import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.neighbors import KNeighborsRegressor
from src.dataset import BaseDataset
from src.NN_models import SimpleMLP, validate, train_epoch, GNN_validate, GNN_train_epoch
from copy import deepcopy
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer, Callback

import sys
from pathlib import Path
sys.path.insert(1, Path("../N2GNN").resolve().as_posix())
from interfaces.pl_data_interface import PlPyGDataTestonValModule
import train_utils
from time import time

import torch

class BaseLearner:
    """ Base class for all active and passive learners """
    def __init__(self, sample_space: BaseDataset):
        self.sample_space = sample_space
    
    def sample(self, sample_space: BaseDataset, n_samples: int) -> any:
        """ Samples n_samples from the sample space.
        Passive learning by default.
        Args:
            n_samples: int, number of samples to draw
            
        Returns:
            samples: np.ndarray, shape (n_samples, n_features)
        """
        
        # Check if sample space is indexable
        if hasattr(sample_space, '__getitem__'):
            # Generate random indices
            indices = np.random.choice(len(sample_space), n_samples, replace=False)
            return sample_space[indices]

        # Sample from sample space
        return sample_space.sample(n_samples)
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    
class GPLearner(BaseLearner):
    """ Gaussian Process learner """
    def __init__(self, sample_space: BaseDataset, kernel: kernels.Kernel, treshold: float, n_restarts_optimizer: int = 0):
        super().__init__(BaseLearner)
        self.kernel = kernel
        self.model = GaussianProcessRegressor(kernel, n_restarts_optimizer=n_restarts_optimizer)
        self.treshold = treshold
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    
    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
    
    def sample(self, n_samples: int, global_problem: BaseDataset, n_reject_samples: int=1000) -> tuple[np.ndarray, np.ndarray]:
        accepted_samples = []
        accepted_indexes = []  # This is for keeping track of labels when it is hard to get a label from X (e.g. when sampling from a list of images)
        
        n_accepted = 0
        n_tries = 0
        while n_accepted < n_samples:
            x, indexes = global_problem.sample(n_reject_samples)
            y_hat, std = self.model.predict(x, return_std=True)
            # Sample from predictive distribution
            y_hat = y_hat + np.random.randn(n_reject_samples)*std
            
            below_thresh = y_hat < self.treshold
            if not np.any(below_thresh):
                n_tries += 1
            else:
                accepted_samples = list(x[below_thresh]) + accepted_samples
                accepted_indexes = list(indexes[below_thresh]) + accepted_indexes
                n_accepted += np.sum(below_thresh)
            if n_tries > 10**4:
                print("Could not find enough samples below the treshold")
                return global_problem.sample(n_samples)
        return np.array(accepted_samples)[:n_samples], np.array(accepted_indexes)[:n_samples]
    
    
class KNNLearner(BaseLearner):
    """ K-Nearest Neighbors learner """
    def __init__(self, sample_space: BaseDataset, n_neighbors: int=5):
        super().__init__(BaseLearner)
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    
    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
    
    def rank_samples(self, unlabelled_data: np.ndarray) -> np.ndarray:
        """ Rank samples by distance to the training data """
        distances, indices = self.model.kneighbors(unlabelled_data, 1)
        return np.argsort(distances[:, 0])[::-1]
    
    def select_samples(self, unlabelled_data: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """ Select n_samples from the unlabelled data """
        ranking = self.rank_samples(unlabelled_data)
        return unlabelled_data[ranking[:n_samples]], ranking[:n_samples]
    
class NNLearner(BaseLearner):
    """ Neural Network learner """
    def __init__(self, sample_space: BaseDataset, hidden_size: int, num_layers: int, output_size: int, loss, device: str = 'cpu', lr: float = 0.001, n_epochs: int = 1000, batch_size: int = 32):
        super().__init__(BaseLearner)
        self.sample_space = sample_space
        self.model_init = lambda: SimpleMLP(input_size=784, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
        self.model = self.model_init()
        self.loss = loss
        self.device = device
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Prepare data
        dataset = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
        train_size = int(0.8*len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Reset model
        self.model = self.model_init()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Train the model
        validation_losses = []
        validation_accuracies = []
        best_model = None
        best_accuracy = 0
        
        for epoch in range(self.n_epochs):
            # Validate model
            val_loss, val_accuracy = validate(self.model, self.loss, validation_loader, self.device)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            # Update best model if necessary
            if best_model is None or val_accuracy > best_accuracy:
                best_model = deepcopy(self.model)
                best_accuracy = val_accuracy
            
            # Train
            for j in range(10):
                train_epoch(self.model, self.optimizer, self.loss, train_loader, self.device)
        print(f"Best Validation Accuracy: {best_accuracy} at epoch {validation_accuracies.index(best_accuracy)}")
        # Set the best model
        self.model = best_model
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X).float().to(self.device)
        return self.model(X).detach().cpu().numpy()
    
    def rank_samples(self, unlabelled_data: np.ndarray) -> np.ndarray:
        """ Rank samples by distance to the training data """
        y_hat = self.predict(unlabelled_data)
        max_unc = np.max(y_hat, axis=1)
        return np.argsort(max_unc)
    
    def select_samples(self, unlabelled_data: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """ Select n_samples from the unlabelled data """
        ranking = self.rank_samples(unlabelled_data)
        return unlabelled_data[ranking[:n_samples]], ranking[:n_samples]
    
    def sample(self, n_samples: int, global_problem: BaseDataset, n_reject_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        accepted_samples = []
        accepted_indexes = []  # This is for keeping track of labels when it is hard to get a label from X (e.g. when sampling from a list of images)
        
        n_accepted = 0
        n_tries = 0
        while n_accepted < n_samples:
            x, indexes = global_problem.sample(n_reject_samples)
            #x = torch.tensor(x).float().to(self.device)
            y_hat = self.model.forward(x)
            rejection_value = torch.rand(n_reject_samples)
            accepted = y_hat > rejection_value
            if not torch.any(accepted):
                n_tries += 1
            else:
                accepted_samples = list(x[accepted]) + accepted_samples
                accepted_indexes = list(indexes[accepted]) + accepted_indexes
                n_accepted += accepted.sum().detach().item()
            if n_tries > 10**4:
                print("Could not find enough samples below the treshold")
                return global_problem.sample(n_samples)
        return list(accepted_samples)[:n_samples], np.array(accepted_indexes)[:n_samples]
    
class StoreBestModel(Callback):
    """ Callback for storing the best model in RAM to avoid saving it to disk """
    def __init__(self, monitor, mode):
        self.best_model = None
        self.best_metric = None
        self.monitor = monitor
        self.mode = mode
    
    def on_validation_end(self, trainer, pl_module):
        if self.best_model is None or (self.mode == "min" and trainer.callback_metrics[self.monitor] < self.best_metric) or (self.mode == "max" and trainer.callback_metrics[self.monitor] > self.best_metric):
            self.best_model = pl_module.state_dict().copy()
            self.best_metric = trainer.callback_metrics[self.monitor]
        
    
class GNNLearner(NNLearner):
    def __init__(self, model_init: BaseDataset, args: any, run_name: str, run_number: int):
        #super().__init__(GNNLearner)
        self.model_init = model_init
        self.model = self.model_init()
        self.args = args
        self.run_name = run_name
        self.run_number = run_number
        self.num_workers = args.num_workers
        
        self.logger = WandbLogger(name=f'{self.run_name}_run_{str(self.run_number)}',
                             project=args.exp_name,
                             save_dir=args.save_dir,
                             offline=args.offline)
        self.logger.log_hyperparams(args)
        
        
    def prepare_data(self, dataset) -> any:
        path, pre_transform, follow_batch = train_utils.data_setup(self.args)
        train_size = int(0.8*len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
        datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=val_dataset,
                                              batch_size=self.args.batch_size,
                                              num_workers=self.num_workers,
                                              follow_batch=follow_batch,
                                              drop_last=False)
        return datamodule
    
    def test(self, test_dataset) -> float:
        datamodule = self.prepare_data(test_dataset)
        
        timer = Timer(duration=dict(weeks=4))
        best_model_checkpoint = StoreBestModel(monitor="val/metric", mode=self.args.mode)
        trainer = Trainer(accelerator="auto",
                          devices="auto",
                          max_epochs=self.args.num_epochs,
                          logger=self.logger,
                          callbacks=[TQDMProgressBar(refresh_rate=20),
                                     best_model_checkpoint,
                                     LearningRateMonitor(logging_interval="epoch"),
                                     timer])
        return trainer.test(self.model, datamodule=datamodule)
        
    def fit(self, dataset) -> None:        
        datamodule = self.prepare_data(dataset)
        timer = Timer(duration=dict(weeks=4))
        best_model_checkpoint = StoreBestModel(monitor="val/metric", mode=self.args.mode)
        trainer = Trainer(accelerator="auto",
                          devices="auto",
                          max_epochs=self.args.num_epochs,
                          logger=self.logger,
                          callbacks=[TQDMProgressBar(refresh_rate=20),
                                     best_model_checkpoint,
                                     LearningRateMonitor(logging_interval="epoch"),
                                     timer])

        trainer.fit(self.model, datamodule=datamodule)
        
        # Set the best model
        self.model.load_state_dict(best_model_checkpoint.best_model)
        
    def rank_samples(self, unlabelled_data: np.ndarray) -> np.ndarray:
        """ Rank samples by distance to the training data """
        y_hat = torch.stack([self.model.forward(x) for x in unlabelled_data])
        unc = torch.abs(torch.nn.functional.softmax(y_hat, 0) - 0.5)
        return torch.argsort(unc, descending=True)
    
    def select_samples(self, unlabelled_data: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """ Select n_samples from the unlabelled data """
        ranking = self.rank_samples(unlabelled_data)
        return [unlabelled_data[index] for index in ranking[:n_samples]], ranking[:n_samples]