import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.neighbors import KNeighborsRegressor
from src.dataset import BaseDataset
from src.NN_models import SimpleMLP, validate, train_epoch, GNN_validate, GNN_train_epoch
from copy import deepcopy
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, Timer, Callback, early_stopping
from torch_geometric.loader import DataLoader

import sys
from pathlib import Path
sys.path.insert(1, Path("../N2GNN").resolve().as_posix())
from interfaces.pl_data_interface import PlPyGDataTestonValModule
import train_utils
from time import time
import wandb

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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        data_loader = self.prepare_data(X)
        return torch.stack([self.model.forward(x) for x in data_loader]).detach().cpu().numpy()
    
    def sample(self, n_samples: int, global_problem: BaseDataset, n_reject_samples: int = 1000, ensure_subset: bool = False, black_listed_indexes: list[int]= [], subset_chance = None) -> tuple[np.ndarray, np.ndarray]:
        accepted_samples = []
        accepted_indexes = []  # This is for keeping track of labels when it is hard to get a label from X (e.g. when sampling from a list of images)
        model_device = self.model.device
        self.model.to("cpu")
        n_accepted = 0
        n_tries = 0
        self.model.eval()
        with torch.no_grad():
            while n_accepted < n_samples:
                x, indexes = global_problem.sample(n_reject_samples, self.device)
                rejection_value = torch.rand(n_reject_samples)
                if ensure_subset:
                    accepted = x.in_subset.bool()
                    if subset_chance is not None:
                        in_subset_indexes = torch.arange(x.in_subset.shape[0])[x.in_subset.bool()]
                        out_subset_indexes = torch.arange(x.in_subset.shape[0])[~x.in_subset.bool()]
                        added = torch.rand(x.in_subset.bool().sum()) < subset_chance
                        amount_added = added.sum()
                        accepted = torch.zeros(x.in_subset.shape[0], dtype=bool)
                        accepted[in_subset_indexes[:amount_added]] = True
                        accepted[out_subset_indexes[:(added.shape[0] - amount_added)]] = True
                else:
                    y_hat = self.model.forward(x)
                    accepted = y_hat > rejection_value
                    
                in_list = torch.isin(indexes, torch.tensor(accepted_indexes + black_listed_indexes))
                accepted = torch.logical_and(accepted, torch.logical_not(in_list))
                if not torch.any(accepted):
                    n_tries += 1
                else:
                    accepted_samples = list(x[accepted]) + accepted_samples
                    accepted_indexes = list(indexes[accepted]) + accepted_indexes
                    n_accepted += accepted.sum().detach().item()
                if n_tries > 10**4:
                    raise ValueError("Could not find enough samples below the treshold")
            self.model.to(model_device)
            
            accepted_samples = list(accepted_samples)
            accepted_indexes = np.array(accepted_indexes)
            # shuffle the samples
            shuffle = np.random.permutation(len(accepted_samples))
            accepted_samples = [accepted_samples[shuffle[i]] for i in range(n_samples)]
            return accepted_samples, accepted_indexes[shuffle][:n_samples]
            
            #return list(accepted_samples)[:n_samples], np.array(accepted_indexes)[:n_samples]
        
    def to(self, device: str): 
        self.model.to(device)
    
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
        
        
class MonitorAccuracy(Callback):
    """ Class for calculating the accuracy of the model """
    def __init__(self, monitor):
        self.monitor = monitor
        self.accuracy = []
    
    def on_validation_end(self, trainer: Trainer, pl_module: any):
        self.accuracy.append(trainer.callback_metrics[self.monitor])
        
    def on_test_end(self, trainer: Trainer, pl_module: any):
        self.accuracy.append(trainer.callback_metrics[self.monitor])
        
    
class GNNLearner(NNLearner):
    def __init__(self, model_init: BaseDataset, args: any, run_name: str, is_classifier: bool = False, device: str = 'cpu'):
        #super().__init__(GNNLearner)
        self.model_init = model_init
        self.model = self.model_init()
        self.args = args
        self.run_name = run_name
        self.is_classifier = is_classifier
        self.num_workers = args.num_workers
        self.device = device
        path, pre_transform, self.follow_batch = train_utils.data_setup(self.args)
        self.logger = WandbLogger(name=self.run_name,
                                    project=args.exp_name,
                                    group=args.group,
                                    save_dir=args.save_dir,
                                    offline=args.offline,
                                    )
        self.logger.log_hyperparams(args)
        
        
    def prepare_data(self, dataset) -> any:
        train_size = int(len(dataset))
        val_size = len(dataset) - train_size  # This is a hack to make the datamodule work
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
        datamodule = PlPyGDataTestonValModule(train_dataset=train_dataset,
                                              val_dataset=val_dataset,
                                              test_dataset=val_dataset,
                                              batch_size=self.args.batch_size,
                                              num_workers=self.num_workers,
                                              persistent_workers=True,
                                              follow_batch=self.follow_batch,
                                              drop_last=False)
        return datamodule
    
    def prepare_test_data(self, dataset, num_workers=None) -> any:
        test_dataset = dataset
        num_workers = num_workers if num_workers is not None else self.num_workers
        datamodule = PlPyGDataTestonValModule(train_dataset=test_dataset,
                                              val_dataset=test_dataset,
                                              test_dataset=test_dataset,
                                              batch_size=self.args.batch_size,
                                              num_workers=num_workers,
                                              persistent_workers=True if num_workers>0 else False,
                                              follow_batch=self.follow_batch,
                                              drop_last=True)
        return datamodule
    
    def test(self, test_dataset, num_workers = None) -> float:
        datamodule = self.prepare_test_data(test_dataset, num_workers)
        #best_model_checkpoint = StoreBestModel(monitor="val/metric", mode=self.args.mode)
        trainer = Trainer(accelerator="auto",
                          devices=self.device,
                          max_epochs=self.args.num_epochs,
                          logger=self.logger,
                          num_sanity_val_steps=0,
                          enable_model_summary=False,
                          enable_progress_bar=False,
                          )
        return trainer.test(self.model, datamodule=datamodule)
        
    def fit(self, dataset) -> None:        
        datamodule = self.prepare_data(dataset)
        trainer =  Trainer(accelerator="auto",
                          devices=self.device,
                          num_sanity_val_steps=0,
                          max_epochs=self.args.num_epochs,
                          logger=self.logger,
                          enable_model_summary=False,
                          enable_progress_bar=False,
                          callbacks=[#best_model_checkpoint,
                                     LearningRateMonitor(logging_interval="epoch"),
                                     #early_stopping.EarlyStopping(monitor="val/metric", patience=self.args.patience),
                                     ])

        trainer.fit(self.model, datamodule=datamodule)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        dataloader = DataLoader(X,
                    batch_size=self.args.batch_size,
                    num_workers=0,
                    shuffle=False,
                    follow_batch=self.follow_batch, 
                    persistent_workers=False, 
                    pin_memory=True)
        self.model.eval()
        with torch.no_grad():
            y_hat = torch.cat([self.model.forward(data) for data in dataloader])
        return y_hat.detach().cpu().numpy()
        
    def rank_samples(self, unlabelled_data: np.ndarray) -> np.ndarray:
        """ Rank samples by distance to the training data """
        y_hat = torch.stack([self.model.forward(x) for x in unlabelled_data])
        unc = torch.abs(torch.nn.functional.softmax(y_hat, 0) - 0.5)
        return torch.argsort(unc, descending=True)
    
    def select_samples(self, unlabelled_data: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """ Select n_samples from the unlabelled data """
        ranking = self.rank_samples(unlabelled_data)
        return [unlabelled_data[index] for index in ranking[:n_samples]], ranking[:n_samples]
    
class CNNLearner(NNLearner):
    def __init__(self, model_init: BaseDataset, args: any, run_name: str, is_subset_model: bool = False, device: str = 'cpu'):
        #super().__init__(GNNLearner)
        self.model_init = model_init
        self.model = self.model_init()
        self.model = self.model.to(device)
        self.model.device = device
        self.args = args
        self.run_name = run_name
        self.is_subset_model = is_subset_model
        self.num_workers = args.num_workers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        if self.is_subset_model:
            self.criteria = torch.nn.BCELoss()
        else:
            self.criteria = torch.nn.CrossEntropyLoss()
            
        
        self.device = device
        self.logger = WandbLogger(name=self.run_name,
                                    project=args.exp_name,
                                    group=args.group,
                                    save_dir=args.save_dir,
                                    offline=args.offline,
                                    )
        self.logger.log_hyperparams(args)
    
    def test(self, test_dataset) -> float:
        data_loader = self.dataloader(test_dataset, 
                                       self.is_subset_model, 
                                       batch_size=self.args.batch_size, 
                                       shuffle=False, 
                                       persistent_workers=False, 
                                       pin_memory=True)  
        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device).squeeze(-1)
                y_hat = self.model(x)
                l = self.criteria(y_hat.squeeze(-1), y)
                if isinstance(self.criteria, torch.nn.BCELoss):
                    y_hat = (y_hat > 0.5).float()
                    val_accuracy += (y_hat == y).sum().item()
                else:
                    val_accuracy += (y==y_hat.argmax(-1)).sum().item()
                
                val_loss += l.item()
        n = len(test_dataset)
        val_loss /= n
        val_accuracy /= n
        return val_accuracy
        
    
    def dataloader(self, dataset, is_subset, **kwargs):
        if is_subset:
            dataset = [(data.x[0], data.in_subset.float()) for data in dataset]
        else:
            dataset = [(data.x[0], data.y) for data in dataset]
        return torch.utils.data.DataLoader(dataset, **kwargs)
        
    def fit(self, dataset, validation_set=None, validation_frequency=10) -> None:        
        train_loader = self.dataloader(dataset, 
                                       self.is_subset_model, 
                                       batch_size=self.args.batch_size, 
                                       drop_last=True,
                                       shuffle=True, 
                                       persistent_workers=False, 
                                       pin_memory=True)
        self.model.train()
        validation_accs = []
        for epoch in range(self.args.num_epochs):
            for i, (x, y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                l = self.criteria(y_hat.squeeze(-1), y.squeeze(-1))
                l.backward()
                self.optimizer.step()

            if (epoch % validation_frequency == 0) and (validation_set is not None):
                val_accuracy = self.test(validation_set)
                validation_accs.append(val_accuracy)
                print(f"Validation Accuracy at epoch {epoch}: {val_accuracy}")
                self.model.train()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        dataloader = torch.utils.data.DataLoader(X, 
                                                 batch_size=self.args.batch_size, 
                                                 shuffle=False, 
                                                 persistent_workers=False, 
                                                 pin_memory=True)
        self.model.eval()
        with torch.no_grad():
            y_hat = torch.cat([self.model.forward(data) for data in dataloader])
        return y_hat.detach().cpu().numpy()
        
    def rank_samples(self, unlabelled_data: np.ndarray) -> np.ndarray:
        """ Rank samples by distance to the training data """
        y_hat = torch.stack([self.model.forward(x) for x in unlabelled_data])
        unc = torch.abs(torch.nn.functional.softmax(y_hat, 0) - 0.5)
        return torch.argsort(unc, descending=True)
    
    def select_samples(self, unlabelled_data: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """ Select n_samples from the unlabelled data """
        ranking = self.rank_samples(unlabelled_data)
        return [unlabelled_data[index] for index in ranking[:n_samples]], ranking[:n_samples]
    
    
class EnsembleModel:
    def __init__(self, models):
        self.models = models
        self.device = models[0].device
        
    def to(self, device):
        for model in self.models:
            model.to(device)
        self.device = device
        
    def train(self):
        for model in self.models:
            model.train()
            
    def eval(self):
        for model in self.models:
            model.eval()
            
    def predict(self, X):
        y_hat = np.stack([model.predict(X) for model in self.models])
        return np.mean(y_hat, axis=0)
            
class EnsembleLearner(NNLearner):
    """ Model for combining multiple ALREADY models """
    def __init__(self, models: list[NNLearner]):
        self.model = EnsembleModel(models)
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        y_hat = np.stack([model.predict(X) for model in self.models])
        return np.mean(y_hat, axis=0)
        

class RandomLearner(BaseLearner):
    """ Random model for baseline comparison """
    def __init__(self, sample_space: BaseDataset):
        super().__init__(sample_space)
    
    def sample(self, n_samples: int, global_problem: BaseDataset, n_reject_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        return global_problem.sample(n_samples)
    
    def rank_samples(self, unlabelled_data: np.ndarray) -> np.ndarray:
        return np.random.permutation(len(unlabelled_data))
    
    def select_samples(self, unlabelled_data: np.ndarray, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        ranking = self.rank_samples(unlabelled_data)
        return unlabelled_data[ranking[:n_samples]], ranking[:n_samples]
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    def sample(self, n_samples: int, global_problem: BaseDataset, n_reject_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        return global_problem.sample(n_samples, "cpu")