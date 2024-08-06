import numpy as np
import scipy.optimize as opt
import torch
from torchdrug import datasets
from typing import Optional
from torchdrug.data.dataloader import graph_collate
#from src.dataset_downloader import get_MNIST

class BaseDataset:
    def __init__(self):
        pass
    
    def label(self, X: any, index: Optional[np.ndarray] = None):
        raise NotImplementedError
    
    def sample(self, n_samples: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

'''
class MNIST_manifold(BaseDataset):
    def __init__(self):
        self.train_data, self.val_data, self.test_data = get_MNIST(0.0)
        self.X_train = np.concatenate([x for x, _ in self.train_data], axis=0)
        self.y_train = np.array([y for _, y in self.train_data])
        self.y_manifold_train = self.convert_data_under_manifold(self.X_train)
        #self.X_val = np.concatenate([x for x, _ in self.val_data], axis=0)
        #self.y_val = self.convert_data_under_manifold(self.val_data)
        self.X_test = np.concatenate([x for x, _ in self.test_data], axis=0)
        self.y_manifold_test = self.convert_data_under_manifold(self.X_test)
        self.y_test = np.array([y for _, y in self.test_data])
        self.mask_test_data()
        
    def mask_test_data(self) -> None:
        """ Mask out test data not part of the manifold """
        y_mask = self.manifold_classifier(self.X_test)
        self.test_data = torch.utils.data.Subset(self.test_data, np.arange(y_mask)[y_mask])
        self.X_test = np.concatenate([x for x, _ in self.test_data], axis=0)
        self.y_manifold_test = self.convert_data_under_manifold(self.X_test)
        self.y_test = np.array([y for _, y in self.test_data])
        
    def convert_data_under_manifold(self, data: np.ndarray) -> np.ndarray:
        y = self.manifold_classifier(data)
        # Convert to one hot encoding
        y_one_hot = np.zeros((len(y), 2))
        for i in range(len(y)):
            y_one_hot[i, y[i]] = 1
        return y_one_hot
    
    def return_test_data(self) -> np.ndarray:
        return self.test_data
        
    def manifold_classifier(self, X: np.ndarray) -> np.ndarray:
        """ Function for classifying if the data belongs to the manifold or not. 
        Args:
            X: The data to classify with shape (n_samples, x_dim, x_dim)"""
        
        length = X.shape[-1]
        top_side_brightness = np.abs(X[..., length//2:, :]).mean(-1).mean(-1)
        bottom_side_brightness = np.abs(X[..., :length//2, :]).mean(-1).mean(-1)
        return top_side_brightness < bottom_side_brightness
    
    def sample(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.train_data), n_samples)
        return self.X_train[indices], indices
    
    def label_global(self, X: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        if index is None:
            raise ValueError("Index must be provided")
        return self.y_train[index]
    
    def label_manifold(self, X: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        if index is None:
            raise ValueError("Index must be provided")
        return self.y_manifold_train[index]
'''  
    
class ZINC250k_manifold(BaseDataset):
    def __init__(self, data_path: str = "data/ZINC250k", logP_interval = (1, 3), lazy: bool = False, reduced_size: int = -1):
        """ Manifold Dataset for storing the Zinc250k dataset.
        Args:
            data_path: Where to store the dataset after downloading
            logP_interval: The interval of logP values within the target manifold
            lazy: Whenver to process the graph structure of the molecules online or all molecules in the beginning
            reduced_size: How much to reduce the dataset size. -1 to remove. Only used for debugging.
        """
        self.logP_tresh = logP_interval
        self.processed = False
        data = datasets.ZINC250k(data_path, lazy=lazy)
        self.node_feature_dim = data.node_feature_dim
        self.tasks = data.tasks
        # Split data
        n_samples = len(data)
        n_train = int(n_samples*0.8)
        n_test = n_samples - n_train
        self.train_data, self.test_data = torch.utils.data.random_split(data, [n_train, n_test])
        if not lazy:
            self.process_data()
            
    def process_data(self):
        y_train_logP = torch.tensor([self.train_data[i]["logP"] for i in range(len(self.train_data))])
        self.X_train = [self.train_data[i]["graph"] for i in range(len(self.train_data))]
        self.y_manifold_train = self.convert_data_under_manifold(y_train_logP)
        self.X_test = [self.test_data[i]["graph"] for i in range(len(self.test_data))]
        y_test_logP = torch.tensor([self.test_data[i]["logP"] for i in range(len(self.test_data))])
        self.y_manifold_test = self.convert_data_under_manifold(y_test_logP)
        self.y_test = torch.tensor([self.test_data[i]["qed"] for i in range(len(self.test_data))])
        self.mask_test_data()
        self.processed = True
        
    def mask_test_data(self) -> None:
        """ Mask out test data not part of the manifold """
        self.test_data = torch.utils.data.Subset(self.test_data, np.arange(len(self.y_manifold_test))[self.y_manifold_test])
        self.X_test = [self.test_data[i]["graph"] for i in range(len(self.test_data))]
        y_test_logP = torch.tensor([self.test_data[i]["logP"] for i in range(len(self.test_data))])
        self.y_manifold_test = self.convert_data_under_manifold(y_test_logP)
        self.y_test = torch.tensor([self.test_data[i]["qed"] for i in range(len(self.test_data))])
        
    def convert_data_under_manifold(self, logP) -> np.ndarray:
        return np.logical_and(self.logP_tresh[0] <= logP, logP <= self.logP_tresh[1])
    
    def return_test_data(self) -> np.ndarray:
        return self.test_data
    
    def sample(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        indices = torch.randperm(len(self.train_data))[:n_samples]
        if self.processed:
            return [graph_collate([self.X_train[i]]) for i in indices], indices
        else:
            return [graph_collate([self.train_data[i]]) for i in indices], indices
    
    def label_global(self, X: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        if index is None:
            raise ValueError("Index must be provided")
        return self.y_train[index]
    
    def label_manifold(self, X: np.ndarray, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if indices is None:
            raise ValueError("Index must be provided")
        if self.processed:
            return self.y_manifold_train[indices]
        else:
            data = torch.Tensor([self.train_data[i]["logP"] for i in indices])
            return self.convert_data_under_manifold(data)
            

class ToySample1:
    def __init__(self, n_features: int = 2):
        self.n_features = 2
        self.min_val = -10
        self.max_val = 10
    
    def label_global(self, X: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        return np.sin(np.abs(X[..., 0])) - np.abs(X[..., 1])
    
    def sample(self, n_samples: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        X = np.random.rand(n_samples, 2)*(self.max_val - self.min_val) + self.min_val
        return X, -1*np.ones(n_samples)
    
    def manifold(self, t: np.ndarray) -> np.ndarray:
        t = t.squeeze()
        return np.stack([t + np.sin(t) , np.sin(t) + np.abs(t)**0.5], axis=-1)
    
    def label_manifold(self, X: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        return np.array([self.manifold_dist(x) for x in X])
    
    def sample_manifold(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.random.rand(n_samples)*(self.max_range - self.min_range) + self.min_range
        return self.manifold(t), np.zeros(n_samples)
    
    def manifold_dist(self, x: np.ndarray, n_init: int = 10) -> float:
        def distance(t): 
            return np.sum((x - self.manifold(t))**2)**0.5
        # Find the minimum of the distance function
        min_dist = np.inf
        x0s = np.linspace(self.min_val, self.max_val, n_init)
        for i in range(n_init):
            res = opt.minimize(distance, x0s[i])
            if res.fun < min_dist:
                min_dist = res.fun
        return min_dist


class IndexSampler(BaseDataset):
    """ Samples from an object with an __len__ method """
    def __init__(self, sample_space: any, labels: any):
        """" Args:
            sample_space: object with __len__ method to sample from
        """
        self.sample_space = sample_space
        self.labels = labels
        
    def sample(self, n_samples: int, replace: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """ Samples n_samples from the sample space. """
        indices = np.random.choice(len(self.sample_space), n_samples, replace=replace)
        return self.sample_space[indices], indices
    
    def label(self, X: any, index: np.ndarray) -> any:
        return self.labels[index] 
    
    def __len__(self):
        return len(self.sample_space)
    
    def __getitem__(self, key):
        return self.sample_space[key]
    
    def __iter__(self):
        return iter(self.sample_space)
    
    def __next__(self):
        return next(self.sample_space)
    