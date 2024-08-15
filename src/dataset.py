from functools import partial
import numpy as np
import scipy.optimize as opt
import torch
from typing import Optional
from copy import copy
from torch_geometric.datasets import ZINC
import torch_geometric
import sys
from pathlib import Path
sys.path.insert(1, Path("../N2GNN").resolve().as_posix())
from data_utils import TupleData

#from src.dataset_downloader import get_MNIST

class BaseDataset:
    def __init__(self):
        pass
    
    def label(self, X: any, index: Optional[np.ndarray] = None):
        raise NotImplementedError
    
    def sample(self, n_samples: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


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
    

class SubsetGraph(torch_geometric.data.Data):
    """ Class for storing the graph data, but with additional information related to the subset"""
    def __init__(self, logP=None, in_subset=None, **kwargs):
        super(SubsetGraph, self).__init__(**kwargs)
        if logP is not None:
            self.logP = logP
            self._data.logP = logP
        if in_subset is not None:
            self.in_subset = in_subset
            self._data.in_subset = in_subset
    
    @property
    def logP(self) -> Optional[torch.Tensor]:
        return self['logP'] if 'logP' in self._store else None
    
    @logP.setter
    def logP(self, value: torch.Tensor) -> None:
        self._store.logP = value
        
    @property
    def in_subset(self) -> Optional[torch.Tensor]:
        return self['in_subset'] if 'in_subset' in self._store else None
    
    @in_subset.setter
    def in_subset(self, value: torch.Tensor) -> None:
        self._store.in_subset = value
        
    
class ZINC250k_manifold(BaseDataset):
    def __init__(self, data_path: str = "data/ZINC250k", logP_interval = (1, 3), 
                 full_dataset: bool = False, pre_transform = None, transform = None):
        """ Manifold Dataset for storing the Zinc250k dataset.
        Args:
            data_path: Where to store the dataset after downloading
            logP_interval: The interval of logP values within the target manifold
            full_dataset: If the full dataset should be used
            pre_transform: Preprocessing transformation
            transform: Transformation to apply to the data
        """
        self.logP_tresh = logP_interval
        dataset_init = partial(ZINC, data_path, subset=not full_dataset, pre_transform=pre_transform, transform=transform)
        self.train_data = dataset_init(split="train")
        self.val_data = dataset_init(split="val")
        self.test_data = dataset_init(split="test")
        self.node_feature_dim = self.train_data.data.x.shape[-1]      
        
        self.process_data()
        self.mask_test_data()
        
        
    def add_attribute(self, attribute: str, attribute_data: torch.Tensor, data: torch_geometric.data.Data):
        data_obj = TupleData(**{attribute: attribute_data}, **data._data)
        data_obj._num_nodes = data._data._num_nodes
        data._data = data_obj
        #data.data = data_obj
        data.slices[attribute] = torch.arange(len(attribute_data)+1)        
        
    def process_data(self):
        # TODO remove temporary workaround for logP value
        self.add_attribute("logP", 6*torch.rand(len(self.train_data)), self.train_data)
        self.add_attribute("logP", 6*torch.rand(len(self.val_data)), self.val_data)
        self.add_attribute("logP", 6*torch.rand(len(self.test_data)), self.test_data)
        
        # Add the in_subset attribute to the data
        in_subset_train = self.tensor_in_subset(self.train_data._data.logP)
        in_subset_val = self.tensor_in_subset(self.val_data._data.logP)
        in_subset_test = self.tensor_in_subset(self.test_data._data.logP)
        
        self.add_attribute("in_subset", in_subset_train, self.train_data)
        self.add_attribute("in_subset", in_subset_val, self.val_data)
        self.add_attribute("in_subset", in_subset_test, self.test_data)
        
    def mask_test_data(self) -> None:
        """ Mask out test data not part of the manifold """
        in_manifold_indices = torch.tensor([i for i, datapoint in enumerate(self.test_data) if datapoint.in_subset])
        self.test_data = self.test_data[in_manifold_indices]
        
    def tensor_in_subset(self, tensor: torch.Tensor) -> torch.Tensor:
        return ((self.logP_tresh[0] <= tensor) & (tensor <= self.logP_tresh[1])).float()
    
    def in_subset(self, datapoint):
        return ((self.logP_tresh[0] <= datapoint.logP) & (datapoint.logP <= self.logP_tresh[1])).float()
    
    def sample(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        indices = torch.randperm(len(self.train_data))[:n_samples]
        return self.train_data[indices], indices
    
    def label_global(self, X: np.ndarray, index: Optional[np.ndarray] = None) -> np.ndarray:
        if index is None:
            raise ValueError("Index must be provided")
        return self.train_data[index]
    
    def label_manifold(self, X: np.ndarray, indices: Optional[np.ndarray] = None) -> np.ndarray:
        if indices is None:
            raise ValueError("Index must be provided")
        return self.train_data[indices]   
        # Return random binary label for now

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
    