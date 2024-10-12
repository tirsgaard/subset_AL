from functools import partial
import numpy as np
import scipy.optimize as opt
import torch
import torchvision
from torchvision import transforms
from typing import Optional
from copy import copy
from torch_geometric.datasets import ZINC
from torch_geometric.data import InMemoryDataset, Data

from torch_geometric.transforms import ToDevice
import torch_geometric
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader
import rdkit
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer
rdkit.rdBase.DisableLog('rdApp.warning')

from tqdm import tqdm

sys.path.insert(1, Path("../N2GNN").resolve().as_posix())
from data_utils import TupleData
from datasets.QM9Dataset import QM9

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
    def __init__(self, data_path: str = "data/ZINC250k", subset_type = "synth", 
                 full_dataset: bool = False, pre_transform = None, transform = None, quantile_threshold: float = 0.9, use_logP = True):
        """ Manifold Dataset for storing the Zinc250k dataset.
        Args:
            data_path: Where to store the dataset after downloading
            logP_interval: The interval of logP values within the target manifold
            full_dataset: If the full dataset should be used
            pre_transform: Preprocessing transformation
            transform: Transformation to apply to the data
        """
        self.atom_min = 23 # Correspond to roughly 50% of the atoms in the dataset, used to define subset
        self.data_path = data_path
        self.full_dataset = full_dataset
        self.use_logP = use_logP
        dataset_init = partial(ZINC, self.data_path, subset=not full_dataset, pre_transform=pre_transform, transform=transform)
        self.train_data = dataset_init(split="train")
        self.val_data = dataset_init(split="val")
        self.test_data = dataset_init(split="test")
        self.node_feature_dim = self.train_data.data.x.shape[-1]
        if subset_type == "synth":
            self.synth_subset_init(quantile_threshold)
        elif subset_type == "size":
            self.size_subset_init()
        else:
            raise ValueError(f"Subset type {subset_type} is not supported")
        self.mask_test_data()
        self.mask_val_data()
        
    def get_smiles(self, split: str, subset: bool = False) -> list[str]:
        name = split if split != "val" else "valid"  # The name of the file is different for the validation set
        smile_path = self.data_path + "/raw/" +  f"/{name}.txt"
        with open(smile_path, 'r') as f:
            smiles = f.readlines()
        # Remove newline characters
        smiles = [smile[:-1] for smile in smiles]
        
        if subset:
            index_path = self.data_path + "/raw/" + f"/{split}.index"
            with open(index_path, 'r') as f:
                indices = [int(index) for index in f.read()[:-1].split(",")]
            smiles = [smiles[i] for i in indices]
        return smiles
        
    def add_attribute(self, attribute: str, attribute_data: torch.Tensor, data: torch_geometric.data.Data):
        data_obj = TupleData(**{attribute: attribute_data}, **data._data)
        data_obj._num_nodes = data._data._num_nodes
        data._data = data_obj
        data.slices[attribute] = torch.arange(len(attribute_data)+1)        
        
    def smiles_to_synthsisability(self, smiles: str, method: str) -> float:
        """ Convert the SMILES string to a synthesisability score """
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if method == "sas":
            return sascorer.calculateScore(mol)
        elif method == "np":
            fscore = npscorer.readNPModel()
            #npscorer.scoreMolWConfidence(mol,fscore)
            return fscore.scoreMol(mol)
        else:
            raise ValueError(f"Method {method} is not supported")
        
    def count_rings_size(self, smiles: str, size: int) -> int:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        return sum(len(ring)>size for ring in mol.GetRingInfo().BondRings())
            
    def size_subset_init(self):
        # Add the in_subset attribute to the data
        in_subset_train = self.tensor_in_subset(self.train_data._data.original_num_nodes, self.atom_min)
        in_subset_val = self.tensor_in_subset(self.val_data._data.original_num_nodes, self.atom_min)
        in_subset_test = self.tensor_in_subset(self.test_data._data.original_num_nodes, self.atom_min)
        
        self.add_attribute("in_subset", in_subset_train, self.train_data)
        self.add_attribute("in_subset", in_subset_val, self.val_data)
        self.add_attribute("in_subset", in_subset_test, self.test_data)
            
    def synth_subset_init(self, quantile_threshold):
        # Add SMILE format to dataset
        train_smiles = self.get_smiles("train", subset=not self.full_dataset)
        val_smiles = self.get_smiles("val", subset=not self.full_dataset)
        test_smiles = self.get_smiles("test", subset=not self.full_dataset)
        
        self.add_attribute("smiles", train_smiles, self.train_data)
        self.add_attribute("smiles", val_smiles, self.val_data)
        self.add_attribute("smiles", test_smiles, self.test_data)
        
        # Add synthisability score
        print("Calculating synthesisability score")
        synth_train = torch.tensor([self.smiles_to_synthsisability(smile, "sas") for smile in tqdm(train_smiles)])
        synth_val = torch.tensor([self.smiles_to_synthsisability(smile, "sas") for smile in tqdm(val_smiles)])
        synth_test = torch.tensor([self.smiles_to_synthsisability(smile, "sas") for smile in tqdm(test_smiles)])
        
        self.add_attribute("synth_score", synth_train, self.train_data)
        self.add_attribute("synth_score", synth_val, self.val_data)
        self.add_attribute("synth_score", synth_test, self.test_data)
        
        # Convert to from logP - SAS - Cycles to logP
        combined_train = self.train_data.data.y
        combined_val = self.val_data.data.y
        combined_test = self.test_data.data.y
        
        rings_train = torch.tensor([self.count_rings_size(smile, 6) for smile in train_smiles])
        rings_val = torch.tensor([self.count_rings_size(smile, 6) for smile in val_smiles])
        rings_test = torch.tensor([self.count_rings_size(smile, 6) for smile in test_smiles])
        
        logP_train = combined_train + synth_train + rings_train
        logP_val = combined_val + synth_val + rings_val
        logP_test = combined_test + synth_test + rings_test
        
        self.add_attribute("logP", logP_train, self.train_data)
        self.add_attribute("logP", logP_val, self.val_data)
        self.add_attribute("logP", logP_test, self.test_data)
        if self.use_logP:
            self.add_attribute("y", logP_train, self.train_data)
            self.add_attribute("y", logP_val, self.val_data)
            self.add_attribute("y", logP_test, self.test_data)
            
        # Calculate top 10% of the synthesisability score from training
        threshold = torch.quantile(self.train_data.synth_score, quantile_threshold)
        
        # Define the subset
        in_subset_train = self.tensor_in_subset(synth_train, threshold)
        in_subset_val = self.tensor_in_subset(synth_val, threshold)
        in_subset_test = self.tensor_in_subset(synth_test, threshold)
        
        self.add_attribute("in_subset", in_subset_train, self.train_data)
        self.add_attribute("in_subset", in_subset_val, self.val_data)
        self.add_attribute("in_subset", in_subset_test, self.test_data)
        
        # Print data summary
        print("Synthesisability score threshold: ", threshold)
        print(f"Number of training samples: {in_subset_train.sum()}")
        print(f"Number of validation samples: {in_subset_val.sum()}")
        print(f"Number of test samples: {in_subset_test.sum()}")
        
        
    def mask_test_data(self) -> None:
        """ Mask out test data not part of the manifold """
        in_manifold_indices = torch.tensor([i for i, datapoint in enumerate(self.test_data) if datapoint.in_subset])
        self.test_data = self.test_data[in_manifold_indices]
        
    def mask_val_data(self) -> None:
        """ Mask out val data not part of the manifold """
        in_manifold_indices = torch.tensor([i for i, datapoint in enumerate(self.val_data) if datapoint.in_subset])
        self.val_data = self.val_data[in_manifold_indices]
        
    def tensor_in_subset(self, tensor: torch.Tensor, tresh_hold: float) -> torch.Tensor:
        return (tensor >= tresh_hold).float()
    
    def sample(self, n_samples: int, device) -> tuple[np.ndarray, np.ndarray]:
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
    
class QM9_manifold(BaseDataset):
    def __init__(self, data_path: str = "data/QM9", subset_type = "synth", pre_transform = None, transform = None, quantile_threshold: float = 0.9):
        """ Manifold Dataset for storing the QM9 dataset.
        Args:
            data_path: Where to store the dataset after downloading
            pre_transform: Preprocessing transformation
            transform: Transformation to apply to the data
        """
        self.data_path = data_path
        dataset = QM9(self.data_path, pre_transform=pre_transform, transform=transform)
        dataset = dataset.shuffle()

        tenprecent = int(len(dataset) * 0.1)
        self.train_mean = dataset.data.y[tenprecent:].mean(dim=0)
        self.train_std = dataset.data.y[tenprecent:].std(dim=0)
        dataset.data.y = (dataset.data.y - self.train_mean) / self.train_std
        self.train_data = dataset[2 * tenprecent:]
        self.test_data = dataset[:tenprecent]
        self.val_data = dataset[tenprecent:2 * tenprecent]
        self.node_feature_dim = self.train_data.data.x.shape[-1]
        
        if subset_type == "synth":
            self.synth_subset_init(quantile_threshold)
        elif subset_type == "size":
            self.size_subset_init()
        else:
            raise ValueError(f"Subset type {subset_type} is not supported")
        self.mask_test_data()
        self.mask_val_data()
        
    def add_attribute(self, attribute: str, attribute_data: torch.Tensor, data: torch_geometric.data.Data):
        data_obj = TupleData(**{attribute: attribute_data}, **data._data)
        data_obj._num_nodes = data._data._num_nodes
        data._data = data_obj
        data.slices[attribute] = torch.arange(len(attribute_data)+1)        
        
    def smiles_to_synthsisability(self, smiles: str, method: str) -> float:
        """ Convert the SMILES string to a synthesisability score """
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if method == "sas":
            return sascorer.calculateScore(mol)
        elif method == "np":
            fscore = npscorer.readNPModel()
            #npscorer.scoreMolWConfidence(mol,fscore)
            return fscore.scoreMol(mol)
        else:
            raise ValueError(f"Method {method} is not supported")
            
    def size_subset_init(self):
        # Add the in_subset attribute to the data
        in_subset_train = self.tensor_in_subset(self.train_data._data.original_num_nodes, self.atom_min)
        in_subset_val = self.tensor_in_subset(self.val_data._data.original_num_nodes, self.atom_min)
        in_subset_test = self.tensor_in_subset(self.test_data._data.original_num_nodes, self.atom_min)
        
        self.add_attribute("in_subset", in_subset_train, self.train_data)
        self.add_attribute("in_subset", in_subset_val, self.val_data)
        self.add_attribute("in_subset", in_subset_test, self.test_data)
            
    def synth_subset_init(self, quantile_threshold):
        # Calculate top 10% of the synthesisability score from training
        threshold = torch.quantile(self.train_data._data.sas, quantile_threshold)
        
        # Define the subset
        in_subset_train = self.tensor_in_subset(self.train_data._data.sas, threshold)
        in_subset_val = self.tensor_in_subset(self.val_data._data.sas, threshold)
        in_subset_test = self.tensor_in_subset(self.test_data._data.sas, threshold)
        
        self.add_attribute("in_subset", in_subset_train, self.train_data)
        self.add_attribute("in_subset", in_subset_val, self.val_data)
        self.add_attribute("in_subset", in_subset_test, self.test_data)
        
        # Print data summary
        print("Synthesisability score threshold: ", threshold)
        print(f"Number of training samples: {in_subset_train.sum()}")
        print(f"Number of validation samples: {in_subset_val.sum()}")
        print(f"Number of test samples: {in_subset_test.sum()}")
        
        
    def mask_test_data(self) -> None:
        """ Mask out test data not part of the manifold """
        in_manifold_indices = torch.tensor([i for i, datapoint in enumerate(self.test_data) if datapoint.in_subset])
        self.test_data = self.test_data[in_manifold_indices]
        
    def mask_val_data(self) -> None:
        """ Mask out val data not part of the manifold """
        in_manifold_indices = torch.tensor([i for i, datapoint in enumerate(self.val_data) if datapoint.in_subset])
        self.val_data = self.val_data[in_manifold_indices]
        
    def tensor_in_subset(self, tensor: torch.Tensor, tresh_hold: float) -> torch.Tensor:
        return (tensor >= tresh_hold).float()
    
    def sample(self, n_samples: int, device) -> tuple[np.ndarray, np.ndarray]:
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

class MNISTDataset(InMemoryDataset):
    """ Class for storing the MNIST data.
    Each datapoint contains:
    x: The image
    y: The class
    in_subset: Whether the datapoint is part of the subset
    """
    def __init__(self, data_list):
        self.data_list = data_list
        super().__init__("", None)
        self.data, self.slices = self.collate(data_list)
    


def get_MNIST_train(device: str = "cpu") -> torchvision.datasets.MNIST:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transform)
    # Precompute the full dataset
    trainset = list(trainset)
    # Transfer to device
    trainset = [(x.to(device), y) for x, y in trainset]
    return trainset

def get_MNIST_test(device: str = "cpu") -> torchvision.datasets.MNIST:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transform)
    # Precompute the full dataset
    testset = list(testset)
    
    testset = [(x.to(device), y) for x, y in testset]
    return testset

def get_rotated_MNIST(val_split: float, leave_out_9: bool = True) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """ Function for loading the binary MNIST dataset with the specified digits and validation split
    Args:
        val_split: The proportion of the training set to use for validation
        leave_out_9: Whether to leave out the digit 9
        
        Returns:
            train_data: The training set
            val_data: The validation set
            test_data: The test set
    """
    # Load the data
    train_data = get_MNIST_train()
    test_data = get_MNIST_test()

    if leave_out_9:
        train_data = list(filter(lambda i: i[1] != 9, train_data))
        test_data = list(filter(lambda i: i[1] != 9, test_data))
    
    # Split the training set into training and validation
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    # Add all 90% rotations of the images
    def rotate_data(data):
        rotated_data = []
        with torch.no_grad():
            for x, y in data:
                for i in range(4):
                    datapoint = Data(x=torch.rot90(x, k=i, dims=(1, 2))[:, None, ...], y=y, in_subset=i % 4 == 0)
                    rotated_data.append(datapoint)
        return MNISTDataset(rotated_data)
    train_data = rotate_data(train_data)
    val_data = rotate_data(val_data)
    test_data = rotate_data(test_data)
    
    return train_data, val_data, test_data


class MNIST_manifold(BaseDataset):
    def __init__(self, data_path: str = "data"):
        """ Manifold Dataset for storing the MNIST dataset.
        Args:
            data_path: Where to store the dataset after downloading
        """
        self.data_path = data_path
        self.train_data, self.val_data, self.test_data = get_rotated_MNIST(0.5)

        self.mask_test_data()
        self.mask_val_data()
        
    def add_attribute(self, attribute: str, attribute_data: torch.Tensor, data: torch_geometric.data.Data):
        data_obj = TupleData(**{attribute: attribute_data}, **data._data)
        data_obj._num_nodes = data._data._num_nodes
        data._data = data_obj
        data.slices[attribute] = torch.arange(len(attribute_data)+1)        
            
        
    def mask_test_data(self) -> None:
        """ Mask out test data not part of the manifold """
        in_manifold_indices = torch.tensor([i for i, datapoint in enumerate(self.test_data) if datapoint.in_subset])
        self.test_data = self.test_data[in_manifold_indices]
        
    def mask_val_data(self) -> None:
        """ Mask out val data not part of the manifold """
        in_manifold_indices = torch.tensor([i for i, datapoint in enumerate(self.val_data) if datapoint.in_subset])
        self.val_data = self.val_data[in_manifold_indices]
        
    def tensor_in_subset(self, tensor: torch.Tensor, tresh_hold: float) -> torch.Tensor:
        return (tensor >= tresh_hold).float()
    
    def sample(self, n_samples: int, device) -> tuple[np.ndarray, np.ndarray]:
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