# Test the ZINC250k dataset
from src.dataset import BaseDataset, ZINC250k_manifold
import smilite
import requests
from tqdm import tqdm
import numpy as np
import rdkit

problem = ZINC250k_manifold(pre_transform = None, transform = None, full_dataset=True)

n_test = 100

def test_split_web(split, n_test: int):
    train_testing_id = np.arange(len(split))
    np.random.shuffle(train_testing_id)
    train_testing_id = train_testing_id[:n_test]
    for i in tqdm(train_testing_id):
        test_string = split[i].smiles
        test = smilite.get_zincid_from_smile(test_string)
        if len(test) == 0:
            continue
        for id in test:
            url = f"https://zinc15.docking.org/substances/{id}.json?output_fields=num_atoms"
            response = requests.get(url)
            try:
                data = response.json()
                break
            except requests.exceptions.JSONDecodeError as e:
                continue
            
        num_atoms = data['num_atoms']
        assert num_atoms == len(split[i].node2graph)
        
def test_split_rdkit(split):
    for data in tqdm(split):
        mol = rdkit.Chem.MolFromSmiles(data.smiles)
        assert mol.GetNumAtoms() == len(data.node2graph)

# Test local
test_split_rdkit(problem.train_data)
test_split_rdkit(problem.val_data)
test_split_rdkit(problem.test_data)

# Test web
test_split_web(problem.train_data, n_test)
test_split_web(problem.val_data, n_test)
test_split_web(problem.test_data, n_test)
