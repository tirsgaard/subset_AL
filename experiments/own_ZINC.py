from load_atoms import load_dataset
import numpy as np

test = load_dataset("QM9", "/scratch/rhti")

smile_array = np.array([data.info["smiles"] for data in test])

np.save("/home/rhti/subset_AL/data/QM9_N2GNN_3_dense_ego_plain/raw/qm9_smiles.npy", smile_array)