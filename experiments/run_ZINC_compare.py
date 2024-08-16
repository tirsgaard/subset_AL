import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels as kernels
from scipy.stats import norm
import scipy.optimize as opt
from src.dataset import BaseDataset, ZINC250k_manifold
from src.AL_agents import GNNLearner
from src.N2GNN_helpers import PlGNNTestonValModule
from src.learning_methods import global_AL_subset_model
from pathlib import Path
import wandb
import sys
sys.path.insert(1, Path("../N2GNN").resolve().as_posix())
from models.input_encoder import EmbeddingEncoder
import train_utils
import copy
import torch
import torch.nn as nn
import torchmetrics
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

# Test code
evaluator = torchmetrics.MeanAbsoluteError()
parser = ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="ZINC", help='Name of dataset.')
parser.add_argument('--runs', type=int, default=10, help='Number of repeat run.')
parser.add_argument('--full', action="store_true", help="If true, run ZINC full." )
parser.add_argument('--save_dir', type=str, default='./save', help='Base directory for saving information.')
parser.add_argument('--config_file', type=str, default=None,
                    help='Additional configuration file for different dataset and models.')
parser.add_argument('--seed', type=int, default=234, help='Random seed for reproducibility.')

#training args
parser.add_argument('--drop_prob', type=float, default=0.0,
                    help='Probability of zeroing an activation in dropout models.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU.')
parser.add_argument('--num_workers', type=int, default=32, help='Number of worker.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
parser.add_argument('--l2_wd', type=float, default=0., help='L2 weight decay.')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
parser.add_argument('--test_eval_interval', type=int, default=10,
                    help='Interval between validation on test dataset.')
parser.add_argument('--factor', type=float, default=0.5,
                    help='Factor in the ReduceLROnPlateau learninfl rate scheduler.')
parser.add_argument('--patience', type=int, default=20,
                    help='Patience in the ReduceLROnPlateau learning rate scheduler.')
parser.add_argument("--offline", action="store_true", help="If true, save the wandb log offline. "
                                                            "Mainly use for debug.")

# data args
parser.add_argument('--policy', default="dense_ego", choices=("dense_ego",
                                                                "dense_noego",
                                                                "sparse_ego",
                                                                "sparse_noego"),
                    help="Policy of data generation in N2GNN. If dense, keep tuple that don't have any aggregation."
                            "if ego, further restrict all tuple mast have distance less than or equal to num_hops.")
parser.add_argument('--message_pool', default="plain", choices=("plain", "hierarchical"),
                    help="message pooling way in N2GNN, if set to plain, pooling all edges together. If set to"
                            "hierarchical, compute index during preprocessing for hierarchical pooling, must be used"
                            "with corresponding gnn convolutional layer.")
parser.add_argument('--reprocess', action="store_true", help='Whether to reprocess the dataset')

# model args
parser.add_argument('--gnn_name', type=str, default="GINEM", choices=("GINEC", "GINEM"),
                    help='Name of base gnn encoder.')
parser.add_argument('--model_name', type=str, default="N2GNN+",
                    choices=("N2GNN+", "N2GNN"), help='Name of GNN model.')
parser.add_argument('--tuple_size', type=int, default=5, help="Length of tuple in tuple aggregation.")
parser.add_argument('--num_hops', type=int, default=3, help="Number of hop in ego-net selection.")
parser.add_argument("--hidden_channels", type=int, default=96, help="Hidden size of the model.")
parser.add_argument("--inner_channels", type=int, default=32,
                    help="Inner channel size when doing tuple aggregation. Mainly used for reduce memory cost "
                            "during the aggregation and gradients saving.")
parser.add_argument('--wo_node_feature', action='store_true',
                    help='If true, remove node feature from model.')
parser.add_argument('--wo_edge_feature', action='store_true',
                    help='If true, remove edge feature from model.')
parser.add_argument("--edge_dim", type=int, default=0, help="Number of edge type.")
parser.add_argument("--num_layers", type=int, default=6, help="Number of layer for GNN.")
parser.add_argument("--JK", type=str, default="last",
                    choices=("sum", "max", "mean", "attention", "last", "concat"), help="Jumping knowledge method.")
parser.add_argument("--residual", action="store_true", help="If ture, use residual connection between each layer.")
parser.add_argument("--eps", type=float, default=0., help="Initial epsilon in GIN.")
parser.add_argument("--train_eps", action="store_true", help="If true, the epsilon is trainable.")
parser.add_argument("--pooling_method", type=str, default="mean", choices=("mean", "sum", "attention"),
                    help="Pooling method in graph level tasks.")
parser.add_argument('--norm_type', type=str, default="Batch",
                    choices=("Batch", "Layer", "Instance", "GraphSize", "Pair", "None"),
                    help="Normalization method in model.")
parser.add_argument('--add_rd', action="store_true", help="If true, additionally add resistance distance into model.")
args = parser.parse_args()

args.exp_name = "ZINC_subset_AL"
args.num_epochs = 100
args.hidden_channels = 64
args.inner_channels = 64
args.mode = "min"
    #training args


np.random.seed(0)
path, pre_transform, follow_batch = train_utils.data_setup(args)
post_transform = train_utils.PostTransform(args.wo_node_feature, args.wo_edge_feature)
problem = ZINC250k_manifold(pre_transform = pre_transform, transform = post_transform)


init_encoder = EmbeddingEncoder(28, args.hidden_channels)
edge_encoder = EmbeddingEncoder(4, args.inner_channels)

class_args = copy.copy(args)
class_args.dataset_name = "classification"
class_args.out_channels = 1
model_init = lambda classification: lambda: PlGNNTestonValModule(target_variable="in_subset" if classification else "y",
                                            loss_criterion=nn.BCELoss() if classification else nn.L1Loss(),
                                           evaluator=evaluator,
                                           args=class_args if classification else args,
                                           init_encoder=init_encoder,
                                           edge_encoder=edge_encoder)

loss_fn = torch.nn.MSELoss()
init_model_manifold = lambda run_name, run_number: GNNLearner(model_init(True), args=class_args, run_name=run_name, run_number=run_number)
init_model_global = lambda run_name, run_number: GNNLearner(model_init(False), args=args, run_name=run_name, run_number=run_number)

n_repeats = 10
data_budget = 1000

def calculate_accuracy(y_hat, y):
    return np.mean(y_hat.argmax(-1) == y.argmax(-1))

def manifold_experiment(init_model_subset: callable,
                        init_model_global: callable,
                        problem: BaseDataset,
                        data_budget: float,
                        manifold_budget: float = 0.25,
                        final_test: bool = False) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    # Precompute the manifold distances for maximum number of budget samples
    data_subset, index_subset = problem.sample(data_budget)
    test_data = problem.test_data if final_test else problem.val_data

    #### Single round of manifold learning followed by sampling points for training
    manifold_budget = int(manifold_budget*data_budget)
    global_budget = data_budget - manifold_budget
    model_subset = init_model_subset("single_round_manifold", 0)
    
    # Fit manifold model
    manifold_subset = data_subset[torch.arange(manifold_budget)]
    model_subset.fit(manifold_subset)
    data_manifold_pred, X_manifold_index = model_subset.sample(global_budget, problem)
    data_manifold_pred = problem.label_manifold(data_manifold_pred, X_manifold_index)
    
    # Sample points from the manifold
    model_global = init_model_global("single_round_global", 0)
    model_global.fit(data_manifold_pred,)
    pred_errors_subset = model_global.test(test_data)
    wandb.finish()

    #### Pure passive learning
    data_global = problem.label_global(data_subset, index_subset)  # Set the global data to be the manifold data for variance reduction
    model_global = init_model_global("passive_global", 0)
    model_global.fit(data_global)
    pred_error_uniform = model_global.test(test_data)
    wandb.finish()
    
    #### Active learning using subset model
    model_global = init_model_global("AL_subset_global", 0)
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global, model_subset, problem, global_budget)
    pred_errors_AL_subset = model_global.test(test_data)
    wandb.finish()
    
    #### Active learning without the subset model
    model_global = init_model_global("AL_non-subset_global", 0)
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global, model_subset, problem, global_budget)
    pred_errors_AL = model_global.test(test_data)
    wandb.finish()
    
    return pred_errors_subset, pred_error_uniform, pred_errors_AL_subset, pred_errors_AL, data_manifold_pred, X_subset_pred_potential

run_experiment = True
if run_experiment:
    results = []
    for _ in tqdm(range(n_repeats)):
        result = manifold_experiment(init_model_manifold, 
                                        init_model_global, 
                                        problem, 
                                        data_budget)
        results.append(result)
        
    pred_errors_subset = np.array([res[0] for res in results])
    pred_error_uniform = np.array([res[1] for res in results])
    pred_errors_AL_subset = np.array([res[2] for res in results])
    pred_errors_AL = np.array([res[3] for res in results])
    AL_X_manifold = np.array([res[4] for res in results])
    AL_global_X_manifold = np.array([res[5] for res in results])
    
    print("Done")
    # Save all data to a file
    np.savez('manifold_data.npz', 
             pred_errors_AL=pred_errors_subset, pred_error_uniform=pred_error_uniform, 
             pred_errors_AL_global=pred_errors_AL_subset, AL_X_manifold=AL_X_manifold, 
             AL_global_X_manifold=AL_global_X_manifold)
else:
    data = np.load('manifold_data.npz')
    pred_errors_subset = data['pred_errors_AL']
    pred_error_uniform = data['pred_error_uniform']
    pred_errors_AL_subset = data['pred_errors_AL_global']
    AL_X_manifold = data['AL_X_manifold']
    AL_global_X_manifold = data['AL_global_X_manifold']
    
def generate_uncertainty_estimate(data_sample, significance_level=0.05):
    mean = data_sample.mean(0)
    std = data_sample.std(0)
    n_root = np.sqrt(np.array([data_sample.shape[0]]))
    q_val = norm().ppf(1 - significance_level / 2)
    lines = q_val*std/n_root
    return mean, lines[0]

subset_mean, subset_bound = generate_uncertainty_estimate(pred_errors_subset)
uniform_mean, uniform_bound = generate_uncertainty_estimate(pred_error_uniform)
AL_subset_mean, AL_subset_bound = generate_uncertainty_estimate(pred_errors_AL_subset)
AL_mean, AL_bound = generate_uncertainty_estimate(pred_errors_AL)


# Print errors and uncertainty to 2 decimal places
print(f"Subset accuracy: {subset_mean:.3f} +- {subset_bound:.3f}")
print(f"Uniform accuracy: {uniform_mean:.3f} +- {uniform_bound:.3f}")
print(f"AL subset accuracy: {AL_subset_mean:.3f} +- {AL_subset_bound:.3f}")
print(f"AL accuracy: {AL_mean:.3f} +- {AL_bound:.3f}")
