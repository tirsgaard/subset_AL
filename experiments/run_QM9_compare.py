import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels as kernels
from scipy.stats import norm
import scipy.optimize as opt
from src.dataset import BaseDataset, QM9_manifold
from src.AL_agents import GNNLearner, RandomLearner
from src.N2GNN_helpers import PlGNNTestonValModule
import torch_geometric.transforms as T
from torch_geometric.data import Data

from pathlib import Path
import wandb
from time import time
import sys
sys.path.insert(1, Path("../N2GNN").resolve().as_posix())
from models.input_encoder import EmbeddingEncoder, QM9InputEncoder
from datasets.QM9Dataset import conversion
import train_utils
from lightning.pytorch import seed_everything 
import copy
import pickle
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError
from torchmetrics.functional.regression.mae import _mean_absolute_error_compute

from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import joblib
import os

class InputTransform(object):
    """QM9 input feature transformation. Concatenate x and z together.
    """
    def __init__(self):
        super().__init__()

    def __call__(self,
                 data: Data) -> Data:
        x = data.x
        z = data.z
        data.x = torch.cat([z.unsqueeze(-1), x], dim=-1)
        data.edge_attr = torch.where(data.edge_attr == 1)[-1]
        return data


class MeanAbsoluteErrorQM9(MeanAbsoluteError):
    def __init__(self,
                 std,
                 conversion,
                 **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.conversion = conversion

    def compute(self) -> torch.Tensor:
        return (_mean_absolute_error_compute(self.sum_abs_error, self.total) * self.std) / self.conversion


# Test code
parser = ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="QM9", help='Name of dataset.')
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
parser.add_argument('--num_workers', type=int, default=16, help='Number of worker.')
parser.add_argument('--n_datapoints', type=int, default=100, help='Number of datapoints to run over.')
parser.add_argument('--subset_quantile', type=float, default=0.9, help='Number of datapoints to run over.')
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
parser.add_argument('--lr_patience', type=int, default=20,
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
parser.add_argument('--search', action="store_true", help="If true, run all first 12 targets.")

# model args
parser.add_argument('--gnn_name', type=str, default="GINEM", choices=("GINEC", "GINEM"),
                    help='Name of base gnn encoder.')
parser.add_argument('--lr_scheduler', type=str, default="ReduceLROnPlateau", choices=("ReduceLROnPlateau", "StepLR"),
                help='Learning rate scheduler.')
parser.add_argument('--group', type=str, default="N2GNN", help='Group name of the experiment.')
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
parser.add_argument('--task', type=int, default=11, choices=list(range(19)), help='Train target.')
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

debug_mode = False
args.exp_name = "QM9_synthesizeability" if not debug_mode else "QM9_debug"
args.num_epochs = 1000 if not debug_mode else 2
args.mode = "min"
args.num_workers = 16
args.patience = 500
args.lr_patience = 50
args.step_size = 50
args.gamma = 0.1
args.lr_scheduler = "StepLR"
args.lr = 0.001
args.l2_wd = 0.005
args.group = "Perfect_manifold_weight_decay"#str(int(time()))
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "offline"
print("starting worker")

seed_everything(0, workers=True)
path, pre_transform, follow_batch = train_utils.data_setup(args)
post_transform = train_utils.PostTransform(args.wo_node_feature, 
                                           args.wo_edge_feature, 
                                           args.task)
start_time = int(time())
problem = QM9_manifold("/scratch/rhti/QM9",
                        pre_transform = T.Compose([InputTransform(), pre_transform]), 
                       transform = post_transform,
                       quantile_threshold=args.subset_quantile)

init_encoder = QM9InputEncoder(args.hidden_channels)
edge_encoder = EmbeddingEncoder(4, args.inner_channels)
evaluator = MeanAbsoluteErrorQM9(problem.train_std[args.task].item(), conversion[args.task].item())


class_args = copy.copy(args)
class_args.dataset_name = "classification"
class_args.out_channels = 1
class_args.pooling_method = "sum"  # For enabling the classifciation model to count the number of nodes in the graph
"""
model_init = lambda classification: lambda: PlGNNTestonValModule(target_variable="in_subset" if classification else "y",
                                            loss_criterion=nn.BCELoss() if classification else nn.L1Loss(),
                                        evaluator=evaluator,
                                        args=class_args if classification else args,
                                        init_encoder=init_encoder,
                                        edge_encoder=edge_encoder,
                                        is_classification=classification)


init_model_manifold = lambda run_name, device: GNNLearner(model_init(True), args=class_args, run_name=run_name, device=device)
init_model_global = lambda run_name, device: GNNLearner(model_init(False), args=args, run_name=run_name, device=device)
"""
def model_init(classification):
    
    def init_encoder2():
        return PlGNNTestonValModule(target_variable="in_subset" if classification else "y",
                                                loss_criterion=nn.BCELoss() if classification else nn.MSELoss(),
                                            evaluator=evaluator,
                                            args=class_args if classification else args,
                                            init_encoder=init_encoder,
                                            edge_encoder=edge_encoder,
                                            is_classification=classification)
                                            
    return init_encoder2

# The above code does not work for multiprocessing, so we need to define the functions in the main function
def init_model_manifold(run_name, device):
    return GNNLearner(model_init(True), args=class_args, run_name=run_name, device=device)

def init_model_global(run_name, device):
    return GNNLearner(model_init(False), args=args, run_name=run_name, device=device)

n_repeats = 20 if not debug_mode else 2
data_budget = args.n_datapoints
print(f"Running {n_repeats} repeats with {data_budget} data points")

def manifold_experiment(init_model_subset: callable,
                        init_model_global: callable,
                        problem: BaseDataset,
                        data_budget: int,
                        manifold_budget: float = 0.25,
                        final_test: bool = False,
                        run_id: int = 0) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    print(f"Running experiment {run_id}")
    seed_everything(run_id, workers=True)
    gpu_id = (run_id % torch.cuda.device_count(), ) if torch.cuda.is_available() else "auto"
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Running on device {device}")
    # Precompute the manifold distances for maximum number of budget samples
    data_subset, index_subset = problem.sample(data_budget, device=device)
    test_data = problem.test_data if final_test else problem.val_data

    #### Single round of manifold learning followed by sampling points for training
    manifold_budget = int(manifold_budget*data_budget)
    global_budget = data_budget  # - manifold_budget # TODO REMOVE THIS!
    model_subset = init_model_subset(f"single_round_manifold_run_{run_id}", gpu_id)
    
    # Fit manifold model
    manifold_subset = data_subset[torch.arange(data_budget)]
    #model_subset.fit(manifold_subset)
    data_manifold_pred, X_manifold_index = model_subset.sample(global_budget, problem, ensure_subset=True)
    data_manifold_pred = problem.label_manifold(data_manifold_pred, X_manifold_index)
    wandb.finish()
    
    # Sample points from the manifold
    model_global = init_model_global(f"single_round_global_run_{run_id}", gpu_id)
    model_global_parameters = model_global.model.state_dict().copy()
    model_global.fit(data_manifold_pred,)
    pred_errors_subset = model_global.test(test_data)
    print(f"Subset accuracy: {pred_errors_subset[0]['test/metric']:.3f}")
    wandb.finish()

    #### Pure passive learning
    seed_everything(run_id, workers=True)
    data_global = problem.label_global(data_subset, index_subset)  # Set the global data to be the manifold data for variance reduction
    model_global = init_model_global(f"passive_global_run_{run_id}", gpu_id)
    model_global.to(device)
    model_global.model.load_state_dict(model_global_parameters)  # Reset the global model to the initial state to ensure fair comparison
    model_global.fit(data_global)
    pred_error_uniform = model_global.test(test_data)
    print(f"Uniform accuracy: {pred_error_uniform[0]['test/metric']:.3f}")
    wandb.finish()
    """
    ### Mix data from the manifold and the global data
    seed_everything(run_id, workers=True)
    model_global = init_model_global(f"mixed_global_run_{run_id}", gpu_id)
    model_global.to(device)
    model_global.model.load_state_dict(model_global_parameters)  # Reset the global model to the initial state to ensure fair comparison
    data_ratio = 0.2
    data_mixed_idx = np.concatenate([X_manifold_index[:int(data_budget*(1-data_ratio))], index_subset[:int(data_budget*data_ratio)]])
    np.random.shuffle(data_mixed_idx)
    data_mixed = problem.train_data[data_mixed_idx]
    model_global.fit(data_mixed)
    pred_error_mixed = model_global.test(test_data)
    print(f"Mixed accuracy: {pred_error_mixed[0]['test/metric']:.3f}")
    wandb.finish()


    #### Active learning using subset model
    seed_everything(run_id, workers=True)
    model_global_init = lambda loop_id: init_model_global(f"AL_subset_global_run_{run_id}_loop_{loop_id}", gpu_id)
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global_init, model_subset, problem, global_budget, n_iterations=5, model_global_state=model_global_parameters)
    pred_errors_AL_subset = model_global.test(test_data)
    print(f"AL subset accuracy: {pred_errors_AL_subset[0]['test/metric']:.3f}")
    wandb.finish()
    
    #### Active learning without the subset model
    seed_everything(run_id, workers=True)
    model_global_init = lambda loop_id: init_model_global(f"AL_non-subset_global_run_{run_id}_loop_{loop_id}", gpu_id)
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global_init, RandomLearner(problem), problem, global_budget, n_iterations=5, model_global_state=model_global_parameters)
    pred_errors_AL = model_global.test(test_data)
    print(f"AL accuracy: {pred_errors_AL[0]['test/metric']:.3f}")
    wandb.finish()
    """
    return pred_errors_subset, pred_error_uniform, #pred_error_mixed, pred_errors_AL_subset, pred_errors_AL#, data_manifold_pred, X_subset_pred_potential

if __name__ == "__main__":
    save_path = Path(args.save_dir) / Path('manifold_data' + str(start_time) + '.pkl')
    run_experiment = True
    if run_experiment:
        n_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
        print(f"Running experiment with {n_workers} workers")
        import os
        import psutil
        pid = os.getpid()
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
        print('memory use:', memoryUse)
        results = joblib.Parallel(n_jobs=n_workers, backend='threading')(
                                                joblib.delayed(manifold_experiment)
                                                    (init_model_manifold, 
                                                    init_model_global, 
                                                    problem, 
                                                    data_budget,
                                                    manifold_budget=0.25,
                                                    run_id=i)
                                                    for i in range(n_repeats))
        print("Saving results.......")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        print("Loading results.......")
        with open(save_path, 'rb') as f:
            results = pickle.load(f)
        
    pred_errors_subset = np.array([res[0][0]["test/metric"] for res in results])
    pred_error_uniform = np.array([res[1][0]["test/metric"] for res in results])
    #pred_errors_AL_subset = np.array([res[2][0]["test/metric"] for res in results])
    #pred_errors_AL = np.array([res[3][0]["test/metric"] for res in results])
    #AL_X_manifold = [res[4] for res in results]
    #AL_global_X_manifold = [res[5] for res in results]
        
    def generate_uncertainty_estimate(data_sample, significance_level=0.05):
        mean = data_sample.mean(0)
        std = data_sample.std(0)
        n_root = np.sqrt(np.array([data_sample.shape[0]]))
        q_val = norm().ppf(1 - significance_level / 2)
        lines = q_val*std/n_root
        return mean, lines[0]

    subset_mean, subset_bound = generate_uncertainty_estimate(pred_errors_subset)
    uniform_mean, uniform_bound = generate_uncertainty_estimate(pred_error_uniform)
    #AL_subset_mean, AL_subset_bound = generate_uncertainty_estimate(pred_errors_AL_subset)
    #AL_mean, AL_bound = generate_uncertainty_estimate(pred_errors_AL)


    # Print errors and uncertainty to 2 decimal places
    print(f"Subset accuracy: {subset_mean:.3f} +- {subset_bound:.3f}")
    print(f"Uniform accuracy: {uniform_mean:.3f} +- {uniform_bound:.3f}")
    #print(f"AL subset accuracy: {AL_subset_mean:.3f} +- {AL_subset_bound:.3f}")
    #print(f"AL accuracy: {AL_mean:.3f} +- {AL_bound:.3f}")


    print("Switching to PL normalized performance")

    subset_mean, subset_bound = generate_uncertainty_estimate(pred_errors_subset - pred_error_uniform)
    uniform_mean, uniform_bound = generate_uncertainty_estimate(pred_error_uniform - pred_error_uniform)
    #AL_subset_mean, AL_subset_bound = generate_uncertainty_estimate(pred_errors_AL_subset - pred_error_uniform)
    #AL_mean, AL_bound = generate_uncertainty_estimate(pred_errors_AL - pred_error_uniform)


    # Print errors and uncertainty to 2 decimal places
    print(f"Subset accuracy: {subset_mean:.3f} +- {subset_bound:.3f}")
    print(f"Uniform accuracy: {uniform_mean:.3f} +- {uniform_bound:.3f}")
    #print(f"AL subset accuracy: {AL_subset_mean:.3f} +- {AL_subset_bound:.3f}")
    #print(f"AL accuracy: {AL_mean:.3f} +- {AL_bound:.3f}")
