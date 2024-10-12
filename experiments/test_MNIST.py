import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels as kernels
from src.NN_models import ResNetModel
from scipy.stats import norm
import scipy.optimize as opt
from src.dataset import BaseDataset, MNIST_manifold
from src.AL_agents import CNNLearner, RandomLearner
from src.N2GNN_helpers import PlGNNTestonValModule
from src.learning_methods import global_AL_subset_model, upper_bound_subset_model
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
from pathlib import Path
import wandb
from time import time
import sys
from lightning.pytorch import seed_everything 
from copy import deepcopy
import pickle
import torch
import torch.nn as nn
import torchmetrics
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import joblib
import os

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
parser.add_argument("--offline", action="store_true", help="If true, save the wandb log offline. "
                                                            "Mainly use for debug.")
parser.add_argument('--debug', action="store_true", help='If true, run in debug mode.')

#training args
parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU.')
parser.add_argument('--num_workers', type=int, default=16, help='Number of worker.')
parser.add_argument('--n_datapoints', type=int, default=1000, help='Number of datapoints to run over.')
parser.add_argument('--subset_quantile', type=float, default=0.9, help='Number of datapoints to run over.')
parser.add_argument('--manifold_fraction', type=float, default=0.25, help='Fraction of data to use for manifold learning.')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')
parser.add_argument('--l2_wd', type=float, default=0., help='L2 weight decay.')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs.')
args = parser.parse_args()

args.exp_name = "MNIST" if not args.debug else "MNIST_debug"
args.num_epochs = 1000 if not args.debug else 100
args.num_workers = 4
args.patience = 500
args.lr_patience = 50
args.step_size = 50
args.gamma = 0.1
args.lr_scheduler = "StepLR"
args.l2_wd = 0.005
n_repeats = 10 if not args.debug else 2

args.group = "Perfect_manifold_weight_decay"#str(int(time()))
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "offline"
print("starting worker")

seed_everything(0, workers=True)
start_time = int(time())

problem = MNIST_manifold()

class_args = deepcopy(args)
class_args.dataset_name = "classification"
class_args.out_channels = 1
#class_args.pooling_method = "sum"  # For enabling the classifciation model to count the number of nodes in the graph
def model_init(is_subset_model):
    def init_encoder2():
        return ResNetModel(1, 1 if is_subset_model else 9)
                                            
    return init_encoder2

# The above code does not work for multiprocessing, so we need to define the functions in the main function
def init_model_manifold(run_name, device):
    return CNNLearner(model_init(True), args=class_args, run_name=run_name, is_subset_model=True, device=device)

def init_model_global(run_name, device):
    return CNNLearner(model_init(False), args=class_args, run_name=run_name, is_subset_model=False, device=device)

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
    #gpu_id = (run_id % torch.cuda.device_count(), ) if torch.cuda.is_available() else "auto"
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    print(f"Running on device {device}")
    # Precompute the manifold distances for maximum number of budget samples
    data_subset, index_subset = problem.sample(data_budget, device=device)
    test_data = problem.test_data if final_test else problem.val_data

    #### Single round of manifold learning followed by sampling points for training
    manifold_budget = int(manifold_budget*data_budget)
    global_budget = data_budget - manifold_budget
    model_subset = init_model_subset(f"single_round_manifold_run_{run_id}", device)
    
    data_manifold_pred = problem.label_manifold(data_subset, index_subset)
    manifold_accuracy = torch.mean(data_manifold_pred.in_subset.float()).detach().item()
    
    # Sample points from the manifold
    model_global = init_model_global(f"single_round_global_run_{run_id}", device)
    if len(data_manifold_pred) > 0:
        val_accs = model_global.fit(data_manifold_pred, test_data, 10)
    pred_errors_subset = model_global.test(test_data)
    print(f"Subset error: {pred_errors_subset:.3f}")
    wandb.finish()
    
    return pred_errors_subset, manifold_accuracy#, upper_errors_subset, #pred_error_uniform, pred_errors_AL_subset, pred_errors_AL#, data_manifold_pred, X_subset_pred_potential

if __name__ == "__main__":
    save_path = Path(args.save_dir) / Path('manifold_data' + str(start_time) + '.pkl')
    run_experiment = True
    if run_experiment:
        n_workers = 1 #torch.cuda.device_count() if torch.cuda.is_available() else 1
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
                                                    manifold_budget=args.manifold_fraction,
                                                    run_id=i)
                                                    for i in range(n_repeats))
        output = {"results": results,
                  "args": args,
                  "time": time() - start_time,
                  "start_time": start_time,
                  "end_time": time(),
                  }
        print("Saving results.......")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        print("Loading results.......")
        with open(save_path, 'rb') as f:
            output = pickle.load(f)
        results = output["results"]
        
    pred_errors_subset = np.array([res[0] for res in results])
    #pred_error_uniform = np.array([res[1] for res in results])
    #pred_errors_upper = np.array([res[2] for res in results])
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
    #uniform_mean, uniform_bound = generate_uncertainty_estimate(pred_error_uniform)
    #subset_upper_mean, subset_upper_bound = generate_uncertainty_estimate(pred_errors_upper)
    #AL_subset_mean, AL_subset_bound = generate_uncertainty_estimate(pred_errors_AL_subset)
    #AL_mean, AL_bound = generate_uncertainty_estimate(pred_errors_AL)


    # Print errors and uncertainty to 2 decimal places
    print(f"Subset error: {subset_mean:.3f} +- {subset_bound:.3f}")
    
