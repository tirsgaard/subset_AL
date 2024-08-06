import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels as kernels
from scipy.stats import norm
import scipy.optimize as opt
from src.dataset import BaseDataset, ZINC250k_manifold
from src.AL_agents import GPLearner, KNNLearner, BaseLearner, NNLearner, GNNLearner
from src.learning_methods import global_AL_subset_model
import torch
from torchdrug import models, tasks, data, datasets
from tqdm import tqdm
from pathlib import Path

np.random.seed(0)
dataset_save_path = Path("data/ZINC250k/ZINC250k.pkl").resolve()
problem = ZINC250k_manifold(lazy=True)

# Test code
model = tasks.PropertyPrediction(models.GIN(problem.node_feature_dim, hidden_dims=[256, 256, 256, 256]), task=problem.tasks,
                                criterion="bce", metric=("auprc", "auroc"))

loss_fn = torch.nn.MSELoss()
classification_loss = torch.nn.CrossEntropyLoss()
model_initialiser = lambda: tasks.PropertyPrediction(models.GIN(problem.node_feature_dim, hidden_dims=[256, 256, 256, 256]), task=problem.tasks,
                                criterion="bce", metric=("auroc"), num_class=1)
init_model_manifold = lambda: GNNLearner(model_initialiser, loss=loss_fn)
init_model_global = lambda: GNNLearner(model_initialiser, loss=loss_fn)

n_repeats = 10
data_budget = 100

def calculate_accuracy(y_hat, y):
    return np.mean(y_hat.argmax(-1) == y.argmax(-1))

def manifold_experiment(init_model_subset: callable,
                        init_model_global: callable,
                        problem: BaseDataset,
                        data_budget: float,
                        manifold_budget: float = 0.25) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    # Precompute the manifold distances for maximum number of budget samples
    X_manifold, X_index = problem.sample(data_budget)
    y_manifold = problem.label_manifold(X_manifold, X_index)

    # Active learning
    manifold_budget = int(manifold_budget*data_budget)
    global_budget = data_budget - manifold_budget
    # Fit manifold model
    model_subset = init_model_subset()
    model_subset.fit(X_manifold[:manifold_budget], y_manifold[:manifold_budget])
    X_manifold_pred, X_manifold_index = model_subset.sample(global_budget, problem)
    y_manifold_pred = problem.label_global(X_manifold_pred, X_manifold_index)
    
    # Define test set
    X_test, y_test = problem.X_test, problem.y_test
    
    # Sample points from the manifold
    model_global = init_model_global()
    model_global.fit(X_manifold_pred, y_manifold_pred)
    pred_errors_subset = calculate_accuracy(model_global.predict(X_test), y_test)

    # Passive learning
    X_global = X_manifold # Set the global data to be the manifold data for variance reduction
    y_global = problem.label_global(X_global, X_index)
    model_global = init_model_global()
    model_global.fit(X_global, y_global)
    pred_error_uniform = calculate_accuracy(model_global.predict(X_test), y_test)
    
    # Active learning using subset model
    model_global = init_model_global()
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global, model_subset, problem, global_budget)
    pred_errors_AL_subset = calculate_accuracy(model_global.predict(X_test), y_test)
    
    # Active learning without the subset model
    model_global = init_model_global()
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global, model_subset, problem, global_budget)
    pred_errors_AL = calculate_accuracy(model_global.predict(X_test), y_test)
    
    return pred_errors_subset, pred_error_uniform, pred_errors_AL_subset, pred_errors_AL, X_manifold_pred, X_subset_pred_potential

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
