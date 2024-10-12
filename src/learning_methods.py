import numpy as np
from src.dataset import BaseDataset
from src.AL_agents import BaseLearner, EnsembleLearner
import torch
import wandb
from typing import Optional, Dict

def global_AL_subset_model(model_global_init: BaseLearner, 
                           model_subset: BaseLearner, 
                           problem: BaseDataset, 
                           data_budget: int, 
                           n_iterations: int = 10, 
                           oversample_multiplier: float = 10.0,
                           model_global_state: Optional[Dict] = None) -> tuple[BaseLearner, np.ndarray]:
    # Schedule the number of samples to be added in each iteration
    budget_left = data_budget
    warm_up_samples = int(0.1*data_budget)
    budget_left = budget_left - warm_up_samples
    n_stage_samples = np.linspace(0, budget_left, n_iterations)
    n_stage_samples = np.diff(n_stage_samples).astype(int)
    
    # Warm up model
    data_manifold_added, manifold_indexes = model_subset.sample(n_samples = warm_up_samples, global_problem = problem)
    data_manifold_added = problem.label_global(data_manifold_added, manifold_indexes)
    for i, n_new_samples in enumerate(n_stage_samples):
        # Fit model
        model_global = model_global_init(i)
        if model_global_state is None:
            model_global_state = model_global.model.state_dict().copy()
        model_global.model.load_state_dict(model_global_state)
        model_global.fit(data_manifold_added)
        wandb.finish()
        # Sample new points
        data_subset_pred_potential, data_index = model_subset.sample(n_samples = int(oversample_multiplier*n_new_samples), global_problem = problem)
        # Select the best points
        data_selected, selected_indexes = model_global.select_samples(data_subset_pred_potential, int(n_new_samples))
        
        # Label the new points and add them to the training set
        manifold_indexes = np.concatenate([manifold_indexes, data_index[selected_indexes]])
        data_manifold_added = data_manifold_added + data_selected
        data_manifold_added = problem.label_global(data_manifold_added, manifold_indexes)
    
    # Fit model
    model_global = model_global_init("final")
    model_global.model.load_state_dict(model_global_state)
    model_global.fit(data_manifold_added)
    return model_global, data_manifold_added

def cross_validate(model_init: BaseLearner, data: np.ndarray, validation_size: int, n_runs: int, device="cpu", return_models: bool = False) -> np.ndarray:
    # Split the data into n_runs with random validation sets of size validation_size
    model_list = []
    validation_scores = np.zeros(n_runs)
    for i in range(n_runs):
        # Randomly shuffle the data
        shuffled_data = data.shuffle()
        # Split the data
        validation_data = shuffled_data[:validation_size]
        training_data = shuffled_data[validation_size:]
        # Fit the model
        model_subset = model_init(f"cross_validate_model_{i}", device)
        model_subset.fit(training_data)
        # Evaluate the model
        model_subset.model.eval()
        with torch.no_grad():
            y_hat = model_subset.predict(validation_data)
        if return_models:
            model_list.append(model_subset)
        
        validation_scores[i] = (1-validation_data.in_subset)*y_hat + validation_data.in_subset*(1-y_hat)
    return validation_scores, model_list


def upper_bound_subset_model(model_subset_init: BaseLearner, 
                           problem: BaseDataset, 
                           data_budget: int, 
                           start_budget: int = 10,
                           batch_sizes: int = 10,
                           runs: int = 10,
                           alpha: Optional[float] = None,
                           model_subset_state: Optional[Dict] = None, 
                           run_ensemble: bool = False,
                           device="cpu") -> tuple[BaseLearner, np.ndarray]:
    # Warm up model
    data_manifold_added, manifold_indexes = problem.sample(n_samples = start_budget, device=device)
    data_manifold_added = problem.label_global(data_manifold_added, manifold_indexes)
    used_budget = start_budget 
    
    change = lambda previous_accuracy, new_accuracy, used_budget, batch_sizes: previous_accuracy - (data_budget - used_budget)*(new_accuracy - previous_accuracy)/batch_sizes
    if alpha is not None:
        stop_criteria = lambda previous_accuracy, new_accuracy, used_budget, batch_sizes: (1 - alpha)*change(previous_accuracy, new_accuracy, used_budget, batch_sizes) - 1 < 0
    else:
        stop_criteria = lambda previous_accuracy, new_accuracy, used_budget, batch_sizes: change(previous_accuracy, new_accuracy, used_budget, batch_sizes) < 0    
    # Start accuracy
    previous_accuracy, _ = cross_validate(model_subset_init, data_manifold_added, validation_size = 1, n_runs = runs, device=device)
    previous_accuracy = previous_accuracy.mean()
    while True:
        # Sample new points
        data_selected, selected_indexes = problem.sample(n_samples = batch_sizes, device=device)
        used_budget += batch_sizes
        
        # Label the new points and add them to the training set
        manifold_indexes = np.concatenate([manifold_indexes, selected_indexes])
        data_manifold_added = problem.train_data[manifold_indexes]  # We can't concatenate if we want to use the same data structure
        
        # Evaluate the model
        new_accuracy, model_list = cross_validate(model_subset_init, data_manifold_added, validation_size = 1, n_runs = runs, device=device, return_models=run_ensemble)
        new_accuracy = new_accuracy.mean()
        # Check stopping criteria
        if stop_criteria(previous_accuracy, new_accuracy, used_budget, batch_sizes):
            break
        
    # Fit model
    if not run_ensemble:
        model_global = model_subset_init("upper_bound_model", device)
        if model_subset_state is not None:
            model_global.model.load_state_dict(model_subset_state)
        model_global.fit(data_manifold_added)
    else:
        model_global = EnsembleLearner(model_list)

    return model_global, data_manifold_added