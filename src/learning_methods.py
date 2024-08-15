import numpy as np
from src.dataset import BaseDataset
from src.AL_agents import BaseLearner

def global_AL_subset_model(model_global: BaseLearner, 
                           model_subset: BaseLearner, 
                           problem: BaseDataset, 
                           data_budget: int, 
                           n_iterations: int = 10, 
                           oversample_multiplier: float = 10.0) -> tuple[BaseLearner, np.ndarray]:
    # Schedule the number of samples to be added in each iteration
    budget_left = data_budget
    warm_up_samples = int(0.1*data_budget)
    budget_left = budget_left - warm_up_samples
    n_stage_samples = np.linspace(0, budget_left, n_iterations)
    n_stage_samples = np.diff(n_stage_samples).astype(int)
    
    # Warm up model
    data_manifold_added, manifold_indexes = model_subset.sample(n_samples = warm_up_samples, global_problem = problem)
    data_manifold_added = problem.label_global(data_manifold_added, manifold_indexes)
    for n_new_samples in n_stage_samples:
        # Fit model
        model_global.fit(data_manifold_added)
        # Sample new points
        data_subset_pred_potential, data_index = model_subset.sample(n_samples = int(oversample_multiplier*n_new_samples), global_problem = problem)
        # Select the best points
        data_selected, selected_indexes = model_global.select_samples(data_subset_pred_potential, int(n_new_samples))
        
        # Label the new points and add them to the training set
        manifold_indexes = np.concatenate([manifold_indexes, data_index[selected_indexes]])
        data_manifold_added = data_manifold_added + data_selected
        data_manifold_added = problem.label_global(data_manifold_added, manifold_indexes)
        
    # Fit model
    
    model_global.fit(data_manifold_added)
    return model_global, data_manifold_added