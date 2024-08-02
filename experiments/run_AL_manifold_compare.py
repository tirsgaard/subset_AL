import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import kernels as kernels
from scipy.stats import norm
import scipy.optimize as opt
from src.dataset import ToySample1, BaseDataset
from src.AL_agents import GPLearner, KNNLearner, BaseLearner
from src.learning_methods import global_AL_subset_model
from tqdm import tqdm

np.random.seed(0)
problem = ToySample1(2)

n_neighbours = 3
manifold_kernel = kernels.RBF(length_scale=1., length_scale_bounds="fixed")*kernels.ConstantKernel(constant_value=3., constant_value_bounds="fixed") + kernels.ConstantKernel(constant_value=4.5)
init_model_manifold = lambda: GPLearner(problem, manifold_kernel, treshold=0.0)
init_model_global = lambda: KNNLearner(problem, n_neighbors=n_neighbours)

n_repeats = 10
data_budget = 1000

def manifold_experiment(init_model_subset: callable,
                        init_model_global: callable,
                        problem: BaseDataset,
                        data_budget: float,
                        manifold_budget: float = 0.25) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    # Precompute the manifold distances for maximum number of budget samples
    X_manifold, X_index = problem.sample(data_budget)
    y_manifold = problem.label_manifold(X_manifold, X_index)

    # Define test set
    t = np.linspace(problem.min_val, problem.max_val, 10000)
    X_test, X_test_index = problem.manifold(t), None
    y_test = problem.label_global(X_test, X_test_index)

    # Active learning
    manifold_budget = int(manifold_budget*data_budget)
    global_budget = data_budget - manifold_budget
    # Fit manifold model
    model_subset = init_model_subset()
    model_subset.fit(X_manifold[:manifold_budget], y_manifold[:manifold_budget])
    X_manifold_pred, X_manifold_index = model_subset.sample(global_budget, problem)
    y_manifold_pred = problem.label_global(X_manifold_pred, X_manifold_index)
    
    # Sample points from the manifold
    model_global = init_model_global()
    model_global.fit(X_manifold_pred, y_manifold_pred)
    pred_errors_subset = np.mean((model_global.predict(X_test) - y_test)**2)

    # Passive learning
    X_global = X_manifold # Set the global data to be the manifold data for variance reduction
    y_global = problem.label_global(X_global, None)
    model_global = init_model_global()
    model_global.fit(X_global, y_global)
    pred_error_uniform = np.mean((model_global.predict(X_test) - y_test)**2)
    
    # Active learning using subset model
    model_global = init_model_global()
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global, model_subset, problem, global_budget)
    pred_errors_AL_subset = np.mean((model_global.predict(X_test) - y_test)**2)
    
    # Active learning without the subset model
    model_global = init_model_global()
    model_global, X_subset_pred_potential = global_AL_subset_model(model_global, problem, problem, global_budget)
    pred_errors_AL = np.mean((model_global.predict(X_test) - y_test)**2)
    
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
             pred_errors_subset=pred_errors_subset, pred_error_uniform=pred_error_uniform,
            pred_errors_AL_subset=pred_errors_AL_subset, pred_errors_AL=pred_errors_AL,
             AL_X_manifold=AL_X_manifold, 
             AL_global_X_manifold=AL_global_X_manifold)
else:
    data = np.load('manifold_data.npz')
    pred_errors_subset = data['pred_errors_AL']
    pred_error_uniform = data['pred_error_uniform']
    pred_errors_AL_subset = data['pred_errors_AL_subset']
    pred_errors_AL = data['pred_errors_AL']
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
print(f"Subset error: {subset_mean:.3f} +- {subset_bound:.3f}")
print(f"Uniform error: {uniform_mean:.3f} +- {uniform_bound:.3f}")
print(f"AL subset error: {AL_subset_mean:.3f} +- {AL_subset_bound:.3f}")
print(f"AL error: {AL_mean:.3f} +- {AL_bound:.3f}")


# Plot the points sampled under the different strategies
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
X_manifold = np.array([problem.manifold(t) for t in np.linspace(problem.min_val, problem.max_val, 1000)])
exp_index = 0
ax[0].scatter(AL_X_manifold[exp_index][:, 0], AL_X_manifold[exp_index][:, 1], label="AL", s=5)
ax[0].plot(X_manifold[:, 0], X_manifold[:, 1], label="Manifold", color="black")
ax[0].set_title("Active learning")
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)

ax[1].scatter(AL_global_X_manifold[exp_index][:, 0], AL_global_X_manifold[exp_index][:, 1], label="AL global", s=5)
ax[1].plot(X_manifold[:, 0], X_manifold[:, 1], label="Manifold", color="black")
ax[1].set_title("Active learning on the manifold")
ax[1].set_xlim(-10, 10)
ax[1].set_ylim(-10, 10)

# Make a 2D histogram plot over all data points
X_flat = AL_X_manifold.reshape(-1, 2)
ax[2].hist2d(X_flat[:, 0], X_flat[:, 1], bins=50, cmap="viridis")
ax[2].plot(X_manifold[:, 0], X_manifold[:, 1], label="Manifold", color="black")
ax[2].set_title("Overall sampling density")
ax[2].set_xlim(-10, 10)
ax[2].set_ylim(-10, 10)
plt.show()
