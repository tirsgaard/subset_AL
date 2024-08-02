import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as kernels
from scipy.stats import norm
import scipy.optimize as opt
from src.dataset import ToySample1
from src.AL_agents import GPLearner
from tqdm import tqdm

np.random.seed(0)
"""

#global_problem = ToySample1()
#manifold_problem = SimpleManifold()

n_neighbours = 3
manifold_kernel = kernels.RBF(length_scale=1., length_scale_bounds="fixed")*kernels.ConstantKernel(constant_value=3., constant_value_bounds="fixed") + kernels.ConstantKernel(constant_value=4.5)
init_model_manifold = lambda: GPLearner(manifold_problem, manifold_kernel)
init_model_global = lambda: KNeighborsRegressor(n_neighbors=n_neighbours)
"""
n_repeats = 10
data_budget = 1000
budget_fractions_to_test = np.linspace(0.001, 0.9, 10)

def manifold_experiment(init_model_manifold, init_model_global, global_problem, manifold_problem, data_budget, budget_fractions_to_test):
    # Precompute the manifold distances for maximum number of budget samples
    X_manifold = global_problem.sample(data_budget)
    y_manifold = np.array([manifold_problem.manifold_dist(x) for x in X_manifold])

    # Define test set
    X_test = np.array([manifold_problem.manifold(t) for t in np.linspace(manifold_problem.min_range, manifold_problem.max_range, 1000)])
    y_test = global_problem.label(X_test)

    # Active learning
    pred_errors_AL = np.zeros(len(budget_fractions_to_test))
    for i, manifold_fraction in enumerate(budget_fractions_to_test):
        manifold_budget = int(manifold_fraction*data_budget)
        global_budget = data_budget - manifold_budget
        
        # Fit manifold model
        model_manifold = init_model_manifold()
        model_manifold.fit(X_manifold[:manifold_budget], y_manifold[:manifold_budget])
        X_manifold_pred = model_manifold.sample(global_budget, global_problem, 0.)
        y_manifold_pred = global_problem.label(X_manifold_pred)
        
        # Sample points from the manifold
        model_global = init_model_global()
        model_global.fit(X_manifold_pred, y_manifold_pred)
        pred_errors_AL[i] = np.mean((model_global.predict(X_test) - y_test)**2)

    # Passive learning
    X_global = X_manifold # Set the global data to be the manifold data for variance reduction
    y_global = global_problem.label(X_global)
    model_global = init_model_global()
    model_global.fit(X_global, y_global)
    pred_error_uniform = np.mean((model_global.predict(X_test) - y_test)**2)
    
    return pred_errors_AL, pred_error_uniform

run_experiment = False
if run_experiment:
    results = []
    for _ in tqdm(range(n_repeats)):
        pred_errors_AL, pred_error_uniform = manifold_experiment(init_model_manifold, 
                                                              init_model_global, 
                                                              global_problem, 
                                                              manifold_problem, 
                                                              data_budget, 
                                                              budget_fractions_to_test)
        results.append((pred_errors_AL, pred_error_uniform))
        
    pred_errors_AL = np.array([res[0] for res in results])
    pred_error_uniform = np.array([res[1] for res in results])
    print("Done")
    # Save all data to a file
    np.savez('manifold_data.npz', pred_errors_AL=pred_errors_AL, pred_error_uniform=pred_error_uniform, budget_fractions_to_test=budget_fractions_to_test)
else:
    data = np.load('manifold_data.npz')
    pred_errors_AL = data['pred_errors_AL']
    pred_error_uniform = data['pred_error_uniform']
    #budget_fractions_to_test = data['budget_fractions_to_test']
    
    
def generate_uncertainty_curve(ax, xaxis, data_sample, label, color, significance_level=0.05):
    mean = data_sample.mean(0)
    std = data_sample.std(0)
    n_root = np.sqrt(np.array([data_sample.shape[0]]))
    q_val = norm().ppf(1 - significance_level / 2)
    lines = q_val*std/n_root
    ax.plot(xaxis, mean, label=label, color=color)
    ax.fill_between(xaxis, mean - lines, mean + lines, color=color, alpha=0.2)
    
def generate_uncertainty_hline(ax, xaxis, data_sample, label, color, significance_level=0.05):
    mean = data_sample.mean(0)
    std = data_sample.std(0)
    n_root = np.sqrt(np.array([data_sample.shape[0]]))
    q_val = norm().ppf(1 - significance_level / 2)
    lines = q_val*std/n_root
    ax.axhline(mean, color=color, linestyle='--', label=label)
    ax.fill_between(xaxis, mean - lines, mean + lines, color=color, alpha=0.2)
    
# Plot error graphs
fig, axs = plt.subplots(1, 1, figsize=(15, 5))
generate_uncertainty_curve(axs, budget_fractions_to_test, pred_errors_AL, 'Active', "green")
generate_uncertainty_hline(axs, budget_fractions_to_test, pred_error_uniform, 'Uniform', "red")
axs.set_ylabel('MSE')
axs.legend()
axs.set_xlim(0, budget_fractions_to_test.max())
axs.set_xlabel('Fraction of budget on manifold')
axs.set_yscale('log')
axs.set_title('Manifold learning')

plt.show()

