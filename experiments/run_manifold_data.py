import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels as kernels
from scipy.stats import norm
import scipy.optimize as opt
from src.dataset import ToySample1, SimpleManifold
from src.AL_agents import GPLearner

np.random.seed(0)


global_problem = ToySample1()
manifold_problem = SimpleManifold()

n_neighbours = 3
manifold_kernel = kernels.RBF(length_scale=1., length_scale_bounds="fixed")*kernels.ConstantKernel(constant_value=3., constant_value_bounds="fixed") + kernels.ConstantKernel(constant_value=4.5)
model_manifold = GPLearner(manifold_problem, manifold_kernel)
model_global = KNeighborsRegressor(n_neighbors=n_neighbours)

N_manifold_budget = 100
N_global_budget = 1000

X_manifold = global_problem.sample(N_manifold_budget)
y_manifold = np.array([manifold_problem.manifold_dist(x) for x in X_manifold])
model_manifold.fit(X_manifold, y_manifold)

X_manifold_pred = model_manifold.sample(N_global_budget, global_problem, 0.)
y_manifold_pred = global_problem.label(X_manifold_pred)

X_global = global_problem.sample(N_global_budget)
y_global = global_problem.label(X_global)

X_test = np.array([manifold_problem.manifold(t) for t in np.linspace(manifold_problem.min_range, manifold_problem.max_range, 1000)])
y_test = global_problem.label(X_test)

point_fits = np.linspace(n_neighbours, N_global_budget, 10).astype(int)
pred_errors_uniform = np.zeros(len(point_fits))

# Passive learning
for i, n_points in enumerate(point_fits):
    model_global.fit(X_global[:n_points], y_global[:n_points])
    pred_errors_uniform[i] = np.mean((model_global.predict(X_test) - y_test)**2)
    
# Active learning
pred_errors_AL = np.zeros(len(point_fits))
for i, n_points in enumerate(point_fits):
    # Sample points from the manifold
    model_global.fit(X_manifold_pred[:n_points], y_manifold_pred[:n_points])
    pred_errors_AL[i] = np.mean((model_global.predict(X_test) - y_test)**2)
    
print("Done")
# Plot error graphs
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(point_fits, pred_errors_uniform, label='Uniform')
axs[0].plot(point_fits, pred_errors_AL, label='Active')
axs[0].legend()
axs[0].set_xlabel('N points')
axs[0].set_ylabel('MSE')
axs[0].set_xscale('log')
axs[0].set_yscale('log')

# plot sampled points
axs[1].scatter(X_global[:, 0], X_global[:, 1], c='r', label='Uniform')
axs[1].scatter(X_manifold_pred[:, 0], X_manifold_pred[:, 1], c='b', label='Manifold')
axs[1].plot(X_test[:, 0], X_test[:, 1], 'r')
axs[1].legend()
axs[1].set_xlabel('N points')
axs[1].set_ylabel('MSE')

# Plot global problem
x_grid = np.linspace(-10, 10, 100)
y_grid = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = global_problem.label(np.stack([X, Y], -1))
axs[2].contourf(X, Y, Z)
# Draw the manifold
t = np.linspace(-10, 10, 100)
manifold_points = manifold_problem.manifold(t)
axs[2].plot(manifold_points[0, :], manifold_points[1, :], 'r')
plt.savefig('manifold_data.svg')
plt.show()

# Plot the manifold fit
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# Show the manifold model's predictions on a grid
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
grid = np.stack([X.flatten(), Y.flatten()], -1)
Z, std = model_manifold.model.predict(grid, return_std=True)
# Plot probability of z < 0
Z = Z.reshape(100, 100)
std = std.reshape(100, 100)
prob = norm.cdf(0, Z, std)
axs[0].contourf(X, Y, prob)
axs[0].scatter(X_manifold[:, 0], X_manifold[:, 1], color='b')
axs[0].plot(manifold_points[0, :], manifold_points[1, :], 'r')
axs[0].set_title('Probability of z < 0')

bar = axs[1].contourf(X, Y, Z)
plt.colorbar(bar, ax=axs[1])
axs[1].set_title('Mean')

bar = axs[2].contourf(X, Y, std)
plt.colorbar(bar, ax=axs[2])
axs[2].set_title('Std')
plt.show()




