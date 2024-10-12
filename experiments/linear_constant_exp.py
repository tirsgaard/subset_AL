import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

save_dir = "figs/"
img_suffix = ".png"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

class LinearModel:
    def __init__(self, subset):
        self.subset = subset
        self.fit_model = None
    
    def add_linear_features(self, x):
        return np.stack([x, x>0.5, 0*x+1])
    
    def add_bias(self, x):
        return np.stack([x, 0*x+1])
    
    def fit(self, x, y):
        x = self.add_bias(x) if self.subset else self.add_linear_features(x)
        self.fit_model = (np.linalg.inv((x @ x.T))@x)@y
        return self.fit_model
    
    def predict(self, x):
        x = self.add_bias(x) if self.subset else self.add_linear_features(x)
        return self.fit_model @ x


def sample_problem(n_points, subset=False, noise=10**-1):
    x = np.random.rand(n_points)
    x = 0.5*x if subset else x
    y = x + np.random.rand(n_points)*noise
    y[x>0.5] += 1
    return x, y

n_samples = [10, 25, 50, 75, 100, 250, 500, 750, 1000]
n_repeats = 1000
x_test, y_test = sample_problem(10**5, subset=True, noise=0)



def experiment(n_samples, x_test, y_test, n_repeats=10, subset=False):
    mse_list = []
    for _ in range(n_repeats):
        x_train, y_train = sample_problem(n_samples, subset=subset)
        linear_fit = LinearModel(1)
        linear_fit.fit(x_train, y_train)
        y_pred = linear_fit.predict(x_test)
        mse = np.mean((y_pred - y_test)**2)
        mse_list.append(mse)
    return np.mean(mse_list), np.std(mse_list)

uniform = [experiment(n, x_test, y_test, n_repeats) for n in tqdm(n_samples)]
mse_uniform = [u[0] for u in uniform]
std_uniform = [u[1] for u in uniform]
uniform_uncertainty = 1.96 * np.array(std_uniform) / np.sqrt(n_repeats)
subset = [experiment(n, x_test, y_test, n_repeats, subset=True) for n in tqdm(n_samples)]
mse_subset = [s[0] for s in subset]
std_subset = [s[1] for s in subset]
subset_uncertainty = 1.96 * np.array(std_subset) / np.sqrt(n_repeats)

# Plot results
plt.plot(n_samples, mse_uniform, label='Uniform')
plt.fill_between(n_samples, mse_uniform-uniform_uncertainty, mse_uniform+uniform_uncertainty, alpha=0.2)
plt.plot(n_samples, mse_subset, label='Subset')
plt.fill_between(n_samples, mse_subset-subset_uncertainty, mse_subset+subset_uncertainty, alpha=0.2)
plt.xlabel('Number of samples')
plt.ylabel('MSE')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig(save_dir + 'test_const_linear' + img_suffix)
plt.show()


