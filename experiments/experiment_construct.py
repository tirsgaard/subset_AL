import numpy as np
import matplotlib.pyplot as plt
import os

x = np.array([100, 250, 500, 750, 1000])
# Un-normalized scores
"""
y_subset = np.array([0.602, 0.471, 0.401, 0.281])
y_uniform = np.array([0.688, 0.506, 0.417, 0.291])
y_subset_uncertainty = np.array([0.023, 0.008, 0.008, 0.013])
y_uniform_uncertainty = np.array([0.033, 0.014, 0.012, 0.012])

# Normalized scores
y_subset_diff = np.array([0.086, 0.035, 0.016, 0.011])
y_subset_diff_uncertainty = np.array([0.029, 0.016, 0.016, 0.015])
"""

# Different sample sizes
y_subset =  np.array([0.606, 0.471, 0.400, 0.323, 0.285])
y_uniform = np.array([0.683, 0.504, 0.407, 0.324, 0.292])
y_mix =     np.array([0.641, 0.474, 0.406, 0.325, 0.273])
y_subset_uncertainty =  np.array([0.023, 0.010, 0.008, 0.015, 0.014])
y_uniform_uncertainty = np.array([0.031, 0.014, 0.014, 0.015, 0.013])
y_mix_uncertainty =     np.array([0.034, 0.009, 0.009, 0.015, 0.010])

save_dir = "figs/"
img_suffix = ".svg"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plot the un-normalized scores
plt.plot(x, y_subset, label='Subset')
plt.plot(x, y_uniform, label='Uniform')
plt.plot(x, y_mix, label='Mix')
# Add uncertainty
plt.fill_between(x, y_subset-y_subset_uncertainty, y_subset+y_subset_uncertainty, alpha=0.2)
plt.fill_between(x, y_uniform-y_uniform_uncertainty, y_uniform+y_uniform_uncertainty, alpha=0.2)
plt.fill_between(x, y_mix-y_mix_uncertainty, y_mix+y_mix_uncertainty, alpha=0.2)
plt.xlabel('Number of training samples')
plt.ylabel('L1 Loss')
plt.xscale('log')
plt.legend()
plt.title('ZINC Atom Subset')
plt.savefig(save_dir + 'unnormalized_scores' + img_suffix)
plt.show()
# Clear the plot
plt.clf()

def calculate_effective_sample_size(x, y, y_target):
    """
    Calculate the effective sample size of y in terms of y_target.
    Args:
        x: The number of training samples
        y: The loss of the model
        y_target: The target loss
    Returns:
        The effective sample size of y in terms of y_target.
    """
    # Interpolate the loss of y_target
    x_int = np.linspace(x[0], x[-1], 1000)
    y_int = np.interp(x_int, x, y)
    y_target_int = np.interp(x_int, x, y_target)
    
    x_corresponding = np.zeros(len(x_int))
    for i in range(len(x_int)):
        res = x_int[(y_int[i] - y_target_int) > 0]
        x_corresponding[i] = res[0] if len(res) > 0 else x_int[-1]
    return x_int, x_corresponding
    
        
x_int, x_sample_cor = calculate_effective_sample_size(x, y_subset, y_uniform)
plt.plot(x_int, x_int, label='Uniform')
plt.plot(x_int, x_sample_cor, label='Subset')
plt.xlabel('Number of training samples')
plt.ylabel('Effective sample size')
plt.legend()
plt.title('ZINC Atom Subset')
plt.savefig(save_dir + 'effective_sample_size' + img_suffix)
plt.show()
plt.clf()

x_int, x_sample_cor = calculate_effective_sample_size(x, y_subset, y_uniform)
plt.plot(x_int, x_int - x_int, label='Uniform')
plt.plot(x_int, x_sample_cor - x_int, label='Subset')
plt.xlabel('Number of training samples')
plt.ylabel('Effective sample size')
plt.legend()
plt.title('ZINC Atom Subset')
plt.savefig(save_dir + 'effective_sample_size_diff' + img_suffix)
plt.show()
plt.clf()


