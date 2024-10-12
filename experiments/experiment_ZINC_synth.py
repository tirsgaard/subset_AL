import numpy as np
import matplotlib.pyplot as plt
import os


### Varying the top fraction split
x = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

# 100 datapoints
y_subset =  np.array([0.740, 0.756, 0.757, 0.805, 0.854, 0.894, 0.968])
y_uniform = np.array([0.771, 0.812, 0.838, 0.888, 0.927, 1.020, 1.155])
y_mix =     np.array([0.750, 0.768, 0.779, 0.800, 0.837, 0.879, 0.960])
y_subset_uncertainty =  np.array([0.029, 0.037, 0.028, 0.031, 0.034, 0.031, 0.039])
y_uniform_uncertainty = np.array([0.030, 0.036, 0.036, 0.032, 0.031, 0.034, 0.049])
y_mix_uncertainty =     np.array([0.026, 0.028, 0.025, 0.028, 0.021, 0.022, 0.033])

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
plt.xlabel('Fraction of top scoring datapoints')
plt.ylabel('L1 Loss')
plt.legend()
plt.title('ZINC Synth Subset 100 datapoints')
plt.savefig(save_dir + 'synthesizability_100_mix' + img_suffix)
plt.show()
# Clear the plot
plt.clf()

### Different synthesisabilty subsets with 1000 data points
x = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

y_subset =              np.array([0.314, 0.323, 0.330, 0.318, 0.320, 0.317, 0.34])
y_uniform =             np.array([0.338, 0.338, 0.376, 0.385, 0.420, 0.445, 0.522])
y_subset_uncertainty =  np.array([0.013, 0.016, 0.018, 0.008, 0.006, 0.005, 0.009])
y_uniform_uncertainty = np.array([0.017, 0.012, 0.022, 0.016, 0.024, 0.020, 0.034])

# Plot the un-normalized scores
plt.plot(x, y_subset, label='Subset')
plt.plot(x, y_uniform, label='Uniform')
# Add uncertainty
plt.fill_between(x, y_subset-y_subset_uncertainty, y_subset+y_subset_uncertainty, alpha=0.2)
plt.fill_between(x, y_uniform-y_uniform_uncertainty, y_uniform+y_uniform_uncertainty, alpha=0.2)
plt.xlabel('Synthisizability Subset Fraction')
plt.ylabel('L1 Loss')
plt.legend()
plt.title('ZINC Synthesizability Subset 1000')
plt.savefig(save_dir + 'synthesizability_1000' + img_suffix)
plt.show()
# Clear the plot
plt.clf()


### Different datapoints with synthesisabilty fraction of 0.9
x = np.array([100, 250, 500, 750, 1000])
y_subset =              np.array([0.969, 0.681, 0.464, 0.376, 0.338])
y_uniform =             np.array([1.142, 0.904, 0.762, 0.641, 0.518])
y_subset_uncertainty =  np.array([0.041, 0.035, 0.023, 0.012, 0.01])
y_uniform_uncertainty = np.array([0.051, 0.023, 0.028, 0.041, 0.029])

# Plot the un-normalized scores
plt.plot(x, y_subset, label='Subset')
plt.plot(x, y_uniform, label='Uniform')
# Add uncertainty
plt.fill_between(x, y_subset-y_subset_uncertainty, y_subset+y_subset_uncertainty, alpha=0.2)
plt.fill_between(x, y_uniform-y_uniform_uncertainty, y_uniform+y_uniform_uncertainty, alpha=0.2)
plt.xlabel('Number of datapoints')
plt.ylabel('L1 Loss')
plt.xscale('log')
plt.legend()
plt.title('ZINC Synthesizability varied datapoints')
plt.savefig(save_dir + 'synthesizability_datapoints' + img_suffix)
plt.show()
# Clear the plot
plt.clf()