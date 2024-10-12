import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use("seaborn-v0_8")


### Varying the fraction values
x = np.array([100, 250, 500, 750, 1000])

# 100 datapoints
y_subset =  np.array([0.615, 0.382, 0.259, 0.218, 0.203])
y_uniform = np.array([0.870, 0.437, 0.298, 0.249, 0.236])
y_subset_uncertainty =  np.array([0.030, 0.026, 0.007, 0.005, 0.003])
y_uniform_uncertainty = np.array([0.084, 0.035, 0.010, 0.005, 0.005])

save_dir = "figs/"
img_suffix = ".png"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plot the un-normalized scores
plt.plot(x, y_subset, label='Subset')
plt.plot(x, y_uniform, label='Uniform')
# Add uncertainty
plt.fill_between(x, y_subset-y_subset_uncertainty, y_subset+y_subset_uncertainty, alpha=0.2)
plt.fill_between(x, y_uniform-y_uniform_uncertainty, y_uniform+y_uniform_uncertainty, alpha=0.2)
plt.xlabel('Fraction of top synthesizable datapoints')
plt.ylabel('MSE Loss')
plt.xscale('log')
plt.legend()
plt.savefig(save_dir + 'QM9_synthesizability_0_9' + img_suffix)
plt.show()
# Clear the plot
plt.clf()


### Varying the top fraction split with 100 samples
x = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# 100 datapoints
y_subset =  np.array([3.320, float("NaN"), 3.172, 3.329, 3.315, 3.599, 2.957])
y_uniform = np.array([3.347, float("NaN"), 3.157,  3.262, 3.423, 3.650, 3.260])
y_subset_uncertainty =  np.array([0.182, float("NaN"), 0.104, 0.228, 0.213, 0.596, 0.173])
y_uniform_uncertainty = np.array([0.253, float("NaN"), 0.116, 0.271, 0.229, 0.422, 0.255])

save_dir = "figs/"
img_suffix = ".png"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plot the un-normalized scores
plt.plot(x, y_subset, label='Subset')
plt.plot(x, y_uniform, label='Uniform')
# Add uncertainty
plt.fill_between(x, y_subset-y_subset_uncertainty, y_subset+y_subset_uncertainty, alpha=0.2)
plt.fill_between(x, y_uniform-y_uniform_uncertainty, y_uniform+y_uniform_uncertainty, alpha=0.2)
plt.xlabel('Fraction of top synthesizable datapoints')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('ZINC Synth Subset 100 datapoints')
plt.savefig(save_dir + 'QM9_synthesizability_100' + img_suffix)
plt.show()
# Clear the plot
plt.clf()


### Varying the top fraction split with 100 samples
x = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# 1000 datapoints
y_subset =  np.array([1.329, float("NaN"), 1.210, 1.384, 0.949, 1.303, 1.008])
y_uniform = np.array([1.478, float("NaN"), 1.478, 1.660, 1.934, 1.835, 1.829])
y_subset_uncertainty =  np.array([0.271, float("NaN"), 0.333, 0.386, 0.214, 0.281, 0.232])
y_uniform_uncertainty = np.array([0.283, float("NaN"), 0.399, 0.457, 0.456, 0.372, 0.479])

save_dir = "figs/"
img_suffix = ".png"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plot the un-normalized scores
plt.plot(x, y_subset, label='Subset')
plt.plot(x, y_uniform, label='Uniform')
# Add uncertainty
plt.fill_between(x, y_subset-y_subset_uncertainty, y_subset+y_subset_uncertainty, alpha=0.2)
plt.fill_between(x, y_uniform-y_uniform_uncertainty, y_uniform+y_uniform_uncertainty, alpha=0.2)
plt.xlabel('Fraction of top synthesizable datapoints')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('ZINC Synth Subset 1000 datapoints')
plt.savefig(save_dir + 'QM9_synthesizability_1000' + img_suffix)
plt.show()
# Clear the plot
plt.clf()


### Varying the top fraction split with 100 samples
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# 1000 datapoints
y_subset =  np.array([0.580, 1.577, 0.009, 0.010, 0.014, 60.075, 0.003, 8.191, 8.243, 8.137, 8.427, 0.619, 39.839, 38.273, 38.858, 35.827, 21.387, 0.200])
y_uniform = np.array([0.840, 2.040, 0.011, 0.014, 0.019, 79.880, 0.005, 10.659, 10.550, 10.997, 10.806, 0.880, 57.942, 58.812, 58.740, 54.388, 30.930, 0.260])
y_subset_uncertainty =  np.array([0.013, 0.057, 0.001, 0.001, 0.001, 2.676, 0.001, 0.419, 0.428, 0.358, 0.494, 0.033, 2.198, 1.896, 2.142, 1.714, 1.751, 0.006])
y_uniform_uncertainty = np.array([0.096, 0.186, 0.001, 0.001, 0.002, 5.092, 0.001, 1.100, 1.169, 1.288, 1.044, 0.088, 5.485, 5.968, 6.201, 4.208, 5.070, 0.023])

save_dir = "figs/"
img_suffix = ".svg"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Plot the un-normalized scores
plt.scatter(x, y_subset, label='Subset', c='g')
plt.scatter(x, y_uniform, label='Uniform', c='b')
# Add uncertainty
plt.fill_between(x, y_subset-y_subset_uncertainty, y_subset+y_subset_uncertainty, alpha=0.3, color='g')
plt.fill_between(x, y_uniform-y_uniform_uncertainty, y_uniform+y_uniform_uncertainty, alpha=0.3, color='b')
plt.xlabel('QM9 Target Index')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.legend()
plt.savefig(save_dir + 'QM9_synthesizability_100_idx' + img_suffix, bbox_inches='tight')
plt.show()
# Clear the plot
plt.clf()


# Relative improvement of subset over uniform plot
y_relative = (y_uniform - y_subset) / y_uniform
worst_relative = (y_uniform+y_uniform_uncertainty - y_subset-y_subset_uncertainty) / (y_uniform+y_uniform_uncertainty)
best_relative = (y_uniform-y_uniform_uncertainty - y_subset+y_subset_uncertainty) / (y_uniform-y_uniform_uncertainty)
plt.scatter(x, y_relative)
plt.xlabel('QM9 Target Index')
plt.ylabel('Relative Improvement')
plt.savefig(save_dir + 'QM9_synthesizability_100_relative' + img_suffix, bbox_inches='tight')
plt.show()
# Clear the plot
plt.clf()
