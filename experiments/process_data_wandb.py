import numpy as np
import matplotlib.pyplot as plt
import csv

path_csv = 'experiments/wandb_export_100_grid.csv'

# Load data
with open(path_csv, newline='') as csvfile:
    data = list(csv.reader(csvfile))
    
# Extract data
data = np.array(data)  
data = data[1:, 1:]  # dim [l2wd, lr, lr_patience, test/metric]

# Replace missing values with 0
data[data == ''] = 0

# Convert to float
data = data.astype(float)
data = data[data[:, 2] < 0.8]
data = data[data[:, 2] != 0.]

# Make 2d plot as a function of lr and and l2wd with points colored by test/metric
lr = np.logspace(-3, -1, 10)
l2wd = np.logspace(-3, -1, 10)
XY = np.meshgrid(lr, l2wd)
test_metric = data[:, 2]
#test_metric[test_metric >1.] = 0

#plt.contour(XY[0], XY[1], test_metric.reshape(10, 10))
plt.scatter(data[:, 1], data[:, 0], c=test_metric, s=300)

# Make a red cross for smallest value
min_idx = np.argmin(test_metric)
plt.scatter(data[min_idx, 1], data[min_idx, 0], c='r', s=300, marker='x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('lr')
plt.ylabel('l2wd')
# Label colorbar
plt.colorbar().set_label('L1 validation error')
plt.savefig('figs/100_data_pointcloud.svg')
plt.show()
