import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

file_path = Path("./save")

def criterion(p_class_n, alpha, remaining_budgets):
    val = p_class_n[0:-1] - remaining_budgets[1:]/(-np.diff(remaining_budgets))*(p_class_n[1:] - p_class_n[0:-1])
    return (1-alpha)*val - 1

def sample_effect(p_class_n, alpha, remaining_budgets, N, pm):
    value = (alpha*p_class_n + (1-p_class_n))*remaining_budgets
    baseline = (alpha*pm + 1*(1-pm))*N
    return value - baseline
    
def worst_sample_effect(p_class_n, remaining_budgets, N, pm):
    value = p_class_n*remaining_budgets
    baseline = pm*N
    return value - baseline

alpha = 2
pm = 0.1
run_data = []
datapoints = [50, 100, 250, 500, 750, 1000]
N = 10000
x_datapoints = []
for n_datapoints in datapoints:
    # load file
    file_name = file_path / f"manifold_data_new{n_datapoints}.pkl"
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    results = np.array(list(value["manifold_rate"] for value in data))
    run_data.append(results)
    x_datapoints.append(len(results)*[n_datapoints])
    
# Fit data with log-linear model
x = np.array(x_datapoints).flatten()
run_data = 1 - np.array(run_data)
log_x = np.log(x)
log_run_data = np.log(run_data).flatten()
linear_fit = np.polyfit(log_x, log_run_data, 1)
#linear_fit[0] = -0.5
print(linear_fit)


# plot
fig, axs = plt.subplots(3, 1, figsize=(5, 7))
axs[0].scatter(x, run_data.flatten(), color="blue", alpha=0.5)
mean_values = np.mean(run_data, axis=1)


axs[0].plot(datapoints, mean_values, color="blue", label="Mean")
x_values = np.linspace(-0.3, N, 10000)
y_values = np.exp(linear_fit[1]) * x_values ** linear_fit[0]
axs[0].plot(x_values, y_values, color="red", label="Fit")
axs[0].set_xlabel("Number of datapoints")
axs[0].set_ylabel("1 - subset rate")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].legend()

condition = criterion(1-mean_values, alpha, N - np.array(datapoints))
condition_smoothed = criterion(1-y_values, alpha, N - np.array(x_values))
axs[1].plot(datapoints[:-1], condition, label="Condition")
axs[1].plot(x_values[:-1], condition_smoothed, label="Condition smoothed")
axs[1].set_xscale("log")
print(f"Optimal value: {x_values[np.sum(condition_smoothed>0)]}")


sample_effects = sample_effect(1-y_values, alpha, N - np.array(x_values), N, pm)
print(f"Last positive value: {x_values[np.sum(sample_effects>0)]}")
axs[2].plot(x_values, sample_effects, label="Sample effect")
axs[2].set_xscale("log")
axs[2].legend()

plt.savefig("figs/fit_classification.png")
plt.show()

