import wandb
import re
import numpy as np
import matplotlib.pyplot as plt
import os


compressor_color_map = {None: 'blue', 'direct': 'orange', 'ef': 'green'}
compressor_marker_map = {None: 's', 'direct': 'o', 'ef': '^'}
compressor_method_map = {None: 'SVFL', 'direct': 'CVFL', 'ef': 'EFVFL'}
metrics_map = {
    "epoch": "Epoch",
    "comm_cost": "Communications (MB)",
    "grad_squared_norm": "Train gradient squared norm",
    "val_acc": "Validation accuracy",
    }


def plot(project_name, run_names, x_metric, y_metric, plot_name, res_path):
    api = wandb.Api()
    runs = api.runs(project_name)
    run_name_to_id = {run.name: run.id for run in runs}
    pattern = re.compile(r"-s\d+$")

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    runs_data = {}
    base_name_compression_type_map = {}
    for run_name in run_names:
        # Get the correct run ID based on the run name
        run_id = run_name_to_id.get(run_name)
        if not run_id:
            print(f"Run with name '{run_name}' not found.")
            continue

        # Fetch the run using its ID
        run = api.run(f"{project_name}/{run_id}")
        
        history = run.history()

        # Extract the relevant columns from the run history
        x_values = history[x_metric].values
        y_values = history[y_metric].values

        # Remove NaN and repeated values
        if x_metric == "comm_cost":
            x_values = x_values[~np.isnan(x_values)]
        elif x_metric == "epoch":
            x_values = x_values[::3]
            x_values += 1
        y_values = y_values[~np.isnan(y_values)]

        base_name = pattern.sub("", run_name)
        # Add the x and y values to the group of runs with the same base name
        if base_name not in runs_data:
            runs_data[base_name] = []
        runs_data[base_name].append((x_values, y_values))
        
        base_name_compression_type_map[base_name] = run.config.get('compression_type')

    # Initialize a variable to track the minimum of the maximum x_values
    min_max_x_value = float('inf')

    num_markers = 8

    # Now plot the mean and std deviation for each group
    for base_name, data in runs_data.items():
        # Stack the y_values to compute mean and std deviation
        y_values_list = [y for _, y in data]

        y_values_array = np.array(y_values_list)
        y_mean = np.mean(y_values_array, axis=0)
        y_std = np.std(y_values_array, axis=0)

        # Use the x_values from the first run (they should be the same across all runs in the group)
        x_values = data[0][0]

        # Update the min_max_x_value to the minimum of the maximum x_values
        max_x_value = max(x_values)
        min_max_x_value = min(min_max_x_value, max_x_value)

        # Fetch color, method_name, and marker using the compression_type associated with the base_name
        compression_type = base_name_compression_type_map[base_name]
        color = compressor_color_map[compression_type]
        method_name = compressor_method_map[compression_type]
        marker = compressor_marker_map[compression_type]

        marker_indices = np.linspace(0, len(x_values) - 1, num_markers, dtype=int)

        # Plot the mean and standard deviation as shaded region
        plt.plot(x_values, y_mean, label=method_name, marker=marker, markevery=marker_indices)
        
        plt.fill_between(x_values, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

    # After the loop, set the right x-axis limit to the minimum of the maximum x_values
    plt.xlim(left=0, right=min_max_x_value)

    plt.xlabel(metrics_map[x_metric])
    plt.ylabel(metrics_map[y_metric])
    plt.legend()
    plt.title(plot_name)
    plt.grid(True)
    if y_metric in ["grad_squared_norm", "train_loss", "val_loss"]:
        plt.yscale('log')
    plt.savefig(f"{res_path}/{y_metric}_per_{x_metric}.pdf")
    plt.close()


if __name__ == '__main__':

    project_name = "pvaldeira-team/efvfl"
    
    run_names = [
                "mnist-fullbatch-svfl-s0",
                "mnist-fullbatch-cvfl-0.01k-s0",
                "mnist-fullbatch-efvfl-0.01k-s0",
                ]
    
    x_metric = "epoch"  # "epoch" | "comm_cost"
    y_metric = "grad_squared_norm"  # "train_loss" | "grad_squared_norm" | "val_acc"
    plot_name = "MNIST full-batch"
    res_path= "results/mnist-fullbatch"
    
    plot(project_name, run_names, x_metric, y_metric, plot_name, res_path)
