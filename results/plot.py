import wandb
import re
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration mappings
COMPRESSOR_COLOR_MAP = {None: 'blue', 'direct': 'orange', 'ef': 'green'}
COMPRESSOR_MARKER_MAP = {None: 's', 'direct': 'o', 'ef': '^'}
COMPRESSOR_METHOD_MAP = {None: 'SVFL', 'direct': 'CVFL', 'ef': 'EFVFL'}
METRICS_MAP = {
    "epoch": "Epoch",
    "comm_cost": "Communications (MB)",
    "grad_squared_norm": "Train gradient squared norm",
    "val_acc": "Validation accuracy",
}

def fetch_run_data(api, project_name, run_name, x_metric, y_metric):
    """Fetch x and y metric data from a given run."""
    run = api.run(f"{project_name}/{run_name}")
    history = run.history()

    x_values = history[x_metric].values
    y_values = history[y_metric].values

    x_values, y_values = clean_data(x_metric, x_values, y_values)
    return run.config.get('compression_type'), x_values, y_values

def clean_data(x_metric, x_values, y_values):
    if x_metric == "comm_cost":
        x_values = x_values[~np.isnan(x_values)]
    elif x_metric == "epoch":
        x_values = x_values[::3] + 1  # Adjust epoch values
        x_values = x_values[:-1] # Drop the "epoch" corresponding to test metrics
    y_values = y_values[~np.isnan(y_values)]
    return x_values, y_values

def group_runs_by_base_name(run_names, run_name_to_id, api, project_name, x_metric, y_metric):
    """Group runs by their base name and fetch corresponding data."""
    pattern = re.compile(r"-s\d+$")
    runs_data = {}
    base_name_compression_type_map = {}

    for run_name in run_names:
        run_id = run_name_to_id.get(run_name)
        if not run_id:
            print(f"Run with name '{run_name}' not found.")
            continue

        base_name = pattern.sub("", run_name)
        compression_type, x_values, y_values = fetch_run_data(api, project_name, run_id, x_metric, y_metric)

        if base_name not in runs_data:
            runs_data[base_name] = []
        runs_data[base_name].append((x_values, y_values))
        base_name_compression_type_map[base_name] = compression_type

    return runs_data, base_name_compression_type_map

def plot_mean_std(x_values, y_values_list, color, method_name, marker, num_markers):
    """Plot mean and std deviation for a set of y values."""
    y_values_array = np.array(y_values_list)
    if y_metric in ["grad_squared_norm", "train_loss", "val_loss"]:
        y_values_array = np.log10(y_values_array)
    y_mean = np.mean(y_values_array, axis=0)
    y_std = np.std(y_values_array, axis=0)

    marker_indices = np.linspace(0, len(x_values) - 1, num_markers, dtype=int)
    plt.plot(x_values, y_mean, label=method_name, marker=marker, markevery=marker_indices)
    plt.fill_between(x_values, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

def save_plot(res_path, x_metric, y_metric, plot_name, min_max_x_value):
    """Save the generated plot."""
    plt.xlim(left=0, right=min_max_x_value)
    plt.xlabel(METRICS_MAP[x_metric])
    plt.ylabel(METRICS_MAP[y_metric])
    plt.legend()
    plt.title(plot_name)
    plt.grid(True)
    # if y_metric in ["grad_squared_norm", "train_loss", "val_loss"]:
    #     plt.yscale('log')
    plt.savefig(f"{res_path}/{y_metric}_per_{x_metric}.pdf")
    plt.close()

def plot(project_name, run_names, x_metric, y_metric, plot_name, res_path, num_markers=8):
    api = wandb.Api()
    runs = api.runs(project_name)
    run_name_to_id = {run.name: run.id for run in runs}

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    runs_data, base_name_compression_type_map = group_runs_by_base_name(
        run_names, run_name_to_id, api, project_name, x_metric, y_metric
    )

    min_max_x_value = float('inf')
    for base_name, data in runs_data.items():
        y_values_list = [y for _, y in data]
        x_values = data[0][0]
        max_x_value = max(x_values)
        min_max_x_value = min(min_max_x_value, max_x_value)

        compression_type = base_name_compression_type_map[base_name]
        color = COMPRESSOR_COLOR_MAP[compression_type]
        method_name = COMPRESSOR_METHOD_MAP[compression_type]
        marker = COMPRESSOR_MARKER_MAP[compression_type]

        plot_mean_std(x_values, y_values_list, color, method_name, marker, num_markers)

    save_plot(res_path, x_metric, y_metric, plot_name, min_max_x_value)

if __name__ == '__main__':
    project_name = "pvaldeira-team/efvfl"
    run_names = [
        "mnist-fullbatch-svfl-s0",
        "mnist-fullbatch-svfl-s1",
        "mnist-fullbatch-svfl-s2",
        "mnist-fullbatch-svfl-s3",
        "mnist-fullbatch-svfl-s4",
        "mnist-fullbatch-cvfl-0.1k-s0",
        "mnist-fullbatch-cvfl-0.1k-s1",
        "mnist-fullbatch-cvfl-0.1k-s2",
        "mnist-fullbatch-cvfl-0.1k-s3",
        "mnist-fullbatch-cvfl-0.1k-s4",
        "mnist-fullbatch-efvfl-0.1k-s0",
        "mnist-fullbatch-efvfl-0.1k-s1",
        "mnist-fullbatch-efvfl-0.1k-s2",
        "mnist-fullbatch-efvfl-0.1k-s3",
        "mnist-fullbatch-efvfl-0.1k-s4",
    ]
    x_metric = "comm_cost" # "epoch" | "comm_cost"
    y_metric = "val_acc" # "grad_squared_norm" | "val_acc"
    plot_name = "MNIST full-batch"
    res_path = "results/mnist-fullbatch"

    plot(project_name, run_names, x_metric, y_metric, plot_name, res_path)
