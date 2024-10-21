import wandb
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import argparse
import os

COMPRESSOR_COLOR_MAP = {None: 'blue', 'direct': 'orange', 'ef': 'green'}
COMPRESSOR_MARKER_MAP = {None: 's', 'direct': 'o', 'ef': '^'}
COMPRESSOR_METHOD_MAP = {None: 'SVFL', 'direct': 'CVFL', 'ef': 'EF-VFL'}
METRICS_MAP = {
    "epoch": "Epoch",
    "comm_cost": "Communications (MB)",
    "grad_squared_norm": "(log) Train gradient squared norm",
    "val_acc": "Validation accuracy (%)",
}

def fetch_run_data(api, project_name, run_name, x_metric, y_metric):
    run = api.run(f"{project_name}/{run_name}")
    history = run.history()

    x_values = history[x_metric].values
    y_values = history[y_metric].values

    x_values, y_values = clean_data(x_metric, x_values, y_metric, y_values)
    return run.config.get('compression_type'), x_values, y_values

def clean_data(x_metric, x_values, y_metric, y_values):
    if x_metric == "comm_cost":
        x_values[0] = 0.0
        x_values = x_values[~np.isnan(x_values)]
    elif x_metric == "epoch":
        x_values = x_values[1:-1:3] + 1  # remove initial and final metrics (val and test)
    y_values = y_values[~np.isnan(y_values)]
    if y_metric == "val_acc":
        y_values *= 100
    return x_values, y_values

def group_runs_by_base_name(run_names, run_name_to_id, api, project_name, x_metric, y_metric):
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

def plot_mean_std(x_values, y_values_list, y_metric, color, method_name, marker, num_markers):
    y_values_array = np.array(y_values_list)
    if y_metric in ["grad_squared_norm", "train_loss", "val_loss"]:
        y_values_array = np.log10(y_values_array)
    y_mean = np.mean(y_values_array, axis=0)
    y_std = np.std(y_values_array, axis=0)

    marker_indices = np.linspace(0, len(x_values) - 1, num_markers, dtype=int)
    plt.plot(x_values, y_mean, label=method_name, marker=marker, markevery=marker_indices)
    plt.fill_between(x_values, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

def save_plot(res_path, x_metric, y_metric, min_max_x_value):
    plt.xlim(left=0, right=min_max_x_value)
    plt.xlabel(METRICS_MAP[x_metric], fontsize=16)
    plt.ylabel(METRICS_MAP[y_metric], fontsize=16)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    if y_metric in ["grad_squared_norm", "train_loss", "val_loss"]:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter(r'$10^{{{x:.1f}}}$'))
    plt.savefig(f"{res_path}/{y_metric}_per_{x_metric}.pdf")
    plt.close()

def plot(project_name, run_names, x_metric, y_metric, res_path, num_markers=8):
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

        plot_mean_std(x_values, y_values_list, y_metric, color, method_name, marker, num_markers)

    save_plot(res_path, x_metric, y_metric, min_max_x_value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default="pvaldeira-team/efvfl", help="Project name in wandb")
    parser.add_argument('--methods', type=str, nargs='+', default=["svfl", "cvfl", "efvfl"])
    parser.add_argument('--experiment', type=str, default="mnist-fullbatch")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--compressor', type=str, default="5b")
    parser.add_argument('--x_metric', type=str, choices=["epoch", "comm_cost"], default="epoch")
    parser.add_argument('--y_metric', type=str, choices=["grad_squared_norm", "val_acc"], default="grad_squared_norm")
    args = parser.parse_args()

    run_names = [f"{args.experiment}-{method}{'-' + args.compressor if method != 'svfl' else ''}-s{seed}"
                 for method in args.methods for seed in args.seeds]

    plot(args.project_name, run_names, args.x_metric, args.y_metric, res_path=f"results/{args.experiment}")
