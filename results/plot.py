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
    "grad_squared_norm": "Train gradient squared norm",
    "val_acc": "Validation accuracy (%)",
    "train_loss": "Train loss",
}

def fetch_run_data(api, project_name, run_name, x_metric, y_metric):
    run = api.run(f"{project_name}/{run_name}")
    history = run.history(keys=[x_metric, y_metric])
    
    x_values = history[x_metric].values
    y_values = history[y_metric].values
    
    return run.config.get('compression_type'), x_values, y_values

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

def plot_mean_std(x_values, y_values_list, y_metric, color, method_name, marker):
    y_values_array = np.array(y_values_list)
    if y_metric in ["grad_squared_norm", "train_loss", "val_loss"]:
        y_values_array = np.log10(y_values_array)
    y_mean = np.mean(y_values_array, axis=0)
    y_std = np.std(y_values_array, axis=0)
    
    plt.plot(x_values, y_mean, label=method_name, marker=marker, markersize=8, markevery=.1)
    plt.fill_between(x_values, y_mean - y_std, y_mean + y_std, color=color, alpha=0.25)

def save_plot(res_path, x_metric, y_metric, min_max_x_value, compressor):
    plt.xlim(left=0, right=min_max_x_value)
    plt.xlabel(METRICS_MAP[x_metric], fontsize=20)
    plt.ylabel(METRICS_MAP[y_metric], fontsize=20)
    plt.legend(loc='best', fontsize=17)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    if y_metric in ["grad_squared_norm", "train_loss", "val_loss"]:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter(r'$10^{{{x:.1f}}}$'))
    plt.savefig(f"{res_path}/{compressor}_{y_metric}_per_{x_metric}.pdf")
    plt.close()

def plot(project_name, run_names, x_metric, y_metric, compressor, res_path):
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

        plot_mean_std(x_values, y_values_list, y_metric, color, method_name, marker)

    save_plot(res_path, x_metric, y_metric, min_max_x_value, compressor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default="pvaldeira-team/efvfl", help="Project name in wandb")
    parser.add_argument('--methods', type=str, nargs='+', default=["svfl", "cvfl", "efvfl"])
    parser.add_argument('--experiment', type=str, default="mnist-fullbatch")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--compressor', type=str, default="0.1k")
    parser.add_argument('--x_metric', type=str, choices=["epoch", "comm_cost"], default="comm_cost")
    parser.add_argument('--y_metric', type=str, choices=["grad_squared_norm", "val_acc", "train_loss"], default="val_acc")
    args = parser.parse_args()

    run_names = [f"{args.experiment}-{method}{'-' + args.compressor if method != 'svfl' else ''}-s{seed}"
                 for method in args.methods for seed in args.seeds]

    plot(args.project_name, run_names, args.x_metric, args.y_metric, args.compressor, res_path=f"results/{args.experiment}")
