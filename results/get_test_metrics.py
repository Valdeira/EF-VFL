import argparse
import wandb
import re
import numpy as np


def fetch_run_data(api, project_name, run_name, test_metric):
    run = api.run(f"{project_name}/{run_name}")
    history = run.history(keys=[test_metric])
    return history[test_metric].values


def group_runs_by_base_name(run_names, run_name_to_id, api, project_name, test_metric):
    pattern = re.compile(r"-s\d+$")
    runs_data = {}

    for run_name in run_names:
        run_id = run_name_to_id.get(run_name)
        if not run_id:
            print(f"Run with name '{run_name}' not found.")
            continue

        base_name = pattern.sub("", run_name)
        test_values = fetch_run_data(api, project_name, run_id, test_metric).item()
        runs_data.setdefault(base_name, []).append(test_values)

    return runs_data


def print_test_metrics(project_name, run_names, test_metric):
    api = wandb.Api()
    runs = api.runs(project_name)
    run_name_to_id = {run.name: run.id for run in runs}

    runs_data = group_runs_by_base_name(run_names, run_name_to_id, api, project_name, test_metric)

    for base_name, test_metric_across_seeds_l in runs_data.items():
        test_metric_across_seeds = np.array(test_metric_across_seeds_l) * 100
        mean_test_metric = np.mean(test_metric_across_seeds)
        std_test_metric = np.std(test_metric_across_seeds)
        print(f"{base_name} accuracy (%): {mean_test_metric:.1f} ± {std_test_metric:.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default="pvaldeira-team/efvfl", help="Project name in wandb")
    parser.add_argument('--methods', type=str, nargs='+', default=["svfl", "cvfl", "efvfl"])
    parser.add_argument('--experiment', type=str, default="mnist-fullbatch")
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--compressor', type=str, default="0.1k")
    parser.add_argument('--test_metric', type=str, default="test_acc")
    args = parser.parse_args()

    run_names = [f"{args.experiment}-{method}{'-' + args.compressor if method != 'svfl' else ''}-s{seed}"
                 for method in args.methods for seed in args.seeds]

    print_test_metrics(args.project_name, run_names, args.test_metric)
