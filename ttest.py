from itertools import combinations
import typing

import numpy as np

import scipy
import wandb


def ttest_xgb(run_ids: typing.List[str]):
    api = wandb.Api()

    runs = []
    for run_id in run_ids:
        runs.append(api.run(f"shashankmanjunath/tda_airflow_sleep_staging/{run_id}"))

    data = {}
    for run in runs:
        experiment_type = run.config["feature_name"]
        if experiment_type in data.keys():
            raise RuntimeError("Repeated Experiment!")

        history_df = run.history()

        data[experiment_type] = history_df["test_ba"].tolist()

    experiments = list(data.keys())
    combos = list(combinations(experiments, 2))

    for exp0, exp1 in combos:
        ttest_res = scipy.stats.ttest_rel(
            #  ttest_res = scipy.stats.ranksums(
            data[exp0],
            data[exp1],
            alternative="two-sided",
        )
        print(f"{exp0}: {np.mean(data[exp0]):.3f}")
        print(f"{exp1}: {np.mean(data[exp1]):.3f}")
        print(f"{exp0} vs. {exp1}: {ttest_res.pvalue}")


if __name__ == "__main__":
    ttest_xgb(
        [
            #  random,
            #  "yzc3pt9o",
            #  classic_6_epoch,
            #  "vm8rhc7i",
            #  hepc,
            #  "04rgwl67",
            #  fft,
            #  "jh36k3qy",
            #  fft_cf,
            #  "9wc1fq7q",
            #  fft_hepc,
            #  "08ewqjlu",
            #  fft_cf_hepc,
            #  "h14hirh4",
            #  classic_6_epoch_hepc,
            #  "f0ck7bnl",
            #  classic_6_epoch_fft,
            #  "5c3jhkq5",
            #  classic_6_epoch_fft_hepc,
            #  "btnbc4c3",
            #  classic_6_epoch_fft_cf,
            #  "e1zcs3i5",
            #  classic_6_epoch_fft_cf_hepc,
            #  "q3gvisx3",
        ]
    )
