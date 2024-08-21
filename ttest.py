from itertools import combinations
import typing

from fire import Fire

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
        #  ttest_res = scipy.stats.ttest_ind(
        ttest_res = scipy.stats.ranksums(
            data[exp0],
            data[exp1],
            alternative="two-sided",
        )
        print(f"{exp0} vs. {exp1}: {ttest_res.pvalue}")


if __name__ == "__main__":
    ttest_xgb(
        [
            # TDA
            #  "p1i9pxcp",
            # Classic
            #  "ymo98u6p",
            # All
            #  "ykjcjg4m",
            # HEPC
            #  "f6vlxoiu",
            # FFT, Abs
            "kz4vsd2c",
            # FFT, RI
            "eimu4xfi",
            # TDA using FFT, Abs
            #  "x2qe14bq",
            #  TDA using FFT, RI
            #  "xovhgjn8",
            # All using FFT, Abs
            #  "zb1b8u2w",
            # All using FFT, RI
            #  "z2qjmkj4",
            # All Homology using FFT, Abs
            #  "3xx1vbha",
            # All Homology using FFT, RI
            #  "1i6rtsrs",
            # All Homology using HEPC
            #  "l399acqj",
        ]
    )
