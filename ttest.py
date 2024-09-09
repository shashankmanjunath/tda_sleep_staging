from itertools import combinations
import typing

from fire import Fire
import numpy as np

import scipy
import wandb


def ttest_xgb(run_ids: typing.List[str], wandb_api_str: str):
    api = wandb.Api()

    runs = []
    for run_id in run_ids:
        runs.append(api.run(f"{wandb_api_str}/{run_id}"))

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
    Fire(ttest_xgb)
