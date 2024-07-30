from itertools import combinations

from fire import Fire

import scipy
import wandb


def ttest_xgb(run_1: str, run_2: str, run_3: str):
    api = wandb.Api()

    run_1_data = api.run(f"shashankmanjunath/tda_airflow_sleep_staging/{run_1}")
    run_2_data = api.run(f"shashankmanjunath/tda_airflow_sleep_staging/{run_2}")
    run_3_data = api.run(f"shashankmanjunath/tda_airflow_sleep_staging/{run_3}")

    runs = [run_1_data, run_2_data, run_3_data]

    data = {}
    for run in runs:
        experiment_type = run.config["feature_name"]
        if experiment_type in data.keys():
            raise RuntimeError("Repeated Experiment!")

        history_df = run.history()

        data[experiment_type] = history_df.iloc[-1]["test_ba_all"]

    experiments = list(data.keys())
    combos = list(combinations(experiments, 2))

    for exp0, exp1 in combos:
        ttest_res = scipy.stats.ttest_ind(
            data[exp0],
            data[exp1],
            alternative="two-sided",
        )
        print(f"{exp0} vs. {exp1}: {ttest_res.pvalue}")


if __name__ == "__main__":
    Fire(ttest_xgb)
