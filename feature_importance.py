from fire import Fire

import scipy
import wandb

import utils


def calculate_feature_importance(run_id: str):
    api = wandb.Api()
    run_data = api.run(f"shashankmanjunath/tda_airflow_sleep_staging/{run_id}")
    history_df = run_data.history()

    experiment_type = run_data.config["feature_name"]

    if experiment_type == "tda_feature":
        feature_names = utils.get_tda_feature_names()
    elif experiment_type == "classic_feature":
        feature_names = utils.get_ntda_feature_names()
    elif experiment_type == "all":
        feature_names = utils.get_tda_feature_names()
        feature_names += utils.get_ntda_feature_names()
    else:
        raise RuntimeError("Experiment type not recognized!")

    cols = history_df.columns
    feature_cols = [x for x in cols if "feature_importance" in x]
    feature_cols = sorted(
        feature_cols,
        key=lambda x: int(x.replace("feature_importance.f", "")),
    )

    feature_importances = {}
    for idx, feature_col in enumerate(feature_cols):
        importance_val = history_df[feature_col].mean(0)
        feature_importances[feature_names[idx]] = importance_val
    sort_importance = utils.sort_dict_list(feature_importances)

    for idx in range(10):
        print(sort_importance[idx])
    return sort_importance


if __name__ == "__main__":
    tda_run_id = "a9cqryo0"
    classic_run_id = "vfvj6x8g"
    all_run_id = "zak66wpy"

    print("TDA:")
    sort_importance = calculate_feature_importance(
        run_id=tda_run_id,
    )

    print("Classic:")
    sort_importance = calculate_feature_importance(
        run_id=classic_run_id,
    )

    print("All:")
    sort_importance = calculate_feature_importance(
        run_id=all_run_id,
    )
