import typing
import os

from xgboost import XGBClassifier
from fire import Fire
from tqdm import tqdm

import sklearn.model_selection
import sklearn.linear_model
import numpy as np
import sklearn
import skopt
import wandb

import tda_utils
import utils


def train(data_dir: str, feature_name: str, use_wandb: bool = False):
    subject_fnames = [x for x in os.listdir(data_dir) if x.endswith(".hdf5")]
    #  subject_fnames = subject_fnames[:50]

    all_paths = np.asarray([os.path.join(data_dir, x) for x in subject_fnames])
    all_demos = utils.get_demographics(
        all_paths,
        data_dir="/work/thesathlab/nchsdb/",
    )
    all_data, all_label = utils.load_data(all_paths, feature_name, sqi_thresh=0.25)

    map_type = utils.wake_nrem_rem_map
    for k, v in all_label.items():
        all_label[k] = np.asarray(list(map(map_type, v)))

    xgb_seed = 999
    kf_seed = 2024

    model = XGBClassifier(
        learning_rate=0.07,
        n_estimators=100,  # 200
        max_depth=4,  # 5
        min_child_weight=1,
        gamma=0,  # 1
        subsample=0.3,  # 0.2
        colsample_bytree=0.8,  # 0.5
        objective="multi:softprob",
        eval_metric="mlogloss",
        seed=xgb_seed,
        #  use_label_encoder=False,
    )

    #  model = sklearn.linear_model.RidgeClassifier(random_state=999)

    n_splits = 5

    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits,
        random_state=kf_seed,
        shuffle=True,
    )
    #  kf = sklearn.model_selection.KFold(
    #      n_splits=n_splits,
    #      random_state=kf_seed,
    #      shuffle=True,
    #  )

    # wandb setup
    if use_wandb:
        wandb.init(
            project="tda_airflow_sleep_staging",
            config={
                "xgb_params": model.get_xgb_params(),
                "n_splits": n_splits,
                "subject_fnames": subject_fnames,
                "data_dir": data_dir,
                "feature_name": feature_name,
                "xgb_seed": xgb_seed,
                "kf_seed": kf_seed,
            },
        )

    train_ba_arr = []
    test_ba_arr = []
    train_acc_arr = []
    test_acc_arr = []
    train_cmat_arr = []
    test_cmat_arr = []
    cohen_kappa = []

    strat_label = [all_demos[x] for x in all_paths]
    pbar = tqdm(kf.split(all_paths, strat_label), total=n_splits)
    for idx, (train_idx, test_idx) in enumerate(pbar):
        train_fnames = all_paths[train_idx]
        test_fnames = all_paths[test_idx]

        if feature_name == "persistence_landscape":
            train_data, test_data = tda_utils.convert_pd_pl_train_test(
                all_data,
                train_fnames,
                test_fnames,
            )
        elif feature_name == "template_function":
            train_data, test_data = tda_utils.apply_template_function(
                all_data,
                train_fnames,
                test_fnames,
            )
        else:
            train_data = np.concatenate(
                [all_data[pt_fname] for pt_fname in train_fnames],
                axis=0,
            )

            test_data = np.concatenate(
                [all_data[pt_fname] for pt_fname in test_fnames],
                axis=0,
            )

        train_label = np.concatenate(
            [all_label[pt_fname] for pt_fname in train_fnames],
            axis=0,
        )
        test_label = np.concatenate(
            [all_label[pt_fname] for pt_fname in test_fnames],
            axis=0,
        )

        # Removing outliers
        #  train_data = utils.remove_outliers(train_data)
        #  test_data = utils.remove_outliers(test_data)

        # Z-normalization
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_data)

        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        sample_weights = sklearn.utils.class_weight.compute_sample_weight(
            class_weight="balanced",
            y=train_label,
        )

        model.fit(train_data, train_label, sample_weight=sample_weights)

        train_pred = model.predict(train_data)
        test_pred = model.predict(test_data)

        train_ba = sklearn.metrics.balanced_accuracy_score(train_label, train_pred)
        test_ba = sklearn.metrics.balanced_accuracy_score(test_label, test_pred)

        train_acc = sklearn.metrics.accuracy_score(train_label, train_pred)
        test_acc = sklearn.metrics.accuracy_score(test_label, test_pred)

        train_cmat = sklearn.metrics.confusion_matrix(train_label, train_pred)
        test_cmat = sklearn.metrics.confusion_matrix(test_label, test_pred)

        cohen_kappa_score = sklearn.metrics.cohen_kappa_score(test_pred, test_label)

        train_cmat_arr.append(train_cmat)
        test_cmat_arr.append(test_cmat)

        train_ba_arr.append(train_ba)
        test_ba_arr.append(test_ba)
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)
        cohen_kappa.append(cohen_kappa_score)

        feature_importance = model.get_booster().get_score(importance_type="gain")

        if use_wandb:
            wandb.log(
                {
                    f"train_ba": train_ba,
                    f"test_ba": test_ba,
                    f"train_acc": train_acc,
                    f"test_acc": test_acc,
                    f"train_cmat": train_cmat,
                    f"test_cmat": test_cmat,
                    f"feature_importance": feature_importance,
                    f"cohen_kappa": cohen_kappa_score,
                }
            )

    print(f"{n_splits}-fold validation:")
    print_result("Train Balanced Accuracy", train_ba_arr)
    print_result("Test Balanced Accuracy", test_ba_arr)
    print_result("Train Accuracy", train_acc_arr)
    print_result("Test Accuracy", test_acc_arr)
    print_result("Cohen's Kappa", cohen_kappa)

    print()

    avg_train_cmat = np.stack(train_cmat_arr).mean(0)
    avg_test_cmat = np.stack(test_cmat_arr).mean(0)
    train_class_acc = avg_train_cmat.diagonal() / avg_train_cmat.sum(axis=1)
    test_class_acc = avg_test_cmat.diagonal() / avg_test_cmat.sum(axis=1)

    print("Average Train Confusion Matrix")
    print(avg_train_cmat)
    print()
    print(train_class_acc)

    print("Average Test Confusion Matrix")
    print(avg_test_cmat)
    print()
    print(test_class_acc)


def print_result(key, arr):
    print(f"{key}: {np.mean(arr):.3f} ({np.std(arr):.3f})")


if __name__ == "__main__":
    Fire(train)
    #  Fire(xgb_param_search)
