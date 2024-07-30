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

import utils


def xgb_param_search(data_dir: str, feature_name: str):
    subject_fnames = os.listdir(data_dir)
    subject_fnames = subject_fnames[:100]

    all_paths = np.asarray([os.path.join(data_dir, x) for x in subject_fnames])
    all_demos = utils.get_demographics(
        all_paths,
        data_dir="/work/thesathlab/nchsdb/",
    )
    all_data, all_label = utils.load_data(all_paths, feature_name, sqi_thresh=0.25)

    map_type = utils.wake_nrem_rem_map
    for k, v in all_label.items():
        all_label[k] = np.asarray(list(map(map_type, v)))

    model = XGBClassifier(
        n_jobs=1,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )

    n_splits = 5
    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits,
        random_state=2024,
        shuffle=True,
    )

    bayes_cv_tuner = skopt.BayesSearchCV(
        estimator=model,
        search_spaces={
            "learning_rate": (0.01, 1.0, "log-uniform"),
            "min_child_weight": (0, 10),
            "max_depth": (0, 50),
            "max_delta_step": (0, 20),
            "subsample": (0.01, 1.0, "uniform"),
            "colsample_bytree": (0.01, 1.0, "uniform"),
            "colsample_bylevel": (0.01, 1.0, "uniform"),
            "reg_lambda": (1e-9, 1000, "log-uniform"),
            "reg_alpha": (1e-9, 1.0, "log-uniform"),
            "gamma": (1e-9, 0.5, "log-uniform"),
            "n_estimators": (50, 100),
        },
        scoring="balanced_accuracy",
        n_jobs=3,
        n_iter=3,
        cv=3,
        verbose=1,
        refit=True,
        random_state=999,
    )

    train_ba_arr = []
    test_ba_arr = []
    train_acc_arr = []
    test_acc_arr = []
    train_cmat_arr = []
    test_cmat_arr = []

    strat_label = [all_demos[x] for x in all_paths]
    pbar = tqdm(kf.split(all_paths, strat_label), total=n_splits)
    for idx, (train_idx, test_idx) in enumerate(pbar):
        train_fnames = all_paths[train_idx]
        test_fnames = all_paths[test_idx]

        train_data = np.concatenate(
            [all_data[pt_fname] for pt_fname in train_fnames],
            axis=0,
        )
        train_label = np.concatenate(
            [all_label[pt_fname] for pt_fname in train_fnames],
            axis=0,
        )

        test_data = np.concatenate(
            [all_data[pt_fname] for pt_fname in test_fnames],
            axis=0,
        )
        test_label = np.concatenate(
            [all_label[pt_fname] for pt_fname in test_fnames],
            axis=0,
        )

        # Z-normalization
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_data)

        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)

        sample_weights = sklearn.utils.class_weight.compute_sample_weight(
            class_weight="balanced",
            y=train_label,
        )

        bayes_cv_tuner.fit(train_data, train_label, sample_weight=sample_weights)

        train_pred = bayes_cv_tuner.predict(train_data)
        test_pred = bayes_cv_tuner.predict(test_data)

        train_ba = sklearn.metrics.balanced_accuracy_score(train_label, train_pred)
        test_ba = sklearn.metrics.balanced_accuracy_score(test_label, test_pred)

        train_acc = sklearn.metrics.accuracy_score(train_label, train_pred)
        test_acc = sklearn.metrics.accuracy_score(test_label, test_pred)

        train_cmat = sklearn.metrics.confusion_matrix(train_label, train_pred)
        test_cmat = sklearn.metrics.confusion_matrix(test_label, test_pred)

        train_cmat_arr.append(train_cmat)
        test_cmat_arr.append(test_cmat)

        train_ba_arr.append(train_ba)
        test_ba_arr.append(test_ba)
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

    print(f"{n_splits}-fold validation:")
    print_result("Train Balanced Accuracy", train_ba_arr)
    print_result("Test Balanced Accuracy", test_ba_arr)
    print_result("Train Accuracy", train_acc_arr)
    print_result("Test Accuracy", test_acc_arr)

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
    pass


def train(data_dir: str, feature_name: str):
    subject_fnames = os.listdir(data_dir)
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

    # wandb setup
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

    strat_label = [all_demos[x] for x in all_paths]
    pbar = tqdm(kf.split(all_paths, strat_label), total=n_splits)
    for idx, (train_idx, test_idx) in enumerate(pbar):
        train_fnames = all_paths[train_idx]
        test_fnames = all_paths[test_idx]

        train_data = np.concatenate(
            [all_data[pt_fname] for pt_fname in train_fnames],
            axis=0,
        )
        train_label = np.concatenate(
            [all_label[pt_fname] for pt_fname in train_fnames],
            axis=0,
        )

        test_data = np.concatenate(
            [all_data[pt_fname] for pt_fname in test_fnames],
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

        train_cmat_arr.append(train_cmat)
        test_cmat_arr.append(test_cmat)

        train_ba_arr.append(train_ba)
        test_ba_arr.append(test_ba)
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

        wandb.log(
            {
                f"train_ba_fold_{idx}": train_ba,
                f"test_ba_fold_{idx}": test_ba,
                f"train_acc_fold_{idx}": train_acc,
                f"test_acc_fold_{idx}": test_acc,
                f"train_cmat_fold_{idx}": train_cmat,
                f"test_cmat_fold_{idx}": test_cmat,
            }
        )

    print(f"{n_splits}-fold validation:")
    print_result("Train Balanced Accuracy", train_ba_arr)
    print_result("Test Balanced Accuracy", test_ba_arr)
    print_result("Train Accuracy", train_acc_arr)
    print_result("Test Accuracy", test_acc_arr)

    print()

    avg_train_cmat = np.stack(train_cmat_arr).mean(0)
    avg_test_cmat = np.stack(test_cmat_arr).mean(0)
    train_class_acc = avg_train_cmat.diagonal() / avg_train_cmat.sum(axis=1)
    test_class_acc = avg_test_cmat.diagonal() / avg_test_cmat.sum(axis=1)

    wandb.log(
        {
            "train_ba_all": train_ba_arr,
            "test_ba_all": test_ba_arr,
            "train_acc_all": train_acc_arr,
            "test_acc_all": test_acc_arr,
            "train_class_acc": train_class_acc,
            "test_class_acc": test_class_acc,
            "avg_train_cmat": avg_train_cmat,
            "avg_test_cmat": avg_test_cmat,
        }
    )

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
