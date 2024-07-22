import typing
import os

from xgboost import XGBClassifier
from fire import Fire
from tqdm import tqdm

import sklearn.model_selection
import numpy as np
import sklearn

import utils


def train(data_dir: str):
    subject_fnames = os.listdir(data_dir)

    all_paths = np.asarray([os.path.join(data_dir, x) for x in subject_fnames])
    all_demos = utils.get_demographics(
        all_paths,
        data_dir="/work/thesathlab/nchsdb/",
    )
    all_data, all_label = utils.load_data(all_paths)

    map_type = utils.wake_nrem_rem_map
    for k, v in all_label.items():
        all_label[k] = np.asarray(list(map(map_type, v)))

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
        seed=999,
        #  use_label_encoder=False,
    )

    n_splits = 5
    #  kf = sklearn.model_selection.KFold(
    #      n_splits=n_splits,
    #      random_state=2024,
    #      shuffle=True,
    #  )

    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits,
        random_state=2024,
        shuffle=True,
    )

    train_ba_arr = []
    test_ba_arr = []
    train_acc_arr = []
    test_acc_arr = []
    train_cmat_arr = []
    test_cmat_arr = []

    strat_label = [all_demos[x] for x in all_paths]
    for idx, (train_idx, test_idx) in enumerate(
        #  tqdm(kf.split(all_paths), total=n_splits)
        tqdm(kf.split(all_paths, strat_label), total=n_splits)
    ):
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

        #  train_class_acc = train_cmat.diagonal() / train_cmat.sum(axis=1)
        #  test_class_acc = test_cmat.diagonal() / test_cmat.sum(axis=1)
        #  print(train_class_acc)
        #  print(test_class_acc)

        train_ba_arr.append(train_ba)
        test_ba_arr.append(test_ba)
        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

        #  print(f"Train Balanced Accuracy: {train_ba:.3f}")
        #  print(f"Test Balanced Accuracy: {test_ba:.3f}")
        #  print()
        #
        #  print(f"Train Accuracy: {train_acc:.3f}")
        #  print(f"Test Accuracy: {test_acc:.3f}")
        #  print()

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


def print_result(key, arr):
    print(f"{key}: {np.mean(arr):.3f} ({np.std(arr):.3f})")


if __name__ == "__main__":
    Fire(train)
