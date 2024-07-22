import typing
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
import h5py


def drop_nan_rows(
    arr: np.ndarray, label: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    #  clean_arr = arr[~np.isnan(arr).any(axis=1)]
    #  clean_label = label[~np.isnan(arr).any(axis=1)]
    clean_arr = arr
    clean_arr[np.isnan(clean_arr)] = 0
    clean_arr[np.isinf(clean_arr)] = 0
    return clean_arr, label


def remove_outliers(arr: np.ndarray) -> np.ndarray:
    med_val = np.median(arr, axis=0)
    ul = med_val + 4 * np.abs(med_val)
    ll = med_val - 4 * np.abs(med_val)

    for idx in range(arr.shape[1]):
        pass
    pass


ORDINAL_MAP = {
    "sleep stage w": 0,
    "sleep stage n1": 1,
    "sleep stage n2": 2,
    "sleep stage n3": 3,
    "sleep stage r": 4,
}


def identity_map(sample: str) -> int:
    return ORDINAL_MAP[sample]


def sleep_wake_map(sample: str) -> int:
    if sample == "sleep stage w":
        return 0
    return 1


def wake_nrem_rem_map(sample: str) -> int:
    if sample == "sleep stage w":
        return 0
    elif sample == "sleep stage r":
        return 2
    return 1


def load_data(fnames_list: typing.List[str]) -> typing.Tuple[dict, dict]:
    """Helper function to load data and labels from a list of hdf5 files"""
    data = {}
    label = {}
    for fname in tqdm(fnames_list):
        fname_label = pd.read_hdf(fname, key="tda_label")
        label_arr = fname_label["description"].to_numpy()
        with h5py.File(fname, "r") as f:
            # Loading data
            data_arr = f["tda_feature"][()]
        # Removing nan data
        data_arr_corr, label_arr_corr = drop_nan_rows(data_arr, label_arr)
        label[fname] = label_arr_corr
        data[fname] = data_arr_corr
    return data, label


def get_demographics(subject_list: typing.List[str], data_dir: str) -> dict:
    study_data_fname = os.path.join(data_dir, "health_data", "SLEEP_STUDY.csv")
    study_data = pd.read_csv(study_data_fname)

    demographic_data_fname = os.path.join(data_dir, "health_data", "DEMOGRAPHIC.csv")
    demographic_data = pd.read_csv(demographic_data_fname)
    demographic_data["STUDY_PAT_ID"] = demographic_data["STUDY_PAT_ID"].astype(str)

    study_data["STUDY_PAT_ID"] = study_data["STUDY_PAT_ID"].astype(str)
    study_data["SLEEP_STUDY_ID"] = study_data["SLEEP_STUDY_ID"].astype(str)
    study_data["PT_ID"] = study_data[["STUDY_PAT_ID", "SLEEP_STUDY_ID"]].agg(
        "_".join, axis=1
    )

    demo_output = {}
    for pt_fname in subject_list:
        study_id = pt_fname.split("/")[-1].replace(".hdf5", "")
        pt_study_data = study_data[study_data["PT_ID"] == study_id]

        age = pt_study_data["AGE_AT_SLEEP_STUDY_DAYS"].item() / 365

        pt_id = study_id.split("_")[0]
        pt_demo_data = demographic_data[demographic_data["STUDY_PAT_ID"] == pt_id]
        sex = pt_demo_data["PCORI_GENDER_CD"].item()

        demo_output[pt_fname] = f"{int(age)}_{sex}"
    return demo_output
