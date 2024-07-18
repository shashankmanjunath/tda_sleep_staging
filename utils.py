import typing

import numpy as np


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
