import typing

import pandas as pd
import numpy as np


def filter_data(
    data: dict[str, np.ndarray],
    label: dict[str, pd.DataFrame],
    all_paths: np.ndarray,
    filter_type: str,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], np.ndarray]:
    filter_func = get_filter_func(filter_type)

    new_paths = []
    for p in all_paths:
        if p in label.keys():
            is_valid_data = filter_func(label[p])
            if not is_valid_data:
                data.pop(p)
                label.pop(p)
            else:
                new_paths.append(p)

    new_paths = np.asarray(new_paths)
    return data, label, new_paths


def get_filter_func(filter_type: str) -> typing.Callable:
    if filter_type == "low_ahi_odi":
        return low_ahi_odi_filter
    elif filter_type == "low_ahi":
        return low_ahi_filter
    else:
        raise RuntimeError("Filter type not recognized!")


def check_filter_func(filter_type: str) -> None:
    """Helper function to check that filter type passed is valid before loading
    data
    """
    _ = get_filter_func(filter_type)
    return


def low_ahi_odi_filter(df: pd.DataFrame) -> bool:
    # Looking for low ahi < 1 and low oxygen desaturation index
    sleep_duration_h = (df["onset"].max() - df["onset"].min()) / 3600.0
    events_df = df[[len(x) > 0 for x in df["events"].tolist()]]

    events_list = events_df["events"].tolist()
    apnea_df = events_df[["apnea" in "\t".join(x) for x in events_list]]
    hypopnea_df = events_df[["hypopnea" in "\t".join(x) for x in events_list]]
    oxygen_desat_df = events_df[
        ["oxygen desaturation" in "\t".join(x) for x in events_list]
    ]

    ahi = (apnea_df.shape[0] + hypopnea_df.shape[0]) / sleep_duration_h
    odi = oxygen_desat_df.shape[0] / sleep_duration_h

    if ahi >= 1 or odi >= 1:
        return False
    return True


def low_ahi_filter(df: pd.DataFrame) -> bool:
    # Looking for low ahi < 1
    sleep_duration_h = (df["onset"].max() - df["onset"].min()) / 3600.0
    events_df = df[[len(x) > 0 for x in df["events"].tolist()]]

    events_list = events_df["events"].tolist()
    apnea_df = events_df[["apnea" in "\t".join(x) for x in events_list]]
    hypopnea_df = events_df[["hypopnea" in "\t".join(x) for x in events_list]]

    ahi = (apnea_df.shape[0] + hypopnea_df.shape[0]) / sleep_duration_h

    if ahi >= 1:
        return False
    return True
