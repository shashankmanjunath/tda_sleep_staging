import typing
import pickle
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy
import h5py
import pywt

import tda_utils

proc_items = {
    "airflow_sublevel": 0,
    "airflow_rips": 0,
    "irr_sublevel": 0,
}


def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25


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


def load_data_hdf5(
    fnames_list: typing.List[str],
    feature_name: str,
    sqi_thresh: float,
) -> typing.Tuple[dict, dict]:
    """Helper function to load data saved in hdf5 files"""
    n_feat = 15
    data = {}
    label = {}

    for fname in tqdm(fnames_list):
        with h5py.File(fname, "r") as f:
            # Loading data
            if feature_name == "all":
                data_arr_tda = np.concatenate(
                    [
                        f["ps_sub_airflow_0"][()],
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["ps_rips_airflow_1"][()],
                        f["ps_irr"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )

                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "all_fft":
                data_arr_tda = np.concatenate(
                    [
                        f["ps_sub_airflow_0"][()],
                        np.abs(f["fft_sub_airflow_0"][()][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_0"][()][:, :n_feat]),
                        f["ps_rips_airflow_1"][()],
                        f["ps_irr"][()],
                        np.abs(f["fft_irr"][()][:, :n_feat]),
                    ],
                    axis=-1,
                )

                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "all_fft_ri":
                data_arr_tda = np.concatenate(
                    [
                        f["ps_sub_airflow_0"][()],
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        f["ps_rips_airflow_1"][()],
                        f["ps_irr"][()],
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )

                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "ps":
                data_arr = np.concatenate(
                    [
                        f["ps_sub_airflow_0"][()],
                        f["ps_rips_airflow_1"][()],
                        f["ps_irr"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "tda":
                data_arr = np.concatenate(
                    [
                        f["ps_sub_airflow_0"][()],
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["ps_rips_airflow_1"][()],
                        f["ps_irr"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "tda_fft":
                data_arr = np.concatenate(
                    [
                        f["ps_sub_airflow_0"][()],
                        np.abs(f["fft_sub_airflow_0"][()][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_0"][()][:, :n_feat]),
                        f["ps_rips_airflow_1"][()],
                        f["ps_irr"][()],
                        np.abs(f["fft_irr"][()][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "tda_fft_ri":
                data_arr = np.concatenate(
                    [
                        f["ps_sub_airflow_0"][()],
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        f["ps_rips_airflow_1"][()],
                        f["ps_irr"][()],
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "classic":
                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
            elif feature_name == "classic_hepc":
                data_arr_tda = np.concatenate(
                    [
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )

                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "classic_fft":
                data_arr_tda = np.concatenate(
                    [
                        np.abs(f["fft_sub_airflow_0"][()][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_0"][()][:, :n_feat]),
                        np.abs(f["fft_irr"][()][:, :n_feat]),
                    ],
                    axis=-1,
                )

                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "classic_fft_ri":
                data_arr_tda = np.concatenate(
                    [
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )

                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "classic_hepc_full":
                data_arr_tda = np.concatenate(
                    [
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["hepc_rips_airflow_1"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )
                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "classic_fft_full":
                data_arr_tda = np.concatenate(
                    [
                        np.abs(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.abs(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "classic_fft_full_ri":
                data_arr_tda = np.concatenate(
                    [
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
                data_arr_classic = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["resp_vol_25_epoch_cent"][()],
                        f["power_25_epoch_cent"][()],
                    ]
                )
                data_arr = np.concatenate((data_arr_tda, data_arr_classic), axis=-1)
            elif feature_name == "hepc":
                data_arr = np.concatenate(
                    [
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "hepc_np":
                data_arr = np.concatenate(
                    [
                        f["hepc_np_sub_airflow_0"][()],
                        f["hepc_np_rips_airflow_0"][()],
                        f["hepc_np_irr"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "hepc_full":
                data_arr = np.concatenate(
                    [
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["hepc_rips_airflow_1"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "fft":
                data_arr = np.concatenate(
                    [
                        np.abs(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "fft_ri":
                data_arr = np.concatenate(
                    [
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "fft_full":
                data_arr = np.concatenate(
                    [
                        np.abs(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.abs(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "fft_full_ri":
                data_arr = np.concatenate(
                    [
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "classic_single_epoch":
                data_arr = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "classic_single_epoch_hepc":
                data_arr = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "classic_single_epoch_fft":
                data_arr = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        np.abs(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.abs(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "classic_single_epoch_fft_ri":
                data_arr = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "classic_single_epoch_hepc_full":
                data_arr = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        f["hepc_sub_airflow_0"][()],
                        f["hepc_rips_airflow_0"][()],
                        f["hepc_rips_airflow_1"][()],
                        f["hepc_irr"][()],
                    ],
                    axis=-1,
                )
            elif feature_name == "classic_single_epoch_fft_full_ri":
                data_arr = np.concatenate(
                    [
                        f["breath_cycle_1_epoch"][()],
                        np.real(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_sub_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_0"][:, :n_feat]),
                        np.real(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.imag(f["fft_rips_airflow_1"][:, :n_feat]),
                        np.real(f["fft_irr"][:, :n_feat]),
                        np.imag(f["fft_irr"][:, :n_feat]),
                    ],
                    axis=-1,
                )
            elif feature_name == "random":
                data_arr = np.random.rand(*f["breath_cycle_1_epoch"].shape)
            else:
                data_arr = f[feature_name][()]
            sqi_arr = f["sqi"][()]

        fname_label = pd.read_hdf(fname, key="label")
        label_arr = fname_label["description"].to_numpy()

        # Removing low SQI data
        data_arr_sqi = data_arr[sqi_arr >= sqi_thresh, :]
        label_arr_sqi = label_arr[sqi_arr >= sqi_thresh]

        # Removing nan data
        data_arr_corr, label_arr_corr = drop_nan_rows(data_arr_sqi, label_arr_sqi)
        label[fname] = label_arr_corr
        data[fname] = data_arr_corr

    return data, label


def load_data_pkl(
    fnames_list: typing.List[str],
    feature_name: str,
    sqi_thresh: float,
) -> typing.Tuple[dict, dict]:
    fnames_list_pkl = [x.replace(".hdf5", ".pkl") for x in fnames_list]

    data = {}
    label = {}
    for fname_pkl, fname_hdf5 in zip(tqdm(fnames_list_pkl), fnames_list):
        with open(fname_pkl, "rb") as f:
            data_arr = pickle.load(f)

        with h5py.File(fname_hdf5, "r") as f:
            sqi_arr = f["sqi"][()]

        feat_arr = []
        for sqi_idx in range(len(sqi_arr)):
            feat_arr_idx = {}
            for k, v in data_arr.items():
                feat_arr_idx[k] = v[sqi_idx]

            feat_arr.append(feat_arr_idx)

        fname_label = pd.read_hdf(fname_hdf5, key="label")
        label_arr = fname_label["description"].to_numpy()

        feat_arr_sqi = []
        for sqi, feat in zip(sqi_arr, feat_arr):
            if sqi >= sqi_thresh:
                if feature_name == "cwt":
                    cwt_feat = calculate_cwt_feat(feat, n_levels=10)
                    feat_arr_sqi.append(cwt_feat)
                elif feature_name == "haar":
                    haar_feat = calculate_haar_feat(feat, n_levels=12)
                    feat_arr_sqi.append(haar_feat)
                elif feature_name == "hermfit":
                    herm_feat = calculate_hermfit(feat, n_levels=15)
                    feat_arr_sqi.append(herm_feat)
                elif feature_name == "lagfit":
                    herm_feat = calculate_lagfit(feat, n_levels=15)
                    feat_arr_sqi.append(herm_feat)
                elif feature_name == "wavedec":
                    w_feat = calculate_wavedec(feat, n_levels=6)
                    feat_arr_sqi.append(w_feat)
                elif feature_name == "polyfit":
                    w_feat = calculate_polyfit(feat, n_levels=15)
                    feat_arr_sqi.append(w_feat)
                elif feature_name == "fft":
                    w_feat = calculate_fft_feat(feat, n_coef=15)
                    feat_arr_sqi.append(w_feat)
                else:
                    feat_arr_sqi.append(feat)
        label_arr_sqi = label_arr[sqi_arr >= sqi_thresh]

        data[fname_hdf5] = feat_arr_sqi
        label[fname_hdf5] = label_arr_sqi
    return data, label


def load_data(
    fnames_list: typing.List[str],
    feature_name: str,
    sqi_thresh: float,
) -> typing.Tuple[dict, dict]:
    """Helper function to load data and labels from a list of files"""
    if feature_name in [
        "persistence_landscape",
        "template_function",
        "cwt",
        "haar",
        "hermfit",
        "wavedec",
        "lagfit",
        "polyfit",
    ]:
        data, label = load_data_pkl(fnames_list, feature_name, sqi_thresh)
    else:
        data, label = load_data_hdf5(fnames_list, feature_name, sqi_thresh)
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


def calculate_sqi(epoch_data: np.ndarray, sampling_freq: float) -> float:
    bp_lo_freq = 0.1
    bp_hi_freq = 0.75
    freq_spread = 4

    sos_butter_bandpass = scipy.signal.butter(
        N=3,
        Wn=[bp_lo_freq, bp_hi_freq],
        btype="bandpass",
        output="sos",
        fs=sampling_freq,
    )

    # Bandpass signal
    bandpass_signal = scipy.signal.sosfilt(sos_butter_bandpass, epoch_data)

    # Fourier transform
    fft_arr = np.abs(scipy.fft.fft(bandpass_signal)).squeeze()
    fft_arr = np.power(fft_arr, 2)
    ts = 1.0 / sampling_freq
    freqs = scipy.fft.fftfreq(fft_arr.shape[-1], d=ts)

    # Finding peak in spectrum
    valid_freqs = (freqs >= bp_lo_freq) & (freqs <= bp_hi_freq)
    fft_arr = fft_arr[valid_freqs]
    freqs = freqs[valid_freqs]
    max_freq_idx = fft_arr.argmax()

    # Calculating SQI
    signal_power = fft_arr.sum()
    low_idx = max_freq_idx - freq_spread
    hi_idx = max_freq_idx + freq_spread
    maxpow_band_power = fft_arr[low_idx:hi_idx].sum()
    sqi = maxpow_band_power / signal_power
    return sqi


def calculate_cwt_feat(data_dict: typing.Dict, n_levels: int):
    levels = {
        "airflow_sublevel": 10,
        "airflow_rips": 2,
        "irr_sublevel": 2,
    }
    #  min_pow = -1
    #  scales = 2.0 ** np.arange(min_pow, min_pow + n_levels).astype(float)
    #  scales = np.arange(1, n_levels + 1)
    scales = np.asarray([0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])
    w = pywt.Wavelet("db2")

    output_arr = []
    for k, dgm_num in proc_items.items():
        target_dgm = data_dict[k][dgm_num]
        dgm_clean = target_dgm[~np.isinf(target_dgm).any(1)]
        psi_dgm = tda_utils.psi(dgm_clean)
        if psi_dgm.size > 0:
            coef, _ = pywt.cwt(psi_dgm, scales, "morl")
            #  coef = pywt.mra(psi_dgm, w, level=levels[k], transform="dwt")
            #  output_arr.append(coef.sum(1))
            #  output_arr.append(np.stack(coef).sum(-1))
            output_arr.append(np.stack(coef).reshape(-1))
        else:
            #  output_arr.append(np.zeros((scales.shape[0],)))
            output_arr.append(np.zeros((psi_dgm.size * (levels[k] + 1),)))
    output_arr = np.concatenate(output_arr, axis=-1)
    return output_arr


def calculate_haar_feat(data_dict: typing.Dict, n_levels: int):
    output_arr = []
    for k, dgm_num in proc_items.items():
        target_dgm = data_dict[k][dgm_num]
        dgm_clean = target_dgm[~np.isinf(target_dgm).any(1)]
        psi_dgm = tda_utils.psi(dgm_clean)
        if psi_dgm.size > 0:
            alpha = []
            n_level_sum = 0.0
            for idx, (b, d) in enumerate(dgm_clean):
                int_val, _ = scipy.integrate.quad(
                    lambda x: tda_utils.h_n(x, n_levels),
                    b,
                    d,
                )
                val = psi_dgm[idx] * int_val
                n_level_sum += val
                alpha.append(n_level_sum)
            output_arr.append(alpha)
        else:
            output_arr.append(np.zeros((n_levels,)))
    output_arr = np.concatenate(output_arr, axis=-1)
    return output_arr


def calculate_hermfit(data_dict: typing.Dict, n_levels: int):
    output_arr = []
    for k, dgm_num in proc_items.items():
        target_dgm = data_dict[k][dgm_num]
        dgm_clean = target_dgm[~np.isinf(target_dgm).any(1)]
        psi_dgm = tda_utils.psi(dgm_clean)
        if psi_dgm.size > 0:
            x = np.linspace(0, dgm_clean.max(), 1000)
            y = np.zeros(x.shape)

            for idx, (b, d) in enumerate(dgm_clean):
                arr_idx = (x >= b) & (x <= d)
                y[arr_idx] += psi_dgm[idx]
            alpha = np.polynomial.hermite.hermfit(x, y, deg=n_levels - 1)
            output_arr.append(alpha)
        else:
            output_arr.append(np.zeros((n_levels,)))
    output_arr = np.concatenate(output_arr, axis=-1)
    return output_arr


def calculate_polyfit(data_dict: typing.Dict, n_levels: int):
    output_arr = []
    for k, dgm_num in proc_items.items():
        target_dgm = data_dict[k][dgm_num]
        dgm_clean = target_dgm[~np.isinf(target_dgm).any(1)]
        psi_dgm = tda_utils.psi(dgm_clean)
        if psi_dgm.size > 0:
            x = np.linspace(0, dgm_clean.max(), 1000)
            y = np.zeros(x.shape)

            for idx, (b, d) in enumerate(dgm_clean):
                arr_idx = (x >= b) & (x <= d)
                y[arr_idx] += psi_dgm[idx]
            alpha = np.polynomial.polynomial.polyfit(x, y, deg=n_levels - 1)
            output_arr.append(alpha)
        else:
            output_arr.append(np.zeros((n_levels,)))
    output_arr = np.concatenate(output_arr, axis=-1)
    return output_arr


def calculate_fft_feat(data_dict: typing.Dict, n_coef: int):
    output_arr = []
    for k, dgm_num in proc_items.items():
        target_dgm = data_dict[k][dgm_num]
        dgm_clean = target_dgm[~np.isinf(target_dgm).any(1)]
        psi_dgm = tda_utils.psi(dgm_clean)
        if psi_dgm.size > 0:
            x = np.linspace(0, dgm_clean.max(), 1000)
            y = np.zeros(x.shape)

            for idx, (b, d) in enumerate(dgm_clean):
                arr_idx = (x >= b) & (x <= d)
                y[arr_idx] += psi_dgm[idx]
            coef = np.fft.rfft(y)
            alpha = coef[:n_coef]
            output_arr.append(np.abs(alpha))
        else:
            output_arr.append(np.zeros((n_coef,)))
    output_arr = np.concatenate(output_arr, axis=-1)
    return output_arr


def calculate_lagfit(data_dict: typing.Dict, n_levels: int):
    output_arr = []
    for k, dgm_num in proc_items.items():
        target_dgm = data_dict[k][dgm_num]
        dgm_clean = target_dgm[~np.isinf(target_dgm).any(1)]
        psi_dgm = tda_utils.psi(dgm_clean)
        if psi_dgm.size > 0:
            x = np.linspace(0, dgm_clean.max(), 1000)
            y = np.zeros(x.shape)

            for idx, (b, d) in enumerate(dgm_clean):
                arr_idx = (x >= b) & (x <= d)
                y[arr_idx] += psi_dgm[idx]
            alpha = np.polynomial.laguerre.lagfit(x, y, deg=n_levels - 1)
            output_arr.append(alpha)
        else:
            output_arr.append(np.zeros((n_levels,)))
    output_arr = np.concatenate(output_arr, axis=-1)
    return output_arr


def calculate_wavedec(data_dict: typing.Dict, n_levels: int):
    w = pywt.Wavelet("haar")
    output_arr = []
    for k, dgm_num in proc_items.items():
        target_dgm = data_dict[k][dgm_num]
        dgm_clean = target_dgm[~np.isinf(target_dgm).any(1)]
        psi_dgm = tda_utils.psi(dgm_clean)
        if psi_dgm.size > 0:
            coef_arr = pywt.wavedec(psi_dgm, wavelet=w, level=3)
            output_arr.append(np.asarray([x.sum() for x in coef_arr]))
        else:
            #  output_arr.append(np.zeros((scales.shape[0],)))
            output_arr.append(np.zeros(4))
    output_arr = np.concatenate(output_arr, axis=-1)
    return output_arr


def get_tda_feature_names() -> typing.List[str]:
    feature_names = [
        # ps_sub_airflow_0
        "airflow_sublevel_midlife_mean_h0",
        "airflow_sublevel_midlife_std_h0",
        "airflow_sublevel_midlife_skew_h0",
        "airflow_sublevel_midlife_kurt_h0",
        "airflow_sublevel_midlife_entropy_h0",
        "airflow_sublevel_lifespan_mean_h0",
        "airflow_sublevel_lifespan_std_h0",
        "airflow_sublevel_lifespan_skew_h0",
        "airflow_sublevel_lifespan_kurt_h0",
        "airflow_sublevel_lifespan_entropy_h0",
        "airflow_gaussian_persistence_curve_h0",
        # hepc_sub_airflow_0
        "airflow_sublevel_hepc_0_h0",
        "airflow_sublevel_hepc_1_h0",
        "airflow_sublevel_hepc_2_h0",
        "airflow_sublevel_hepc_3_h0",
        "airflow_sublevel_hepc_4_h0",
        "airflow_sublevel_hepc_5_h0",
        "airflow_sublevel_hepc_6_h0",
        "airflow_sublevel_hepc_7_h0",
        "airflow_sublevel_hepc_8_h0",
        "airflow_sublevel_hepc_9_h0",
        "airflow_sublevel_hepc_10_h0",
        "airflow_sublevel_hepc_11_h0",
        "airflow_sublevel_hepc_12_h0",
        "airflow_sublevel_hepc_13_h0",
        "airflow_sublevel_hepc_14_h0",
        # hepc_rips_airflow_0
        "airflow_rips_hepc_0_h0",
        "airflow_rips_hepc_1_h0",
        "airflow_rips_hepc_2_h0",
        "airflow_rips_hepc_3_h0",
        "airflow_rips_hepc_4_h0",
        "airflow_rips_hepc_5_h0",
        "airflow_rips_hepc_6_h0",
        "airflow_rips_hepc_7_h0",
        "airflow_rips_hepc_8_h0",
        "airflow_rips_hepc_9_h0",
        "airflow_rips_hepc_10_h0",
        "airflow_rips_hepc_11_h0",
        "airflow_rips_hepc_12_h0",
        "airflow_rips_hepc_13_h0",
        "airflow_rips_hepc_14_h0",
        # ps_rips_airflow_1
        "airflow_sublevel_midlife_mean_h1",
        "airflow_sublevel_midlife_std_h1",
        "airflow_sublevel_midlife_skew_h1",
        "airflow_sublevel_midlife_kurt_h1",
        "airflow_sublevel_midlife_entropy_h1",
        "airflow_sublevel_lifespan_mean_h1",
        "airflow_sublevel_lifespan_std_h1",
        "airflow_sublevel_lifespan_skew_h1",
        "airflow_sublevel_lifespan_kurt_h1",
        "airflow_sublevel_lifespan_entropy_h1",
        "airflow_gaussian_persistence_curve_h1",
        # ps_irr
        "irr_sublevel_midlife_mean_h0",
        "irr_sublevel_midlife_std_h0",
        "irr_sublevel_midlife_skew_h0",
        "irr_sublevel_midlife_kurt_h0",
        "irr_sublevel_midlife_entropy_h0",
        "irr_sublevel_lifespan_mean_h0",
        "irr_sublevel_lifespan_std_h0",
        "irr_sublevel_lifespan_skew_h0",
        "irr_sublevel_lifespan_kurt_h0",
        "irr_sublevel_lifespan_entropy_h0",
        "irr_gaussian_persistence_curve_h0",
        # hepc_irr
        "irr_sublevel_hepc_0_h0",
        "irr_sublevel_hepc_1_h0",
        "irr_sublevel_hepc_2_h0",
        "irr_sublevel_hepc_3_h0",
        "irr_sublevel_hepc_4_h0",
        "irr_sublevel_hepc_5_h0",
        "irr_sublevel_hepc_6_h0",
        "irr_sublevel_hepc_7_h0",
        "irr_sublevel_hepc_8_h0",
        "irr_sublevel_hepc_9_h0",
        "irr_sublevel_hepc_10_h0",
        "irr_sublevel_hepc_11_h0",
        "irr_sublevel_hepc_12_h0",
        "irr_sublevel_hepc_13_h0",
        "irr_sublevel_hepc_14_h0",
    ]
    return feature_names


def get_ntda_feature_names() -> typing.List[str]:
    feature_names = [
        "breathing_cycle_amp_med",
        "breathing_cycle_amp_iqr",
        "breathing_cycle_width_med",
        "breathing_cycle_width_iqr",
        "breathing_cycle_peak_med",
        "breathing_cycle_peak_iqr",
        "breathing_cycle_trough_med",
        "breathing_cycle_trough_iqr",
        "mai",
        "mae",
        "mai/mae",
        "med_peaks/iqr_peaks",
        "med_troughs/iqr_troughs",
        "cycle_amp_med",
        "resp_vol_cycle_med",
        "resp_vol_inhale_med",
        "resp_vol_exhale_med",
        "resp_flow_cycle_med",
        "resp_flow_inhale_med",
        "resp_flow_exhale_med",
        "resp_flow_inhale_med/resp_flow_exhale_med",
        "sample_entropy",
        "log_vlf_pow",
        "log_lf_pow",
        "log_hf_pow",
        "lf/hf",
        "peak_freq_hf",
        "peak_freq_power_hf",
    ]
    return feature_names


def sort_dict_list(data_dict: typing.Dict) -> typing.List:
    output_arr = []
    for k, v in data_dict.items():
        output_arr.append((k, v))

    output_arr = sorted(output_arr, key=lambda x: x[1], reverse=True)
    return output_arr
