import pickle
import typing
import time
import os

from ripser import ripser
from fire import Fire
from tqdm import tqdm

import gtda.time_series
import neurokit2 as nk
import scipy.sparse
import pandas as pd
import numpy as np
import cripser
import h5py
import dtw
import mne

import tda_utils
import utils


fapc_support = {
    "rips_airflow_h0": [0, 0.0002],
    "rips_airflow_h1": [0, 0.0005],
    "sublevel_airflow_h0": [-0.0015, 0.0015],
    "sublevel_irr_h0": [10, 50],
}

hepc_scale = {
    "rips_airflow_h0": 90442.544,
    "rips_airflow_h1": 55034.829,
    "sublevel_airflow_h0": 15909.436,
    "sublevel_irr_h0": 0.164,
}


class EpochItem:
    def __init__(
        self,
        epoch: np.ndarray,
        t_start: float,
        t_len: float,
        label: pd.Series,
    ):
        self.epoch = epoch
        self.t_start = t_start
        self.t_len = t_len
        self.label = label

        # Slack time
        self.t_slack = 61.0

    def overlaps(self, other) -> bool:
        # Allow t_slack difference between old end time and new start time
        end_time = self.t_start + self.t_len
        slack_time = end_time + self.t_slack

        if slack_time < other.t_start:
            #  print(f"Transition: diff time {self.diff_time(other)}")
            return False
        else:
            return True

    def diff_time(self, other) -> float:
        end_time = self.t_start + self.t_len
        diff_time = other.t_start - end_time
        return diff_time


class EpochCache:
    def __init__(self, sampling_freq: float):
        self.min_epochs_valid = 25
        self.sampling_freq = sampling_freq

        self.epoch_cache = [[]]
        self.valid_indexes = []

    def add_epoch(
        self,
        epoch: np.ndarray,
        t_start: float,
        t_len: float,
        label: pd.Series,
    ) -> None:
        if len(self.epoch_cache[-1]) == 0:
            # No epochs in cache; add to cache
            self.epoch_cache[-1].append(EpochItem(epoch, t_start, t_len, label))
        else:
            cur_epoch = EpochItem(epoch, t_start, t_len, label)
            last_epoch = self.epoch_cache[-1][-1]

            if last_epoch.overlaps(cur_epoch):
                # Add epoch to end of cache
                self.epoch_cache[-1].append(cur_epoch)

                if len(self.epoch_cache[-1]) >= self.min_epochs_valid:
                    # Centered epoch is valid
                    n_seq = len(self.epoch_cache)
                    n_item = len(self.epoch_cache[-1]) - (self.min_epochs_valid // 2)

                    # Correcting for 0-indexing
                    n_seq_idx = n_seq - 1
                    n_item_idx = n_item - 1
                    self.valid_indexes.append((n_seq_idx, n_item_idx))
            else:
                # New epoch is not consecutive to previous epoch or is new sleep
                # stage, so we create new sequence and restart
                self.epoch_cache.append([cur_epoch])

    def __len__(self):
        return len(self.valid_indexes)

    def __getitem__(self, idx: int) -> typing.Tuple[np.ndarray, pd.Series]:
        # Getting indexes for cache and sequence
        cache_idx, seq_idx = self.valid_indexes[idx]
        epoch_item = self.epoch_cache[cache_idx][seq_idx]
        data = epoch_item.epoch
        label = epoch_item.label
        return data, label

    def get_epoch_sequence(self, idx: int, n_epochs: int) -> np.ndarray:
        # Getting indexes for cache and sequence
        cache_idx, seq_idx = self.valid_indexes[idx]

        # Calculating indexes of last n_epochs, including the current one
        start_idx = seq_idx - n_epochs + 1
        end_idx = seq_idx + 1

        if start_idx < 0:
            raise ValueError("Too many epochs requested!")

        epoch_seq = [x.epoch for x in self.epoch_cache[cache_idx][start_idx:end_idx]]
        epoch_data = np.concatenate(epoch_seq, axis=-1)
        return epoch_data

    def get_centered_epoch_sequence(self, idx: int, n_epochs: int) -> np.ndarray:
        # Getting indexes for cache and sequence
        cache_idx, seq_idx = self.valid_indexes[idx]

        start_idx = seq_idx - (n_epochs // 2)
        end_idx = seq_idx + (n_epochs // 2) + 1

        epoch_seq = [x.epoch for x in self.epoch_cache[cache_idx][start_idx:end_idx]]
        epoch_data = np.concatenate(epoch_seq, axis=-1)
        return epoch_data

    def get_template_epochs(self, idx: int, n_epochs: int) -> np.ndarray:
        # Getting indexes for cache and sequence
        cache_idx, seq_idx = self.valid_indexes[idx]

        start_idx = seq_idx - (n_epochs // 2)
        end_idx = seq_idx + (n_epochs // 2) + 1

        epoch_seq = []
        for epoch_seq_idx in range(start_idx, end_idx):
            if epoch_seq_idx != seq_idx:
                epoch_seq.append(self.epoch_cache[cache_idx][epoch_seq_idx].epoch)
        epoch_data = np.stack(epoch_seq, axis=0)
        return epoch_data


class AirflowSignalProcessor:
    def __init__(self, pt_id: str, data_dir: str, save_dir: str):
        self.pt_id = pt_id
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.save_fname_hdf5 = os.path.join(self.save_dir, f"{pt_id}.hdf5")
        self.save_fname_pkl = os.path.join(self.save_dir, f"{pt_id}.pkl")

        self.edf_fname = os.path.join(self.data_dir, "sleep_data", f"{self.pt_id}.edf")
        self.tsv_fname = os.path.join(self.data_dir, "sleep_data", f"{self.pt_id}.tsv")
        self.study_data = pd.read_csv(
            os.path.join(self.data_dir, "health_data", "SLEEP_STUDY.csv")
        )

        self.sleep_stage_kw = [
            "sleep stage w",
            "sleep stage n1",
            "sleep stage n2",
            "sleep stage n3",
            "sleep stage r",
        ]

        self.reject_kw = [
            "central apnea",
            "mixed apnea",
            "obstructive apnea",
            "hypoponea",
            "oxygen desaturation",
        ]

        self.d = 3
        self.target_resamp_rate = 4
        self.n_epochs_sqi = 6

        if not os.path.exists(self.edf_fname):
            raise RuntimeError("EDF file for patient id not found!")
        elif not os.path.exists(self.tsv_fname):
            raise RuntimeError("TSV file for patient id not found!")

    def load_pt_study_metadata(self) -> pd.DataFrame:
        self.study_data["STUDY_PAT_ID"] = self.study_data["STUDY_PAT_ID"].astype(str)
        self.study_data["SLEEP_STUDY_ID"] = self.study_data["SLEEP_STUDY_ID"].astype(
            str
        )
        self.study_data["PT_ID"] = self.study_data[
            ["STUDY_PAT_ID", "SLEEP_STUDY_ID"]
        ].agg("_".join, axis=1)
        pt_study_data = self.study_data[self.study_data["PT_ID"] == self.pt_id]

        if pt_study_data.shape[0] > 1:
            raise RuntimeError("Too many sleep studies found!")
        return pt_study_data

    def get_ahi(self) -> float:
        pt_study_data = self.load_pt_study_metadata()
        study_duration = pt_study_data["SLEEP_STUDY_DURATION_DATETIME"].item().strip()

        tm = time.strptime(study_duration, "%H:%M:%S")
        study_dur_h = tm.tm_hour + (tm.tm_min / 60) + (tm.tm_sec / 3600)

        raw_tsv = pd.read_csv(self.tsv_fname, sep="\t")
        raw_tsv["description"] = raw_tsv["description"].apply(lambda x: x.lower())
        event_list = raw_tsv["description"].to_list()
        events = [x for x in event_list if "apnea" in x or "hypopnea" in x]
        ahi = len(events) / study_dur_h
        return ahi

    def get_age(self):
        pt_study_data = self.load_pt_study_metadata()
        age = pt_study_data["AGE_AT_SLEEP_STUDY_DAYS"].item() / 365
        return age

    def find_target_intervals(self):
        raw_tsv = pd.read_csv(self.tsv_fname, sep="\t")
        raw_tsv["description"] = raw_tsv["description"].apply(lambda x: x.lower())

        raw_tsv.insert(raw_tsv.shape[1], "end", raw_tsv["onset"] + raw_tsv["duration"])
        interval = raw_tsv.apply(
            lambda x: pd.Interval(x["onset"], x["end"], closed="both"),
            axis=1,
        )
        raw_tsv.insert(
            raw_tsv.shape[1],
            "interval",
            interval,
        )

        sleep_stage_tsv = raw_tsv[
            [x in self.sleep_stage_kw for x in raw_tsv["description"]]
        ]

        reject_tsv = raw_tsv[[x in self.reject_kw for x in raw_tsv["description"]]]

        for r_idx, (_, _, _, _, r_interval) in reject_tsv.iterrows():
            overlap_rows = sleep_stage_tsv[
                sleep_stage_tsv["interval"].apply(lambda x: x.overlaps(r_interval))
            ]

            sleep_stage_tsv = sleep_stage_tsv.drop(index=overlap_rows.index, axis=0)
        return sleep_stage_tsv

    def load_epoch_cache(
        self,
        raw_edf: mne.io.BaseRaw,
        target_intervals: pd.DataFrame,
    ) -> EpochCache:
        sfreq = raw_edf.info["sfreq"]
        n_rows = target_intervals.shape[0]

        # Running epochs for signal quality index calculation
        airflow_cache = EpochCache(sampling_freq=sfreq)

        pbar = tqdm(
            target_intervals.iterrows(),
            total=n_rows,
            desc=f"Loading {self.pt_id}...",
        )
        for idx, (_, interval) in enumerate(pbar):
            interval_start_idx = int(interval.onset * sfreq)
            interval_end_idx = int(interval.end * sfreq)

            # Getting airflow signal
            cur_airflow = raw_edf.get_data(
                #  picks=["Resp Airflow"],
                picks=["Resp PTAF"],
                start=interval_start_idx,
                stop=interval_end_idx,
            )

            airflow_cache.add_epoch(
                cur_airflow,
                t_start=interval["onset"],
                t_len=interval["duration"],
                label=interval,
            )
        return airflow_cache

    def process(self):
        raw_edf = mne.io.read_raw_edf(self.edf_fname, verbose=False)
        sfreq = raw_edf.info["sfreq"]

        target_intervals = self.find_target_intervals()
        airflow_cache = self.load_epoch_cache(raw_edf, target_intervals)

        data = []
        dgms_dict = {
            "irr_sublevel": [],
            "airflow_sublevel": [],
            "airflow_rips": [],
        }
        for idx in tqdm(range(len(airflow_cache)), desc=f"Processing {self.pt_id}..."):
            data_arr = airflow_cache.get_epoch_sequence(idx, n_epochs=6)
            sqi = utils.calculate_sqi(data_arr, sfreq)

            # Calculate IRR signal
            irr_signal = self.calc_irr(data_arr.squeeze(), sampling_freq=sfreq)

            # Sublevel set filtration of IRR signal
            sublevel_dgms_irr = self.sublevel_set_filtration(irr_signal)

            # Skipping if we have very few points in filtration
            if np.isnan(sublevel_dgms_irr[0]).sum() > 0:
                continue

            hepc_irr = self.hepc(
                sublevel_dgms_irr[0],
                scale=hepc_scale["sublevel_irr_h0"],
            )
            ap_fapc_irr = self.fapc_closed_form(sublevel_dgms_irr[0])
            L_sublevel_irr = (
                fapc_support["sublevel_irr_h0"][1] - fapc_support["sublevel_irr_h0"][0]
            )
            sp_fapc_irr = self.fapc_closed_form(
                sublevel_dgms_irr[0],
                L=L_sublevel_irr,
            )

            # Applying time-delay embedding and Rips filtration to airflow
            # signal
            rips_dgms_airflow = self.rips_filtration(data_arr, sfreq)

            hepc_rips_airflow_0 = self.hepc(
                rips_dgms_airflow[0],
                scale=hepc_scale["rips_airflow_h0"],
            )
            hepc_rips_airflow_1 = self.hepc(
                rips_dgms_airflow[1],
                scale=hepc_scale["rips_airflow_h1"],
            )
            ap_fapc_rips_airflow_0 = self.fapc_closed_form(rips_dgms_airflow[0])
            ap_fapc_rips_airflow_1 = self.fapc_closed_form(rips_dgms_airflow[1])
            L_rips_airflow_h0 = (
                fapc_support["rips_airflow_h0"][1] - fapc_support["rips_airflow_h0"][0]
            )
            sp_fapc_rips_airflow_0 = self.fapc_closed_form(
                rips_dgms_airflow[0],
                L=L_rips_airflow_h0,
            )
            L_rips_airflow_h1 = (
                fapc_support["rips_airflow_h1"][1] - fapc_support["rips_airflow_h1"][0]
            )
            sp_fapc_rips_airflow_1 = self.fapc_closed_form(
                rips_dgms_airflow[1],
                L=L_rips_airflow_h1,
            )

            # Sublevel set filtration of airflow signal
            sublevel_dgms_airflow = self.sublevel_set_filtration(data_arr)
            hepc_sub_airflow_0 = self.hepc(
                sublevel_dgms_airflow[0],
                scale=hepc_scale["sublevel_airflow_h0"],
            )
            ap_fapc_sub_airflow_0 = self.fapc_closed_form(sublevel_dgms_airflow[0])
            L_sublevel_airflow_h0 = (
                fapc_support["sublevel_airflow_h0"][1]
                - fapc_support["sublevel_airflow_h0"][0]
            )
            sp_fapc_sub_airflow_0 = self.fapc_closed_form(
                sublevel_dgms_airflow[0],
                L=L_sublevel_airflow_h0,
            )

            # Calculating Non-TDA Features
            _, interval_data = airflow_cache[idx]
            epoch_6 = data_arr
            breath_cycle_6_epoch = self.classic_features_breath_cycle(epoch_6, sfreq)

            # Grouping features
            data.append(
                {
                    # HEPC Features
                    "hepc_sub_airflow_0": hepc_sub_airflow_0,
                    "hepc_rips_airflow_0": hepc_rips_airflow_0,
                    "hepc_rips_airflow_1": hepc_rips_airflow_1,
                    "hepc_irr": hepc_irr,
                    # AP-FAPC Features
                    "ap_fapc_sub_airflow_0": ap_fapc_sub_airflow_0,
                    "ap_fapc_rips_airflow_0": ap_fapc_rips_airflow_0,
                    "ap_fapc_rips_airflow_1": ap_fapc_rips_airflow_1,
                    "ap_fapc_irr": ap_fapc_irr,
                    # SP-FAPC Features
                    "sp_fapc_sub_airflow_0": sp_fapc_sub_airflow_0,
                    "sp_fapc_rips_airflow_0": sp_fapc_rips_airflow_0,
                    "sp_fapc_rips_airflow_1": sp_fapc_rips_airflow_1,
                    "sp_fapc_irr": sp_fapc_irr,
                    # Classic Features
                    "breath_cycle_6_epoch": breath_cycle_6_epoch,
                    # Label/SQI
                    "label": interval_data,
                    "sqi": sqi,
                }
            )

            # Diagrams -- placed in .pkl file since they have inconsistence
            # sizes
            dgms_dict["irr_sublevel"].append(sublevel_dgms_irr)
            dgms_dict["airflow_sublevel"].append(sublevel_dgms_airflow)
            dgms_dict["airflow_rips"].append(rips_dgms_airflow)

        label_df = pd.concat([x["label"] for x in data], axis=1).T
        label_df = label_df.drop("interval", axis=1)

        # Converting types
        label_df["onset"] = label_df["onset"].astype(float)
        label_df["duration"] = label_df["duration"].astype(float)
        label_df["description"] = label_df["description"].astype(str)
        label_df["end"] = label_df["end"].astype(str)

        # Saving h5py File
        with h5py.File(self.save_fname_hdf5, "a") as f:
            keys = data[0].keys()
            for k in keys:
                if k != "label":
                    f.create_dataset(k, data=np.stack([x[k] for x in data]))
        label_df.to_hdf(self.save_fname_hdf5, key=f"label")

        #  Saving pkl file with dgms
        with open(self.save_fname_pkl, "wb") as f:
            pickle.dump(dgms_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def calc_irr(self, arr: np.ndarray, sampling_freq: float) -> np.ndarray:
        clean_arr = self.clean_rsp_signal(arr.squeeze(), sampling_rate=sampling_freq)
        _, peaks = nk.rsp_peaks(
            clean_arr,
            sampling_rate=sampling_freq,
            method="biosppy",
        )
        rsp_rate = nk.rsp_rate(
            clean_arr, peaks, method="troughs", sampling_rate=sampling_freq
        )

        T = arr.shape[-1] / sampling_freq
        n_samp = int(T * self.target_resamp_rate)

        # Resampling to 4 Hz
        rsp_resamp = scipy.signal.resample(rsp_rate, num=n_samp)
        return rsp_resamp

    def clean_rsp_signal(self, arr: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Custom function to clean respiratory signal following paper
        implementation"""
        assert len(arr.shape) == 1

        # Linear detrend of signal
        detrend_arr = scipy.signal.detrend(arr, type="linear")

        # 2Hz 5th order butterworth low pass filter
        sos_arr = scipy.signal.butter(
            N=5,
            Wn=2,
            btype="lowpass",
            output="sos",
            fs=sampling_rate,
        )
        clean_arr = scipy.signal.sosfilt(sos_arr, detrend_arr)
        return clean_arr

    def sublevel_set_filtration(self, arr: np.ndarray) -> typing.List[np.ndarray]:
        """Performs sublevel set filtration"""
        arr = arr.squeeze()
        assert len(arr.shape) == 1

        pd = cripser.computePH(arr, maxdim=0)
        dgm0 = pd[:, 1:3]

        # Since we only do 0th diagram, last death in diagram is inf. Changing
        # to inf for easy downstream processing
        dgm0[-1, -1] = np.inf

        return [dgm0]

    def rips_filtration(
        self, arr: np.ndarray, sampling_freq: float
    ) -> typing.List[np.ndarray]:
        n_seconds = 1
        tau = int(sampling_freq * n_seconds)
        embedder = gtda.time_series.SingleTakensEmbedding(
            time_delay=tau,
            dimension=3,
        )

        embedded_signal = embedder.fit_transform(arr.squeeze())
        dgms = ripser(embedded_signal, n_perm=128)["dgms"]
        return dgms

    def persistence_summary(self, dgm: np.ndarray):
        dgm_clean = dgm[~np.isinf(dgm).any(1)]

        dm = tda_utils.midlife_persistence(dgm_clean)
        dl = tda_utils.lifespan_persistence(dgm_clean)

        feat = [
            # Midlife persistence
            tda_utils.mean_pers(dm),
            tda_utils.std_pers(dm),
            tda_utils.skew_pers(dm),
            tda_utils.kurt_pers(dm),
            tda_utils.entr_pers_midlife(dgm_clean),
            # Lifespan persistence
            tda_utils.mean_pers(dl),
            tda_utils.std_pers(dl),
            tda_utils.skew_pers(dl),
            tda_utils.kurt_pers(dl),
            tda_utils.entr_pers_lifespan(dgm_clean),
            # 1-norm of Gaussian persistence curve
            tda_utils.gaussian_persistence_curve(dgm_clean, sigma=1.0),
        ]
        feat = np.asarray(feat)
        return feat

    def calculate_persistence_curve(
        self,
        dgm: np.ndarray,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        dgm_clean = dgm[~np.isinf(dgm).any(1)]
        psi_dgms = tda_utils.psi(dgm_clean)
        x = np.linspace(dgm_clean.min(), dgm_clean.max(), 1000)
        y = np.zeros(x.shape)

        for idx, (b, d) in enumerate(dgm_clean):
            arr_idx = (x >= b) & (x <= d)
            y[arr_idx] += psi_dgms[idx]
        return x, y

    def hepc(
        self,
        dgm: np.ndarray,
        scale: typing.Union[float, None] = None,
    ) -> np.ndarray:
        dgm_clean = dgm[~np.isinf(dgm).any(1)]

        if scale:
            dgm_clean = dgm_clean * scale

        hepc_feat = tda_utils.hepc(dgm_clean)
        return hepc_feat

    def fapc_fft_pc(self, dgm: np.ndarray) -> np.ndarray:
        dgm_clean = dgm[~np.isinf(dgm).any(1)]
        fft_feat = tda_utils.fapc_fft_pc(dgm_clean)
        return fft_feat

    def fapc_closed_form(
        self,
        dgm: np.ndarray,
        L: typing.Union[float, None] = None,
    ) -> np.ndarray:
        dgm_clean = dgm[~np.isinf(dgm).any(1)]
        fft_feat = tda_utils.fapc_closed_form(dgm_clean, L)
        return fft_feat

    def classic_features_breath_cycle(
        self, arr: np.ndarray, sfreq: float
    ) -> np.ndarray:
        arr = arr.squeeze()
        arr = arr - np.mean(arr)
        cleaned = nk.rsp_clean(arr, sampling_rate=sfreq)
        _, peaks_dict = nk.rsp_peaks(cleaned)
        info = nk.rsp_fixpeaks(peaks_dict)
        peaks = cleaned[info["RSP_Peaks"]]

        if len(peaks) > 5:
            # Calculating necessary information for features
            troughs = cleaned[info["RSP_Troughs"]]

            pidx = info["RSP_Peaks"]
            tidx = info["RSP_Troughs"]

            inhalation = []
            exhalation = []
            IER = []
            for idx in range(1, len(peaks)):
                inhalation_signal = cleaned[pidx[idx - 1] : tidx[idx]]
                exhalation_signal = cleaned[tidx[idx - 1] : pidx[idx - 1]]

                inhalation.append(np.sum(np.abs(inhalation_signal)) / sfreq)
                exhalation.append(np.sum(np.abs(exhalation_signal)) / sfreq)

                inhalation_diff = tidx[idx] - pidx[idx - 1]
                exhalation_diff = pidx[idx - 1] - tidx[idx - 1]
                inhalation_exhalation_ratio = inhalation_diff / exhalation_diff
                IER.append(inhalation_exhalation_ratio)

            # Features 1 and 2
            med_amp = np.median(peaks - troughs)
            iqr_amp = utils.iqr(peaks - troughs)

            # Features 3 and 4
            med_width = np.median(IER)
            iqr_width = utils.iqr(IER)

            # Features 5 and 6
            med_peaks = np.median(peaks)
            iqr_peaks = utils.iqr(peaks)

            # Features 7 and 8
            med_troughs = np.median(troughs)
            iqr_troughs = utils.iqr(troughs)

            # Features 9, 10, and 11
            mai = np.median(inhalation)
            mae = np.median(exhalation)
            mai_mae_ratio = mai / mae

            feat_arr = np.asarray(
                [
                    med_amp,
                    iqr_amp,
                    med_width,
                    iqr_width,
                    med_peaks,
                    iqr_peaks,
                    med_troughs,
                    iqr_troughs,
                    mai,
                    mae,
                    mai_mae_ratio,
                ]
            )
        else:
            feat_arr = np.zeros(11)

        return feat_arr

    def classic_features_dtw(
        self,
        arr: np.ndarray,
        template_arr: np.ndarray,
        sfreq: float,
    ) -> np.ndarray:
        window_size = 50
        freq_size = 50

        arr = arr.squeeze()
        rsp = arr - np.mean(arr)
        template_arr = template_arr.squeeze()

        # Decimating array for performance/memory
        rsp = scipy.signal.decimate(arr, 10, zero_phase=True)

        dist_t = []
        dist_f = []
        for idx in range(5, template_arr.shape[0]):
            template_arr_iter = template_arr[idx - 5 : idx]
            template_arr_iter = template_arr_iter.reshape(-1)
            template_arr_iter = scipy.signal.decimate(
                template_arr_iter,
                10,
                zero_phase=True,
            )

            alignment_t = dtw.dtw(
                rsp,
                template_arr_iter,
                keep_internals=True,
                window_type="sakoechiba",
                window_args={"window_size": window_size},
            )
            dist_t.append(alignment_t.normalizedDistance)

            freq_rsp, psd_rsp = scipy.signal.periodogram(
                rsp,
                fs=sfreq // 10,
                window="hann",
                scaling="spectrum",
            )

            freq_template, psd_template = scipy.signal.periodogram(
                template_arr_iter,
                fs=sfreq // 10,
                window="hann",
                scaling="spectrum",
            )
            alignment_f = dtw.dtw(
                psd_rsp,
                psd_template,
                keep_internals=True,
                window_type="sakoechiba",
                window_args={"window_size": freq_size},
            )
            dist_f.append(alignment_f.normalizedDistance)

        dtw_t = min(dist_t)
        dtw_f = min(dist_f)

        # Calculating standard deviation of breathing frequency
        cleaned = nk.rsp_clean(rsp, sampling_rate=sfreq)
        _, peaks_dict = nk.rsp_peaks(cleaned)
        info = nk.rsp_fixpeaks(peaks_dict)
        rsp_rate = nk.rsp_rate(cleaned, peaks_dict, sampling_rate=sfreq)
        rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=sfreq, show=False)
        breathing_freq_std = 1.0 / rrv["RRV_SDBB"].item()

        dtw_feat = np.asarray([dtw_t, dtw_f, breathing_freq_std])
        return dtw_feat

    def classic_features_power(self, arr: np.ndarray, sfreq: float) -> np.ndarray:
        arr = arr.squeeze()
        rsp = arr - np.mean(arr)
        cleaned = nk.rsp_clean(rsp, sampling_rate=sfreq)
        _, peaks_dict = nk.rsp_peaks(cleaned)
        info = nk.rsp_fixpeaks(peaks_dict)
        #  peaks = cleaned[info["RSP_Peaks"]]

        # Calculating RRV Features
        rsp_rate = nk.rsp_rate(cleaned, peaks_dict, sampling_rate=sfreq)
        rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=sfreq, show=False)

        samp_entr = rrv["RRV_SampEn"]
        vlf_logpow = rrv["RRV_VLF"]
        lf_logpow = rrv["RRV_LF"]
        hf_logpow = rrv["RRV_HF"]
        lf_hf = rrv["RRV_LFHF"]

        # Calculating power features
        b, a = scipy.signal.butter(
            3,
            [0.1 * 2 / sfreq, 0.75 * 2 / sfreq],
            btype="bandpass",
        )
        sec = arr.shape[-1] / sfreq
        filt_signal = scipy.signal.filtfilt(b, a, rsp)
        Fsignal = scipy.fft.fft(filt_signal)

        power = np.abs(Fsignal) * np.abs(Fsignal) / arr.shape[-1]

        max_power = np.max(power[int(sec * 0.15) : int(sec * 0.5)])
        Mm = np.argwhere(power[int(sec * 0.15) : int(sec * 0.5)] == max_power)
        max_power_loc = int(sec * 0.15) + Mm[0].item()

        feat = np.asarray(
            [
                samp_entr.item(),
                vlf_logpow.item(),
                lf_logpow.item(),
                hf_logpow.item(),
                lf_hf.item(),
                max_power,
                max_power_loc,
            ]
        )
        return feat

    def classic_features_resp_vol(self, arr: np.ndarray, sfreq: float) -> np.ndarray:
        arr = arr.squeeze()
        arr = arr - np.mean(arr)

        cleaned = nk.rsp_clean(arr, sampling_rate=sfreq)
        df, peaks_dict = nk.rsp_peaks(cleaned)

        info = nk.rsp_fixpeaks(peaks_dict)
        peaks = cleaned[info["RSP_Peaks"]]
        troughs = cleaned[info["RSP_Troughs"]]

        peak_med = np.median(peaks)
        peak_iqr = utils.iqr(peaks)

        feat_15 = peak_med / peak_iqr

        trough_med = np.median(troughs)
        trough_iqr = utils.iqr(troughs)

        feat_16 = trough_med / trough_iqr

        cycle_amp_med = np.median(peaks - troughs)

        vol_breath = []
        vol_inhale = []
        vol_exhale = []
        flow_breath = []
        flow_inhale = []
        flow_exhale = []

        pidx = info["RSP_Peaks"]
        tidx = info["RSP_Troughs"]
        for i in range(len(peaks) - 1):
            vol_breath.append(np.trapz(cleaned[pidx[i] : tidx[i + 1]]))
            vol_inhale.append(np.trapz(cleaned[tidx[i] : pidx[i]]))
            vol_exhale.append(np.trapz(cleaned[pidx[i] : tidx[i + 1]]))

            flow_breath.append(vol_breath[i] / (tidx[i + 1] - tidx[i]))
            flow_inhale.append(vol_inhale[i] / (pidx[i] - tidx[i]))
            flow_exhale.append(vol_exhale[i] / (tidx[i + 1] - pidx[i]))

        feat_18 = np.median(vol_breath)
        feat_19 = np.median(vol_inhale)
        feat_20 = np.median(vol_exhale)
        feat_21 = np.median(flow_breath)
        feat_22 = np.median(flow_inhale)
        feat_23 = np.median(flow_exhale)
        feat_24 = feat_23 / feat_22

        feat_arr = np.asarray(
            [
                feat_15,
                feat_16,
                cycle_amp_med,
                feat_18,
                feat_19,
                feat_20,
                feat_21,
                feat_22,
                feat_23,
                feat_24,
            ]
        )
        return feat_arr

    def classic_features_motion(self, arr: np.ndarray, sfreq: float) -> np.ndarray:
        raise NotImplementedError()


def process_idx(idx: int, data_dir: str, save_dir: str):
    pt_ids = []
    for pt_file in os.listdir(os.path.join(data_dir, "sleep_data")):
        if pt_file.endswith(".edf"):
            pt_ids.append(pt_file.replace(".edf", ""))

    pt_id = pt_ids[idx]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loader = AirflowSignalProcessor(
        pt_id=pt_id,
        data_dir=data_dir,
        save_dir=save_dir,
    )
    subject_ahi = loader.get_ahi()
    subject_age = loader.get_age()

    if (subject_ahi < 1) and (subject_age >= 2) and (subject_age < 18):
        loader.process()
    else:
        if subject_ahi >= 1:
            print("Subject AHI too high!")

        if subject_age < 2:
            print("Subject age too low!")

        if subject_age >= 18:
            print("Subject age too high!")


if __name__ == "__main__":
    Fire(process_idx)
