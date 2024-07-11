import typing
import os

from ripser import ripser
from tqdm import tqdm

import gtda.time_series
import neurokit2 as nk
import scipy.sparse
import pandas as pd
import numpy as np
import mne

import tda_utils


class EpochItem:
    def __init__(self, epoch, t_start, t_len):
        self.epoch = epoch
        self.t_start = t_start
        self.t_len = t_len

        # Slack time
        self.t_slack = 1.0

    def overlaps(self, other) -> bool:
        # Allow t_slack difference between old end time and new start time
        end_time = self.t_start + self.t_len
        slack_time = end_time + self.t_slack

        if slack_time < other.t_start:
            return False
        else:
            return True


class EpochCache:
    def __init__(self, n_epochs: int, sampling_freq: float):
        self.n_epochs = n_epochs
        self.sampling_freq = sampling_freq
        self.freq_spread = 4

        self.bp_lo_freq = 0.1
        self.bp_hi_freq = 0.75

        self.epoch_cache = []
        self.sos_butter_bandpass = scipy.signal.butter(
            N=3,
            Wn=[self.bp_lo_freq, self.bp_hi_freq],
            btype="bandpass",
            output="sos",
            fs=self.sampling_freq,
        )

    def add_epoch(self, epoch: np.ndarray, t_start: float, t_len: float) -> None:
        if len(self.epoch_cache) == 0:
            # No epochs in cache; add to cache
            self.epoch_cache.append(EpochItem(epoch, t_start, t_len))
        else:
            cur_epoch = EpochItem(epoch, t_start, t_len)
            last_epoch = self.epoch_cache[-1]

            if last_epoch.overlaps(cur_epoch):
                # New epoch is consecutive to previous epoch
                if len(self.epoch_cache) == self.n_epochs:
                    # Pop out first item if adding current will make more than
                    # the maximum number of epochs
                    self.epoch_cache = self.epoch_cache[1:]
                elif len(self.epoch_cache) > self.n_epochs:
                    raise RuntimeError("SQI Epoch Cache too large!")
                # Add epoch to end of cache
                self.epoch_cache.append(cur_epoch)
            else:
                # New epoch is not consecutive to previous epoch, so we clear
                # cache and restart
                self.epoch_cache = []
                self.epoch_cache.append(cur_epoch)

    def get_epoch_sequence(self) -> np.ndarray:
        if len(self.epoch_cache) < 5:
            raise RuntimeError("Insufficient data in epoch cache!")
        epoch_data = np.concatenate([x.epoch for x in self.epoch_cache], axis=-1)
        return epoch_data

    def get_sqi(self) -> float:
        if len(self.epoch_cache) < 5:
            return -1.0
        epoch_data = np.concatenate([x.epoch for x in self.epoch_cache], axis=-1)

        # Bandpass signal
        bandpass_signal = scipy.signal.sosfilt(self.sos_butter_bandpass, epoch_data)

        # Fourier transform
        fft_arr = np.abs(scipy.fft.fft(bandpass_signal)).squeeze()
        fft_arr = np.power(fft_arr, 2)
        ts = 1.0 / self.sampling_freq
        freqs = scipy.fft.fftfreq(fft_arr.shape[-1], d=ts)

        # Finding peak in spectrum
        valid_freqs = (freqs >= self.bp_lo_freq) & (freqs <= self.bp_hi_freq)
        fft_arr = fft_arr[valid_freqs]
        freqs = freqs[valid_freqs]
        max_freq_idx = fft_arr.argmax()

        # Calculating SQI
        signal_power = fft_arr.sum()
        low_idx = max_freq_idx - self.freq_spread
        hi_idx = max_freq_idx + self.freq_spread
        maxpow_band_power = fft_arr[low_idx:hi_idx].sum()
        sqi = maxpow_band_power / signal_power
        return sqi


class AirflowSignalProcessor:
    def __init__(self, pt_id: str, data_dir: str):
        self.pt_id = pt_id
        self.data_dir = data_dir
        self.edf_fname = os.path.join(self.data_dir, "sleep_data", f"{self.pt_id}.edf")
        self.tsv_fname = os.path.join(self.data_dir, "sleep_data", f"{self.pt_id}.tsv")

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

    def process(self):
        raw_edf = mne.io.read_raw_edf(self.edf_fname)
        sfreq = raw_edf.info["sfreq"]

        target_intervals = self.find_target_intervals()
        n_rows = target_intervals.shape[0]

        data = []

        # Running epochs for signal quality index calculation
        epoch_cache = EpochCache(self.n_epochs_sqi, sampling_freq=sfreq)

        pbar = tqdm(target_intervals.iterrows(), total=n_rows)
        for idx, (_, interval) in enumerate(pbar):
            interval_start_idx = int(interval.onset * sfreq)
            interval_end_idx = int(interval.end * sfreq)

            # Getting airflow signal
            cur_epoch = raw_edf.get_data(
                picks=["Resp Airflow"],
                start=interval_start_idx,
                stop=interval_end_idx,
            )

            epoch_cache.add_epoch(
                cur_epoch,
                t_start=interval["onset"],
                t_len=interval["duration"],
            )

            sqi = epoch_cache.get_sqi()

            if sqi < 0.25:
                print(f"Skipping {idx}, SQI: {sqi}")
                # SQI is too low, reject epoch
                continue

            data_arr = epoch_cache.get_epoch_sequence()

            # Calculate IRR signal
            irr_signal = self.get_irr(data_arr, sfreq)

            # Sublevel set filtration of IRR signal
            sublevel_dgms_irr = self.sublevel_set_filtration(irr_signal)

            # Skipping if we have very few points in filtration
            if np.isnan(sublevel_dgms_irr[0]).sum() > 0:
                continue

            ps_irr = self.persistence_summary(sublevel_dgms_irr[0])
            hepc_irr = self.hepc(sublevel_dgms_irr[0])

            # Applying time-delay embedding and Rips filtration to airflow
            # signal
            rips_dgms_airflow = self.rips_filtration(data_arr, sfreq)
            hepc_rips_airflow_0 = self.hepc(rips_dgms_airflow[0])
            ps_rips_airflow_1 = self.persistence_summary(rips_dgms_airflow[1])

            # Sublevel set filtration of airflow signal
            sublevel_dgms_airflow = self.sublevel_set_filtration(data_arr)
            hepc_sub_airflow_0 = self.hepc(sublevel_dgms_airflow[0])
            ps_sub_airflow_0 = self.persistence_summary(sublevel_dgms_airflow[0])

            feat_arr = [
                ps_sub_airflow_0,
                hepc_sub_airflow_0,
                hepc_rips_airflow_0,
                ps_rips_airflow_1,
                ps_irr,
                hepc_irr,
            ]
            feat = np.concatenate(feat_arr, axis=0)
            data.append((feat, interval))
        return

    def get_irr(self, arr: np.ndarray, sampling_freq: float) -> np.ndarray:
        #  clean_arr = nk.rsp_clean(arr.squeeze(), sampling_rate=sampling_freq)
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

    def signal_quality_check(self, arr: np.ndarray, sampling_freq: float) -> np.ndarray:
        pass

    def sublevel_set_filtration(self, arr: np.ndarray) -> typing.List[np.ndarray]:
        """Performs sublevel set filtration based on ripser.
        Code modified from:
            https://ripser.scikit-tda.org/en/latest/notebooks/Lower%20Star%20Time%20Series.html
        """
        arr = arr.squeeze()
        assert len(arr.shape) == 1

        # Add edges between adjacent points in the time series, with the
        # "distance" along the edge equal to the max value of the points it
        # connects
        N = arr.shape[0]
        I = np.arange(N - 1)
        J = np.arange(1, N)
        V = np.maximum(arr[0:-1], arr[1::])

        # Add vertex birth times along the diagonal of the distance matrix
        I = np.concatenate((I, np.arange(N)))
        J = np.concatenate((J, np.arange(N)))
        V = np.concatenate((V, arr))

        # Create the sparse distance matrix
        D = scipy.sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
        dgm0 = ripser(D, maxdim=0, distance_matrix=True)["dgms"][0]
        #  dgm0 = dgm0[dgm0[:, 1] - dgm0[:, 0] > 1e-3, :]
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

        # Scaling array to have maximum value 1
        #  arr = arr / np.abs(arr).max()

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

    def hepc(self, dgm: np.ndarray) -> np.ndarray:
        dgm_clean = dgm[~np.isinf(dgm).any(1)]
        hepc_feat = tda_utils.hepc(dgm_clean)
        return hepc_feat


def process_idx(idx):
    data_dir = "/work/thesathlab/nchsdb/"

    pt_ids = []
    for pt_file in os.listdir(os.path.join(data_dir, "sleep_data")):
        if pt_file.endswith(".edf"):
            pt_ids.append(pt_file.replace(".edf", ""))

    pt_id = pt_ids[idx]
    loader = AirflowSignalProcessor(pt_id=pt_id, data_dir=data_dir)
    loader.process()


if __name__ == "__main__":
    process_idx(0)
