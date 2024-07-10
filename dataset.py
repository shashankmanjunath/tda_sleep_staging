import os

from ripser import ripser
from tqdm import tqdm

import gtda.time_series
import pandas as pd
import numpy as np
import mne

import tda_utils


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
        for idx, interval in tqdm(target_intervals.iterrows(), total=n_rows):
            interval_start_idx = int(interval.onset * sfreq)
            interval_end_idx = int(interval.end * sfreq)
            data_arr = raw_edf.get_data(
                picks=["Resp Airflow"],
                start=interval_start_idx,
                stop=interval_end_idx,
            )
            dgms = self.apply_tda(data_arr, sfreq)
            data.append((dgms, interval.description))
        return

    def apply_tda(self, arr: np.ndarray, sampling_freq: float):
        n_seconds = 1
        tau = int(sampling_freq * n_seconds)
        embedder = gtda.time_series.SingleTakensEmbedding(
            time_delay=tau,
            dimension=3,
        )

        # Scaling array to have maximum value 1
        arr = arr / np.abs(arr).max()
        embedded_signal = embedder.fit_transform(arr.squeeze())
        dgms = ripser(embedded_signal, n_perm=128)["dgms"]

        for dgm in dgms:
            feat = []
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
            hepc_feat = tda_utils.hepc(dgm_clean)
        return -1


if __name__ == "__main__":
    pt_id = "10000_17728"
    loader = AirflowSignalProcessor(pt_id=pt_id, data_dir="/work/thesathlab/nchsdb/")
    loader.process()
