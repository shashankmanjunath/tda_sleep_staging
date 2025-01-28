import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np


class FilterFunc:
    def __init__(self, filter_name: str, data_dir: str):
        self.filter_name = filter_name
        self.data_dir = data_dir

        self.filter_func_map = {
            "low_ahi_odi": self.low_ahi_odi_filter,
            "low_ahi": self.low_ahi_filter,
            "mid_ahi": self.mid_ahi_filter,
            "high_ahi": self.high_ahi_filter,
            "all": self.all_filter,
        }
        if self.filter_name not in self.filter_func_map:
            raise RuntimeError("Filter type not recognized!")
        self.filter_func = self.filter_func_map[self.filter_name]

    def __call__(
        self,
        data: dict[str, np.ndarray],
        label: dict[str, pd.DataFrame],
        all_paths: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], np.ndarray]:
        new_paths = []
        for p in tqdm(all_paths, desc="Filtering...."):
            if p in label.keys():
                is_valid_data = self.filter_func(label_dict=label, subject_path=p)
                if not is_valid_data:
                    data.pop(p)
                    label.pop(p)
                else:
                    new_paths.append(p)

        new_paths = np.asarray(new_paths)
        return data, label, new_paths

    def get_ahi(self, subject_id: str) -> float:
        study_data = pd.read_csv(
            os.path.join(
                self.data_dir,
                "health_data",
                "SLEEP_STUDY.csv",
            )
        )

        study_data["STUDY_PAT_ID"] = study_data["STUDY_PAT_ID"].astype(str)
        study_data["SLEEP_STUDY_ID"] = study_data["SLEEP_STUDY_ID"].astype(str)
        study_data["PT_ID"] = study_data[["STUDY_PAT_ID", "SLEEP_STUDY_ID"]].agg(
            "_".join,
            axis=1,
        )
        pt_study_data = study_data[study_data["PT_ID"] == subject_id]

        if pt_study_data.shape[0] > 1:
            raise RuntimeError("Too many sleep studies found!")

        study_duration = pt_study_data["SLEEP_STUDY_DURATION_DATETIME"].item().strip()

        tm = time.strptime(study_duration, "%H:%M:%S")
        study_dur_h = tm.tm_hour + (tm.tm_min / 60) + (tm.tm_sec / 3600)

        tsv_fname = os.path.join(self.data_dir, "sleep_data", f"{subject_id}.tsv")
        if not os.path.exists(tsv_fname):
            raise RuntimeError("TSV file for subject not found!")

        raw_tsv = pd.read_csv(tsv_fname, sep="\t")
        raw_tsv["description"] = raw_tsv["description"].apply(lambda x: x.lower())
        event_list = raw_tsv["description"].to_list()
        events = [x for x in event_list if "apnea" in x or "hypopnea" in x]
        ahi = len(events) / study_dur_h
        return ahi

    def low_ahi_odi_filter(
        self,
        **kwargs,
    ) -> bool:
        # Getting subject data
        label_dict = kwargs["label_dict"]
        subject_path = kwargs["subject_path"]
        df = label_dict[subject_path]

        # Looking for low ahi < 1 and low oxygen desaturation index
        sleep_duration_h = (df["onset"].max() - df["onset"].min()) / 3600.0
        events_df = df[[len(x) > 0 for x in df["events"].tolist()]]

        events_list = events_df["events"].tolist()
        #  apnea_df = events_df[["apnea" in "\t".join(x) for x in events_list]]
        #  hypopnea_df = events_df[["hypopnea" in "\t".join(x) for x in events_list]]
        oxygen_desat_df = events_df[
            ["oxygen desaturation" in "\t".join(x) for x in events_list]
        ]

        #  ahi = (apnea_df.shape[0] + hypopnea_df.shape[0]) / sleep_duration_h
        subject_id = subject_path.split("/")[-1].replace(".hdf5", "")
        ahi = self.get_ahi(subject_id)
        odi = oxygen_desat_df.shape[0] / sleep_duration_h

        if ahi >= 1 or odi >= 1:
            return False
        return True

    def mid_ahi_odi_filter(self, **kwargs) -> bool:
        pass

    def high_ahi_odi_filter(self, **kwargs) -> bool:
        pass

    def low_ahi_filter(self, **kwargs) -> bool:
        # Looking for low ahi < 1
        subject_path = kwargs["subject_path"]
        subject_id = subject_path.split("/")[-1].replace(".hdf5", "")
        ahi = self.get_ahi(subject_id)

        if ahi >= 1:
            return False
        return True

    def mid_ahi_filter(self, **kwargs) -> bool:
        subject_path = kwargs["subject_path"]
        subject_id = subject_path.split("/")[-1].replace(".hdf5", "")
        ahi = self.get_ahi(subject_id)

        if 1 <= ahi <= 5:
            return True
        return False

    def high_ahi_filter(self, **kwargs) -> bool:
        subject_path = kwargs["subject_path"]
        subject_id = subject_path.split("/")[-1].replace(".hdf5", "")
        ahi = self.get_ahi(subject_id)

        if 5 <= ahi:
            return True
        return False

    def low_odi_filter(self, **kwargs) -> bool:
        pass

    def mid_odi_filter(self, **kwargs) -> bool:
        pass

    def high_odi_filter(self, **kwargs) -> bool:
        pass

    def all_filter(self, **kwargs) -> bool:
        return True
