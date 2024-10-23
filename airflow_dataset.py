import typing
import time

from tqdm import tqdm

import torch.utils.data
import pandas as pd
import numpy as np
import torch
import h5py

import utils


class AirflowSignalDataset(torch.utils.data.Dataset):
    def __init__(self, fnames: typing.List):
        self.fnames = fnames

        self.idx_label_cache = []
        self.reject_kws = {
            "oxygen desaturation",
            "obstructive apnea",
            "central apnea",
            "mixed apnea",
        }
        self.label_map = utils.wake_nrem_rem_map

        for f_idx, fname in enumerate(tqdm(self.fnames)):
            with h5py.File(fname) as f:
                if "airflow" not in f.keys() or "label" not in f.keys():
                    continue

                n_airflow = f["airflow"].shape[0]

            t1 = time.time()
            label_df = pd.read_hdf(fname, key="label")
            #  print(f"PD Load time: {time.time() - t1}")

            if label_df.shape[0] != n_airflow:
                raise RuntimeError(
                    "Number of airflow signals not equal to number of labels!"
                )

            t1 = time.time()
            # Dropping rows with reject keywords
            label_df["label"] = label_df.label.apply(lambda x: x.split(","))
            label_df["drop_row"] = label_df.label.apply(
                lambda x: len(set(x).intersection(self.reject_kws))
            )
            label_df = label_df[label_df.drop_row == 0]
            label_df["label"] = label_df["label"].apply(
                lambda x: [t for t in x if "sleep stage" in t]
            )
            label_df["airflow_idx"] = np.arange(label_df.shape[0])
            label_df["fname"] = fname
            #  print(f"Drop time: {time.time() - t1}")

            t1 = time.time()
            self.idx_label_cache += label_df.to_dict("records")
            #  print(f"tolist time: {time.time() - t1}")

            if f_idx > 10:
                break

    def __len__(self) -> int:
        return len(self.idx_label_cache)

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, int]:
        sample = self.idx_label_cache[idx]
        with h5py.File(sample["fname"]) as f:
            data_arr = f["airflow"][sample["airflow_idx"], :]
        data_arr = torch.as_tensor(data_arr)[None, None, :]
        label = self.label_map(sample["label"])
        return data_arr, label
