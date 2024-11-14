import typing
import time

from tqdm import tqdm

import torch.utils.data
import sklearn.utils
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
        self.file_handle_cache = {}

        label_arr = []

        for f_idx, fname in enumerate(tqdm(self.fnames)):
            f = h5py.File(fname)

            if "airflow" not in f.keys() or "label" not in f.keys():
                f.close()
                continue

            n_airflow = f["airflow"].shape[0]
            n_time = f["airflow"].shape[1]

            if n_time != 46080:
                f.close()
                continue

            self.file_handle_cache[fname] = f

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

            label_arr += [self.label_map(x["label"][0]) for x in self.idx_label_cache]

            #  if f_idx >= 5:
            #      break

        self.class_weights = sklearn.utils.class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(label_arr),
            y=label_arr,
        )

    def __len__(self) -> int:
        return len(self.idx_label_cache)

    def __getitem__(self, idx: int) -> typing.Tuple[torch.Tensor, int]:
        sample = self.idx_label_cache[idx]

        f = self.file_handle_cache[sample["fname"]]

        data_arr = f["airflow"][sample["airflow_idx"], :]
        data_arr = torch.as_tensor(data_arr)[None, :, None]
        label = self.label_map(sample["label"][0])
        return data_arr, label
