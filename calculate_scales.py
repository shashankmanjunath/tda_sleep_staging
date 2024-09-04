import random
import os

from fire import Fire
import numpy as np

import tda_utils
import utils


def compute_scale(preproc_dir: str):
    subject_fnames = [x for x in os.listdir(preproc_dir) if x.endswith(".hdf5")]
    subject_fnames = utils.get_unique_subjects(subject_fnames)

    all_paths = np.asarray([os.path.join(preproc_dir, x) for x in subject_fnames])
    all_data, _ = utils.load_data_pkl(all_paths, sqi_thresh=0.25)

    data = []
    for v in all_data.values():
        data += v

    for k, total_hom_deg in tda_utils.n_hom_deg.items():
        for h in range(total_hom_deg):
            # Shuffling
            random.shuffle(data)

            # Using a subset, otherwise we run out of memory
            train_data_arr = [x[k][h] for x in data[:10000]]

            # Removing inf values
            train_clean_data = [x[~np.isinf(x).any(axis=1)] for x in train_data_arr]

            max_arr = [x.max() for x in train_clean_data]
            min_arr = [x.min() for x in train_clean_data]
            max_val = np.max(max_arr)
            min_val = np.min(min_arr)

            max_ul = np.percentile(max_arr, 75)
            min_ll = np.percentile(min_arr, 25)

            scale_val = np.mean([5 / np.abs(x).max() for x in train_clean_data])
            print(f"{k} H_{h}")
            print(f"    Min {min_val}, Max {max_val}, Scale {scale_val}")
            print(f"    Range: {max_ul}, {min_ll}")


if __name__ == "__main__":
    Fire(compute_scale)
