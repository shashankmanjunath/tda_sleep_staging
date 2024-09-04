import collections
import os

from fire import Fire
import numpy as np

import utils


def get_demographics(data_dir: str, preproc_dir: str):
    subject_fnames = [x for x in os.listdir(preproc_dir) if x.endswith(".hdf5")]
    subject_fnames = utils.get_unique_subjects(subject_fnames)
    all_paths = np.asarray([os.path.join(preproc_dir, x) for x in subject_fnames])
    all_demos = utils.get_demographics(
        all_paths,
        data_dir=data_dir,
    )

    output_dict = collections.defaultdict(list)
    for k, v in all_demos.items():
        output_dict[v].append(k)

    keys = sorted(output_dict.keys(), key=lambda x: int(x.split("_")[0]))
    for k in keys:
        print(f"{k} & {len(output_dict[k])} & \\\\")
        print("\\hline")


def get_sample_counts(preproc_dir: str):
    subject_fnames = [x for x in os.listdir(preproc_dir) if x.endswith(".hdf5")]
    subject_fnames = utils.get_unique_subjects(subject_fnames)
    all_paths = np.asarray([os.path.join(preproc_dir, x) for x in subject_fnames])
    _, all_label = utils.load_data(all_paths, "random", sqi_thresh=0.25)

    output_counts = {
        "sleep stage w": 0,
        "sleep stage nrem": 0,
        "sleep stage r": 0,
    }
    total = 0
    for v in all_label.values():
        label_name, label_count = np.unique(v, return_counts=True)

        for ln, lc in zip(label_name, label_count):
            total += lc
            if ln in ["sleep stage n1", "sleep stage n2", "sleep stage n3"]:
                output_counts["sleep stage nrem"] += lc
            else:
                output_counts[ln] += lc

    for k, v in output_counts.items():
        v_pct = (v / total) * 100
        print(f"{k}: {v} Samples, {v_pct:.3f} % of Dataset")


if __name__ == "__main__":
    Fire(
        {
            "demographics": get_demographics,
            "sample_counts": get_sample_counts,
        }
    )
