import pathlib
import pickle
import typing
import ntpath
import os

from fire import Fire
from tqdm import tqdm

import sklearn.model_selection
import numpy as np
import wandb

import torch.nn.functional as F
import torch.nn as nn
import torch

from airflow_dataset import AirflowSignalDataset
import train_airflow
import models.vit
import utils


class ModelTester:
    def __init__(self, test_dataset, model):
        self.test_dataset = test_dataset
        self.batch_size = 512
        self.num_workers = 8

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.model_save_dir = os.path.join("./", "best_models")
        self.model_save_fname = os.path.join(self.model_save_dir, "best_model.pth")

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.model = model.to(self.device)

    def test(self):
        with torch.no_grad():
            self.model = self.model.eval()
            all_preds = []
            all_labels = []

            pbar = tqdm(self.test_loader)
            for idx, (data, labels) in enumerate(pbar):
                all_labels.append(labels.numpy())

                data = data.to(self.device).float()
                labels = labels.to(self.device).long()

                logits, _ = self.model(data)
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)

                all_preds.append(preds.cpu().numpy())

            all_preds = np.concatenate(all_preds).astype(int)
            all_labels = np.concatenate(all_labels).astype(int)

            ba = sklearn.metrics.balanced_accuracy_score(all_labels, all_preds)
            f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="macro")
            precision = sklearn.metrics.precision_score(
                all_labels,
                all_preds,
                average="macro",
            )
            recall = sklearn.metrics.recall_score(
                all_labels,
                all_preds,
                average="macro",
            )
            cohen_kappa = sklearn.metrics.cohen_kappa_score(all_labels, all_preds)
            cmat = sklearn.metrics.confusion_matrix(all_labels, all_preds)

            output_dict = {
                "ba": ba,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "cohen_kappa": cohen_kappa,
                "cmat": cmat,
            }

            #  print("Test Results:")
            #  print(f"Test Balanced Accuracy: {test_ba}")
            #  print(f"Test F1 Score: {test_f1}")
            #  print(f"Test Precision: {test_precision}")
            #  print(f"Test Recall: {test_recall}")
        return output_dict


def test(
    preproc_dir: str,
    data_dir: str,
    wandb_username: str,
    wandb_project_name: str,
    fold_run_ids: typing.List[str],
    save_fname: str,
):
    parent_dir, _ = ntpath.split(save_fname)
    pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)

    subject_fnames = [x for x in os.listdir(preproc_dir) if x.endswith(".hdf5")]
    subject_fnames = utils.get_unique_subjects(subject_fnames)

    all_paths = np.asarray([os.path.join(preproc_dir, x) for x in subject_fnames])
    print("Loading demographics...")
    all_demos = utils.get_demographics(
        all_paths,
        data_dir,
    )
    print("Demographics loaded!")

    n_splits = 5
    kf_seed = 2024
    kf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits,
        random_state=kf_seed,
        shuffle=True,
    )
    strat_label = [all_demos[x] for x in all_paths]
    config = train_airflow.transformer_config()

    output_dict = {
        "train_ba": [],
        "test_ba": [],
        "cohen_kappa": [],
        "test_cmat": [],
    }

    for split_idx, (train_idx, test_idx) in enumerate(kf.split(all_paths, strat_label)):
        print(f"Testing Split {split_idx}")
        fold_id = fold_run_ids[split_idx]
        best_model_fname = os.path.join(
            "./", "best_models", f"split_{split_idx}", f"best_model_{split_idx}.pth"
        )
        data_fname = wandb.restore(
            best_model_fname,
            run_path=f"{wandb_username}/{wandb_project_name}/{fold_id}",
        )
        weights_data = torch.load(data_fname.name, weights_only=False)

        model = models.vit.VisionTransformer(
            config,
            img_size=(1, 46080),
            num_classes=3,
        )
        model.load_state_dict(weights_data["model_state_dict"])

        train_fnames = all_paths[train_idx]
        test_fnames = all_paths[test_idx]

        train_dataset = AirflowSignalDataset(train_fnames)
        test_dataset = AirflowSignalDataset(test_fnames)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters: {params}")

        trainset_tester = ModelTester(train_dataset, model)
        testset_tester = ModelTester(test_dataset, model)

        print("Testing Train Set")
        train_output_dict = trainset_tester.test()
        print("Testing Test Set")
        test_output_dict = testset_tester.test()

        output_dict["train_ba"].append(train_output_dict["ba"])
        output_dict["test_ba"].append(test_output_dict["ba"])
        output_dict["cohen_kappa"].append(test_output_dict["cohen_kappa"])
        output_dict["test_cmat"].append(test_output_dict["cmat"])

    with open(save_fname, "wb") as f:
        pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    print("Starting...")
    Fire(test)
    #  test(
    #      preproc_dir="/work/thesathlab/manjunath.sh/tda_sleep_staging_airflow",
    #      data_dir="/work/thesathlab/nchsdb/",
    #  )
