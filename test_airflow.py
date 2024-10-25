import os

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
        self.batch_size = 128
        self.num_workers = 16

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.criterion = nn.CrossEntropyLoss()

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
            losses = []

            pbar = tqdm(self.test_loader, desc="Testing...")
            for idx, (data, labels) in enumerate(pbar):
                all_labels.append(labels.numpy())

                data = data.to(self.device).float()
                labels = labels.to(self.device).long()

                logits, attn_weights = self.model(data)
                loss = self.criterion(logits, labels)
                losses.append(loss.item() * labels.shape[0])

                preds = F.softmax(logits, dim=-1).argmax(dim=-1)

                all_preds.append(preds.cpu().numpy())

            all_preds = np.concatenate(all_preds).astype(int)
            all_labels = np.concatenate(all_labels).astype(int)
            #  test_acc = (all_preds == all_labels).sum() / all_labels.size

            test_ba = sklearn.metrics.balanced_accuracy_score(all_labels, all_preds)
            test_f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="macro")
            test_precision = sklearn.metrics.precision_score(
                all_labels,
                all_preds,
                average="macro",
            )
            test_recall = sklearn.metrics.recall_score(
                all_labels,
                all_preds,
                average="macro",
            )

            test_loss = np.sum(losses) / all_labels.size

            print("Test Results:")
            print(f"Test loss: {test_loss}")
            print(f"Test Balanced Accuracy: {test_ba}")
            print(f"Test F1 Score: {test_f1}")
            print(f"Test Precision: {test_precision}")
            print(f"Test Recall: {test_recall}")


def test(
    preproc_dir: str,
    data_dir: str,
    calc_demos: bool = False,
    use_wandb: bool = False,
    wandb_project_name: str = "",
):
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
    selected_split = 0

    for idx, (train_idx, test_idx) in enumerate(kf.split(all_paths, strat_label)):
        if idx != selected_split:
            continue

        print(f"Testing Split {idx}")

        test_fnames = all_paths[test_idx]
        test_dataset = AirflowSignalDataset(test_fnames)

        model = models.vit.VisionTransformer(
            config,
            img_size=(1, 46080),
            num_classes=3,
        )

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters: {params}")

        tester = ModelTester(test_dataset, model)

        run = wandb.init(project="tda_airflow_sleep_staging", job_type="evaluate_model")
        downloaded_model_path = run.use_model(name="run-pnqzhlxl-best_model.pth:latest")
        best_model_state_dict = torch.load(downloaded_model_path)
        tester.model.load_state_dict(best_model_state_dict)
        tester.test()


if __name__ == "__main__":
    print("Starting...")
    test(
        preproc_dir="/work/thesathlab/manjunath.sh/tda_sleep_staging_airflow",
        data_dir="/work/thesathlab/nchsdb/",
    )
