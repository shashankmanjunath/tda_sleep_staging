import typing
import time
import os

from fire import Fire
from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as nn
import torch

import sklearn.model_selection
import ml_collections
import numpy as np
import wandb

from airflow_dataset import AirflowSignalDataset
import models.vit
import utils


def transformer_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (128, 1)})
    config.hidden_size_1 = 128
    config.hidden_size_2 = 64
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 128
    config.transformer.num_heads = 4
    config.transformer.num_layers = 8
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


class ModelTrainer:
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        model: nn.Module,
        split_idx: int,
        use_wandb: bool = False,
        continue_run_path: typing.Optional[str] = None,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split_idx = split_idx
        self.use_wandb = use_wandb
        self.continue_run_path = continue_run_path
        self.batch_size = 512
        self.start_epoch = 0
        self.epochs = 1000
        self.num_workers = 8
        self.model_save_dir = os.path.join(
            "./",
            "best_models",
            f"split_{self.split_idx}",
        )
        self.model_save_fname = os.path.join(
            self.model_save_dir,
            f"best_model_{self.split_idx}.pth",
        )

        self.model_train_checkpoint = os.path.join(
            self.model_save_dir, f"model_checkpoint_{self.split_idx}.pth"
        )

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            #  prefetch_factor=8,
            shuffle=True,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            #  prefetch_factor=8,
            shuffle=True,
            pin_memory=True,
        )

        self.n_test_steps = 5000
        self.best_test_ba = -1.0

        self.model = model.to(self.device)
        self.lr = 1e-4
        t1 = time.time()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print(f"Optimizer initialization time: {time.time() - t1}")

        self.train_weights = torch.as_tensor(self.train_dataset.class_weights)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.train_weights.to(self.device).float(),
        )

        if self.use_wandb:
            if not self.continue_run_path:
                self.run = wandb.init(
                    project="tda_airflow_sleep_staging",
                    name=f"fold_{self.split_idx}",
                    config={
                        "batch_size": self.batch_size,
                        "learning_rate": self.lr,
                        "epochs": self.epochs,
                    },
                )
            else:
                data_fname = wandb.restore(
                    self.model_train_checkpoint,
                    run_path=self.continue_run_path,
                )
                weights_data = torch.load(data_fname.name, weights_only=False)
                self.model.load_state_dict(weights_data["model_state_dict"])
                self.optim.load_state_dict(weights_data["optimizer"])
                self.start_epoch = weights_data["epoch"]
                self.best_test_ba = weights_data["best_test_acc"]

                run_id = self.continue_run_path.split("/")[-1]
                self.run = wandb.init(
                    project="tda_airflow_sleep_staging",
                    id=run_id,
                    resume="must",
                )

    def train(self):
        steps = 0
        for epoch in range(self.start_epoch, self.epochs):
            self.model = self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Training [{epoch+1}/{self.epochs}]")
            for idx, (train_data, train_label) in enumerate(pbar):
                if steps % self.n_test_steps == 0 and steps > 0:
                    test_ba = self.test()
                    pbar.set_postfix({"test_acc": f"{test_ba:.3f}"})

                    if test_ba > self.best_test_ba:
                        print("Saving new best model...")
                        torch.save(
                            {
                                "model_state_dict": self.model.state_dict(),
                                "epoch": epoch,
                                "optimizer": self.optim.state_dict(),
                                "best_test_acc": test_ba,
                            },
                            self.model_save_fname,
                        )

                        if self.use_wandb:
                            wandb.save(self.model_save_fname)

                        self.best_test_ba = test_ba

                train_data = train_data.to(self.device).float()
                train_label = train_label.to(self.device)

                self.optim.zero_grad()

                logits, attn_weights = self.model(train_data)

                loss = self.criterion(logits, train_label)
                loss.backward()
                self.optim.step()

                preds = F.softmax(logits, dim=-1).argmax(-1)

                train_acc = (preds == train_label).sum() / preds.size()[0]

                if self.use_wandb:
                    self.run.log(
                        {
                            "train_loss": loss.item(),
                            "train_acc": train_acc.item(),
                        }
                    )

                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "epoch": epoch,
                            "optimizer": self.optim.state_dict(),
                            "best_test_acc": self.best_test_ba,
                        },
                        self.model_train_checkpoint,
                    )

                    if self.use_wandb:
                        wandb.save(self.model_train_checkpoint)

                steps += 1

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

            loss = np.sum(losses) / all_labels.size

            if self.use_wandb:
                self.run.log(
                    {
                        "test_loss": loss,
                        "test_ba": test_ba,
                        "test_f1": test_f1,
                        "test_precision": test_precision,
                        "test_recall": test_recall,
                    }
                )
        return test_ba


def train(
    preproc_dir: str,
    data_dir: str,
    target_split: int,
    #  calc_demos: bool = False,
    use_wandb: bool = False,
    continue_run_path: typing.Optional[str] = None,
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
    config = transformer_config()

    for split_idx, (train_idx, test_idx) in enumerate(kf.split(all_paths, strat_label)):
        if split_idx != target_split:
            continue

        print(f"Running Split {split_idx}")

        train_fnames = all_paths[train_idx][:10]
        test_fnames = all_paths[test_idx][:10]

        train_dataset = AirflowSignalDataset(train_fnames)
        test_dataset = AirflowSignalDataset(test_fnames)

        model = models.vit.VisionTransformer(
            config,
            img_size=(1, 46080),
            num_classes=3,
        )
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters: {params}")

        trainer = ModelTrainer(
            train_dataset,
            test_dataset,
            model,
            split_idx=split_idx,
            use_wandb=use_wandb,
            continue_run_path=continue_run_path,
        )
        trainer.train()


if __name__ == "__main__":
    print("Starting...")
    Fire(train)
