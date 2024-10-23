import os

#  from tqdm import tqdm

#  import torch.nn as nn
#  import torch

#  import sklearn.model_selection
#  import ml_collections
#  import numpy as np

#  from airflow_dataset import AirflowSignalDataset
#  import models.vit
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
    def __init__(self, train_dataset, test_dataset, model):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = 1024

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            self.batch_size,
            pin_memory=True,
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            self.batch_size,
            pin_memory=True,
        )

        self.model = model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for train_data, train_label in self.train_loader:
            train_data = train_data.to(self.device)
            train_label = train_label.to(self.device)

            outputs = self.model(train_data)
            loss = self.criterion(outputs, train_label)
            loss.backward()
            self.optim.step()
            pass
        pass


def train(
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
    config = transformer_config()
    pbar = tqdm(kf.split(all_paths, strat_label), total=n_splits)

    for idx, (train_idx, test_idx) in enumerate(pbar):
        train_fnames = all_paths[train_idx]
        test_fnames = all_paths[test_idx]

        train_dataset = AirflowSignalDataset(train_fnames)
        test_dataset = AirflowSignalDataset(test_fnames)

        model = models.vit.VisionTransformer(
            config,
            img_size=(1, 46080),
            num_classes=3,
        )

        trainer = ModelTrainer(train_dataset, test_dataset, model)
        trainer.train()
        pass
    pass


if __name__ == "__main__":
    print("Starting...")
    train(
        preproc_dir="/work/thesathlab/manjunath.sh/tda_sleep_staging_airflow",
        data_dir="/work/thesathlab/nchsdb/",
    )
