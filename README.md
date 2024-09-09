# Sleep Staging from Airflow Signals Using Fourier Approximations of Persistence Curves

This repository accompanies the paper "Sleep Staging from Airflow Signals Using
Fourier Approximations of Persistence Curves." This repository is anonymized for
submission purposes.

## Environment Setup

First install Miniconda using the appropriate instructions for your operating
system from [this link](https://docs.anaconda.com/miniconda/miniconda-install/).

Once you have miniconda installed, install the environment from the
environment.yml file provided with the repository by running the following
command:

```
conda env create -f environment.yml
```

You can now activate the environment with the following command:

```
conda activate tda_sleep_staging_env
```

Once you have the conda environment activated, we require manual installation of
a single third-party python package. This is specifically to calculate Hermite
functions Run:

```
git clone https://github.com/rob217/hermite-functions.git
```

Once this has downloaded, run:

```
cd hermite-functions
python setup.py install
```

Once this is complete the environment should be ready for use.

## Dataset and Preprocessing

This research uses the Nationwide Children's Hospital Sleep DataBank (NCHSDB).
To obtain this dataset, go to the
[NCHSDB website](https://sleepdata.org/datasets/nchsdb) and request access. Note
that it may take a day or two to obtain access to the dataset. This dataset
contains 3984 polysomnogram (PSG) studies from primarily pediatric subjects.

Once you have obtained access, download the dataset to an appropriate location
on your machine. The script `dataset.py` contains code for preprocessing each
subject individually. To preprocess the dataset, run the following command:

```
python dataset.py --idx {IDX} --data_dir {DATASET_DIR} --save_dir {SAVE_DIR}
```

where `{IDX}` is an integer between 0 and 3984 corresponding to one of the 3984
PSG studies in the dataset, `{DATASET_DIR}` is the directory where the dataset
is located, and `{SAVE_DIR}` is the directory where preprocessed data is to be
saved. Note that many subjects are skipped due to high apnea-hypopnea index
(AHI), being too young (age < 2) or too old (age > 18). Additionally, many
subjects included in this dataset do not have the requisite nasal cannula
airflow ("Resp PTAF") sensor included in their polysomnogram; these subjects are
also not processed. This script only preprocesses a single subject which the
`{IDX}` parameter refers to, and must be run for each subject included in the
dataset. We recommend parallelizing processing across multiple
machines as processing time can be significant. A single subject which meets
the requisite criteria can take hour or longer to process.

## Running Experiments

This repository uses weights and biases to record information from runs; we
recommend setting up and logging into a weights and biases account at [this
link](https://wandb.ai/site), then connecting your local installation to record
model statistics to your account.

### Model Performance Metrics

To reproduce the model performance metrics included in the paper, run
the script `train_multi_xgb.py` as follows:

```
python train_multi_xgb.py --preproc_dir {SAVE_DIR} --data_dir {DATASET_DIR} --wandb_project_name {WANDB_PROJECT_NAME}
```

`{SAVE_DIR}` should be the directory where the preprocessed data is located.
`{DATASET_DIR}` should be the directory where the original NCHSDB data is
located, and `{WANDB_PROJECT_NAME}` should be the name of the wandb project you
set up. If you wish to run only a single experiment, run the script
`train_xgb.py` as follows:

```
python train_xgb.py --preproc_dir {PREPROC_DIR} --data_dir {DATASET_DIR} --feature_name {FEATURE_NAME} --calc_demos --use_wandb --wandb_project_name {WANDB_PROJECT_NAME}
```

The feature name should be one of the allowed feature names in the following
table.

|          Feature Key           | Corresponding Feature in Results Table |
| :----------------------------: | :------------------------------------: |
|            `random`            |            Random Features             |
|       `classic_6_epoch`        |           Baseline Features            |
|             `hepc`             |                  HEPC                  |
|           `ap_fapc`            |                AP-FAPC                 |
|           `sp_fapc`            |                SP-FAPC                 |
|         `ap_fapc_hepc`         |             AP-FAPC + HEPC             |
|         `sp_fapc_hepc`         |             SP-FAPC + HEPC             |
|     `classic_6_epoch_hepc`     |            Baseline + HEPC             |
|   `classic_6_epoch_ap_fapc`    |           Baseline + AP-FAPC           |
|   `classic_6_epoch_sp_fapc`    |           Baseline + SP-FAPC           |
| `classic_6_epoch_ap_fapc_hepc` |       Baseline + AP-FAPC + HEPC        |
| `classic_6_epoch_sp_fapc_hepc` |       Baseline + SP-FAPC + HEPC        |

`--calc_demos` is a flag which can be omitted. If included, test accuracies on
individual age/sex demographics will be calculated and printed. `--use_wandb`
is a flag which indicates whether to save statistics about the run to weights
and biases.

### Residual Values

To reproduce the residual values and $d_{\text{min}}$/$d_{\text{max}}$ metrics
included in the paper, first run the following command:

```
python calc_residual.py process --idx {IDX} --preproc_dir {PREPROC_DIR} --data_dir {DATASET_DIR} --residual_save_dir {RESIDUAL_SAVE_DIR}
```

`{IDX}` should be an integer between 0 and 3984 which corresponds to one of the
3984 PSG studies in the dataset. This script will only calculate residuals for a
single subject, and must be individually run for each subject index. We
recommend parallelizing this process across multiple machines. Note that,
similarly to the preprocessing script, many of the subjects will be skipped due
to having a high AHI, not having the appropriate sensor in their data, or not
being within the selected age range. `{PREPROC_DIR}` should be the directory
where the preprocessed data is located. `{DATASET_DIR}` should be the directory
where the original NCHSDB dataset is located. `{RESIDUAL_SAVE_DIR}` should be
the location where the intermediate file containing residual values for each fit
type should be located.

Once residuals have been calculated, we include a script to display the results.
Run the following command:

```
python calc_residual.py display --residual_save_dir {RESIDUAL_SAVE_DIR}
```

where `{RESIDUAL_SAVE_DIR}` is the directory where the intermediate residual
value files were saved.

### $d_{\text{min}}$ and $d_{\text{max}}$ Statistics

To calculate the data on $d_{\text{min}}$ and $d_{\text{max}}$ of the
persistence diagrams in the dataset, i.e. the domain of the persistence curves,
run the following command:

```
python calc_residual.py get_domain_stats --data_dir {DATASET_DIR} --preproc_dir {PREPROC_DIR}
```

where `{DATASET_DIR}` is the directory where the original NCHSDB dataset is
located and `{PREPROC_DIR}` is the directory where the preprocessed data is
located. This function will print out $d_{\text{max}}$ and $d_{\text{min}}$
statistics once loading is completed.

## Images in Paper

Generating the images included in the paper requires using a Jupyter notebook.
First, run the following command:

```
jupyter lab
```

Once Jupyterlab starts and a connection has been established through a browser,
and open the `images.ipynb` notebook. Follow the instructions in the notebook to
generate images.

## Scaling Parameter Calculation

To recalculate the scaling parameters used for the HEPC and SP-FAPC
approximations, run the following command:

```
python calculate_scales.py --preproc_dir {PREPROC_DIR}
```

where `{PREPROC_DIR}` is the directory where the preprocessed data is
located. Note that these values may differ slightly from those used in the paper
since the subset selected is random.

## Demographic Counts and Sample Counts from Dataset

To get the demographic age/sex group counts, run the following command:

```
python get_demos_sample_counts.py demographics --data_dir {DATASET_DIR} --preproc_dir {PREPROC_DIR}
```

where `{DATASET_DIR}` is the directory where the original NCHSDB dataset is
located and `{PREPROC_DIR}` is the directory where the preprocessed data is
located.

To get the sample counts from the whole dataset, run the following command:

```
python get_demos_sample_counts.py sample_counts --preproc_dir {PREPROC_DIR}
```

where `{PREPROC_DIR}` is the directory where the preprocessed data is
located.
