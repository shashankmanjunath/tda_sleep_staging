from fire import Fire
import train_xgb


def main(preproc_dir: str, data_dir: str):
    feature_set_names = [
        "random",
        "classic_6_epoch",
        "hepc",
        "fft",
        "fft_cf",
        "fft_hepc",
        "fft_cf_hepc",
        "classic_6_epoch_hepc",
        "classic_6_epoch_fft",
        "classic_6_epoch_fft_hepc",
        "classic_6_epoch_fft_cf",
        "classic_6_epoch_fft_cf_hepc",
    ]
    n_feature_set = len(feature_set_names)

    for idx, feature_name in enumerate(feature_set_names):
        print(f"[{idx+1}/{n_feature_set}]: Running {feature_name}")
        train_xgb.train(
            preproc_dir,
            data_dir,
            feature_name=feature_name,
            calc_demos=False,
            use_wandb=True,
        )


if __name__ == "__main__":
    Fire(main)
