from fire import Fire
import train_xgb


def main(preproc_dir: str, data_dir: str, wandb_project_name: str):
    feature_set_names = [
        "random",
        "classic_6_epoch",
        "hepc",
        "hepc_30",
        "ap_fapc",
        "ap_fapc_30",
        "sp_fapc",
        "sp_fapc_30",
        "ap_fapc_hepc",
        "sp_fapc_hepc",
        "classic_6_epoch_hepc",
        "classic_6_epoch_ap_fapc",
        "classic_6_epoch_ap_fapc_hepc",
        "classic_6_epoch_sp_fapc",
        "classic_6_epoch_sp_fapc_hepc",
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
            wandb_project_name=wandb_project_name,
        )


if __name__ == "__main__":
    Fire(main)
