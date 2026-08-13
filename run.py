"""Public entry point for the released CFX-GNN implementation."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

PAPER_METHOD = {
    "cf_ref_mode": "local_quantile",
    "stage1_map_loss": "none",
    "cf_stage1_regularizer": "orth",
    "final_pred_source": "zy",
    "phase2_mask_role": "bias_clean",
    "phase3_update_mode": "clean_head_only",
    "reduce_loss_type": "repr",
    "lambda_pred_exp": 0.0,
    "lambda_h_cf_inv": 0.0,
    "lambda_local_keep": 0.0,
    "calib_epochs": 0,
}


def required_data_files(dataset):
    folder = ROOT / "cfx_gnn" / "dataset" / ("pokec" if dataset.startswith("pokec_") else dataset)
    return (folder / f"{dataset}_graph.bin", folder / f"{dataset}_index.bin")


def validate_data(dataset):
    if dataset == "toy" and not all(path.is_file() for path in required_data_files(dataset)):
        from cfx_gnn.toy_data import generate_toy_dataset
        generate_toy_dataset(ROOT / "cfx_gnn" / "dataset" / "toy")
    missing = [path for path in required_data_files(dataset) if not path.is_file()]
    if missing:
        relative = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(
            f"Missing prepared {dataset} dataset cache:\n{relative}\n"
            "See cfx_gnn/dataset/README.md for the expected layout."
        )


def build_command(dataset, gpu, smoke_test=False):
    config = yaml.safe_load((ROOT / "configs" / f"{dataset}.yaml").read_text(encoding="utf-8"))
    forbidden = sorted(PAPER_METHOD.keys() & config.keys())
    if forbidden:
        raise ValueError(
            "Paper method fields must not be selectable in dataset configs: "
            + ", ".join(forbidden)
        )
    command = [sys.executable, str(ROOT / "cfx_gnn" / "main.py"),
        "--method", "cf_risk_self_explainer", "--gpu", str(gpu),
        "--ckpt_dir", str((ROOT / "checkpoints" / dataset).resolve())]
    for key, value in config.items():
        if isinstance(value, bool):
            if value: command.append(f"--{key}")
        elif value is not None: command.extend((f"--{key}", str(value)))
    if smoke_test:
        command.append("--smoke_test")
        for key in ("pre_epochs", "epochs", "cf_warm_epochs", "cf_exp_epochs", "cf_joint_epochs"):
            flag = f"--{key}"; command[command.index(flag) + 1] = "1"
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bail", "pokec_n", "pokec_z", "toy"), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    validate_data(args.dataset)
    subprocess.run(build_command(args.dataset, args.gpu, args.smoke_test),
                   cwd=ROOT / "cfx_gnn", check=True)


if __name__ == "__main__": main()
