"""Small evaluation helpers required by the paper training path."""

import os

import torch


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def compute_mask_sparsity(feature_mask, structure_mask, is_self_loop, sens_index=None):
    feature_candidates = torch.ones_like(feature_mask, dtype=torch.bool)
    if sens_index is not None:
        feature_candidates[:, sens_index] = False
    feature_values = feature_mask[feature_candidates].float()
    non_loop = ~is_self_loop
    structure_values = structure_mask[non_loop].float() if non_loop.any() else structure_mask.float()
    feature_ratio = feature_values.mean().item() if feature_values.numel() else 0.0
    structure_ratio = structure_values.mean().item() if structure_values.numel() else 0.0
    return {
        "feature_kept_ratio": float(feature_ratio),
        "feature_sparsity": float(1.0 - feature_ratio),
        "feature_kept_count": int((feature_values > 0).sum().item()),
        "feature_total_count": int(feature_values.numel()),
        "structure_kept_ratio": float(structure_ratio),
        "structure_sparsity": float(1.0 - structure_ratio),
        "structure_kept_count": int((structure_values > 1e-5).sum().item()),
        "structure_total_count": int(structure_values.numel()),
    }


def dataset_mask_target_ranges(dataset):
    ranges = {
        "bail": ((0.08, 0.20), (0.25, 0.70)),
        "pokec_z": ((0.15, 0.35), (0.06, 0.18)),
        "pokec_n": ((0.12, 0.30), (0.10, 0.25)),
        "toy": ((0.03, 0.05), (0.03, 0.05)),
    }
    return ranges[dataset]


def case_feature_names(cfg, num_features):
    dataset_files = {
        "bail": ("bail.csv", "RECID"),
        "pokec_z": ("region_job.csv", "I_am_working_in_field"),
        "pokec_n": ("region_job_2.csv", "I_am_working_in_field"),
    }
    filename, label_name = dataset_files.get(cfg.dataset, (None, None))
    csv_path = os.path.join(cfg.data_dir, filename) if filename else None
    if not csv_path or not os.path.exists(csv_path):
        return [f"feature_{index}" for index in range(num_features)]
    import pandas as pd
    columns = list(pd.read_csv(csv_path, nrows=1).columns)
    for column in (label_name, "user_id"):
        if column in columns:
            columns.remove(column)
    if len(columns) != num_features:
        return [f"feature_{index}" for index in range(num_features)]
    return columns
