"""Data/model utilities for the paper implementation."""

import dgl
import torch

from models import Backbone, FairDis


def _infer_sens_index(dataset):
    defaults = {"bail": 0, "pokec_z": 3, "pokec_n": 3, "toy": 0}
    if dataset not in defaults:
        raise KeyError(f"No sensitive-feature index is known for {dataset!r}.")
    return defaults[dataset]


def get_model(cfg):
    graphs, _ = dgl.load_graphs(cfg.graph_path)
    graph = graphs[0]
    info = torch.load(cfg.index_path)
    info.setdefault("sens_index", _infer_sens_index(cfg.dataset))
    in_dim = graph.ndata["nfeat"].shape[1]
    hidden_dim = int(getattr(cfg, "hidden_dim", 64))
    if cfg.method != "cf_risk_self_explainer":
        raise ValueError("Only the paper CFX-GNN method is available.")
    if cfg.encoder_type == "gcn":
        encoder = Backbone.GCNEncoder(in_dim, hidden_dim, hidden_dim, 2)
    elif cfg.encoder_type == "appnp":
        encoder = Backbone.APPNPEncoder(in_dim, hidden_dim, hidden_dim, 2)
    else:
        raise ValueError("The paper implementation supports only GCN and APPNP.")
    model = FairDis.self_explainer(encoder, in_dim, hidden_dim, 1)
    return graph, model, info


def flip_sensitive_feature(features, sens_idx):
    counterfactual = features.clone()
    counterfactual[:, sens_idx] = 1.0 - counterfactual[:, sens_idx]
    return counterfactual


def check_correlation(labels, sensitive, mask=None, name=""):
    labels = labels.squeeze()
    sensitive = sensitive.squeeze()
    valid = labels >= 0
    if mask is not None:
        valid &= mask.bool()
    y = labels[valid].float()
    s = sensitive[valid].float()
    print("\n" + "-" * 60)
    print(f"[Correlation Check] {name}" if name else "[Correlation Check]")
    print("-" * 60)
    print(f"Valid labeled nodes: {y.numel()}")
    if y.numel() == 0:
        return None
    result = {}
    for value in (0, 1):
        group = s == value
        result[f"p_y1_s{value}"] = y[group].mean().item() if group.any() else None
        print(f"P(Y=1|S={value}): {result[f'p_y1_s{value}']}")
    if y.std() < 1e-12 or s.std() < 1e-12:
        correlation = 0.0
    else:
        correlation = torch.corrcoef(torch.stack((y, s)))[0, 1].item()
    result["corr"] = correlation
    print(f"Correlation (Y, S): {correlation:.4f}")
    return result
