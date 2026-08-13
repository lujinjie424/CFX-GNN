"""Training path for counterfactual-risk-guided intrinsic explanations."""

import copy
import json
import os

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.counterfactual_operator import CounterfactualOperator
from modules.donor_selector import DonorSelector
from modules.losses import (
    balanced_supervised_contrastive_loss,
    binary_task_loss,
    conditional_sensitive_contrastive_loss,
    label_conditioned_prototype_contrastive_loss,
    mask_sparsity_loss,
    prototype_contrastive_loss,
    supervised_contrastive_loss,
    weighted_reduce_loss,
)
from modules.risk_weighting import compute_cf_risk, risk_to_weight
from train import metric
from train.paper_helpers import (
    case_feature_names,
    compute_mask_sparsity,
    dataset_mask_target_ranges,
    to_float,
)
from train.utils import check_correlation, flip_sensitive_feature, get_model


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def _grad_reverse(x, scale=1.0):
    return _GradientReversal.apply(x, float(scale))


def _nan_guard(name, value):
    if torch.isnan(value).any() or torch.isinf(value).any():
        raise FloatingPointError(f"{name} became NaN/Inf")


def _cfg_int(cfg, name, default):
    value = getattr(cfg, name, None)
    return int(default if value is None else value)


def _select_case_study_nodes(
    cfg,
    candidate_nodes,
    case_score,
    feature_soft,
    labels,
    sens,
    sens_index,
    prob_full=None,
):
    top_nodes = min(int(getattr(cfg, "case_top_nodes", 2)), candidate_nodes.numel())
    if top_nodes <= 0 or candidate_nodes.numel() == 0:
        return [], {}

    mode = getattr(cfg, "case_selection_mode", "top")
    feature_for_score = feature_soft.clone()
    feature_for_score[:, sens_index] = -1.0
    feature_top_value, feature_top_index = torch.max(feature_for_score, dim=1)

    if mode == "top":
        selected = candidate_nodes[
            torch.topk(case_score[candidate_nodes], k=top_nodes, largest=True).indices
        ].detach().cpu().tolist()
        return [int(node) for node in selected], {
            "selection_mode": mode,
            "feature_top_value": feature_top_value.detach().cpu().numpy(),
            "feature_top_index": feature_top_index.detach().cpu().numpy(),
        }

    candidate_count = min(int(getattr(cfg, "case_candidate_pool", 2000)), candidate_nodes.numel())
    min_feature_soft = float(getattr(cfg, "case_min_feature_soft", 0.05))
    candidate_case = case_score[candidate_nodes]
    candidate_feat = feature_top_value[candidate_nodes]
    case_min, case_max = candidate_case.min(), candidate_case.max()
    feat_min, feat_max = candidate_feat.min(), candidate_feat.max()
    case_norm = (candidate_case - case_min) / (case_max - case_min + 1e-12)
    feat_norm = (candidate_feat - feat_min) / (feat_max - feat_min + 1e-12)
    composite = case_norm + 1.5 * feat_norm

    pool_nodes = candidate_nodes[torch.topk(composite, k=candidate_count, largest=True).indices]
    preferred = pool_nodes[feature_top_value[pool_nodes] >= min_feature_soft]
    if preferred.numel() >= top_nodes:
        pool_nodes = preferred

    rows = []
    prob_cpu = None if prob_full is None else prob_full.detach().cpu().numpy()
    for rank, node in enumerate(pool_nodes.detach().cpu().tolist()):
        node = int(node)
        pred_bucket = 0
        if prob_cpu is not None:
            pred_bucket = int(min(2, max(0, np.floor(float(prob_cpu[node]) * 3.0))))
        rows.append(
            {
                "node": node,
                "label": int(labels[node].detach().cpu().item()),
                "sensitive": int(sens[node].detach().cpu().item()),
                "top_feature_index": int(feature_top_index[node].detach().cpu().item()),
                "top_feature_soft": float(feature_top_value[node].detach().cpu().item()),
                "case_score": float(case_score[node].detach().cpu().item()),
                "composite_rank": rank,
                "pred_bucket": pred_bucket,
            }
        )

    if mode == "sensitive_diverse":
        diversity_keys = ("label", "sensitive")
    elif mode == "feature_diverse":
        diversity_keys = ("label", "sensitive", "top_feature_index", "pred_bucket")
    else:
        raise ValueError(f"Unknown case_selection_mode: {mode}")

    selected = []
    seen = {key: set() for key in diversity_keys}
    remaining = rows.copy()
    while remaining and len(selected) < top_nodes:
        def row_priority(row):
            novelty = sum(row[key] not in seen[key] for key in diversity_keys)
            return (
                novelty,
                row["top_feature_soft"],
                row["case_score"],
                -row["composite_rank"],
            )

        best = max(remaining, key=row_priority)
        selected.append(best)
        remaining.remove(best)
        for key in diversity_keys:
            seen[key].add(best[key])

    return [row["node"] for row in selected], {
        "selection_mode": mode,
        "candidate_pool": int(candidate_count),
        "case_min_feature_soft": min_feature_soft,
        "preferred_candidates": int((feature_top_value[pool_nodes] >= min_feature_soft).sum().detach().cpu().item()),
        "feature_top_value": feature_top_value.detach().cpu().numpy(),
        "feature_top_index": feature_top_index.detach().cpu().numpy(),
        "selected_rows": selected,
    }


def _write_cf_case_study_export(
    cfg,
    gw,
    features,
    labels,
    sens,
    test_mask,
    sens_index,
    feature_soft,
    structure_soft,
    feature_mask,
    structure_mask,
    logits_full,
    logits_cf_set,
    logits_exp,
    logits_exp_cf_set,
    logits_comp,
    logits_comp_cf_set,
    repr_full,
    repr_exp,
    repr_comp,
):
    output_path = getattr(cfg, "case_study_output", None)
    if not output_path:
        return None
    if not os.path.isabs(output_path):
        output_path = os.path.join(cfg.abs_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        prob_full = torch.sigmoid(logits_full).squeeze()
        prob_full_cf = torch.sigmoid(logits_cf_set.mean(dim=1)).squeeze()
        prob_bias = torch.sigmoid(logits_exp).squeeze()
        prob_bias_cf = torch.sigmoid(logits_exp_cf_set.mean(dim=1)).squeeze()
        prob_clean = torch.sigmoid(logits_comp).squeeze()
        prob_clean_cf = torch.sigmoid(logits_comp_cf_set.mean(dim=1)).squeeze()
        case_score = repr_exp + torch.relu(repr_exp - repr_comp) + 0.25 * repr_full
        candidate_nodes = torch.where(test_mask.to(case_score.device))[0]
        selected_nodes, selection_meta = _select_case_study_nodes(
            cfg,
            candidate_nodes,
            case_score,
            feature_soft,
            labels,
            sens,
            sens_index,
            prob_full=prob_full,
        )

    feature_names = case_feature_names(cfg, features.shape[1])
    top_features = int(getattr(cfg, "case_top_features", 5))
    top_neighbors = int(getattr(cfg, "case_top_neighbors", 5))
    src, dst = gw.edges()
    src_cpu = src.detach().cpu().numpy()
    dst_cpu = dst.detach().cpu().numpy()
    feature_soft_cpu = feature_soft.detach().cpu().numpy()
    structure_soft_cpu = structure_soft.detach().cpu().numpy()
    feature_hard_cpu = feature_mask.detach().cpu().numpy()
    structure_hard_cpu = structure_mask.detach().cpu().numpy()
    features_cpu = features.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    sens_cpu = sens.detach().cpu().numpy()

    cases = []
    markdown = [
        "# Bail Learned Bias-Mask Case Study",
        "",
        f"- Dataset: `{cfg.dataset}`",
        f"- Encoder: `{cfg.encoder_type}`",
        f"- Method: `{cfg.method}`",
        f"- Seed: `{cfg.seed}`",
        "- Mask role: `bias_clean`",
        "- CF reference: `local_quantile`",
        f"- Node selection mode: `{selection_meta.get('selection_mode', 'top')}`.",
        "- Base node score: `bias h-cf distance + relu(bias h-cf distance - "
        "clean h-cf distance) + 0.25 * full h-cf distance`.",
        f"- Feature-diverse settings: candidate pool `{selection_meta.get('candidate_pool', 'NA')}`, "
        f"preferred top-feature soft-mask minimum `{selection_meta.get('case_min_feature_soft', 'NA')}`.",
        "- Top masks are soft values from the trained CFX-GNN mask generators; "
        "hard keep is the binary mask used for graph construction.",
        "",
    ]

    for node in selected_nodes:
        node = int(node)
        feature_scores = feature_soft_cpu[node].copy()
        feature_scores[sens_index] = -1.0
        feature_order = np.argsort(-feature_scores)[:top_features]

        neighbor_scores = {}
        neighbor_hard = {}
        for edge_idx, (u, v) in enumerate(zip(src_cpu, dst_cpu)):
            u, v = int(u), int(v)
            if u == v:
                continue
            neighbor = None
            if u == node:
                neighbor = v
            elif v == node:
                neighbor = u
            if neighbor is None:
                continue
            score = float(structure_soft_cpu[edge_idx])
            hard = float(structure_hard_cpu[edge_idx])
            if neighbor not in neighbor_scores or score > neighbor_scores[neighbor]:
                neighbor_scores[neighbor] = score
                neighbor_hard[neighbor] = hard
        neighbor_order = sorted(neighbor_scores, key=lambda item: (-neighbor_scores[item], item))[:top_neighbors]

        case = {
            "node": node,
            "label": int(labels_cpu[node]),
            "sensitive": int(sens_cpu[node]),
            "prob_full": float(prob_full[node].detach().cpu().item()),
            "prob_full_cf": float(prob_full_cf[node].detach().cpu().item()),
            "prob_bias": float(prob_bias[node].detach().cpu().item()),
            "prob_bias_cf": float(prob_bias_cf[node].detach().cpu().item()),
            "prob_clean": float(prob_clean[node].detach().cpu().item()),
            "prob_clean_cf": float(prob_clean_cf[node].detach().cpu().item()),
            "repr_full": float(repr_full[node].detach().cpu().item()),
            "repr_bias": float(repr_exp[node].detach().cpu().item()),
            "repr_clean": float(repr_comp[node].detach().cpu().item()),
            "case_score": float(case_score[node].detach().cpu().item()),
            "top_feature_soft": float(selection_meta["feature_top_value"][node]),
            "top_feature_index": int(selection_meta["feature_top_index"][node]),
            "top_feature_name": feature_names[int(selection_meta["feature_top_index"][node])],
            "top_features": [],
            "top_neighbors": [],
        }
        markdown.extend(
            [
                f"## Case Node {node}",
                "",
                "| Node | Label | Sensitive | P(full) | P(full CF) | P(bias) | P(bias CF) | "
                "P(clean) | P(clean CF) | d_full | d_bias | d_clean |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {node} | {case['label']} | {case['sensitive']} | "
                f"{case['prob_full']:.4f} | {case['prob_full_cf']:.4f} | "
                f"{case['prob_bias']:.4f} | {case['prob_bias_cf']:.4f} | "
                f"{case['prob_clean']:.4f} | {case['prob_clean_cf']:.4f} | "
                f"{case['repr_full']:.4f} | {case['repr_bias']:.4f} | {case['repr_clean']:.4f} |",
                "",
                f"- Selection score: `{case['case_score']:.4f}`.",
                f"- Strongest non-sensitive feature mask: `{case['top_feature_name']}` "
                f"=`{case['top_feature_soft']:.4f}`.",
                "",
                "### Top Feature Masks",
                "",
                "| Rank | Feature | Soft bias mask | Hard keep | Node value |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, feat_idx in enumerate(feature_order, start=1):
            row = {
                "rank": rank,
                "feature_index": int(feat_idx),
                "feature": feature_names[int(feat_idx)],
                "soft_mask": float(feature_soft_cpu[node, feat_idx]),
                "hard_keep": int(feature_hard_cpu[node, feat_idx] > 0),
                "value": float(features_cpu[node, feat_idx]),
            }
            case["top_features"].append(row)
            markdown.append(
                f"| {rank} | `{row['feature']}` | {row['soft_mask']:.4f} | "
                f"{row['hard_keep']} | {row['value']:.4f} |"
            )

        markdown.extend(
            [
                "",
                "### Top Structure Masks",
                "",
                "| Rank | Neighbor | Soft bias mask | Hard keep | Neighbor label | Neighbor sensitive | Same sensitive |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, neighbor in enumerate(neighbor_order, start=1):
            row = {
                "rank": rank,
                "neighbor": int(neighbor),
                "soft_mask": float(neighbor_scores[neighbor]),
                "hard_keep": int(neighbor_hard[neighbor] > 0),
                "neighbor_label": int(labels_cpu[neighbor]),
                "neighbor_sensitive": int(sens_cpu[neighbor]),
                "same_sensitive": int(sens_cpu[neighbor] == sens_cpu[node]),
            }
            case["top_neighbors"].append(row)
            markdown.append(
                f"| {rank} | {row['neighbor']} | {row['soft_mask']:.4f} | "
                f"{row['hard_keep']} | {row['neighbor_label']} | "
                f"{row['neighbor_sensitive']} | {row['same_sensitive']} |"
            )
        markdown.append("")
        cases.append(case)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown))
    json_path = os.path.splitext(output_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"cases": cases}, handle, indent=2)
    print(f"Case-study learned masks exported to {output_path}")
    print(f"Case-study JSON exported to {json_path}")
    return {"markdown": output_path, "json": json_path, "nodes": [case["node"] for case in cases]}


def _set_requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad = flag


def _zy_dim_from_model(model, fallback):
    return int(getattr(model.fairINN, "y_dim", getattr(model.fairINN, "u_dim", fallback)))


def _configure_warm_y_head(model, cfg, device):
    mode = getattr(cfg, "warm_y_loss", "mlp_bce")
    if mode == "linear_bce":
        z_y_dim = _zy_dim_from_model(model, getattr(model.fairINN, "y_dim", 0))
        model.fairINN.classifier_y = torch.nn.Linear(z_y_dim, 1).to(device)
    return model.fairINN.classifier_y


def _classifier_logits(classifier, z_y):
    return classifier(z_y)


def _uses_h_prediction(cfg):
    return getattr(cfg, "final_pred_source", "zy") == "h"


def _logits_from_h(model, classifier, h, cfg):
    z_y, z_s, _, pred_y, _ = model.fairINN(h)
    if _uses_h_prediction(cfg):
        return classifier(h), z_y, z_s
    return pred_y, z_y, z_s


def _sensitive_proxy_loss(model, z_u, z_b, pred_s, labels, sens, train_mask, cfg):
    """Warm-up z_b as a sensitive-related proxy without making CE mandatory."""
    mode = getattr(cfg, "warm_sensitive_loss", "contrastive")
    temperature = getattr(cfg, "contrast_temperature", 0.07)
    max_samples = getattr(cfg, "contrast_max_samples", 2048)
    z_train = z_b[train_mask]
    s_train = sens[train_mask]
    y_train = labels[train_mask]

    losses = {}
    zero = z_b.new_tensor(0.0)
    if mode == "distance_softplus":
        raw_di = model.fairINN.loss_fn.loss_di(z_train, s_train, y_train)
        losses["s_con"] = F.softplus(raw_di)
    elif mode == "distance":
        losses["s_con"] = model.fairINN.loss_fn.loss_di(z_train, s_train, y_train)
    elif mode == "joo_di":
        z_y_train = z_u[train_mask]
        if max_samples is not None and z_train.size(0) > int(max_samples):
            perm = torch.randperm(z_train.size(0), device=z_train.device)[: int(max_samples)]
            z_train = z_train[perm]
            z_y_train = z_y_train[perm]
            s_train = s_train[perm]
            y_train = y_train[perm]
        raw_zs = model.fairINN.loss_fn.loss_di(z_train, s_train, y_train)
        raw_zy = model.fairINN.loss_fn.loss_di(z_y_train, y_train, s_train)
        losses["s_con"] = F.softplus(raw_zs) + float(getattr(cfg, "joo_di_y_weight", 1.0)) * F.softplus(raw_zy)
    elif mode == "conditional_contrastive":
        losses["s_con"] = conditional_sensitive_contrastive_loss(
            z_train,
            s_train,
            y_train,
            margin=getattr(cfg, "s_contrast_margin", 1.0),
            max_samples=max_samples,
        )
    elif mode in ("contrastive", "contrastive_ce"):
        secondary = None if getattr(cfg, "no_label_conditioned_zb", False) else y_train
        losses["s_con"] = supervised_contrastive_loss(
            z_train,
            s_train,
            temperature=temperature,
            secondary_key=secondary,
            max_samples=max_samples,
        )
    elif mode == "prototype":
        losses["s_con"] = prototype_contrastive_loss(z_train, s_train, temperature=temperature)
    elif mode == "label_conditioned_prototype":
        losses["s_con"] = label_conditioned_prototype_contrastive_loss(
            z_train,
            s_train,
            y_train,
            temperature=temperature,
            max_samples=max_samples,
        )
    else:
        losses["s_con"] = zero

    if mode in ("ce", "contrastive_ce") or float(getattr(cfg, "lambda_s_ce", 0.0)) > 0.0:
        losses["s_ce"] = F.binary_cross_entropy_with_logits(
            pred_s[train_mask], s_train.float().view(-1, 1)
        )
    else:
        losses["s_ce"] = zero
    return losses


def _cf_logits(model, classifier, h_cf_set, cfg):
    """Run the configured task head on [N, K, d] counterfactual h representations."""
    num_nodes, num_donors, hidden_dim = h_cf_set.shape
    h_flat = h_cf_set.reshape(num_nodes * num_donors, hidden_dim)
    if _uses_h_prediction(cfg):
        return classifier(h_flat).reshape(num_nodes, num_donors, -1)
    _, _, _, logits, _ = model.fairINN(h_flat)
    return logits.reshape(num_nodes, num_donors, -1)


def _sensitive_exposure_from_z_s(model, z_s):
    """How confidently z_s carries sensitive information, in [0, 1]."""
    logits_s = model.fairINN.classifier_s(z_s).view(-1)
    return torch.abs(torch.sigmoid(logits_s) - 0.5) * 2.0


def _binary_concrete_sample(logits, temperature):
    eps = 1e-6
    u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
    noise = torch.log(u) - torch.log1p(-u)
    return torch.sigmoid((logits + noise) / max(float(temperature), eps))


def _build_soft_masks(
    model,
    g,
    x,
    h,
    h_cf_mean,
    train_mask,
    sens_index,
    tau=1.0,
    stochastic_structure=False,
    stochastic_temperature=0.5,
):
    """Build differentiable feature and edge masks.

    feature_mask: [N, F]; structure_mask: [E], matching the current graph edges.
    """
    feature_mask = model.featureMask_Generator(x, h, h_cf_mean.detach())
    feature_mask = feature_mask.clone()
    feature_mask[:, sens_index] = 0.0

    src, dst = g.edges()
    is_self_loop = src == dst
    edge_feat = torch.cat([h[src], h[dst], h_cf_mean[src].detach(), h_cf_mean[dst].detach()], dim=-1)
    edge_logits = model.structureMask_Generator(edge_feat).squeeze(-1)
    if stochastic_structure:
        structure_mask = _binary_concrete_sample(edge_logits, stochastic_temperature)
    else:
        structure_mask = torch.sigmoid(edge_logits / max(float(tau), 1e-6))
    structure_mask = structure_mask.clamp(min=1e-6, max=1.0 - 1e-6)
    structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

    ablation_mode = getattr(model, "mask_ablation_mode", "full")
    if ablation_mode == "feature_only":
        structure_mask = torch.full_like(structure_mask, 1e-6).masked_fill(is_self_loop, 1.0)
    elif ablation_mode == "structure_only":
        feature_mask = torch.zeros_like(feature_mask)
    elif ablation_mode != "full":
        raise ValueError(f"Unknown mask_ablation_mode: {ablation_mode}")

    x_base = model.get_x_base(x, train_mask)
    x_exp = feature_mask * x + (1.0 - feature_mask) * x_base
    return feature_mask, structure_mask, x_exp, is_self_loop


def _build_dynamic_soft_masks(
    model,
    g,
    x,
    h_seed,
    h_cf_seed,
    train_mask,
    sens_index,
    tau,
    cf_operator,
    donor_indices,
    z_b_bank,
    dynamic=True,
    stochastic_structure=False,
    stochastic_temperature=0.5,
):
    """Build masks conditioned on the current explanation-state counterfactual.

    The first pass produces a provisional explanation state. The second pass
    conditions the mask generators on that state's own counterfactual direction,
    aligning the explainer input with the loss that is optimized downstream.
    """
    if not dynamic:
        return _build_soft_masks(
            model,
            g,
            x,
            h_seed,
            h_cf_seed,
            train_mask,
            sens_index,
            tau=tau,
            stochastic_structure=stochastic_structure,
            stochastic_temperature=stochastic_temperature,
        )

    with torch.no_grad():
        feature_seed, structure_seed, x_seed, _ = _build_soft_masks(
            model, g, x, h_seed, h_cf_seed, train_mask, sens_index, tau=tau
        )
        h_state = model.encoder(g, x_seed, edge_weight=structure_seed)
        h_state_cf = cf_operator.apply(h_state, donor_indices, z_b_bank=z_b_bank).mean(dim=1)

    return _build_soft_masks(
        model,
        g,
        x,
        h_state.detach(),
        h_state_cf.detach(),
        train_mask,
        sens_index,
        tau=tau,
        stochastic_structure=stochastic_structure,
        stochastic_temperature=stochastic_temperature,
    )


def _cf_representation_invariance_loss(h, h_cf_set, weights=None, mask=None):
    """Weighted distance between current explanation states and their CF states."""
    h_cf_mean = h_cf_set.mean(dim=1).detach()
    dist = torch.norm(h - h_cf_mean, p=2, dim=1) / (h.size(1) ** 0.5)
    if weights is not None:
        dist = dist * weights.to(dist.device).detach()
    if mask is not None:
        dist = dist[mask.to(dist.device).bool()]
    return dist.mean()


def _cf_representation_distance(h, h_cf_set):
    """Per-node donor-wise representation shift under counterfactual intervention."""
    dist = torch.norm(h_cf_set - h.unsqueeze(1), p=2, dim=-1) / (h.size(1) ** 0.5)
    return dist.mean(dim=1)


def _prediction_sufficiency_loss(logits_exp, labels, mask=None):
    """Keep the explanation graph predictive in the same space used for evaluation."""
    labels = labels.float().view(-1, 1).to(logits_exp.device)
    per_node = F.binary_cross_entropy_with_logits(logits_exp, labels, reduction="none").view(-1)
    if mask is not None:
        per_node = per_node[mask.to(per_node.device).bool()]
    return per_node.mean()


def _masked_weighted_reduce_loss(risk_exp, risk_full, weights, mask, rho=0.5):
    margin = risk_exp - float(rho) * risk_full.detach()
    loss = weights.detach() * F.relu(margin)
    mask = mask.to(loss.device).bool()
    return loss[mask].mean() if mask.any() else loss.mean()


def _masked_weighted_mean(values, weights, mask):
    values = values.view(-1)
    loss = values * weights.to(values.device).detach().view(-1)
    mask = mask.to(loss.device).bool()
    return loss[mask].mean() if mask.any() else loss.mean()


def _apply_donor_confidence_weight(weights, donor_reliability, cfg):
    if donor_reliability is None:
        return weights
    reliability = donor_reliability.to(weights.device).float().view(-1)
    if getattr(cfg, "use_donor_reliability_gate", False):
        tau = float(getattr(cfg, "donor_reliability_gate_tau", 0.8))
        floor = float(getattr(cfg, "donor_reliability_gate_floor", 0.5))
        gate = torch.where(
            reliability < tau,
            reliability.new_full(reliability.shape, floor),
            torch.ones_like(reliability),
        )
        return weights * gate.detach()
    if not getattr(cfg, "use_donor_confidence_weight", False):
        return weights
    floor = float(getattr(cfg, "donor_reliability_floor", 0.05))
    reliability = reliability.clamp(min=floor, max=1.0)
    return weights * reliability.detach()


def _cf_group_calibration_loss(logits, logits_cf_mean, sensitive, mask):
    """Align each observed group's prediction distribution to its CF reference."""
    prob = torch.sigmoid(logits.view(-1))
    prob_cf = torch.sigmoid(logits_cf_mean.view(-1))
    sensitive = sensitive.to(prob.device).long().view(-1)
    mask = mask.to(prob.device).bool().view(-1)
    losses = []
    for group in (0, 1):
        group_mask = mask & (sensitive == group)
        if group_mask.any():
            losses.append(torch.abs(prob[group_mask].mean() - prob_cf[group_mask].mean()))
    return sum(losses) / len(losses) if losses else prob.new_tensor(0.0)


def _cf_pair_calibration_loss(logits, logits_cf_set, weights, mask):
    """Node-level prediction consistency to all CF references."""
    logits = logits.view(-1, 1, 1)
    pair_gap = torch.abs(logits - logits_cf_set).mean(dim=(1, 2))
    return _masked_weighted_mean(pair_gap, weights, mask)


@torch.no_grad()
def _calibration_selection_metrics(logits, logits_cf_set, labels, sens, eval_mask):
    eval_metric = metric.evaluate(
        logits[eval_mask],
        labels[eval_mask],
        sens[eval_mask],
        logits_cf_set.mean(dim=1)[eval_mask],
    )
    logit_gap = torch.abs(
        logits[eval_mask].view(-1) - logits_cf_set.mean(dim=1)[eval_mask].view(-1)
    ).mean()
    eval_metric["cf_logit_gap"] = float(logit_gap.item())
    return eval_metric


def _calibration_selection_score(cfg, eval_metric):
    return (
        float(eval_metric["parity"])
        + float(eval_metric["equality"])
        + float(getattr(cfg, "calib_select_cf_weight", 1.0)) * float(eval_metric["cf"])
        + float(getattr(cfg, "calib_select_logit_weight", 0.5)) * float(eval_metric["cf_logit_gap"])
        - float(getattr(cfg, "calib_select_auc_weight", 0.0)) * float(eval_metric["auc_roc"])
    )


def _attach_comp_metrics(eval_metric, comp_metric):
    eval_metric["comp_auc_roc"] = float(comp_metric["auc_roc"])
    eval_metric["comp_cf"] = float(comp_metric["cf"])
    eval_metric["exp_comp_auc_gap"] = float(eval_metric["auc_roc"] - comp_metric["auc_roc"])
    return eval_metric


def _fair_checkpoint_selection_score(cfg, eval_metric):
    return (
        float(eval_metric["parity"])
        + float(eval_metric["equality"])
        + float(getattr(cfg, "fair_stop_cf_weight", 1.0)) * float(eval_metric["cf"])
        + float(getattr(cfg, "fair_stop_logit_weight", 0.5)) * float(eval_metric["cf_logit_gap"])
        - float(getattr(cfg, "fair_stop_auc_weight", 0.0)) * float(eval_metric["auc_roc"])
    )


def _local_mask_sparsity_loss(feature_mask, structure_mask, is_self_loop, sens_index, local_node_mask, local_edge_mask):
    def _bernoulli_ib_kl(values):
        if values.numel() == 0:
            return feature_mask.new_tensor(0.0)
        prob = values.clamp(min=1e-6, max=1.0 - 1e-6)
        prior = prob.detach().mean().clamp(min=1e-6, max=1.0 - 1e-6)
        kl = prob * (torch.log(prob) - torch.log(prior))
        kl = kl + (1.0 - prob) * (torch.log1p(-prob) - torch.log1p(-prior))
        return kl.mean()

    feature_candidates = local_node_mask.to(feature_mask.device).bool().unsqueeze(1).expand_as(feature_mask).clone()
    if sens_index is not None:
        feature_candidates[:, sens_index] = False
    if feature_candidates.any():
        feature_part = _bernoulli_ib_kl(feature_mask[feature_candidates])
    else:
        feature_part = feature_mask.new_tensor(0.0)

    edge_candidates = local_edge_mask.to(structure_mask.device).bool() & (~is_self_loop)
    if edge_candidates.any():
        structure_part = _bernoulli_ib_kl(structure_mask[edge_candidates])
    else:
        fallback = structure_mask[~is_self_loop] if (~is_self_loop).any() else structure_mask
        structure_part = _bernoulli_ib_kl(fallback)
    return feature_part + structure_part


def _local_mask_means(feature_mask, structure_mask, is_self_loop, sens_index, local_node_mask, local_edge_mask):
    feature_candidates = local_node_mask.to(feature_mask.device).bool().unsqueeze(1).expand_as(feature_mask).clone()
    if sens_index is not None:
        feature_candidates[:, sens_index] = False
    if feature_candidates.any():
        feature_mean = feature_mask[feature_candidates].mean()
    else:
        feature_mean = feature_mask.new_tensor(0.0)

    edge_candidates = local_edge_mask.to(structure_mask.device).bool() & (~is_self_loop)
    if edge_candidates.any():
        structure_mean = structure_mask[edge_candidates].mean()
    else:
        structure_mean = structure_mask[~is_self_loop].mean() if (~is_self_loop).any() else structure_mask.mean()
    return feature_mean, structure_mean


def _local_mask_budget_loss(
    feature_mask,
    structure_mask,
    is_self_loop,
    sens_index,
    local_node_mask,
    local_edge_mask,
    feature_budget=None,
    structure_budget=None,
):
    loss = feature_mask.new_tensor(0.0)
    feature_candidates = local_node_mask.to(feature_mask.device).bool().unsqueeze(1).expand_as(feature_mask).clone()
    if sens_index is not None:
        feature_candidates[:, sens_index] = False
    if feature_budget is not None and feature_candidates.any():
        feature_mean = feature_mask[feature_candidates].mean()
        loss = loss + F.relu(feature_mean - float(feature_budget)).pow(2)

    edge_candidates = local_edge_mask.to(structure_mask.device).bool() & (~is_self_loop)
    if structure_budget is not None and edge_candidates.any():
        structure_mean = structure_mask[edge_candidates].mean()
        loss = loss + F.relu(structure_mean - float(structure_budget)).pow(2)
    return loss


def _local_mask_entropy_loss(structure_mask, is_self_loop, local_edge_mask):
    edge_candidates = local_edge_mask.to(structure_mask.device).bool() & (~is_self_loop)
    if not edge_candidates.any():
        return structure_mask.new_tensor(0.0)
    prob = structure_mask[edge_candidates].clamp(min=1e-6, max=1.0 - 1e-6)
    entropy = -(prob * torch.log(prob) + (1.0 - prob) * torch.log1p(-prob))
    return entropy.mean()


def _apply_fixed_local_scope(feature_mask, structure_mask, is_self_loop, local_node_mask, local_edge_mask):
    """Keep non-local features/edges unchanged so only the selected region is explained."""
    local_node_mask = local_node_mask.to(feature_mask.device).bool()
    local_edge_mask = local_edge_mask.to(structure_mask.device).bool()
    scoped_feature = torch.where(
        local_node_mask.unsqueeze(1),
        feature_mask,
        torch.ones_like(feature_mask),
    )
    scoped_structure = torch.where(
        local_edge_mask | is_self_loop,
        structure_mask,
        torch.ones_like(structure_mask),
    )
    return scoped_feature, scoped_structure


def _apply_fixed_bias_scope(feature_mask, structure_mask, is_self_loop, local_node_mask, local_edge_mask):
    """Restrict the bias mask to the selected local region; clean complement keeps non-local graph."""
    local_node_mask = local_node_mask.to(feature_mask.device).bool()
    local_edge_mask = local_edge_mask.to(structure_mask.device).bool()
    scoped_feature = torch.where(
        local_node_mask.unsqueeze(1),
        feature_mask,
        torch.zeros_like(feature_mask),
    )
    scoped_structure = torch.where(
        local_edge_mask,
        structure_mask,
        torch.full_like(structure_mask, 1e-6),
    )
    scoped_structure = scoped_structure.masked_fill(is_self_loop, 1.0)
    return scoped_feature, scoped_structure


def _is_bias_clean_mode(cfg):
    return True


def _nonlocal_keep_loss(h_exp, h_full, local_node_mask):
    nonlocal_mask = ~local_node_mask.to(h_exp.device).bool()
    if not nonlocal_mask.any():
        return h_exp.new_tensor(0.0)
    dist = torch.norm(h_exp - h_full.detach(), p=2, dim=1) / (h_exp.size(1) ** 0.5)
    return dist[nonlocal_mask].mean()


@torch.no_grad()
def _select_local_region(g, scores, candidate_mask, top_k=512, hops=2):
    device = scores.device
    candidate_mask = candidate_mask.to(device).bool()
    candidate_idx = torch.where(candidate_mask)[0]
    if candidate_idx.numel() == 0:
        candidate_idx = torch.arange(scores.numel(), device=device)

    k = min(max(1, int(top_k)), candidate_idx.numel())
    _, local_top = torch.topk(scores[candidate_idx], k=k, largest=True)
    anchors = candidate_idx[local_top]

    src, dst = g.edges()
    src = src.to(device)
    dst = dst.to(device)
    node_mask = torch.zeros(g.num_nodes(), dtype=torch.bool, device=device)
    edge_mask = torch.zeros(g.num_edges(), dtype=torch.bool, device=device)
    frontier = torch.zeros_like(node_mask)
    frontier[anchors] = True
    node_mask[anchors] = True

    for _ in range(max(1, int(hops))):
        incident = frontier[src] | frontier[dst]
        edge_mask |= incident
        new_nodes = torch.zeros_like(node_mask)
        if incident.any():
            new_nodes[src[incident]] = True
            new_nodes[dst[incident]] = True
        frontier = new_nodes & (~node_mask)
        node_mask |= new_nodes
        if not frontier.any():
            break

    anchor_mask = torch.zeros_like(node_mask)
    anchor_mask[anchors] = True
    return anchor_mask, node_mask, edge_mask


def _cap_mask_by_ratio(scores, candidates, max_ratio):
    selected = torch.zeros_like(scores)
    values = scores[candidates]
    if values.numel() == 0:
        return selected
    keep = max(1, int(values.numel() * float(max_ratio)))
    top = torch.topk(values, k=min(keep, values.numel()), largest=True).indices
    flat = torch.zeros_like(values)
    flat[top] = 1.0
    selected[candidates] = flat
    return selected


@torch.no_grad()
def _hard_masks_from_soft(
    feature_soft,
    structure_soft,
    is_self_loop,
    sens_index,
    feature_max,
    structure_max,
    ablation_mode="full",
):
    feature_candidates = torch.ones_like(feature_soft, dtype=torch.bool)
    feature_candidates[:, sens_index] = False
    feature_mask = _cap_mask_by_ratio(feature_soft, feature_candidates, feature_max)
    feature_mask[:, sens_index] = 0.0

    non_loop = ~is_self_loop
    structure_mask = _cap_mask_by_ratio(structure_soft, non_loop, structure_max)
    structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)
    if ablation_mode == "feature_only":
        structure_mask = torch.zeros_like(structure_mask).masked_fill(is_self_loop, 1.0)
    elif ablation_mode == "structure_only":
        feature_mask = torch.zeros_like(feature_mask)
    elif ablation_mode != "full":
        raise ValueError(f"Unknown mask_ablation_mode: {ablation_mode}")
    return feature_mask, structure_mask


def _straight_through_hard_masks(
    feature_soft,
    structure_soft,
    is_self_loop,
    sens_index,
    feature_max,
    structure_max,
    ablation_mode="full",
):
    feature_hard, structure_hard = _hard_masks_from_soft(
        feature_soft,
        structure_soft,
        is_self_loop,
        sens_index,
        feature_max,
        structure_max,
        ablation_mode=ablation_mode,
    )
    structure_hard = structure_hard.clamp(min=1e-6)
    feature_mask = feature_hard.detach() + feature_soft - feature_soft.detach()
    structure_mask = structure_hard.detach() + structure_soft - structure_soft.detach()
    return feature_mask, structure_mask


@torch.no_grad()
def _matched_random_bias_clean_diagnostics(
    model,
    gw,
    features,
    labels,
    sens,
    test_mask,
    train_mask,
    classifier,
    cf_operator,
    donor_indices,
    z_b_bank,
    feature_mask,
    structure_mask,
    is_self_loop,
    sens_index,
    cfg,
):
    num_trials = max(1, int(getattr(cfg, "random_mask_trials", 5)))
    feature_candidates = torch.ones_like(feature_mask, dtype=torch.bool)
    feature_candidates[:, sens_index] = False
    feature_positions = torch.where(feature_candidates.flatten())[0]
    feature_keep = int((feature_mask[feature_candidates] > 0.5).sum().item())

    non_loop = ~is_self_loop
    edge_positions = torch.where(non_loop)[0]
    edge_keep = int((structure_mask[non_loop] > 1e-5).sum().item()) if non_loop.any() else 0

    trials = []
    x_base = model.get_x_base(features, train_mask)
    for _ in range(num_trials):
        random_feature = torch.zeros_like(feature_mask)
        if feature_keep > 0 and feature_positions.numel() > 0:
            selected = feature_positions[
                torch.randperm(feature_positions.numel(), device=feature_positions.device)[
                    : min(feature_keep, feature_positions.numel())
                ]
            ]
            random_feature.flatten()[selected] = 1.0
        random_feature[:, sens_index] = 0.0

        random_structure = torch.full_like(structure_mask, 1e-6)
        random_structure = random_structure.masked_fill(is_self_loop, 1.0)
        if edge_keep > 0 and edge_positions.numel() > 0:
            selected = edge_positions[
                torch.randperm(edge_positions.numel(), device=edge_positions.device)[
                    : min(edge_keep, edge_positions.numel())
                ]
            ]
            random_structure[selected] = 1.0

        feature_clean, structure_clean = model.get_complement_masks(
            random_feature, random_structure, is_self_loop
        )
        structure_clean = structure_clean.clamp(min=1e-6)
        x_bias = random_feature * features + (1.0 - random_feature) * x_base
        x_clean = feature_clean * features + (1.0 - feature_clean) * x_base

        h_bias = model.encoder(gw, x_bias, edge_weight=random_structure)
        logits_bias, logits_bias_cf_set, risk_bias, _ = _counterfactual_risk_snapshot(
            model, h_bias, classifier, cf_operator, donor_indices, z_b_bank, cfg
        )
        h_clean = model.encoder(gw, x_clean, edge_weight=structure_clean)
        logits_clean, logits_clean_cf_set, risk_clean, _ = _counterfactual_risk_snapshot(
            model, h_clean, classifier, cf_operator, donor_indices, z_b_bank, cfg
        )

        bias_metric = metric.evaluate(
            logits_bias[test_mask],
            labels[test_mask],
            sens[test_mask],
            logits_bias_cf_set.mean(dim=1)[test_mask],
        )
        clean_metric = metric.evaluate(
            logits_clean[test_mask],
            labels[test_mask],
            sens[test_mask],
            logits_clean_cf_set.mean(dim=1)[test_mask],
        )
        trials.append(
            {
                "bias_auc": to_float(bias_metric["auc_roc"]),
                "bias_dp": to_float(bias_metric["parity"]),
                "bias_eo": to_float(bias_metric["equality"]),
                "bias_cf": to_float(bias_metric["cf"]),
                "clean_auc": to_float(clean_metric["auc_roc"]),
                "clean_dp": to_float(clean_metric["parity"]),
                "clean_eo": to_float(clean_metric["equality"]),
                "clean_cf": to_float(clean_metric["cf"]),
                "clean_bias_auc_gap": to_float(clean_metric["auc_roc"]) - to_float(bias_metric["auc_roc"]),
                "risk_bias_mean": float(risk_bias[test_mask].mean().item()),
                "risk_clean_mean": float(risk_clean[test_mask].mean().item()),
                "risk_bias_minus_clean": float((risk_bias[test_mask] - risk_clean[test_mask]).mean().item()),
            }
        )

    summary = {
        "enabled": True,
        "num_trials": int(num_trials),
        "matched_feature_keep": int(feature_keep),
        "matched_structure_keep": int(edge_keep),
    }
    for key in trials[0]:
        values = [float(trial[key]) for trial in trials]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        summary[key] = float(mean)
        summary[f"{key}_std"] = float(var ** 0.5)
    summary["trials"] = trials
    return summary


def _counterfactual_risk_snapshot(model, h, classifier, cf_operator, donor_indices, z_b_bank, args):
    h_cf_set = cf_operator.apply(h, donor_indices, z_b_bank=z_b_bank)
    logits, _, z_b = _logits_from_h(model, classifier, h, args)
    logits_cf_set = _cf_logits(model, classifier, h_cf_set, args)
    cf_risk = compute_cf_risk(
        logits,
        logits_cf_set,
        mode=getattr(args, "risk_mode", "mean_logit_gap"),
        var_gamma=getattr(args, "risk_var_gamma", 0.0),
    )
    z_s_exposure = _sensitive_exposure_from_z_s(model, z_b)
    risk = cf_risk + float(getattr(args, "lambda_zs_fair", 1.0)) * z_s_exposure
    return logits, logits_cf_set, risk, h_cf_set


@torch.no_grad()
def _hard_explanation_selection_metrics(
    model,
    gw,
    features,
    labels,
    sens,
    eval_mask,
    train_mask,
    classifier,
    cf_operator,
    donor_indices,
    sens_index,
    feature_max,
    structure_max,
    local_node_mask,
    local_edge_mask,
    cfg,
):
    """Evaluate validation metrics on the same hard explanation graph used at test time."""
    h_full = model.encoder(gw, features)
    _, z_b_full = cf_operator.encode(h_full)
    h_cf_set = cf_operator.apply(h_full, donor_indices, z_b_bank=z_b_full.detach())
    feature_soft, structure_soft, _, is_self_loop = _build_dynamic_soft_masks(
        model,
        gw,
        features,
        h_full,
        h_cf_set.mean(dim=1),
        train_mask,
        sens_index,
        tau=getattr(cfg, "mask_tau", 1.0),
        cf_operator=cf_operator,
        donor_indices=donor_indices,
        z_b_bank=z_b_full.detach(),
        dynamic=not getattr(cfg, "static_exp_condition", False),
        stochastic_structure=False,
    )
    scope_fn = _apply_fixed_bias_scope if _is_bias_clean_mode(cfg) else _apply_fixed_local_scope
    feature_soft, structure_soft = scope_fn(
        feature_soft, structure_soft, is_self_loop, local_node_mask, local_edge_mask
    )
    feature_mask, structure_mask = _hard_masks_from_soft(
        feature_soft,
        structure_soft,
        is_self_loop,
        sens_index,
        feature_max,
        structure_max,
        ablation_mode=getattr(cfg, "mask_ablation_mode", "full"),
    )
    feature_mask, structure_mask = scope_fn(
        feature_mask, structure_mask, is_self_loop, local_node_mask, local_edge_mask
    )
    exp_graph, exp_feature, _, _, _ = model.get_explained_graph(
        gw, features, feature_mask, structure_mask, train_mask
    )
    feature_comp, structure_comp = model.get_complement_masks(feature_mask, structure_mask, is_self_loop)
    comp_graph, comp_feature, _, _, _ = model.get_explained_graph(
        gw, features, feature_comp, structure_comp, train_mask
    )
    h_exp = model.encoder(exp_graph, exp_feature)
    h_exp_cf_set = cf_operator.apply(h_exp, donor_indices, z_b_bank=z_b_full.detach())
    logits_exp, _, _ = _logits_from_h(model, classifier, h_exp, cfg)
    logits_exp_cf_set = _cf_logits(model, classifier, h_exp_cf_set, cfg)
    bias_metric = _calibration_selection_metrics(logits_exp, logits_exp_cf_set, labels, sens, eval_mask)
    h_comp = model.encoder(comp_graph, comp_feature)
    h_comp_cf_set = cf_operator.apply(h_comp, donor_indices, z_b_bank=z_b_full.detach())
    logits_comp, _, _ = _logits_from_h(model, classifier, h_comp, cfg)
    logits_comp_cf_set = _cf_logits(model, classifier, h_comp_cf_set, cfg)
    clean_metric = _calibration_selection_metrics(logits_comp, logits_comp_cf_set, labels, sens, eval_mask)
    if _is_bias_clean_mode(cfg):
        eval_metric = dict(clean_metric)
        eval_metric["bias_auc_roc"] = float(bias_metric["auc_roc"])
        eval_metric["bias_cf"] = float(bias_metric["cf"])
        eval_metric["clean_auc_roc"] = float(clean_metric["auc_roc"])
        eval_metric["clean_cf"] = float(clean_metric["cf"])
        eval_metric["comp_auc_roc"] = float(bias_metric["auc_roc"])
        eval_metric["comp_cf"] = float(bias_metric["cf"])
        eval_metric["exp_comp_auc_gap"] = float(clean_metric["auc_roc"] - bias_metric["auc_roc"])
    else:
        eval_metric = dict(bias_metric)
        eval_metric["comp_auc_roc"] = float(clean_metric["auc_roc"])
        eval_metric["comp_cf"] = float(clean_metric["cf"])
        eval_metric["exp_comp_auc_gap"] = float(bias_metric["auc_roc"] - clean_metric["auc_roc"])
    eval_metric["hard_feature_keep"] = float(feature_mask.mean().item())
    eval_metric["hard_structure_keep"] = float(structure_mask[~is_self_loop].mean().item())
    return eval_metric


def train(cfg):
    paper_method = {
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
    mismatches = {
        name: (getattr(cfg, name, None), expected)
        for name, expected in paper_method.items()
        if getattr(cfg, name, None) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Unsupported non-paper method configuration: {details}")
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    gw, model, info = get_model(cfg)
    gw = dgl.remove_self_loop(gw)
    gw = dgl.add_self_loop(gw).to(device)
    model.to(device)
    model.mask_ablation_mode = getattr(cfg, "mask_ablation_mode", "full")

    train_node = info["train_index"].long()[:cfg.index] if cfg.index else info["train_index"].long()
    valid_node = info["valid_index"].long()
    test_node = info["test_index"].long()
    labels = info["label"].long().to(device)
    sens = info["sens"].long().to(device)
    sens_index = int(info["sens_index"])
    features = gw.ndata["nfeat"].to(device)

    train_mask = torch.zeros(gw.num_nodes(), dtype=torch.bool, device=device)
    valid_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    train_mask[train_node.to(device)] = True
    valid_mask[valid_node.to(device)] = True
    test_mask[test_node.to(device)] = True

    feature_range, structure_range = dataset_mask_target_ranges(cfg.dataset)
    feature_max = float(getattr(cfg, "mask_feature_target_max", None) or feature_range[1])
    structure_max = float(getattr(cfg, "mask_structure_target_max", None) or structure_range[1])

    print("Sensitive attribute distribution:", sens.sum(), "/", len(sens))
    check_correlation(labels, sens)

    h_dim = int(getattr(cfg, "hidden_dim", 64))
    z_y_dim = _zy_dim_from_model(model, h_dim // 2)
    z_b_dim = int(getattr(model.fairINN, "s_dim", h_dim - z_y_dim))
    classifier_dim = h_dim if _uses_h_prediction(cfg) else z_y_dim
    classifier = torch.nn.Linear(classifier_dim, 1).to(device)
    z_b_label_adversary = torch.nn.Sequential(
        torch.nn.Linear(z_b_dim, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 1),
    ).to(device)
    cf_operator = CounterfactualOperator(model.fairINN)
    cf_operator.configure_reference(sensitive=sens)
    _configure_warm_y_head(model, cfg, device)

    print("\n================ Phase 1: Counterfactual operator warm-up ================")
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.fairINN.parameters())
        + list(classifier.parameters())
        + list(z_b_label_adversary.parameters()),
        lr=cfg.lr,
        weight_decay=1e-4,
    )
    best_state = {
        "encoder": copy.deepcopy(model.encoder.state_dict()),
        "fairINN": copy.deepcopy(model.fairINN.state_dict()),
        "classifier": copy.deepcopy(classifier.state_dict()),
        "z_b_label_adversary": copy.deepcopy(z_b_label_adversary.state_dict()),
    }
    best_tradeoff = -float("inf")
    warm_history = []
    warm_epochs = _cfg_int(cfg, "cf_warm_epochs", getattr(cfg, "pre_epochs", 250))
    for epoch in tqdm(range(warm_epochs), leave=False):
        model.train()
        classifier.train()
        z_b_label_adversary.train()
        optimizer.zero_grad()
        h = model.encoder(gw, features)
        z_u, z_b, pred_s, pred_y, _ = model.fairINN(h)
        logits = classifier(h) if _uses_h_prediction(cfg) else classifier(z_u)
        h_rec = cf_operator.inverse(z_u, z_b)

        l_cls = binary_task_loss(logits, labels, train_mask)
        warm_y_mode = getattr(cfg, "warm_y_loss", "mlp_bce")
        if warm_y_mode in ("mlp_bce", "linear_bce"):
            l_y = F.binary_cross_entropy_with_logits(pred_y[train_mask], labels[train_mask].float().view(-1, 1))
            l_y_bce = l_y
            l_y_supcon = torch.tensor(0.0, device=device)
        elif warm_y_mode == "supcon":
            l_y_bce = torch.tensor(0.0, device=device)
            l_y_supcon = supervised_contrastive_loss(
                z_u[train_mask],
                labels[train_mask],
                temperature=getattr(cfg, "contrast_temperature", 0.07),
                max_samples=getattr(cfg, "contrast_max_samples", 2048),
            )
            l_y = l_y_supcon
        elif warm_y_mode in ("balanced_supcon", "cross_s_supcon"):
            l_y_bce = torch.tensor(0.0, device=device)
            l_y_supcon = balanced_supervised_contrastive_loss(
                z_u[train_mask],
                labels[train_mask],
                sens[train_mask],
                temperature=getattr(cfg, "contrast_temperature", 0.07),
                max_samples=getattr(cfg, "contrast_max_samples", 2048),
                cross_sensitive_positive=(warm_y_mode == "cross_s_supcon"),
            )
            l_y = l_y_supcon
        elif warm_y_mode == "none":
            l_y_bce = torch.tensor(0.0, device=device)
            l_y_supcon = torch.tensor(0.0, device=device)
            l_y = torch.tensor(0.0, device=device)
        else:
            raise ValueError(f"Unknown warm_y_loss: {warm_y_mode}")
        sens_losses = _sensitive_proxy_loss(model, z_u, z_b, pred_s, labels, sens, train_mask, cfg)
        l_s_con = sens_losses["s_con"]
        l_s_ce = sens_losses["s_ce"]
        l_zu_con = supervised_contrastive_loss(
            z_u[train_mask],
            labels[train_mask],
            temperature=getattr(cfg, "contrast_temperature", 0.07),
            max_samples=getattr(cfg, "contrast_max_samples", 2048),
        )
        l_zb_y_adv = torch.tensor(0.0, device=device)
        if float(getattr(cfg, "lambda_zb_y_adv", 0.0)) > 0.0:
            adv_logits = z_b_label_adversary(_grad_reverse(z_b, scale=getattr(cfg, "lambda_zb_y_adv", 0.0)))
            l_zb_y_adv = F.binary_cross_entropy_with_logits(
                adv_logits[train_mask], labels[train_mask].float().view(-1, 1)
            )
        l_ind = model.fairINN.loss_fn.loss_hsic(z_u[train_mask], z_b[train_mask])
        l_orth = model.fairINN.loss_fn.loss_orth(z_u[train_mask], z_b[train_mask])
        stage1_regularizer = getattr(cfg, "cf_stage1_regularizer", "hsic")
        if stage1_regularizer == "orth":
            l_separation = float(getattr(cfg, "lambda_orth", 0.05)) * l_orth
        elif stage1_regularizer == "both":
            l_separation = (
                float(getattr(cfg, "lambda_ind", 1.0)) * l_ind
                + float(getattr(cfg, "lambda_orth", 0.05)) * l_orth
            )
        else:
            l_separation = float(getattr(cfg, "lambda_ind", 1.0)) * l_ind
        l_rec = F.mse_loss(h_rec, h)
        l_stage1_map, stage1_map_stats = cf_operator.stage1_map_loss(
            h,
            z_u,
            z_b,
            sens,
            mode=getattr(cfg, "stage1_map_loss", "none"),
            mask=train_mask,
        )
        loss = (
            float(getattr(cfg, "lambda_y", 1.0)) * l_y
            + float(getattr(cfg, "lambda_zu_con", 0.1)) * l_zu_con
            + float(getattr(cfg, "lambda_s", 1.0)) * l_s_con
            + float(getattr(cfg, "lambda_s_ce", 0.0)) * l_s_ce
            + l_zb_y_adv
            + l_separation
            + float(getattr(cfg, "lambda_stage1_map", 0.0)) * l_stage1_map
        )
        if _uses_h_prediction(cfg) and not getattr(cfg, "no_warm_cls", False):
            loss = loss + float(getattr(cfg, "lambda_y", 1.0)) * l_cls
        _nan_guard("warm loss", loss)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == warm_epochs - 1:
            model.eval()
            classifier.eval()
            with torch.no_grad():
                h_eval = model.encoder(gw, features)
                if _uses_h_prediction(cfg) and getattr(cfg, "no_warm_cls", False):
                    _, _, _, logits_eval, _ = model.fairINN(h_eval)
                else:
                    logits_eval, _, _ = _logits_from_h(model, classifier, h_eval, cfg)
                val_metric = metric.evaluate(logits_eval[valid_mask], labels[valid_mask], sens[valid_mask], logits_eval[valid_mask])
                if warm_y_mode == "supcon":
                    tradeoff = -float(loss.detach().item())
                else:
                    tradeoff = val_metric["auc_roc"] + val_metric["F1"] + val_metric["acc"] - 0.5 * (
                        val_metric["parity"] + val_metric["equality"]
                    )
                if tradeoff > best_tradeoff:
                    best_tradeoff = tradeoff
                    best_state = {
                        "encoder": copy.deepcopy(model.encoder.state_dict()),
                        "fairINN": copy.deepcopy(model.fairINN.state_dict()),
                        "classifier": copy.deepcopy(classifier.state_dict()),
                        "z_b_label_adversary": copy.deepcopy(z_b_label_adversary.state_dict()),
                    }
            warm_history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().item()),
                    "cls": float(l_cls.detach().item()),
                    "y_loss": float(l_y.detach().item()),
                    "y_ce": float(l_y_bce.detach().item()),
                    "y_supcon": float(l_y_supcon.detach().item()),
                    "zu_con": float(l_zu_con.detach().item()),
                    "zb_proxy": float(l_s_con.detach().item()),
                    "zb_ce": float(l_s_ce.detach().item()),
                    "zb_y_adv": float(l_zb_y_adv.detach().item()),
                    "ind_hsic": float(l_ind.detach().item()),
                    "orth": float(l_orth.detach().item()),
                    "rec": float(l_rec.detach().item()),
                    "stage1_map_loss": getattr(cfg, "stage1_map_loss", "none"),
                    **stage1_map_stats,
                    "val_auc": float(val_metric["auc_roc"]),
                    "val_acc": float(val_metric["acc"]),
                    "val_f1": float(val_metric["F1"]),
                    "val_dp": float(val_metric["parity"]),
                    "val_eo": float(val_metric["equality"]),
                    "selection_tradeoff": float(tradeoff),
                }
            )
            print(
                f"[Warm {epoch}] loss={loss.item():.4f} cls={l_cls.item():.4f} "
                f"Ly={l_y.item():.4f} LyCE={l_y_bce.item():.4f} LySup={l_y_supcon.item():.4f} "
                f"ZuCon={l_zu_con.item():.4f} "
                f"ZbCon={l_s_con.item():.4f} ZbCE={l_s_ce.item():.4f} "
                f"ZbYAdv={l_zb_y_adv.item():.4f} Lind={l_ind.item():.6f} "
                f"Lorth={l_orth.item():.6f} reg={stage1_regularizer} rec={l_rec.item():.6f} "
                f"Stage1Map={l_stage1_map.item():.6f}"
            )

    model.encoder.load_state_dict(best_state["encoder"])
    model.fairINN.load_state_dict(best_state["fairINN"])
    classifier.load_state_dict(best_state["classifier"])
    z_b_label_adversary.load_state_dict(best_state["z_b_label_adversary"])
    ckpt_dir = f"{cfg.ckpt_dir}/{cfg.dataset}/CfRisk/{cfg.encoder_type}/{cfg.seed}"
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.encoder.state_dict(), f"{ckpt_dir}/encoder.pt")
    torch.save(model.fairINN.state_dict(), f"{ckpt_dir}/fairINN.pt")
    torch.save(model.fairINN.classifier_y.state_dict(), f"{ckpt_dir}/pred_y_head.pt")

    model.eval()
    classifier.eval()
    with torch.no_grad():
        h_full = model.encoder(gw, features)
        z_u_full, z_b_full, _, pred_y_full, _ = model.fairINN(h_full)
        donor_label_scores = torch.sigmoid(pred_y_full).view(-1)
        rec_error = cf_operator.reconstruction_error(h_full)
        zs_h_geometry = cf_operator.zs_h_geometry_diagnostics(
            h_full,
            z_u_full,
            z_b_full,
            sens,
            mask=test_mask,
        )

    selector = DonorSelector(struct_beta=getattr(cfg, "struct_donor_beta", 0.1))
    donor_indices = selector.select(
        z_u=z_u_full.detach(),
        sensitive=sens,
        train_mask=train_mask,
        graph=gw,
        k=getattr(cfg, "num_donors", getattr(cfg, "donor_top_k", 5)),
        use_structure=getattr(cfg, "use_struct_donor", False) and not getattr(cfg, "no_struct_donor", False),
        labels=labels,
        label_scores=donor_label_scores,
        label_penalty=getattr(cfg, "donor_fallback_label_penalty", 0.0),
        label_score_max_diff=getattr(cfg, "donor_label_score_max_diff", None),
        require_same_label=getattr(cfg, "donor_require_same_label", False),
        chunk_size=getattr(cfg, "donor_select_chunk_size", 4096),
        confidence_score_tau=getattr(cfg, "donor_confidence_score_tau", 0.05),
        selection_mode=getattr(cfg, "donor_selection_mode", "distance"),
        label_rank_weight=getattr(cfg, "donor_label_rank_weight", 0.25),
    )
    donor_stats = selector.last_stats
    donor_reliability = selector.last_reliability

    with torch.no_grad():
        logits_full, logits_cf_set, risk_full, h_cf_set = _counterfactual_risk_snapshot(
            model, h_full, classifier, cf_operator, donor_indices, z_b_full.detach(), cfg
        )
        risk_weight = risk_to_weight(
            risk_full,
            tau=getattr(cfg, "risk_tau", 0.1),
            temperature=getattr(cfg, "risk_temperature", 0.05),
            no_risk_weight=getattr(cfg, "no_risk_weight", False),
        )
        risk_weight = _apply_donor_confidence_weight(risk_weight, donor_reliability, cfg)
        h_cf_mean = h_cf_set.mean(dim=1)
        repr_risk_full = _cf_representation_distance(h_full, h_cf_set).detach()
        repr_reduce_top_ratio = float(getattr(cfg, "repr_reduce_top_ratio", 0.0))
        repr_reduce_mask = None
        if repr_reduce_top_ratio > 0.0:
            repr_reduce_mask = torch.zeros(gw.num_nodes(), dtype=torch.bool, device=device)
            num_repr_reduce = min(
                gw.num_nodes(),
                max(1, int(round(gw.num_nodes() * repr_reduce_top_ratio))),
            )
            repr_reduce_mask[torch.topk(repr_risk_full, k=num_repr_reduce, largest=True).indices] = True
        if getattr(cfg, "explainer_scope", "local") == "all_nodes":
            fixed_anchor_mask = torch.ones(gw.num_nodes(), dtype=torch.bool, device=device)
            fixed_local_node_mask = fixed_anchor_mask.clone()
            fixed_local_edge_mask = torch.ones(gw.num_edges(), dtype=torch.bool, device=device)
        else:
            fixed_anchor_mask, fixed_local_node_mask, fixed_local_edge_mask = _select_local_region(
                gw,
                (risk_weight * risk_full).detach(),
                train_mask,
                top_k=getattr(cfg, "local_top_k", 512),
                hops=getattr(cfg, "local_hops", 2),
            )
        fixed_local_train_mask = fixed_local_node_mask & train_mask

    print(
        f"Full CF risk mean={risk_full.mean().item():.6f} "
        f"high_ratio={(risk_full > getattr(cfg, 'risk_tau', 0.1)).float().mean().item():.4f} "
        f"rec_error={rec_error.item():.8f}"
    )
    print("Donor stats:", donor_stats)
    print(
        "Fixed local explainer scope:",
        {
            "anchors": int(fixed_anchor_mask.sum().item()),
            "local_node_ratio": float(fixed_local_node_mask.float().mean().item()),
            "local_edge_ratio": float(fixed_local_edge_mask.float().mean().item()),
        },
    )

    if getattr(cfg, "freeze_inn_in_exp", True):
        _set_requires_grad(model.fairINN, False)
    if getattr(cfg, "freeze_encoder_in_exp", True):
        _set_requires_grad(model.encoder, False)
    predictor_y = classifier if _uses_h_prediction(cfg) else model.fairINN.classifier_y
    _set_requires_grad(predictor_y, True)

    if not _uses_h_prediction(cfg):
        classifier_ref = classifier
        classifier_exp = torch.nn.Linear(z_y_dim, 1).to(device)
        classifier_exp.load_state_dict(classifier_ref.state_dict())
        classifier = classifier_exp

    print("\n================ Phase 2: Self-explaining prediction learning ================")
    exp_params = (
        list(model.featureMask_Generator.parameters())
        + list(model.structureMask_Generator.parameters())
        + list(predictor_y.parameters())
    )
    exp_optimizer = torch.optim.Adam(exp_params, lr=getattr(cfg, "explainer_lr", cfg.lr), weight_decay=1e-4)
    best_exp_state = {
        "predictor_y": copy.deepcopy(predictor_y.state_dict()),
        "feature": copy.deepcopy(model.featureMask_Generator.state_dict()),
        "structure": copy.deepcopy(model.structureMask_Generator.state_dict()),
    }
    best_exp_score = float("inf")
    exp_history = []
    exp_epochs = _cfg_int(cfg, "cf_exp_epochs", getattr(cfg, "epochs", 150))
    phase2_bias_clean = _is_bias_clean_mode(cfg)
    scope_fn = _apply_fixed_bias_scope if phase2_bias_clean else _apply_fixed_local_scope
    for epoch in tqdm(range(exp_epochs), leave=False):
        model.train()
        classifier.train()
        exp_optimizer.zero_grad()

        h_cond = h_full.detach() if getattr(cfg, "freeze_encoder_in_exp", True) else model.encoder(gw, features)
        feature_mask, structure_mask, x_exp, is_self_loop = _build_dynamic_soft_masks(
            model,
            gw,
            features,
            h_cond,
            h_cf_mean,
            train_mask,
            sens_index,
            tau=getattr(cfg, "mask_tau", 1.0),
            cf_operator=cf_operator,
            donor_indices=donor_indices,
            z_b_bank=z_b_full.detach(),
            dynamic=not getattr(cfg, "static_exp_condition", False),
            stochastic_structure=bool(getattr(cfg, "gsat_stochastic_mask", False)),
            stochastic_temperature=getattr(cfg, "gsat_temperature", 0.5),
        )
        feature_mask, structure_mask = scope_fn(
            feature_mask, structure_mask, is_self_loop, fixed_local_node_mask, fixed_local_edge_mask
        )
        if getattr(cfg, "joint_mask_mode", "soft") == "straight_through_hard":
            feature_mask, structure_mask = _straight_through_hard_masks(
                feature_mask,
                structure_mask,
                is_self_loop,
                sens_index,
                feature_max,
                structure_max,
                ablation_mode=getattr(cfg, "mask_ablation_mode", "full"),
            )
            feature_mask, structure_mask = scope_fn(
                feature_mask, structure_mask, is_self_loop, fixed_local_node_mask, fixed_local_edge_mask
            )
        x_base = model.get_x_base(features, train_mask)
        x_exp = feature_mask * features + (1.0 - feature_mask) * x_base
        h_exp = model.encoder(gw, x_exp, edge_weight=structure_mask)
        logits_exp, z_u_exp, z_b_exp = _logits_from_h(model, classifier, h_exp, cfg)
        h_exp_cf_set = cf_operator.apply(h_exp, donor_indices, z_b_bank=z_b_full.detach())
        logits_exp_cf_set = _cf_logits(model, classifier, h_exp_cf_set, cfg)
        cf_risk_exp = compute_cf_risk(
            logits_exp,
            logits_exp_cf_set,
            mode=getattr(cfg, "risk_mode", "mean_logit_gap"),
            var_gamma=getattr(cfg, "risk_var_gamma", 0.0),
        )
        z_s_exp = _sensitive_exposure_from_z_s(model, z_b_exp)
        risk_exp = cf_risk_exp + float(getattr(cfg, "lambda_zs_fair", 1.0)) * z_s_exp
        logits_target = logits_exp
        h_target = h_exp
        h_target_cf_set = h_exp_cf_set
        risk_target = risk_exp
        z_s_target = z_s_exp
        cf_risk_target = cf_risk_exp
        risk_clean = risk_exp
        cf_risk_clean = cf_risk_exp
        z_s_clean = z_s_exp
        if phase2_bias_clean:
            feature_clean, structure_clean = model.get_complement_masks(feature_mask, structure_mask, is_self_loop)
            x_clean = feature_clean * features + (1.0 - feature_clean) * x_base
            h_clean = model.encoder(gw, x_clean, edge_weight=structure_clean)
            logits_clean, _, z_b_clean = _logits_from_h(model, classifier, h_clean, cfg)
            h_clean_cf_set = cf_operator.apply(h_clean, donor_indices, z_b_bank=z_b_full.detach())
            logits_clean_cf_set = _cf_logits(model, classifier, h_clean_cf_set, cfg)
            cf_risk_clean = compute_cf_risk(
                logits_clean,
                logits_clean_cf_set,
                mode=getattr(cfg, "risk_mode", "mean_logit_gap"),
                var_gamma=getattr(cfg, "risk_var_gamma", 0.0),
            )
            z_s_clean = _sensitive_exposure_from_z_s(model, z_b_clean)
            risk_clean = cf_risk_clean + float(getattr(cfg, "lambda_zs_fair", 1.0)) * z_s_clean
            logits_target = logits_clean
            h_target = h_clean
            h_target_cf_set = h_clean_cf_set
            risk_target = risk_clean
            z_s_target = z_s_clean
            cf_risk_target = cf_risk_clean
        anchor_mask = fixed_anchor_mask
        local_node_mask = fixed_local_node_mask
        local_edge_mask = fixed_local_edge_mask
        local_train_mask = fixed_local_train_mask

        l_reduce = torch.tensor(0.0, device=device)
        if not getattr(cfg, "no_reduce_loss", False):
            reduce_source = getattr(cfg, "reduce_loss_type", "prediction")
            reduce_exp = risk_target
            reduce_full = risk_full
            if reduce_source == "repr":
                reduce_exp = _cf_representation_distance(h_target, h_target_cf_set)
                reduce_full = repr_risk_full
            reduce_mask = anchor_mask
            reduce_weight = risk_weight
            if reduce_source == "repr" and repr_reduce_mask is not None:
                reduce_mask = repr_reduce_mask
                reduce_weight = torch.ones_like(risk_weight)
            l_reduce = _masked_weighted_reduce_loss(
                reduce_exp,
                reduce_full,
                reduce_weight,
                reduce_mask,
                rho=getattr(cfg, "rho_reduce", 0.5),
            )
        l_u = torch.tensor(0.0, device=device)
        if not getattr(cfg, "no_u_suf", False):
            l_u = _prediction_sufficiency_loss(logits_target, labels, local_train_mask)
        l_h_inv = _cf_representation_invariance_loss(h_target, h_target_cf_set, weights=risk_weight, mask=anchor_mask)
        l_zs_fair = _masked_weighted_mean(z_s_target, risk_weight, anchor_mask)
        l_pred = binary_task_loss(logits_target, labels, local_train_mask)
        l_bias_margin = torch.tensor(0.0, device=device)
        if phase2_bias_clean and float(getattr(cfg, "lambda_bias_margin", 0.0)) > 0.0:
            margin = float(getattr(cfg, "bias_margin", 0.0))
            margin_values = F.relu(margin - (risk_exp - risk_clean))
            l_bias_margin = _masked_weighted_mean(margin_values, risk_weight, anchor_mask)
        l_sp = _local_mask_sparsity_loss(
            feature_mask, structure_mask, is_self_loop, sens_index, local_node_mask, local_edge_mask
        )
        local_feature_mean, local_structure_mean = _local_mask_means(
            feature_mask, structure_mask, is_self_loop, sens_index, local_node_mask, local_edge_mask
        )
        l_gsat_entropy = _local_mask_entropy_loss(structure_mask, is_self_loop, local_edge_mask)
        l_budget = _local_mask_budget_loss(
            feature_mask,
            structure_mask,
            is_self_loop,
            sens_index,
            local_node_mask,
            local_edge_mask,
            feature_budget=getattr(cfg, "mask_feature_budget", None),
            structure_budget=getattr(cfg, "mask_structure_budget", None),
        )
        l_keep = _nonlocal_keep_loss(h_target, h_full.detach(), local_node_mask)
        loss = (
            float(getattr(cfg, "lambda_reduce", 1.0)) * l_reduce
            + float(getattr(cfg, "lambda_u_suf", 1.0)) * l_u
            + float(getattr(cfg, "lambda_h_cf_inv", 0.1)) * l_h_inv
            + float(getattr(cfg, "lambda_zs_fair", 1.0)) * l_zs_fair
            + float(getattr(cfg, "lambda_pred_exp", 1.0)) * l_pred
            + float(getattr(cfg, "lambda_bias_margin", 0.0)) * l_bias_margin
            + float(getattr(cfg, "lambda_sp", 0.01)) * l_sp
            + float(getattr(cfg, "lambda_gsat_entropy", 0.0)) * l_gsat_entropy
            + float(getattr(cfg, "lambda_mask_budget", 0.0)) * l_budget
            + float(getattr(cfg, "lambda_local_keep", 1.0)) * l_keep
        )
        _nan_guard("explanation loss", loss)
        loss.backward()
        exp_optimizer.step()

        with torch.no_grad():
            repr_exp = _cf_representation_distance(h_target, h_target_cf_set)
            exp_history.append(
                {
                    "epoch": int(epoch),
                    "loss": float(loss.detach().item()),
                    "reduce": float(l_reduce.detach().item()),
                    "u": float(l_u.detach().item()),
                    "h_inv": float(l_h_inv.detach().item()),
                    "z_s_fair": float(l_zs_fair.detach().item()),
                    "pred": float(l_pred.detach().item()),
                    "bias_margin": float(l_bias_margin.detach().item()),
                    "sp": float(l_sp.detach().item()),
                    "gsat_entropy": float(l_gsat_entropy.detach().item()),
                    "budget": float(l_budget.detach().item()),
                    "keep": float(l_keep.detach().item()),
                    "repr_exp_anchor": float(repr_exp[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "repr_full_anchor": float(repr_risk_full[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "repr_margin_anchor": (
                        float(
                            (
                                repr_exp[anchor_mask]
                                - float(getattr(cfg, "rho_reduce", 0.5)) * repr_risk_full[anchor_mask]
                            ).mean().item()
                        )
                        if anchor_mask.any()
                        else 0.0
                    ),
                    "repr_reduce_nodes": int(
                        repr_reduce_mask.sum().item() if repr_reduce_mask is not None else anchor_mask.sum().item()
                    ),
                    "cf_risk_anchor": float(cf_risk_target[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "risk_exp_anchor": float(risk_target[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "risk_bias_anchor": float(risk_exp[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "risk_clean_anchor": float(risk_clean[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "risk_bias_minus_clean_anchor": (
                        float((risk_exp[anchor_mask] - risk_clean[anchor_mask]).mean().item())
                        if anchor_mask.any()
                        else 0.0
                    ),
                    "z_s_exp_anchor": float(z_s_target[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "z_s_bias_anchor": float(z_s_exp[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "z_s_clean_anchor": float(z_s_clean[anchor_mask].mean().item()) if anchor_mask.any() else 0.0,
                    "local_feature_keep": float(local_feature_mean.detach().item()),
                    "local_structure_keep": float(local_structure_mean.detach().item()),
                }
            )

        score = (
            float(getattr(cfg, "lambda_reduce", 1.0)) * l_reduce
            + 0.2 * float(getattr(cfg, "lambda_u_suf", 1.0)) * l_u
            + float(getattr(cfg, "lambda_h_cf_inv", 0.1)) * l_h_inv
            + float(getattr(cfg, "lambda_zs_fair", 1.0)) * l_zs_fair
            + float(getattr(cfg, "lambda_bias_margin", 0.0)) * l_bias_margin
            + float(getattr(cfg, "lambda_sp", 0.01)) * l_sp
            + float(getattr(cfg, "lambda_gsat_entropy", 0.0)) * l_gsat_entropy
            + float(getattr(cfg, "lambda_mask_budget", 0.0)) * l_budget
        ).detach().item()
        if score < best_exp_score:
            best_exp_score = score
            best_exp_state = {
                "predictor_y": copy.deepcopy(predictor_y.state_dict()),
                "feature": copy.deepcopy(model.featureMask_Generator.state_dict()),
                "structure": copy.deepcopy(model.structureMask_Generator.state_dict()),
            }
        if epoch % 10 == 0 or epoch == exp_epochs - 1:
            print(
                f"[Exp {epoch}] loss={loss.item():.4f} reduce={l_reduce.item():.4f} "
                f"u={l_u.item():.4f} hinv={l_h_inv.item():.4f} zs={l_zs_fair.item():.4f} pred={l_pred.item():.4f} "
                f"bmargin={l_bias_margin.item():.4f} "
                f"sp={l_sp.item():.4f} ent={l_gsat_entropy.item():.4f} budget={l_budget.item():.4f} keep={l_keep.item():.4f} "
                f"risk_target={risk_target[anchor_mask].mean().item():.6f} cf={cf_risk_target[anchor_mask].mean().item():.6f} "
                f"risk_bias={risk_exp[anchor_mask].mean().item():.6f} risk_clean={risk_clean[anchor_mask].mean().item():.6f} "
                f"anchors={int(anchor_mask.sum().item())}"
            )

    predictor_y.load_state_dict(best_exp_state["predictor_y"])
    model.featureMask_Generator.load_state_dict(best_exp_state["feature"])
    model.structureMask_Generator.load_state_dict(best_exp_state["structure"])

    print("\n================ Phase 3: Explanation-driven joint refinement ================")
    phase3_update_mode = getattr(cfg, "phase3_update_mode", "joint")
    _set_requires_grad(model.fairINN, False)
    _set_requires_grad(predictor_y, True)
    _set_requires_grad(model.featureMask_Generator, phase3_update_mode == "joint")
    _set_requires_grad(model.structureMask_Generator, phase3_update_mode == "joint")
    if not getattr(cfg, "freeze_encoder_in_exp", True):
        _set_requires_grad(model.encoder, True)
    joint_params = list(predictor_y.parameters())
    if phase3_update_mode == "joint":
        joint_params += list(model.featureMask_Generator.parameters()) + list(model.structureMask_Generator.parameters())
    if not getattr(cfg, "freeze_encoder_in_exp", True):
        joint_params += list(model.encoder.parameters())
    joint_optimizer = torch.optim.Adam(joint_params, lr=getattr(cfg, "explainer_lr", cfg.lr), weight_decay=1e-4)
    best_joint_state = {
        "predictor_y": copy.deepcopy(predictor_y.state_dict()),
        "feature": copy.deepcopy(model.featureMask_Generator.state_dict()),
        "structure": copy.deepcopy(model.structureMask_Generator.state_dict()),
    }
    best_tradeoff = -float("inf")
    best_fair_score = float("inf")
    best_joint_gate_satisfied = False
    best_joint_epoch = -1
    best_joint_val = None
    joint_patience = max(0, int(getattr(cfg, "joint_early_stop_patience", 0)))
    joint_min_delta = max(0.0, float(getattr(cfg, "joint_early_stop_min_delta", 1e-4)))
    joint_checks_without_improvement = 0
    joint_stopped_early = False
    joint_stop_epoch = None
    joint_stop_reason = None
    joint_epochs = _cfg_int(cfg, "cf_joint_epochs", max(1, getattr(cfg, "epochs", 150) // 2))
    for epoch in tqdm(range(joint_epochs), leave=False):
        model.train()
        classifier.train()
        joint_optimizer.zero_grad()

        h_cond = h_full.detach() if getattr(cfg, "freeze_encoder_in_exp", True) else model.encoder(gw, features)
        feature_mask, structure_mask, x_exp, is_self_loop = _build_dynamic_soft_masks(
            model,
            gw,
            features,
            h_cond,
            h_cf_mean,
            train_mask,
            sens_index,
            tau=getattr(cfg, "mask_tau", 1.0),
            cf_operator=cf_operator,
            donor_indices=donor_indices,
            z_b_bank=z_b_full.detach(),
            dynamic=not getattr(cfg, "static_exp_condition", False),
            stochastic_structure=bool(getattr(cfg, "gsat_stochastic_mask", False)),
            stochastic_temperature=getattr(cfg, "gsat_temperature", 0.5),
        )
        feature_mask, structure_mask = scope_fn(
            feature_mask, structure_mask, is_self_loop, fixed_local_node_mask, fixed_local_edge_mask
        )
        x_base = model.get_x_base(features, train_mask)
        x_exp = feature_mask * features + (1.0 - feature_mask) * x_base
        h_exp = model.encoder(gw, x_exp, edge_weight=structure_mask)
        logits_exp, z_u_exp, z_b_exp = _logits_from_h(model, classifier, h_exp, cfg)
        h_exp_cf_set = cf_operator.apply(h_exp, donor_indices, z_b_bank=z_b_full.detach())
        logits_exp_cf_set = _cf_logits(model, classifier, h_exp_cf_set, cfg)
        cf_risk_exp = compute_cf_risk(
            logits_exp,
            logits_exp_cf_set,
            mode=getattr(cfg, "risk_mode", "mean_logit_gap"),
            var_gamma=getattr(cfg, "risk_var_gamma", 0.0),
        )
        z_s_exp = _sensitive_exposure_from_z_s(model, z_b_exp)
        risk_exp = cf_risk_exp + float(getattr(cfg, "lambda_zs_fair", 1.0)) * z_s_exp
        logits_target = logits_exp
        logits_target_cf_set = logits_exp_cf_set
        h_target = h_exp
        h_target_cf_set = h_exp_cf_set
        risk_target = risk_exp
        z_s_target = z_s_exp
        risk_clean = risk_exp
        if phase2_bias_clean:
            feature_clean, structure_clean = model.get_complement_masks(feature_mask, structure_mask, is_self_loop)
            x_clean = feature_clean * features + (1.0 - feature_clean) * x_base
            h_clean = model.encoder(gw, x_clean, edge_weight=structure_clean)
            logits_clean, _, z_b_clean = _logits_from_h(model, classifier, h_clean, cfg)
            h_clean_cf_set = cf_operator.apply(h_clean, donor_indices, z_b_bank=z_b_full.detach())
            logits_clean_cf_set = _cf_logits(model, classifier, h_clean_cf_set, cfg)
            cf_risk_clean = compute_cf_risk(
                logits_clean,
                logits_clean_cf_set,
                mode=getattr(cfg, "risk_mode", "mean_logit_gap"),
                var_gamma=getattr(cfg, "risk_var_gamma", 0.0),
            )
            z_s_clean = _sensitive_exposure_from_z_s(model, z_b_clean)
            risk_clean = cf_risk_clean + float(getattr(cfg, "lambda_zs_fair", 1.0)) * z_s_clean
            logits_target = logits_clean
            logits_target_cf_set = logits_clean_cf_set
            h_target = h_clean
            h_target_cf_set = h_clean_cf_set
            risk_target = risk_clean
            z_s_target = z_s_clean
        anchor_mask = fixed_anchor_mask
        local_node_mask = fixed_local_node_mask
        local_edge_mask = fixed_local_edge_mask
        local_train_mask = fixed_local_train_mask
        l_pred = binary_task_loss(logits_target, labels, local_train_mask)
        l_cf = _masked_weighted_mean(risk_target, risk_weight, anchor_mask)
        l_u = torch.tensor(0.0, device=device)
        if not getattr(cfg, "no_u_suf", False):
            l_u = _prediction_sufficiency_loss(logits_target, labels, local_train_mask)
        l_h_inv = _cf_representation_invariance_loss(h_target, h_target_cf_set, weights=risk_weight, mask=anchor_mask)
        l_zs_fair = _masked_weighted_mean(z_s_target, risk_weight, anchor_mask)
        l_bias_margin = torch.tensor(0.0, device=device)
        if phase2_bias_clean and float(getattr(cfg, "lambda_bias_margin", 0.0)) > 0.0:
            margin = float(getattr(cfg, "bias_margin", 0.0))
            margin_values = F.relu(margin - (risk_exp - risk_clean))
            l_bias_margin = _masked_weighted_mean(margin_values, risk_weight, anchor_mask)
        l_sp = _local_mask_sparsity_loss(
            feature_mask, structure_mask, is_self_loop, sens_index, local_node_mask, local_edge_mask
        )
        local_feature_mean, local_structure_mean = _local_mask_means(
            feature_mask, structure_mask, is_self_loop, sens_index, local_node_mask, local_edge_mask
        )
        l_gsat_entropy = _local_mask_entropy_loss(structure_mask, is_self_loop, local_edge_mask)
        l_budget = _local_mask_budget_loss(
            feature_mask,
            structure_mask,
            is_self_loop,
            sens_index,
            local_node_mask,
            local_edge_mask,
            feature_budget=getattr(cfg, "mask_feature_budget", None),
            structure_budget=getattr(cfg, "mask_structure_budget", None),
        )
        l_keep = _nonlocal_keep_loss(h_target, h_full.detach(), local_node_mask)
        loss = (
            l_pred
            + float(getattr(cfg, "lambda_cf", getattr(cfg, "lambda_reduce", 1.0))) * l_cf
            + float(getattr(cfg, "lambda_u_suf", 1.0)) * l_u
            + float(getattr(cfg, "lambda_h_cf_inv", 0.1)) * l_h_inv
            + float(getattr(cfg, "lambda_zs_fair", 1.0)) * l_zs_fair
            + float(getattr(cfg, "lambda_bias_margin", 0.0)) * l_bias_margin
            + float(getattr(cfg, "lambda_sp", 0.01)) * l_sp
            + float(getattr(cfg, "lambda_gsat_entropy", 0.0)) * l_gsat_entropy
            + float(getattr(cfg, "lambda_mask_budget", 0.0)) * l_budget
            + float(getattr(cfg, "lambda_local_keep", 1.0)) * l_keep
        )
        _nan_guard("joint loss", loss)
        loss.backward()
        joint_optimizer.step()

        if epoch % 5 == 0 or epoch == joint_epochs - 1:
            model.eval()
            classifier.eval()
            with torch.no_grad():
                selection_mode = getattr(cfg, "checkpoint_selection_mode", "soft")
                if selection_mode == "hard":
                    val_metric = _hard_explanation_selection_metrics(
                        model,
                        gw,
                        features,
                        labels,
                        sens,
                        valid_mask,
                        train_mask,
                        classifier,
                        cf_operator,
                        donor_indices,
                        sens_index,
                        feature_max,
                        structure_max,
                        fixed_local_node_mask,
                        fixed_local_edge_mask,
                        cfg,
                    )
                else:
                    val_metric = _calibration_selection_metrics(
                        logits_target, logits_target_cf_set, labels, sens, valid_mask
                    )
                    if phase2_bias_clean:
                        bias_metric = _calibration_selection_metrics(
                            logits_exp, logits_exp_cf_set, labels, sens, valid_mask
                        )
                        val_metric["bias_auc_roc"] = float(bias_metric["auc_roc"])
                        val_metric["bias_cf"] = float(bias_metric["cf"])
                        val_metric["clean_auc_roc"] = float(val_metric["auc_roc"])
                        val_metric["clean_cf"] = float(val_metric["cf"])
                        val_metric["comp_auc_roc"] = float(bias_metric["auc_roc"])
                        val_metric["comp_cf"] = float(bias_metric["cf"])
                        val_metric["exp_comp_auc_gap"] = float(val_metric["auc_roc"] - bias_metric["auc_roc"])
                    else:
                        val_metric["comp_auc_roc"] = 0.0
                        val_metric["comp_cf"] = 0.0
                        val_metric["exp_comp_auc_gap"] = 0.0
                    val_metric["hard_feature_keep"] = float(local_feature_mean.detach().item())
                    val_metric["hard_structure_keep"] = float(local_structure_mean.detach().item())
                soft_min_auc = getattr(cfg, "soft_select_min_auc", None)
                if soft_min_auc is None:
                    soft_min_auc = getattr(cfg, "selection_min_auc", None)
                soft_auc_floor_active = selection_mode == "soft" and soft_min_auc is not None
                soft_auc_satisfied = (not soft_auc_floor_active) or (
                    val_metric["auc_roc"] >= float(soft_min_auc)
                )
                min_hard_auc = getattr(cfg, "hard_select_min_auc", None)
                if min_hard_auc is None:
                    min_hard_auc = getattr(cfg, "selection_min_auc", 0.55) or 0.55
                min_hard_margin = float(getattr(cfg, "hard_select_min_margin", 0.0))
                gate_satisfied = (
                    selection_mode == "hard"
                    and
                    val_metric["auc_roc"] >= float(min_hard_auc)
                    and val_metric["exp_comp_auc_gap"] >= min_hard_margin
                )
                if getattr(cfg, "fair_early_stop", False):
                    fair_score = _fair_checkpoint_selection_score(cfg, val_metric)
                    passes_fair_floor = val_metric["auc_roc"] >= float(getattr(cfg, "fair_stop_min_auc", 0.0))
                    should_update = soft_auc_satisfied and passes_fair_floor and (
                        (gate_satisfied and not best_joint_gate_satisfied)
                        or (
                            gate_satisfied == best_joint_gate_satisfied
                            and fair_score < best_fair_score - joint_min_delta
                        )
                    )
                    if should_update:
                        best_fair_score = fair_score
                else:
                    if selection_mode == "soft":
                        tradeoff = val_metric["auc_roc"] + val_metric["F1"] + val_metric["acc"] - 2.0 * (
                            val_metric["parity"] + val_metric["equality"] + val_metric["cf"]
                        )
                    elif gate_satisfied:
                        tradeoff = (
                            val_metric["auc_roc"]
                            + val_metric["F1"]
                            + val_metric["acc"]
                            + val_metric["exp_comp_auc_gap"]
                            - 2.0 * (val_metric["parity"] + val_metric["equality"] + val_metric["cf"])
                        )
                    else:
                        tradeoff = (
                            val_metric["auc_roc"]
                            + val_metric["F1"]
                            + val_metric["acc"]
                            + 0.5 * val_metric["exp_comp_auc_gap"]
                            - 0.5 * (val_metric["parity"] + val_metric["equality"])
                            - 0.25 * val_metric["cf"]
                        )
                    should_update = soft_auc_satisfied and (
                        (gate_satisfied and not best_joint_gate_satisfied)
                        or (
                            gate_satisfied == best_joint_gate_satisfied
                            and tradeoff > best_tradeoff + joint_min_delta
                        )
                    )
                    if should_update:
                        best_tradeoff = tradeoff
                if should_update:
                    best_joint_gate_satisfied = bool(gate_satisfied)
                    best_joint_epoch = int(epoch)
                    best_joint_val = {k: to_float(v) for k, v in val_metric.items()}
                    best_joint_state = {
                        "predictor_y": copy.deepcopy(predictor_y.state_dict()),
                        "feature": copy.deepcopy(model.featureMask_Generator.state_dict()),
                        "structure": copy.deepcopy(model.structureMask_Generator.state_dict()),
                    }
                    joint_checks_without_improvement = 0
                else:
                    joint_checks_without_improvement += 1
            print(
                f"[Joint {epoch}] loss={loss.item():.4f} pred={l_pred.item():.4f} "
                f"cf={l_cf.item():.4f} u={l_u.item():.4f} hinv={l_h_inv.item():.4f} zs={l_zs_fair.item():.4f} "
                f"bmargin={l_bias_margin.item():.4f} sp={l_sp.item():.4f} ent={l_gsat_entropy.item():.4f} "
                f"budget={l_budget.item():.4f} keep={l_keep.item():.4f} "
                f"anchors={int(anchor_mask.sum().item())} sel={selection_mode} val_auc={val_metric['auc_roc']:.4f} "
                f"val_dp={val_metric['parity']:.4f} val_eo={val_metric['equality']:.4f} "
                f"val_cf={val_metric['cf']:.4f} hard_feat={val_metric['hard_feature_keep']:.4f} "
                f"hard_struct={val_metric['hard_structure_keep']:.4f} comp_auc={val_metric['comp_auc_roc']:.4f} "
                f"auc_gap={val_metric['exp_comp_auc_gap']:.4f} gate={int(gate_satisfied)} "
                f"soft_auc_ok={int(soft_auc_satisfied)}"
            )
            if joint_patience > 0 and joint_checks_without_improvement >= joint_patience:
                joint_stopped_early = True
                joint_stop_epoch = int(epoch)
                joint_stop_reason = f"no checkpoint improvement for {joint_patience} validation checks"
                print(f"[Joint early stop] epoch={epoch} best_epoch={best_joint_epoch} reason={joint_stop_reason}")
                break

    predictor_y.load_state_dict(best_joint_state["predictor_y"])
    model.featureMask_Generator.load_state_dict(best_joint_state["feature"])
    model.structureMask_Generator.load_state_dict(best_joint_state["structure"])

    calibration_diagnostics = {
        "enabled": bool(getattr(cfg, "cf_group_calibration", False)),
        "best_epoch": None,
        "best_score": None,
        "pre_val": None,
        "best_val": None,
        "auc_floor": None,
    }

    if getattr(cfg, "cf_group_calibration", False):
        print("\n================ Phase 4: CF-guided group calibration ================")
        model.eval()
        classifier.eval()
        with torch.no_grad():
            h_full_calib = model.encoder(gw, features)
            z_u_calib, z_b_calib = cf_operator.encode(h_full_calib)
            logits_full_calib, logits_cf_calib, risk_full_calib, h_cf_calib = _counterfactual_risk_snapshot(
                model, h_full_calib, classifier, cf_operator, donor_indices, z_b_calib.detach(), cfg
            )
            risk_weight_calib = risk_to_weight(
                risk_full_calib,
                tau=getattr(cfg, "risk_tau", 0.1),
                temperature=getattr(cfg, "risk_temperature", 0.05),
                no_risk_weight=getattr(cfg, "no_risk_weight", False),
            )
            risk_weight_calib = _apply_donor_confidence_weight(risk_weight_calib, donor_reliability, cfg)
            feature_soft, structure_soft, is_self_loop = None, None, None
            feature_soft, structure_soft, _, is_self_loop = _build_dynamic_soft_masks(
                model,
                gw,
                features,
                h_full_calib,
                h_cf_calib.mean(dim=1),
                train_mask,
                sens_index,
                tau=getattr(cfg, "mask_tau", 1.0),
                cf_operator=cf_operator,
                donor_indices=donor_indices,
                z_b_bank=z_b_calib.detach(),
                dynamic=not getattr(cfg, "static_exp_condition", False),
            )
            feature_soft, structure_soft = scope_fn(
                feature_soft, structure_soft, is_self_loop, fixed_local_node_mask, fixed_local_edge_mask
            )
            feature_mask_calib, structure_mask_calib = _hard_masks_from_soft(
                feature_soft,
                structure_soft,
                is_self_loop,
                sens_index,
                feature_max,
                structure_max,
                ablation_mode=getattr(cfg, "mask_ablation_mode", "full"),
            )
            feature_mask_calib, structure_mask_calib = scope_fn(
                feature_mask_calib, structure_mask_calib, is_self_loop, fixed_local_node_mask, fixed_local_edge_mask
            )
            exp_graph_calib, exp_feature_calib, _, _, _ = model.get_explained_graph(
                gw, features, feature_mask_calib, structure_mask_calib, train_mask
            )
            feature_comp_calib, structure_comp_calib = model.get_complement_masks(
                feature_mask_calib, structure_mask_calib, is_self_loop
            )
            comp_graph_calib, comp_feature_calib, _, _, _ = model.get_explained_graph(
                gw, features, feature_comp_calib, structure_comp_calib, train_mask
            )
            h_exp_calib = model.encoder(exp_graph_calib, exp_feature_calib).detach()
            h_exp_cf_calib = cf_operator.apply(h_exp_calib, donor_indices, z_b_bank=z_b_calib.detach()).detach()
            h_comp_calib = model.encoder(comp_graph_calib, comp_feature_calib).detach()
            h_comp_cf_calib = cf_operator.apply(h_comp_calib, donor_indices, z_b_bank=z_b_calib.detach()).detach()
            h_teacher_calib = h_comp_calib if phase2_bias_clean else h_exp_calib
            h_teacher_cf_calib = h_comp_cf_calib if phase2_bias_clean else h_exp_cf_calib
            h_other_calib = h_exp_calib if phase2_bias_clean else h_comp_calib
            h_other_cf_calib = h_exp_cf_calib if phase2_bias_clean else h_comp_cf_calib
            logits_teacher, _, _ = _logits_from_h(model, classifier, h_teacher_calib, cfg)
            logits_teacher = logits_teacher.detach()
            logits_cf_teacher = _cf_logits(model, classifier, h_teacher_cf_calib, cfg).detach()
            pre_calib_metric = _calibration_selection_metrics(
                logits_teacher, logits_cf_teacher, labels, sens, valid_mask
            )
            logits_comp_pre, _, _ = _logits_from_h(model, classifier, h_other_calib, cfg)
            logits_comp_cf_pre = _cf_logits(model, classifier, h_other_cf_calib, cfg)
            comp_pre_metric = _calibration_selection_metrics(
                logits_comp_pre, logits_comp_cf_pre, labels, sens, valid_mask
            )
            pre_calib_metric = _attach_comp_metrics(pre_calib_metric, comp_pre_metric)

        _set_requires_grad(model.fairINN, False)
        _set_requires_grad(predictor_y, True)
        calib_optimizer = torch.optim.Adam(predictor_y.parameters(), lr=getattr(cfg, "calib_lr", cfg.lr), weight_decay=1e-4)
        calib_epochs = _cfg_int(cfg, "calib_epochs", 50)
        auc_floor = max(
            float(getattr(cfg, "calib_min_auc", 0.0)),
            float(pre_calib_metric["auc_roc"]) - float(getattr(cfg, "calib_auc_drop_tol", 0.01)),
        )
        best_calib_state = copy.deepcopy(predictor_y.state_dict())
        best_calib_metric = dict(pre_calib_metric)
        best_calib_epoch = -1
        best_calib_score = _calibration_selection_score(cfg, pre_calib_metric)
        best_calib_guard_satisfied = bool(
            pre_calib_metric["auc_roc"] >= float(getattr(cfg, "calib_guard_min_auc", 0.55))
            and pre_calib_metric["exp_comp_auc_gap"] >= float(getattr(cfg, "calib_guard_min_margin", 0.0))
        )
        print(
            f"[Calib pre] val_auc={pre_calib_metric['auc_roc']:.4f} "
            f"val_dp={pre_calib_metric['parity']:.4f} val_eo={pre_calib_metric['equality']:.4f} "
            f"val_cf={pre_calib_metric['cf']:.4f} val_logit={pre_calib_metric['cf_logit_gap']:.4f} "
            f"comp_auc={pre_calib_metric['comp_auc_roc']:.4f} "
            f"auc_gap={pre_calib_metric['exp_comp_auc_gap']:.4f} auc_floor={auc_floor:.4f}"
        )
        calibration_diagnostics.update(
            {
                "pre_val": {k: to_float(v) for k, v in pre_calib_metric.items()},
                "auc_floor": float(auc_floor),
                "hard_guard": bool(getattr(cfg, "calib_hard_guard", False)),
                "guard_min_auc": float(getattr(cfg, "calib_guard_min_auc", 0.55)),
                "guard_min_margin": float(getattr(cfg, "calib_guard_min_margin", 0.0)),
            }
        )
        for epoch in tqdm(range(calib_epochs), leave=False):
            predictor_y.train()
            calib_optimizer.zero_grad()
            logits_calib, _, z_b_exp_calib = _logits_from_h(model, classifier, h_teacher_calib, cfg)
            logits_cf_set_calib = _cf_logits(model, classifier, h_teacher_cf_calib, cfg)
            cf_risk_calib = compute_cf_risk(
                logits_calib,
                logits_cf_set_calib,
                mode=getattr(cfg, "risk_mode", "mean_logit_gap"),
                var_gamma=getattr(cfg, "risk_var_gamma", 0.0),
            )
            z_s_calib = _sensitive_exposure_from_z_s(model, z_b_exp_calib)
            risk_calib = cf_risk_calib + float(getattr(cfg, "lambda_zs_fair", 1.0)) * z_s_calib
            l_task_calib = binary_task_loss(logits_calib, labels, train_mask)
            l_ind_calib = _masked_weighted_mean(risk_calib, risk_weight_calib, train_mask)
            l_group_calib = _cf_group_calibration_loss(
                logits_calib, logits_cf_set_calib.mean(dim=1), sens, train_mask
            )
            l_pair_calib = _cf_pair_calibration_loss(
                logits_calib, logits_cf_set_calib, risk_weight_calib, train_mask
            )
            l_distill_calib = F.mse_loss(logits_calib[train_mask], logits_teacher[train_mask])
            loss_calib = (
                l_task_calib
                + float(getattr(cfg, "lambda_calib_ind", 0.5)) * l_ind_calib
                + float(getattr(cfg, "lambda_calib_group", 0.5)) * l_group_calib
                + float(getattr(cfg, "lambda_calib_pair", 0.0)) * l_pair_calib
                + float(getattr(cfg, "lambda_calib_distill", 0.1)) * l_distill_calib
            )
            _nan_guard("calibration loss", loss_calib)
            loss_calib.backward()
            calib_optimizer.step()
            with torch.no_grad():
                predictor_y.eval()
                logits_val_calib, _, _ = _logits_from_h(model, classifier, h_teacher_calib, cfg)
                logits_cf_val_calib = _cf_logits(model, classifier, h_teacher_cf_calib, cfg)
                val_calib_metric = _calibration_selection_metrics(
                    logits_val_calib, logits_cf_val_calib, labels, sens, valid_mask
                )
                logits_comp_val_calib, _, _ = _logits_from_h(model, classifier, h_other_calib, cfg)
                logits_comp_cf_val_calib = _cf_logits(model, classifier, h_other_cf_calib, cfg)
                comp_val_calib_metric = _calibration_selection_metrics(
                    logits_comp_val_calib, logits_comp_cf_val_calib, labels, sens, valid_mask
                )
                val_calib_metric = _attach_comp_metrics(val_calib_metric, comp_val_calib_metric)
                val_score = _calibration_selection_score(cfg, val_calib_metric)
                guard_satisfied = bool(
                    val_calib_metric["auc_roc"] >= float(getattr(cfg, "calib_guard_min_auc", 0.55))
                    and val_calib_metric["exp_comp_auc_gap"] >= float(getattr(cfg, "calib_guard_min_margin", 0.0))
                )
                if getattr(cfg, "calib_hard_guard", False):
                    should_update_calib = (
                        val_calib_metric["auc_roc"] >= auc_floor
                        and guard_satisfied
                        and ((not best_calib_guard_satisfied) or val_score < best_calib_score)
                    )
                else:
                    should_update_calib = val_calib_metric["auc_roc"] >= auc_floor and val_score < best_calib_score
                if should_update_calib:
                    best_calib_score = val_score
                    best_calib_epoch = int(epoch)
                    best_calib_metric = dict(val_calib_metric)
                    best_calib_guard_satisfied = bool(guard_satisfied)
                    best_calib_state = copy.deepcopy(predictor_y.state_dict())
            if epoch % 10 == 0 or epoch == calib_epochs - 1:
                print(
                    f"[Calib {epoch}] loss={loss_calib.item():.4f} task={l_task_calib.item():.4f} "
                    f"ind={l_ind_calib.item():.4f} group={l_group_calib.item():.4f} "
                    f"pair={l_pair_calib.item():.4f} distill={l_distill_calib.item():.4f} "
                    f"val_auc={val_calib_metric['auc_roc']:.4f} val_dp={val_calib_metric['parity']:.4f} "
                    f"val_eo={val_calib_metric['equality']:.4f} val_cf={val_calib_metric['cf']:.4f} "
                    f"comp_auc={val_calib_metric['comp_auc_roc']:.4f} "
                    f"auc_gap={val_calib_metric['exp_comp_auc_gap']:.4f} gate={int(guard_satisfied)}"
                )
        predictor_y.load_state_dict(best_calib_state)
        print(
            f"[Calib best] epoch={best_calib_epoch} score={best_calib_score:.4f} "
            f"val_auc={best_calib_metric['auc_roc']:.4f} val_dp={best_calib_metric['parity']:.4f} "
            f"val_eo={best_calib_metric['equality']:.4f} val_cf={best_calib_metric['cf']:.4f} "
            f"val_logit={best_calib_metric['cf_logit_gap']:.4f} "
            f"comp_auc={best_calib_metric['comp_auc_roc']:.4f} "
            f"auc_gap={best_calib_metric['exp_comp_auc_gap']:.4f} gate={int(best_calib_guard_satisfied)}"
        )
        calibration_diagnostics.update(
            {
                "best_epoch": int(best_calib_epoch),
                "best_score": float(best_calib_score),
                "best_guard_satisfied": bool(best_calib_guard_satisfied),
                "best_val": {k: to_float(v) for k, v in best_calib_metric.items()},
            }
        )

    print("\n================ Risk-guided evaluation ================")
    model.eval()
    classifier.eval()
    with torch.no_grad():
        h_full = model.encoder(gw, features)
        z_u_full, z_b_full = cf_operator.encode(h_full)
        logits_full, logits_cf_set, risk_full, h_cf_set = _counterfactual_risk_snapshot(
            model, h_full, classifier, cf_operator, donor_indices, z_b_full.detach(), cfg
        )
        risk_weight = risk_to_weight(
            risk_full,
            tau=getattr(cfg, "risk_tau", 0.1),
            temperature=getattr(cfg, "risk_temperature", 0.05),
            no_risk_weight=getattr(cfg, "no_risk_weight", False),
        )
        risk_weight = _apply_donor_confidence_weight(risk_weight, donor_reliability, cfg)
        feature_soft, structure_soft, x_exp_soft, is_self_loop = _build_dynamic_soft_masks(
            model,
            gw,
            features,
            h_full,
            h_cf_set.mean(dim=1),
            train_mask,
            sens_index,
            tau=getattr(cfg, "mask_tau", 1.0),
            cf_operator=cf_operator,
            donor_indices=donor_indices,
            z_b_bank=z_b_full.detach(),
            dynamic=not getattr(cfg, "static_exp_condition", False),
        )
        feature_soft, structure_soft = scope_fn(
            feature_soft, structure_soft, is_self_loop, fixed_local_node_mask, fixed_local_edge_mask
        )
        feature_mask, structure_mask = _hard_masks_from_soft(
            feature_soft,
            structure_soft,
            is_self_loop,
            sens_index,
            feature_max,
            structure_max,
            ablation_mode=getattr(cfg, "mask_ablation_mode", "full"),
        )
        feature_mask, structure_mask = scope_fn(
            feature_mask, structure_mask, is_self_loop, fixed_local_node_mask, fixed_local_edge_mask
        )
        feature_comp, structure_comp = model.get_complement_masks(feature_mask, structure_mask, is_self_loop)
        exp_graph, exp_feature, _, _, _ = model.get_explained_graph(gw, features, feature_mask, structure_mask, train_mask)
        comp_graph, comp_feature, _, _, _ = model.get_explained_graph(gw, features, feature_comp, structure_comp, train_mask)

        h_exp = model.encoder(exp_graph, exp_feature)
        logits_exp, logits_exp_cf_set, risk_exp, h_exp_cf_set = _counterfactual_risk_snapshot(
            model, h_exp, classifier, cf_operator, donor_indices, z_b_full.detach(), cfg
        )
        _, z_b_exp = cf_operator.encode(h_exp)
        h_comp = model.encoder(comp_graph, comp_feature)
        logits_comp, logits_comp_cf_set, risk_comp, h_comp_cf_set = _counterfactual_risk_snapshot(
            model, h_comp, classifier, cf_operator, donor_indices, z_b_full.detach(), cfg
        )
        _, z_b_comp = cf_operator.encode(h_comp)

        z_s_exposure_full = _sensitive_exposure_from_z_s(model, z_b_full)
        z_s_exposure_exp = _sensitive_exposure_from_z_s(model, z_b_exp)
        z_s_exposure_comp = _sensitive_exposure_from_z_s(model, z_b_comp)

        full_metric = metric.evaluate(logits_full[test_mask], labels[test_mask], sens[test_mask], logits_cf_set.mean(dim=1)[test_mask])
        exp_metric = metric.evaluate(logits_exp[test_mask], labels[test_mask], sens[test_mask], logits_exp_cf_set.mean(dim=1)[test_mask])
        comp_metric = metric.evaluate(logits_comp[test_mask], labels[test_mask], sens[test_mask], logits_comp_cf_set.mean(dim=1)[test_mask])

        # External NIFTY-style intervention: flip the raw sensitive feature while
        # keeping the learned explanation masks fixed.
        external_cf = {}
        if sens_index is not None:
            features_sensitive_cf = features.clone()
            features_sensitive_cf[:, sens_index] = 1.0 - features_sensitive_cf[:, sens_index]
            ext_exp_graph, ext_exp_feature, _, _, _ = model.get_explained_graph(
                gw, features_sensitive_cf, feature_mask, structure_mask, train_mask
            )
            ext_comp_graph, ext_comp_feature, _, _, _ = model.get_explained_graph(
                gw, features_sensitive_cf, feature_comp, structure_comp, train_mask
            )
            ext_logits_full, _, _ = _logits_from_h(
                model, classifier, model.encoder(gw, features_sensitive_cf), cfg
            )
            ext_logits_exp, _, _ = _logits_from_h(
                model, classifier, model.encoder(ext_exp_graph, ext_exp_feature), cfg
            )
            ext_logits_comp, _, _ = _logits_from_h(
                model, classifier, model.encoder(ext_comp_graph, ext_comp_feature), cfg
            )

            def _external_pair(factual, counterfactual):
                factual_prob = torch.sigmoid(factual[test_mask].squeeze())
                counter_prob = torch.sigmoid(counterfactual[test_mask].squeeze())
                return {
                    "unfairness": float(((factual_prob > 0.5) != (counter_prob > 0.5)).float().mean().item()),
                    "cf_prob_gap": float(torch.abs(factual_prob - counter_prob).mean().item()),
                }

            external_cf = {
                "protocol": "global_sensitive_feature_flip_fixed_explanation",
                "full_graph": _external_pair(logits_full, ext_logits_full),
                "explain_graph": _external_pair(logits_exp, ext_logits_exp),
                "complement_graph": _external_pair(logits_comp, ext_logits_comp),
            }

        p_full = torch.sigmoid(logits_full[test_mask])
        p_full_cf = torch.sigmoid(logits_cf_set.mean(dim=1)[test_mask])
        p_exp = torch.sigmoid(logits_exp[test_mask])
        p_exp_cf = torch.sigmoid(logits_exp_cf_set.mean(dim=1)[test_mask])
        p_comp = torch.sigmoid(logits_comp[test_mask])
        p_comp_cf = torch.sigmoid(logits_comp_cf_set.mean(dim=1)[test_mask])

        pred_full = p_full > 0.5
        pred_full_cf = p_full_cf > 0.5
        pred_exp = p_exp > 0.5
        pred_exp_cf = p_exp_cf > 0.5
        pred_comp = p_comp > 0.5
        pred_comp_cf = p_comp_cf > 0.5
        repr_full = _cf_representation_distance(h_full, h_cf_set)
        repr_exp = _cf_representation_distance(h_exp, h_exp_cf_set)
        repr_comp = _cf_representation_distance(h_comp, h_comp_cf_set)

    mask_sparsity = compute_mask_sparsity(feature_mask, structure_mask, is_self_loop, sens_index)
    complement_mask_sparsity = compute_mask_sparsity(
        feature_comp, structure_comp, is_self_loop, sens_index
    )

    # Manuscript RQ2 metrics. d_cf follows Eq. (24): mean L2 distance
    # normalized by sqrt(embedding dimension). Fidelity follows Eq. (25):
    # hard-label agreement with the full graph (not disagreement/Fid+).
    embedding_scale = float(max(h_full.shape[1], 1)) ** 0.5
    clean_fidelity = float((pred_comp == pred_full).float().mean().item())
    bias_fidelity = float((pred_exp == pred_full).float().mean().item())
    paper_explanation_eval = {
        "full": {
            "auc": float(full_metric["auc_roc"]),
            "delta_sp": float(full_metric["parity"]),
            "delta_eo": float(full_metric["equality"]),
            "d_cf": float(repr_full[test_mask].mean().item() / embedding_scale),
            "fidelity": None,
            "feature_retained": 1.0,
            "structure_retained": 1.0,
        },
        "clean": {
            "auc": float(comp_metric["auc_roc"]),
            "delta_sp": float(comp_metric["parity"]),
            "delta_eo": float(comp_metric["equality"]),
            "d_cf": float(repr_comp[test_mask].mean().item() / embedding_scale),
            "fidelity": clean_fidelity,
            "feature_retained": float(complement_mask_sparsity["feature_kept_ratio"]),
            "structure_retained": float(complement_mask_sparsity["structure_kept_ratio"]),
        },
        "bias": {
            "auc": float(exp_metric["auc_roc"]),
            "delta_sp": float(exp_metric["parity"]),
            "delta_eo": float(exp_metric["equality"]),
            "d_cf": float(repr_exp[test_mask].mean().item() / embedding_scale),
            "fidelity": bias_fidelity,
            "feature_retained": float(mask_sparsity["feature_kept_ratio"]),
            "structure_retained": float(mask_sparsity["structure_kept_ratio"]),
        },
    }
    if phase2_bias_clean:
        target_metric = comp_metric
        opposite_metric = exp_metric
        target_risk = risk_comp
        opposite_risk = risk_exp
        target_z_s_exposure = z_s_exposure_comp
        opposite_z_s_exposure = z_s_exposure_exp
        target_cf_flip = float(comp_metric["cf"]) if comp_metric.get("cf", None) is not None else None
        validity_target_name = "clean_graph"
        validity_opposite_name = "bias_graph"
    else:
        target_metric = exp_metric
        opposite_metric = comp_metric
        target_risk = risk_exp
        opposite_risk = risk_comp
        target_z_s_exposure = z_s_exposure_exp
        opposite_z_s_exposure = z_s_exposure_comp
        target_cf_flip = float(exp_metric["cf"]) if exp_metric.get("cf", None) is not None else None
        validity_target_name = "explain_graph"
        validity_opposite_name = "complement_graph"

    risk_reduction = float(risk_full[test_mask].mean().item() - target_risk[test_mask].mean().item())
    risk_reduction_ratio = float(1.0 - target_risk[test_mask].mean().item() / (risk_full[test_mask].mean().item() + 1e-8))
    risk_tau = float(getattr(cfg, "risk_tau", 0.1))
    high_mask = test_mask & (risk_full > risk_tau)
    repr_target = repr_comp if phase2_bias_clean else repr_exp
    repr_opposite = repr_exp if phase2_bias_clean else repr_comp
    cf_repr_consistency = {
        "full_h_cf_dist": float(repr_full[test_mask].mean().item()),
        "explain_h_cf_dist": float(repr_exp[test_mask].mean().item()),
        "complement_h_cf_dist": float(repr_comp[test_mask].mean().item()),
        "target_h_cf_dist": float(repr_target[test_mask].mean().item()),
        "opposite_h_cf_dist": float(repr_opposite[test_mask].mean().item()),
        "target_minus_full": float((repr_target[test_mask] - repr_full[test_mask]).mean().item()),
        "opposite_minus_target": float((repr_opposite[test_mask] - repr_target[test_mask]).mean().item()),
        "bias_h_cf_dist": float(repr_exp[test_mask].mean().item()) if phase2_bias_clean else None,
        "clean_h_cf_dist": float(repr_comp[test_mask].mean().item()) if phase2_bias_clean else None,
        "clean_minus_full": (
            float((repr_comp[test_mask] - repr_full[test_mask]).mean().item()) if phase2_bias_clean else None
        ),
        "bias_minus_clean": (
            float((repr_exp[test_mask] - repr_comp[test_mask]).mean().item()) if phase2_bias_clean else None
        ),
    }
    if repr_reduce_mask is not None:
        repr_reduce_test_mask = test_mask & repr_reduce_mask
        cf_repr_consistency.update(
            {
                "repr_reduce_top_ratio": float(repr_reduce_top_ratio),
                "repr_reduce_test_nodes": int(repr_reduce_test_mask.sum().item()),
                "top_full_h_cf_dist": (
                    float(repr_full[repr_reduce_test_mask].mean().item()) if repr_reduce_test_mask.any() else None
                ),
                "top_clean_h_cf_dist": (
                    float(repr_comp[repr_reduce_test_mask].mean().item())
                    if phase2_bias_clean and repr_reduce_test_mask.any()
                    else None
                ),
                "top_bias_h_cf_dist": (
                    float(repr_exp[repr_reduce_test_mask].mean().item())
                    if phase2_bias_clean and repr_reduce_test_mask.any()
                    else None
                ),
                "top_clean_minus_full": (
                    float((repr_comp[repr_reduce_test_mask] - repr_full[repr_reduce_test_mask]).mean().item())
                    if phase2_bias_clean and repr_reduce_test_mask.any()
                    else None
                ),
                "top_bias_minus_clean": (
                    float((repr_exp[repr_reduce_test_mask] - repr_comp[repr_reduce_test_mask]).mean().item())
                    if phase2_bias_clean and repr_reduce_test_mask.any()
                    else None
                ),
            }
        )
    eval_anchor_mask, eval_local_node_mask, eval_local_edge_mask = _select_local_region(
        gw,
        (risk_weight * risk_full).detach(),
        train_mask,
        top_k=getattr(cfg, "local_top_k", 512),
        hops=getattr(cfg, "local_hops", 2),
    )

    hard_explain_auc = float(target_metric.get("auc_roc", 0.0))
    hard_complement_auc = float(opposite_metric.get("auc_roc", 0.0))
    hard_exp_comp_gap = hard_explain_auc - hard_complement_auc
    hard_cf_flip = target_cf_flip
    min_valid_auc = float(getattr(cfg, "validity_min_explain_auc", 0.55))
    min_valid_gap = float(getattr(cfg, "validity_min_exp_comp_gap", 0.0))
    max_valid_cf = getattr(cfg, "validity_max_cf_flip", None)
    if max_valid_cf is not None:
        max_valid_cf = float(max_valid_cf)

    hard_validity_failures = []
    hard_explain_auc_ok = hard_explain_auc >= min_valid_auc
    hard_exp_comp_gap_ok = hard_exp_comp_gap >= min_valid_gap
    hard_cf_flip_ok = True if max_valid_cf is None or hard_cf_flip is None else hard_cf_flip <= max_valid_cf
    if not hard_explain_auc_ok:
        hard_validity_failures.append(f"{validity_target_name}_auc_below_min")
    if not hard_exp_comp_gap_ok:
        hard_validity_failures.append(f"{validity_target_name}_{validity_opposite_name}_auc_gap_below_min")
    if not hard_cf_flip_ok:
        hard_validity_failures.append("cf_flip_above_max")
    hard_mask_validity = {
        "valid": not hard_validity_failures,
        "failures": ";".join(hard_validity_failures),
        "explain_auc": hard_explain_auc,
        "complement_auc": hard_complement_auc,
        "exp_comp_auc_gap": hard_exp_comp_gap,
        "target_graph": validity_target_name,
        "opposite_graph": validity_opposite_name,
        "bias_auc": float(exp_metric.get("auc_roc", 0.0)) if phase2_bias_clean else None,
        "clean_auc": float(comp_metric.get("auc_roc", 0.0)) if phase2_bias_clean else None,
        "clean_bias_auc_gap": (
            float(comp_metric.get("auc_roc", 0.0)) - float(exp_metric.get("auc_roc", 0.0))
            if phase2_bias_clean
            else None
        ),
        "cf_flip": hard_cf_flip,
        "min_explain_auc": min_valid_auc,
        "min_exp_comp_gap": min_valid_gap,
        "max_cf_flip": max_valid_cf,
        "explain_auc_ok": hard_explain_auc_ok,
        "exp_comp_gap_ok": hard_exp_comp_gap_ok,
        "cf_flip_ok": hard_cf_flip_ok,
        "strict": bool(getattr(cfg, "strict_run_validity", False)),
    }
    random_mask_diagnostics = {"enabled": False}
    if phase2_bias_clean and int(getattr(cfg, "random_mask_trials", 5)) > 0:
        random_mask_diagnostics = _matched_random_bias_clean_diagnostics(
            model,
            gw,
            features,
            labels,
            sens,
            test_mask,
            train_mask,
            classifier,
            cf_operator,
            donor_indices,
            z_b_full.detach(),
            feature_mask,
            structure_mask,
            is_self_loop,
            sens_index,
            cfg,
        )
    run_failures = []
    if getattr(cfg, "strict_run_validity", False) and hard_validity_failures:
        run_failures.extend([f"hard_mask_validity:{failure}" for failure in hard_validity_failures])
    if hard_validity_failures and getattr(cfg, "strict_run_validity", False):
        run_status = "objective_failure"
    elif run_failures:
        run_status = "diagnostic_failure"
    else:
        run_status = "ok"

    case_study_export = _write_cf_case_study_export(
        cfg=cfg,
        gw=gw,
        features=features,
        labels=labels,
        sens=sens,
        test_mask=test_mask,
        sens_index=sens_index,
        feature_soft=feature_soft,
        structure_soft=structure_soft,
        feature_mask=feature_mask,
        structure_mask=structure_mask,
        logits_full=logits_full,
        logits_cf_set=logits_cf_set,
        logits_exp=logits_exp,
        logits_exp_cf_set=logits_exp_cf_set,
        logits_comp=logits_comp,
        logits_comp_cf_set=logits_comp_cf_set,
        repr_full=repr_full,
        repr_exp=repr_exp,
        repr_comp=repr_comp,
    )

    mask_score_export = {"enabled": False}
    export_template = getattr(cfg, "export_mask_scores", None)
    if export_template:
        export_path = str(export_template).format(seed=cfg.seed, dataset=cfg.dataset)
        export_dir = os.path.dirname(os.path.abspath(export_path))
        os.makedirs(export_dir, exist_ok=True)
        if phase2_bias_clean:
            feature_bias_scores = feature_soft
            structure_bias_scores = structure_soft
        else:
            feature_bias_scores = 1.0 - feature_soft
            structure_bias_scores = 1.0 - structure_soft
        source, target = gw.edges()
        edge_index = torch.stack([target, source], dim=0)
        np.savez_compressed(
            export_path,
            feature_scores=feature_bias_scores.detach().cpu().numpy(),
            edge_scores=structure_bias_scores.detach().cpu().numpy(),
            edge_index=edge_index.detach().cpu().numpy(),
        )
        mask_score_export = {
            "enabled": True,
            "path": export_path,
            "score_semantics": "larger_is_more_bias_related",
            "feature_shape": list(feature_bias_scores.shape),
            "edge_count": int(structure_bias_scores.numel()),
        }

    explanation_artifact_export = {"enabled": False}
    artifact_template = getattr(cfg, "export_explanation_artifacts", None)
    if artifact_template:
        artifact_path = str(artifact_template).format(seed=cfg.seed, dataset=cfg.dataset)
        os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
        h_cf_mean = h_cf_set.mean(dim=1)
        z_u_full, z_b_factual = cf_operator.encode(h_full)
        z_u_cf, z_b_counterfactual = cf_operator.encode(h_cf_mean)
        np.savez_compressed(
            artifact_path,
            node_index=torch.where(test_mask)[0].detach().cpu().numpy(),
            label=labels[test_mask].detach().cpu().numpy(),
            sensitive=sens[test_mask].detach().cpu().numpy(),
            h_factual=h_full[test_mask].detach().cpu().numpy(),
            h_counterfactual=h_cf_mean[test_mask].detach().cpu().numpy(),
            z_b_factual=z_b_factual[test_mask].detach().cpu().numpy(),
            z_b_counterfactual=z_b_counterfactual[test_mask].detach().cpu().numpy(),
        )
        explanation_artifact_export = {"enabled": True, "path": artifact_path}

    diagnostics = {
        "mask_sparsity": mask_sparsity,
        "complement_mask_sparsity": complement_mask_sparsity,
        "paper_explanation_eval": paper_explanation_eval,
        "explanation_artifact_export": explanation_artifact_export,
        "hard_mask_validity": hard_mask_validity,
        "random_mask": random_mask_diagnostics,
        "case_study_export": case_study_export or {"enabled": False},
        "mask_score_export": mask_score_export,
        "run_status": {
            "status": run_status,
            "primary_objective": "counterfactual_risk_guided_explanation",
            "failure_reason": ";".join(run_failures),
        },
        "cf_risk": {
            "full_mean": float(risk_full[test_mask].mean().item()),
            "explanation_mean": float(risk_exp[test_mask].mean().item()),
            "complement_mean": float(risk_comp[test_mask].mean().item()),
            "target_mean": float(target_risk[test_mask].mean().item()),
            "opposite_mean": float(opposite_risk[test_mask].mean().item()),
            "bias_mean": float(risk_exp[test_mask].mean().item()) if phase2_bias_clean else None,
            "clean_mean": float(risk_comp[test_mask].mean().item()) if phase2_bias_clean else None,
            "bias_minus_clean": (
                float((risk_exp[test_mask] - risk_comp[test_mask]).mean().item())
                if phase2_bias_clean
                else None
            ),
            "risk_reduction": risk_reduction,
            "risk_reduction_ratio": risk_reduction_ratio,
            "high_risk_ratio": float(high_mask.float().mean().item()),
            "risk_weight_mean": float(risk_weight[test_mask].mean().item()),
        },
        "cf_representation_consistency": cf_repr_consistency,
        "sensitive_exposure": {
            "full_mean": float(z_s_exposure_full[test_mask].mean().item()),
            "explanation_mean": float(z_s_exposure_exp[test_mask].mean().item()),
            "complement_mean": float(z_s_exposure_comp[test_mask].mean().item()),
            "target_mean": float(target_z_s_exposure[test_mask].mean().item()),
            "opposite_mean": float(opposite_z_s_exposure[test_mask].mean().item()),
            "bias_mean": float(z_s_exposure_exp[test_mask].mean().item()) if phase2_bias_clean else None,
            "clean_mean": float(z_s_exposure_comp[test_mask].mean().item()) if phase2_bias_clean else None,
        },
        "high_risk_subset": {
            "num_nodes": int(high_mask.sum().item()),
            "risk_full_mean": float(risk_full[high_mask].mean().item()) if high_mask.any() else 0.0,
            "risk_exp_mean": float(target_risk[high_mask].mean().item()) if high_mask.any() else 0.0,
        },
        "disentangle_diagnostics": {"rec_error": float(rec_error.item())},
        "donor_selector": donor_stats,
        "counterfactual_reference": cf_operator.last_cf_stats,
        "external_cf_sensitivity": external_cf,
        "zs_h_geometry": zs_h_geometry,
        "local_explainer": {
            "local_top_k": int(getattr(cfg, "local_top_k", 512)),
            "local_hops": int(getattr(cfg, "local_hops", 2)),
            "anchor_count": int(eval_anchor_mask.sum().item()),
            "local_node_ratio": float(eval_local_node_mask.float().mean().item()),
            "local_edge_ratio": float(eval_local_edge_mask.float().mean().item()),
            "anchor_full_risk_mean": float(risk_full[eval_anchor_mask].mean().item()) if eval_anchor_mask.any() else 0.0,
            "anchor_exp_risk_mean": float(target_risk[eval_anchor_mask].mean().item()) if eval_anchor_mask.any() else 0.0,
        },
        "bias_clean": {
            "enabled": bool(phase2_bias_clean),
            "bias_graph_alias": "explain_graph",
            "clean_graph_alias": "complement_graph",
            "bias_auc": float(exp_metric.get("auc_roc", 0.0)),
            "clean_auc": float(comp_metric.get("auc_roc", 0.0)),
            "clean_bias_auc_gap": float(comp_metric.get("auc_roc", 0.0)) - float(exp_metric.get("auc_roc", 0.0)),
            "bias_dp": float(exp_metric.get("parity", 0.0)),
            "clean_dp": float(comp_metric.get("parity", 0.0)),
            "bias_eo": float(exp_metric.get("equality", 0.0)),
            "clean_eo": float(comp_metric.get("equality", 0.0)),
            "bias_cf": float(exp_metric.get("cf", 0.0)),
            "clean_cf": float(comp_metric.get("cf", 0.0)),
            "risk_bias_mean": float(risk_exp[test_mask].mean().item()),
            "risk_clean_mean": float(risk_comp[test_mask].mean().item()),
            "risk_bias_minus_clean": float((risk_exp[test_mask] - risk_comp[test_mask]).mean().item()),
        },
        "early_stopping": {
            "fair_early_stop": bool(getattr(cfg, "fair_early_stop", False)),
            "phase3_update_mode": phase3_update_mode,
            "checkpoint_selection_mode": getattr(cfg, "checkpoint_selection_mode", "soft"),
            "soft_select_min_auc": (
                None
                if getattr(cfg, "soft_select_min_auc", None) is None
                else float(getattr(cfg, "soft_select_min_auc"))
            ),
            "hard_gate_satisfied": bool(best_joint_gate_satisfied),
            "hard_select_min_auc": float(getattr(cfg, "hard_select_min_auc", None) or (getattr(cfg, "selection_min_auc", 0.55) or 0.55)),
            "hard_select_min_margin": float(getattr(cfg, "hard_select_min_margin", 0.0)),
            "best_joint_epoch": int(best_joint_epoch),
            "best_joint_score": float(best_fair_score if getattr(cfg, "fair_early_stop", False) else best_tradeoff),
            "best_joint_val": best_joint_val,
            "joint_early_stop_patience": int(joint_patience),
            "joint_early_stop_min_delta": float(joint_min_delta),
            "stopped_early": bool(joint_stopped_early),
            "stop_epoch": joint_stop_epoch,
            "stop_reason": joint_stop_reason,
            "validation_checks_without_improvement": int(joint_checks_without_improvement),
        },
        "calibration": calibration_diagnostics,
        "run_config": {
            "method": cfg.method,
            "phase2_mask_role": "bias_clean",
            "explainer_scope": getattr(cfg, "explainer_scope", "local"),
            "num_donors": int(getattr(cfg, "num_donors", getattr(cfg, "donor_top_k", 5))),
            "donor_label_score_max_diff": (
                None
                if getattr(cfg, "donor_label_score_max_diff", None) is None
                else float(getattr(cfg, "donor_label_score_max_diff"))
            ),
            "use_donor_confidence_weight": bool(getattr(cfg, "use_donor_confidence_weight", False)),
            "use_donor_reliability_gate": bool(getattr(cfg, "use_donor_reliability_gate", False)),
            "donor_reliability_gate_tau": float(getattr(cfg, "donor_reliability_gate_tau", 0.8)),
            "donor_reliability_gate_floor": float(getattr(cfg, "donor_reliability_gate_floor", 0.5)),
            "donor_require_same_label": bool(getattr(cfg, "donor_require_same_label", False)),
            "donor_select_chunk_size": int(getattr(cfg, "donor_select_chunk_size", 4096)),
            "donor_confidence_score_tau": float(getattr(cfg, "donor_confidence_score_tau", 0.05)),
            "donor_reliability_floor": float(getattr(cfg, "donor_reliability_floor", 0.05)),
            "donor_selection_mode": getattr(cfg, "donor_selection_mode", "distance"),
            "donor_label_rank_weight": float(getattr(cfg, "donor_label_rank_weight", 0.25)),
            "cf_ref_mode": "local_quantile",
            "stage1_map_loss": getattr(cfg, "stage1_map_loss", "none"),
            "lambda_stage1_map": float(getattr(cfg, "lambda_stage1_map", 0.0)),
            "phase3_update_mode": phase3_update_mode,
            "risk_mode": getattr(cfg, "risk_mode", "mean_logit_gap"),
            "risk_tau": float(getattr(cfg, "risk_tau", 0.1)),
            "rho_reduce": float(getattr(cfg, "rho_reduce", 0.5)),
            "repr_reduce_top_ratio": float(getattr(cfg, "repr_reduce_top_ratio", 0.0)),
            "reduce_loss_type": getattr(cfg, "reduce_loss_type", "prediction"),
            "lambda_reduce": float(getattr(cfg, "lambda_reduce", 1.0)),
            "u_sufficiency_type": "prediction_bce",
            "lambda_u_suf": float(getattr(cfg, "lambda_u_suf", 1.0)),
            "lambda_h_cf_inv": float(getattr(cfg, "lambda_h_cf_inv", 0.1)),
            "lambda_zs_fair": float(getattr(cfg, "lambda_zs_fair", 1.0)),
            "sp_regularizer_type": "local_bernoulli_ib_kl",
            "hidden_dim": int(getattr(cfg, "hidden_dim", 64)),
            "warm_y_loss": getattr(cfg, "warm_y_loss", "mlp_bce"),
            "warm_sensitive_loss": getattr(cfg, "warm_sensitive_loss", "distance_softplus"),
            "joo_di_y_weight": float(getattr(cfg, "joo_di_y_weight", 1.0)),
            "lambda_y": float(getattr(cfg, "lambda_y", 1.0)),
            "lambda_s": float(getattr(cfg, "lambda_s", 1.0)),
            "cf_stage1_regularizer": getattr(cfg, "cf_stage1_regularizer", "hsic"),
            "lambda_ind": float(getattr(cfg, "lambda_ind", 1.0)),
            "lambda_orth": float(getattr(cfg, "lambda_orth", 0.05)),
            "lambda_zu_con": float(getattr(cfg, "lambda_zu_con", 0.1)),
            "lambda_rec": 0.0,
            "lambda_local_keep": float(getattr(cfg, "lambda_local_keep", 1.0)),
            "lambda_pred_exp": float(getattr(cfg, "lambda_pred_exp", 1.0)),
            "lambda_sp": float(getattr(cfg, "lambda_sp", 0.01)),
            "joint_mask_mode": getattr(cfg, "joint_mask_mode", "soft"),
            "gsat_stochastic_mask": bool(getattr(cfg, "gsat_stochastic_mask", False)),
            "gsat_temperature": float(getattr(cfg, "gsat_temperature", 0.5)),
            "lambda_gsat_entropy": float(getattr(cfg, "lambda_gsat_entropy", 0.0)),
            "lambda_mask_budget": float(getattr(cfg, "lambda_mask_budget", 0.0)),
            "mask_feature_budget": (
                None if getattr(cfg, "mask_feature_budget", None) is None else float(getattr(cfg, "mask_feature_budget"))
            ),
            "mask_structure_budget": (
                None if getattr(cfg, "mask_structure_budget", None) is None else float(getattr(cfg, "mask_structure_budget"))
            ),
            "final_pred_source": getattr(cfg, "final_pred_source", "zy"),
            "prediction_head": "encoder_h_linear" if _uses_h_prediction(cfg) else "fairINN_pred_y",
            "warm_cls_used": bool(_uses_h_prediction(cfg) and not getattr(cfg, "no_warm_cls", False)),
            "no_warm_cls": bool(getattr(cfg, "no_warm_cls", False)),
            "warm_task_source": "fairINN_pred_y",
            "risk_definition": "cf_logit_gap_plus_z_s_sensitive_exposure",
            "static_exp_condition": bool(getattr(cfg, "static_exp_condition", False)),
            "local_top_k": int(getattr(cfg, "local_top_k", 512)),
            "local_hops": int(getattr(cfg, "local_hops", 2)),
            "fair_early_stop": bool(getattr(cfg, "fair_early_stop", False)),
            "checkpoint_selection_mode": getattr(cfg, "checkpoint_selection_mode", "soft"),
            "soft_select_min_auc": (
                None
                if getattr(cfg, "soft_select_min_auc", None) is None
                else float(getattr(cfg, "soft_select_min_auc"))
            ),
            "fair_stop_min_auc": float(getattr(cfg, "fair_stop_min_auc", 0.0)),
            "fair_stop_cf_weight": float(getattr(cfg, "fair_stop_cf_weight", 1.0)),
            "fair_stop_logit_weight": float(getattr(cfg, "fair_stop_logit_weight", 0.5)),
            "fair_stop_auc_weight": float(getattr(cfg, "fair_stop_auc_weight", 0.0)),
            "joint_early_stop_patience": int(getattr(cfg, "joint_early_stop_patience", 0)),
            "joint_early_stop_min_delta": float(getattr(cfg, "joint_early_stop_min_delta", 1e-4)),
            "cf_group_calibration": bool(getattr(cfg, "cf_group_calibration", False)),
            "calib_epochs": int(getattr(cfg, "calib_epochs", 50)),
            "calib_hard_guard": bool(getattr(cfg, "calib_hard_guard", False)),
            "calib_guard_min_auc": float(getattr(cfg, "calib_guard_min_auc", 0.55)),
            "calib_guard_min_margin": float(getattr(cfg, "calib_guard_min_margin", 0.0)),
            "strict_run_validity": bool(getattr(cfg, "strict_run_validity", False)),
            "validity_min_explain_auc": float(getattr(cfg, "validity_min_explain_auc", 0.55)),
            "validity_min_exp_comp_gap": float(getattr(cfg, "validity_min_exp_comp_gap", 0.0)),
            "validity_max_cf_flip": (
                None
                if getattr(cfg, "validity_max_cf_flip", None) is None
                else float(getattr(cfg, "validity_max_cf_flip"))
            ),
            "lambda_calib_ind": float(getattr(cfg, "lambda_calib_ind", 0.5)),
            "lambda_calib_group": float(getattr(cfg, "lambda_calib_group", 0.5)),
            "lambda_calib_distill": float(getattr(cfg, "lambda_calib_distill", 0.1)),
            "lambda_calib_pair": float(getattr(cfg, "lambda_calib_pair", 0.3)),
            "calib_auc_drop_tol": float(getattr(cfg, "calib_auc_drop_tol", 0.01)),
            "calib_min_auc": float(getattr(cfg, "calib_min_auc", 0.0)),
            "calib_select_cf_weight": float(getattr(cfg, "calib_select_cf_weight", 1.0)),
            "calib_select_logit_weight": float(getattr(cfg, "calib_select_logit_weight", 0.5)),
            "calib_select_auc_weight": float(getattr(cfg, "calib_select_auc_weight", 0.0)),
        },
        "warmup_history": warm_history,
        "exp_history": exp_history,
    }

    system_fairness = {
        "original_us": float(torch.abs(p_full - p_full_cf).mean().item()),
        "explain_us": float(torch.abs(p_exp - p_exp_cf).mean().item()),
        "comp_us": float(torch.abs(p_comp - p_comp_cf).mean().item()),
        "original_flip": float((pred_full != pred_full_cf).float().mean().item()),
        "explain_flip": float((pred_exp != pred_exp_cf).float().mean().item()),
        "comp_flip": float((pred_comp != pred_comp_cf).float().mean().item()),
    }
    fid_accuracy = {
        "Fid_plus_flip": float((pred_full != pred_comp).float().mean().item()),
        "Fid_minus_flip": float((pred_full != pred_exp).float().mean().item()),
        "Fid_plus_prob": float(torch.abs(p_full - p_comp).mean().item()),
        "Fid_minus_prob": float(torch.abs(p_full - p_exp).mean().item()),
    }
    fid_cf = {
        "CF_fid_plus_prob": float(torch.abs(torch.abs(p_full - p_full_cf) - torch.abs(p_comp - p_comp_cf)).mean().item()),
        "CF_fid_minus_prob": float(torch.abs(torch.abs(p_full - p_full_cf) - torch.abs(p_exp - p_exp_cf)).mean().item()),
        "CF_fid_plus_flip": float(((pred_full != pred_full_cf) != (pred_comp != pred_comp_cf)).float().mean().item()),
        "CF_fid_minus_flip": float(((pred_full != pred_full_cf) != (pred_exp != pred_exp_cf)).float().mean().item()),
    }

    if phase2_bias_clean:
        print("Bias graph (explain_graph):", exp_metric)
        print("Clean graph (complement_graph):", comp_metric)
    else:
        print("Explain graph:", exp_metric)
        print("Complement graph:", comp_metric)
    print("Full graph:", full_metric)
    print("CF risk diagnostics:", diagnostics["cf_risk"])
    print("Mask sparsity:", mask_sparsity)

    return exp_metric, full_metric, comp_metric, system_fairness, fid_accuracy, fid_cf, diagnostics
