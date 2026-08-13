"""Counterfactual risk scoring and soft risk weighting."""

import torch


def compute_cf_risk(logits, logits_cf_set, mode="mean_logit_gap", var_gamma=0.0):
    """Compute per-node counterfactual risk.

    logits: [N, 1] or [N, C].
    logits_cf_set: [N, K, 1] or [N, K, C].
    returns: [N] risk scores.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if logits_cf_set.dim() == 2:
        logits_cf_set = logits_cf_set.unsqueeze(-1)
    gap = torch.abs(logits_cf_set - logits.unsqueeze(1)).mean(dim=-1)  # [N, K]
    if mode == "mean_logit_gap":
        return gap.mean(dim=1)
    if mode == "mean_var_logit_gap":
        return gap.mean(dim=1) + float(var_gamma) * gap.var(dim=1, unbiased=False)
    raise ValueError(f"Unsupported risk mode: {mode}")


def risk_to_weight(risk, tau=0.1, temperature=0.05, no_risk_weight=False):
    """Convert risk [N] to detached soft weights [N]."""
    if no_risk_weight:
        return torch.ones_like(risk).detach()
    temperature = max(float(temperature), 1e-6)
    return torch.sigmoid((risk - float(tau)) / temperature).detach()
