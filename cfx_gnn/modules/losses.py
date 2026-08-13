"""Loss helpers for counterfactual-risk-guided explanation learning."""

import torch
import torch.nn.functional as F


def binary_task_loss(logits, labels, mask):
    """BCE loss over a node mask for binary logits with shape [N, 1]."""
    labels = labels.float().view(-1, 1).to(logits.device)
    mask = mask.to(logits.device).bool()
    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


def utility_sufficiency_loss(z_u_exp, z_u_full, mask=None):
    """Cosine distance between explanation and full utility proxies."""
    dist = 1.0 - F.cosine_similarity(z_u_exp, z_u_full.detach(), dim=-1)
    if mask is not None:
        dist = dist[mask.to(dist.device).bool()]
    return dist.mean()


def weighted_reduce_loss(risk_exp, risk_full, weights, rho=0.5):
    """Penalize explanation risk above a fraction of full risk."""
    margin = risk_exp - float(rho) * risk_full.detach()
    return (weights.detach() * F.relu(margin)).mean()


def mask_sparsity_loss(feature_mask, structure_mask, is_self_loop, sens_index=None):
    """Mean kept ratio over allowed features and non-self-loop edges."""
    feature_candidates = torch.ones_like(feature_mask, dtype=torch.bool)
    if sens_index is not None:
        feature_candidates[:, sens_index] = False
    feature_part = feature_mask[feature_candidates].mean()
    non_loop = ~is_self_loop
    if non_loop.any():
        structure_part = structure_mask[non_loop].mean()
    else:
        structure_part = structure_mask.mean()
    return feature_part + structure_part


def supervised_contrastive_loss(z, positive_key, temperature=0.07, secondary_key=None, max_samples=None):
    """Supervised contrastive loss with optional label-conditioned positives.

    If secondary_key is provided, positives must share both positive_key and
    secondary_key. This is useful for grouping z_b by sensitive attribute while
    avoiding label-only shortcuts inside each sensitive group.
    """
    if max_samples is not None and z.size(0) > int(max_samples):
        perm = torch.randperm(z.size(0), device=z.device)[: int(max_samples)]
        z = z[perm]
        positive_key = positive_key[perm]
        if secondary_key is not None:
            secondary_key = secondary_key[perm]

    z = F.normalize(z, dim=1)
    positive_key = positive_key.contiguous().view(-1, 1)
    positive_mask = torch.eq(positive_key, positive_key.T)
    if secondary_key is not None:
        secondary_key = secondary_key.contiguous().view(-1, 1)
        positive_mask = positive_mask & torch.eq(secondary_key, secondary_key.T)

    logits_mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    positive_mask = positive_mask & logits_mask
    valid = positive_mask.sum(dim=1) > 0
    if not valid.any():
        return z.new_tensor(0.0)

    logits = torch.mm(z, z.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
    return -mean_log_prob_pos[valid].mean()


def balanced_supervised_contrastive_loss(
    z,
    labels,
    sensitive,
    temperature=0.07,
    max_samples=None,
    cross_sensitive_positive=False,
):
    """SupCon for z_y with sensitive-balanced pair weights.

    Positives share the task label. If cross_sensitive_positive is true,
    positives must also come from the opposite sensitive group, forcing
    same-label cross-group alignment. Pair weights are inverse group-count
    weights over (label, sensitive), so majority sensitive groups do not
    dominate the contrastive denominator.
    """
    if max_samples is not None and z.size(0) > int(max_samples):
        perm = torch.randperm(z.size(0), device=z.device)[: int(max_samples)]
        z = z[perm]
        labels = labels[perm]
        sensitive = sensitive[perm]

    z = F.normalize(z, dim=1)
    labels = labels.long().view(-1)
    sensitive = sensitive.long().view(-1)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_sensitive = sensitive.unsqueeze(0) == sensitive.unsqueeze(1)
    logits_mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    positive_mask = same_label & logits_mask
    if cross_sensitive_positive:
        positive_mask = positive_mask & (~same_sensitive)
    valid = positive_mask.sum(dim=1) > 0
    if not valid.any():
        return z.new_tensor(0.0)

    group_key = labels * (sensitive.max().long() + 1) + sensitive
    unique, counts = torch.unique(group_key, return_counts=True)
    group_weight = torch.ones_like(group_key, dtype=z.dtype)
    for key, count in zip(unique, counts):
        group_weight[group_key == key] = 1.0 / count.float().clamp(min=1.0)
    group_weight = group_weight / group_weight.mean().clamp(min=1e-8)

    logits = torch.mm(z, z.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    pair_weight = group_weight.view(1, -1) * logits_mask.float()
    exp_logits = torch.exp(logits) * pair_weight
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-8))

    pos_weight = positive_mask.float() * group_weight.view(1, -1)
    mean_log_prob_pos = (pos_weight * log_prob).sum(dim=1) / pos_weight.sum(dim=1).clamp(min=1e-8)
    return -mean_log_prob_pos[valid].mean()


def prototype_contrastive_loss(z, group, temperature=0.07):
    """Two-group prototype contrast for sensitive proxy geometry."""
    group = group.long().view(-1)
    unique = torch.unique(group)
    if unique.numel() < 2:
        return z.new_tensor(0.0)
    z_norm = F.normalize(z, dim=1)
    prototypes = []
    proto_labels = []
    for value in unique:
        mask = group == value
        if mask.any():
            prototypes.append(F.normalize(z_norm[mask].mean(dim=0, keepdim=True), dim=1))
            proto_labels.append(value)
    prototypes = torch.cat(prototypes, dim=0)
    proto_labels = torch.stack(proto_labels).to(z.device)
    logits = torch.mm(z_norm, prototypes.T) / max(float(temperature), 1e-6)
    targets = (group.unsqueeze(1) == proto_labels.unsqueeze(0)).float().argmax(dim=1)
    return F.cross_entropy(logits, targets)


def label_conditioned_prototype_contrastive_loss(z, sensitive, labels, temperature=0.07, max_samples=None):
    """Sensitive prototype contrast computed within each task-label stratum.

    This shapes z_b to separate sensitive groups while avoiding an easy
    label-separation shortcut: each node is contrasted only against prototypes
    built from nodes with the same task label.
    """
    if max_samples is not None and z.size(0) > int(max_samples):
        perm = torch.randperm(z.size(0), device=z.device)[: int(max_samples)]
        z = z[perm]
        sensitive = sensitive[perm]
        labels = labels[perm]

    z_norm = F.normalize(z, dim=1)
    sensitive = sensitive.long().view(-1)
    labels = labels.long().view(-1)
    losses = []
    for label_value in torch.unique(labels):
        label_mask = labels == label_value
        local_sensitive = sensitive[label_mask]
        unique_sensitive = torch.unique(local_sensitive)
        if unique_sensitive.numel() < 2:
            continue
        z_label = z_norm[label_mask]
        prototypes = []
        proto_labels = []
        for sensitive_value in unique_sensitive:
            group_mask = local_sensitive == sensitive_value
            if group_mask.any():
                prototypes.append(F.normalize(z_label[group_mask].mean(dim=0, keepdim=True), dim=1))
                proto_labels.append(sensitive_value)
        prototypes = torch.cat(prototypes, dim=0)
        proto_labels = torch.stack(proto_labels).to(z.device)
        logits = torch.mm(z_label, prototypes.T) / max(float(temperature), 1e-6)
        targets = (local_sensitive.unsqueeze(1) == proto_labels.unsqueeze(0)).float().argmax(dim=1)
        losses.append(F.cross_entropy(logits, targets))
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def conditional_sensitive_contrastive_loss(
    z,
    sensitive,
    labels,
    margin=1.0,
    max_samples=None,
):
    """Pairwise contrastive loss for z_b conditioned on task label.

    Within each label stratum, same-sensitive pairs are pulled together and
    different-sensitive pairs are pushed apart. Cross-label pairs are ignored
    so z_b is discouraged from solving the easier label-separation problem.
    """
    if max_samples is not None and z.size(0) > int(max_samples):
        perm = torch.randperm(z.size(0), device=z.device)[: int(max_samples)]
        z = z[perm]
        sensitive = sensitive[perm]
        labels = labels[perm]

    z = F.normalize(z, dim=1)
    sensitive = sensitive.long().view(-1)
    labels = labels.long().view(-1)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_sensitive = sensitive.unsqueeze(0) == sensitive.unsqueeze(1)
    not_self = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    pos_mask = same_label & same_sensitive & not_self
    neg_mask = same_label & (~same_sensitive) & not_self
    dist = torch.cdist(z, z, p=2)

    losses = []
    if pos_mask.any():
        losses.append(dist[pos_mask].pow(2).mean())
    if neg_mask.any():
        losses.append(F.relu(float(margin) - dist[neg_mask]).pow(2).mean())
    if not losses:
        return z.new_tensor(0.0)
    return sum(losses)
