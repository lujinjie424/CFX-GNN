"""Top-k donor selection for representation-level counterfactual intervention."""

import torch


class DonorSelector:
    """Select opposite-sensitive donor nodes with optional degree compatibility.

    Args:
        struct_beta: Weight for the degree-based structural distance.
        eps: Numerical stability constant.
    """

    def __init__(self, struct_beta=0.1, eps=1e-8):
        self.struct_beta = float(struct_beta)
        self.eps = float(eps)
        self.last_stats = {}
        self.last_reliability = None

    @torch.no_grad()
    def select(
        self,
        z_u,
        sensitive,
        train_mask=None,
        graph=None,
        k=5,
        use_structure=False,
        labels=None,
        label_scores=None,
        label_penalty=0.0,
        label_score_max_diff=None,
        require_same_label=False,
        chunk_size=4096,
        confidence_score_tau=0.05,
        selection_mode="distance",
        label_rank_weight=0.25,
    ):
        """Return donor indices with shape [N, K].

        z_u: [N, d_u] utility-related proxy representation.
        sensitive: [N] binary sensitive group labels.
        train_mask: optional [N] bool mask defining the donor bank.
        graph: optional DGL graph used for degree-based compatibility.
        """
        device = z_u.device
        num_nodes = z_u.size(0)
        k = max(1, int(k))
        sensitive = sensitive.to(device).long().view(-1)
        labels = None if labels is None else labels.to(device).long().view(-1)
        label_scores = None if label_scores is None else label_scores.to(device).float().view(-1)
        if train_mask is None:
            train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        else:
            train_mask = train_mask.to(device).bool()

        bank_idx = torch.where(train_mask)[0]
        if bank_idx.numel() == 0:
            raise ValueError("DonorSelector requires at least one donor-bank node.")

        degree = None
        if use_structure and graph is not None:
            degree = (graph.in_degrees().float() + graph.out_degrees().float()).to(device)
            degree = torch.log1p(degree).view(-1, 1)

        donor_indices = torch.empty(num_nodes, k, dtype=torch.long, device=device)
        donor_u_dist = torch.empty(num_nodes, k, dtype=z_u.dtype, device=device)
        donor_degree_diff = torch.empty(num_nodes, k, dtype=z_u.dtype, device=device)
        donor_label_score_diff = torch.zeros(num_nodes, k, dtype=z_u.dtype, device=device)
        donor_label_filter_valid = torch.ones(num_nodes, k, dtype=torch.bool, device=device)
        fallback_count = 0
        label_filter_query_count = 0
        label_filter_fallback_count = 0

        for src_group, dst_group in ((0, 1), (1, 0)):
            query_mask = sensitive == src_group
            pool_idx = bank_idx[sensitive[bank_idx] == dst_group]
            if pool_idx.numel() == 0:
                pool_idx = bank_idx
                fallback_count += int(query_mask.sum().item())
            if not query_mask.any():
                continue

            query_idx_all = torch.where(query_mask)[0]
            kk = min(k, pool_idx.numel())
            chunk_size = max(1, int(chunk_size))
            for start in range(0, query_idx_all.numel(), chunk_size):
                query_idx = query_idx_all[start : start + chunk_size]
                base_dist = torch.cdist(z_u[query_idx].detach(), z_u[pool_idx].detach(), p=2)
                dist = base_dist
                label_score_diff = None
                if float(label_penalty) > 0.0 and label_scores is not None:
                    label_score_diff = torch.abs(
                        label_scores[query_idx].unsqueeze(1) - label_scores[pool_idx].unsqueeze(0)
                    )
                    scale = dist.detach().median().clamp(min=self.eps)
                    dist = dist + float(label_penalty) * scale * label_score_diff
                elif float(label_penalty) > 0.0 and labels is not None:
                    mismatch = (labels[query_idx].unsqueeze(1) != labels[pool_idx].unsqueeze(0)).float()
                    scale = dist.detach().median().clamp(min=self.eps)
                    dist = dist + float(label_penalty) * scale * mismatch
                if degree is not None:
                    dist = dist + self.struct_beta * torch.cdist(
                        degree[query_idx], degree[pool_idx], p=1
                    )
                if selection_mode == "rank_fusion" and label_scores is not None:
                    if label_score_diff is None:
                        label_score_diff = torch.abs(
                            label_scores[query_idx].unsqueeze(1) - label_scores[pool_idx].unsqueeze(0)
                        )
                    utility_order = torch.argsort(base_dist, dim=1)
                    utility_rank = torch.empty_like(base_dist)
                    rank_values = torch.arange(base_dist.size(1), device=device, dtype=base_dist.dtype)
                    utility_rank.scatter_(1, utility_order, rank_values.unsqueeze(0).expand_as(base_dist))
                    score_order = torch.argsort(label_score_diff, dim=1)
                    score_rank = torch.empty_like(label_score_diff)
                    score_rank.scatter_(1, score_order, rank_values.unsqueeze(0).expand_as(label_score_diff))
                    denom = max(base_dist.size(1) - 1, 1)
                    dist = utility_rank / denom + float(label_rank_weight) * score_rank / denom

                topk_dist = dist
                candidate_filter = None
                row_filter_valid = None
                if label_score_max_diff is not None and label_scores is not None:
                    if label_score_diff is None:
                        label_score_diff = torch.abs(
                            label_scores[query_idx].unsqueeze(1) - label_scores[pool_idx].unsqueeze(0)
                        )
                    candidate_filter = label_score_diff <= float(label_score_max_diff)
                if require_same_label and labels is not None:
                    query_has_label = labels[query_idx] >= 0
                    same_label = labels[query_idx].unsqueeze(1) == labels[pool_idx].unsqueeze(0)
                    valid_same_label = torch.where(
                        query_has_label.unsqueeze(1),
                        same_label,
                        torch.ones_like(same_label, dtype=torch.bool),
                    )
                    candidate_filter = valid_same_label if candidate_filter is None else (candidate_filter & valid_same_label)
                if candidate_filter is not None:
                    row_filter_valid = candidate_filter.sum(dim=1) >= kk
                    filtered_dist = dist.masked_fill(~candidate_filter, float("inf"))
                    topk_dist = torch.where(row_filter_valid.unsqueeze(1), filtered_dist, dist)
                    label_filter_query_count += int(row_filter_valid.sum().item())
                    label_filter_fallback_count += int((~row_filter_valid).sum().item())
                _, local_top = torch.topk(topk_dist, k=kk, largest=False, dim=1)
                selected = pool_idx[local_top]
                selected_u_dist = base_dist.gather(1, local_top)
                if label_scores is not None:
                    if label_score_diff is None:
                        label_score_diff = torch.abs(
                            label_scores[query_idx].unsqueeze(1) - label_scores[pool_idx].unsqueeze(0)
                        )
                    selected_label_score_diff = label_score_diff.gather(1, local_top)
                    if candidate_filter is not None:
                        selected_filter_valid = candidate_filter.gather(1, local_top)
                        if row_filter_valid is not None:
                            selected_filter_valid = torch.where(
                                row_filter_valid.unsqueeze(1),
                                selected_filter_valid,
                                torch.zeros_like(selected_filter_valid),
                            )
                    else:
                        selected_filter_valid = torch.ones_like(selected_label_score_diff, dtype=torch.bool)
                else:
                    selected_label_score_diff = torch.zeros_like(selected_u_dist)
                    selected_filter_valid = torch.ones_like(selected_u_dist, dtype=torch.bool)
                if kk < k:
                    pad = selected[:, -1:].expand(-1, k - kk)
                    selected = torch.cat([selected, pad], dim=1)
                    selected_u_dist = torch.cat(
                        [selected_u_dist, selected_u_dist[:, -1:].expand(-1, k - kk)], dim=1
                    )
                    selected_label_score_diff = torch.cat(
                        [
                            selected_label_score_diff,
                            selected_label_score_diff[:, -1:].expand(-1, k - kk),
                        ],
                        dim=1,
                    )
                    selected_filter_valid = torch.cat(
                        [selected_filter_valid, selected_filter_valid[:, -1:].expand(-1, k - kk)],
                        dim=1,
                    )
                donor_indices[query_idx] = selected
                donor_u_dist[query_idx] = selected_u_dist
                donor_label_score_diff[query_idx] = selected_label_score_diff
                donor_label_filter_valid[query_idx] = selected_filter_valid
                if degree is not None:
                    donor_degree_diff[query_idx] = torch.abs(degree[query_idx] - degree[selected].squeeze(-1))
                else:
                    donor_degree_diff[query_idx] = 0.0

        opposite_sensitive = (sensitive[donor_indices] != sensitive.unsqueeze(1)).float()
        unique_ratio = []
        for row in donor_indices:
            unique_ratio.append(float(torch.unique(row).numel() / max(k, 1)))
        unique_ratio = torch.tensor(unique_ratio, device=device)

        stats = {
            "num_donors": int(k),
            "bank_size": int(bank_idx.numel()),
            "fallback_ratio": float(fallback_count / max(num_nodes, 1)),
            "use_structure": bool(use_structure),
            "struct_beta": float(self.struct_beta),
            "label_penalty": float(label_penalty),
            "selection_mode": str(selection_mode),
            "label_rank_weight": float(label_rank_weight),
            "label_score_max_diff": None if label_score_max_diff is None else float(label_score_max_diff),
            "require_same_label": bool(require_same_label),
            "uses_soft_label_scores": bool(label_scores is not None and float(label_penalty) > 0.0),
            "uses_label_score_filter": bool(
                (label_score_max_diff is not None and label_scores is not None)
                or (require_same_label and labels is not None)
            ),
            "label_filter_query_ratio": float(label_filter_query_count / max(num_nodes, 1)),
            "label_filter_fallback_ratio": float(label_filter_fallback_count / max(num_nodes, 1)),
            "confidence_score_tau": float(confidence_score_tau),
            "opposite_sensitive_ratio": float(opposite_sensitive.mean().item()),
            "unique_donor_ratio": float(unique_ratio.mean().item()),
            "z_u_dist_mean": float(donor_u_dist.mean().item()),
            "z_u_dist_top1_mean": float(donor_u_dist[:, 0].mean().item()),
            "z_u_dist_std": float(donor_u_dist.std(unbiased=False).item()),
            "degree_diff_mean": float(donor_degree_diff.mean().item()),
            "degree_diff_top1_mean": float(donor_degree_diff[:, 0].mean().item()),
        }
        if label_scores is not None:
            label_score_diff = torch.abs(
                label_scores[donor_indices] - label_scores.unsqueeze(1)
            )
            score_tau = max(float(confidence_score_tau), self.eps)
            score_reliability = torch.exp(-label_score_diff / score_tau).mean(dim=1)
            filter_reliability = donor_label_filter_valid.float().mean(dim=1)
            donor_reliability = score_reliability * filter_reliability
            stats.update(
                {
                    "label_score_diff_mean": float(label_score_diff.mean().item()),
                    "label_score_diff_top1_mean": float(label_score_diff[:, 0].mean().item()),
                    "filtered_label_score_diff_mean": float(donor_label_score_diff.mean().item()),
                    "label_score_filter_valid_ratio": float(donor_label_filter_valid.float().mean().item()),
                    "donor_reliability_mean": float(donor_reliability.mean().item()),
                    "donor_reliability_min": float(donor_reliability.min().item()),
                    "donor_reliability_p10": float(torch.quantile(donor_reliability, 0.10).item()),
                    "donor_reliability_top1_proxy": float(torch.exp(-label_score_diff[:, 0] / score_tau).mean().item()),
                }
            )
        else:
            donor_reliability = donor_label_filter_valid.float().mean(dim=1)
        if labels is not None:
            same_label = (labels[donor_indices] == labels.unsqueeze(1)).float()
            valid_label = labels >= 0
            stats.update(
                {
                    "same_label_ratio": float(same_label.mean().item()),
                    "same_label_top1_ratio": float(same_label[:, 0].mean().item()),
                    "valid_label_query_ratio": float(valid_label.float().mean().item()),
                    "same_label_labeled_ratio": float(same_label[valid_label].mean().item()) if valid_label.any() else 0.0,
                    "same_label_labeled_top1_ratio": float(same_label[valid_label, 0].mean().item()) if valid_label.any() else 0.0,
                }
            )

        self.last_stats = stats
        self.last_reliability = donor_reliability.detach()
        return donor_indices
