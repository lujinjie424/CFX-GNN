"""Representation-level counterfactual intervention operator."""

import torch
import torch.nn.functional as F


class CounterfactualOperator:
    """Apply donor-based identity transplantation through an INN editor."""

    def __init__(self, inn_model, cf_ref_mode="local_quantile", sensitive=None):
        self.inn_model = inn_model
        self.cf_ref_mode = cf_ref_mode
        self.sensitive = sensitive
        self.last_cf_stats = {}

    def configure_reference(self, cf_ref_mode="local_quantile", sensitive=None):
        if cf_ref_mode != "local_quantile":
            raise ValueError("Only the paper local-quantile reference is supported.")
        self.cf_ref_mode = "local_quantile"
        if sensitive is not None:
            self.sensitive = sensitive

    def encode(self, h):
        """Return z_u, z_b for h with shapes [N, d_u] and [N, d_b]."""
        z_u, z_b, _, _, _ = self.inn_model(h)
        return z_u, z_b

    def inverse(self, z_u, z_b):
        """Invert concatenated proxy latents back to h-space."""
        z = torch.cat([z_u, z_b], dim=-1)
        return self.inn_model.inn(z, reverse=True)

    def reconstruction_error(self, h):
        """Mean squared reconstruction error for h -> z -> h."""
        z_u, z_b = self.encode(h)
        h_rec = self.inverse(z_u, z_b)
        return torch.mean((h - h_rec) ** 2)

    @staticmethod
    def _rank_quantiles(values, reference):
        """Hard per-dimension empirical quantiles of values in a reference bank."""
        if reference.size(0) < 2:
            return None
        sorted_ref, _ = torch.sort(reference.detach(), dim=0)
        value_detached = values.detach().contiguous()
        ranks = []
        for dim in range(value_detached.size(1)):
            ranks.append(
                torch.searchsorted(
                    sorted_ref[:, dim].contiguous(),
                    value_detached[:, dim].contiguous(),
                ).float()
            )
        rank = torch.stack(ranks, dim=1)
        return (rank / max(reference.size(0) - 1, 1)).clamp(0.0, 1.0)

    @staticmethod
    def _select_by_quantile(target_bank, quantiles):
        """Select per-dimension target values from a local target bank by quantile."""
        sorted_target, _ = torch.sort(target_bank.detach(), dim=1)
        k = sorted_target.size(1)
        idx = torch.round(quantiles * max(k - 1, 0)).long().clamp(0, max(k - 1, 0))
        return sorted_target.gather(1, idx.unsqueeze(1)).squeeze(1)

    @staticmethod
    def _select_global_by_quantile(target_reference, quantiles):
        """Select per-dimension target values from one global target reference."""
        sorted_target, _ = torch.sort(target_reference.detach(), dim=0)
        k = sorted_target.size(0)
        idx = torch.round(quantiles * max(k - 1, 0)).long().clamp(0, max(k - 1, 0))
        cols = torch.arange(sorted_target.size(1), device=sorted_target.device)
        return sorted_target[idx, cols.unsqueeze(0).expand_as(idx)]

    def _local_quantile_zb(self, z_b_current, donor_z_b):
        sensitive = None if self.sensitive is None else self.sensitive.to(z_b_current.device).long().view(-1)
        if sensitive is None or donor_z_b.size(1) < 2:
            self.last_cf_stats = {
                "cf_ref_mode": "local_quantile",
                "cf_ref_fallback_ratio": 1.0,
                "cf_ref_fallback_reason": "missing_sensitive_or_single_donor",
            }
            return donor_z_b[:, 0, :].detach()

        z_b_cf = donor_z_b[:, 0, :].detach().clone()
        fallback = torch.ones(z_b_current.size(0), dtype=torch.bool, device=z_b_current.device)
        for group in (0, 1):
            group_mask = sensitive == group
            if not group_mask.any():
                continue
            source = z_b_current[group_mask].detach()
            quantiles = self._rank_quantiles(z_b_current[group_mask], source)
            if quantiles is None:
                continue
            z_b_cf[group_mask] = self._select_by_quantile(donor_z_b[group_mask], quantiles)
            fallback[group_mask] = False

        delta_zb = torch.norm(z_b_current.detach() - z_b_cf, p=2, dim=1) / (z_b_current.size(1) ** 0.5)
        self.last_cf_stats = {
            "cf_ref_mode": "local_quantile",
            "cf_ref_fallback_ratio": float(fallback.float().mean().item()),
            "quantile_delta_zb_mean": float(delta_zb.mean().item()),
        }
        return z_b_cf

    def apply(self, h, donor_indices, z_b_bank=None, cf_ref_mode=None):
        if cf_ref_mode not in (None, "local_quantile"):
            raise ValueError("Only the paper local-quantile reference is supported.")

        z_u_current, z_b_current = self.encode(h)
        if z_b_bank is None:
            z_b_bank = z_b_current.detach()
        donor_indices = donor_indices.to(h.device).long()
        donor_z_b = z_b_bank.to(h.device)[donor_indices]
        z_b_cf = self._local_quantile_zb(z_b_current, donor_z_b)
        h_cf = self.inverse(z_u_current, z_b_cf)
        delta_zb = torch.norm(z_b_current.detach() - z_b_cf.detach(), p=2, dim=1) / (z_b_current.size(1) ** 0.5)
        delta_h = torch.norm(h.detach() - h_cf.detach(), p=2, dim=1) / (h.size(1) ** 0.5)
        corr = ((delta_zb - delta_zb.mean()) * (delta_h - delta_h.mean())).mean() / (
            delta_zb.std(unbiased=False).clamp_min(1e-8) * delta_h.std(unbiased=False).clamp_min(1e-8)
        )
        self.last_cf_stats.update(
            {
                "quantile_delta_zb_mean": float(delta_zb.mean().item()),
                "quantile_delta_h_mean": float(delta_h.mean().item()),
                "quantile_corr_zb_h": float(corr.item()),
            }
        )
        return h_cf.unsqueeze(1)

    def global_quantile_zb(self, z_b, sensitive):
        """Per-dimension quantile match from each sensitive group to the opposite group."""
        sensitive = sensitive.to(z_b.device).long().view(-1)
        z_b_cf = z_b.detach().clone()
        fallback = torch.ones(z_b.size(0), dtype=torch.bool, device=z_b.device)
        for src_group, dst_group in ((0, 1), (1, 0)):
            src_mask = sensitive == src_group
            dst_mask = sensitive == dst_group
            if not src_mask.any() or dst_mask.sum() < 2 or src_mask.sum() < 2:
                continue
            quantiles = self._rank_quantiles(z_b[src_mask], z_b[src_mask])
            z_b_cf[src_mask] = self._select_global_by_quantile(z_b[dst_mask], quantiles)
            fallback[src_mask] = False
        return z_b_cf, fallback

    def stage1_map_loss(self, h, z_u, z_b, sensitive, mode="none", mask=None):
        """The paper does not add an auxiliary reference-mapping loss."""
        if mode not in (None, "none"):
            raise ValueError("The paper uses no auxiliary Stage-1 mapping loss.")
        return h.new_tensor(0.0), {}

    @torch.no_grad()
    def zs_h_geometry_diagnostics(self, h, z_u, z_b, sensitive, mask=None, num_samples=2048, knn_values=(10, 20)):
        """Monitor whether z_b counterfactual distances align with h distances.

        This is a pure diagnostic used to inspect the learned geometry; it must
        not affect optimization.
        """
        z_b_cf, fallback = self.global_quantile_zb(z_b, sensitive)
        h_cf = self.inverse(z_u, z_b_cf)
        valid = ~fallback
        if mask is not None:
            valid = valid & mask.to(valid.device).bool()
        idx = valid.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return {
                "enabled": True,
                "num_valid": 0,
                "num_samples": 0,
                "cf_ref_fallback_ratio": float(fallback.float().mean().item()),
            }
        if idx.numel() > int(num_samples):
            perm = torch.randperm(idx.numel(), device=idx.device)[: int(num_samples)]
            idx = idx[perm]

        hs = h[idx]
        hcs = h_cf[idx]
        zbs = z_b[idx]
        zbcfs = z_b_cf[idx]
        d_zs = torch.norm(zbs - zbcfs, p=2, dim=1) / (z_b.size(1) ** 0.5)
        d_h = torch.norm(hs - hcs, p=2, dim=1) / (h.size(1) ** 0.5)
        gain = d_h / d_zs.clamp_min(1e-6)

        def _pearson(a, b):
            if a.numel() < 2:
                return a.new_tensor(0.0)
            ac = a - a.mean()
            bc = b - b.mean()
            return (ac * bc).mean() / (ac.std(unbiased=False).clamp_min(1e-8) * bc.std(unbiased=False).clamp_min(1e-8))

        def _rank(x):
            order = torch.argsort(x)
            ranks = torch.empty_like(order, dtype=torch.float)
            ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float)
            return ranks

        def _spearman(a, b):
            if a.numel() < 2:
                return a.new_tensor(0.0)
            return _pearson(_rank(a), _rank(b))

        def _quantile(x, q):
            if x.numel() == 0:
                return x.new_tensor(0.0)
            return torch.quantile(x.float(), x.new_tensor(float(q)))

        spearman = _spearman(d_zs, d_h)
        shuffled = d_h[torch.randperm(d_h.numel(), device=d_h.device)]
        spearman_shuffle = _spearman(d_zs, shuffled)

        stats = {
            "enabled": True,
            "num_valid": int(valid.sum().item()),
            "num_samples": int(idx.numel()),
            "cf_ref_fallback_ratio": float(fallback.float().mean().item()),
            "d_zs_mean": float(d_zs.mean().item()),
            "d_zs_p10": float(_quantile(d_zs, 0.10).item()),
            "d_zs_p50": float(_quantile(d_zs, 0.50).item()),
            "d_zs_p90": float(_quantile(d_zs, 0.90).item()),
            "d_h_mean": float(d_h.mean().item()),
            "d_h_p10": float(_quantile(d_h, 0.10).item()),
            "d_h_p50": float(_quantile(d_h, 0.50).item()),
            "d_h_p90": float(_quantile(d_h, 0.90).item()),
            "dist_pearson": float(_pearson(d_zs, d_h).item()),
            "dist_spearman": float(spearman.item()),
            "dist_spearman_shuffle": float(spearman_shuffle.item()),
            "dist_spearman_lift": float((spearman - spearman_shuffle).item()),
            "gain_mean": float(gain.mean().item()),
            "gain_p10": float(_quantile(gain, 0.10).item()),
            "gain_p50": float(_quantile(gain, 0.50).item()),
            "gain_p90": float(_quantile(gain, 0.90).item()),
            "gain_max": float(gain.max().item()),
            "gain_out_of_range_ratio": float(((gain < 0.1) | (gain > 5.0)).float().mean().item()),
        }

        n = idx.numel()
        if n > 1:
            z_dist = torch.cdist(zbs.float(), zbs.float())
            h_dist = torch.cdist(hs.float(), hs.float())
            eye = torch.eye(n, device=h.device, dtype=torch.bool)
            z_dist.masked_fill_(eye, float("inf"))
            h_dist.masked_fill_(eye, float("inf"))
            for k in knn_values:
                k_eff = min(int(k), n - 1)
                if k_eff <= 0:
                    stats[f"knn_jaccard_{int(k)}"] = 0.0
                    continue
                z_nn = torch.topk(z_dist, k=k_eff, dim=1, largest=False).indices
                h_nn = torch.topk(h_dist, k=k_eff, dim=1, largest=False).indices
                overlap = torch.zeros(n, device=h.device)
                for row in range(n):
                    overlap[row] = torch.isin(z_nn[row], h_nn[row]).float().sum()
                stats[f"knn_jaccard_{int(k)}"] = float((overlap / (2 * k_eff - overlap).clamp_min(1.0)).mean().item())
        else:
            for k in knn_values:
                stats[f"knn_jaccard_{int(k)}"] = 0.0
        return stats
