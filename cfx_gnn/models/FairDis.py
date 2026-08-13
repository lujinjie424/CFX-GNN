import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.half_dim = input_dim // 2

        self.s_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half_dim),
        )
        self.scale = nn.Parameter(torch.tensor(-2.0))

        self.t_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half_dim)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]

        if not reverse:
            s_raw = self.s_net(x1)
            s = torch.tanh(s_raw) * F.softplus(self.scale)

            t = self.t_net(x1)
            y2 = x2 * torch.exp(s) + t
            return torch.cat([x1, y2], dim=1), torch.sum(s, dim=1)
        else:
            s_raw = self.s_net(x1)
            s = torch.tanh(s_raw) * F.softplus(self.scale)

            t = self.t_net(x1)
            x2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, x2], dim=1)


class INN_Node(nn.Module):
    def __init__(self, input_dim, num_layers=4):
        super().__init__()
        assert input_dim % 2 == 0, "Input dimension must be even for simple coupling."

        self.layers = nn.ModuleList([
            CouplingLayer(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x, reverse=False):
        log_det_jacobian = 0

        if not reverse:
            for i, layer in enumerate(self.layers):
                x, ld = layer(x, reverse=False)
                log_det_jacobian += ld

                if i < len(self.layers) - 1:
                    x = x.flip(dims=[1])
            return x, log_det_jacobian

        else:
            for i, layer in enumerate(reversed(self.layers)):
                if i > 0:
                    x = x.flip(dims=[1])

                x = layer(x, reverse=True)
            return x

class FairLatentLoss(nn.Module):
    def __init__(self, z_dim, epsilon=1e-6):
        super().__init__()
        self.z_dim = z_dim
        self.epsilon = epsilon
        self.register_buffer("I", torch.eye(z_dim))

    def compute_covariance(self, z):
        batch_size = z.size(0)
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = torch.matmul(z_centered.T, z_centered) / (batch_size - 1)
        return cov

    def loss_orth(self, z_u, z_b, eps=1e-8):
        """
        Orthogonality loss between task-dominant and bias-dominant representations.

        z_u: [N, d_u]
        z_b: [N, d_b]

        If d_u == d_b, we compute instance-wise cosine similarity.
        If d_u != d_b, use covariance-based orthogonality instead.
        """
        if z_u.size(1) == z_b.size(1):
            z_u_norm = F.normalize(z_u, p=2, dim=1, eps=eps)
            z_b_norm = F.normalize(z_b, p=2, dim=1, eps=eps)
            cos = (z_u_norm * z_b_norm).sum(dim=1)
            return cos.abs().mean()
        else:
            z_u_centered = z_u - z_u.mean(dim=0, keepdim=True)
            z_b_centered = z_b - z_b.mean(dim=0, keepdim=True)

            n = z_u.size(0)
            cross_cov = torch.matmul(z_u_centered.T, z_b_centered) / max(n - 1, 1)

            return cross_cov.pow(2).mean()

    def loss_dg(self, z):
        """
        Diagonalizing Loss (L_dg) [cite: 164]
        目标：使非对角元素趋于0，去相关。
        """
        C = self.compute_covariance(z)
        off_diag = C - C * self.I
        l_dg = (off_diag.pow(2)).sum() / self.z_dim
        return l_dg

    def loss_eq(self, z, c=0.5):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.epsilon)
        return torch.relu(c - std).mean()

    def loss_di(self, z, labels, sensitives):
        """
        Distance Loss (L_di) [cite: 174]
        目标：
        1. 相同标签(Y)但不同敏感属性(S) -> 最小化距离 (Mask_max)
        2. 相同敏感属性(S)但不同标签(Y) -> 最大化距离 (Mask_min)
        """
        batch_size = z.size(0)

        diff = z.unsqueeze(1) - z.unsqueeze(0)
        dist_sq = diff.pow(2).sum(dim=2)

        D_val = torch.log((dist_sq + 1) / (dist_sq + self.epsilon))

        label_match = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        sens_match = (sensitives.unsqueeze(1) == sensitives.unsqueeze(0)).float()

        M_max = label_match * (1 - sens_match)

        M_min = sens_match * (1 - label_match)

        mask_identity = torch.eye(batch_size).to(z.device)
        M_max = M_max * (1 - mask_identity)
        M_min = M_min * (1 - mask_identity)

        sum_m_max = M_max.sum() + 1e-8
        sum_m_min = M_min.sum() + 1e-8

        term1 = - (1.0 / sum_m_max) * (M_max * D_val).sum()
        term2 = (1.0 / sum_m_min) * (M_min * D_val).sum()

        return term1 + term2

    def compute_kernel(self, x, y):
        dim = x.size(1)
        dist = torch.cdist(x, x, p=2).pow(2)
        sigma = torch.median(dist).detach()
        if sigma == 0: sigma = 1.0

        return torch.exp(-dist / (2 * sigma))

    def loss_hsic(self, z_y, z_s):
        """
        最大化 Z_y 和 Z_s 的独立性 (最小化 HSIC 值)
        """
        n = z_y.size(0)

        K = self.compute_kernel(z_y, z_y)
        L = self.compute_kernel(z_s, z_s)

        H = torch.eye(n, device=z_y.device) - (1.0 / n) * torch.ones((n, n), device=z_y.device)

        hsic = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)

        return hsic


class FairINNModel(nn.Module):
    def __init__(self, input_dim, y_dim, s_dim):
        """
        input_dim: 预训练嵌入的维度
        y_dim: 分配给 Z^Y 的维度 [cite: 182]
        s_dim: 分配给 Z^S 的维度
        """
        super().__init__()
        self.inn = INN_Node(input_dim)
        self.y_dim = y_dim
        self.s_dim = s_dim

        assert y_dim + s_dim == input_dim

        self.classifier_y = nn.Sequential(
            nn.Linear(y_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.classifier_s = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

        self.loss_fn = FairLatentLoss(y_dim)

    def forward(self, e):
        z, log_det = self.inn(e)

        z_y = z[:, :self.y_dim]
        z_s = z[:, self.y_dim:]

        pred_y = self.classifier_y(z_y)
        pred_s = self.classifier_s(z_s)

        return z_y, z_s, pred_s, pred_y, log_det

    def unit(self, v, eps=1e-12):
        return v / (v.norm(p=2) + eps)

    @torch.no_grad()
    def get_dirs(self):
        hy = self.unit(self.classifier_y.weight[0])
        hs = self.unit(self.classifier_s.weight[0])
        return hy, hs



class StructureMask(nn.Module):
    """邻居掩码生成器：接受每条边的拼接输入，预测 mask 的 Logits"""

    def __init__(self, input_dim, hidden_dim, init_bias=3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        self.init_bias = init_bias
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.out_features == 1:
                    nn.init.constant_(m.bias, self.init_bias)
                else:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, edge_feat):
        """
        returns scores: [E, 1] (Logits, range: -inf ~ +inf)
        """
        scores = self.net(edge_feat)

        return scores


class FeatureMask(nn.Module):
    def __init__(self, feature_dim, emb_dim):
        super().__init__()
        self.hidden_dim = 64
        self.input_dim = feature_dim + 2 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        last_layer = self.net[-1]

        nn.init.normal_(last_layer.weight, mean=0, std=0.001)

        nn.init.constant_(last_layer.bias, 3.0)

    def forward(self, x, e, e_cf):

        delta_e = e_cf - e
        emb = torch.cat([x, e, delta_e], dim=1)

        mask_logits = self.net(emb)
        mask = torch.sigmoid(mask_logits)

        return mask


def supcon_loss(z, labels, temperature=0.07):
    """
    真正的 Supervised Contrastive Loss (InfoNCE 变体)
    """
    z = F.normalize(z, dim=1)

    sim_matrix = torch.mm(z, z.T)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(z.shape[0]).view(-1, 1).to(z.device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(sim_matrix / temperature) * logits_mask

    log_prob = sim_matrix / temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    loss = - mean_log_prob_pos.mean()

    return loss

class self_explainer(nn.Module):
    def __init__(self, pret_encoder, input_dim, hidden_dim, class_num, isINN=True, dropout=0, num_channels=8, mask_hidden=64):
        super().__init__()
        self.encoder = pret_encoder

        self.classifier = nn.Linear(hidden_dim // 2, 1)


        self.fairINN = FairINNModel(hidden_dim , hidden_dim // 2, hidden_dim // 2)

        self.featureMask_Generator = FeatureMask(input_dim, hidden_dim)
        self.structureMask_Generator = StructureMask(hidden_dim  * 4, hidden_dim )

    def Disentangler(self, g, x, y, s, train_mask, pos_weight=None, phase='Disentangler'):

        if phase == 'Disentangler':
            h = self.encoder(g, x)
        elif phase == 'Prediction':
            with torch.no_grad():
                h = self.encoder(g, x)


        z_y, z_s, pred_s, pred_y, log_det = self.fairINN(h)

        z_full = torch.cat([z_y, z_s], dim=1)
        l_g = torch.mean(0.5 * torch.sum(z_full ** 2, dim=1) - log_det)

        l_cls_y = F.binary_cross_entropy_with_logits(pred_y[train_mask], y[train_mask].unsqueeze(1).float())
        pos_weight_tensor = torch.tensor([pos_weight]).to(s.device)
        l_cls_s = F.binary_cross_entropy_with_logits(pred_s[train_mask], s[train_mask].unsqueeze(1).float())
        l_con_y = supcon_loss(z_y[train_mask], y[train_mask])
        l_con_s = supcon_loss(z_s[train_mask], s[train_mask])

        loss_module = self.fairINN.loss_fn
        l_dg_y = loss_module.loss_dg(z_y)
        l_eq_y = loss_module.loss_eq(z_y)
        l_dg_s = loss_module.loss_dg(z_s)
        l_eq_s = loss_module.loss_eq(z_s)
        l_eq = 0.5 * l_eq_s + 0.5 * l_eq_y
        l_di_y = loss_module.loss_di(z_y[train_mask], y[train_mask], s[train_mask])
        l_di_s = loss_module.loss_di(z_s[train_mask], s[train_mask], y[train_mask])
        l_orth = loss_module.loss_orth(z_y, z_s)
        l_independence = loss_module.loss_hsic(z_y[train_mask], z_s[train_mask])

        return l_con_y,l_con_s, l_cls_y, l_cls_s, l_g,l_di_y,l_di_s, l_eq, l_independence,h, z_y, z_s, l_orth

    @torch.no_grad()
    def get_emb_dis(self, g, x):
        self.encoder.eval()
        self.fairINN.eval()
        with torch.no_grad():
            h = self.encoder(g, x)
            z_y, z_s, pred_s, pred_y, log_det = self.fairINN(h)

        return h, z_y, z_s


    def generate_nearest_neighbor_counterfactual(self, z_s, z_y, sens, labels, train_mask):
        """
        [极速版] 基于矩阵运算的最近邻反事实生成。
        移除了所有 Python 循环，使用 torch.cdist 并行计算距离。
        """
        self.fairINN.eval()

        z_y_train = z_y[train_mask].detach()
        z_s_train = z_s[train_mask].detach()
        s_train = sens[train_mask].detach()

        mask_0 = (s_train == 0)
        mask_1 = (s_train == 1)

        z_y_pool_0 = z_y_train[mask_0]
        z_s_pool_0 = z_s_train[mask_0]

        z_y_pool_1 = z_y_train[mask_1]
        z_s_pool_1 = z_s_train[mask_1]


        current_s = sens.long()


        z_s_new = z_s.clone()

        query_mask_0 = (current_s == 0)
        if query_mask_0.any() and len(z_y_pool_1) > 0:
            queries_0 = z_y[query_mask_0]

            dists = torch.cdist(queries_0, z_y_pool_1, p=2)

            min_indices = torch.argmin(dists, dim=1)

            z_s_matched = z_s_pool_1[min_indices]

            z_s_new[query_mask_0] = z_s_matched

        query_mask_1 = (current_s == 1)
        if query_mask_1.any() and len(z_y_pool_0) > 0:
            queries_1 = z_y[query_mask_1]

            dists = torch.cdist(queries_1, z_y_pool_0, p=2)
            min_indices = torch.argmin(dists, dim=1)
            z_s_matched = z_s_pool_0[min_indices]

            z_s_new[query_mask_1] = z_s_matched

        z_cf = torch.cat([z_y, z_s_new], dim=1)
        e_cf = self.fairINN.inn(z_cf, reverse=True)

        return e_cf, z_y, z_s_new

    import torch
    import torch.nn.functional as F

    def generate_nearest_neighbor_counterfactual_with_R(
            self, z_s, z_y, sens, labels, train_mask, g=None,
            top_k=5, donor_tau=0.5, fallback_label_penalty=0.25,
            reliability_floor=0.05):
        """
        Construct a plausible counterfactual latent by softly averaging a
        same-label/opposite-sensitive donor set. The INN is treated as a latent
        editor; donor reliability is diagnostic weak-supervision weight, not
        ground-truth counterfactual validity.
        """
        self.fairINN.eval()

        device = z_y.device
        eps = 1e-8
        top_k = int(getattr(self, "donor_top_k", top_k))
        donor_tau = float(getattr(self, "donor_tau", donor_tau))
        fallback_label_penalty = float(getattr(self, "donor_fallback_label_penalty", fallback_label_penalty))
        reliability_floor = float(getattr(self, "donor_reliability_floor", reliability_floor))
        R_scores = torch.zeros(z_y.size(0), device=device)
        entropy_scores = torch.zeros(z_y.size(0), device=device)
        fallback_scores = torch.zeros(z_y.size(0), device=device)
        z_s_new = z_s.clone()

        train_indices = torch.where(train_mask)[0].to(device)
        z_y_train = z_y[train_mask].detach()
        z_s_train = z_s[train_mask].detach()
        s_train = sens[train_mask].detach()
        y_train = labels[train_mask].detach()

        if g is not None:
            with torch.no_grad():
                in_deg = g.in_degrees().float().to(device)
                out_deg = g.out_degrees().float().to(device)
                degree = (in_deg + out_deg).clamp(min=1.0)
                src, dst = g.edges()
                src_s = sens[src].float()
                same_src_dst = (sens[src] == sens[dst]).float()
                same_sum = torch.zeros(g.num_nodes(), device=device).scatter_add_(0, dst, same_src_dst)
                neigh_s_sum = torch.zeros(g.num_nodes(), device=device).scatter_add_(0, dst, src_s)
                same_ratio = same_sum / degree
                neigh_s_mean = neigh_s_sum / degree
                topo_all = torch.stack(
                    [torch.log1p(degree), same_ratio, neigh_s_mean],
                    dim=1,
                )
        else:
            topo_all = None

        def select_pool(target_sensitive):
            pool_mask = s_train == target_sensitive
            return (
                z_y_train[pool_mask],
                z_s_train[pool_mask],
                y_train[pool_mask],
                train_indices[pool_mask],
            )

        def process_batch(query_mask, target_sensitive):
            if not query_mask.any():
                return

            pool_zy, pool_zs, pool_y, pool_nodes = select_pool(target_sensitive)
            if pool_zy.numel() == 0:
                return

            queries = z_y[query_mask]
            query_labels = labels[query_mask]
            query_nodes = torch.where(query_mask)[0]
            dists = torch.cdist(queries, pool_zy, p=2)
            same_label = query_labels.unsqueeze(1) == pool_y.unsqueeze(0)
            has_same_label = same_label.any(dim=1)

            label_compat = torch.where(
                same_label,
                torch.ones_like(dists),
                torch.full_like(dists, fallback_label_penalty),
            )
            masked_dists = dists.masked_fill(~same_label, float("inf"))
            search_dists = torch.where(has_same_label.unsqueeze(1), masked_dists, dists)

            k = min(top_k, pool_zy.size(0))
            top_vals, top_idx = torch.topk(search_dists, k=k, largest=False, dim=1)
            finite = torch.isfinite(top_vals)
            top_vals = top_vals.masked_fill(~finite, 1e6)

            weights = torch.softmax(-top_vals / max(donor_tau, eps), dim=1) * finite.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=eps)

            selected_zs = pool_zs[top_idx]
            selected_zy = pool_zy[top_idx]
            selected_labels = pool_y[top_idx]
            selected_label_compat = label_compat.gather(1, top_idx)

            util_sim = (1.0 + F.cosine_similarity(
                queries.unsqueeze(1).expand_as(selected_zy),
                selected_zy,
                dim=2,
            )) / 2.0

            if topo_all is not None:
                q_topo = topo_all[query_nodes].unsqueeze(1)
                d_topo = topo_all[pool_nodes[top_idx]]
                topo_dist = torch.norm(q_topo - d_topo, p=2, dim=2)
                topo_compat = torch.exp(-topo_dist)
            else:
                topo_compat = torch.ones_like(util_sim)

            weighted_zs = (weights.unsqueeze(-1) * selected_zs).sum(dim=1)
            z_s_new[query_mask] = weighted_zs

            label_match = (query_labels.unsqueeze(1) == selected_labels).float()
            raw_reliability = (
                weights * util_sim.detach() * selected_label_compat.detach() * topo_compat.detach()
            ).sum(dim=1)
            raw_reliability = raw_reliability * torch.where(
                has_same_label,
                torch.ones_like(raw_reliability),
                torch.full_like(raw_reliability, fallback_label_penalty),
            )
            entropy = -(weights * torch.log(weights.clamp(min=eps))).sum(dim=1)
            entropy_norm = entropy / np.log(max(k, 2))
            uncertainty_discount = (1.0 - entropy_norm).clamp(min=0.0, max=1.0)

            R_scores[query_mask] = raw_reliability * uncertainty_discount
            entropy_scores[query_mask] = entropy_norm
            fallback_scores[query_mask] = (~has_same_label).float()

        current_s = sens.long()
        valid_label = labels != -1
        process_batch((current_s == 0) & valid_label, target_sensitive=1)
        process_batch((current_s == 1) & valid_label, target_sensitive=0)

        z_cf = torch.cat([z_y, z_s_new], dim=1)
        e_cf = self.fairINN.inn(z_cf, reverse=True)

        train_weight = torch.where(
            R_scores > 0,
            R_scores.clamp(min=reliability_floor),
            torch.zeros_like(R_scores),
        )
        self.last_donor_diag = {
            "raw_mean": float(R_scores.detach().mean().item()),
            "raw_std": float(R_scores.detach().std(unbiased=False).item()),
            "raw_nonzero_ratio": float((R_scores.detach() > 0).float().mean().item()),
            "train_weight_mean": float(train_weight.detach().mean().item()),
            "train_weight_nonzero_ratio": float((train_weight.detach() > 0).float().mean().item()),
            "fallback_ratio": float(fallback_scores.detach().mean().item()),
            "entropy_mean": float(entropy_scores.detach().mean().item()),
            "top_k": int(top_k),
        }

        return e_cf, z_y, z_s_new, train_weight

    def get_logits(self, g, x):
        h = self.encoder(g, x)

        z_y, z_s, logits_s, logits_y, log_det = self.fairINN(h)

        return h, z_y, z_s, logits_s, logits_y

    def kl_softmax(self, p_logits, q_logits, T=1.0):
        """ KL( softmax(p/T) || softmax(q/T) ). """
        p = F.log_softmax(p_logits / T, dim=-1)
        q = F.softmax(q_logits / T, dim=-1)
        return F.kl_div(p, q, reduction="batchmean") * (T * T)

    def logit_l2(self,p, q):
        return ((p - q) ** 2).mean()

    def compute_cosine_loss(self, emb1, emb2):
        return 1.0 - F.cosine_similarity(emb1, emb2, dim=-1).mean()

    @torch.no_grad()
    def get_x_base(self, x, train_mask):
        xb = x[train_mask].mean(dim=0, keepdim=True)
        return xb

    def calculate_geometric_weights(self, e, e_cf):
        with torch.no_grad():
            dist = torch.norm(e - e_cf, p=2, dim=1)

            mean_dist = dist.mean() + 1e-8
            weights = 1.0 + 0.5 * (dist / mean_dist)


        return weights

    def mask_range_loss(self, keep_ratio, target_range):
        min_keep, max_keep = target_range
        return F.relu(torch.as_tensor(min_keep, device=keep_ratio.device) - keep_ratio).pow(2) + \
            F.relu(keep_ratio - torch.as_tensor(max_keep, device=keep_ratio.device)).pow(2)

    def get_mask_target_ranges(self):
        return getattr(self, "mask_feature_target_range", (0.10, 0.30)), \
            getattr(self, "mask_structure_target_range", (0.10, 0.30))

    def compute_feature_mask_loss(self, g, x, sens, train_mask, labels):
        e = self.encoder(g, x)

        z_y, z_s, _, _, _ = self.fairINN(e)
        e_cf, _, _, R = self.generate_nearest_neighbor_counterfactual_with_R(
            z_s, z_y, sens, labels, train_mask, g=g
        )
        weights = R.detach()

        mask = self.featureMask_Generator(x, e, e_cf)
        mask = mask

        x_base = self.get_x_base(x, train_mask)
        x_tilde = mask * x + (1 - mask) * x_base

        x_masked = x.clone()
        x_masked = x_tilde
        e_tilde = self.encoder(g, x_masked)

        e_tilde_tr = e_tilde
        z_y_t, z_s_t, _, _, _ = self.fairINN(e_tilde_tr)
        e_tilde_cf, _, _, R_tilde = self.generate_nearest_neighbor_counterfactual_with_R(
            z_s_t, z_y_t, sens, labels, train_mask, g=g
        )



        e_tilde_train = e_tilde

        z_train = z_y.detach()
        z_y_t_train = z_y_t
        e_train = e.detach()

        loss_suf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_train, e_tilde_train, dim=1)
        L_suf = loss_suf_elem.mean()

        loss_cf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_tilde_train, e_tilde_cf, dim=1)
        cf_weight = torch.maximum(weights, R_tilde.detach())
        valid_cf = cf_weight > 0
        if valid_cf.any():
            L_cf = (loss_cf_elem[valid_cf] * cf_weight[valid_cf]).sum() / cf_weight[valid_cf].sum().clamp(min=1e-8)
        else:
            L_cf = torch.tensor(0.0, device=x.device)

        feature_target_range, _ = self.get_mask_target_ranges()
        feature_candidates = torch.ones_like(mask, dtype=torch.bool)
        sens_idx = getattr(self, "current_sens_index", None)
        if sens_idx is not None:
            feature_candidates[:, sens_idx] = False
        keep_ratio = mask[feature_candidates].mean() if feature_candidates.any() else mask.mean()
        L_sp = self.mask_range_loss(keep_ratio, feature_target_range)
        L_bin = (mask * (1 - mask)).mean()

        return L_suf, L_cf, L_sp, L_bin


    def compute_structure_mask_loss_Gumbel(self, g, x, sens, train_mask, labels, mask_threshold=0.3, tau=1.0, eps=1e-6):
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        e = self.encoder(g, x)

        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e)
            e_cf, _, _, R = self.generate_nearest_neighbor_counterfactual_with_R(
                z_s, z_y, sens, labels, train_mask, g=g
            )

        weights = R.detach()


        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()


        if self.training:
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        e_mask = self.encoder(g, x, edge_weight=structure_mask)
        z_y_t, z_s_t, _, _, _ = self.fairINN(e_mask)

        e_train = e.detach()
        e_mask_train = e_mask
        loss_suf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_train, e_mask_train, dim=1)



        z_y_train = z_y[train_mask].detach()
        z_y_t_train = z_y_t[train_mask]

        L_suf = loss_suf_elem.mean()


        with torch.no_grad():
            e_mask_cf, _, _, R_mask = self.generate_nearest_neighbor_counterfactual_with_R(
                z_s_t, z_y_t, sens, labels, train_mask, g=g
            )
            e_mask_cf = e_mask_cf.detach()

        loss_cf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_mask_train, e_mask_cf, dim=1)
        cf_weight = torch.maximum(weights, R_mask.detach())
        valid_cf = cf_weight > 0
        if valid_cf.any():
            L_cf = (loss_cf_elem[valid_cf] * cf_weight[valid_cf]).sum() / cf_weight[valid_cf].sum().clamp(min=1e-8)
        else:
            L_cf = torch.tensor(0.0, device=x.device)

        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            _, structure_target_range = self.get_mask_target_ranges()
            L_sp_structure = self.mask_range_loss(m_rest_soft.mean(), structure_target_range)
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        with torch.no_grad():
            if (~is_self_loop).any():
                real_sparsity_hard = 1.0 - y_hard01[~is_self_loop].mean().item()
                kept_ratio_hard = y_hard01[~is_self_loop].mean().item()
            else:
                real_sparsity_hard = 0.0
                kept_ratio_hard = 1.0

            kept_ratio_soft = m_rest_soft.mean().item() if m_rest_soft.numel() > 0 else 1.0

        count = {
            "mask_mean": structure_mask.mean().item(),
            "mask_min": structure_mask.min().item(),
            "mask_max": structure_mask.max().item(),

            "kept_ratio_hard": kept_ratio_hard,
            "sparsity_hard": real_sparsity_hard,
            "kept_ratio_soft": kept_ratio_soft,

            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            "edge_weight_sum": structure_mask.detach().sum().item(),
        }

        return L_suf, L_cf, L_sp_structure, count


    def compute_mask_loss_Gumbel(self, g, x, sens, train_mask, labels, mask_threshold=0.3, tau=1.0, eps=1e-6):
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        e = self.encoder(g, x)

        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e)
            e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)

        weights_all = self.calculate_geometric_weights(e, e_cf)
        weights = weights_all[train_mask].detach()

        feature_mask = self.featureMask_Generator(x, e, e_cf)

        x_base = self.get_x_base(x, train_mask)
        x_tilde = feature_mask * x + (1 - feature_mask) * x_base

        x_masked = x_tilde

        L_sp_feature = feature_mask.mean()
        L_bin = (feature_mask * (1 - feature_mask)).mean()

        e_tilde = self.encoder(g, x_masked)
        with torch.no_grad():
            z_y_tilde, z_s_tilde, _, _, _ = self.fairINN(e_tilde)
            e_cf_tilde, _, _ = self.generate_nearest_neighbor_counterfactual(z_s_tilde, z_y_tilde, sens, train_mask)

        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e_tilde[src], e_tilde[dst], e_cf_tilde[src], e_cf_tilde[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()


        if self.training:
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        e_mask = self.encoder(g, x_masked, edge_weight=structure_mask)
        z_y_t, z_s_t, _, _, _ = self.fairINN(e_mask)

        e_train = e[train_mask].detach()
        e_mask_train = e_mask[train_mask]
        loss_suf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_train, e_mask_train, dim=1)

        L_suf = (loss_suf_elem * weights).mean()


        with torch.no_grad():
            e_mask_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s_t, z_y_t, sens, labels, train_mask)
            e_mask_cf = e_mask_cf[train_mask].detach()

        loss_cf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_mask_train, e_mask_cf, dim=1)
        L_cf = (loss_cf_elem * weights).mean()

        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            L_sp_structure = m_rest_soft.mean()
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        with torch.no_grad():
            if (~is_self_loop).any():
                real_sparsity_hard = 1.0 - y_hard01[~is_self_loop].mean().item()
                kept_ratio_hard = y_hard01[~is_self_loop].mean().item()
            else:
                real_sparsity_hard = 0.0
                kept_ratio_hard = 1.0

            kept_ratio_soft = m_rest_soft.mean().item() if m_rest_soft.numel() > 0 else 1.0

        count = {
            "mask_mean": structure_mask.mean().item(),
            "mask_min": structure_mask.min().item(),
            "mask_max": structure_mask.max().item(),

            "kept_ratio_hard": kept_ratio_hard,
            "sparsity_hard": real_sparsity_hard,
            "kept_ratio_soft": kept_ratio_soft,

            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            "edge_weight_sum": structure_mask.detach().sum().item(),
        }

        return L_suf, L_cf, L_sp_structure, L_sp_feature, L_bin, count

    import torch

    def weighted_contrastive_loss(self, q, k, weights, tau=0.1):
        """
        [辅助方法] 加权对比损失 (Weighted InfoNCE Loss)
        q: Query Embedding (e.g., e_mask) [N, D]
        k: Key Embedding (e.g., e_origin) [N, D]
        weights: Sample weights [N]
        tau: Temperature coefficient
        """
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        sim_matrix = torch.mm(q, k.t()) / tau

        batch_size = q.shape[0]
        labels = torch.arange(batch_size, device=q.device)

        loss_ce = F.cross_entropy(sim_matrix, labels, reduction='none')

        loss = (loss_ce * weights).mean()

        return loss

    def compute_structure_mask_loss_contrastive(self, g, x, sens, train_mask, labels, mask_threshold=0.3, tau=1.0, eps=1e-6,
                                                con_tau=0.1):
        """
        基于对比学习 (Contrastive Learning) 的结构掩码损失计算
        Args:
            con_tau: 对比学习的温度系数 (建议 0.07 ~ 0.2)
        """
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        e_origin = self.encoder(g, x)
        e_origin_detach = e_origin.detach()

        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e_origin)
            e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)

        weights_all = self.calculate_geometric_weights(e_origin, e_cf)
        weights = weights_all[train_mask].detach()

        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e_origin[src], e_origin[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()


        if self.training:
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        e_mask = self.encoder(g, x, edge_weight=structure_mask)

        z_y_t, z_s_t, _, pred_y_t, _ = self.fairINN(e_mask)


        q_mask = e_mask[train_mask]
        k_origin = e_origin_detach[train_mask]
        k_cf = e_cf[train_mask]


        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights, tau=con_tau)

        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            L_sp_structure = m_rest_soft.mean()
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        with torch.no_grad():
            if (~is_self_loop).any():
                real_sparsity_hard = 1.0 - y_hard01[~is_self_loop].mean().item()
                kept_ratio_hard = y_hard01[~is_self_loop].mean().item()
            else:
                real_sparsity_hard = 0.0
                kept_ratio_hard = 1.0

            kept_ratio_soft = m_rest_soft.mean().item() if m_rest_soft.numel() > 0 else 1.0

        count = {
            "mask_mean": structure_mask.mean().item(),
            "mask_min": structure_mask.min().item(),
            "mask_max": structure_mask.max().item(),

            "kept_ratio_hard": kept_ratio_hard,
            "sparsity_hard": real_sparsity_hard,
            "kept_ratio_soft": kept_ratio_soft,

            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            "edge_weight_sum": structure_mask.detach().sum().item(),

            "L_suf_con": L_suf.item(),
            "L_cf_con": L_cf.item()
        }

        return L_suf, L_cf, L_sp_structure, count

    def compute_feature_mask_loss_contrastive(self, g, x, sens, train_mask, labels, con_tau=0.5):
        """
        基于对比学习 (Contrastive Learning) 的特征掩码损失计算
        Args:
            con_tau: 对比学习的温度系数
        """
        e = self.encoder(g, x)
        e_detach = e.detach()

        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e)
            e_cf, _, _, R = self.generate_nearest_neighbor_counterfactual_with_R(z_s, z_y, sens, labels, train_mask)

        weights_all = self.calculate_geometric_weights(e, e_cf)
        weights = weights_all[train_mask].detach()

        mask = self.featureMask_Generator(x, e, e_cf)
        mask = mask[train_mask]

        x_base = self.get_x_base(x, train_mask)
        x_tilde = mask * x[train_mask] + (1 - mask) * x_base


        x_masked_full = x.clone()
        x_masked_full[train_mask] = x_tilde

        e_tilde = self.encoder(g, x_masked_full)
        _,_,_,logits_y, _ = self.fairINN(e_tilde)


        q_mask = e_tilde[train_mask]
        k_origin = e_detach[train_mask]
        k_cf = e_cf[train_mask]



        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights, tau=con_tau)

        L_sp = mask.mean()

        L_bin = (mask * (1 - mask)).mean()

        count = {
            "feat_mask_mean": mask.mean().item(),
            "feat_mask_min": mask.min().item(),
            "feat_mask_max": mask.max().item(),
            "feat_weight_mean": weights.mean().item(),
            "L_feat_suf_con": L_suf.item(),
            "L_feat_cf_con": L_cf.item()
        }

        return L_suf, L_cf, L_sp, L_bin


    def compute_feature_mask_loss_contrastive_ete(self, g, x, e, z_y, z_s, e_cf, R, sens, train_mask, labels, con_tau=0.5):
        """
        基于对比学习 (Contrastive Learning) 的特征掩码损失计算
        Args:
            con_tau: 对比学习的温度系数
        """


        weights_all = self.calculate_geometric_weights(e, e_cf)
        weights = weights_all[train_mask].detach()
        R_score = R[train_mask].detach()
        weights_cf = R_score * weights

        mask = self.featureMask_Generator(x, e, e_cf)
        mask = mask[train_mask]

        x_base = self.get_x_base(x, train_mask)
        x_tilde = mask * x[train_mask] + (1 - mask) * x_base


        x_masked_full = x.clone()
        x_masked_full[train_mask] = x_tilde

        e_tilde = self.encoder(g, x_masked_full)


        e_detach = e.detach()

        q_mask = e_tilde[train_mask]
        k_origin = e_detach[train_mask]
        k_cf = e_cf[train_mask]



        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights_cf, tau=con_tau)

        L_sp = mask.mean()

        L_bin = (mask * (1 - mask)).mean()

        count = {
            "feat_mask_mean": mask.mean().item(),
            "feat_mask_min": mask.min().item(),
            "feat_mask_max": mask.max().item(),
            "feat_weight_mean": weights.mean().item(),
            "L_feat_suf_con": L_suf.item(),
            "L_feat_cf_con": L_cf.item()
        }

        return x_masked_full, L_suf, L_cf, L_sp, L_bin



    def compute_structure_mask_loss_contrastive_ete(self, g, x, e, z_y, z_s, e_cf, R, sens, train_mask, labels, mask_threshold=0.3, tau=1.0, eps=1e-6,
                                                con_tau=0.1):
        """
        基于对比学习 (Contrastive Learning) 的结构掩码损失计算
        Args:
            con_tau: 对比学习的温度系数 (建议 0.07 ~ 0.2)
        """
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        e_origin = e
        e_origin_detach = e_origin.detach()



        weights_all = self.calculate_geometric_weights(e_origin, e_cf)
        weights = weights_all[train_mask].detach()
        R_score = R[train_mask].detach()
        weights_cf = R_score * weights


        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e_origin[src], e_origin[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()


        if self.training:
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        e_mask = self.encoder(g, x, edge_weight=structure_mask)

        z_y_t, z_s_t, _, pred_y_t, _ = self.fairINN(e_mask)


        q_mask = e_mask[train_mask]
        k_origin = e_origin_detach[train_mask]
        k_cf = e_cf[train_mask]


        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights_cf, tau=con_tau)

        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            L_sp_structure = m_rest_soft.mean()
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        with torch.no_grad():
            if (~is_self_loop).any():
                real_sparsity_hard = 1.0 - y_hard01[~is_self_loop].mean().item()
                kept_ratio_hard = y_hard01[~is_self_loop].mean().item()
            else:
                real_sparsity_hard = 0.0
                kept_ratio_hard = 1.0

            kept_ratio_soft = m_rest_soft.mean().item() if m_rest_soft.numel() > 0 else 1.0

        count = {
            "mask_mean": structure_mask.mean().item(),
            "mask_min": structure_mask.min().item(),
            "mask_max": structure_mask.max().item(),

            "kept_ratio_hard": kept_ratio_hard,
            "sparsity_hard": real_sparsity_hard,
            "kept_ratio_soft": kept_ratio_soft,

            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            "edge_weight_sum": structure_mask.detach().sum().item(),

            "L_suf_con": L_suf.item(),
            "L_cf_con": L_cf.item()
        }

        return structure_mask, L_suf, L_cf, L_sp_structure, count

    def get_hard_masks(
            self, g, x, sens, sens_idx, train_mask, labels,
            mask_threshold=0.3, min_feature_keep_ratio=0.05,
            max_feature_keep_ratio=None, max_structure_keep_ratio=None):
        """
        Step 1: 模型推理 -> 生成 Soft Mask -> 截断为 Hard Mask
        """
        self.eval()
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)


        with torch.no_grad():
            e = self.encoder(g, x)
            z_y, z_s, _, _, _ = self.fairINN(e)

            e_cf, _, _, _ = self.generate_nearest_neighbor_counterfactual_with_R(z_s, z_y, sens, labels, train_mask)

            mask_score_feat = self.featureMask_Generator(x, e, e_cf)
            mask_score_feat[:, sens_idx] = 0.0

            src, dst = g.edges()
            is_self_loop = (src == dst)
            edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
            mask_logits_struct = self.structureMask_Generator(edge_feat).squeeze(-1)
            structure_probs = torch.sigmoid(mask_logits_struct)

            structure_probs = structure_probs.masked_fill(is_self_loop, 1.0)

            feat_mask_hard = (mask_score_feat > mask_threshold).float()
            feature_candidates = torch.ones_like(feat_mask_hard, dtype=torch.bool)
            feature_candidates[:, sens_idx] = False
            if min_feature_keep_ratio > 0:
                candidate_scores = mask_score_feat[feature_candidates]
                candidate_mask = feat_mask_hard[feature_candidates]
                min_keep = int(candidate_scores.numel() * min_feature_keep_ratio)
                if min_keep > 0 and int(candidate_mask.sum().item()) < min_keep:
                    topk_idx = torch.topk(candidate_scores, k=min_keep, largest=True).indices
                    fallback_mask = torch.zeros_like(candidate_scores)
                    fallback_mask[topk_idx] = 1.0
                    feat_mask_hard = feat_mask_hard.clone()
                    feat_mask_hard[feature_candidates] = fallback_mask
                    feat_mask_hard[:, sens_idx] = 0.0
            if max_feature_keep_ratio is None:
                max_feature_keep_ratio = self.get_mask_target_ranges()[0][1]
            if max_feature_keep_ratio is not None and max_feature_keep_ratio > 0:
                candidate_scores = mask_score_feat[feature_candidates]
                candidate_mask = feat_mask_hard[feature_candidates]
                max_keep = max(1, int(candidate_scores.numel() * max_feature_keep_ratio))
                current_keep = int(candidate_mask.sum().item())
                if current_keep > max_keep:
                    topk_idx = torch.topk(candidate_scores, k=max_keep, largest=True).indices
                    capped_mask = torch.zeros_like(candidate_scores)
                    capped_mask[topk_idx] = 1.0
                    feat_mask_hard = feat_mask_hard.clone()
                    feat_mask_hard[feature_candidates] = capped_mask
                    feat_mask_hard[:, sens_idx] = 0.0

            struct_mask_hard = (structure_probs > mask_threshold).float()
            if max_structure_keep_ratio is None:
                max_structure_keep_ratio = self.get_mask_target_ranges()[1][1]
            if max_structure_keep_ratio is not None and max_structure_keep_ratio > 0:
                non_loop = ~is_self_loop
                non_loop_scores = structure_probs[non_loop]
                non_loop_mask = struct_mask_hard[non_loop]
                if non_loop_scores.numel() > 0:
                    max_keep = max(1, int(non_loop_scores.numel() * max_structure_keep_ratio))
                    current_keep = int(non_loop_mask.sum().item())
                    if current_keep > max_keep:
                        topk_idx = torch.topk(non_loop_scores, k=max_keep, largest=True).indices
                        capped_mask = torch.zeros_like(non_loop_scores)
                        capped_mask[topk_idx] = 1.0
                        struct_mask_hard = struct_mask_hard.clone()
                        struct_mask_hard[non_loop] = capped_mask
            struct_mask_hard = struct_mask_hard.masked_fill(is_self_loop, 1.0)

        return feat_mask_hard, struct_mask_hard, is_self_loop

    def get_mask_prediction(self, g, x, sens, train_mask, loss_type=None, mask_threshold=0.3, tau=0.1):
        self.eval()
        e = self.encoder(g, x)
        src, dst = g.edges()
        is_self_loop = (src == dst)

        z_y, z_s, pred_s, pred_y, _ = self.fairINN(e)
        e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)

        mask = self.featureMask_Generator(x, e, e_cf)
        x_base = self.get_x_base(x, train_mask)
        x_tilde = mask * x + (1.0 - mask) * x_base

        src, dst = g.edges()
        edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()

        if loss_type == 'Soft' or loss_type == 'Entropy':
            structure_mask = torch.sigmoid(mask_logits)
        elif loss_type == 'STE':

            mask_probs = torch.sigmoid(mask_logits)
            mask_probs = mask_probs.masked_fill(is_self_loop, 1.0)
            mask_hard = (mask_probs > mask_threshold).float()
            mask_hard = mask_hard.masked_fill(is_self_loop, 1.0)
            structure_mask = mask_hard - mask_probs.detach() + mask_probs
            structure_mask = torch.clamp(structure_mask, min=1e-6)
        elif loss_type == 'Temp':
            structure_mask = torch.sigmoid(mask_logits / tau)
        else:
            raise NotImplementedError("Unsupported loss type.")

        e_tilde = self.encoder(g, x_tilde, edge_weight=structure_mask)
        _, _, _, logit_tilde, _ = self.fairINN(e_tilde)

        return x_tilde, logit_tilde, structure_mask

    def get_complement_masks(self, feat_mask, struct_mask, is_self_loop):
        """
        Step 2: 输入解释掩码 -> 生成反向(补图)掩码
        """
        comp_feat_mask = 1.0 - feat_mask

        comp_struct_mask = 1.0 - struct_mask

        comp_struct_mask = comp_struct_mask.masked_fill(is_self_loop, 1.0)

        return comp_feat_mask, comp_struct_mask

    def get_explained_graph(self, g, x, feat_mask, struct_mask, train_mask):
        """
        Step 3: 输入掩码 + 原数据 -> 构建物理子图对象 (DGLGraph) 和 掩码后特征
        """
        x_base = self.get_x_base(x, train_mask)
        x_new = feat_mask * x + (1.0 - feat_mask) * x_base

        kept_indices = torch.nonzero(struct_mask > 0).squeeze(-1)

        if kept_indices.numel() > 0:
            g_new = dgl.edge_subgraph(g, kept_indices, relabel_nodes=False)
        else:
            g_new = dgl.graph(([], []), num_nodes=g.number_of_nodes(), device=g.device)

        g_new = dgl.remove_self_loop(g_new)
        g_new = dgl.add_self_loop(g_new)

        e_new = self.encoder(g_new, x_new)
        z_y, z_s, _, _, _ = self.fairINN(e_new)


        return g_new, x_new, z_y, z_s, e_new

    def get_hard_mask_prediction(
            self,
            g,
            x,
            sens,
            train_mask,
            mask_threshold=0.3,
            tau=0.1,
            return_subgraph=True,
            eps=1e-6,
    ):
        """
        推理/解释阶段：生成硬掩码，构建新图，进行预测。

        与“新训练代码”保持一致的关键点：
        1) 训练中对结构掩码使用 hard=eps/1 来满足 APPNPConv(norm="both") 对 edge_weight>0 的要求；
        2) 推理阶段你现在是“物理删边 + 重建自环”，这本身不会触发 edge_weight>0 的报错（因为不传 edge_weight）。
           但为了与训练口径一致，我们：
             - 仍然锁定自环为必保留（prob=1）
             - kept_indices 基于 hard(0/1) 的阈值逻辑
             - 额外返回结构的 hard mask（可选）以及 soft probs（可选）
        """

        self.eval()

        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        with torch.no_grad():
            e = self.encoder(g, x)
            z_y, z_s, pred_s, pred_y, _ = self.fairINN(e)

            e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)

            mask_score_feat = self.featureMask_Generator(x, e, e_cf)

            src, dst = g.edges()
            is_self_loop = (src == dst)

            edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
            mask_logits_struct = self.structureMask_Generator(edge_feat).squeeze(-1)
            structure_probs = torch.sigmoid(mask_logits_struct)

            structure_probs = structure_probs.masked_fill(is_self_loop, 1.0)

            feature_mask_hard = (mask_score_feat > mask_threshold).float()
            x_base = self.get_x_base(x, train_mask)
            x_final = feature_mask_hard * x + (1.0 - feature_mask_hard) * x_base

            struct_mask_hard01 = (structure_probs > mask_threshold).float()
            struct_mask_hard01 = struct_mask_hard01.masked_fill(is_self_loop, 1.0)


            kept_indices = torch.nonzero(struct_mask_hard01 > 0).squeeze(-1)

            if kept_indices.numel() > 0:
                new_g = dgl.edge_subgraph(g, kept_indices, relabel_nodes=False)
            else:
                new_g = dgl.graph(([], []), num_nodes=g.number_of_nodes(), device=g.device)

            new_g = dgl.remove_self_loop(new_g)
            new_g = dgl.add_self_loop(new_g)

            e_final = self.encoder(new_g, x_final)
            z_y, z_s, _, logit_final, _ = self.fairINN(e_final)

        if return_subgraph:
            return x_final, logit_final, new_g, struct_mask_hard01, structure_probs, e_final, z_y, z_s
        else:
            return x_final, logit_final, struct_mask_hard01, structure_probs

    @torch.no_grad()
    def compute_flip_and_consistency(self, z_y, z_s, z_s_cf, sens, labels, thr_logit=0.0):
        """
        model: 你的 FairINNModel (含 classifier_y / classifier_s)
        z_y, z_s: 原表示 [N, y_dim], [N, s_dim]
        z_y_cf, z_s_cf: 反事实表示 [N, y_dim], [N, s_dim]
        thr_logit: logit 阈值，二分类默认 0.0

        返回:
          flip_rate_s: scalar
          consistency_y: scalar
          以及一些可选的诊断量
        """
        w = self.fairINN.classifier_s.weight.data
        b = self.fairINN.classifier_s.bias.data
        print("||w||:", w.norm().item(), "bias:", b.item())


        z_s_cf_eval = z_s_cf[:, self.fairINN.y_dim:]
        z_y_cf_eval = z_s_cf[:, :self.fairINN.y_dim]
        w = self.fairINN.classifier_s.weight.squeeze(0)
        b = self.fairINN.classifier_s.bias.squeeze(0)

        proj = z_s_cf_eval @ w + b
        print("proj (==logit_s_cf) min/max/std:",
              proj.min().item(), proj.max().item(), proj.std().item())

        logit_s = self.fairINN.classifier_s(z_s)
        logit_s_cf = self.fairINN.classifier_s(z_s_cf_eval)
        print("logit_s      min/max/std:",
              logit_s.min().item(), logit_s.max().item(), logit_s.std().item())
        print("logit_s_cf   min/max/std:",
              logit_s_cf.min().item(), logit_s_cf.max().item(), logit_s_cf.std().item())

        prob_s = torch.sigmoid(logit_s)
        prob_s_cf = torch.sigmoid(logit_s_cf)

        print("prob_s       min/max/std:",
              prob_s.min().item(), prob_s.max().item(), prob_s.std().item())
        print("prob_s_cf    min/max/std:",
              prob_s_cf.min().item(), prob_s_cf.max().item(), prob_s_cf.std().item())

        pred_s = (logit_s.squeeze() > 0).type_as(sens)
        pred_s_cf = (logit_s_cf.squeeze() > 0).type_as(sens)
        print("pred_s   std:", pred_s.float().std().item(), "min/max:", pred_s.min().item(), pred_s.max().item())
        print("pred_s_cf std:", pred_s_cf.float().std().item(), "min/max:", pred_s_cf.min().item(),
              pred_s_cf.max().item())

        flip_rate_s = (pred_s != pred_s_cf).float().mean()

        logit_y = self.fairINN.classifier_y(z_y).squeeze(-1)
        logit_y_cf = self.fairINN.classifier_y(z_y_cf_eval).squeeze(-1)
        pred_y = (logit_y > thr_logit)
        pred_y_cf = (logit_y_cf > thr_logit)

        consistency_y = (pred_y == pred_y_cf).float().mean()

        prob_s = torch.sigmoid(logit_s)
        prob_s_cf = torch.sigmoid(logit_s_cf)
        prob_y = torch.sigmoid(logit_y)
        prob_y_cf = torch.sigmoid(logit_y_cf)

        diagnostics = {
            "delta_prob_s_mean": (prob_s_cf - prob_s).abs().mean(),
            "delta_prob_y_mean": (prob_y_cf - prob_y).abs().mean(),
            "delta_logit_s_mean": (logit_s_cf - logit_s).abs().mean(),
            "delta_logit_y_mean": (logit_y_cf - logit_y).abs().mean(),
        }

        return flip_rate_s.item(), consistency_y.item(), diagnostics



