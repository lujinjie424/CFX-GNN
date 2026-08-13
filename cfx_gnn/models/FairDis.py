import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.half_dim = input_dim // 2

        # [改进] 去掉 Tanh，允许更大的变形能力
        # 或者使用 Clamp 保证数值稳定，但不限制在 (-1, 1) 这么死
        self.s_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half_dim),
            # nn.Tanh()  <-- 建议移除，或者换成下面的 scaling
        )
        # 为了防止训练初期梯度爆炸，可以加一个可学习的缩放因子，初始化为小值
        self.scale = nn.Parameter(torch.tensor(-2.0))

        self.t_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.half_dim)
        )

    def forward(self, x, reverse=False):
        x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]

        if not reverse:
            # 正向: E -> Z
            s_raw = self.s_net(x1)
            # 使用 tanh * scale 既保证稳定又能通过学习扩大范围
            # 或者直接用 s = s_raw，但要注意梯度裁剪
            s = torch.tanh(s_raw) * F.softplus(self.scale)

            t = self.t_net(x1)
            y2 = x2 * torch.exp(s) + t
            return torch.cat([x1, y2], dim=1), torch.sum(s, dim=1)
        else:
            # 逆向: Z -> E
            s_raw = self.s_net(x1)
            s = torch.tanh(s_raw) * F.softplus(self.scale)

            t = self.t_net(x1)
            x2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, x2], dim=1)


class INN_Node(nn.Module):
    def __init__(self, input_dim, num_layers=4):
        super().__init__()
        # 必须确保 input_dim 是偶数，否则切分会出问题
        assert input_dim % 2 == 0, "Input dimension must be even for simple coupling."

        self.layers = nn.ModuleList([
            CouplingLayer(input_dim) for _ in range(num_layers)
        ])

    def forward(self, x, reverse=False):
        log_det_jacobian = 0

        if not reverse:
            # --- 正向: E -> Z ---
            for i, layer in enumerate(self.layers):
                # 1. 通过耦合层
                x, ld = layer(x, reverse=False)
                log_det_jacobian += ld

                # 2. 【关键修复】交换通道 (Channel Swap)
                # 除了最后一层外，每一层输出后都把向量翻转一下
                # 这样上一层没变的 x1，在下一层就会变成 x2 从而被处理
                if i < len(self.layers) - 1:
                    x = x.flip(dims=[1])
            return x, log_det_jacobian

        else:
            # --- 逆向: Z -> E ---
            # 注意：逆向时顺序要完全相反
            for i, layer in enumerate(reversed(self.layers)):
                # 1. 【关键修复】先交换回来（因为正向是后交换的）
                # 逻辑：正向是 Layer -> Flip ... Layer -> Flip -> Layer
                # 逆向是 Layer -> UnFlip -> Layer ... UnFlip -> Layer
                # 注意索引 i 是从 0 开始的倒序，需要判断是否是第一层(即正向的最后一层)
                # 正向最后一层没有 Flip，所以逆向第一层也不用 UnFlip
                if i > 0:
                    x = x.flip(dims=[1])

                # 2. 通过耦合层逆向
                x = layer(x, reverse=True)
            return x

class FairLatentLoss(nn.Module):
    def __init__(self, z_dim, epsilon=1e-6):
        super().__init__()
        self.z_dim = z_dim
        self.epsilon = epsilon
        self.register_buffer("I", torch.eye(z_dim))

    def compute_covariance(self, z):
        # 计算协方差矩阵 C(Z) [cite: 161]
        # z: [Batch, Dim]
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
            # covariance-based version for different dimensions
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
        # 移除对角线元素: C - C * I
        off_diag = C - C * self.I
        # 损失为非对角线元素的平方和，归一化
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

        # 计算成对距离矩阵
        # expand dims for broadcasting: [N, 1, D] - [1, N, D]
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        dist_sq = diff.pow(2).sum(dim=2)  # L2 距离的平方

        # 定义论文中的单调递减函数 D(x, y) [cite: 178]
        # D = log((||x-y||^2 + 1) / (||x-y||^2 + epsilon))
        D_val = torch.log((dist_sq + 1) / (dist_sq + self.epsilon))

        # 构建 Mask [cite: 176]
        # labels: [N], sensitives: [N]
        # Kronecker delta: (labels[i] == labels[j])
        label_match = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        sens_match = (sensitives.unsqueeze(1) == sensitives.unsqueeze(0)).float()

        # M_max: Y相同，S不同 (我们希望这些样本在 Z^Y 空间接近，即 maximize D)
        # 注意：论文公式(7)中第一项是负号，表示最大化第一项的和
        M_max = label_match * (1 - sens_match)

        # M_min: S相同，Y不同 (我们希望这些样本远离，即 minimize D)
        M_min = sens_match * (1 - label_match)

        # 忽略自身对自身的比较 (对角线)
        mask_identity = torch.eye(batch_size).to(z.device)
        M_max = M_max * (1 - mask_identity)
        M_min = M_min * (1 - mask_identity)

        # 计算 Loss [cite: 174]
        # 避免除以0
        sum_m_max = M_max.sum() + 1e-8
        sum_m_min = M_min.sum() + 1e-8

        term1 = - (1.0 / sum_m_max) * (M_max * D_val).sum()
        term2 = (1.0 / sum_m_min) * (M_min * D_val).sum()

        return term1 + term2

    def compute_kernel(self, x, y):
        # 使用高斯核计算核矩阵
        dim = x.size(1)
        # 计算成对距离
        dist = torch.cdist(x, x, p=2).pow(2)
        # 动态计算 sigma (基于中位数技巧)
        sigma = torch.median(dist).detach()
        if sigma == 0: sigma = 1.0  # 防止除零

        return torch.exp(-dist / (2 * sigma))

    def loss_hsic(self, z_y, z_s):
        """
        最大化 Z_y 和 Z_s 的独立性 (最小化 HSIC 值)
        """
        n = z_y.size(0)

        # 2. 计算核矩阵
        K = self.compute_kernel(z_y, z_y)
        L = self.compute_kernel(z_s, z_s)

        # 3. 动态生成单位矩阵 (不要用 self.I)
        # H = I - 1/n * 11^T
        # H shape: [n, n]
        H = torch.eye(n, device=z_y.device) - (1.0 / n) * torch.ones((n, n), device=z_y.device)

        # 4. 计算 HSIC
        # hsic = tr(KHLH)
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

        # 确保总维度匹配
        assert y_dim + s_dim == input_dim

        # 用于分类的简单线性层 [cite: 204]
        self.classifier_y = nn.Sequential(
            nn.Linear(y_dim, 64),
            nn.LeakyReLU(),  # LeakyReLU 防止梯度死区
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.classifier_s = nn.Sequential(
            nn.Linear(s_dim, 64),
            nn.LeakyReLU(),  # LeakyReLU 防止梯度死区
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        # self.classifier_y = nn.Linear(y_dim, 1)
        # self.classifier_s = nn.Linear(s_dim, 1)

        self.loss_fn = FairLatentLoss(y_dim)  # 主要对 Z^Y 施加公平约束

    def forward(self, e):
        # e: 预训练节点嵌入 [Batch, Dim]
        z, log_det = self.inn(e)

        # 切分 Z -> Z^Y, Z^S [cite: 182]
        z_y = z[:, :self.y_dim]
        z_s = z[:, self.y_dim:]

        # 预测
        pred_y = self.classifier_y(z_y)
        pred_s = self.classifier_s(z_s)

        return z_y, z_s, pred_s, pred_y, log_det

    def unit(self, v, eps=1e-12):
        return v / (v.norm(p=2) + eps)

    @torch.no_grad()
    def get_dirs(self):
        # 线性分类器：weight shape [1, dim]
        hy = self.unit(self.classifier_y.weight[0])  # [y_dim]
        hs = self.unit(self.classifier_s.weight[0])  # [s_dim]
        return hy, hs



class StructureMask(nn.Module):
    """邻居掩码生成器：接受每条边的拼接输入，预测 mask 的 Logits"""

    def __init__(self, input_dim, hidden_dim, init_bias=3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1)  # 输出维度为1
        )
        self.init_bias = init_bias
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # 这里的逻辑是：如果是最后一层，我们希望 bias 稍微大一点，让初始 mask 偏向 1
                if m.out_features == 1:
                    nn.init.constant_(m.bias, self.init_bias)
                else:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, edge_feat):
        """
        returns scores: [E, 1] (Logits, range: -inf ~ +inf)
        """
        scores = self.net(edge_feat)

        # 移除 forward 中的噪声注入，移到 Loss 计算处控制会更灵活
        # 移除 Sigmoid，返回 Logits
        return scores


class FeatureMask(nn.Module):
    def __init__(self, feature_dim, emb_dim):
        super().__init__()
        self.hidden_dim = 64
        self.input_dim = feature_dim + 2 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            # [关键] 最后一层，输出 logits
            nn.Linear(self.hidden_dim, feature_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # 1. 通用初始化 (针对所有层)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)  # 中间层 bias 设为 0 即可

        # 2. [核心修正] 特别初始化最后一层 (Output Layer)
        # 获取最后一层
        last_layer = self.net[-1]

        # 将最后一层的权重设得很小 (接近 0)，减少输入的波动影响
        nn.init.normal_(last_layer.weight, mean=0, std=0.001)

        # [关键] 将最后一层的偏置设为大的正数 (例如 3.0 或 5.0)
        # Sigmoid(3.0) ≈ 0.95
        # Sigmoid(5.0) ≈ 0.993
        nn.init.constant_(last_layer.bias, 3.0)

    def forward(self, x, e, e_cf):

        # 1. 拼接条件信息（推荐显式用 delta）
        delta_e = e_cf - e  # [N, d]
        emb = torch.cat([x, e, delta_e], dim=1)  # [N, F + 2d]

        # 2. 生成特征级 mask
        mask_logits = self.net(emb)  # [N, F]
        mask = torch.sigmoid(mask_logits)  # [N, F]

        return mask


def supcon_loss(z, labels, temperature=0.07):
    """
    真正的 Supervised Contrastive Loss (InfoNCE 变体)
    """
    # 1. 特征归一化 (使用余弦相似度)
    z = F.normalize(z, dim=1)

    # 2. 计算相似度矩阵 [N, N]
    sim_matrix = torch.mm(z, z.T)

    # 3. 构造 Mask
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)

    # 4. 屏蔽对角线 (自己与自己不算正样本)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(z.shape[0]).view(-1, 1).to(z.device),
        0
    )
    mask = mask * logits_mask

    # 5. 计算 Logits
    # 除以温度系数，增加对难例的关注
    exp_logits = torch.exp(sim_matrix / temperature) * logits_mask

    # 分母：每个样本与其他所有样本(除自己外)的相似度之和
    log_prob = sim_matrix / temperature - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    # 6. 计算 Mean Log-Likelihood
    # 只计算正样本对 (同类) 的 Log 概率
    # mask.sum(1) 是每个样本对应的正样本数量
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    # 损失取负
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

        # 2. 基础损失
        # L_g: Gaussian Loss (Negative Log Likelihood) [cite: 203]
        # 假设先验 p(z) 是标准正态分布，log p(z) 正比于 -||z||^2
        z_full = torch.cat([z_y, z_s], dim=1)
        l_g = torch.mean(0.5 * torch.sum(z_full ** 2, dim=1) - log_det)

        # L_cls: 分类损失 [cite: 204]
        # l_cls_encoder = F.binary_cross_entropy_with_logits(logit[train_mask], y[train_mask].unsqueeze(1).float())
        l_cls_y = F.binary_cross_entropy_with_logits(pred_y[train_mask], y[train_mask].unsqueeze(1).float())
        pos_weight_tensor = torch.tensor([pos_weight]).to(s.device)
        # criterion_sens = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        l_cls_s = F.binary_cross_entropy_with_logits(pred_s[train_mask], s[train_mask].unsqueeze(1).float())
        l_con_y = supcon_loss(z_y[train_mask], y[train_mask])
        l_con_s = supcon_loss(z_s[train_mask], s[train_mask])

        # 3. 公平性损失 L_fair(Z^Y) [cite: 183]
        # 注意：论文主要对 Z^Y 施加这些约束以去除敏感信息
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

        # 1. 准备搜索库 (Search Database) - 只在训练集中找
        # 这一步和之前一样，构建 "男池子" 和 "女池子"
        # detach() 很重要，我们不需要对替身库求导
        z_y_train = z_y[train_mask].detach()
        z_s_train = z_s[train_mask].detach()
        s_train = sens[train_mask].detach()

        # 分离 Group 0 和 Group 1 的库
        mask_0 = (s_train == 0)
        mask_1 = (s_train == 1)

        # Pool 0: S=0 的所有样本 (供 S=1 的人搜索)
        z_y_pool_0 = z_y_train[mask_0]
        z_s_pool_0 = z_s_train[mask_0]

        # Pool 1: S=1 的所有样本 (供 S=0 的人搜索)
        z_y_pool_1 = z_y_train[mask_1]
        z_s_pool_1 = z_s_train[mask_1]

        # 2. 准备查询对象 (Query)
        # 我们需要知道当前输入 z_s, z_y 中，哪些是 Group 0，哪些是 Group 1
        # 假设输入的是全图或一个 Batch

        # 获取当前样本的真实敏感属性 (用于决定去哪个池子找)
        # 注意：如果传入的 z_s, z_y 是全图，这里 sensitive_labels 也应该是全图
        current_s = sens.long()

        # 3. [核心优化] 矩阵化搜索
        # 我们不能写循环，而是要把所有需要去 Pool 1 找的人打包，一次性算完

        # 创建一个容器存放结果，初始化为原始 z_s (防止有孤立点没找到替身)
        z_s_new = z_s.clone()

        # === 批次 A: 当前是 S=0 的样本 -> 去 Pool 1 找 ===
        query_mask_0 = (current_s == 0)
        if query_mask_0.any() and len(z_y_pool_1) > 0:
            # 取出所有 S=0 的 z_y
            queries_0 = z_y[query_mask_0]  # [K, dim]

            # 计算距离矩阵 [K, M] (K个查询者, M个候选人)
            # torch.cdist 是高度优化的 CUDA 实现
            dists = torch.cdist(queries_0, z_y_pool_1, p=2)

            # 找到每行最小值的索引 [K]
            min_indices = torch.argmin(dists, dim=1)

            # 使用索引从 z_s_pool_1 中取出替身的 z_s
            # 这一步叫 "Gather"
            z_s_matched = z_s_pool_1[min_indices]

            # 填回结果容器
            z_s_new[query_mask_0] = z_s_matched

        # === 批次 B: 当前是 S=1 的样本 -> 去 Pool 0 找 ===
        query_mask_1 = (current_s == 1)
        if query_mask_1.any() and len(z_y_pool_0) > 0:
            queries_1 = z_y[query_mask_1]

            dists = torch.cdist(queries_1, z_y_pool_0, p=2)
            min_indices = torch.argmin(dists, dim=1)
            z_s_matched = z_s_pool_0[min_indices]

            z_s_new[query_mask_1] = z_s_matched

        # 4. 解码 (一次性解码整个 Batch)
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
        # logits = self.encoder.prediction(h)

        z_y, z_s, logits_s, logits_y, log_det = self.fairINN(h)
        # e_cf, z_cf = self.counterfactual_embedding(z_s, z_y)
        # logits_cf = self.encoder.prediction(e_cf)

        return h, z_y, z_s, logits_s, logits_y

    def kl_softmax(self, p_logits, q_logits, T=1.0):
        """ KL( softmax(p/T) || softmax(q/T) ). """
        p = F.log_softmax(p_logits / T, dim=-1)  # log p
        q = F.softmax(q_logits / T, dim=-1)  # q
        return F.kl_div(p, q, reduction="batchmean") * (T * T)

    def logit_l2(self,p, q):
        return ((p - q) ** 2).mean()

    def compute_cosine_loss(self, emb1, emb2):
        # 目标是最大化相似度，即最小化 (1 - cos)
        return 1.0 - F.cosine_similarity(emb1, emb2, dim=-1).mean()

    @torch.no_grad()
    def get_x_base(self, x, train_mask):
        # x: [N, F], train_mask: [N] bool
        xb = x[train_mask].mean(dim=0, keepdim=True)  # [1, F]
        return xb

    def calculate_geometric_weights(self, e, e_cf):
        with torch.no_grad():
            dist = torch.norm(e - e_cf, p=2, dim=1)

            # 方案 A: 均值归一化 (推荐，比较温和)
            # 让平均距离的权重为 1.0，距离远的 > 1.0
            # 添加 epsilon 防止除以 0
            mean_dist = dist.mean() + 1e-8
            weights = 1.0 + 0.5 * (dist / mean_dist)
            # 结果通常在 [1.0, 2.0] 附近分布，但保留了真实的分布形态

            # 方案 B: Sigmoid 缩放 (如果你想严格限制在 1-2 之间)
            # weights = 1.0 + torch.sigmoid(dist - dist.mean())

        return weights

    def mask_range_loss(self, keep_ratio, target_range):
        min_keep, max_keep = target_range
        return F.relu(torch.as_tensor(min_keep, device=keep_ratio.device) - keep_ratio).pow(2) + \
            F.relu(keep_ratio - torch.as_tensor(max_keep, device=keep_ratio.device)).pow(2)

    def get_mask_target_ranges(self):
        return getattr(self, "mask_feature_target_range", (0.10, 0.30)), \
            getattr(self, "mask_structure_target_range", (0.10, 0.30))

    def compute_feature_mask_loss(self, g, x, sens, train_mask, labels):
        # 1) 原始 embedding
        e = self.encoder(g, x)  # [N, d]
        # [REMOVED] logit = self.encoder.prediction(e)

        # 2) 生成原始反事实 embedding (Target Generation)
        z_y, z_s, _, _, _ = self.fairINN(e)
        # 建议：这里的 generate 内部 alpha 设为 1.0 或 None，确保产生强反事实
        e_cf, _, _, R = self.generate_nearest_neighbor_counterfactual_with_R(
            z_s, z_y, sens, labels, train_mask, g=g
        )
        weights = R.detach()

        # 3) Mask 生成
        mask = self.featureMask_Generator(x, e, e_cf)
        mask = mask

        # 4) 构造 Masked 特征
        x_base = self.get_x_base(x, train_mask)
        x_tilde = mask * x + (1 - mask) * x_base

        # 5) Masked Embedding
        x_masked = x.clone()
        x_masked = x_tilde
        e_tilde = self.encoder(g, x_masked)
        # [REMOVED] logit_tilde ...

        # 6) [Consistency Check] 在 Masked Embedding 上再生成一次 CF
        e_tilde_tr = e_tilde
        z_y_t, z_s_t, _, _, _ = self.fairINN(e_tilde_tr)
        e_tilde_cf, _, _, R_tilde = self.generate_nearest_neighbor_counterfactual_with_R(
            z_s_t, z_y_t, sens, labels, train_mask, g=g
        )
        # [REMOVED] logit_tilde_cf ...

        # --- [MODIFIED] 加权 Loss 计算 (基于 Embedding) ---

        # 获取训练集的原始 embedding (Detach，作为 Target)

        e_tilde_train = e_tilde # 实际上 e_tilde 已经在 5) 步算出来只针对 mask 部分变化了，但为了维度对齐

        z_train = z_y.detach()
        z_y_t_train = z_y_t
        e_train = e.detach()

        # (A) Sufficiency Loss (L_suf): 语义保留
        # 目标：Mask 后的 embedding 应该尽可能接近原始 embedding (保留任务信息 Z_y)
        # 使用 Cosine Similarity 往往比 MSE 更好，因为它关注方向（语义）而非模长（掩码可能导致模长变小）
        # Loss = 1 - CosSim
        loss_suf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_train, e_tilde_train, dim=1)
        # 如果你想用 MSE 也可以: loss_suf_elem = torch.sum((e_train - e_tilde_train)**2, dim=1)
        L_suf = loss_suf_elem.mean()

        # (B) Counterfactual Consistency (L_cf): 敏感不变性
        # 目标：Mask 后的图，经过反事实变换，Embedding 应该保持不变
        # 这意味着 e_tilde 里面已经不包含敏感信息了
        # 同样使用 Cosine Distance 或 MSE
        # 注意 e_tilde_cf 计算时用的是 e_tilde 的 z_y，所以它们理应很像
        loss_cf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_tilde_train, e_tilde_cf, dim=1)
        # 或者 MSE: loss_cf_elem = torch.sum((e_tilde_train - e_tilde_cf[train_mask])**2, dim=1)
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
        # [1] 显式重构自环
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        # 1. 基础 Embedding
        e = self.encoder(g, x)

        # 2. 反事实相关
        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e)
            e_cf, _, _, R = self.generate_nearest_neighbor_counterfactual_with_R(
                z_s, z_y, sens, labels, train_mask, g=g
            )

        weights = R.detach()


        # 3. 计算 Logits
        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()

        # =======================================================
        # [核心技巧] Hard Gumbel-Sigmoid
        # =======================================================

        # 1. 生成 Gumbel 噪声
        # Gumbel(0,1) = -log(-log(U(0,1)))
        if self.training:
            # 更稳：避免 U 取到 0 或 1
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        # 锁定自环概率为 1（soft 阶段就锁）
        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        # 2) hard mask（0/1）
        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        # 3) 关键修复：将 hard 的 0 变成 eps，保证 edge_weight 严格 > 0
        #    hard: 0 -> eps, 1 -> 1
        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        # 4) STE：forward 用 y_hard_pos（eps/1），backward 用 y_soft（可导）
        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        # 双重保险：确保严格为正（APPNPConv norm="both" 的要求）
        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        # 5) 基于 Mask 的前向传播（APPNP 用 edge_weight 参与传播）
        e_mask = self.encoder(g, x, edge_weight=structure_mask)
        z_y_t, z_s_t, _, _, _ = self.fairINN(e_mask)

        # --- Loss 计算 ---
        e_train = e.detach()
        e_mask_train = e_mask
        #
        loss_suf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_train, e_mask_train, dim=1)



        z_y_train = z_y[train_mask].detach()
        z_y_t_train = z_y_t[train_mask]
        # L_y = F.binary_cross_entropy_with_logits(pred_y_t[train_mask], labels[train_mask].unsqueeze(1).float())
        # L_y = F.mse_loss(pred_y_t[train_mask], pred_y[train_mask])
        # loss_suf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_train, e_mask_train, dim=1)

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

        # 正则化：稀疏性（用 soft 的均值更平滑）
        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            _, structure_target_range = self.get_mask_target_ranges()
            L_sp_structure = self.mask_range_loss(m_rest_soft.mean(), structure_target_range)
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        # --- 统计信息 ---
        with torch.no_grad():
            # hard(0/1) 的真实稀疏度（不含自环）
            if (~is_self_loop).any():
                real_sparsity_hard = 1.0 - y_hard01[~is_self_loop].mean().item()
                kept_ratio_hard = y_hard01[~is_self_loop].mean().item()
            else:
                real_sparsity_hard = 0.0
                kept_ratio_hard = 1.0

            # soft 的“平均保留强度”（不含自环）
            kept_ratio_soft = m_rest_soft.mean().item() if m_rest_soft.numel() > 0 else 1.0

        count = {
            "mask_mean": structure_mask.mean().item(),
            "mask_min": structure_mask.min().item(),
            "mask_max": structure_mask.max().item(),

            # hard / soft 两套口径
            "kept_ratio_hard": kept_ratio_hard,  # hard(0/1) 保留边比例
            "sparsity_hard": real_sparsity_hard,  # hard(0/1) 稀疏度
            "kept_ratio_soft": kept_ratio_soft,  # soft 平均强度

            # 你原本的权重监控
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            # 边数统计（对 m 是 eps/1 的 forward 权重，不等价于 hard(0/1) 的 kept_edge_count）
            "edge_weight_sum": structure_mask.detach().sum().item(),
        }

        return L_suf, L_cf, L_sp_structure, count


    def compute_mask_loss_Gumbel(self, g, x, sens, train_mask, labels, mask_threshold=0.3, tau=1.0, eps=1e-6):
        # [1] 显式重构自环
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        # 1. 基础 Embedding
        e = self.encoder(g, x)

        # 2. 反事实相关
        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e)
            e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)

        weights_all = self.calculate_geometric_weights(e, e_cf)
        weights = weights_all[train_mask].detach()

        feature_mask = self.featureMask_Generator(x, e, e_cf)
        # feature_mask = feature_mask[train_mask]

        # 4) 构造 Masked 特征
        x_base = self.get_x_base(x, train_mask)
        x_tilde = feature_mask * x + (1 - feature_mask) * x_base

        # 5) Masked Embedding
        # x_masked = x.clone()
        x_masked = x_tilde

        L_sp_feature = feature_mask.mean()
        L_bin = (feature_mask * (1 - feature_mask)).mean()

        e_tilde = self.encoder(g, x_masked)
        with torch.no_grad():
            z_y_tilde, z_s_tilde, _, _, _ = self.fairINN(e_tilde)
            e_cf_tilde, _, _ = self.generate_nearest_neighbor_counterfactual(z_s_tilde, z_y_tilde, sens, train_mask)

        # 3. 计算 Logits
        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e_tilde[src], e_tilde[dst], e_cf_tilde[src], e_cf_tilde[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()

        # =======================================================
        # [核心技巧] Hard Gumbel-Sigmoid
        # =======================================================

        # 1. 生成 Gumbel 噪声
        # Gumbel(0,1) = -log(-log(U(0,1)))
        if self.training:
            # 更稳：避免 U 取到 0 或 1
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        # 锁定自环概率为 1（soft 阶段就锁）
        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        # 2) hard mask（0/1）
        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        # 3) 关键修复：将 hard 的 0 变成 eps，保证 edge_weight 严格 > 0
        #    hard: 0 -> eps, 1 -> 1
        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        # 4) STE：forward 用 y_hard_pos（eps/1），backward 用 y_soft（可导）
        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        # 双重保险：确保严格为正（APPNPConv norm="both" 的要求）
        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        # 5) 基于 Mask 的前向传播（APPNP 用 edge_weight 参与传播）
        e_mask = self.encoder(g, x_masked, edge_weight=structure_mask)
        z_y_t, z_s_t, _, _, _ = self.fairINN(e_mask)

        # --- Loss 计算 ---
        e_train = e[train_mask].detach()
        e_mask_train = e_mask[train_mask]
        #
        loss_suf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_train, e_mask_train, dim=1)

        L_suf = (loss_suf_elem * weights).mean()


        with torch.no_grad():
            e_mask_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s_t, z_y_t, sens, labels, train_mask)
            e_mask_cf = e_mask_cf[train_mask].detach()

        loss_cf_elem = 1.0 - torch.nn.functional.cosine_similarity(e_mask_train, e_mask_cf, dim=1)
        L_cf = (loss_cf_elem * weights).mean()

        # 正则化：稀疏性（用 soft 的均值更平滑）
        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            L_sp_structure = m_rest_soft.mean()
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        # --- 统计信息 ---
        with torch.no_grad():
            # hard(0/1) 的真实稀疏度（不含自环）
            if (~is_self_loop).any():
                real_sparsity_hard = 1.0 - y_hard01[~is_self_loop].mean().item()
                kept_ratio_hard = y_hard01[~is_self_loop].mean().item()
            else:
                real_sparsity_hard = 0.0
                kept_ratio_hard = 1.0

            # soft 的“平均保留强度”（不含自环）
            kept_ratio_soft = m_rest_soft.mean().item() if m_rest_soft.numel() > 0 else 1.0

        count = {
            "mask_mean": structure_mask.mean().item(),
            "mask_min": structure_mask.min().item(),
            "mask_max": structure_mask.max().item(),

            # hard / soft 两套口径
            "kept_ratio_hard": kept_ratio_hard,  # hard(0/1) 保留边比例
            "sparsity_hard": real_sparsity_hard,  # hard(0/1) 稀疏度
            "kept_ratio_soft": kept_ratio_soft,  # soft 平均强度

            # 你原本的权重监控
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            # 边数统计（对 m 是 eps/1 的 forward 权重，不等价于 hard(0/1) 的 kept_edge_count）
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
        # 1. 归一化 (L2 Normalize)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # 2. 计算相似度矩阵 (Dot Product)
        # sim_matrix[i][j] = q[i] * k[j]
        # 对角线元素是正样本对 (Positives)，其他是负样本对 (Negatives)
        sim_matrix = torch.mm(q, k.t()) / tau

        # 3. 构造标签 (正样本在对角线上)
        # 这里的任务是让第 i 个 q 找到第 i 个 k
        batch_size = q.shape[0]
        labels = torch.arange(batch_size, device=q.device)

        # 4. 计算加权交叉熵
        # reduction='none' 允许我们对每个样本单独加权
        loss_ce = F.cross_entropy(sim_matrix, labels, reduction='none')

        # 5. 应用权重
        loss = (loss_ce * weights).mean()

        return loss

    def compute_structure_mask_loss_contrastive(self, g, x, sens, train_mask, labels, mask_threshold=0.3, tau=1.0, eps=1e-6,
                                                con_tau=0.1):
        """
        基于对比学习 (Contrastive Learning) 的结构掩码损失计算
        Args:
            con_tau: 对比学习的温度系数 (建议 0.07 ~ 0.2)
        """
        # [1] 显式重构自环
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        # 1. 基础 Embedding (作为 Target Key)
        # [关键] 这里必须 detach，防止梯度流向 Encoder 导致"为了迎合 Mask 而改变原特征"
        e_origin = self.encoder(g, x)
        e_origin_detach = e_origin.detach()

        # 2. 反事实相关 & 权重计算
        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e_origin)
            # 生成反事实目标 (作为 L_cf 的 Target Key)
            e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)

        weights_all = self.calculate_geometric_weights(e_origin, e_cf)
        weights = weights_all[train_mask].detach()

        # 3. 计算 Mask Logits
        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e_origin[src], e_origin[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()

        # =======================================================
        # [核心技巧] Hard Gumbel-Sigmoid
        # =======================================================

        # 1. 生成 Gumbel 噪声
        if self.training:
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        # 锁定自环概率为 1
        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        # 2. Hard Mask (0/1)
        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        # 3. 保证 edge_weight 严格 > 0 (APPNP 要求)
        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        # 4. STE: Forward 用 hard, Backward 用 soft
        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        # 双重保险
        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        # 5. 基于 Mask 的前向传播 (Query)
        e_mask = self.encoder(g, x, edge_weight=structure_mask)

        # 这里的 fairINN 输出可用于额外的 loss (如 L_y)，如果需要的话
        z_y_t, z_s_t, _, pred_y_t, _ = self.fairINN(e_mask)

        # =======================================================
        # [修改点] 计算对比损失 (Contrastive Loss)
        # =======================================================

        # 提取训练集部分的 Embedding
        q_mask = e_mask[train_mask]  # Query
        k_origin = e_origin_detach[train_mask]  # Key (Positive for L_suf)
        k_cf = e_cf[train_mask]  # Key (Positive for L_cf)
        # L_suf = F.binary_cross_entropy_with_logits(pred_y_t[train_mask], labels[train_mask].unsqueeze(1).float())


        # 1. L_suf (Sufficiency): Mask 后的图应该像原图，且与其他节点的原图区分开
        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        # 2. L_cf (Necessary/Counterfactual): Mask 后的图应该像反事实目标
        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights, tau=con_tau)

        # =======================================================
        # 正则化 (Sparsity)
        # =======================================================
        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            L_sp_structure = m_rest_soft.mean()
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        # =======================================================
        # 统计信息 (Count) - 完整保留
        # =======================================================
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

            # hard / soft 两套口径
            "kept_ratio_hard": kept_ratio_hard,
            "sparsity_hard": real_sparsity_hard,
            "kept_ratio_soft": kept_ratio_soft,

            # 权重监控
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            # 边数统计
            "edge_weight_sum": structure_mask.detach().sum().item(),

            # [新增] 监控对比损失数值
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
        # 1. 原始 Embedding (作为 L_suf 的 Target Key)
        # [关键] 必须 detach，我们不希望 Mask Loss 去更新 Encoder 的参数来迎合 Mask
        e = self.encoder(g, x)
        e_detach = e.detach()

        # 2. 生成原始反事实 Embedding (作为 L_cf 的 Target Key)
        with torch.no_grad():
            z_y, z_s, _, _, _ = self.fairINN(e)
            # 建议：这里的 generate 内部 alpha 设为 1.0 或 None，确保产生强反事实
            e_cf, _, _, R = self.generate_nearest_neighbor_counterfactual_with_R(z_s, z_y, sens, labels, train_mask)

        # 3. 计算样本权重 (Hard Sample Mining)
        weights_all = self.calculate_geometric_weights(e, e_cf)
        weights = weights_all[train_mask].detach()

        # 4. Mask 生成
        # 注意：Generator 需要梯度，所以这里输入的 e 不 detach (或者根据 Generator 设计决定)
        # 通常 Generator 依赖 e 的上下文信息
        mask = self.featureMask_Generator(x, e, e_cf)
        mask = mask[train_mask]

        # 5. 构造 Masked 特征
        x_base = self.get_x_base(x, train_mask)
        x_tilde = mask * x[train_mask] + (1 - mask) * x_base


        # 6. Masked Embedding (Query)
        # 我们需要构建一个全图的 x_masked 才能输入 Encoder
        x_masked_full = x.clone()
        x_masked_full[train_mask] = x_tilde

        e_tilde = self.encoder(g, x_masked_full)
        _,_,_,logits_y, _ = self.fairINN(e_tilde)

        # =======================================================
        # [核心修改] 转换为对比损失 (Contrastive Loss)
        # =======================================================

        # 提取训练集部分的 Embedding
        q_mask = e_tilde[train_mask]  # Query: 掩码后的特征表示
        k_origin = e_detach[train_mask]  # Key (Positive for L_suf): 原始特征表示
        k_cf = e_cf[train_mask]  # Key (Positive for L_cf): 反事实特征表示
        # L_suf = F.binary_cross_entropy_with_logits(logits_y[train_mask], labels[train_mask].unsqueeze(1).float())



        # (A) Sufficiency Loss (L_suf)
        # 目标：Mask 后的特征应当保留原始节点的语义身份 (Identity Preservation)
        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        # (B) Counterfactual Consistency (L_cf)
        # 目标：Mask 后的特征应当在潜空间中对齐反事实目标 (Fairness Alignment)
        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights, tau=con_tau)

        # =======================================================
        # (C) Regularization (Sparsity & Binarization)
        # =======================================================
        # 稀疏性约束：希望 mask 尽可能接近 0 (去掉非必要特征)
        L_sp = mask.mean()

        # 二值化约束：希望 mask 接近 0 或 1，而不是中间值
        # mask * (1 - mask) 在 0.5 时最大，在 0/1 时最小
        L_bin = (mask * (1 - mask)).mean()

        # =======================================================
        # 统计信息
        # =======================================================
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


        # 3. 计算样本权重 (Hard Sample Mining)
        weights_all = self.calculate_geometric_weights(e, e_cf)
        weights = weights_all[train_mask].detach()
        R_score = R[train_mask].detach()
        weights_cf = R_score * weights

        # 4. Mask 生成
        # 注意：Generator 需要梯度，所以这里输入的 e 不 detach (或者根据 Generator 设计决定)
        # 通常 Generator 依赖 e 的上下文信息
        mask = self.featureMask_Generator(x, e, e_cf)
        mask = mask[train_mask]

        # 5. 构造 Masked 特征
        x_base = self.get_x_base(x, train_mask)
        x_tilde = mask * x[train_mask] + (1 - mask) * x_base


        # 6. Masked Embedding (Query)
        # 我们需要构建一个全图的 x_masked 才能输入 Encoder
        x_masked_full = x.clone()
        x_masked_full[train_mask] = x_tilde

        e_tilde = self.encoder(g, x_masked_full)
        # _,_,_,logits_y, _ = self.fairINN(e_tilde)

        # =======================================================
        # [核心修改] 转换为对比损失 (Contrastive Loss)
        # =======================================================

        e_detach = e.detach()

        # 提取训练集部分的 Embedding
        q_mask = e_tilde[train_mask]  # Query: 掩码后的特征表示
        k_origin = e_detach[train_mask]  # Key (Positive for L_suf): 原始特征表示
        k_cf = e_cf[train_mask]  # Key (Positive for L_cf): 反事实特征表示
        # L_suf = F.binary_cross_entropy_with_logits(logits_y[train_mask], labels[train_mask].unsqueeze(1).float())



        # (A) Sufficiency Loss (L_suf)
        # 目标：Mask 后的特征应当保留原始节点的语义身份 (Identity Preservation)
        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        # (B) Counterfactual Consistency (L_cf)
        # 目标：Mask 后的特征应当在潜空间中对齐反事实目标 (Fairness Alignment)
        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights_cf, tau=con_tau)

        # =======================================================
        # (C) Regularization (Sparsity & Binarization)
        # =======================================================
        # 稀疏性约束：希望 mask 尽可能接近 0 (去掉非必要特征)
        L_sp = mask.mean()

        # 二值化约束：希望 mask 接近 0 或 1，而不是中间值
        # mask * (1 - mask) 在 0.5 时最大，在 0/1 时最小
        L_bin = (mask * (1 - mask)).mean()

        # =======================================================
        # 统计信息
        # =======================================================
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
        # [1] 显式重构自环
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        # 1. 基础 Embedding (作为 Target Key)
        # [关键] 这里必须 detach，防止梯度流向 Encoder 导致"为了迎合 Mask 而改变原特征"
        e_origin = e
        e_origin_detach = e_origin.detach()



        weights_all = self.calculate_geometric_weights(e_origin, e_cf)
        weights = weights_all[train_mask].detach()
        R_score = R[train_mask].detach()
        weights_cf = R_score * weights


        # 3. 计算 Mask Logits
        src, dst = g.edges()
        is_self_loop = (src == dst)
        edge_feat = torch.cat([e_origin[src], e_origin[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()

        # =======================================================
        # [核心技巧] Hard Gumbel-Sigmoid
        # =======================================================

        # 1. 生成 Gumbel 噪声
        if self.training:
            U = torch.rand_like(mask_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(U))
            y_soft = torch.sigmoid((mask_logits + gumbel_noise) / max(tau, 1e-8))
        else:
            y_soft = torch.sigmoid(mask_logits)

        # 锁定自环概率为 1
        y_soft = y_soft.masked_fill(is_self_loop, 1.0)

        # 2. Hard Mask (0/1)
        y_hard01 = (y_soft > mask_threshold).float()
        y_hard01 = y_hard01.masked_fill(is_self_loop, 1.0)

        # 3. 保证 edge_weight 严格 > 0 (APPNP 要求)
        y_hard_pos = y_hard01 * (1.0 - eps) + eps
        y_hard_pos = y_hard_pos.masked_fill(is_self_loop, 1.0)

        # 4. STE: Forward 用 hard, Backward 用 soft
        structure_mask = y_hard_pos - y_soft.detach() + y_soft

        # 双重保险
        structure_mask = structure_mask.clamp(min=eps)
        structure_mask = structure_mask.masked_fill(is_self_loop, 1.0)

        # 5. 基于 Mask 的前向传播 (Query)
        e_mask = self.encoder(g, x, edge_weight=structure_mask)

        # 这里的 fairINN 输出可用于额外的 loss (如 L_y)，如果需要的话
        z_y_t, z_s_t, _, pred_y_t, _ = self.fairINN(e_mask)

        # =======================================================
        # [修改点] 计算对比损失 (Contrastive Loss)
        # =======================================================

        # 提取训练集部分的 Embedding
        q_mask = e_mask[train_mask]  # Query
        k_origin = e_origin_detach[train_mask]  # Key (Positive for L_suf)
        k_cf = e_cf[train_mask]  # Key (Positive for L_cf)
        # L_suf = F.binary_cross_entropy_with_logits(pred_y_t[train_mask], labels[train_mask].unsqueeze(1).float())


        # 1. L_suf (Sufficiency): Mask 后的图应该像原图，且与其他节点的原图区分开
        L_suf = self.weighted_contrastive_loss(q_mask, k_origin, weights, tau=con_tau)

        # 2. L_cf (Necessary/Counterfactual): Mask 后的图应该像反事实目标
        L_cf = self.weighted_contrastive_loss(q_mask, k_cf, weights_cf, tau=con_tau)

        # =======================================================
        # 正则化 (Sparsity)
        # =======================================================
        m_rest_soft = y_soft[~is_self_loop]
        if m_rest_soft.numel() > 0:
            L_sp_structure = m_rest_soft.mean()
        else:
            L_sp_structure = torch.tensor(0.0, device=structure_mask.device)

        # =======================================================
        # 统计信息 (Count) - 完整保留
        # =======================================================
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

            # hard / soft 两套口径
            "kept_ratio_hard": kept_ratio_hard,
            "sparsity_hard": real_sparsity_hard,
            "kept_ratio_soft": kept_ratio_soft,

            # 权重监控
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),

            # 边数统计
            "edge_weight_sum": structure_mask.detach().sum().item(),

            # [新增] 监控对比损失数值
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
        # mean_val = feature_cf[train_node, sens_index].mean()
        # feature_cf[:, sens_index] = mean_val
        # 0. 统一处理自环口径 (确保与训练一致)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)


        with torch.no_grad():
            # 1. 前向传播
            e = self.encoder(g, x)
            z_y, z_s, _, _, _ = self.fairINN(e)

            # 2. 生成反事实目标
            e_cf, _, _, _ = self.generate_nearest_neighbor_counterfactual_with_R(z_s, z_y, sens, labels, train_mask)

            # 3. 生成 Soft Masks
            # 3.1 特征
            mask_score_feat = self.featureMask_Generator(x, e, e_cf)
            mask_score_feat[:, sens_idx] = 0.0

            # 3.2 结构
            src, dst = g.edges()
            is_self_loop = (src == dst)
            edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
            mask_logits_struct = self.structureMask_Generator(edge_feat).squeeze(-1)
            structure_probs = torch.sigmoid(mask_logits_struct)

            # [关键] Soft 阶段锁定自环必保留
            structure_probs = structure_probs.masked_fill(is_self_loop, 1.0)

            # 4. 生成 Hard Masks (0/1)
            # 4.1 特征硬掩码
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

            # 4.2 结构硬掩码
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
            # [关键] Hard 阶段再次确保自环为 1
            struct_mask_hard = struct_mask_hard.masked_fill(is_self_loop, 1.0)

        # 返回掩码 Tensor 和 自环指示器(供后续使用)
        return feat_mask_hard, struct_mask_hard, is_self_loop

    def get_mask_prediction(self, g, x, sens, train_mask, loss_type=None, mask_threshold=0.3, tau=0.1):
        self.eval()
        e = self.encoder(g, x)
        # logit = self.encoder.prediction(e)
        src, dst = g.edges()
        is_self_loop = (src == dst)  # 识别自环

        z_y, z_s, pred_s, pred_y, _ = self.fairINN(e)  # 注意你的 fairINN.forward 返回 (z_y, z_s, pred_s, pred_y, log_det)
        # 用你已有的 counterfactual_embedding：返回 e_cf, z_cf
        e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)  # 你需要把 margin/clamp 参数接进去

        mask = self.featureMask_Generator(x, e, e_cf)
        x_base = self.get_x_base(x, train_mask)  # [1, F]
        x_tilde = mask * x + (1.0 - mask) * x_base

        src, dst = g.edges()
        edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
        mask_logits = self.structureMask_Generator(edge_feat).squeeze()

        if loss_type == 'Soft' or loss_type == 'Entropy':
            structure_mask = torch.sigmoid(mask_logits)
        elif loss_type == 'STE':
            # mask_probs = torch.sigmoid(mask_logits)
            # mask_hard = (mask_probs > mask_threshold).float()
            # structure_mask = mask_hard - mask_probs.detach() + mask_probs

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

        e_tilde = self.encoder(g, x_tilde, edge_weight=structure_mask)  # [N,d]
        _, _, _, logit_tilde, _ = self.fairINN(e_tilde)
        # logit_tilde = self.encoder.prediction(e_tilde)

        return x_tilde, logit_tilde, structure_mask

    def get_complement_masks(self, feat_mask, struct_mask, is_self_loop):
        """
        Step 2: 输入解释掩码 -> 生成反向(补图)掩码
        """
        # 1. 特征反向: 直接取反 (1 -> 0, 0 -> 1)
        comp_feat_mask = 1.0 - feat_mask

        # 2. 结构反向:
        # 先取反: 原本保留的边(1)变成0，原本删除的边(0)变成1
        comp_struct_mask = 1.0 - struct_mask

        # [关键] 物理修正:
        # 取反后，原本必留的自环变成了 0。
        # 但为了让 GNN 能在反向图上运行，必须强制把自环加回来！
        comp_struct_mask = comp_struct_mask.masked_fill(is_self_loop, 1.0)

        return comp_feat_mask, comp_struct_mask

    def get_explained_graph(self, g, x, feat_mask, struct_mask, train_mask):
        """
        Step 3: 输入掩码 + 原数据 -> 构建物理子图对象 (DGLGraph) 和 掩码后特征
        """
        # 1. 应用特征掩码
        # 获取基准值 (均值或其他)
        x_base = self.get_x_base(x, train_mask)
        # 掩码公式: m*x + (1-m)*base
        x_new = feat_mask * x + (1.0 - feat_mask) * x_base

        # 2. 构建物理子图
        # 找出 mask 为 1 的边的索引
        kept_indices = torch.nonzero(struct_mask > 0).squeeze(-1)

        if kept_indices.numel() > 0:
            # 构建子图 (relabel_nodes=False 保证特征矩阵行号对齐)
            g_new = dgl.edge_subgraph(g, kept_indices, relabel_nodes=False)
        else:
            # 极端情况：空图
            g_new = dgl.graph(([], []), num_nodes=g.number_of_nodes(), device=g.device)

        # 3. 统一自环处理 (防止 DGL 报错)
        # 移除可能残留的旧自环，添加统一的新自环
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

        # 统一处理自环口径（与训练一致）
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        with torch.no_grad():
            # 1) 前向传播获取必要信息
            e = self.encoder(g, x)
            z_y, z_s, pred_s, pred_y, _ = self.fairINN(e)

            # 生成反事实目标
            e_cf, _, _ = self.generate_nearest_neighbor_counterfactual(z_s, z_y, sens, train_mask)

            # 2) 计算 Soft Mask
            # 2.1 特征掩码（你原逻辑保持不变）
            mask_score_feat = self.featureMask_Generator(x, e, e_cf)

            # 2.2 结构掩码（与训练一致：logits->sigmoid；推理不加gumbel噪声）
            src, dst = g.edges()
            is_self_loop = (src == dst)

            edge_feat = torch.cat([e[src], e[dst], e_cf[src], e_cf[dst]], dim=-1)
            mask_logits_struct = self.structureMask_Generator(edge_feat).squeeze(-1)
            structure_probs = torch.sigmoid(mask_logits_struct)

            # 锁定自环必保留（与训练一致：soft阶段就锁）
            structure_probs = structure_probs.masked_fill(is_self_loop, 1.0)

            # 3) 生成 Hard Mask（二值化）
            # 3.1 特征硬掩码
            feature_mask_hard = (mask_score_feat > mask_threshold).float()
            x_base = self.get_x_base(x, train_mask)
            x_final = feature_mask_hard * x + (1.0 - feature_mask_hard) * x_base

            # 3.2 结构硬掩码（0/1）
            struct_mask_hard01 = (structure_probs > mask_threshold).float()
            struct_mask_hard01 = struct_mask_hard01.masked_fill(is_self_loop, 1.0)

            # （可选）如果你想“推理阶段也用 eps/1 权重传播，而不是删边”，可以用下面这个：
            # struct_mask_hard_pos = struct_mask_hard01 * (1.0 - eps) + eps
            # struct_mask_hard_pos = struct_mask_hard_pos.masked_fill(is_self_loop, 1.0)

            # 4) 构建新图：物理删除边 + 重建自环（你原思路保留）
            kept_indices = torch.nonzero(struct_mask_hard01 > 0).squeeze(-1)

            if kept_indices.numel() > 0:
                new_g = dgl.edge_subgraph(g, kept_indices, relabel_nodes=False)
            else:
                new_g = dgl.graph(([], []), num_nodes=g.number_of_nodes(), device=g.device)

            # 统一处理自环：移除残留自环，再添加全自环（确保每个节点度数>=1）
            new_g = dgl.remove_self_loop(new_g)
            new_g = dgl.add_self_loop(new_g)

            # 5) 基于新图预测（不传 edge_weight）
            e_final = self.encoder(new_g, x_final)
            z_y, z_s, _, logit_final, _ = self.fairINN(e_final)

        # 返回结果
        if return_subgraph:
            # 多返回一个 hard mask 方便你做解释可视化/统计
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
        # --- Sensitive flip rate ---
        w = self.fairINN.classifier_s.weight.data
        b = self.fairINN.classifier_s.bias.data
        print("||w||:", w.norm().item(), "bias:", b.item())


        z_s_cf_eval = z_s_cf[:, self.fairINN.y_dim:]  # [N, 16]
        z_y_cf_eval = z_s_cf[:, :self.fairINN.y_dim]  # [N, 16]
        w = self.fairINN.classifier_s.weight.squeeze(0)  # [16]
        b = self.fairINN.classifier_s.bias.squeeze(0)  # scalar

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

        # --- Label consistency ---
        logit_y = self.fairINN.classifier_y(z_y).squeeze(-1)  # [N]
        logit_y_cf = self.fairINN.classifier_y(z_y_cf_eval).squeeze(-1)
        pred_y = (logit_y > thr_logit)
        pred_y_cf = (logit_y_cf > thr_logit)

        consistency_y = (pred_y == pred_y_cf).float().mean()

        # 可选：连续量诊断（更细）
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



