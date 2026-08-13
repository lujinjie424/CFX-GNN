import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

def accuracy(y_pred, y_true):
    """计算分类准确率
    Args:
        y_pred (Tensor): 模型预测logits [N, C]
        y_true (Tensor): 真实标签 [N]

    Returns:
        acc (float): 正确预测的比例
    """
    correct = y_pred.eq(y_true).double()
    return correct.sum().item() / len(y_true)

def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                 sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))

    return parity.item(), equality.item()

def evaluate(pre, label, sens, cf_pre=None):
    y_true = label.squeeze().long()
    sens = sens.squeeze().long()
    preds = (pre.squeeze() > 0).type_as(label)

    metrics = {}

    metrics['acc'] = accuracy(preds, y_true)

    metrics['F1'] = f1_score(
        y_true.cpu().numpy(),
        preds.cpu().numpy()
    )

    metrics['auc_roc'] = roc_auc_score(
        y_true.cpu().numpy(),
        preds.detach().cpu().numpy()
    )

    metrics['parity'], metrics['equality'] = fair_metric(
        preds.cpu().numpy(),
        y_true.cpu().numpy(),
        sens.cpu().numpy()
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true.cpu().numpy(),
        preds.cpu().numpy()
    ).ravel()
    metrics['fpr'] = fp / (fp + tn + 1e-12)

    if cf_pre is not None:
        cf_preds = (cf_pre.squeeze() > 0).type_as(label)
        metrics['cf'] = 1.0 - preds.eq(cf_preds).float().mean().item()
        metrics['cf_logit_gap'] = torch.abs(
            pre.squeeze().float() - cf_pre.squeeze().float()
        ).mean().item()
        metrics['cf_prob_gap'] = torch.abs(
            torch.sigmoid(pre.squeeze().float()) - torch.sigmoid(cf_pre.squeeze().float())
        ).mean().item()
    else:
        metrics['cf'] = None
        metrics['cf_logit_gap'] = None
        metrics['cf_prob_gap'] = None

    return metrics
