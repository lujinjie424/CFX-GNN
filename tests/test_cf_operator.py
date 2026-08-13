"""Smoke tests for counterfactual operator and risk scoring."""

import torch

from cfx_gnn.modules.counterfactual_operator import CounterfactualOperator
from cfx_gnn.modules.risk_weighting import compute_cf_risk


class ToyINN(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.y_dim = dim // 2

    def forward(self, h):
        z_u = h[:, : self.y_dim]
        z_b = h[:, self.y_dim :]
        return z_u, z_b, None, None, None

    def inn(self, z, reverse=False):
        if not reverse:
            return z, torch.zeros(z.size(0), device=z.device)
        return z


def test_counterfactual_operator_shapes():
    h = torch.randn(7, 6)
    donor_indices = torch.tensor(
        [
            [1, 2],
            [0, 2],
            [0, 1],
            [4, 5],
            [3, 5],
            [3, 4],
            [0, 3],
        ]
    )
    operator = CounterfactualOperator(ToyINN(dim=6))
    operator.configure_reference(sensitive=torch.tensor([0, 0, 0, 1, 1, 1, 0]))
    h_cf_set = operator.apply(h, donor_indices)
    assert h_cf_set.shape == (7, 1, 6)
    assert operator.last_cf_stats["cf_ref_mode"] == "local_quantile"
    assert operator.reconstruction_error(h).item() < 1e-10


def test_compute_cf_risk_shapes():
    logits = torch.zeros(7, 1)
    logits_cf = torch.ones(7, 2, 1)
    risk = compute_cf_risk(logits, logits_cf)
    assert risk.shape == (7,)
    assert torch.allclose(risk, torch.ones(7))
