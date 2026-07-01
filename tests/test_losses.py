from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm_pipeline.losses import AsymmetricRULLoss, official_score_torch  # noqa: E402


def test_official_score_piecewise_halves_at_expected_errors() -> None:
    actual = torch.tensor([100.0, 100.0, 100.0])
    pred = torch.tensor([120.0, 100.0, 50.0])
    score = official_score_torch(actual, pred)
    expected = torch.tensor([0.5, 1.0, 0.5])
    assert torch.allclose(score, expected, atol=1e-6)


def test_asymmetric_loss_penalizes_over_prediction_more() -> None:
    target = torch.tensor([100.0, 100.0, 100.0])
    pred_over = torch.tensor([110.0, 112.0, 108.0])
    pred_under = torch.tensor([90.0, 88.0, 92.0])
    loss_fn = AsymmetricRULLoss(conservative_weight=0.0)

    loss_over = loss_fn(pred_over, target)
    loss_under = loss_fn(pred_under, target)
    assert loss_over > loss_under


def test_loss_accepts_uncertainty_head_tuple() -> None:
    pred = torch.tensor([95.0, 100.0, 102.0])
    log_var = torch.tensor([-1.0, 0.0, 0.3])
    target = torch.tensor([100.0, 98.0, 101.0])
    loss = AsymmetricRULLoss()((pred, log_var), target)
    assert torch.isfinite(loss)

