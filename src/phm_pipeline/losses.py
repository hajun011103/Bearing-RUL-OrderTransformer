"""Official score and conservative asymmetric training objectives."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


LOG_HALF = math.log(0.5)


def official_score_numpy(
    actual_rul: np.ndarray,
    predicted_rul: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute the competition score for each sample.

    Error is positive for under-prediction, which is safer in maintenance. The
    over-prediction branch decays to 0.5 at only -20% error, while the safe
    branch reaches 0.5 at +50% error.
    """

    actual = np.maximum(np.asarray(actual_rul, dtype=float), eps)
    pred = np.asarray(predicted_rul, dtype=float)
    er = 100.0 * (actual - pred) / actual
    score = np.where(
        er <= 0.0,
        np.exp(-LOG_HALF * (er / 20.0)),
        np.exp(LOG_HALF * (er / 50.0)),
    )
    return score


def official_score_torch(
    actual_rul: torch.Tensor,
    predicted_rul: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Torch version of the official score."""

    actual = actual_rul.clamp_min(eps)
    er = 100.0 * (actual - predicted_rul) / actual
    return torch.where(
        er <= 0.0,
        torch.exp(-LOG_HALF * (er / 20.0)),
        torch.exp(LOG_HALF * (er / 50.0)),
    )


class AsymmetricRULLoss(nn.Module):
    """Differentiable loss aligned to the official asymmetric scoring metric.

    The main term is negative log score. Extra regularization nudges the model
    toward a conservative error band, e.g. +5% to +10% Er, and strongly penalizes
    dangerous over-prediction.
    """

    def __init__(
        self,
        *,
        conservative_er_target: float = 7.5,
        conservative_weight: float = 0.03,
        score_weight: float = 1.0,
        relative_huber_weight: float = 0.0,
        over_prediction_weight: float = 2.0,
        uncertainty_weight: float = 0.02,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.conservative_er_target = conservative_er_target
        self.conservative_weight = conservative_weight
        self.score_weight = score_weight
        self.relative_huber_weight = relative_huber_weight
        self.over_prediction_weight = over_prediction_weight
        self.uncertainty_weight = uncertainty_weight
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        actual_rul: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(prediction, tuple):
            pred_rul, log_var = prediction
        else:
            pred_rul = prediction
            log_var = None

        pred_rul = pred_rul.squeeze(-1)
        actual = actual_rul.squeeze(-1).clamp_min(self.eps)
        er = 100.0 * (actual - pred_rul) / actual

        neg_log_score = torch.where(
            er <= 0.0,
            (-LOG_HALF) * (-er / 20.0),
            -LOG_HALF * (er / 50.0),
        )

        # Directly discourage predicted RUL beyond the true RUL. This is the
        # competition-dangerous side and deserves more pressure than symmetric
        # regression loss would give it.
        relative_error = (pred_rul - actual) / actual
        relative_over = F.relu(relative_error)
        over_penalty = self.over_prediction_weight * relative_over.pow(2)

        conservative_penalty = self.conservative_weight * F.smooth_l1_loss(
            er,
            torch.full_like(er, self.conservative_er_target),
            reduction="none",
            beta=10.0,
        )
        relative_huber = self.relative_huber_weight * F.smooth_l1_loss(
            relative_error,
            torch.zeros_like(relative_error),
            reduction="none",
            beta=0.25,
        )

        loss = self.score_weight * neg_log_score + relative_huber + over_penalty + conservative_penalty
        if log_var is not None:
            log_var = log_var.squeeze(-1).clamp(-8.0, 8.0)
            gaussian_nll = 0.5 * (torch.exp(-log_var) * relative_error.pow(2) + log_var)
            loss = loss + self.uncertainty_weight * gaussian_nll
        return loss.mean()
