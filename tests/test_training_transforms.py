from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.training import (  # noqa: E402
    apply_temporal_postprocess,
    inverse_transform_rul_target,
    scale_sweep_metrics,
    transform_rul_target,
)
import pandas as pd  # noqa: E402


def test_log1p_target_transform_round_trips() -> None:
    rul = np.array([1.0, 149.0, 10_000.0], dtype=float)
    transformed = transform_rul_target(rul, "log1p")
    recovered = inverse_transform_rul_target(torch.tensor(transformed), "log1p")
    np.testing.assert_allclose(recovered.numpy(), rul, rtol=1e-6)


def test_prediction_scale_is_applied_after_inverse_transform() -> None:
    transformed = torch.tensor([np.log1p(100.0)], dtype=torch.float32)
    recovered = inverse_transform_rul_target(
        transformed,
        "log1p",
        prediction_scale=0.5,
    )
    assert torch.allclose(recovered, torch.tensor([50.0]), atol=1e-4)


def test_scale_sweep_metrics_selects_best_validation_scale() -> None:
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([200.0, 400.0, 600.0])
    metrics = scale_sweep_metrics(
        actual,
        predicted,
        scale_min=0.25,
        scale_max=1.0,
        scale_steps=4,
    )
    assert metrics["prediction_scale"] == 0.5
    assert metrics["score"] > 0.99


def test_eol_quantile_smoothing_is_causal() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["Train1"] * 4,
            "mid_time_s": [0.0, 600.0, 1200.0, 1800.0],
            "predicted_rul_s": [1000.0, 800.0, 650.0, 400.0],
        }
    )
    smoothed = apply_temporal_postprocess(
        frame,
        method="eol_quantile",
        group_column="run_id",
        time_column="mid_time_s",
        prediction_column="predicted_rul_s",
        eol_quantile=0.5,
        blend=1.0,
        floor_s=1.0,
    )
    # First point cannot use any future information: it re-derives from its own EOL.
    assert abs(float(smoothed.iloc[0]) - 1000.0) < 1e-6
    assert (smoothed.to_numpy() >= 1.0).all()


def test_decay_smoothing_is_monotone_non_increasing() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["Train1"] * 4,
            "mid_time_s": [0.0, 600.0, 1200.0, 1800.0],
            "predicted_rul_s": [1000.0, 1200.0, 650.0, 700.0],
        }
    )
    smoothed = apply_temporal_postprocess(
        frame,
        method="decay",
        group_column="run_id",
        time_column="mid_time_s",
        prediction_column="predicted_rul_s",
        blend=1.0,
        floor_s=1.0,
    ).to_numpy()
    assert np.all(np.diff(smoothed) <= 1e-6)
