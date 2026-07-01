from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm_pipeline.config import ModelConfig  # noqa: E402
from phm_pipeline.model import (  # noqa: E402
    RULTransformer,
    build_model,
    model_config_from_dict,
)


def test_rul_transformer_forward_shapes_with_mask() -> None:
    torch.manual_seed(0)
    model = RULTransformer(
        ModelConfig(
            feature_dim=8,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            use_uncertainty_head=True,
        )
    )
    x = torch.randn(4, 12, 8)
    times = torch.arange(12, dtype=torch.float32).repeat(4, 1) * 600.0
    mask = torch.ones(4, 12, dtype=torch.bool)
    mask[0, :3] = False
    rul, log_var = model(x, times, mask)
    assert rul.shape == (4,)
    assert log_var.shape == (4,)
    assert torch.isfinite(rul).all()
    assert (rul > 0.0).all()


def test_rul_transformer_without_uncertainty_head_returns_tensor() -> None:
    torch.manual_seed(0)
    model = RULTransformer(
        ModelConfig(feature_dim=6, hidden_dim=16, num_layers=1, num_heads=2, dropout=0.0, use_uncertainty_head=False)
    )
    x = torch.randn(2, 5, 6)
    times = torch.arange(5, dtype=torch.float32).repeat(2, 1) * 600.0
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = model(x, times, mask)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2,)
    assert (out > 0.0).all()


def test_build_model_rejects_unknown_architecture() -> None:
    try:
        build_model(ModelConfig(feature_dim=4, architecture="wavelet_transformer"))
    except ValueError as exc:
        assert "architecture" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown architecture")


def test_model_config_from_dict_ignores_unknown_keys() -> None:
    # Old checkpoints carried fields for architectures that no longer exist.
    payload = {
        "feature_dim": 10,
        "hidden_dim": 32,
        "fourier_modes": 16,
        "wavelet_components": 4,
        "condition_feature_indices": [0, 1, 2],
    }
    config = model_config_from_dict(payload)
    assert config.feature_dim == 10
    assert config.hidden_dim == 32
