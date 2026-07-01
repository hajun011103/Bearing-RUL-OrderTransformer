"""Time-gap-aware Transformer for bearing RUL prediction.

This module intentionally contains only the architecture used by the abstract:
a Transformer encoder that consumes padded order-domain feature windows plus the
true elapsed timestamps and acquisition gaps. Exploratory encoders (wavelet /
PINNsFormer, Fourier neural operator, GRU autoregressive, FiLM conditioning)
that were tried during development are documented in ``docs/experiments.md``.
"""

from __future__ import annotations

from dataclasses import fields
import math

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class ContinuousTimeEncoding(nn.Module):
    """Sinusoidal encoding from actual elapsed time and acquisition gaps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        frequencies = torch.exp(
            torch.linspace(math.log(1e-4), math.log(1.0), math.ceil(dim / 2))
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, times_s: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Log-compress time so early and late life both get usable resolution.
        times = torch.log1p(times_s.clamp_min(0.0) / 60.0)
        angles = times.unsqueeze(-1) * self.frequencies
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)[..., : self.dim]
        if mask is not None:
            enc = enc * mask.unsqueeze(-1).to(enc.dtype)
        return enc


class GapEncoding(nn.Module):
    """Learned embedding of elapsed time since the previous observation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, times_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(times_s)
        delta[:, 1:] = (times_s[:, 1:] - times_s[:, :-1]).clamp_min(0.0)
        # Three scales help the network distinguish a normal 10-minute jump
        # from shorter padding artifacts or future test schedules.
        features = torch.stack(
            [
                torch.log1p(delta / 60.0),
                torch.log1p(delta / 600.0),
                (delta > 300.0).to(times_s.dtype),
            ],
            dim=-1,
        )
        return self.proj(features) * mask.unsqueeze(-1).to(times_s.dtype)


class GatedResidualBlock(nn.Module):
    """Small feed-forward block that stabilizes tabular time-series features."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        value, gate = self.fc(x).chunk(2, dim=-1)
        x = F.silu(value) * torch.sigmoid(gate)
        x = self.out(self.dropout(x))
        return residual + self.dropout(x)


def _last_valid(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.long().sum(dim=1).clamp_min(1)
    last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
    return encoded.gather(dim=1, index=last_idx).squeeze(1)


class RULTransformer(nn.Module):
    """Transformer encoder for discontinuous segment-level RUL prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        self.time_encoding = ContinuousTimeEncoding(config.hidden_dim)
        self.gap_encoding = GapEncoding(config.hidden_dim)
        self.pre_blocks = nn.Sequential(
            GatedResidualBlock(config.hidden_dim, config.dropout),
            GatedResidualBlock(config.hidden_dim, config.dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.uncertainty_head = (
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            if config.use_uncertainty_head
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        times_s: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict RUL from padded sequences.

        Args:
            x: ``(batch, time, features)`` standardized feature tensor.
            times_s: actual elapsed timestamps in seconds.
            mask: boolean tensor where True marks real observations.
        """

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.feature_proj(self.input_norm(x))
        h = h + self.time_encoding(times_s, mask=mask) + self.gap_encoding(times_s, mask)
        h = self.pre_blocks(h)
        key_padding_mask = ~mask.bool()
        encoded = self.encoder(h, src_key_padding_mask=key_padding_mask)

        pooled = _last_valid(encoded, mask)

        # Softplus enforces non-negative RUL while leaving gradients healthy.
        rul = F.softplus(self.head(pooled)).squeeze(-1) + 1e-3
        if self.uncertainty_head is None:
            return rul
        log_var = self.uncertainty_head(pooled).squeeze(-1)
        return rul, log_var


def model_config_from_dict(payload: dict) -> ModelConfig:
    """Build a ``ModelConfig`` from a checkpoint dict, ignoring unknown keys.

    Older checkpoints saved fields for architectures that no longer exist. This
    keeps only the fields the current ``ModelConfig`` defines so such
    checkpoints still load.
    """

    valid = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in payload.items() if k in valid})


def build_model(config: ModelConfig) -> nn.Module:
    """Factory kept small so scripts can swap architecture later."""

    if config.architecture == "transformer":
        return RULTransformer(config)
    raise ValueError(f"Unknown model architecture {config.architecture!r}")


def load_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a checkpoint, tolerating buffers that are not persisted."""

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if unexpected or missing:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise RuntimeError("Error(s) in loading state_dict: " + "; ".join(details))
