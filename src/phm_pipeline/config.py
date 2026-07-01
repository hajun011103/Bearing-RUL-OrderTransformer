"""Configuration objects shared by feature extraction and modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class BearingGeometry:
    """Bearing geometry used to compute fault orders.

    Competition organizers do not always publish full bearing geometry in the
    same place as the raw files. If exact values are known, fill in the physical
    dimensions. Otherwise pass explicit fault orders. Orders are frequency
    ratios relative to shaft frequency, so they remain stable under RPM changes.
    """

    rolling_elements: int | None = None
    ball_diameter_mm: float | None = None
    pitch_diameter_mm: float | None = None
    contact_angle_deg: float = 0.0
    fault_orders: dict[str, float] = field(
        default_factory=lambda: {
            # Conservative generic deep-groove defaults. Replace with the exact
            # bearing geometry when it is available from the challenge notice.
            "ftf": 0.40,
            "bpfo": 3.58,
            "bpfi": 5.42,
            "bsf": 2.36,
        }
    )

    def resolved_fault_orders(self) -> dict[str, float]:
        """Return fault orders from explicit values or physical geometry."""

        if (
            self.rolling_elements is None
            or self.ball_diameter_mm is None
            or self.pitch_diameter_mm is None
        ):
            return dict(self.fault_orders)

        import math

        n = float(self.rolling_elements)
        ratio = self.ball_diameter_mm / self.pitch_diameter_mm
        cos_a = math.cos(math.radians(self.contact_angle_deg))
        bpfo = n * 0.5 * (1.0 - ratio * cos_a)
        bpfi = n * 0.5 * (1.0 + ratio * cos_a)
        bsf = 0.5 / ratio * (1.0 - (ratio * cos_a) ** 2)
        ftf = 0.5 * (1.0 - ratio * cos_a)
        out = dict(self.fault_orders)
        out.update({"ftf": ftf, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf})
        return out


@dataclass(slots=True)
class FeatureConfig:
    """Feature extraction settings for the order-domain pipeline."""

    sample_rate_hz: float = 25_600.0
    operation_rate_hz: float = 0.1
    segment_seconds: float = 60.0
    rest_seconds: float = 540.0
    acquisition_period_seconds: float = 600.0
    samples_per_revolution: int = 128
    max_order: float = 40.0
    order_band_width: float = 0.15
    order_harmonics: int = 5
    envelope_band_hz: tuple[float, float] = (500.0, 12_000.0)
    analysis_decimate: int = 2
    geometry: BearingGeometry = field(default_factory=BearingGeometry)


@dataclass(slots=True)
class ModelConfig:
    """Neural network architecture settings."""

    feature_dim: int
    architecture: str = "transformer"
    hidden_dim: int = 192
    num_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.15
    max_context: int = 64
    use_uncertainty_head: bool = True


@dataclass(slots=True)
class TrainingConfig:
    """Training loop settings."""

    features_path: Path
    output_dir: Path = Path("artifacts/runs/default")
    target_column: str = "rul_s"
    group_column: str = "run_id"
    time_column: str = "mid_time_s"
    context_length: int = 48
    min_context: int = 4
    split_mode: str = "run"
    val_runs: tuple[str, ...] = ()
    val_fraction: float = 0.25
    target_transform: str = "identity"
    prediction_scale: float = 1.0
    temporal_postprocess: str = "none"
    temporal_decay_slack_s: float = 0.0
    temporal_eol_quantile: float = 0.5
    temporal_postprocess_blend: float = 1.0
    temporal_postprocess_floor_s: float = 1.0
    optimize_val_scale: bool = False
    val_scale_min: float = 0.05
    val_scale_max: float = 0.80
    val_scale_steps: int = 151
    batch_size: int = 64
    epochs: int = 150
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 25
    num_workers: int = 0
    seed: int = 2026
    device: str = "auto"
    conservative_er_target: float = 7.5
    conservative_weight: float = 0.03
    score_loss_weight: float = 1.0
    relative_huber_weight: float = 0.0
    over_prediction_weight: float = 2.0
    uncertainty_weight: float = 0.02
    model_architecture: str = "transformer"
    hidden_dim: int = 192
    num_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.15
