"""Windowed datasets, score-driven training loop, and causal smoothing.

Exploratory training extras (governing-dynamics data augmentation, learned RUL
calibration, and a PINO-style physics loss) were tried during development and
removed; see ``docs/experiments.md`` for the rationale.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import ModelConfig, TrainingConfig
from .losses import AsymmetricRULLoss, official_score_numpy
from .model import build_model, load_model_state, model_config_from_dict


NON_FEATURE_COLUMNS = {
    "run_id",
    "segment_id",
    "member",
    "zip_path",
    "start_time_s",
    "mid_time_s",
    "end_time_s",
    "run_end_s",
    "rul_s",
    "rul_segments",
    "life_fraction",
}


class Standardizer:
    """Numpy standardizer with NaN-safe fit/transform."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.mean_ = np.nan_to_num(self.mean_, nan=0.0)
        self.scale_ = np.where(np.isfinite(self.scale_) & (self.scale_ > 1e-8), self.scale_, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has not been fitted")
        out = (x - self.mean_) / self.scale_
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def to_dict(self) -> dict[str, list[float]]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer has not been fitted")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "Standardizer":
        obj = cls()
        obj.mean_ = np.asarray(payload["mean"], dtype=np.float32)
        obj.scale_ = np.asarray(payload["scale"], dtype=np.float32)
        return obj


class WindowedRULDataset(Dataset):
    """Left-padded causal windows ending at each labeled segment."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        group_column: str,
        time_column: str,
        target_column: str,
        context_length: int,
        min_context: int,
        standardizer: Standardizer | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.context_length = context_length
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self.metadata: list[dict[str, object]] = []

        for group_name, group in frame.sort_values([group_column, time_column]).groupby(group_column):
            features = group[feature_columns].to_numpy(dtype=np.float32)
            times = group[time_column].to_numpy(dtype=np.float32)
            targets = group[target_column].to_numpy(dtype=np.float32)
            segment_ids = (
                group["segment_id"].to_numpy()
                if "segment_id" in group.columns
                else np.arange(len(group))
            )
            if standardizer is not None:
                features = standardizer.transform(features).astype(np.float32)
            for end in range(max(min_context - 1, 0), len(group)):
                start = max(0, end + 1 - context_length)
                x = features[start : end + 1]
                t = times[start : end + 1]
                mask = np.ones(x.shape[0], dtype=bool)
                pad = context_length - x.shape[0]
                if pad > 0:
                    x = np.pad(x, ((pad, 0), (0, 0)), mode="constant")
                    t = np.pad(t, (pad, 0), mode="edge")
                    mask = np.pad(mask, (pad, 0), mode="constant", constant_values=False)
                self.samples.append((x, t, mask, float(targets[end])))
                self.metadata.append(
                    {
                        "run_id": group_name,
                        "segment_id": int(segment_ids[end]),
                        time_column: float(times[end]),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x, t, mask, y = self.samples[idx]
        return {
            "x": torch.from_numpy(x),
            "times_s": torch.from_numpy(t),
            "mask": torch.from_numpy(mask),
            "target": torch.tensor(y, dtype=torch.float32),
        }


class PredictionWindowDataset(Dataset):
    """Causal windows for every row in an unlabeled or labeled feature table."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        group_column: str,
        time_column: str,
        context_length: int,
        standardizer: Standardizer,
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []
        for group_name, group in frame.sort_values([group_column, time_column]).groupby(group_column):
            features = group[feature_columns].to_numpy(dtype=np.float32)
            features = standardizer.transform(features).astype(np.float32)
            times = group[time_column].to_numpy(dtype=np.float32)
            segment_ids = group["segment_id"].to_numpy() if "segment_id" in group else np.arange(len(group))
            for end in range(len(group)):
                start = max(0, end + 1 - context_length)
                x = features[start : end + 1]
                t = times[start : end + 1]
                mask = np.ones(x.shape[0], dtype=bool)
                pad = context_length - x.shape[0]
                if pad > 0:
                    x = np.pad(x, ((pad, 0), (0, 0)), mode="constant")
                    t = np.pad(t, (pad, 0), mode="edge")
                    mask = np.pad(mask, (pad, 0), mode="constant", constant_values=False)
                meta = {"run_id": group_name, "segment_id": segment_ids[end]}
                self.samples.append((x, t, mask, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        x, t, mask, meta = self.samples[idx]
        return {
            "x": torch.from_numpy(x),
            "times_s": torch.from_numpy(t),
            "mask": torch.from_numpy(mask),
            "meta": meta,
        }


def prediction_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),  # type: ignore[arg-type]
        "times_s": torch.stack([item["times_s"] for item in batch]),  # type: ignore[arg-type]
        "mask": torch.stack([item["mask"] for item in batch]),  # type: ignore[arg-type]
        "meta": [item["meta"] for item in batch],
    }


def load_feature_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def infer_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    cols = []
    for col in df.columns:
        if col == target_column or col in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns were found")
    return cols


def transform_rul_target(rul_s: float | np.ndarray, transform: str) -> float | np.ndarray:
    """Map physical RUL seconds to the model's output space."""

    values = np.asarray(rul_s, dtype=np.float64)
    if transform == "identity":
        out = values
    elif transform == "log1p":
        out = np.log1p(np.maximum(values, 0.0))
    else:
        raise ValueError(f"Unknown target_transform {transform!r}")
    if np.isscalar(rul_s):
        return float(out)
    return out


def inverse_transform_rul_target(
    prediction: torch.Tensor,
    transform: str,
    *,
    prediction_scale: float = 1.0,
) -> torch.Tensor:
    """Map model output back to physical RUL seconds."""

    if transform == "identity":
        rul = prediction
    elif transform == "log1p":
        rul = torch.expm1(prediction.clamp_max(20.0))
    else:
        raise ValueError(f"Unknown target_transform {transform!r}")
    return rul.clamp_min(1e-6) * float(prediction_scale)


def prediction_to_rul(
    prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    transform: str,
    *,
    prediction_scale: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Convert model output to RUL seconds while preserving uncertainty."""

    if isinstance(prediction, tuple):
        pred, log_var = prediction
        rul = inverse_transform_rul_target(pred, transform, prediction_scale=prediction_scale)
        return rul, log_var
    return inverse_transform_rul_target(prediction, transform, prediction_scale=prediction_scale)


def split_by_run(
    df: pd.DataFrame,
    *,
    group_column: str,
    val_runs: tuple[str, ...],
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = list(df[group_column].drop_duplicates())
    if val_runs:
        val_set = set(val_runs)
    else:
        n_val = max(1, int(round(len(runs) * val_fraction)))
        val_set = set(runs[-n_val:])
    train = df[~df[group_column].isin(val_set)].copy()
    val = df[df[group_column].isin(val_set)].copy()
    if train.empty or val.empty:
        raise ValueError("Train/validation split is empty; adjust val_runs or val_fraction")
    return train, val


def split_per_run_tail(
    df: pd.DataFrame,
    *,
    group_column: str,
    time_column: str,
    val_fraction: float,
    min_train_rows: int,
    min_val_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically reserve the tail of every run for validation."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1) for per_run_tail split")
    train_parts = []
    val_parts = []
    for run_id, group in df.sort_values([group_column, time_column]).groupby(group_column):
        n = len(group)
        n_val = max(min_val_rows, int(round(n * val_fraction)))
        n_val = min(n_val, n - min_train_rows)
        if n_val <= 0:
            raise ValueError(
                f"Run {run_id!r} has {n} rows, not enough for "
                f"{min_train_rows} train and {min_val_rows} validation rows"
            )
        split_idx = n - n_val
        train_parts.append(group.iloc[:split_idx].copy())
        val_parts.append(group.iloc[split_idx:].copy())
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    return train, val


def split_train_validation(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch train/validation splitting strategy."""

    if config.split_mode == "run":
        return split_by_run(
            df,
            group_column=config.group_column,
            val_runs=config.val_runs,
            val_fraction=config.val_fraction,
        )
    if config.split_mode == "per_run_tail":
        if config.val_runs:
            raise ValueError("val_runs is only supported with split_mode='run'")
        return split_per_run_tail(
            df,
            group_column=config.group_column,
            time_column=config.time_column,
            val_fraction=config.val_fraction,
            min_train_rows=max(config.context_length, config.min_context),
            min_val_rows=config.min_context,
        )
    raise ValueError(f"Unknown split_mode {config.split_mode!r}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN reproducible when a GPU is used; these are no-ops on CPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class RULTrainer:
    """Owns model training, score validation, and checkpointing."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> dict[str, float]:
        seed_everything(self.config.seed)
        if self.config.prediction_scale <= 0.0:
            raise ValueError("prediction_scale must be positive")
        transform_rul_target(1.0, self.config.target_transform)
        df = load_feature_table(self.config.features_path)
        feature_columns = infer_feature_columns(df, self.config.target_column)
        train_df, val_df = split_train_validation(df, self.config)

        standardizer = Standardizer().fit(train_df[feature_columns].to_numpy(dtype=np.float32))
        train_ds = WindowedRULDataset(
            train_df,
            feature_columns=feature_columns,
            group_column=self.config.group_column,
            time_column=self.config.time_column,
            target_column=self.config.target_column,
            context_length=self.config.context_length,
            min_context=self.config.min_context,
            standardizer=standardizer,
        )
        val_ds = WindowedRULDataset(
            val_df,
            feature_columns=feature_columns,
            group_column=self.config.group_column,
            time_column=self.config.time_column,
            target_column=self.config.target_column,
            context_length=self.config.context_length,
            min_context=self.config.min_context,
            standardizer=standardizer,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        model_config = ModelConfig(
            feature_dim=len(feature_columns),
            architecture=self.config.model_architecture,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            max_context=self.config.context_length,
        )
        model = build_model(model_config).to(self.device)
        initial_physical_target = float(train_df[self.config.target_column].median())
        initial_target = transform_rul_target(
            initial_physical_target / max(self.config.prediction_scale, 1e-6),
            self.config.target_transform,
        )
        initialize_rul_head(model, float(initial_target))
        criterion = AsymmetricRULLoss(
            conservative_er_target=self.config.conservative_er_target,
            conservative_weight=self.config.conservative_weight,
            score_weight=self.config.score_loss_weight,
            relative_huber_weight=self.config.relative_huber_weight,
            over_prediction_weight=self.config.over_prediction_weight,
            uncertainty_weight=self.config.uncertainty_weight,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.config.epochs, 1),
        )

        best_score = -float("inf")
        best_epoch = -1
        best_prediction_scale = self.config.prediction_scale
        bad_epochs = 0
        history = []
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            val_metrics = self.evaluate(model, val_loader)
            row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            history.append(row)
            pd.DataFrame(history).to_csv(self.output_dir / "history.csv", index=False)

            if val_metrics["score"] > best_score:
                best_score = val_metrics["score"]
                best_epoch = epoch
                best_prediction_scale = float(
                    val_metrics.get("prediction_scale", self.config.prediction_scale)
                )
                bad_epochs = 0
                self._save_checkpoint(
                    model,
                    model_config,
                    standardizer,
                    feature_columns,
                    best_score=best_score,
                    epoch=epoch,
                    best_prediction_scale=best_prediction_scale,
                )
            else:
                bad_epochs += 1

            print(
                f"epoch={epoch:03d} loss={train_loss:.5f} "
                f"val_score={val_metrics['score']:.5f} "
                f"val_mae={val_metrics['mae']:.1f}s bad={bad_epochs}"
            )
            if bad_epochs >= self.config.patience:
                break

        summary = {
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_prediction_scale": best_prediction_scale,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: AsymmetricRULLoss,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        losses = []
        progress = tqdm(loader, desc="train", leave=False)
        for batch in progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["x"], batch["times_s"], batch["mask"])
            pred = prediction_to_rul(
                pred,
                self.config.target_transform,
                prediction_scale=self.config.prediction_scale,
            )
            loss = criterion(pred, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=np.mean(losses[-20:]))
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
    ) -> dict[str, float]:
        model.eval()
        preds = []
        targets = []
        metadata_frame = None
        dataset_metadata = getattr(loader.dataset, "metadata", None)
        if dataset_metadata is not None:
            metadata_frame = pd.DataFrame(dataset_metadata)
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = model(batch["x"], batch["times_s"], batch["mask"])
            prediction_scale = 1.0 if self.config.optimize_val_scale else self.config.prediction_scale
            out = prediction_to_rul(
                out,
                self.config.target_transform,
                prediction_scale=prediction_scale,
            )
            pred = out[0] if isinstance(out, tuple) else out
            preds.append(pred.detach().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
        pred_arr = np.concatenate(preds).reshape(-1)
        target_arr = np.concatenate(targets).reshape(-1)
        if self.config.optimize_val_scale:
            return scale_sweep_metrics(
                target_arr,
                pred_arr,
                scale_min=self.config.val_scale_min,
                scale_max=self.config.val_scale_max,
                scale_steps=self.config.val_scale_steps,
                metadata_frame=metadata_frame,
                group_column=self.config.group_column,
                time_column=self.config.time_column,
                temporal_postprocess=self.config.temporal_postprocess,
                temporal_decay_slack_s=self.config.temporal_decay_slack_s,
                temporal_eol_quantile=self.config.temporal_eol_quantile,
                temporal_postprocess_blend=self.config.temporal_postprocess_blend,
                temporal_postprocess_floor_s=self.config.temporal_postprocess_floor_s,
            )
        pred_arr = apply_temporal_postprocess_to_array(
            pred_arr,
            metadata_frame=metadata_frame,
            method=self.config.temporal_postprocess,
            group_column=self.config.group_column,
            time_column=self.config.time_column,
            decay_slack_s=self.config.temporal_decay_slack_s,
            eol_quantile=self.config.temporal_eol_quantile,
            blend=self.config.temporal_postprocess_blend,
            floor_s=self.config.temporal_postprocess_floor_s,
        )
        scores = official_score_numpy(target_arr, pred_arr)
        return {
            "score": float(np.mean(scores)),
            "mae": float(np.mean(np.abs(pred_arr - target_arr))),
            "rmse": float(np.sqrt(np.mean((pred_arr - target_arr) ** 2))),
            "mean_er": float(np.mean(100.0 * (target_arr - pred_arr) / np.maximum(target_arr, 1e-6))),
            "prediction_scale": float(self.config.prediction_scale),
        }

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        standardizer: Standardizer,
        feature_columns: list[str],
        *,
        best_score: float,
        epoch: int,
        best_prediction_scale: float,
    ) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "training_config": _jsonable(asdict(self.config)),
            "standardizer": standardizer.to_dict(),
            "feature_columns": feature_columns,
            "best_score": best_score,
            "epoch": epoch,
            "best_prediction_scale": best_prediction_scale,
        }
        torch.save(checkpoint, self.output_dir / "best_model.pt")


def scale_sweep_metrics(
    target_arr: np.ndarray,
    pred_arr: np.ndarray,
    *,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
    metadata_frame: pd.DataFrame | None = None,
    group_column: str = "run_id",
    time_column: str = "mid_time_s",
    temporal_postprocess: str = "none",
    temporal_decay_slack_s: float = 0.0,
    temporal_eol_quantile: float = 0.5,
    temporal_postprocess_blend: float = 1.0,
    temporal_postprocess_floor_s: float = 1.0,
) -> dict[str, float]:
    """Return validation metrics at the best multiplicative prediction scale.

    Note: this selects a scale on the data it is given. Only ever call it on an
    inner validation split, never on the set whose score you intend to report.
    """

    if scale_min <= 0.0 or scale_max <= 0.0:
        raise ValueError("validation scale bounds must be positive")
    if scale_max < scale_min:
        raise ValueError("val_scale_max must be >= val_scale_min")
    if scale_steps <= 0:
        raise ValueError("val_scale_steps must be positive")

    target_arr = np.asarray(target_arr, dtype=float)
    pred_arr = np.asarray(pred_arr, dtype=float)
    scales = np.linspace(scale_min, scale_max, scale_steps)
    best: dict[str, float] | None = None
    for scale in scales:
        scaled = pred_arr * float(scale)
        scaled = apply_temporal_postprocess_to_array(
            scaled,
            metadata_frame=metadata_frame,
            method=temporal_postprocess,
            group_column=group_column,
            time_column=time_column,
            decay_slack_s=temporal_decay_slack_s,
            eol_quantile=temporal_eol_quantile,
            blend=temporal_postprocess_blend,
            floor_s=temporal_postprocess_floor_s,
        )
        scores = official_score_numpy(target_arr, scaled)
        metrics = {
            "score": float(np.mean(scores)),
            "mae": float(np.mean(np.abs(scaled - target_arr))),
            "rmse": float(np.sqrt(np.mean((scaled - target_arr) ** 2))),
            "mean_er": float(
                np.mean(100.0 * (target_arr - scaled) / np.maximum(target_arr, 1e-6))
            ),
            "prediction_scale": float(scale),
        }
        if best is None or metrics["score"] > best["score"]:
            best = metrics
    if best is None:
        raise RuntimeError("validation scale sweep produced no candidates")
    return best


def _jsonable(obj: object) -> object:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def initialize_rul_head(model: torch.nn.Module, initial_rul_s: float) -> None:
    """Initialize the positive RUL head near the label scale.

    The challenge labels are tens of thousands of seconds. A Softplus head
    initialized around zero predicts roughly one second, which lands on the flat
    extreme-underprediction side of the score. Setting the final bias to a
    robust train-set RUL scale makes the first epoch meaningful.
    """

    if not np.isfinite(initial_rul_s) or initial_rul_s <= 0.0:
        return
    head = getattr(model, "head", None)
    if not isinstance(head, torch.nn.Sequential):
        return
    final = head[-1]
    if not isinstance(final, torch.nn.Linear):
        return
    with torch.no_grad():
        final.bias.fill_(float(initial_rul_s))


def apply_temporal_postprocess(
    frame: pd.DataFrame,
    *,
    method: str,
    group_column: str,
    time_column: str,
    prediction_column: str = "predicted_rul_s",
    decay_slack_s: float = 0.0,
    eol_quantile: float = 0.5,
    blend: float = 1.0,
    floor_s: float = 1.0,
) -> pd.Series:
    """Apply causal temporal consistency to row-level predictions.

    This is the single source of truth for the causal smoothers. ``eol_quantile``
    tracks a causal running quantile of the *predicted* end-of-life
    (``time + predicted_RUL``) and re-derives RUL from it; ``decay`` enforces a
    monotone, at-least-1s/s decline. Both use only information up to the current
    time step, so they are safe for online use.
    """

    method = str(method or "none")
    if method == "none":
        return pd.Series(frame[prediction_column].to_numpy(dtype=float), index=frame.index, dtype=float)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("temporal_postprocess_blend must be in [0, 1]")
    if not 0.0 <= eol_quantile <= 1.0:
        raise ValueError("temporal_eol_quantile must be in [0, 1]")

    values = pd.Series(index=frame.index, dtype=float)
    ordered = frame.sort_values([group_column, time_column])
    for _name, group in ordered.groupby(group_column, sort=False):
        raw = group[prediction_column].to_numpy(dtype=float)
        times = group[time_column].to_numpy(dtype=float)
        corrected: list[float] = []
        if method == "decay":
            current = float(raw[0])
            for idx, pred in enumerate(raw):
                if idx == 0:
                    current = float(pred)
                else:
                    dt = max(float(times[idx] - times[idx - 1]), 0.0)
                    current = min(float(pred), current - dt + float(decay_slack_s))
                corrected.append(max(float(floor_s), blend * current + (1.0 - blend) * float(pred)))
        elif method == "eol_quantile":
            seen: list[float] = []
            for time_s, pred in zip(times, raw, strict=True):
                seen.append(float(time_s + pred))
                causal_pred = max(float(floor_s), float(np.quantile(seen, eol_quantile)) - float(time_s))
                corrected.append(max(float(floor_s), blend * causal_pred + (1.0 - blend) * float(pred)))
        else:
            raise ValueError(f"Unsupported temporal_postprocess '{method}'")
        values.loc[group.index] = corrected
    return values


def apply_temporal_postprocess_to_array(
    pred_arr: np.ndarray,
    *,
    metadata_frame: pd.DataFrame | None,
    method: str,
    group_column: str,
    time_column: str,
    decay_slack_s: float,
    eol_quantile: float,
    blend: float,
    floor_s: float,
) -> np.ndarray:
    """Apply temporal postprocessing to a prediction array using row metadata."""

    pred_arr = np.asarray(pred_arr, dtype=float).reshape(-1)
    if method == "none":
        return pred_arr
    if metadata_frame is None:
        raise ValueError("temporal postprocessing needs metadata_frame")
    work = metadata_frame.copy()
    work["predicted_rul_s"] = pred_arr
    corrected = apply_temporal_postprocess(
        work,
        method=method,
        group_column=group_column,
        time_column=time_column,
        prediction_column="predicted_rul_s",
        decay_slack_s=decay_slack_s,
        eol_quantile=eol_quantile,
        blend=blend,
        floor_s=floor_s,
    )
    return corrected.to_numpy(dtype=float)


def select_temporal_smoothing(
    frame: pd.DataFrame,
    candidates: list[tuple[str, float | None, float | None]],
    *,
    group_column: str,
    time_column: str,
    target_column: str,
    prediction_column: str = "predicted_rul_s",
    floor_s: float = 1.0,
) -> tuple[dict[str, object], float]:
    """Pick the smoothing candidate that maximizes the official score on ``frame``.

    ``candidates`` is a list of ``(method, quantile, blend)`` tuples; ``method``
    may be ``"none"`` (no smoothing at all). This is meant for **leak-free**
    selection on an inner split — never call it on the set whose score you intend
    to report. Returns ``(choice, score)`` where ``choice`` is a dict with keys
    ``method``/``quantile``/``blend``.
    """

    if not candidates:
        raise ValueError("no smoothing candidates provided")
    actual = frame[target_column].to_numpy(dtype=float)
    best: tuple[dict[str, object], float] | None = None
    for method, quantile, blend in candidates:
        if method == "none":
            pred = frame[prediction_column].to_numpy(dtype=float)
        else:
            pred = apply_temporal_postprocess(
                frame,
                method=method,
                group_column=group_column,
                time_column=time_column,
                prediction_column=prediction_column,
                eol_quantile=float(quantile),
                blend=float(blend),
                floor_s=floor_s,
            ).to_numpy(dtype=float)
        score = float(np.mean(official_score_numpy(actual, pred)))
        if best is None or score > best[1]:
            best = ({"method": method, "quantile": quantile, "blend": blend}, score)
    assert best is not None
    return best


@torch.no_grad()
def export_predictions(
    *,
    checkpoint_path: str | Path,
    features_path: str | Path,
    output_csv: str | Path,
    batch_size: int = 128,
    device: str = "auto",
    prediction_scale: float | None = None,
) -> pd.DataFrame:
    """Load a trained checkpoint and export row-level RUL predictions."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = model_config_from_dict(checkpoint["model_config"])
    model = build_model(model_config)
    load_model_state(model, checkpoint["model_state_dict"])
    resolved = resolve_device(device)
    model.to(resolved)
    model.eval()

    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    feature_columns = list(checkpoint["feature_columns"])
    training_config = checkpoint.get("training_config", {})
    group_column = training_config.get("group_column", "run_id")
    time_column = training_config.get("time_column", "mid_time_s")
    context_length = int(training_config.get("context_length", model_config.max_context))
    target_transform = training_config.get("target_transform", "identity")
    resolved_prediction_scale = float(
        checkpoint.get("best_prediction_scale", training_config.get("prediction_scale", 1.0))
        if prediction_scale is None
        else prediction_scale
    )
    temporal_postprocess = str(training_config.get("temporal_postprocess", "none"))
    temporal_decay_slack_s = float(training_config.get("temporal_decay_slack_s", 0.0))
    temporal_eol_quantile = float(training_config.get("temporal_eol_quantile", 0.5))
    temporal_postprocess_blend = float(training_config.get("temporal_postprocess_blend", 1.0))
    temporal_postprocess_floor_s = float(training_config.get("temporal_postprocess_floor_s", 1.0))

    frame = load_feature_table(features_path)
    dataset = PredictionWindowDataset(
        frame,
        feature_columns=feature_columns,
        group_column=group_column,
        time_column=time_column,
        context_length=context_length,
        standardizer=standardizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=prediction_collate,
    )
    rows = []
    for batch in loader:
        x = batch["x"].to(resolved)
        times = batch["times_s"].to(resolved)
        mask = batch["mask"].to(resolved)
        out = model(x, times, mask)
        out = prediction_to_rul(
            out,
            target_transform,
            prediction_scale=resolved_prediction_scale,
        )
        pred = out[0] if isinstance(out, tuple) else out
        for meta, value in zip(batch["meta"], pred.detach().cpu().numpy(), strict=False):
            row = dict(meta)
            row["predicted_rul_s"] = float(value)
            rows.append(row)
    result = pd.DataFrame(rows)
    if temporal_postprocess != "none":
        result["raw_predicted_rul_s"] = result["predicted_rul_s"]
        result["predicted_rul_s"] = apply_temporal_postprocess(
            result,
            method=temporal_postprocess,
            group_column=group_column,
            time_column=time_column,
            prediction_column="predicted_rul_s",
            decay_slack_s=temporal_decay_slack_s,
            eol_quantile=temporal_eol_quantile,
            blend=temporal_postprocess_blend,
            floor_s=temporal_postprocess_floor_s,
        )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
