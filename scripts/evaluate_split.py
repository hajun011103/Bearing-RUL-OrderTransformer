#!/usr/bin/env python
"""Evaluate a checkpoint on the same split policy used by training.

This is intentionally separate from ``predict.py``. The prediction exporter
uses all previous rows in a bearing as context, while validation during
training builds windows only inside the validation split. This script mirrors
the training-time validation behavior and writes per-bearing diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.config import TrainingConfig
from phm_pipeline.losses import official_score_numpy
from phm_pipeline.model import build_model, load_model_state, model_config_from_dict
from phm_pipeline.training import (
    Standardizer,
    apply_temporal_postprocess,
    load_feature_table,
    prediction_to_rul,
    resolve_device,
    split_train_validation,
)


class EvaluationWindowDataset(Dataset):
    """Validation windows with row metadata preserved for diagnostics."""

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
        standardizer: Standardizer,
    ) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []
        metadata_columns = [
            col
            for col in (
                group_column,
                "segment_id",
                time_column,
                target_column,
                "rul_segments",
                "life_fraction",
            )
            if col in frame.columns
        ]

        for _group_name, group in frame.sort_values([group_column, time_column]).groupby(group_column):
            features = group[feature_columns].to_numpy(dtype=np.float32)
            features = standardizer.transform(features).astype(np.float32)
            times = group[time_column].to_numpy(dtype=np.float32)

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

                meta = group.iloc[end][metadata_columns].to_dict()
                self.samples.append((x, t, mask, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        x, times, mask, meta = self.samples[idx]
        return {
            "x": torch.from_numpy(x),
            "times_s": torch.from_numpy(times),
            "mask": torch.from_numpy(mask),
            "meta": meta,
        }


def collate_eval(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),  # type: ignore[arg-type]
        "times_s": torch.stack([item["times_s"] for item in batch]),  # type: ignore[arg-type]
        "mask": torch.stack([item["mask"] for item in batch]),  # type: ignore[arg-type]
        "meta": [item["meta"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-mode", choices=("checkpoint", "run", "per_run_tail"), default="checkpoint")
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--val-runs", nargs="*", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--min-context", type=int, default=None)
    parser.add_argument("--prediction-scale", type=float, default=None)
    parser.add_argument("--temporal-postprocess", choices=("checkpoint", "none", "decay", "eol_quantile"), default="checkpoint")
    parser.add_argument("--temporal-decay-slack-s", type=float, default=None)
    parser.add_argument("--temporal-eol-quantile", type=float, default=None)
    parser.add_argument("--temporal-postprocess-blend", type=float, default=None)
    parser.add_argument("--temporal-postprocess-floor-s", type=float, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_eval_config(
    args: argparse.Namespace,
    training_payload: dict[str, object],
    checkpoint: dict[str, object],
) -> TrainingConfig:
    split_mode = (
        str(training_payload.get("split_mode", "run"))
        if args.split_mode == "checkpoint"
        else args.split_mode
    )
    val_runs = args.val_runs
    if val_runs is None:
        val_runs = list(training_payload.get("val_runs", ()))

    temporal_postprocess = (
        str(training_payload.get("temporal_postprocess", "none"))
        if args.temporal_postprocess == "checkpoint"
        else args.temporal_postprocess
    )
    tuned_prediction_scale = float(
        checkpoint.get("best_prediction_scale", training_payload.get("prediction_scale", 1.0))
    )
    return TrainingConfig(
        features_path=args.features,
        output_dir=args.output_dir,
        target_column=str(training_payload.get("target_column", "rul_s")),
        group_column=str(training_payload.get("group_column", "run_id")),
        time_column=str(training_payload.get("time_column", "mid_time_s")),
        context_length=int(args.context_length or training_payload.get("context_length", 48)),
        min_context=int(args.min_context or training_payload.get("min_context", 4)),
        split_mode=split_mode,
        val_runs=tuple(str(run) for run in val_runs),
        val_fraction=float(args.val_fraction or training_payload.get("val_fraction", 0.25)),
        target_transform=str(training_payload.get("target_transform", "identity")),
        prediction_scale=float(args.prediction_scale if args.prediction_scale is not None else tuned_prediction_scale),
        temporal_postprocess=temporal_postprocess,
        temporal_decay_slack_s=float(
            training_payload.get("temporal_decay_slack_s", 0.0)
            if args.temporal_decay_slack_s is None
            else args.temporal_decay_slack_s
        ),
        temporal_eol_quantile=float(
            training_payload.get("temporal_eol_quantile", 0.5)
            if args.temporal_eol_quantile is None
            else args.temporal_eol_quantile
        ),
        temporal_postprocess_blend=float(
            training_payload.get("temporal_postprocess_blend", 1.0)
            if args.temporal_postprocess_blend is None
            else args.temporal_postprocess_blend
        ),
        temporal_postprocess_floor_s=float(
            training_payload.get("temporal_postprocess_floor_s", 1.0)
            if args.temporal_postprocess_floor_s is None
            else args.temporal_postprocess_floor_s
        ),
    )


def summarize(frame: pd.DataFrame, *, group_column: str, target_column: str) -> tuple[pd.DataFrame, dict[str, float]]:
    actual = frame[target_column].to_numpy(dtype=float)
    predicted = frame["predicted_rul_s"].to_numpy(dtype=float)
    frame["er_percent"] = 100.0 * (actual - predicted) / np.maximum(actual, 1e-6)
    frame["score"] = official_score_numpy(actual, predicted)
    frame["abs_error_s"] = np.abs(predicted - actual)
    frame["over_predicted"] = predicted > actual

    by_run = (
        frame.groupby(group_column, sort=True)
        .agg(
            rows=("score", "size"),
            score=("score", "mean"),
            mae_s=("abs_error_s", "mean"),
            rmse_s=("abs_error_s", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mean_er_percent=("er_percent", "mean"),
            over_prediction_rate=("over_predicted", "mean"),
            mean_actual_rul_s=(target_column, "mean"),
            mean_predicted_rul_s=("predicted_rul_s", "mean"),
        )
        .reset_index()
    )
    overall = {
        "rows": int(len(frame)),
        "score": float(frame["score"].mean()),
        "mae_s": float(frame["abs_error_s"].mean()),
        "rmse_s": float(np.sqrt(np.mean(np.square(frame["abs_error_s"])))),
        "mean_er_percent": float(frame["er_percent"].mean()),
        "over_prediction_rate": float(frame["over_predicted"].mean()),
    }
    return by_run, overall


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    training_payload = dict(checkpoint.get("training_config", {}))
    config = build_eval_config(args, training_payload, checkpoint)

    features = load_feature_table(args.features)
    _train_df, val_df = split_train_validation(features, config)
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])  # type: ignore[arg-type]
    feature_columns = list(checkpoint["feature_columns"])

    missing = sorted(set(feature_columns) - set(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing checkpoint columns: {missing[:10]}")

    dataset = EvaluationWindowDataset(
        val_df,
        feature_columns=feature_columns,
        group_column=config.group_column,
        time_column=config.time_column,
        target_column=config.target_column,
        context_length=config.context_length,
        min_context=config.min_context,
        standardizer=standardizer,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval)

    model_config = model_config_from_dict(checkpoint["model_config"])  # type: ignore[arg-type]
    model = build_model(model_config)
    load_model_state(model, checkpoint["model_state_dict"])  # type: ignore[arg-type]
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    rows = []
    for batch in loader:
        x = batch["x"].to(device)
        times = batch["times_s"].to(device)
        mask = batch["mask"].to(device)
        out = model(x, times, mask)
        out = prediction_to_rul(
            out,
            config.target_transform,
            prediction_scale=config.prediction_scale,
        )
        pred = out[0] if isinstance(out, tuple) else out
        for meta, value in zip(batch["meta"], pred.detach().cpu().numpy(), strict=False):
            row = dict(meta)
            row["predicted_rul_s"] = float(np.asarray(value).squeeze())
            rows.append(row)

    result = pd.DataFrame(rows)
    if config.temporal_postprocess != "none":
        result["raw_predicted_rul_s"] = result["predicted_rul_s"]
        result["predicted_rul_s"] = apply_temporal_postprocess(
            result,
            method=config.temporal_postprocess,
            group_column=config.group_column,
            time_column=config.time_column,
            prediction_column="predicted_rul_s",
            decay_slack_s=config.temporal_decay_slack_s,
            eol_quantile=config.temporal_eol_quantile,
            blend=config.temporal_postprocess_blend,
            floor_s=config.temporal_postprocess_floor_s,
        )
    by_run, overall = summarize(result, group_column=config.group_column, target_column=config.target_column)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "val_predictions_with_actual.csv", index=False)
    by_run.to_csv(args.output_dir / "val_metrics_by_run.csv", index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "features": str(args.features),
        "split_mode": config.split_mode,
        "val_fraction": config.val_fraction,
        "val_runs": list(config.val_runs),
        "target_transform": config.target_transform,
        "prediction_scale": config.prediction_scale,
        "temporal_postprocess": config.temporal_postprocess,
        "temporal_decay_slack_s": config.temporal_decay_slack_s,
        "temporal_eol_quantile": config.temporal_eol_quantile,
        "temporal_postprocess_blend": config.temporal_postprocess_blend,
        "temporal_postprocess_floor_s": config.temporal_postprocess_floor_s,
        "context_length": config.context_length,
        "min_context": config.min_context,
        "overall": overall,
    }
    (args.output_dir / "val_metrics_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(by_run.to_string(index=False))


if __name__ == "__main__":
    main()
