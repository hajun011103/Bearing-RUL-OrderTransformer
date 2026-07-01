#!/usr/bin/env python
"""Leak-free late-life ("tail") validation of the order-domain Transformer.

This is a secondary diagnostic to the leave-one-bearing-out headline
([`scripts/run_lobo.py`](run_lobo.py)). It asks an easier question: given the
early life of a bearing, how well is its *late* life predicted?

To keep it honest, each bearing is split chronologically into three parts:

* inner-train  (earliest ~60%)  - model gradient updates,
* inner-val    (next ~15%)      - early stopping AND selection of the prediction
                                  scale and EOL-smoothing hyper-parameters,
* test-tail    (last 25%)       - held out; only scored, never tuned on.

The earlier version of this experiment selected the prediction scale on the
test tail itself, which tunes on the reported set. Here the scale and smoothing
are frozen from the inner split before the test tail is touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.config import TrainingConfig
from phm_pipeline.losses import official_score_numpy
from phm_pipeline.model import build_model, load_model_state, model_config_from_dict
from phm_pipeline.training import (
    RULTrainer,
    Standardizer,
    apply_temporal_postprocess,
    load_feature_table,
    prediction_to_rul,
    resolve_device,
)

SMOOTHING_GRID: list[tuple[str, float | None, float | None]] = [("none", None, None)]
for _q in (0.5, 0.75, 0.9, 1.0):
    for _b in (0.5, 0.75, 1.0):
        SMOOTHING_GRID.append(("eol_quantile", _q, _b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "artifacts/features/train_full_order_domain.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/runs/per_run_tail_nested")
    parser.add_argument("--group-column", default="run_id")
    parser.add_argument("--time-column", default="mid_time_s")
    parser.add_argument("--target-column", default="rul_s")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--inner-val-fraction", type=float, default=0.20)
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--min-context", type=int, default=4)
    parser.add_argument("--target-transform", choices=("identity", "log1p"), default="log1p")
    parser.add_argument("--scale-min", type=float, default=0.05)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--scale-steps", type=int, default=116)
    parser.add_argument("--floor-s", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def chronological_head(df: pd.DataFrame, args: argparse.Namespace, keep_fraction: float) -> pd.DataFrame:
    """Keep the earliest ``keep_fraction`` of every bearing (chronologically)."""

    parts = []
    for _run, group in df.sort_values([args.group_column, args.time_column]).groupby(args.group_column):
        n = len(group)
        n_keep = max(args.context_length, int(round(n * keep_fraction)))
        n_keep = min(n_keep, n)
        parts.append(group.iloc[:n_keep])
    return pd.concat(parts, ignore_index=True)


def chronological_tail(df: pd.DataFrame, args: argparse.Namespace, tail_fraction: float) -> pd.DataFrame:
    """Keep the latest ``tail_fraction`` of every bearing (chronologically)."""

    parts = []
    for _run, group in df.sort_values([args.group_column, args.time_column]).groupby(args.group_column):
        n = len(group)
        n_tail = max(args.min_context, int(round(n * tail_fraction)))
        n_tail = min(n_tail, n)
        parts.append(group.iloc[n - n_tail :])
    return pd.concat(parts, ignore_index=True)


@torch.no_grad()
def predict_rows(checkpoint_path: Path, features: pd.DataFrame, rows: pd.DataFrame, args, device) -> pd.DataFrame:
    """Predict causal RUL for the given ``rows`` using history from ``features``.

    Context for a row is the preceding segments of the same bearing in the full
    ``features`` table (early life is legitimately observable at deployment).
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(model_config_from_dict(checkpoint["model_config"]))
    load_model_state(model, checkpoint["model_state_dict"])
    model.to(device).eval()
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    feature_columns = list(checkpoint["feature_columns"])
    tc = dict(checkpoint.get("training_config", {}))
    context_length = int(tc.get("context_length", args.context_length))
    target_transform = str(tc.get("target_transform", args.target_transform))
    prediction_scale = float(checkpoint.get("best_prediction_scale", tc.get("prediction_scale", 1.0)))

    out_rows = []
    for bearing, target_group in rows.groupby(args.group_column):
        full = features[features[args.group_column] == bearing].sort_values(args.time_column).reset_index(drop=True)
        feats = standardizer.transform(full[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
        times = full[args.time_column].to_numpy(dtype=np.float32)
        target_segments = set(target_group["segment_id"].tolist())
        windows_x, windows_t, windows_mask, keep_idx = [], [], [], []
        for end in range(len(full)):
            if int(full.iloc[end]["segment_id"]) not in target_segments:
                continue
            start = max(0, end + 1 - context_length)
            x = feats[start : end + 1]
            t = times[start : end + 1]
            mask = np.ones(x.shape[0], dtype=bool)
            pad = context_length - x.shape[0]
            if pad > 0:
                x = np.pad(x, ((pad, 0), (0, 0)), mode="constant")
                t = np.pad(t, (pad, 0), mode="edge")
                mask = np.pad(mask, (pad, 0), mode="constant", constant_values=False)
            windows_x.append(x)
            windows_t.append(t)
            windows_mask.append(mask)
            keep_idx.append(end)
        if not windows_x:
            continue
        pred = model(
            torch.from_numpy(np.stack(windows_x)).to(device),
            torch.from_numpy(np.stack(windows_t)).to(device),
            torch.from_numpy(np.stack(windows_mask)).to(device),
        )
        pred = prediction_to_rul(pred, target_transform, prediction_scale=prediction_scale)
        pred = (pred[0] if isinstance(pred, tuple) else pred).detach().cpu().numpy().reshape(-1)
        block = full.iloc[keep_idx][
            [args.group_column, "segment_id", args.time_column, args.target_column, "life_fraction"]
        ].copy()
        block["predicted_rul_s"] = pred
        out_rows.append(block)
    return pd.concat(out_rows, ignore_index=True)


def best_scale_and_smoothing(pred: pd.DataFrame, args) -> tuple[float, dict, float]:
    """Grid-search prediction scale x smoothing on the inner validation rows."""

    actual = pred[args.target_column].to_numpy(dtype=float)
    raw = pred["predicted_rul_s"].to_numpy(dtype=float)
    scales = np.linspace(args.scale_min, args.scale_max, args.scale_steps)
    best = None
    for scale in scales:
        scaled = pred.copy()
        scaled["predicted_rul_s"] = raw * float(scale)
        for method, q, b in SMOOTHING_GRID:
            if method == "none":
                out = scaled["predicted_rul_s"].to_numpy(dtype=float)
            else:
                out = apply_temporal_postprocess(
                    scaled,
                    method=method,
                    group_column=args.group_column,
                    time_column=args.time_column,
                    prediction_column="predicted_rul_s",
                    eol_quantile=float(q),
                    blend=float(b),
                    floor_s=args.floor_s,
                ).to_numpy(dtype=float)
            score = float(np.mean(official_score_numpy(actual, out)))
            if best is None or score > best[2]:
                best = (float(scale), {"method": method, "quantile": q, "blend": b}, score)
    assert best is not None
    return best


def summarize(pred: pd.DataFrame, args) -> dict[str, float]:
    actual = pred[args.target_column].to_numpy(dtype=float)
    predicted = pred["predicted_rul_s"].to_numpy(dtype=float)
    er = 100.0 * (actual - predicted) / np.maximum(actual, 1e-6)
    return {
        "rows": int(len(pred)),
        "score": float(np.mean(official_score_numpy(actual, predicted))),
        "mae_s": float(np.mean(np.abs(predicted - actual))),
        "rmse_s": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mean_er_percent": float(np.mean(er)),
        "over_prediction_rate": float(np.mean(predicted > actual)),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = load_feature_table(args.features)

    train_pool = chronological_head(features, args, keep_fraction=1.0 - args.test_fraction)
    test_tail = chronological_tail(features, args, tail_fraction=args.test_fraction)

    run_dir = args.output_dir / "inner_model"
    checkpoint = run_dir / "best_model.pt"
    if not (args.skip_existing and checkpoint.exists()):
        run_dir.mkdir(parents=True, exist_ok=True)
        pool_path = run_dir / "train_pool.parquet"
        train_pool.to_parquet(pool_path, index=False)
        config = TrainingConfig(
            features_path=pool_path,
            output_dir=run_dir,
            split_mode="per_run_tail",
            val_fraction=args.inner_val_fraction,
            target_transform=args.target_transform,
            context_length=args.context_length,
            min_context=args.min_context,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
        )
        RULTrainer(config).fit()

    # Inner validation rows = the tail of the train pool (never the test tail).
    inner_val = chronological_tail(train_pool, args, tail_fraction=args.inner_val_fraction)
    inner_pred = predict_rows(checkpoint, features, inner_val, args, device)
    scale, smoothing, inner_score = best_scale_and_smoothing(inner_pred, args)
    print(f"[tail] selected scale={scale:.3f} smoothing={smoothing} (inner score={inner_score:.4f})", flush=True)

    test_pred = predict_rows(checkpoint, features, test_tail, args, device)
    test_pred["raw_predicted_rul_s"] = test_pred["predicted_rul_s"]
    test_pred["predicted_rul_s"] = test_pred["predicted_rul_s"] * scale
    if smoothing["method"] != "none":
        test_pred["predicted_rul_s"] = apply_temporal_postprocess(
            test_pred,
            method=smoothing["method"],
            group_column=args.group_column,
            time_column=args.time_column,
            prediction_column="predicted_rul_s",
            eol_quantile=float(smoothing["quantile"]),
            blend=float(smoothing["blend"]),
            floor_s=args.floor_s,
        )

    result = {
        "selected_scale": scale,
        "selected_smoothing": smoothing,
        "inner_selection_score": inner_score,
        "honest": summarize(test_pred, args),
    }
    test_pred.to_csv(args.output_dir / "tail_predictions_with_actual.csv", index=False)
    (args.output_dir / "tail_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
