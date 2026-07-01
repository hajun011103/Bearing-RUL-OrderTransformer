#!/usr/bin/env python
"""Leak-free nested leave-one-bearing-out (LOBO) evaluation.

The headline result of this project is a leave-one-bearing-out out-of-fold
(OOF) score. To keep that number honest, every hyper-parameter that touches the
held-out bearing must be chosen *without looking at it*:

* Model selection / early stopping uses a chronological tail split of the
  training bearings only, never the held-out bearing.
* The causal end-of-life (EOL) smoothing hyper-parameters (quantile, blend) are
  selected with an *inner* LOBO over the training bearings, then frozen before
  they are applied to the held-out bearing.

For transparency the script reports three numbers on the same OOF predictions:

* ``raw``     - no temporal smoothing at all.
* ``honest``  - smoothing selected by the inner LOBO (leak-free, headline).
* ``oracle``  - smoothing selected to maximize the pooled OOF score itself.
                This peeks at the test folds and is only an optimistic ceiling.

Because the pre-extracted order-domain feature table is tiny (a few hundred
segment rows), the whole nested procedure retrains a handful of small
Transformers and runs on CPU in a few minutes.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
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


# EOL-smoothing search grid. "none" (raw) is always a candidate so the inner
# selection can decide that smoothing does not help on the training folds.
SMOOTHING_GRID: list[tuple[str, float | None, float | None]] = [("none", None, None)]
for _q in (0.5, 0.75, 0.9, 1.0):
    for _b in (0.5, 0.75, 1.0):
        SMOOTHING_GRID.append(("eol_quantile", _q, _b))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/features/train_full_order_domain.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/runs/lobo_order_domain_nested",
    )
    parser.add_argument("--group-column", default="run_id")
    parser.add_argument("--time-column", default="mid_time_s")
    parser.add_argument("--target-column", default="rul_s")
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--min-context", type=int, default=4)
    parser.add_argument("--target-transform", choices=("identity", "log1p"), default="log1p")
    parser.add_argument("--inner-val-fraction", type=float, default=0.25)
    parser.add_argument("--floor-s", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing best_model.pt if the fold directory already has one.",
    )
    return parser.parse_args()


def train_on_bearings(
    bearings: tuple[str, ...],
    features: pd.DataFrame,
    args: argparse.Namespace,
    run_dir: Path,
) -> Path:
    """Train the order-domain Transformer on ``bearings`` with tail early stopping.

    Early stopping uses a chronological ``per_run_tail`` split of the training
    bearings only. No smoothing is applied during training or validation, so the
    saved checkpoint is chosen purely from raw validation score on data that
    never includes the outer held-out bearing.
    """

    checkpoint = run_dir / "best_model.pt"
    if args.skip_existing and checkpoint.exists():
        return checkpoint

    run_dir.mkdir(parents=True, exist_ok=True)
    subset = features[features[args.group_column].isin(bearings)].copy()
    features_path = run_dir / "features_subset.parquet"
    subset.to_parquet(features_path, index=False)

    config = TrainingConfig(
        features_path=features_path,
        output_dir=run_dir,
        target_column=args.target_column,
        group_column=args.group_column,
        time_column=args.time_column,
        context_length=args.context_length,
        min_context=args.min_context,
        split_mode="per_run_tail",
        val_fraction=args.inner_val_fraction,
        target_transform=args.target_transform,
        prediction_scale=1.0,
        temporal_postprocess="none",
        optimize_val_scale=False,
        model_architecture="transformer",
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )
    RULTrainer(config).fit()
    return checkpoint


@torch.no_grad()
def predict_bearing_raw(
    checkpoint_path: Path,
    features: pd.DataFrame,
    bearing: str,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    """Return raw (un-smoothed) causal RUL predictions for one bearing.

    Every labeled segment (from ``min_context`` onward) is predicted using only
    the preceding segments in the same bearing as context, matching how the
    model would run online.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = model_config_from_dict(checkpoint["model_config"])
    model = build_model(model_config)
    load_model_state(model, checkpoint["model_state_dict"])
    model.to(device).eval()

    standardizer = Standardizer.from_dict(checkpoint["standardizer"])
    feature_columns = list(checkpoint["feature_columns"])
    training_config = dict(checkpoint.get("training_config", {}))
    context_length = int(training_config.get("context_length", args.context_length))
    min_context = int(training_config.get("min_context", args.min_context))
    target_transform = str(training_config.get("target_transform", args.target_transform))
    prediction_scale = float(
        checkpoint.get("best_prediction_scale", training_config.get("prediction_scale", 1.0))
    )

    group = (
        features[features[args.group_column] == bearing]
        .sort_values(args.time_column)
        .reset_index(drop=True)
    )
    feats = standardizer.transform(group[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
    times = group[args.time_column].to_numpy(dtype=np.float32)

    windows_x, windows_t, windows_mask, end_indices = [], [], [], []
    for end in range(max(min_context - 1, 0), len(group)):
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
        end_indices.append(end)

    x_tensor = torch.from_numpy(np.stack(windows_x)).to(device)
    t_tensor = torch.from_numpy(np.stack(windows_t)).to(device)
    mask_tensor = torch.from_numpy(np.stack(windows_mask)).to(device)
    out = model(x_tensor, t_tensor, mask_tensor)
    out = prediction_to_rul(
        out,
        target_transform,
        prediction_scale=prediction_scale,
    )
    pred = (out[0] if isinstance(out, tuple) else out).detach().cpu().numpy().reshape(-1)

    result = group.iloc[end_indices][
        [args.group_column, "segment_id", args.time_column, args.target_column, "life_fraction"]
    ].copy()
    result["predicted_rul_s"] = pred
    return result.reset_index(drop=True)


def apply_smoothing(
    frame: pd.DataFrame,
    method: str,
    quantile: float | None,
    blend: float | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Apply a smoothing candidate and return the resulting RUL array."""

    if method == "none":
        return frame["predicted_rul_s"].to_numpy(dtype=float)
    smoothed = apply_temporal_postprocess(
        frame,
        method=method,
        group_column=args.group_column,
        time_column=args.time_column,
        prediction_column="predicted_rul_s",
        eol_quantile=float(quantile),
        blend=float(blend),
        floor_s=args.floor_s,
    )
    return smoothed.to_numpy(dtype=float)


def score_frame(frame: pd.DataFrame, prediction: np.ndarray, target_column: str) -> float:
    actual = frame[target_column].to_numpy(dtype=float)
    return float(np.mean(official_score_numpy(actual, prediction)))


def select_smoothing(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, float]:
    """Pick the smoothing candidate that maximizes score on ``frame``."""

    best_choice: dict | None = None
    best_score = -np.inf
    for method, quantile, blend in SMOOTHING_GRID:
        prediction = apply_smoothing(frame, method, quantile, blend, args)
        score = score_frame(frame, prediction, args.target_column)
        if score > best_score:
            best_score = score
            best_choice = {"method": method, "quantile": quantile, "blend": blend}
    assert best_choice is not None
    return best_choice, best_score


def summarize(frame: pd.DataFrame, args: argparse.Namespace) -> dict[str, float]:
    actual = frame[args.target_column].to_numpy(dtype=float)
    predicted = frame["predicted_rul_s"].to_numpy(dtype=float)
    er = 100.0 * (actual - predicted) / np.maximum(actual, 1e-6)
    return {
        "rows": int(len(frame)),
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
    bearings = tuple(sorted(features[args.group_column].unique()))
    print(f"[lobo] bearings: {bearings} | device={device}", flush=True)

    honest_parts: list[pd.DataFrame] = []
    raw_parts: list[pd.DataFrame] = []
    fold_records: list[dict] = []

    for held_out in bearings:
        train_pool = tuple(b for b in bearings if b != held_out)
        print(f"\n[lobo] ===== held-out {held_out} | train {train_pool} =====", flush=True)

        # --- inner LOBO over the training bearings to select smoothing ---
        inner_oof_parts: list[pd.DataFrame] = []
        for inner_held in train_pool:
            inner_train = tuple(b for b in train_pool if b != inner_held)
            inner_dir = args.output_dir / f"outer_{held_out}" / f"inner_holdout_{inner_held}"
            print(f"[lobo]   inner: train {inner_train} -> predict {inner_held}", flush=True)
            inner_ckpt = train_on_bearings(inner_train, features, args, inner_dir)
            inner_oof_parts.append(predict_bearing_raw(inner_ckpt, features, inner_held, args, device))
        inner_oof = pd.concat(inner_oof_parts, ignore_index=True)
        choice, inner_score = select_smoothing(inner_oof, args)
        print(
            f"[lobo]   selected smoothing for {held_out}: {choice} "
            f"(inner score={inner_score:.4f})",
            flush=True,
        )

        # --- outer model trained on the full training pool ---
        outer_dir = args.output_dir / f"outer_{held_out}" / "outer_model"
        outer_ckpt = train_on_bearings(train_pool, features, args, outer_dir)
        outer_pred = predict_bearing_raw(outer_ckpt, features, held_out, args, device)

        raw_parts.append(outer_pred.copy())
        honest = outer_pred.copy()
        honest["predicted_rul_s"] = apply_smoothing(
            honest, choice["method"], choice["quantile"], choice["blend"], args
        )
        honest_parts.append(honest)

        fold_records.append(
            {
                "held_out": held_out,
                "train_pool": list(train_pool),
                "selected_smoothing": choice,
                "inner_selection_score": inner_score,
                "raw_fold_score": summarize(outer_pred, args)["score"],
                "honest_fold_score": summarize(honest, args)["score"],
            }
        )

    raw_oof = pd.concat(raw_parts, ignore_index=True)
    honest_oof = pd.concat(honest_parts, ignore_index=True)

    # Oracle: pick a single smoothing that maximizes the pooled OOF score itself.
    # This peeks at the test folds and is reported only as an optimistic ceiling.
    oracle_choice, _ = select_smoothing(raw_oof, args)
    oracle_pred = apply_smoothing(
        raw_oof, oracle_choice["method"], oracle_choice["quantile"], oracle_choice["blend"], args
    )
    oracle_frame = raw_oof.copy()
    oracle_frame["predicted_rul_s"] = oracle_pred

    results = {
        "n_bearings": len(bearings),
        "bearings": list(bearings),
        "raw": summarize(raw_oof, args),
        "honest": summarize(honest_oof, args),
        "oracle": {**summarize(oracle_frame, args), "selected_smoothing": oracle_choice},
        "folds": fold_records,
    }

    raw_oof.to_csv(args.output_dir / "oof_predictions_raw.csv", index=False)
    honest_oof.to_csv(args.output_dir / "oof_predictions_honest.csv", index=False)
    (args.output_dir / "nested_lobo_summary.json").write_text(json.dumps(results, indent=2))

    print("\n[lobo] ===== summary =====")
    print(json.dumps({k: results[k] for k in ("raw", "honest", "oracle")}, indent=2))


if __name__ == "__main__":
    main()
