#!/usr/bin/env python
"""Apply causal temporal consistency to validation prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.losses import official_score_numpy
from phm_pipeline.training import apply_temporal_postprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("decay", "eol_quantile"), default="decay")
    parser.add_argument("--slack-s", type=float, default=0.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--floor-s", type=float, default=1.0)
    parser.add_argument("--group-column", default="run_id")
    parser.add_argument("--time-column", default="mid_time_s")
    parser.add_argument("--target-column", default="rul_s")
    return parser.parse_args()


def summarize(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
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


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    if not 0.0 <= args.quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    frame = pd.read_csv(args.predictions)
    raw_predictions = frame["predicted_rul_s"].copy()

    # Single source of truth for causal smoothing lives in phm_pipeline.training.
    corrected = apply_temporal_postprocess(
        frame,
        method=args.method,
        group_column=args.group_column,
        time_column=args.time_column,
        prediction_column="predicted_rul_s",
        decay_slack_s=args.slack_s,
        eol_quantile=args.quantile,
        blend=args.blend,
        floor_s=args.floor_s,
    )
    if args.method == "decay":
        params = {"slack_s": args.slack_s, "blend": args.blend, "floor_s": args.floor_s}
    else:
        params = {"quantile": args.quantile, "blend": args.blend, "floor_s": args.floor_s}

    frame["raw_predicted_rul_s"] = raw_predictions
    frame["predicted_rul_s"] = corrected
    by_run, overall = summarize(frame, group_column=args.group_column, target_column=args.target_column)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "val_predictions_with_actual.csv", index=False)
    by_run.to_csv(args.output_dir / "val_metrics_by_run.csv", index=False)
    summary = {
        "source_predictions": str(args.predictions),
        "method": args.method,
        "params": params,
        "overall": overall,
    }
    (args.output_dir / "val_metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(by_run.to_string(index=False))


if __name__ == "__main__":
    main()
