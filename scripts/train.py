#!/usr/bin/env python
"""Train the order-domain RUL Transformer with score-based early stopping."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.config import TrainingConfig
from phm_pipeline.training import RULTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/default"))
    parser.add_argument("--context-length", type=int, default=48)
    parser.add_argument("--min-context", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--split-mode", choices=("run", "per_run_tail"), default="run")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--val-runs", nargs="*", default=())
    parser.add_argument("--target-transform", choices=("identity", "log1p"), default="identity")
    parser.add_argument("--prediction-scale", type=float, default=1.0)
    parser.add_argument(
        "--temporal-postprocess",
        choices=("none", "decay", "eol_quantile"),
        default="none",
        help="Apply causal temporal consistency to validation predictions before scoring.",
    )
    parser.add_argument("--temporal-decay-slack-s", type=float, default=0.0)
    parser.add_argument("--temporal-eol-quantile", type=float, default=0.5)
    parser.add_argument("--temporal-postprocess-blend", type=float, default=1.0)
    parser.add_argument("--temporal-postprocess-floor-s", type=float, default=1.0)
    parser.add_argument(
        "--optimize-val-scale",
        action="store_true",
        help=(
            "Sweep a multiplicative prediction scale on the validation split each "
            "epoch. Only ever use this on an inner validation split, never on the "
            "set whose score you report (it tunes on that data)."
        ),
    )
    parser.add_argument("--val-scale-min", type=float, default=0.05)
    parser.add_argument("--val-scale-max", type=float, default=0.80)
    parser.add_argument("--val-scale-steps", type=int, default=151)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conservative-er-target", type=float, default=7.5)
    parser.add_argument("--conservative-weight", type=float, default=0.03)
    parser.add_argument("--score-loss-weight", type=float, default=1.0)
    parser.add_argument("--relative-huber-weight", type=float, default=0.0)
    parser.add_argument("--over-prediction-weight", type=float, default=2.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        features_path=args.features,
        output_dir=args.output_dir,
        context_length=args.context_length,
        min_context=args.min_context,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        val_runs=tuple(args.val_runs),
        target_transform=args.target_transform,
        prediction_scale=args.prediction_scale,
        temporal_postprocess=args.temporal_postprocess,
        temporal_decay_slack_s=args.temporal_decay_slack_s,
        temporal_eol_quantile=args.temporal_eol_quantile,
        temporal_postprocess_blend=args.temporal_postprocess_blend,
        temporal_postprocess_floor_s=args.temporal_postprocess_floor_s,
        optimize_val_scale=args.optimize_val_scale,
        val_scale_min=args.val_scale_min,
        val_scale_max=args.val_scale_max,
        val_scale_steps=args.val_scale_steps,
        device=args.device,
        conservative_er_target=args.conservative_er_target,
        conservative_weight=args.conservative_weight,
        score_loss_weight=args.score_loss_weight,
        relative_huber_weight=args.relative_huber_weight,
        over_prediction_weight=args.over_prediction_weight,
        uncertainty_weight=args.uncertainty_weight,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    summary = RULTrainer(config).fit()
    print(summary)


if __name__ == "__main__":
    main()
