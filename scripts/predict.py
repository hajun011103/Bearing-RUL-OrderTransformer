#!/usr/bin/env python
"""Export RUL predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.training import export_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prediction-scale", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_predictions(
        checkpoint_path=args.checkpoint,
        features_path=args.features,
        output_csv=args.output,
        batch_size=args.batch_size,
        device=args.device,
        prediction_scale=args.prediction_scale,
    )
    print(f"wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
