#!/usr/bin/env python
"""Extract order-domain feature tables from the PHM Korea training data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.config import BearingGeometry, FeatureConfig
from phm_pipeline.features import extract_all_features, write_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/Train"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/features/train_features.parquet"))
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--channels", type=int, nargs="*", default=None)
    parser.add_argument("--analysis-decimate", type=int, default=2)
    parser.add_argument("--bpfo-order", type=float, default=None)
    parser.add_argument("--bpfi-order", type=float, default=None)
    parser.add_argument("--bsf-order", type=float, default=None)
    parser.add_argument("--ftf-order", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry = BearingGeometry()
    overrides = {
        "bpfo": args.bpfo_order,
        "bpfi": args.bpfi_order,
        "bsf": args.bsf_order,
        "ftf": args.ftf_order,
    }
    for key, value in overrides.items():
        if value is not None:
            geometry.fault_orders[key] = value
    config = FeatureConfig(
        geometry=geometry,
        analysis_decimate=args.analysis_decimate,
    )
    df = extract_all_features(
        args.data_root,
        config,
        limit_segments=args.limit_segments,
        channels=args.channels,
    )
    write_features(df, args.output)
    print(f"wrote {len(df)} rows x {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
