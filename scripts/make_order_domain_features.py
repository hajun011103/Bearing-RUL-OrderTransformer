#!/usr/bin/env python
"""Create domain-specific feature tables from extracted PHM features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.training import NON_FEATURE_COLUMNS, load_feature_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/features/train_full.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/features/train_full_order_domain.parquet"),
    )
    parser.add_argument(
        "--drop-operation",
        action="store_true",
        help="Drop operation covariates too, leaving only order-domain vibration features plus metadata.",
    )
    parser.add_argument(
        "--drop-temperature",
        action="store_true",
        help="Drop only front/rear temperature operation covariates while keeping torque and RPM.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("order", "envelope", "time"),
        default="order",
        help="Use order, envelope-order, or time-domain vibration features.",
    )
    parser.add_argument(
        "--include-time-stats",
        action="store_true",
        help=(
            "Also keep impulsiveness time-domain stats such as kurtosis and crest factor. "
            "Useful for order/envelope tables where these columns would otherwise be dropped."
        ),
    )
    parser.add_argument(
        "--include-dynamics",
        action="store_true",
        help="Also keep causal DMD/SINDy degradation-dynamics features.",
    )
    parser.add_argument(
        "--stat-terms",
        nargs="*",
        default=("kurtosis", "crest_factor"),
        help="Time-domain stat name fragments to keep when --include-time-stats is set.",
    )
    return parser.parse_args()


def keep_column(
    name: str,
    *,
    drop_operation: bool,
    drop_temperature: bool,
    feature_mode: str,
    include_time_stats: bool,
    include_dynamics: bool,
    stat_terms: tuple[str, ...],
) -> bool:
    if name in NON_FEATURE_COLUMNS or name == "target":
        return True
    if name.startswith(("cond_", "condres_", "condbase_", "condratio_", "eq_age_")):
        return True
    if "_sk_band_" in name or name.endswith("_sk_score"):
        return True
    if feature_mode == "order" and "_order" in name:
        return True
    if feature_mode == "envelope" and "_env_order" in name:
        return True
    if feature_mode == "time" and "_order" not in name:
        return True
    if include_time_stats and "_order" not in name and any(term in name for term in stat_terms):
        return True
    if include_dynamics and (name.startswith("dmd_") or name.startswith("sindy_")):
        return True
    if drop_temperature and name.startswith(("op_", "opwin_")) and "temp" in name:
        return False
    if not drop_operation and (name.startswith("op_") or name.startswith("opwin_")):
        return True
    return False


def main() -> None:
    args = parse_args()
    df = load_feature_table(args.source)
    kept = [
        col
        for col in df.columns
        if keep_column(
            col,
            drop_operation=args.drop_operation,
            drop_temperature=args.drop_temperature,
            feature_mode=args.feature_mode,
            include_time_stats=args.include_time_stats,
            include_dynamics=args.include_dynamics,
            stat_terms=tuple(args.stat_terms),
        )
    ]
    out = df.loc[:, kept].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        out.to_csv(args.output, index=False)
    else:
        out.to_parquet(args.output, index=False)

    order_cols = [col for col in out.columns if "_order" in col]
    time_cols = [
        col
        for col in out.columns
        if col not in NON_FEATURE_COLUMNS
        and col != "target"
        and not col.startswith("op_")
        and not col.startswith("opwin_")
        and "_order" not in col
    ]
    operation_cols = [col for col in out.columns if col.startswith("op_") or col.startswith("opwin_")]
    temperature_cols = [col for col in out.columns if col.startswith(("op_", "opwin_")) and "temp" in col]
    stat_cols = [
        col
        for col in out.columns
        if "_order" not in col and any(term in col for term in args.stat_terms)
    ]
    dynamics_cols = [
        col
        for col in out.columns
        if col.startswith("dmd_") or col.startswith("sindy_")
    ]
    print(
        f"wrote {len(out)} rows x {len(out.columns)} columns to {args.output}\n"
        f"kept {len(order_cols)} order-domain columns, {len(time_cols)} time-domain columns, "
        f"{len(stat_cols)} selected time-stat columns, {len(dynamics_cols)} dynamics columns, "
        f"{len(operation_cols)} operation columns ({len(temperature_cols)} temperature); "
        f"dropped {len(df.columns) - len(out.columns)} other columns"
    )


if __name__ == "__main__":
    main()
