#!/usr/bin/env python
"""Run an external PRONOSTIA/FEMTO order-domain RUL smoke benchmark.

The KSPHM abstract method is an order-domain Transformer with causal
end-of-life smoothing. PRONOSTIA uses CSV vibration snapshots rather than the
KSPHM TDMS archives, so this script builds a compact compatible feature table
directly from the PRONOSTIA ZIP files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.config import FeatureConfig, TrainingConfig
from phm_pipeline.features import (
    aggregate_domain_fault_features,
    order_band_features,
    order_spectrum,
    robust_signal_stats,
    spectral_shape_features,
)
from phm_pipeline.losses import official_score_numpy
from phm_pipeline.training import RULTrainer, apply_temporal_postprocess, export_predictions


CONDITION_BY_PREFIX = {
    # Standard PRONOSTIA operating conditions.
    # Bearing1_x: 1800 rpm, 4000 N; Bearing2_x: 1650 rpm, 4200 N;
    # Bearing3_x: 1500 rpm, 5000 N.
    "Bearing1": {"rpm": 1800.0, "load_n": 4000.0},
    "Bearing2": {"rpm": 1650.0, "load_n": 4200.0},
    "Bearing3": {"rpm": 1500.0, "load_n": 5000.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--femto-root",
        type=Path,
        default=Path("data/external/femto/FEMTOBearingDataSet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/external/pronostia_order_transformer"),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-run", default="Bearing3_2")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def run_name(member: str) -> str:
    parts = member.split("/")
    for part in parts:
        if re.fullmatch(r"Bearing\d+_\d+", part):
            return part
    raise ValueError(f"Could not parse bearing run from {member}")


def segment_id(member: str) -> int:
    match = re.search(r"acc_(\d+)\.csv$", member)
    if not match:
        raise ValueError(f"Could not parse segment id from {member}")
    return int(match.group(1))


def condition_for(run_id: str) -> dict[str, float]:
    for prefix, values in CONDITION_BY_PREFIX.items():
        if run_id.startswith(prefix):
            return values
    raise ValueError(f"Unknown PRONOSTIA condition for {run_id}")


def read_vibration_csv(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member) as handle:
        first = handle.readline()
        delimiter = b";" if b";" in first and b"," not in first else b","
        handle.seek(0)
        data = np.loadtxt(handle, delimiter=delimiter.decode("ascii"), usecols=(4, 5), dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"Unexpected vibration shape for {member}: {data.shape}")
    return data.T


def extract_order_features(vibration: np.ndarray, *, rpm: float, config: FeatureConfig) -> dict[str, float]:
    fault_orders = config.geometry.resolved_fault_orders()
    out: dict[str, float] = {}
    for channel_idx, signal in enumerate(vibration, start=1):
        prefix = f"ch{channel_idx}"
        for key, value in robust_signal_stats(signal).items():
            # Keep a few health stats; the spectral representation remains order-domain.
            if key in {"rms", "kurtosis", "crest_factor", "p2p"}:
                out[f"{prefix}_{key}"] = value
        orders, amplitude = order_spectrum(
            signal,
            rpm,
            sample_rate_hz=config.sample_rate_hz,
            samples_per_revolution=config.samples_per_revolution,
            max_order=config.max_order,
        )
        out.update(spectral_shape_features(orders, amplitude, prefix=f"{prefix}_order"))
        out.update(
            order_band_features(
                orders,
                amplitude,
                fault_orders,
                width=config.order_band_width,
                harmonics=config.order_harmonics,
                prefix=f"{prefix}_order",
            )
        )
    out.update(
        aggregate_domain_fault_features(
            out,
            n_channels=int(vibration.shape[0]),
            fault_orders=fault_orders,
            domain_prefixes=("order",),
        )
    )
    return out


def build_feature_table(
    zip_path: Path,
    *,
    root_name: str,
    split_name: str,
    stride: int,
    config: FeatureConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith(f"{root_name}/")
            and name.lower().endswith(".csv")
            and re.search(r"/acc_\d+\.csv$", name)
        ]
        by_run: dict[str, list[str]] = {}
        for member in members:
            by_run.setdefault(run_name(member), []).append(member)

        for run_id in sorted(by_run):
            run_members = sorted(by_run[run_id], key=segment_id)
            selected = run_members[:: max(int(stride), 1)]
            n_total = len(run_members)
            cond = condition_for(run_id)
            for count, member in enumerate(selected, start=1):
                seg = segment_id(member)
                if count == 1 or count % 250 == 0 or count == len(selected):
                    print(
                        f"[pronostia] {split_name}/{run_id}: {count}/{len(selected)}",
                        flush=True,
                    )
                start_s = float((seg - 1) * 10.0)
                vibration = read_vibration_csv(archive, member)
                feats = extract_order_features(vibration, rpm=cond["rpm"], config=config)
                row: dict[str, float | int | str] = {
                    "run_id": run_id,
                    "split": split_name,
                    "segment_id": seg,
                    "member": member,
                    "start_time_s": start_s,
                    "mid_time_s": start_s + 0.05,
                    "end_time_s": start_s + 0.1,
                    "op_rpm": cond["rpm"],
                    "op_load_n": cond["load_n"],
                    "run_end_s": float((n_total - 1) * 10.0),
                    "rul_s": max(float((n_total - seg) * 10.0), 1.0),
                    "rul_segments": max(n_total - seg, 0),
                    "life_fraction": float((seg - 1) / max(n_total - 1, 1)),
                }
                row.update(feats)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["run_id", "segment_id"]).reset_index(drop=True)


def summarize_predictions(predictions: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = ["run_id", "segment_id"]
    actual_cols = keys + ["mid_time_s", "rul_s", "run_end_s", "life_fraction"]
    merged = predictions.merge(features[actual_cols], on=keys, how="left")
    if merged["rul_s"].isna().any():
        raise ValueError("Some predictions could not be matched to PRONOSTIA labels")
    merged["raw_predicted_rul_s"] = merged["predicted_rul_s"]
    merged["predicted_rul_s"] = apply_temporal_postprocess(
        merged,
        method="eol_quantile",
        group_column="run_id",
        time_column="mid_time_s",
        prediction_column="predicted_rul_s",
        eol_quantile=0.9,
        blend=1.0,
        floor_s=1.0,
    )
    actual = merged["rul_s"].to_numpy(dtype=float)
    pred = merged["predicted_rul_s"].to_numpy(dtype=float)
    merged["score"] = official_score_numpy(actual, pred)
    merged["abs_error_s"] = np.abs(pred - actual)
    merged["normalized_abs_error"] = merged["abs_error_s"] / np.maximum(
        merged["run_end_s"].to_numpy(dtype=float),
        1.0,
    )
    merged["over_predicted"] = pred > actual
    merged["er_percent"] = 100.0 * (actual - pred) / np.maximum(actual, 1e-6)
    by_run = (
        merged.groupby("run_id", sort=True)
        .agg(
            rows=("score", "size"),
            score=("score", "mean"),
            mae_s=("abs_error_s", "mean"),
            normalized_mae=("normalized_abs_error", "mean"),
            over_prediction_rate=("over_predicted", "mean"),
            mean_er_percent=("er_percent", "mean"),
        )
        .reset_index()
    )
    overall = {
        "rows": int(len(merged)),
        "score": float(merged["score"].mean()),
        "mae_s": float(merged["abs_error_s"].mean()),
        "normalized_mae": float(merged["normalized_abs_error"].mean()),
        "over_prediction_rate": float(merged["over_predicted"].mean()),
        "mean_er_percent": float(merged["er_percent"].mean()),
    }
    return merged, {"overall": overall, "by_run": by_run.to_dict(orient="records")}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_features_path = args.output_dir / f"pronostia_learning_stride{args.stride}.parquet"
    validation_features_path = args.output_dir / f"pronostia_full_validation_stride{args.stride}.parquet"

    config = FeatureConfig(
        sample_rate_hz=25_600.0,
        segment_seconds=0.1,
        acquisition_period_seconds=10.0,
        samples_per_revolution=128,
        max_order=40.0,
        order_band_width=0.15,
        order_harmonics=5,
    )

    if args.rebuild_features or not train_features_path.exists():
        train_df = build_feature_table(
            args.femto_root / "Training_set.zip",
            root_name="Learning_set",
            split_name="learning",
            stride=args.stride,
            config=config,
        )
        train_df.to_parquet(train_features_path, index=False)
    else:
        train_df = pd.read_parquet(train_features_path)

    if args.rebuild_features or not validation_features_path.exists():
        validation_df = build_feature_table(
            args.femto_root / "Validation_Set.zip",
            root_name="Full_Test_Set",
            split_name="full_validation",
            stride=args.stride,
            config=config,
        )
        validation_df.to_parquet(validation_features_path, index=False)
    else:
        validation_df = pd.read_parquet(validation_features_path)

    if args.val_run not in set(train_df["run_id"]):
        raise ValueError(f"--val-run must be one of {sorted(train_df['run_id'].unique())}")

    run_dir = args.output_dir / f"model_stride{args.stride}_{args.val_run}"
    trainer_config = TrainingConfig(
        features_path=train_features_path,
        output_dir=run_dir,
        split_mode="run",
        val_runs=(args.val_run,),
        target_transform="log1p",
        optimize_val_scale=True,
        val_scale_min=0.05,
        val_scale_max=1.2,
        val_scale_steps=116,
        temporal_postprocess="eol_quantile",
        temporal_eol_quantile=0.9,
        temporal_postprocess_blend=1.0,
        temporal_postprocess_floor_s=1.0,
        context_length=args.context_length,
        min_context=4,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.10,
        learning_rate=3e-4,
        device=args.device,
        seed=2026,
    )
    if (run_dir / "best_model.pt").exists() and (run_dir / "summary.json").exists():
        train_summary = json.loads((run_dir / "summary.json").read_text())
    else:
        train_summary = RULTrainer(trainer_config).fit()

    checkpoint_for_export = run_dir / "best_model_export_raw.pt"
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    checkpoint.setdefault("training_config", {})["temporal_postprocess"] = "none"
    torch.save(checkpoint, checkpoint_for_export)

    prediction_csv = run_dir / "full_validation_raw_predictions.csv"
    predictions = export_predictions(
        checkpoint_path=checkpoint_for_export,
        features_path=validation_features_path,
        output_csv=prediction_csv,
        batch_size=args.batch_size,
        device=args.device,
    )
    merged, summary = summarize_predictions(predictions, validation_df)
    merged.to_csv(run_dir / "full_validation_predictions_with_actual.csv", index=False)
    pd.DataFrame(summary["by_run"]).to_csv(run_dir / "full_validation_metrics_by_run.csv", index=False)
    final_summary = {
        "dataset": "PRONOSTIA/FEMTO Bearing",
        "protocol": (
            "Train on NASA FEMTO Training_set/Learning_set with one held-out "
            "learning bearing for early stopping; evaluate on Validation_Set/Full_Test_Set."
        ),
        "stride": args.stride,
        "internal_training_summary": train_summary,
        **summary,
    }
    (run_dir / "full_validation_summary.json").write_text(json.dumps(final_summary, indent=2))
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
