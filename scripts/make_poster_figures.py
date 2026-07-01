#!/usr/bin/env python
"""Generate poster figures for the PHM Korea abstract pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phmkorea_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy import signal
import torch

sys.path.insert(0, str(ROOT / "src"))

from phm_pipeline.config import BearingGeometry, FeatureConfig
from phm_pipeline.data import VibrationZipReader, discover_train_runs, read_operation_csv
from phm_pipeline.features import order_spectrum, rpm_trace
from phm_pipeline.model import build_model, load_model_state, model_config_from_dict
from phm_pipeline.training import Standardizer


OUT_DIR = ROOT / "figures" / "poster"
FEATURES_PATH = ROOT / "artifacts" / "features" / "train_full_order_domain.parquet"
LOBO_PRED_PATH = (
    ROOT
    / "artifacts"
    / "runs"
    / "abstract_lobo_order_domain_eol_q09_test"
    / "val_predictions_with_actual.csv"
)
LOBO_RUN_METRICS_PATH = (
    ROOT
    / "artifacts"
    / "runs"
    / "abstract_lobo_order_domain_eol_q09_test"
    / "val_metrics_by_run.csv"
)
LOBO_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "runs"
    / "abstract_lobo_order_domain_eol_q09_test"
    / "val_metrics_summary.json"
)
TAIL_RUN_METRICS_PATH = (
    ROOT
    / "artifacts"
    / "runs"
    / "abstract_per_run_tail_order_domain_eol_q09_test"
    / "val_metrics_by_run.csv"
)
TAIL_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "runs"
    / "abstract_per_run_tail_order_domain_eol_q09_test"
    / "val_metrics_summary.json"
)
PRONOSTIA_SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "external"
    / "pronostia_order_transformer"
    / "model_stride5_Bearing3_2"
    / "full_validation_summary.json"
)
ATTENTION_CHECKPOINT_PATH = (
    ROOT / "artifacts" / "runs" / "per_run_tail_log_order_domain" / "best_model.pt"
)

COLORS = {
    "ink": "#202124",
    "muted": "#667085",
    "grid": "#D0D5DD",
    "light": "#EEF2F6",
    "navy": "#284B63",
    "teal": "#2A9D8F",
    "rust": "#C65D3A",
    "gold": "#E9A93A",
    "purple": "#6B5CA5",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.9,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT_DIR / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


def hours(values_s: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values_s, dtype=float) / 3600.0


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def pick_high_rpm_variation_segment(features: pd.DataFrame) -> pd.Series:
    return features.sort_values("opwin_rpm_std", ascending=False).iloc[0]


def read_example_vibration(features: pd.DataFrame) -> tuple[pd.Series, np.ndarray, np.ndarray, float]:
    row = pick_high_rpm_variation_segment(features)
    run = next(r for r in discover_train_runs(ROOT / "data" / "Train") if r.run_id == row["run_id"])
    operation = read_operation_csv(run.operation_csv)
    op_window = operation[
        (operation["time_s"] >= float(row["start_time_s"]))
        & (operation["time_s"] <= float(row["end_time_s"]))
    ]
    rpm_values = op_window["rpm"].to_numpy(dtype=float)
    if rpm_values.size < 2:
        rpm_values = np.array([float(row["op_rpm"])], dtype=float)
    fs = FeatureConfig().sample_rate_hz
    with VibrationZipReader(run.vibration_zip) as reader:
        x = reader.read_segment(str(row["member"]), channels=[0])[0].astype(float)
    return row, x - np.mean(x), rpm_values, fs


def angular_resample(
    x: np.ndarray,
    rpm: np.ndarray,
    *,
    sample_rate_hz: float,
    samples_per_revolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = np.maximum(rpm_trace(rpm, x.size, sample_rate_hz), 1.0)
    rotations = np.cumsum(trace / 60.0) / sample_rate_hz
    rotations -= rotations[0]
    total_revs = float(rotations[-1])
    n_angle = max(256, int(total_revs * samples_per_revolution))
    uniform_revs = np.linspace(0.0, total_revs, n_angle, endpoint=False)
    angular = np.interp(uniform_revs, rotations, x - np.mean(x))
    return rotations, uniform_revs, angular


def figure_domain_example(features: pd.DataFrame) -> None:
    config = FeatureConfig()
    row, x, rpm_values, fs = read_example_vibration(features)

    time_s = np.arange(x.size) / fs
    view = slice(0, int(2.0 * fs), 16)
    freq, psd = signal.welch(x, fs=fs, nperseg=16384, noverlap=8192, scaling="spectrum")
    amp = np.sqrt(np.maximum(psd, 0.0))
    keep_freq = freq <= 350.0
    orders, order_amp = order_spectrum(
        x,
        rpm_values,
        sample_rate_hz=fs,
        samples_per_revolution=config.samples_per_revolution,
        max_order=40.0,
    )

    fig = plt.figure(figsize=(12, 6.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[0.8, 1.0], figure=fig)
    ax_time = fig.add_subplot(gs[0, :])
    ax_freq = fig.add_subplot(gs[1, 0])
    ax_order = fig.add_subplot(gs[1, 1])

    ax_time.plot(time_s[view], x[view], color=COLORS["navy"], lw=0.8)
    ax_time.set_title("A 60 s vibration segment is measured while shaft speed changes")
    ax_time.set_xlabel("Time inside acquisition (s)")
    ax_time.set_ylabel("Acceleration")
    ax_time.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_time.text(
        0.99,
        0.86,
        f"{row['run_id']} segment {int(row['segment_id'])} | RPM std {row['opwin_rpm_std']:.1f}",
        transform=ax_time.transAxes,
        ha="right",
        va="center",
        color=COLORS["muted"],
    )

    ax_freq.plot(freq[keep_freq], amp[keep_freq], color=COLORS["rust"], lw=1.0)
    ax_freq.set_title("Frequency domain: fault bands move with RPM")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Amplitude")
    ax_freq.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_freq.set_xlim(0, 350)

    ax_order.plot(orders, order_amp, color=COLORS["teal"], lw=1.0)
    ax_order.set_title("Order domain: fault orders stay fixed")
    ax_order.set_xlabel("Order (cycles / shaft revolution)")
    ax_order.set_ylabel("Amplitude")
    ax_order.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_order.set_xlim(0, 40)

    rpm_min = float(row["opwin_rpm_min"])
    rpm_max = float(row["opwin_rpm_max"])
    fault_orders = BearingGeometry().resolved_fault_orders()
    shown = [("BPFO", fault_orders["bpfo"], COLORS["gold"]), ("BPFI", fault_orders["bpfi"], COLORS["purple"]), ("BSF", fault_orders["bsf"], COLORS["navy"])]
    ymax_freq = float(np.nanmax(amp[keep_freq])) if np.any(keep_freq) else 1.0
    ymax_order = float(np.nanmax(order_amp)) if order_amp.size else 1.0
    for idx, (label, order, color) in enumerate(shown):
        low = order * rpm_min / 60.0
        high = order * rpm_max / 60.0
        ax_freq.axvspan(low, high, color=color, alpha=0.16)
        ax_freq.text(
            (low + high) / 2,
            ymax_freq * (0.88 - 0.10 * idx),
            label,
            ha="center",
            va="top",
            color=color,
            fontsize=9,
        )
        ax_order.axvline(order, color=color, lw=1.2, ls="--")
        ax_order.text(
            order + 0.25,
            ymax_order * (0.86 - 0.10 * idx),
            label,
            ha="left",
            va="top",
            color=color,
            fontsize=9,
        )

    fig.suptitle("Time/Frequency Features vs Order-Domain Representation", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "01_time_frequency_vs_order_domain")


def figure_order_tracking_algorithm_steps(features: pd.DataFrame) -> None:
    config = FeatureConfig()
    row, x, rpm_values, fs = read_example_vibration(features)
    rotations, uniform_revs, angular = angular_resample(
        x,
        rpm_values,
        sample_rate_hz=fs,
        samples_per_revolution=config.samples_per_revolution,
    )
    trace = rpm_trace(rpm_values, x.size, fs)
    time_s = np.arange(x.size, dtype=float) / fs
    time_view = slice(0, x.size, max(x.size // 2400, 1))
    trace_view = slice(0, trace.size, max(trace.size // 800, 1))
    angle_view = uniform_revs <= min(25.0, float(uniform_revs[-1]))

    freq, psd = signal.welch(x, fs=fs, nperseg=16384, noverlap=8192, scaling="spectrum")
    amp = np.sqrt(np.maximum(psd, 0.0))
    keep_freq = freq <= 350.0
    orders, order_amp = order_spectrum(
        x,
        rpm_values,
        sample_rate_hz=fs,
        samples_per_revolution=config.samples_per_revolution,
        max_order=40.0,
    )

    fig = plt.figure(figsize=(12.5, 7.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.28)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_angle = fig.add_subplot(gs[0, 1])
    ax_freq = fig.add_subplot(gs[1, 0])
    ax_order = fig.add_subplot(gs[1, 1])

    ax_raw.plot(time_s[time_view], x[time_view], color=COLORS["navy"], lw=0.7)
    ax_raw.set_title("1. Raw vibration in time")
    ax_raw.set_xlabel("Time inside acquisition (s)")
    ax_raw.set_ylabel("Acceleration")
    ax_raw.set_xlim(0, 60)
    ax_raw.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)

    ax_rpm = ax_raw.twinx()
    ax_rpm.plot(time_s[trace_view], trace[trace_view], color=COLORS["gold"], lw=1.2, alpha=0.9)
    ax_rpm.set_ylabel("RPM", color=COLORS["gold"])
    ax_rpm.tick_params(axis="y", colors=COLORS["gold"])
    ax_rpm.spines["right"].set_color(COLORS["gold"])

    ax_angle.plot(uniform_revs[angle_view], angular[angle_view], color=COLORS["teal"], lw=0.8)
    ax_angle.set_title("2. Resample to uniform shaft angle")
    ax_angle.set_xlabel("Shaft revolutions")
    ax_angle.set_ylabel("Acceleration")
    ax_angle.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_angle.text(
        0.02,
        0.88,
        r"$\theta[n]=\sum rpm[n]/(60 f_s)$" + "\nlinear interpolation to uniform theta",
        transform=ax_angle.transAxes,
        color=COLORS["muted"],
        fontsize=10,
        va="top",
    )

    ax_freq.plot(freq[keep_freq], amp[keep_freq], color=COLORS["rust"], lw=1.0)
    ax_freq.set_title("3. Ordinary FFT: Hz axis shifts with RPM")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Amplitude")
    ax_freq.set_xlim(0, 350)
    ax_freq.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)

    ax_order.plot(orders, order_amp, color=COLORS["teal"], lw=1.0)
    ax_order.set_title("4. Angle-domain FFT: order axis is RPM-normalized")
    ax_order.set_xlabel("Order (cycles / shaft revolution)")
    ax_order.set_ylabel("Amplitude")
    ax_order.set_xlim(0, 40)
    ax_order.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)

    fault_orders = BearingGeometry().resolved_fault_orders()
    for label, order, color in [
        ("BSF", fault_orders["bsf"], COLORS["navy"]),
        ("BPFO", fault_orders["bpfo"], COLORS["gold"]),
        ("BPFI", fault_orders["bpfi"], COLORS["purple"]),
    ]:
        ax_order.axvline(order, color=color, lw=1.1, ls="--")
        ax_order.text(order + 0.25, ax_order.get_ylim()[1] * 0.82, label, color=color, fontsize=9, va="top")

    fig.suptitle(
        f"Order Tracking Algorithm Used in This Pipeline ({row['run_id']} segment {int(row['segment_id'])})",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, "08_order_tracking_algorithm_steps")


def figure_data_acquisition_timeline(features: pd.DataFrame) -> None:
    run_summary = (
        features.groupby("run_id", sort=True)
        .agg(start_s=("start_time_s", "min"), end_s=("run_end_s", "max"), segments=("segment_id", "count"))
        .reset_index()
    )

    fig = plt.figure(figsize=(12, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.05, 1.0], figure=fig, hspace=0.42)
    ax_runs = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])

    y = np.arange(len(run_summary))
    for idx, row in run_summary.iterrows():
        duration_h = row["end_s"] / 3600.0
        ax_runs.barh(idx, duration_h, color=COLORS["teal"], height=0.5)
        ax_runs.text(duration_h + 0.25, idx, f"{int(row['segments'])} segments", va="center", color=COLORS["muted"])
    ax_runs.set_yticks(y)
    ax_runs.set_yticklabels(run_summary["run_id"])
    ax_runs.invert_yaxis()
    ax_runs.set_xlabel("Run time (hours)")
    ax_runs.set_title("Run-to-failure bearing records")
    ax_runs.grid(True, axis="x", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_runs.spines[["top", "right"]].set_visible(False)

    total_min = 32
    ax_zoom.set_xlim(0, total_min)
    ax_zoom.set_ylim(0, 1)
    ax_zoom.set_yticks([])
    ax_zoom.set_xlabel("Elapsed time from run start (min)")
    ax_zoom.set_title("Acquisition pattern used to build one row of the feature table")
    for start_min in [0, 10, 20, 30]:
        ax_zoom.add_patch(Rectangle((start_min, 0.28), 1.0, 0.34, color=COLORS["rust"], alpha=0.9))
        if start_min < total_min - 1:
            ax_zoom.add_patch(Rectangle((start_min + 1.0, 0.28), 9.0, 0.34, color=COLORS["light"], alpha=1.0))
    ax_zoom.hlines(0.78, 0, total_min, color=COLORS["navy"], lw=1.3)
    sample_ticks = np.arange(0, total_min + 0.001, 1 / 6)
    ax_zoom.vlines(sample_ticks, 0.75, 0.81, color=COLORS["navy"], lw=0.25, alpha=0.55)
    ax_zoom.text(0.02, 0.11, "Red blocks: 60 s vibration TDMS", transform=ax_zoom.transAxes, ha="left", color=COLORS["rust"], fontweight="bold")
    ax_zoom.text(0.02, 0.03, "Light band: 540 s rest while operation is logged", transform=ax_zoom.transAxes, ha="left", color=COLORS["muted"])
    ax_zoom.text(total_min - 0.2, 0.86, "Operation CSV sampled through time", ha="right", color=COLORS["navy"])
    arrow = FancyArrowPatch((20.5, 0.62), (27.2, 0.92), arrowstyle="-|>", mutation_scale=14, lw=1.2, color=COLORS["ink"])
    ax_zoom.add_patch(arrow)
    ax_zoom.text(27.4, 0.94, "RUL label = failure time - segment midpoint", va="center", color=COLORS["ink"])
    ax_zoom.spines[["top", "right", "left"]].set_visible(False)
    ax_zoom.grid(True, axis="x", color=COLORS["grid"], lw=0.4, alpha=0.7)

    fig.suptitle("Data Acquisition Timeline", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    save_figure(fig, "02_data_acquisition_timeline")


def normalize_window(values: pd.DataFrame) -> pd.DataFrame:
    out = values.copy()
    for col in out.columns:
        x = out[col].to_numpy(dtype=float)
        finite = np.isfinite(x)
        if finite.sum() == 0:
            out[col] = 0.0
            continue
        med = np.nanmedian(x)
        q25, q75 = np.nanpercentile(x, [25, 75])
        scale = max(q75 - q25, np.nanstd(x), 1e-9)
        out[col] = np.clip((x - med) / scale, -3, 3)
    return out


def channel_columns(features: pd.DataFrame, suffix: str) -> list[str]:
    return [f"ch{i}_{suffix}" for i in range(1, 5) if f"ch{i}_{suffix}" in features.columns]


def channel_summary(features: pd.DataFrame, suffix: str, *, how: str = "max") -> pd.Series:
    cols = channel_columns(features, suffix)
    if not cols:
        raise KeyError(f"No channel columns found for suffix '{suffix}'")
    values = features[cols].apply(pd.to_numeric, errors="coerce")
    if how == "mean":
        return values.mean(axis=1)
    return values.max(axis=1)


def robust_unit_scale(values: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    lo, hi = np.nanpercentile(x[finite], [5, 95])
    if not np.isfinite(hi - lo) or abs(hi - lo) < 1e-12:
        lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    scale = max(float(hi - lo), 1e-12)
    return np.clip((x - float(lo)) / scale, 0.0, 1.0)


def active_feature_family_counts(features: pd.DataFrame) -> dict[str, int]:
    metadata = {
        "run_id",
        "segment_id",
        "member",
        "start_time_s",
        "mid_time_s",
        "end_time_s",
        "run_end_s",
        "rul_s",
        "rul_segments",
        "life_fraction",
    }
    cols = [c for c in features.columns if c not in metadata]
    operation = [c for c in cols if c.startswith(("op_", "opwin_"))]
    shape_terms = ("_centroid", "_spread", "_entropy", "_peak_order", "_peak_amp")
    shape = [c for c in cols if "_order" in c and c.endswith(shape_terms)]
    fault = [
        c
        for c in cols
        if any(token in c for token in ("_ftf_", "_bpfo_", "_bpfi_", "_bsf_"))
        and "_order" in c
    ]
    other = [c for c in cols if c not in set(operation).union(shape).union(fault)]
    return {
        "Operation": len(operation),
        "Order shape": len(shape),
        "Fault-order bands": len(fault),
        "Other kept": len(other),
    }


def figure_extracted_order_features(features: pd.DataFrame) -> None:
    run_id = "Train3"
    run = features[features["run_id"] == run_id].sort_values("life_fraction").reset_index(drop=True)
    if run.empty:
        run_id = str(features["run_id"].iloc[0])
        run = features[features["run_id"] == run_id].sort_values("life_fraction").reset_index(drop=True)

    selected = {
        "Order peak amp": channel_summary(run, "order_peak_amp", how="max"),
        "Envelope peak amp": channel_summary(run, "env_order_peak_amp", how="max"),
        "Order entropy": channel_summary(run, "order_entropy", how="mean"),
        "BPFO energy": channel_summary(run, "order_bpfo_energy_sum", how="max"),
        "BPFI energy": channel_summary(run, "order_bpfi_energy_sum", how="max"),
        "BSF energy": channel_summary(run, "order_bsf_energy_sum", how="max"),
        "Env BPFO energy": channel_summary(run, "env_order_bpfo_energy_sum", how="max"),
        "Env BPFI energy": channel_summary(run, "env_order_bpfi_energy_sum", how="max"),
    }
    x = run["life_fraction"].to_numpy(dtype=float)
    line_names = ["Order peak amp", "Envelope peak amp", "BPFO energy", "BPFI energy", "BSF energy"]
    heat_names = list(selected)
    heat = np.vstack([robust_unit_scale(selected[name]) for name in heat_names])

    fig = plt.figure(figsize=(12.5, 7.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.12, 0.95], width_ratios=[1.6, 1.0], figure=fig, hspace=0.42, wspace=0.50)
    ax_lines = fig.add_subplot(gs[0, :])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_counts = fig.add_subplot(gs[1, 1])

    line_colors = [COLORS["teal"], COLORS["rust"], COLORS["gold"], COLORS["purple"], COLORS["navy"]]
    for name, color in zip(line_names, line_colors, strict=True):
        ax_lines.plot(x, robust_unit_scale(selected[name]), lw=2.0, label=name, color=color)
    ax_lines.set_title(f"Selected extracted order-domain features over bearing life ({run_id})")
    ax_lines.set_xlabel("Life fraction")
    ax_lines.set_ylabel("Robust normalized value")
    ax_lines.set_xlim(0, 1)
    ax_lines.set_ylim(-0.05, 1.05)
    ax_lines.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_lines.legend(frameon=False, ncol=3, loc="upper left")
    ax_lines.spines[["top", "right"]].set_visible(False)

    im = ax_heat.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax_heat.set_title("Feature map used by the sequence model")
    ax_heat.set_yticks(np.arange(len(heat_names)))
    ax_heat.set_yticklabels(heat_names)
    ax_heat.set_xticks([0, max(len(run) // 2, 1), len(run) - 1])
    ax_heat.set_xticklabels(["early", "middle", "late"])
    ax_heat.set_xlabel("Segment order in run")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.015)
    cbar.set_label("")

    counts = active_feature_family_counts(features)
    labels = list(counts)
    values = [counts[label] for label in labels]
    colors = [COLORS["navy"], COLORS["teal"], COLORS["gold"], COLORS["muted"]]
    bars = ax_counts.barh(labels, values, color=colors)
    ax_counts.set_title("Active feature table composition")
    ax_counts.set_xlabel("Number of columns")
    ax_counts.set_ylabel("")
    ax_counts.grid(True, axis="x", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_counts.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax_counts.text(value + 2, bar.get_y() + bar.get_height() / 2, str(value), va="center", color=COLORS["muted"])
    ax_counts.text(
        0.02,
        -0.28,
        "Active abstract table keeps order/envelope-order features plus operation covariates.\n"
        "RMS/kurtosis are available in the base extractor, but not kept in this active order-only table.",
        transform=ax_counts.transAxes,
        color=COLORS["muted"],
        fontsize=9,
        va="top",
    )

    fig.suptitle(
        "Extracted Order-Domain Feature Visualization",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, "09_extracted_order_features")


def figure_transformer_input_window(features: pd.DataFrame) -> None:
    run_id = "Train3"
    run_df = features[features["run_id"] == run_id].sort_values("segment_id").reset_index(drop=True)
    context = 48
    window = run_df.tail(context)
    feature_cols = [
        "op_rpm",
        "op_torque_nm",
        "ch1_order_peak_order",
        "ch1_order_peak_amp",
        "ch1_order_bpfo_energy_sum",
        "ch1_order_bpfi_energy_sum",
        "ch1_order_bsf_energy_sum",
        "ch1_env_order_peak_amp",
        "ch2_order_peak_amp",
        "ch2_env_order_peak_amp",
    ]
    labels = [
        "RPM",
        "Torque",
        "Peak order",
        "Ch1 peak amp",
        "Ch1 BPFO energy",
        "Ch1 BPFI energy",
        "Ch1 BSF energy",
        "Ch1 envelope amp",
        "Ch2 peak amp",
        "Ch2 envelope amp",
    ]
    mat = normalize_window(window[feature_cols]).T.to_numpy()

    fig = plt.figure(figsize=(12, 6.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.3, 1.0], figure=fig, wspace=0.22)
    ax_heat = fig.add_subplot(gs[0])
    ax_diag = fig.add_subplot(gs[1])

    image = ax_heat.imshow(mat, aspect="auto", cmap="viridis", vmin=-3, vmax=3, interpolation="nearest")
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_xticks([0, 11, 23, 35, 47])
    ax_heat.set_xticklabels(["t-47", "t-36", "t-24", "t-12", "t"])
    ax_heat.set_xlabel("Causal segment index")
    ax_heat.set_title(f"Transformer input window: {context} historical segments from {run_id}")
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.015)
    cbar.set_label("Robust normalized feature value")

    ax_diag.axis("off")
    ax_diag.set_xlim(0, 1)
    ax_diag.set_ylim(0, 1)
    boxes = [
        (0.08, 0.72, 0.84, 0.15, "Order-domain\nfeature matrix", COLORS["teal"]),
        (0.08, 0.47, 0.84, 0.16, "Time-gap and\ncausal position encoding", COLORS["gold"]),
        (0.08, 0.22, 0.84, 0.16, "Transformer\nencoder", COLORS["navy"]),
        (0.22, 0.03, 0.56, 0.12, "RUL head", COLORS["rust"]),
    ]
    for x, y, w, h, text, color in boxes:
        ax_diag.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.95))
        ax_diag.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontweight="bold")
    for y0, y1 in [(0.72, 0.63), (0.47, 0.38), (0.22, 0.15)]:
        ax_diag.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1), arrowstyle="-|>", mutation_scale=15, lw=1.2, color=COLORS["ink"]))
    ax_diag.text(0.5, 0.92, "Only past segments are visible", ha="center", va="center", color=COLORS["muted"])

    fig.suptitle("Transformer Input Window", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    save_figure(fig, "03_transformer_input_window")


def transformer_attention_weights(
    features: pd.DataFrame,
    *,
    run_id: str = "Train3",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(ATTENTION_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_config = model_config_from_dict(checkpoint["model_config"])
    model = build_model(model_config)
    load_model_state(model, checkpoint["model_state_dict"])
    model.eval()

    training_config = checkpoint["training_config"]
    context_length = int(training_config.get("context_length", model_config.max_context))
    feature_columns = list(checkpoint["feature_columns"])
    standardizer = Standardizer.from_dict(checkpoint["standardizer"])

    run = features[features["run_id"] == run_id].sort_values("mid_time_s").reset_index(drop=True)
    if run.empty:
        run = features.sort_values(["run_id", "mid_time_s"]).tail(context_length).reset_index(drop=True)
    window = run.tail(context_length).copy()

    x_np = standardizer.transform(window[feature_columns].to_numpy(dtype=np.float32)).astype(np.float32)
    t_np = window["mid_time_s"].to_numpy(dtype=np.float32)
    mask_np = np.ones(len(window), dtype=bool)
    if len(window) < context_length:
        pad = context_length - len(window)
        x_np = np.pad(x_np, ((pad, 0), (0, 0)), mode="constant")
        t_np = np.pad(t_np, (pad, 0), mode="edge")
        mask_np = np.pad(mask_np, (pad, 0), mode="constant", constant_values=False)

    x = torch.from_numpy(x_np).unsqueeze(0)
    times_s = torch.from_numpy(t_np).unsqueeze(0)
    mask = torch.from_numpy(mask_np).unsqueeze(0)

    with torch.no_grad():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = model.feature_proj(model.input_norm(x))
        h = h + model.time_encoding(times_s, mask=mask) + model.gap_encoding(times_s, mask)
        h = model.pre_blocks(h)
        key_padding_mask = ~mask.bool()
        layer_weights: list[np.ndarray] = []
        encoded = h
        for layer in model.encoder.layers:
            query = layer.norm1(encoded) if getattr(layer, "norm_first", False) else encoded
            _attn_out, weights = layer.self_attn(
                query,
                query,
                query,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            layer_weights.append(weights.squeeze(0).detach().cpu().numpy())
            encoded = layer(encoded, src_key_padding_mask=key_padding_mask)

    stacked = np.stack(layer_weights, axis=0)
    # Shape: layers, heads, query_token, key_token.
    layer_head_mean = stacked.mean(axis=(0, 1))
    first_layer_heads = stacked[0]
    return layer_head_mean, first_layer_heads, mask_np


def figure_transformer_attention_map(features: pd.DataFrame) -> None:
    attention, first_layer_heads, mask_np = transformer_attention_weights(features, run_id="Train3")
    valid = np.flatnonzero(mask_np)
    attention = attention[np.ix_(valid, valid)]
    first_layer = first_layer_heads.mean(axis=0)[np.ix_(valid, valid)]

    labels = [f"t-{len(valid) - 1 - i}" if i < len(valid) - 1 else "t" for i in range(len(valid))]
    tick_positions = [0, max(len(valid) // 4, 1), max(len(valid) // 2, 1), max(3 * len(valid) // 4, 1), len(valid) - 1]
    tick_labels = [labels[pos] for pos in tick_positions]

    fig = plt.figure(figsize=(12.5, 5.9))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.30)
    ax_first = fig.add_subplot(gs[0])
    ax_all = fig.add_subplot(gs[1])

    vmax = float(max(np.quantile(attention, 0.995), np.quantile(first_layer, 0.995), 1.0 / max(len(valid), 1)))
    for ax, matrix, title in [
        (ax_first, first_layer, "Layer 1 average self-attention"),
        (ax_all, attention, "Average across layers and heads"),
    ]:
        im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax, aspect="equal", interpolation="nearest")
        ax.plot([-0.5, len(valid) - 0.5], [-0.5, len(valid) - 0.5], color="white", lw=1.0, ls="--", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("Key segment being attended to")
        ax.set_ylabel("Query segment")
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    cbar = fig.colorbar(im, ax=[ax_first, ax_all], fraction=0.026, pad=0.02)
    cbar.set_label("Attention weight")
    fig.text(
        0.02,
        0.02,
        "This is a self-attention map, not the input feature heatmap. A diagonal is not required: each past segment can attend to any other segment in the past-only window.",
        color=COLORS["muted"],
        fontsize=10,
    )
    fig.suptitle(
        "Transformer Self-Attention Map for the 48-Segment Input Window",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, "10_transformer_self_attention_map")


def poster_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
    *,
    title_size: float = 11.5,
    body_size: float = 9.5,
) -> None:
    header_h = min(0.20, max(0.10, 0.32 * h))
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.045",
        facecolor="white",
        edgecolor=color,
        linewidth=2.0,
    )
    ax.add_patch(box)
    ax.add_patch(
        Rectangle(
            (x, y + h - header_h),
            w,
            header_h,
            facecolor=color,
            edgecolor="none",
        )
    )
    ax.text(x + 0.07, y + h - header_h / 2, title, ha="left", va="center", color="white", fontsize=title_size, fontweight="bold")
    ax.text(x + 0.07, y + h - header_h - 0.06, body, ha="left", va="top", color=COLORS["ink"], fontsize=body_size, linespacing=1.18)


def feature_highlight_matrix(features: pd.DataFrame, run_id: str = "Train3") -> tuple[pd.DataFrame, list[str], np.ndarray]:
    run = features[features["run_id"] == run_id].sort_values("life_fraction").reset_index(drop=True)
    if run.empty:
        run_id = str(features["run_id"].iloc[0])
        run = features[features["run_id"] == run_id].sort_values("life_fraction").reset_index(drop=True)
    selected = {
        "RPM": run["op_rpm"],
        "Order peak": channel_summary(run, "order_peak_amp", how="max"),
        "Envelope peak": channel_summary(run, "env_order_peak_amp", how="max"),
        "BPFO energy": channel_summary(run, "order_bpfo_energy_sum", how="max"),
        "BPFI energy": channel_summary(run, "order_bpfi_energy_sum", how="max"),
        "BSF energy": channel_summary(run, "order_bsf_energy_sum", how="max"),
        "Env BPFO": channel_summary(run, "env_order_bpfo_energy_sum", how="max"),
        "Env BPFI": channel_summary(run, "env_order_bpfi_energy_sum", how="max"),
    }
    labels = list(selected)
    matrix = np.vstack([robust_unit_scale(selected[label]) for label in labels])
    return run, labels, matrix


def figure_order_domain_features_poster_highlight(features: pd.DataFrame) -> None:
    run, labels, matrix = feature_highlight_matrix(features)
    x = run["life_fraction"].to_numpy(dtype=float)
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.05, 1.18, 1.18], height_ratios=[0.88, 1.12], figure=fig, hspace=0.36, wspace=0.32)
    ax_flow = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1:])
    ax_lines = fig.add_subplot(gs[1, :])

    ax_flow.axis("off")
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    poster_box(ax_flow, 0.04, 0.67, 0.86, 0.24, "Input", "vibration + RPM", COLORS["navy"], title_size=10.0, body_size=8.7)
    poster_box(ax_flow, 0.04, 0.37, 0.86, 0.24, "Tracking", "angle resampling", COLORS["teal"], title_size=10.0, body_size=8.7)
    poster_box(ax_flow, 0.04, 0.07, 0.86, 0.24, "Features", "fault-order energy", COLORS["gold"], title_size=10.0, body_size=8.7)
    for y0, y1 in [(0.67, 0.61), (0.37, 0.31)]:
        ax_flow.add_patch(FancyArrowPatch((0.47, y0), (0.47, y1), arrowstyle="-|>", mutation_scale=16, lw=1.5, color=COLORS["ink"]))

    im = ax_heat.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax_heat.set_title("Order-domain feature map across bearing life")
    ax_heat.set_yticks(np.arange(len(labels)))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_xticks([0, max(len(run) // 2, 1), len(run) - 1])
    ax_heat.set_xticklabels(["early", "middle", "near failure"])
    ax_heat.set_xlabel("Run-to-failure segment order")
    ax_heat.tick_params(axis="y", labelsize=9)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.015)
    cbar.set_label("normalized feature level")

    line_specs = [
        ("Order peak", COLORS["teal"]),
        ("BPFO energy", COLORS["gold"]),
        ("BPFI energy", COLORS["purple"]),
        ("BSF energy", COLORS["navy"]),
        ("Envelope peak", COLORS["rust"]),
    ]
    for label, color in line_specs:
        row_idx = labels.index(label)
        smooth = pd.Series(matrix[row_idx]).rolling(5, min_periods=1, center=True).median().to_numpy()
        ax_lines.plot(x, smooth, lw=2.4, color=color, label=label)
    ax_lines.axvspan(0.78, 1.0, color=COLORS["light"], alpha=0.9, zorder=0)
    ax_lines.text(0.79, 0.96, "late-life region", color=COLORS["muted"], va="top", fontsize=10)
    ax_lines.set_title(f"Selected feature trajectories ({run['run_id'].iloc[0]})")
    ax_lines.set_xlabel("Life fraction")
    ax_lines.set_ylabel("Normalized value")
    ax_lines.set_xlim(0, 1)
    ax_lines.set_ylim(-0.05, 1.08)
    ax_lines.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.75)
    ax_lines.legend(frameon=False, ncol=5, loc="upper left")
    ax_lines.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.02,
        0.02,
        "Takeaway: order features make speed-dependent bearing fault components easier to compare across time.",
        color=COLORS["muted"],
        fontsize=10.5,
    )
    fig.suptitle("Order-Domain Features Used by the RUL Model", x=0.02, y=1.01, ha="left", fontsize=20, fontweight="bold")
    save_figure(fig, "12_order_domain_features_poster_highlight")


def transformer_window_matrix(features: pd.DataFrame, run_id: str = "Train3") -> tuple[np.ndarray, list[str], pd.DataFrame]:
    run_df = features[features["run_id"] == run_id].sort_values("segment_id").reset_index(drop=True)
    if run_df.empty:
        run_df = features.sort_values(["run_id", "segment_id"]).tail(48).reset_index(drop=True)
    context = 48
    window = run_df.tail(context)
    feature_cols = [
        "op_rpm",
        "op_torque_nm",
        "ch1_order_peak_order",
        "ch1_order_peak_amp",
        "ch1_order_bpfo_energy_sum",
        "ch1_order_bpfi_energy_sum",
        "ch1_order_bsf_energy_sum",
        "ch1_env_order_peak_amp",
        "ch2_order_peak_amp",
        "ch2_env_order_peak_amp",
    ]
    labels = [
        "RPM",
        "Torque",
        "Peak order",
        "Ch1 peak amp",
        "Ch1 BPFO",
        "Ch1 BPFI",
        "Ch1 BSF",
        "Ch1 envelope",
        "Ch2 peak amp",
        "Ch2 envelope",
    ]
    matrix = normalize_window(window[feature_cols]).T.to_numpy()
    return matrix, labels, window


def figure_transformer_poster_highlight(features: pd.DataFrame) -> None:
    matrix, labels, window = transformer_window_matrix(features)
    attention, _, mask_np = transformer_attention_weights(features, run_id=str(window["run_id"].iloc[0]))
    valid = np.flatnonzero(mask_np)
    attention = attention[np.ix_(valid, valid)]
    labels_x = [f"t-{len(valid) - 1 - i}" if i < len(valid) - 1 else "t" for i in range(len(valid))]
    ticks = [0, max(len(valid) // 3, 1), max(2 * len(valid) // 3, 1), len(valid) - 1]

    fig = plt.figure(figsize=(13.2, 7.0))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.72], figure=fig, hspace=0.34, wspace=0.30)
    ax_window = fig.add_subplot(gs[:, 0])
    ax_attn = fig.add_subplot(gs[0, 1])
    ax_arch = fig.add_subplot(gs[1, 1])

    im = ax_window.imshow(matrix, aspect="auto", cmap="viridis", vmin=-3, vmax=3, interpolation="nearest")
    ax_window.set_title("Causal input window: 48 past segments")
    ax_window.set_yticks(np.arange(len(labels)))
    ax_window.set_yticklabels(labels)
    ax_window.set_xticks([0, 11, 23, 35, 47])
    ax_window.set_xticklabels(["t-47", "t-36", "t-24", "t-12", "t"])
    ax_window.set_xlabel("Historical segment index")
    ax_window.set_ylabel("Order-domain feature")
    cbar = fig.colorbar(im, ax=ax_window, fraction=0.035, pad=0.015)
    cbar.set_label("robust normalized value")

    vmax = float(max(np.quantile(attention, 0.995), 1.0 / max(len(valid), 1)))
    attn_im = ax_attn.imshow(attention, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    ax_attn.plot([-0.5, len(valid) - 0.5], [-0.5, len(valid) - 0.5], color="white", ls="--", lw=1.0, alpha=0.75)
    ax_attn.set_title("Self-attention map")
    ax_attn.set_xlabel("key segment")
    ax_attn.set_ylabel("query segment")
    ax_attn.set_xticks(ticks)
    ax_attn.set_yticks(ticks)
    ax_attn.set_xticklabels([labels_x[i] for i in ticks])
    ax_attn.set_yticklabels([labels_x[i] for i in ticks])
    cbar2 = fig.colorbar(attn_im, ax=ax_attn, fraction=0.046, pad=0.02)
    cbar2.set_label("attention weight")

    ax_arch.axis("off")
    ax_arch.set_xlim(0, 1)
    ax_arch.set_ylim(0, 1)
    poster_box(ax_arch, 0.05, 0.62, 0.90, 0.24, "Token sequence", "each segment becomes one feature token", COLORS["teal"], body_size=9.0)
    poster_box(ax_arch, 0.05, 0.34, 0.90, 0.22, "Temporal encoding", "time gap + causal position information", COLORS["gold"], body_size=9.0)
    poster_box(ax_arch, 0.05, 0.07, 0.90, 0.21, "Transformer head", "degradation context -> RUL estimate", COLORS["navy"], body_size=9.0)
    for y0, y1 in [(0.62, 0.56), (0.34, 0.28)]:
        ax_arch.add_patch(FancyArrowPatch((0.50, y0), (0.50, y1), arrowstyle="-|>", mutation_scale=15, lw=1.5, color=COLORS["ink"]))

    fig.text(
        0.57,
        0.035,
        "Attention map note: a diagonal is not required because each query can use context from any past segment.",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    fig.suptitle("Transformer Learns the Degradation Trajectory from Order Features", x=0.02, y=1.01, ha="left", fontsize=20, fontweight="bold")
    save_figure(fig, "13_transformer_window_attention_poster_highlight")


def figure_causal_eol_smoothing_poster_highlight(pred: pd.DataFrame) -> None:
    run = select_prediction_run(pred)
    observed_h = hours(run["mid_time_s"])
    true_rul_h = hours(run["rul_s"])
    raw_rul_h = hours(run["raw_predicted_rul_s"])
    smooth_rul_h = hours(run["predicted_rul_s"])
    raw_eol_h = hours(run["mid_time_s"] + run["raw_predicted_rul_s"])
    smooth_eol_h = hours(run["mid_time_s"] + run["predicted_rul_s"])
    actual_eol = float(np.nanmedian(hours(run["mid_time_s"] + run["rul_s"])))
    cursor = max(8, int(len(run) * 0.68))
    cursor = min(cursor, len(run) - 1)
    cursor_time = float(observed_h[cursor])
    history_slice = slice(max(0, cursor - 23), cursor + 1)
    history_x = np.arange(cursor - max(0, cursor - 23) + 1)
    q90 = float(np.nanquantile(raw_eol_h[: cursor + 1], 0.90))

    fig = plt.figure(figsize=(13.2, 6.7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.38, 1.0], figure=fig, wspace=0.30)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_quantile = fig.add_subplot(gs[0, 1])

    ax_main.plot(observed_h, true_rul_h, color=COLORS["ink"], lw=2.0, ls="--", label="True RUL")
    ax_main.plot(observed_h, raw_rul_h, color=COLORS["rust"], lw=1.2, alpha=0.55, label="Raw Transformer")
    ax_main.plot(observed_h, smooth_rul_h, color=COLORS["teal"], lw=2.6, label="Causal EOL smoothed")
    ax_main.axvline(cursor_time, color=COLORS["navy"], lw=1.5, ls=":")
    ax_main.axvspan(cursor_time, float(observed_h[-1]), color=COLORS["light"], alpha=0.75)
    ax_main.text(cursor_time + 0.05, ax_main.get_ylim()[1] * 0.86, "future not used", color=COLORS["muted"], fontsize=10)
    ax_main.set_title(f"RUL output becomes steadier after causal smoothing ({run['run_id'].iloc[0]})")
    ax_main.set_xlabel("Observed run time (hours)")
    ax_main.set_ylabel("Remaining useful life (hours)")
    ax_main.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.75)
    ax_main.legend(frameon=False, loc="upper right")
    ax_main.spines[["top", "right"]].set_visible(False)

    ax_quantile.plot(history_x, raw_eol_h[history_slice], marker="o", ms=4.0, lw=1.4, color=COLORS["rust"], label="raw EOL candidates")
    ax_quantile.axhline(actual_eol, color=COLORS["ink"], lw=1.8, ls="--", label="actual EOL")
    ax_quantile.axhline(q90, color=COLORS["teal"], lw=2.4, label="causal q0.90 EOL")
    ax_quantile.scatter([history_x[-1]], [smooth_eol_h[cursor]], s=85, color=COLORS["teal"], edgecolor="white", linewidth=1.5, zorder=5)
    ax_quantile.text(
        history_x[-1],
        smooth_eol_h[cursor],
        "  current output",
        va="center",
        color=COLORS["teal"],
        fontweight="bold",
    )
    ax_quantile.set_title("Causal EOL quantile at one time step")
    ax_quantile.set_xlabel("Past model outputs only")
    ax_quantile.set_ylabel("Predicted EOL time (hours)")
    ax_quantile.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.75)
    ax_quantile.legend(frameon=False, loc="best", fontsize=8.5)
    ax_quantile.spines[["top", "right"]].set_visible(False)
    ax_quantile.text(
        0.02,
        -0.18,
        r"$RUL_t = \widehat{EOL}^{smooth}_t - t$",
        transform=ax_quantile.transAxes,
        color=COLORS["ink"],
        fontsize=12,
        fontweight="bold",
    )

    fig.text(
        0.02,
        0.02,
        "Takeaway: smoothing is causal, so the plotted RUL is stable without using future observations.",
        color=COLORS["muted"],
        fontsize=10.5,
    )
    fig.suptitle("Causal End-of-Life Smoothing for Stable RUL Prediction", x=0.02, y=1.01, ha="left", fontsize=20, fontweight="bold")
    save_figure(fig, "14_causal_eol_smoothing_poster_highlight")


def draw_wind_turbine(ax: plt.Axes, cx: float, cy: float, scale: float) -> None:
    ax.plot([cx, cx], [cy - 1.4 * scale, cy + 0.35 * scale], color=COLORS["ink"], lw=3.0, solid_capstyle="round")
    ax.add_patch(Circle((cx, cy + 0.42 * scale), 0.11 * scale, facecolor=COLORS["navy"], edgecolor="none"))
    blade_color = COLORS["teal"]
    for angle in [90, 210, 330]:
        theta = np.deg2rad(angle)
        tip = np.array([cx + np.cos(theta) * 0.78 * scale, cy + 0.42 * scale + np.sin(theta) * 0.78 * scale])
        left = np.array([cx + np.cos(theta + 0.18) * 0.13 * scale, cy + 0.42 * scale + np.sin(theta + 0.18) * 0.13 * scale])
        right = np.array([cx + np.cos(theta - 0.18) * 0.13 * scale, cy + 0.42 * scale + np.sin(theta - 0.18) * 0.13 * scale])
        ax.add_patch(Polygon([left, tip, right], closed=True, facecolor=blade_color, edgecolor="none", alpha=0.9))
    ax.add_patch(Rectangle((cx - 0.35 * scale, cy + 0.28 * scale), 0.42 * scale, 0.18 * scale, facecolor=COLORS["light"], edgecolor=COLORS["ink"], lw=1.2))
    ax.text(cx, cy - 1.58 * scale, "variable-speed\nwind turbine", ha="center", va="top", color=COLORS["ink"], fontsize=10, fontweight="bold")


def draw_sensor_node(ax: plt.Axes, x: float, y: float, color: str, label: str) -> None:
    ax.add_patch(Circle((x, y), 0.12, facecolor=color, edgecolor="white", lw=1.5))
    ax.text(x, y - 0.23, label, ha="center", va="top", color=COLORS["ink"], fontsize=9)


def pipeline_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.8))
    ax.add_patch(Rectangle((x, y + h - 0.20), w, 0.20, facecolor=color, edgecolor="none"))
    ax.text(x + 0.04, y + h - 0.10, title, va="center", ha="left", color="white", fontweight="bold", fontsize=11)
    ax.text(x + 0.04, y + h - 0.30, body, va="top", ha="left", color=COLORS["ink"], fontsize=10, linespacing=1.25)


def figure_real_world_application_context() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 6.4)

    ax.add_patch(Rectangle((0, 0), 12.8, 6.4, facecolor="white", edgecolor="none"))
    ax.text(
        0.25,
        6.10,
        "Real-World Application: Predictive Maintenance for Variable-Speed Rotating Machinery",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )

    draw_wind_turbine(ax, 1.15, 3.55, 1.35)
    draw_sensor_node(ax, 2.10, 4.40, COLORS["rust"], "vibration")
    draw_sensor_node(ax, 2.48, 3.85, COLORS["gold"], "RPM")
    draw_sensor_node(ax, 2.16, 3.25, COLORS["purple"], "temperature")
    ax.add_patch(FancyArrowPatch((2.35, 3.80), (3.15, 3.80), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=COLORS["ink"]))

    pipeline_box(
        ax,
        3.20,
        3.05,
        2.15,
        1.50,
        "1. Sensor data",
        "vibration + RPM\noperation history\nbearing segments",
        COLORS["navy"],
    )
    ax.add_patch(FancyArrowPatch((5.40, 3.80), (5.95, 3.80), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=COLORS["ink"]))
    pipeline_box(
        ax,
        6.00,
        3.05,
        2.15,
        1.50,
        "2. Order features",
        "RPM-normalized\nBPFO/BPFI/BSF bands\nenvelope-order energy",
        COLORS["teal"],
    )
    ax.add_patch(FancyArrowPatch((8.20, 3.80), (8.75, 3.80), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=COLORS["ink"]))
    pipeline_box(
        ax,
        8.80,
        3.05,
        2.15,
        1.50,
        "3. RUL model",
        "Transformer watches\ndegradation trajectory\ncausal EOL smoothing",
        COLORS["gold"],
    )
    ax.add_patch(FancyArrowPatch((9.90, 3.02), (9.90, 2.35), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=COLORS["ink"]))

    ax.add_patch(Rectangle((7.55, 0.80), 4.70, 1.35, facecolor="#F7FAFC", edgecolor=COLORS["rust"], linewidth=2.0))
    ax.text(7.78, 1.83, "Maintenance decision", ha="left", va="center", color=COLORS["rust"], fontsize=13, fontweight="bold")
    ax.text(
        7.78,
        1.45,
        "Schedule replacement before failure\nReduce unplanned downtime\nLower emergency repair cost",
        ha="left",
        va="top",
        color=COLORS["ink"],
        fontsize=10.5,
        linespacing=1.25,
    )

    ax.plot([0.45, 12.25], [0.45, 0.45], color=COLORS["grid"], lw=1.0)
    ax.text(
        0.45,
        0.23,
        "Poster sentence: Order-domain RUL prediction supports predictive maintenance by stabilizing RPM-dependent bearing fault features\n"
        "and estimating failure time early enough for maintenance planning.",
        ha="left",
        va="center",
        color=COLORS["muted"],
        fontsize=9.3,
    )
    save_figure(fig, "11_real_world_application_context")


def select_prediction_run(pred: pd.DataFrame, preferred: str = "Train3") -> pd.DataFrame:
    if preferred in set(pred["run_id"]):
        return pred[pred["run_id"] == preferred].sort_values("mid_time_s").copy()
    run_id = pred.groupby("run_id")["score"].mean().idxmax()
    return pred[pred["run_id"] == run_id].sort_values("mid_time_s").copy()


def figure_causal_eol_smoothing(pred: pd.DataFrame) -> None:
    run = select_prediction_run(pred)
    x_h = hours(run["mid_time_s"])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(x_h, hours(run["rul_s"]), color=COLORS["ink"], lw=2.0, ls="--", label="True RUL")
    ax.plot(x_h, hours(run["raw_predicted_rul_s"]), color=COLORS["rust"], lw=1.2, alpha=0.65, label="Raw Transformer")
    ax.plot(x_h, hours(run["predicted_rul_s"]), color=COLORS["teal"], lw=2.2, label="Causal EOL q=0.90")
    ax.fill_between(x_h, hours(run["raw_predicted_rul_s"]), hours(run["predicted_rul_s"]), color=COLORS["teal"], alpha=0.08)
    ax.set_title(f"Causal EOL smoothing on {run['run_id'].iloc[0]}")
    ax.set_xlabel("Observed run time (hours)")
    ax.set_ylabel("Remaining useful life (hours)")
    ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.08,
        "The smoother updates the end-of-life estimate using predictions up to the current segment only.",
        transform=ax.transAxes,
        color=COLORS["muted"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Causal End-of-Life Smoothing", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "04_causal_eol_smoothing")


def figure_predicted_eol(pred: pd.DataFrame) -> None:
    run = select_prediction_run(pred)
    observed_h = hours(run["mid_time_s"])
    actual_eol_h = hours(run["mid_time_s"] + run["rul_s"])
    raw_eol_h = hours(run["mid_time_s"] + run["raw_predicted_rul_s"])
    smooth_eol_h = hours(run["mid_time_s"] + run["predicted_rul_s"])
    actual_value = float(np.nanmedian(actual_eol_h))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.axhline(actual_value, color=COLORS["ink"], lw=2.0, ls="--", label="Actual EOL")
    ax.plot(observed_h, raw_eol_h, color=COLORS["rust"], lw=1.1, alpha=0.65, label="Raw predicted EOL")
    ax.plot(observed_h, smooth_eol_h, color=COLORS["teal"], lw=2.2, label="Smoothed predicted EOL")
    ax.set_title(f"Predicted failure time converges over the run ({run['run_id'].iloc[0]})")
    ax.set_xlabel("Observed run time (hours)")
    ax.set_ylabel("Predicted end-of-life time (hours from start)")
    ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        0.90,
        f"Actual EOL: {actual_value:.1f} h",
        transform=ax.transAxes,
        color=COLORS["ink"],
        fontweight="bold",
    )
    fig.suptitle("Predicted End-of-Life Trajectory", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "05_predicted_eol_plot")


def metric_card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, lines: list[str], color: str) -> None:
    header_h = min(0.18, max(0.08, 0.24 * h))
    first_line_y = y + h - header_h - 0.08
    line_gap = min(0.11, max(0.075, 0.16 * h))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.6))
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor=color, edgecolor="none"))
    ax.text(x + 0.03, y + h - header_h / 2, title, va="center", ha="left", color="white", fontweight="bold")
    for idx, line in enumerate(lines):
        ax.text(x + 0.03, first_line_y - idx * line_gap, line, va="center", ha="left", color=COLORS["ink"], fontsize=10.5)


def plot_rul_trajectory(
    ax: plt.Axes,
    run: pd.DataFrame,
    *,
    title: str,
    show_raw: bool,
    show_legend: bool,
) -> None:
    x_h = hours(run["mid_time_s"])
    ax.plot(x_h, hours(run["rul_s"]), color=COLORS["ink"], lw=2.0, ls="--", label="True RUL")
    if show_raw:
        ax.plot(
            x_h,
            hours(run["raw_predicted_rul_s"]),
            color=COLORS["rust"],
            lw=1.1,
            alpha=0.55,
            label="Raw Transformer",
        )
    ax.plot(
        x_h,
        hours(run["predicted_rul_s"]),
        color=COLORS["teal"],
        lw=2.4,
        label="Causal EOL smoothed",
    )
    ax.set_title(title)
    ax.set_xlabel("Observed run time (hours)")
    ax.set_ylabel("RUL (hours)")
    ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, loc="upper right")


def figure_rul_prediction_result_panel(pred: pd.DataFrame) -> None:
    lobo_summary = read_json(LOBO_SUMMARY_PATH)["overall"]
    tail_summary = read_json(TAIL_SUMMARY_PATH)["overall"]
    lobo_by_run = pd.read_csv(LOBO_RUN_METRICS_PATH).set_index("run_id")

    main_run = select_prediction_run(pred, preferred="Train3")
    hard_run = select_prediction_run(pred, preferred="Train4")
    main_run_id = str(main_run["run_id"].iloc[0])
    hard_run_id = str(hard_run["run_id"].iloc[0])

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = gridspec.GridSpec(
        3,
        3,
        width_ratios=[1.35, 1.35, 1.10],
        height_ratios=[1.0, 1.0, 0.58],
        figure=fig,
        hspace=0.55,
        wspace=0.34,
    )
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    ax_eol = fig.add_subplot(gs[0, 2])
    ax_hard = fig.add_subplot(gs[1, 2])
    ax_cards = fig.add_subplot(gs[2, :])

    plot_rul_trajectory(
        ax_main,
        main_run,
        title=f"Representative RUL trajectory ({main_run_id})",
        show_raw=True,
        show_legend=True,
    )
    main_metrics = lobo_by_run.loc[main_run_id]
    ax_main.text(
        0.02,
        0.08,
        f"{main_run_id}: score {main_metrics['score']:.3f}, MAE {main_metrics['mae_s'] / 1000:.2f} ks",
        transform=ax_main.transAxes,
        color=COLORS["ink"],
        fontweight="bold",
    )

    observed_h = hours(main_run["mid_time_s"])
    actual_eol_h = hours(main_run["mid_time_s"] + main_run["rul_s"])
    raw_eol_h = hours(main_run["mid_time_s"] + main_run["raw_predicted_rul_s"])
    smooth_eol_h = hours(main_run["mid_time_s"] + main_run["predicted_rul_s"])
    actual_eol = float(np.nanmedian(actual_eol_h))
    ax_eol.axhline(actual_eol, color=COLORS["ink"], lw=1.8, ls="--", label="Actual EOL")
    ax_eol.plot(observed_h, raw_eol_h, color=COLORS["rust"], lw=1.0, alpha=0.55, label="Raw")
    ax_eol.plot(observed_h, smooth_eol_h, color=COLORS["teal"], lw=2.0, label="Smoothed")
    ax_eol.set_title("Predicted EOL convergence")
    ax_eol.set_xlabel("Observed time (h)")
    ax_eol.set_ylabel("EOL time (h)")
    ax_eol.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_eol.spines[["top", "right"]].set_visible(False)
    ax_eol.legend(frameon=False, loc="lower right", fontsize=8)

    plot_rul_trajectory(
        ax_hard,
        hard_run,
        title=f"Hard case ({hard_run_id})",
        show_raw=False,
        show_legend=False,
    )
    hard_metrics = lobo_by_run.loc[hard_run_id]
    ax_hard.text(
        0.04,
        0.08,
        f"score {hard_metrics['score']:.3f}\nMAE {hard_metrics['mae_s'] / 1000:.2f} ks",
        transform=ax_hard.transAxes,
        color=COLORS["muted"],
        fontsize=9,
    )

    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    metric_card(
        ax_cards,
        0.00,
        0.10,
        0.235,
        0.78,
        "Overall LOBO",
        [
            f"score {lobo_summary['score']:.3f}",
            f"MAE {lobo_summary['mae_s'] / 1000:.2f} ks",
            f"overprediction {100 * lobo_summary['over_prediction_rate']:.1f}%",
        ],
        COLORS["teal"],
    )
    metric_card(
        ax_cards,
        0.255,
        0.10,
        0.235,
        0.78,
        "Tail validation",
        [
            f"score {tail_summary['score']:.3f}",
            f"MAE {tail_summary['mae_s'] / 1000:.2f} ks",
            f"overprediction {100 * tail_summary['over_prediction_rate']:.1f}%",
        ],
        COLORS["navy"],
    )
    metric_card(
        ax_cards,
        0.510,
        0.10,
        0.235,
        0.78,
        "Displayed run",
        [
            f"{main_run_id}",
            f"MAE {main_metrics['mae_s'] / 1000:.2f} ks",
            f"score {main_metrics['score']:.3f}",
        ],
        COLORS["gold"],
    )
    metric_card(
        ax_cards,
        0.765,
        0.10,
        0.235,
        0.78,
        "Takeaway",
        [
            "tracks RUL trend",
            "stabilizes EOL",
            "shows hard case",
        ],
        COLORS["rust"],
    )

    fig.suptitle(
        "RUL Prediction Result: Trajectory, End-of-Life Stability, and Error",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    save_figure(fig, "07_rul_prediction_result_panel")


def figure_results() -> None:
    lobo = pd.read_csv(LOBO_RUN_METRICS_PATH)
    tail = pd.read_csv(TAIL_RUN_METRICS_PATH)
    lobo_summary = read_json(LOBO_SUMMARY_PATH)["overall"]
    tail_summary = read_json(TAIL_SUMMARY_PATH)["overall"]
    pron_summary = read_json(PRONOSTIA_SUMMARY_PATH)["overall"] if PRONOSTIA_SUMMARY_PATH.exists() else None

    fig = plt.figure(figsize=(12, 6.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.72, 1.0], figure=fig, hspace=0.35, wspace=0.25)
    ax_cards = fig.add_subplot(gs[0, :])
    ax_score = fig.add_subplot(gs[1, 0])
    ax_mae = fig.add_subplot(gs[1, 1])
    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)

    metric_card(
        ax_cards,
        0.00,
        0.08,
        0.31,
        0.78,
        "KSPHM LOBO",
        [
            f"score {lobo_summary['score']:.3f}",
            f"MAE {lobo_summary['mae_s'] / 1000:.2f} ks",
            f"overprediction {100 * lobo_summary['over_prediction_rate']:.1f}%",
        ],
        COLORS["teal"],
    )
    metric_card(
        ax_cards,
        0.345,
        0.08,
        0.31,
        0.78,
        "KSPHM Tail",
        [
            f"score {tail_summary['score']:.3f}",
            f"MAE {tail_summary['mae_s'] / 1000:.2f} ks",
            f"overprediction {100 * tail_summary['over_prediction_rate']:.1f}%",
        ],
        COLORS["navy"],
    )
    if pron_summary is not None:
        pron_lines = [
            f"score {pron_summary['score']:.3f}",
            f"MAE {pron_summary['mae_s'] / 1000:.2f} ks",
            f"normalized MAE {pron_summary['normalized_mae']:.3f}",
        ]
    else:
        pron_lines = ["not available", "", ""]
    metric_card(ax_cards, 0.69, 0.08, 0.31, 0.78, "PRONOSTIA diagnostic", pron_lines, COLORS["rust"])

    runs = list(lobo["run_id"])
    x = np.arange(len(runs))
    width = 0.36
    tail_by_run = tail.set_index("run_id").reindex(runs).reset_index()
    ax_score.bar(x - width / 2, lobo["score"], width, label="LOBO", color=COLORS["teal"])
    ax_score.bar(x + width / 2, tail_by_run["score"], width, label="Tail", color=COLORS["navy"])
    ax_score.set_xticks(x)
    ax_score.set_xticklabels(runs)
    ax_score.set_ylabel("Score")
    ax_score.set_title("Score by bearing run")
    ax_score.grid(True, axis="y", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_score.legend(frameon=False)
    ax_score.spines[["top", "right"]].set_visible(False)

    ax_mae.bar(x - width / 2, lobo["mae_s"] / 1000.0, width, label="LOBO", color=COLORS["teal"])
    ax_mae.bar(x + width / 2, tail_by_run["mae_s"] / 1000.0, width, label="Tail", color=COLORS["navy"])
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(runs)
    ax_mae.set_ylabel("MAE (ks)")
    ax_mae.set_title("Absolute RUL error by bearing run")
    ax_mae.grid(True, axis="y", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_mae.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Result Summary", x=0.02, y=1.02, ha="left", fontsize=18, fontweight="bold")
    save_figure(fig, "06_results_summary")


def main() -> None:
    set_style()
    features = pd.read_parquet(FEATURES_PATH)
    predictions = pd.read_csv(LOBO_PRED_PATH)
    figure_domain_example(features)
    figure_order_tracking_algorithm_steps(features)
    figure_extracted_order_features(features)
    figure_data_acquisition_timeline(features)
    figure_transformer_input_window(features)
    figure_transformer_attention_map(features)
    figure_order_domain_features_poster_highlight(features)
    figure_transformer_poster_highlight(features)
    figure_real_world_application_context()
    figure_causal_eol_smoothing(predictions)
    figure_causal_eol_smoothing_poster_highlight(predictions)
    figure_predicted_eol(predictions)
    figure_results()
    figure_rul_prediction_result_panel(predictions)
    print(f"wrote poster figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
