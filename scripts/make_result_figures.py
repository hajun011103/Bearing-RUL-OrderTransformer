#!/usr/bin/env python
"""Generate result-focused figures for the PHM Korea RUL poster."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phmkorea_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


OUT_DIR = ROOT / "figures" / "results"

LOBO_DIR = ROOT / "artifacts" / "runs" / "abstract_lobo_order_domain_eol_q09_test"
TAIL_DIR = ROOT / "artifacts" / "runs" / "abstract_per_run_tail_order_domain_eol_q09_test"
BASE_LOBO_DIR = ROOT / "artifacts" / "runs" / "lobo_order_domain"
PRONOSTIA_DIR = ROOT / "artifacts" / "external" / "pronostia_order_transformer" / "model_stride5_Bearing3_2"

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
            "axes.titlesize": 14,
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
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def hours(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) / 3600.0


def ks(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) / 1000.0


def run_sort_key(run_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in str(run_id) if ch.isdigit())
    return (int(digits) if digits else 10_000, str(run_id))


def load_kspm_lobo() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred = pd.read_csv(LOBO_DIR / "val_predictions_with_actual.csv")
    metrics = pd.read_csv(LOBO_DIR / "val_metrics_by_run.csv").sort_values("run_id", key=lambda s: s.map(run_sort_key))
    summary = read_json(LOBO_DIR / "val_metrics_summary.json")["overall"]
    return pred, metrics, summary


def load_kspm_tail() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred = pd.read_csv(TAIL_DIR / "val_predictions_with_actual.csv")
    metrics = pd.read_csv(TAIL_DIR / "val_metrics_by_run.csv").sort_values("run_id", key=lambda s: s.map(run_sort_key))
    summary = read_json(TAIL_DIR / "val_metrics_summary.json")["overall"]
    return pred, metrics, summary


def load_pronostia() -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    pred_path = PRONOSTIA_DIR / "full_validation_predictions_with_actual.csv"
    metrics_path = PRONOSTIA_DIR / "full_validation_metrics_by_run.csv"
    summary_path = PRONOSTIA_DIR / "full_validation_summary.json"
    if not (pred_path.exists() and metrics_path.exists() and summary_path.exists()):
        return None
    return (
        pd.read_csv(pred_path),
        pd.read_csv(metrics_path).sort_values("score", ascending=False),
        read_json(summary_path)["overall"],
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": COLORS["grid"], "boxstyle": "round,pad=0.25"},
    )


def metric_card(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, lines: list[str], color: str) -> None:
    header_h = 0.23 * h
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.8))
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor=color, edgecolor="none"))
    ax.text(x + 0.03, y + h - header_h / 2, title, ha="left", va="center", color="white", fontweight="bold")
    for idx, line in enumerate(lines):
        ax.text(x + 0.03, y + h - header_h - 0.10 - idx * 0.17, line, ha="left", va="center", color=COLORS["ink"])


def plot_rul(
    ax: plt.Axes,
    run: pd.DataFrame,
    *,
    title: str,
    show_raw: bool = True,
    show_legend: bool = False,
) -> None:
    x = hours(run["mid_time_s"])
    ax.plot(x, hours(run["rul_s"]), color=COLORS["ink"], lw=2.0, ls="--", label="True RUL")
    if show_raw and "raw_predicted_rul_s" in run:
        ax.plot(x, hours(run["raw_predicted_rul_s"]), color=COLORS["rust"], lw=1.1, alpha=0.55, label="Raw Transformer")
    ax.plot(x, hours(run["predicted_rul_s"]), color=COLORS["teal"], lw=2.4, label="Causal EOL smoothed")
    ax.set_title(title)
    ax.set_xlabel("Observed time (h)")
    ax.set_ylabel("RUL (h)")
    ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="upper right")


def plot_eol(ax: plt.Axes, run: pd.DataFrame, *, title: str, show_legend: bool = False) -> None:
    x = hours(run["mid_time_s"])
    actual_eol = float(np.nanmedian(hours(run["mid_time_s"] + run["rul_s"])))
    ax.axhline(actual_eol, color=COLORS["ink"], lw=1.8, ls="--", label="Actual EOL")
    if "raw_predicted_rul_s" in run:
        ax.plot(x, hours(run["mid_time_s"] + run["raw_predicted_rul_s"]), color=COLORS["rust"], lw=1.1, alpha=0.55, label="Raw EOL")
    ax.plot(x, hours(run["mid_time_s"] + run["predicted_rul_s"]), color=COLORS["teal"], lw=2.2, label="Smoothed EOL")
    ax.set_title(title)
    ax.set_xlabel("Observed time (h)")
    ax.set_ylabel("Predicted EOL time (h)")
    ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, fontsize=9, loc="best")


def figure_overall_scorecards(lobo_metrics: pd.DataFrame, lobo_summary: dict, tail_summary: dict, pron_summary: dict | None) -> None:
    fig = plt.figure(figsize=(13.2, 6.8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.72, 1.0], figure=fig, hspace=0.42, wspace=0.26)
    ax_cards = fig.add_subplot(gs[0, :])
    ax_score = fig.add_subplot(gs[1, 0])
    ax_mae = fig.add_subplot(gs[1, 1])
    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)

    metric_card(
        ax_cards,
        0.00,
        0.10,
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
        0.10,
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
    if pron_summary:
        pron_lines = [
            f"score {pron_summary['score']:.3f}",
            f"MAE {pron_summary['mae_s'] / 1000:.2f} ks",
            f"normalized MAE {pron_summary['normalized_mae']:.3f}",
        ]
    else:
        pron_lines = ["not available", "", ""]
    metric_card(ax_cards, 0.69, 0.10, 0.31, 0.78, "PRONOSTIA external", pron_lines, COLORS["rust"])

    x = np.arange(len(lobo_metrics))
    ax_score.bar(x, lobo_metrics["score"], color=COLORS["teal"])
    ax_score.set_xticks(x)
    ax_score.set_xticklabels(lobo_metrics["run_id"])
    ax_score.set_ylim(0, max(0.85, float(lobo_metrics["score"].max()) * 1.15))
    ax_score.set_ylabel("Score (higher is better)")
    ax_score.set_title("LOBO score by held-out run")
    ax_score.grid(True, axis="y", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_score.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(lobo_metrics["score"]):
        ax_score.text(idx, value + 0.02, f"{value:.3f}", ha="center", va="bottom", color=COLORS["ink"], fontsize=9)

    ax_mae.bar(x, lobo_metrics["mae_s"] / 1000.0, color=COLORS["gold"])
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(lobo_metrics["run_id"])
    ax_mae.set_ylabel("MAE (ks, lower is better)")
    ax_mae.set_title("LOBO absolute RUL error by held-out run")
    ax_mae.grid(True, axis="y", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_mae.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(lobo_metrics["mae_s"] / 1000.0):
        ax_mae.text(idx, value + 0.35, f"{value:.1f}", ha="center", va="bottom", color=COLORS["ink"], fontsize=9)

    fig.suptitle("Result Summary: Order-Domain Transformer + Causal EOL Smoothing", x=0.02, y=1.02, ha="left", fontsize=19, fontweight="bold")
    save_figure(fig, "result_01_overall_scorecards")


def figure_kspm_lobo_rul(pred: pd.DataFrame, metrics: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=False, sharey=False)
    metric_by_run = metrics.set_index("run_id")
    for ax, run_id in zip(axes.flat, runs, strict=False):
        run = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        plot_rul(ax, run, title=f"{run_id} RUL trajectory", show_raw=True, show_legend=(run_id == runs[0]))
        if run_id in metric_by_run.index:
            row = metric_by_run.loc[run_id]
            ax.text(
                0.03,
                0.07,
                f"score {row['score']:.3f} | MAE {row['mae_s'] / 1000:.2f} ks",
                transform=ax.transAxes,
                color=COLORS["ink"],
                fontweight="bold",
            )
    fig.suptitle("KSPHM LOBO RUL Trajectories", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_02_kspm_lobo_rul_trajectories")


def figure_kspm_lobo_eol(pred: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=False, sharey=False)
    for ax, run_id in zip(axes.flat, runs, strict=False):
        run = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        plot_eol(ax, run, title=f"{run_id} predicted EOL convergence", show_legend=(run_id == runs[0]))
    fig.suptitle("KSPHM Predicted End-of-Life Convergence", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_03_kspm_lobo_eol_convergence")


def figure_kspm_lobo_error(pred: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=True, sharey=False)
    for ax, run_id in zip(axes.flat, runs, strict=False):
        run = pred[pred["run_id"] == run_id].sort_values("life_fraction")
        color = np.where(run["over_predicted"].to_numpy(bool), COLORS["rust"], COLORS["teal"])
        ax.scatter(run["life_fraction"], hours(run["abs_error_s"]), c=color, s=22, alpha=0.82, edgecolors="none")
        smooth = pd.Series(hours(run["abs_error_s"])).rolling(9, min_periods=1, center=True).median()
        ax.plot(run["life_fraction"], smooth, color=COLORS["ink"], lw=1.7, label="rolling median")
        ax.set_title(f"{run_id} absolute error over life")
        ax.set_xlabel("Life fraction")
        ax.set_ylabel("Absolute RUL error (h)")
        ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.03, 0.90, "rust = overprediction", transform=ax.transAxes, color=COLORS["muted"], fontsize=9)
    fig.suptitle("KSPHM LOBO Error Distribution Over Bearing Life", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_04_kspm_lobo_error_over_life")


def figure_tail_rul(pred: pd.DataFrame, metrics: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), sharex=False, sharey=False)
    metric_by_run = metrics.set_index("run_id")
    for ax, run_id in zip(axes.flat, runs, strict=False):
        run = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        plot_rul(ax, run, title=f"{run_id} tail-validation RUL", show_raw=True, show_legend=(run_id == runs[0]))
        ax.axvspan(float(hours(run["mid_time_s"]).min()), float(hours(run["mid_time_s"]).max()), color=COLORS["light"], alpha=0.35, zorder=0)
        if run_id in metric_by_run.index:
            row = metric_by_run.loc[run_id]
            ax.text(0.03, 0.07, f"score {row['score']:.3f} | MAE {row['mae_s'] / 1000:.2f} ks", transform=ax.transAxes, color=COLORS["ink"], fontweight="bold")
    fig.suptitle("KSPHM Late-Life Tail Validation", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_05_tail_validation_rul_trajectories")


def figure_lobo_tail_metric_bars(lobo: pd.DataFrame, tail: pd.DataFrame) -> None:
    runs = list(lobo["run_id"])
    tail = tail.set_index("run_id").reindex(runs).reset_index()
    x = np.arange(len(runs))
    width = 0.36

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    specs = [
        ("score", "Score (higher is better)", "Score by run", 1.0),
        ("mae_s", "MAE (ks, lower is better)", "Mean absolute RUL error", 1000.0),
        ("rmse_s", "RMSE (ks, lower is better)", "Root mean square RUL error", 1000.0),
        ("over_prediction_rate", "Overprediction rate", "Dangerous overprediction rate", 1.0),
    ]
    for ax, (col, ylabel, title, divisor) in zip(axes, specs, strict=True):
        lobo_values = lobo[col] / divisor
        tail_values = tail[col] / divisor
        ax.bar(x - width / 2, lobo_values, width, label="LOBO", color=COLORS["teal"])
        ax.bar(x + width / 2, tail_values, width, label="Tail", color=COLORS["navy"])
        ax.set_xticks(x)
        ax.set_xticklabels(runs)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", color=COLORS["grid"], lw=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("KSPHM LOBO vs Late-Life Tail Validation Metrics", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    save_figure(fig, "result_06_lobo_vs_tail_metric_bars")


def figure_predicted_vs_true(lobo_pred: pd.DataFrame, tail_pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharex=False, sharey=False)
    datasets = [
        ("LOBO", lobo_pred, COLORS["teal"]),
        ("Tail validation", tail_pred, COLORS["navy"]),
    ]
    for ax, (title, frame, color) in zip(axes, datasets, strict=True):
        true_h = hours(frame["rul_s"])
        pred_h = hours(frame["predicted_rul_s"])
        limit = float(max(np.nanmax(true_h), np.nanmax(pred_h))) * 1.05
        ax.plot([0, limit], [0, limit], color=COLORS["ink"], lw=1.6, ls="--", label="perfect prediction")
        ax.scatter(true_h, pred_h, c=color, s=24, alpha=0.62, edgecolors="none")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_title(title)
        ax.set_xlabel("True RUL (h)")
        ax.set_ylabel("Predicted RUL (h)")
        ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Predicted vs True RUL", x=0.02, y=1.02, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_07_predicted_vs_true_scatter")


def summary_from_dir(path: Path) -> dict | None:
    for name in ("val_metrics_summary.json", "lobo_summary.json", "full_validation_summary.json"):
        candidate = path / name
        if candidate.exists():
            data = read_json(candidate)
            return data.get("overall", data)
    return None


def figure_ablation_comparison() -> None:
    records = []
    candidates = [
        ("LOBO raw Transformer", BASE_LOBO_DIR),
        ("Temporal decay strict", ROOT / "artifacts" / "runs" / "lobo_order_domain_temporal_decay_strict"),
        ("EOL q0.75 blend0.5", ROOT / "artifacts" / "runs" / "lobo_order_domain_temporal_eol_q075_blend05"),
        ("EOL q0.90 final", LOBO_DIR),
        ("Tail base", ROOT / "artifacts" / "runs" / "per_run_tail_log_order_domain"),
        ("Tail scale0.41", ROOT / "artifacts" / "runs" / "per_run_tail_log_order_domain_scale041"),
        ("Tail final q0.90", TAIL_DIR),
    ]
    for label, path in candidates:
        summary = summary_from_dir(path)
        if summary:
            records.append({"label": label, **summary})
    frame = pd.DataFrame(records)

    fig = plt.figure(figsize=(13.2, 6.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)
    ax_score = fig.add_subplot(gs[0])
    ax_mae = fig.add_subplot(gs[1])
    y = np.arange(len(frame))
    colors = [COLORS["teal"] if "LOBO" in label or "EOL" in label else COLORS["navy"] for label in frame["label"]]
    ax_score.barh(y, frame["score"], color=colors)
    ax_score.set_yticks(y)
    ax_score.set_yticklabels(frame["label"])
    ax_score.invert_yaxis()
    ax_score.set_xlabel("Score (higher is better)")
    ax_score.set_title("Postprocessing and validation variants")
    ax_score.grid(True, axis="x", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_score.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(frame["score"]):
        ax_score.text(value + 0.012, idx, f"{value:.3f}", va="center", color=COLORS["ink"], fontsize=9)

    ax_mae.barh(y, frame["mae_s"] / 1000.0, color=colors)
    ax_mae.set_yticks(y)
    ax_mae.set_yticklabels([])
    ax_mae.invert_yaxis()
    ax_mae.set_xlabel("MAE (ks, lower is better)")
    ax_mae.set_title("Absolute error by variant")
    ax_mae.grid(True, axis="x", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_mae.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(frame["mae_s"] / 1000.0):
        ax_mae.text(value + 0.35, idx, f"{value:.1f}", va="center", color=COLORS["ink"], fontsize=9)

    fig.suptitle("Ablation / Variant Result Comparison", x=0.02, y=1.02, ha="left", fontsize=19, fontweight="bold")
    save_figure(fig, "result_08_ablation_comparison")


def figure_pronostia_external(pron: tuple[pd.DataFrame, pd.DataFrame, dict] | None) -> None:
    if pron is None:
        return
    pred, metrics, summary = pron
    selected = [
        metrics.iloc[0]["run_id"],
        metrics.iloc[len(metrics) // 2]["run_id"],
        metrics.iloc[-1]["run_id"],
    ]
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.82, 1.0], figure=fig, hspace=0.42, wspace=0.30)
    ax_bar = fig.add_subplot(gs[0, :2])
    ax_card = fig.add_subplot(gs[0, 2])
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    y = np.arange(len(metrics))
    colors = [COLORS["teal"] if run in selected else COLORS["muted"] for run in metrics["run_id"]]
    ax_bar.barh(y, metrics["score"], color=colors, alpha=0.92)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(metrics["run_id"], fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Score (higher is better)")
    ax_bar.set_title("External PRONOSTIA validation score by bearing")
    ax_bar.grid(True, axis="x", color=COLORS["grid"], lw=0.5, alpha=0.8)
    ax_bar.spines[["top", "right"]].set_visible(False)

    ax_card.axis("off")
    ax_card.set_xlim(0, 1)
    ax_card.set_ylim(0, 1)
    metric_card(
        ax_card,
        0.04,
        0.12,
        0.92,
        0.78,
        "PRONOSTIA overall",
        [
            f"score {summary['score']:.3f}",
            f"MAE {summary['mae_s'] / 1000:.2f} ks",
            f"normalized MAE {summary['normalized_mae']:.3f}",
        ],
        COLORS["rust"],
    )

    metric_by_run = metrics.set_index("run_id")
    for ax, run_id in zip(axes, selected, strict=True):
        run = pred[pred["run_id"] == run_id].sort_values("life_fraction")
        run_end = float(run["run_end_s"].max())
        true_norm = run["rul_s"] / run_end
        pred_norm = run["predicted_rul_s"] / run_end
        ax.plot(run["life_fraction"], true_norm, color=COLORS["ink"], lw=2.0, ls="--", label="True normalized RUL")
        ax.plot(run["life_fraction"], pred_norm, color=COLORS["teal"], lw=2.0, label="Predicted")
        ax.set_title(f"{run_id}: score {metric_by_run.loc[run_id, 'score']:.3f}")
        ax.set_xlabel("Life fraction")
        ax.set_ylabel("Normalized RUL")
        ymax = min(2.1, max(1.05, float(np.nanmax([true_norm.max(), pred_norm.max()])) * 1.08))
        ax.set_ylim(-0.05, ymax)
        ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.text(
        0.02,
        0.02,
        "External result is a domain-shift diagnostic; PRONOSTIA operating conditions differ from the KSPHM variable-speed setup.",
        color=COLORS["muted"],
        fontsize=10,
    )
    fig.suptitle("External Dataset Result: PRONOSTIA / FEMTO-ST", x=0.02, y=1.02, ha="left", fontsize=19, fontweight="bold")
    save_figure(fig, "result_09_pronostia_external_validation")


def figure_pronostia_rul_trajectories(pron: tuple[pd.DataFrame, pd.DataFrame, dict] | None) -> None:
    if pron is None:
        return
    pred, metrics, _summary = pron
    runs = sorted(pred["run_id"].unique(), key=run_sort_key)
    metric_by_run = metrics.set_index("run_id")
    ncols = 3
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 10.8), sharex=True, sharey=False)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(runs):
            ax.axis("off")
            continue
        run_id = runs[idx]
        run = pred[pred["run_id"] == run_id].sort_values("life_fraction")
        run_end = float(run["run_end_s"].max())
        x = run["life_fraction"].to_numpy(dtype=float)
        true_norm = run["rul_s"].to_numpy(dtype=float) / run_end
        raw_norm = run["raw_predicted_rul_s"].to_numpy(dtype=float) / run_end
        pred_norm = run["predicted_rul_s"].to_numpy(dtype=float) / run_end

        ax.plot(x, true_norm, color=COLORS["ink"], lw=1.8, ls="--", label="True normalized RUL")
        ax.plot(x, raw_norm, color=COLORS["rust"], lw=1.0, alpha=0.50, label="Raw Transformer")
        ax.plot(x, pred_norm, color=COLORS["teal"], lw=2.0, label="Predicted")
        ymax = min(4.2, max(1.05, float(np.nanmax([true_norm.max(), raw_norm.max(), pred_norm.max()])) * 1.08))
        ax.set_ylim(-0.05, ymax)
        ax.set_title(f"{run_id} RUL trajectory", fontsize=11)
        ax.set_xlabel("Life fraction")
        ax.set_ylabel("Normalized RUL")
        ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if run_id in metric_by_run.index:
            row = metric_by_run.loc[run_id]
            ax.text(
                0.03,
                0.07,
                f"score {row['score']:.3f} | nMAE {row['normalized_mae']:.3f}",
                transform=ax.transAxes,
                color=COLORS["ink"],
                fontsize=8.5,
                fontweight="bold",
            )
        if idx == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.suptitle("PRONOSTIA RUL Trajectories", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.text(
        0.02,
        0.012,
        "Normalized RUL is used because PRONOSTIA validation bearings have different run lengths.",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    save_figure(fig, "result_11_pronostia_rul_trajectories")


def figure_pronostia_error_over_life(pron: tuple[pd.DataFrame, pd.DataFrame, dict] | None) -> None:
    if pron is None:
        return
    pred, _metrics, _summary = pron
    runs = sorted(pred["run_id"].unique(), key=run_sort_key)
    ncols = 3
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 10.8), sharex=True, sharey=False)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(runs):
            ax.axis("off")
            continue
        run_id = runs[idx]
        run = pred[pred["run_id"] == run_id].sort_values("life_fraction")
        if "normalized_abs_error" in run:
            error = run["normalized_abs_error"].to_numpy(dtype=float)
        else:
            error = run["abs_error_s"].to_numpy(dtype=float) / np.maximum(run["run_end_s"].to_numpy(dtype=float), 1e-9)
        color = np.where(run["over_predicted"].to_numpy(bool), COLORS["rust"], COLORS["teal"])
        ax.scatter(run["life_fraction"], error, c=color, s=14, alpha=0.82, edgecolors="none")
        smooth = pd.Series(error).rolling(13, min_periods=1, center=True).median()
        ax.plot(run["life_fraction"], smooth, color=COLORS["ink"], lw=1.5, label="rolling median")
        ymax = min(4.2, max(0.55, float(np.nanmax(error)) * 1.12))
        ax.set_ylim(-0.03, ymax)
        ax.set_title(f"{run_id} normalized error over life", fontsize=11)
        ax.set_xlabel("Life fraction")
        ax.set_ylabel("Normalized absolute error")
        ax.grid(True, color=COLORS["grid"], lw=0.5, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.03, 0.90, "rust = overprediction", transform=ax.transAxes, color=COLORS["muted"], fontsize=7.5)

    fig.suptitle("PRONOSTIA Error Distribution Over Bearing Life", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    fig.text(
        0.02,
        0.012,
        "Error is normalized by each bearing run length to compare bearings with different lifetimes.",
        color=COLORS["muted"],
        fontsize=9.5,
    )
    save_figure(fig, "result_12_pronostia_error_over_life")


def figure_poster_result_panel(lobo_pred: pd.DataFrame, lobo_metrics: pd.DataFrame, lobo_summary: dict, tail_summary: dict) -> None:
    metric_by_run = lobo_metrics.set_index("run_id")
    main_run = lobo_pred[lobo_pred["run_id"] == "Train3"].sort_values("mid_time_s")
    hard_run = lobo_pred[lobo_pred["run_id"] == "Train4"].sort_values("mid_time_s")

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.35, 1.35, 1.0], height_ratios=[1.0, 0.48], figure=fig, hspace=0.40, wspace=0.30)
    ax_main = fig.add_subplot(gs[0, :2])
    ax_hard = fig.add_subplot(gs[0, 2])
    ax_cards = fig.add_subplot(gs[1, :])

    plot_rul(ax_main, main_run, title="Representative result: Train3", show_raw=True, show_legend=True)
    row = metric_by_run.loc["Train3"]
    ax_main.text(0.03, 0.08, f"Train3 score {row['score']:.3f}, MAE {row['mae_s'] / 1000:.2f} ks", transform=ax_main.transAxes, color=COLORS["ink"], fontweight="bold")

    plot_rul(ax_hard, hard_run, title="Hard case: Train4", show_raw=False, show_legend=False)
    row = metric_by_run.loc["Train4"]
    ax_hard.text(0.06, 0.08, f"score {row['score']:.3f}\nMAE {row['mae_s'] / 1000:.2f} ks", transform=ax_hard.transAxes, color=COLORS["muted"], fontsize=9)

    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    metric_card(ax_cards, 0.00, 0.10, 0.24, 0.78, "Overall LOBO", [f"score {lobo_summary['score']:.3f}", f"MAE {lobo_summary['mae_s'] / 1000:.2f} ks", f"overprediction {100*lobo_summary['over_prediction_rate']:.1f}%"], COLORS["teal"])
    metric_card(ax_cards, 0.255, 0.10, 0.24, 0.78, "Tail validation", [f"score {tail_summary['score']:.3f}", f"MAE {tail_summary['mae_s'] / 1000:.2f} ks", f"overprediction {100*tail_summary['over_prediction_rate']:.1f}%"], COLORS["navy"])
    metric_card(ax_cards, 0.510, 0.10, 0.24, 0.78, "Best run", ["Train1", f"score {metric_by_run.loc['Train1', 'score']:.3f}", f"MAE {metric_by_run.loc['Train1', 'mae_s'] / 1000:.2f} ks"], COLORS["gold"])
    metric_card(ax_cards, 0.765, 0.10, 0.24, 0.78, "Takeaway", ["tracks RUL trend", "stabilizes EOL", "hard case remains"], COLORS["rust"])

    fig.suptitle("RUL Prediction Result", x=0.02, y=1.01, ha="left", fontsize=19, fontweight="bold")
    save_figure(fig, "result_10_poster_result_panel")


def write_result_tables(lobo_summary: dict, tail_summary: dict, pron_summary: dict | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        {"dataset": "KSPHM", "protocol": "LOBO + causal EOL q0.90", **lobo_summary},
        {"dataset": "KSPHM", "protocol": "Tail + causal EOL q0.90", **tail_summary},
    ]
    if pron_summary:
        rows.append({"dataset": "PRONOSTIA", "protocol": "External first-pass", **pron_summary})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "result_metrics_summary.csv", index=False)

    figures = [
        ("result_01_overall_scorecards", "Overall result cards and LOBO per-run bars."),
        ("result_02_kspm_lobo_rul_trajectories", "All KSPHM LOBO RUL trajectories."),
        ("result_03_kspm_lobo_eol_convergence", "Predicted EOL convergence for all KSPHM runs."),
        ("result_04_kspm_lobo_error_over_life", "Absolute error distribution over bearing life."),
        ("result_05_tail_validation_rul_trajectories", "Late-life tail validation trajectories."),
        ("result_06_lobo_vs_tail_metric_bars", "LOBO vs tail score, MAE, RMSE, and overprediction."),
        ("result_07_predicted_vs_true_scatter", "Predicted RUL vs true RUL scatter."),
        ("result_08_ablation_comparison", "Available postprocessing and validation variants."),
        ("result_09_pronostia_external_validation", "External PRONOSTIA domain-shift check."),
        ("result_10_poster_result_panel", "Compact poster result panel."),
        ("result_11_pronostia_rul_trajectories", "All PRONOSTIA normalized RUL trajectories."),
        ("result_12_pronostia_error_over_life", "PRONOSTIA normalized error distribution over bearing life."),
    ]
    lines = ["# Result Figure Index", ""]
    for stem, description in figures:
        png = OUT_DIR / f"{stem}.png"
        if png.exists():
            lines.append(f"- `{stem}.png` / `{stem}.pdf`: {description}")
    lines.append("")
    lines.append("Recommended poster picks: `result_10_poster_result_panel`, `result_06_lobo_vs_tail_metric_bars`, and optionally `result_09_pronostia_external_validation`.")
    (OUT_DIR / "result_figure_index.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    set_style()
    lobo_pred, lobo_metrics, lobo_summary = load_kspm_lobo()
    tail_pred, tail_metrics, tail_summary = load_kspm_tail()
    pron = load_pronostia()
    pron_summary = pron[2] if pron else None

    figure_overall_scorecards(lobo_metrics, lobo_summary, tail_summary, pron_summary)
    figure_kspm_lobo_rul(lobo_pred, lobo_metrics)
    figure_kspm_lobo_eol(lobo_pred)
    figure_kspm_lobo_error(lobo_pred)
    figure_tail_rul(tail_pred, tail_metrics)
    figure_lobo_tail_metric_bars(lobo_metrics, tail_metrics)
    figure_predicted_vs_true(lobo_pred, tail_pred)
    figure_ablation_comparison()
    figure_pronostia_external(pron)
    figure_pronostia_rul_trajectories(pron)
    figure_pronostia_error_over_life(pron)
    figure_poster_result_panel(lobo_pred, lobo_metrics, lobo_summary, tail_summary)
    write_result_tables(lobo_summary, tail_summary, pron_summary)
    print(f"wrote result figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
