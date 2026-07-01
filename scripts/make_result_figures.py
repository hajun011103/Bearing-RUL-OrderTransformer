#!/usr/bin/env python
"""Regenerate result figures from the HONEST, leak-free runs.

Reads the outputs of ``scripts/run_lobo.py`` (nested leave-one-bearing-out) and
``scripts/run_tail_validation.py`` and produces a consistent set of result
figures that reflect the honest scores (LOBO ~0.462), not the earlier leaky
0.670. The poster figures under ``figures/poster/`` are the as-submitted artifact
and are intentionally NOT regenerated here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phmkorea_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from phm_pipeline.losses import official_score_numpy

OUT_DIR = ROOT / "figures" / "results"
LOBO_DIR = ROOT / "artifacts" / "runs" / "lobo_order_domain_nested"
TAIL_DIR = ROOT / "artifacts" / "runs" / "per_run_tail_nested"

# Previously reported (double-leak) LOBO score, shown only for transparency.
PREVIOUSLY_REPORTED_LOBO = 0.670

COLORS = {
    "ink": "#202124",
    "muted": "#667085",
    "grid": "#D0D5DD",
    "light": "#EEF2F6",
    "navy": "#284B63",
    "teal": "#2A9D8F",
    "rust": "#C65D3A",
    "gold": "#E9A93A",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def hours(values) -> np.ndarray:
    return np.asarray(values, dtype=float) / 3600.0


def run_key(run_id: str) -> int:
    digits = "".join(ch for ch in str(run_id) if ch.isdigit())
    return int(digits) if digits else 10_000


def load_lobo() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    honest = pd.read_csv(LOBO_DIR / "oof_predictions_honest.csv")
    raw = pd.read_csv(LOBO_DIR / "oof_predictions_raw.csv")[["run_id", "segment_id", "predicted_rul_s"]]
    raw = raw.rename(columns={"predicted_rul_s": "raw_predicted_rul_s"})
    pred = honest.merge(raw, on=["run_id", "segment_id"], how="left")
    pred = _add_derived(pred)
    per_run = _per_run_metrics(pred)
    summary = json.loads((LOBO_DIR / "nested_lobo_summary.json").read_text())
    return pred, per_run, summary


def _add_derived(pred: pd.DataFrame) -> pd.DataFrame:
    actual = pred["rul_s"].to_numpy(dtype=float)
    p = pred["predicted_rul_s"].to_numpy(dtype=float)
    pred["score"] = official_score_numpy(actual, p)
    pred["abs_error_s"] = np.abs(p - actual)
    pred["over_predicted"] = p > actual
    pred["er_percent"] = 100.0 * (actual - p) / np.maximum(actual, 1e-6)
    return pred


def _per_run_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, g in pred.groupby("run_id"):
        rows.append(
            {
                "run_id": run_id,
                "rows": len(g),
                "score": float(g["score"].mean()),
                "mae_s": float(g["abs_error_s"].mean()),
                "over_prediction_rate": float(g["over_predicted"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("run_id", key=lambda s: s.map(run_key)).reset_index(drop=True)


def metric_card(ax, x, y, w, h, title, lines, color) -> None:
    header_h = 0.24 * h
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color, linewidth=1.8))
    ax.add_patch(Rectangle((x, y + h - header_h), w, header_h, facecolor=color, edgecolor="none"))
    ax.text(x + 0.03, y + h - header_h / 2, title, ha="left", va="center", color="white", fontweight="bold")
    for idx, line in enumerate(lines):
        ax.text(x + 0.03, y + h - header_h - 0.12 - idx * 0.19, line, ha="left", va="center", color=COLORS["ink"])


def fig_scorecards(per_run: pd.DataFrame, summary: dict) -> None:
    raw, honest, oracle = summary["raw"], summary["honest"], summary["oracle"]
    fig = plt.figure(figsize=(12.6, 6.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.7, 1.0], figure=fig, hspace=0.45, wspace=0.24)
    ax_cards = fig.add_subplot(gs[0, :])
    ax_score = fig.add_subplot(gs[1, 0])
    ax_mae = fig.add_subplot(gs[1, 1])
    ax_cards.axis("off")
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)

    metric_card(ax_cards, 0.00, 0.08, 0.31, 0.82, "Honest LOBO",
                [f"score {honest['score']:.3f}", f"MAE {honest['mae_s'] / 1000:.1f} ks",
                 f"overpred {100 * honest['over_prediction_rate']:.0f}%"], COLORS["teal"])
    metric_card(ax_cards, 0.345, 0.08, 0.31, 0.82, "Raw (no smoothing)",
                [f"score {raw['score']:.3f}", f"MAE {raw['mae_s'] / 1000:.1f} ks",
                 f"overpred {100 * raw['over_prediction_rate']:.0f}%"], COLORS["navy"])
    metric_card(ax_cards, 0.69, 0.08, 0.31, 0.82, "Oracle ceiling",
                [f"score {oracle['score']:.3f}", "smoothing tuned", "on test folds"], COLORS["gold"])

    x = np.arange(len(per_run))
    ax_score.bar(x, per_run["score"], color=COLORS["teal"])
    ax_score.set_xticks(x)
    ax_score.set_xticklabels(per_run["run_id"])
    ax_score.set_ylim(0, 0.85)
    ax_score.set_ylabel("Score (higher is better)")
    ax_score.set_title("Honest LOBO score by held-out bearing")
    ax_score.grid(True, axis="y", color=COLORS["grid"], lw=0.5)
    ax_score.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(per_run["score"]):
        ax_score.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax_mae.bar(x, per_run["mae_s"] / 1000.0, color=COLORS["gold"])
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(per_run["run_id"])
    ax_mae.set_ylabel("MAE (ks, lower is better)")
    ax_mae.set_title("Absolute RUL error by held-out bearing")
    ax_mae.grid(True, axis="y", color=COLORS["grid"], lw=0.5)
    ax_mae.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(per_run["mae_s"] / 1000.0):
        ax_mae.text(i, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Honest leave-one-bearing-out result", x=0.02, y=1.00, ha="left", fontsize=17, fontweight="bold")
    save_figure(fig, "result_lobo_scorecards")


def fig_trajectories(pred: pd.DataFrame, per_run: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0))
    metric_by_run = per_run.set_index("run_id")
    for ax, run_id in zip(axes.flat, runs):
        g = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        x = hours(g["mid_time_s"])
        ax.plot(x, hours(g["rul_s"]), color=COLORS["ink"], lw=2.0, ls="--", label="True RUL")
        ax.plot(x, hours(g["raw_predicted_rul_s"]), color=COLORS["rust"], lw=1.1, alpha=0.6, label="Raw Transformer")
        ax.plot(x, hours(g["predicted_rul_s"]), color=COLORS["teal"], lw=2.3, label="Causal EOL smoothed")
        ax.set_title(f"{run_id} RUL trajectory")
        ax.set_xlabel("Observed time (h)")
        ax.set_ylabel("RUL (h)")
        ax.grid(True, color=COLORS["grid"], lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        row = metric_by_run.loc[run_id]
        ax.text(0.03, 0.07, f"score {row['score']:.3f} | MAE {row['mae_s'] / 1000:.1f} ks",
                transform=ax.transAxes, fontweight="bold", color=COLORS["ink"])
        if run_id == runs[0]:
            ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Honest LOBO RUL trajectories", x=0.02, y=1.01, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_lobo_rul_trajectories")


def fig_eol(pred: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0))
    for ax, run_id in zip(axes.flat, runs):
        g = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        x = hours(g["mid_time_s"])
        actual_eol = float(np.nanmedian(hours(g["mid_time_s"] + g["rul_s"])))
        ax.axhline(actual_eol, color=COLORS["ink"], lw=1.8, ls="--", label="Actual EOL")
        ax.plot(x, hours(g["mid_time_s"] + g["raw_predicted_rul_s"]), color=COLORS["rust"], lw=1.1, alpha=0.6, label="Raw EOL")
        ax.plot(x, hours(g["mid_time_s"] + g["predicted_rul_s"]), color=COLORS["teal"], lw=2.2, label="Smoothed EOL")
        ax.set_title(f"{run_id} predicted EOL convergence")
        ax.set_xlabel("Observed time (h)")
        ax.set_ylabel("Predicted EOL time (h)")
        ax.grid(True, color=COLORS["grid"], lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if run_id == runs[0]:
            ax.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Honest LOBO predicted end-of-life convergence", x=0.02, y=1.01, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_lobo_eol_convergence")


def fig_scatter(pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    true_h = hours(pred["rul_s"])
    pred_h = hours(pred["predicted_rul_s"])
    limit = float(max(np.nanmax(true_h), np.nanmax(pred_h))) * 1.05
    ax.plot([0, limit], [0, limit], color=COLORS["ink"], lw=1.6, ls="--", label="perfect prediction")
    for run_id, g in pred.groupby("run_id"):
        ax.scatter(hours(g["rul_s"]), hours(g["predicted_rul_s"]), s=22, alpha=0.6, edgecolors="none", label=run_id)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("True RUL (h)")
    ax.set_ylabel("Predicted RUL (h)")
    ax.set_title("Honest LOBO: predicted vs true RUL")
    ax.grid(True, color=COLORS["grid"], lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save_figure(fig, "result_lobo_predicted_vs_true")


def fig_error_over_life(pred: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), sharex=True)
    for ax, run_id in zip(axes.flat, runs):
        g = pred[pred["run_id"] == run_id].sort_values("life_fraction")
        color = np.where(g["over_predicted"].to_numpy(bool), COLORS["rust"], COLORS["teal"])
        ax.scatter(g["life_fraction"], hours(g["abs_error_s"]), c=color, s=20, alpha=0.8, edgecolors="none")
        smooth = pd.Series(hours(g["abs_error_s"])).rolling(9, min_periods=1, center=True).median()
        ax.plot(g["life_fraction"], smooth, color=COLORS["ink"], lw=1.6)
        ax.set_title(f"{run_id} absolute error over life")
        ax.set_xlabel("Life fraction")
        ax.set_ylabel("Absolute RUL error (h)")
        ax.grid(True, color=COLORS["grid"], lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.03, 0.9, "rust = overprediction", transform=ax.transAxes, color=COLORS["muted"], fontsize=9)
    fig.suptitle("Honest LOBO error distribution over bearing life", x=0.02, y=1.01, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "result_lobo_error_over_life")


def fig_honest_vs_reported(summary: dict) -> None:
    labels = ["Previously\nreported\n(2 leaks)", "Raw\n(honest models)", "Honest\n(leak-free)", "Oracle\nceiling"]
    values = [PREVIOUSLY_REPORTED_LOBO, summary["raw"]["score"], summary["honest"]["score"], summary["oracle"]["score"]]
    colors = [COLORS["muted"], COLORS["navy"], COLORS["teal"], COLORS["gold"]]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("LOBO official score")
    ax.set_title("Removing test-set leakage: 0.670 → 0.462")
    ax.grid(True, axis="y", color=COLORS["grid"], lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(values):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.text(0.0, -0.16,
            "Leaks removed: (1) smoothing tuned on test folds, (2) early stopping on the held-out bearing.",
            transform=ax.transAxes, color=COLORS["muted"], fontsize=9)
    save_figure(fig, "result_honest_vs_reported")


def fig_tail_limitation() -> None:
    tail_pred_path = TAIL_DIR / "tail_predictions_with_actual.csv"
    tail_summary_path = TAIL_DIR / "tail_summary.json"
    if not (tail_pred_path.exists() and tail_summary_path.exists()):
        return
    pred = pd.read_csv(tail_pred_path)
    summary = json.loads(tail_summary_path.read_text())
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 4.3), sharey=False)
    if len(runs) == 1:
        axes = [axes]
    for ax, run_id in zip(axes, runs):
        g = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        x = hours(g["mid_time_s"])
        ax.plot(x, hours(g["rul_s"]), color=COLORS["ink"], lw=2.0, ls="--", label="True RUL")
        ax.plot(x, hours(g["predicted_rul_s"]), color=COLORS["rust"], lw=2.2, label="Predicted (test tail)")
        ax.set_title(run_id)
        ax.set_xlabel("Observed time (h)")
        ax.set_ylabel("RUL (h)")
        ax.grid(True, color=COLORS["grid"], lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if run_id == runs[0]:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
    honest = summary["honest"]
    fig.suptitle(
        f"Honest late-life tail: score {honest['score']:.3f}, {100 * honest['over_prediction_rate']:.0f}% over-predicted "
        "— a negative result (model never saw a near-death bearing)",
        x=0.02, y=1.03, ha="left", fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "result_tail_limitation")


def write_index(summary: dict) -> None:
    lines = [
        "# Result figure index (honest runs)",
        "",
        "Generated by `scripts/make_result_figures.py` from the leak-free runs.",
        f"Honest LOBO score: **{summary['honest']['score']:.3f}** "
        f"(raw {summary['raw']['score']:.3f}, oracle {summary['oracle']['score']:.3f}; "
        f"previously reported {PREVIOUSLY_REPORTED_LOBO:.3f} with two test-set leaks).",
        "",
        "- `result_lobo_scorecards`: overall cards + per-bearing score and MAE.",
        "- `result_lobo_rul_trajectories`: raw vs causal-smoothed vs true RUL per bearing.",
        "- `result_lobo_eol_convergence`: predicted end-of-life convergence per bearing.",
        "- `result_lobo_predicted_vs_true`: predicted vs true RUL scatter.",
        "- `result_lobo_error_over_life`: absolute error over life fraction per bearing.",
        "- `result_honest_vs_reported`: leakage correction (0.670 -> 0.462).",
        "- `result_tail_limitation`: honest late-life tail negative result.",
        "",
        "The poster figures under `figures/poster/` are the as-submitted artifact and are not regenerated.",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result_figure_index.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    set_style()
    pred, per_run, summary = load_lobo()
    fig_scorecards(per_run, summary)
    fig_trajectories(pred, per_run)
    fig_eol(pred)
    fig_scatter(pred)
    fig_error_over_life(pred)
    fig_honest_vs_reported(summary)
    fig_tail_limitation()
    write_index(summary)
    print(f"wrote honest result figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
