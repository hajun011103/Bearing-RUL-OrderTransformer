#!/usr/bin/env python
"""Regenerate result figures from the HONEST, leak-free runs.

Reads the outputs of ``scripts/run_lobo.py`` (nested leave-one-bearing-out) and
``scripts/run_tail_validation.py`` and produces a small, consistent, polished set
of result figures that reflect the honest scores (LOBO ~0.462), not the earlier
leaky 0.670. The poster is the as-submitted artifact and is not regenerated here.
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
import numpy as np
import pandas as pd

from phm_pipeline.losses import official_score_numpy

OUT_DIR = ROOT / "figures" / "results"
LOBO_DIR = ROOT / "artifacts" / "runs" / "lobo_order_domain_nested"
TAIL_DIR = ROOT / "artifacts" / "runs" / "per_run_tail_nested"

PREVIOUSLY_REPORTED_LOBO = 0.670

# Cohesive palette.
INK = "#101828"
SLATE = "#475467"
FAINT = "#98A2B3"
GRID = "#EAECF0"
TEAL = "#0E9384"
CORAL = "#F97066"
AMBER = "#F79009"
NAVY = "#1D4E89"
GRAY = "#C4CAD4"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.edgecolor": SLATE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": SLATE,
            "axes.labelsize": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlepad": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": 9.5,
        }
    )


def clean(ax, *, y_grid_only: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    if y_grid_only:
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")


def save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def hours(values) -> np.ndarray:
    return np.asarray(values, dtype=float) / 3600.0


def run_key(run_id: str) -> int:
    digits = "".join(ch for ch in str(run_id) if ch.isdigit())
    return int(digits) if digits else 10_000


def badge(ax, text: str, *, loc=(0.035, 0.06)) -> None:
    ax.text(
        loc[0], loc[1], text, transform=ax.transAxes, fontsize=9.5, color=INK,
        va="bottom", ha="left",
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.4", "linewidth": 1.0},
    )


def load_lobo() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    honest = pd.read_csv(LOBO_DIR / "oof_predictions_honest.csv")
    raw = pd.read_csv(LOBO_DIR / "oof_predictions_raw.csv")[["run_id", "segment_id", "predicted_rul_s"]]
    raw = raw.rename(columns={"predicted_rul_s": "raw_predicted_rul_s"})
    pred = honest.merge(raw, on=["run_id", "segment_id"], how="left")
    actual = pred["rul_s"].to_numpy(dtype=float)
    p = pred["predicted_rul_s"].to_numpy(dtype=float)
    pred["score"] = official_score_numpy(actual, p)
    pred["abs_error_s"] = np.abs(p - actual)
    pred["over_predicted"] = p > actual
    rows = []
    for run_id, g in pred.groupby("run_id"):
        rows.append(
            {"run_id": run_id, "score": float(g["score"].mean()),
             "mae_s": float(g["abs_error_s"].mean()),
             "over_prediction_rate": float(g["over_predicted"].mean())}
        )
    per_run = pd.DataFrame(rows).sort_values("run_id", key=lambda s: s.map(run_key)).reset_index(drop=True)
    summary = json.loads((LOBO_DIR / "nested_lobo_summary.json").read_text())
    return pred, per_run, summary


def fig_trajectories(pred: pd.DataFrame, per_run: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)
    metric = per_run.set_index("run_id")
    handles = None
    for ax, run_id in zip(axes.flat, runs):
        g = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        x = hours(g["mid_time_s"])
        ax.fill_between(x, 0, hours(g["predicted_rul_s"]), color=TEAL, alpha=0.08, lw=0)
        h_raw, = ax.plot(x, hours(g["raw_predicted_rul_s"]), color=CORAL, lw=1.1, alpha=0.85, label="Raw Transformer")
        h_sm, = ax.plot(x, hours(g["predicted_rul_s"]), color=TEAL, lw=2.6, solid_capstyle="round", label="Causal EOL smoothed")
        h_tr, = ax.plot(x, hours(g["rul_s"]), color=INK, lw=1.7, ls=(0, (5, 3)), label="True RUL")
        handles = [h_tr, h_raw, h_sm]
        ax.set_title(f"Bearing {run_id}")
        ax.set_xlabel("Observed time (h)")
        ax.set_ylabel("RUL (h)")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.01)
        clean(ax)
        row = metric.loc[run_id]
        badge(ax, f"score {row['score']:.3f}   ·   MAE {row['mae_s'] / 1000:.1f} ks")
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975),
               handlelength=1.9, columnspacing=1.8)
    save(fig, "result_lobo_rul_trajectories")


def fig_per_bearing(per_run: pd.DataFrame, summary: dict) -> None:
    fig, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(11.6, 4.0))
    fig.subplots_adjust(wspace=0.24, top=0.86, bottom=0.14)
    x = np.arange(len(per_run))
    labels = [f"Bearing\n{r}" for r in per_run["run_id"]]

    ax_s.axhline(summary["honest"]["score"], color=SLATE, lw=1.1, ls=(0, (4, 3)))
    ax_s.text(2.0, summary["honest"]["score"] + 0.016,
              f"overall {summary['honest']['score']:.3f}", ha="center", va="bottom", color=SLATE, fontsize=9)
    ax_s.bar(x, per_run["score"], width=0.6, color=TEAL, zorder=3)
    ax_s.set_ylim(0, 0.82)
    ax_s.set_ylabel("Official score  (higher is better)")
    ax_s.set_title("LOBO score by held-out bearing")
    for i, v in enumerate(per_run["score"]):
        ax_s.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="medium")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(labels)
    clean(ax_s)

    ax_m.bar(x, per_run["mae_s"] / 1000.0, width=0.6, color=AMBER, zorder=3)
    ax_m.set_ylabel("MAE (ks)  (lower is better)")
    ax_m.set_title("Absolute RUL error by held-out bearing")
    for i, v in enumerate(per_run["mae_s"] / 1000.0):
        ax_m.text(i, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="medium")
    ax_m.set_ylim(top=float((per_run["mae_s"] / 1000.0).max()) * 1.2)
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(labels)
    clean(ax_m)
    save(fig, "result_lobo_by_bearing")


def fig_honest_vs_reported(summary: dict) -> None:
    items = [
        ("Previously\nreported", PREVIOUSLY_REPORTED_LOBO, GRAY, "two stacked leaks"),
        ("Raw\n(honest models)", summary["raw"]["score"], NAVY, "no smoothing"),
        ("Honest\n(leak-free)", summary["honest"]["score"], TEAL, "headline"),
        ("Oracle\nceiling", summary["oracle"]["score"], AMBER, "tuned on test"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    fig.subplots_adjust(bottom=0.2, top=0.9)
    x = np.arange(len(items))
    values = [v for _, v, _, _ in items]
    colors = [c for _, _, c, _ in items]
    bars = ax.bar(x, values, width=0.62, color=colors, zorder=3)
    # Emphasize the honest bar.
    bars[2].set_edgecolor(INK)
    bars[2].set_linewidth(1.6)
    ax.set_ylim(0, 0.76)
    ax.set_ylabel("LOBO official score")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _, _, _ in items])
    for i, (_, v, _, sub) in enumerate(items):
        ax.text(i, v + 0.014, f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.text(i, 0.02, sub, ha="center", va="bottom", fontsize=8.2, color=SLATE)
    # Drop annotation from previously-reported to honest.
    drop = PREVIOUSLY_REPORTED_LOBO - summary["honest"]["score"]
    ax.annotate(
        f"−{drop:.3f}\nremoving leakage",
        xy=(1.72, 0.40), xytext=(0.7, 0.60),
        ha="center", fontsize=9.5, color=CORAL, fontweight="semibold",
        arrowprops={"arrowstyle": "-|>", "color": CORAL, "lw": 1.6,
                    "connectionstyle": "arc3,rad=-0.25"},
    )
    clean(ax)
    save(fig, "result_honest_vs_reported")


def fig_eol(pred: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)
    handles = None
    for ax, run_id in zip(axes.flat, runs):
        g = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        x = hours(g["mid_time_s"])
        actual_eol = float(np.nanmedian(hours(g["mid_time_s"] + g["rul_s"])))
        h_a = ax.axhline(actual_eol, color=INK, lw=1.7, ls=(0, (5, 3)), label="Actual EOL")
        h_r, = ax.plot(x, hours(g["mid_time_s"] + g["raw_predicted_rul_s"]), color=CORAL, lw=1.1, alpha=0.85, label="Raw EOL")
        h_s, = ax.plot(x, hours(g["mid_time_s"] + g["predicted_rul_s"]), color=TEAL, lw=2.4, solid_capstyle="round", label="Smoothed EOL")
        handles = [h_a, h_r, h_s]
        ax.set_title(f"Bearing {run_id}")
        ax.set_xlabel("Observed time (h)")
        ax.set_ylabel("Predicted EOL (h)")
        ax.margins(x=0.01)
        clean(ax)
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975), handlelength=1.9, columnspacing=1.8)
    save(fig, "result_lobo_eol_convergence")


def fig_scatter(pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    true_h = hours(pred["rul_s"])
    pred_h = hours(pred["predicted_rul_s"])
    limit = float(max(np.nanmax(true_h), np.nanmax(pred_h))) * 1.05
    ax.plot([0, limit], [0, limit], color=SLATE, lw=1.4, ls=(0, (5, 3)), label="perfect prediction")
    palette = {r: c for r, c in zip(sorted(pred["run_id"].unique(), key=run_key), [TEAL, NAVY, AMBER, CORAL])}
    for run_id, g in pred.groupby("run_id"):
        ax.scatter(hours(g["rul_s"]), hours(g["predicted_rul_s"]), s=26, alpha=0.65,
                   color=palette.get(run_id, SLATE), edgecolors="none", label=f"Bearing {run_id}")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("True RUL (h)")
    ax.set_ylabel("Predicted RUL (h)")
    ax.set_aspect("equal")
    clean(ax, y_grid_only=False)
    ax.grid(True, color=GRID)
    ax.legend(loc="upper left")
    save(fig, "result_lobo_predicted_vs_true")


def fig_error_over_life(pred: pd.DataFrame) -> None:
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True)
    fig.subplots_adjust(top=0.92, hspace=0.34, wspace=0.22)
    for ax, run_id in zip(axes.flat, runs):
        g = pred[pred["run_id"] == run_id].sort_values("life_fraction")
        over = g["over_predicted"].to_numpy(bool)
        ax.scatter(g["life_fraction"][~over], hours(g["abs_error_s"])[~over], c=TEAL, s=20, alpha=0.8, edgecolors="none")
        ax.scatter(g["life_fraction"][over], hours(g["abs_error_s"])[over], c=CORAL, s=20, alpha=0.85, edgecolors="none")
        smooth = pd.Series(hours(g["abs_error_s"])).rolling(9, min_periods=1, center=True).median()
        ax.plot(g["life_fraction"], smooth, color=INK, lw=1.6)
        ax.set_title(f"Bearing {run_id}")
        ax.set_xlabel("Life fraction")
        ax.set_ylabel("Absolute error (h)")
        ax.margins(x=0.02)
        clean(ax)
        ax.text(0.035, 0.9, "coral = overprediction", transform=ax.transAxes, color=CORAL, fontsize=8.5)
    save(fig, "result_lobo_error_over_life")


def fig_tail_limitation() -> None:
    tail_pred_path = TAIL_DIR / "tail_predictions_with_actual.csv"
    tail_summary_path = TAIL_DIR / "tail_summary.json"
    if not (tail_pred_path.exists() and tail_summary_path.exists()):
        return
    pred = pd.read_csv(tail_pred_path)
    runs = sorted(pred["run_id"].unique(), key=run_key)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.1 * len(runs), 3.6))
    if len(runs) == 1:
        axes = [axes]
    handles = None
    for ax, run_id in zip(axes, runs):
        g = pred[pred["run_id"] == run_id].sort_values("mid_time_s")
        x = hours(g["mid_time_s"])
        h_t, = ax.plot(x, hours(g["rul_s"]), color=INK, lw=1.7, ls=(0, (5, 3)), label="True RUL")
        h_p, = ax.plot(x, hours(g["predicted_rul_s"]), color=CORAL, lw=2.3, solid_capstyle="round", label="Predicted (test tail)")
        handles = [h_t, h_p]
        ax.set_title(f"Bearing {run_id}")
        ax.set_xlabel("Observed time (h)")
        ax.set_ylabel("RUL (h)")
        ax.margins(x=0.02)
        clean(ax)
    fig.subplots_adjust(top=0.82, wspace=0.3)
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99), handlelength=1.9)
    save(fig, "result_tail_limitation")


def write_index(summary: dict) -> None:
    lines = [
        "# Result figure index (honest runs)",
        "",
        "Generated by `scripts/make_result_figures.py` from the leak-free runs.",
        f"Honest LOBO score **{summary['honest']['score']:.3f}** "
        f"(raw {summary['raw']['score']:.3f}, oracle {summary['oracle']['score']:.3f}; "
        f"previously reported {PREVIOUSLY_REPORTED_LOBO:.3f} with two test-set leaks).",
        "",
        "- `result_honest_vs_reported`: the leakage correction (0.670 -> 0.462).",
        "- `result_lobo_rul_trajectories`: raw vs causal-smoothed vs true RUL per bearing.",
        "- `result_lobo_by_bearing`: per-bearing score and MAE.",
        "- `result_lobo_eol_convergence`: predicted end-of-life convergence per bearing.",
        "- `result_lobo_predicted_vs_true`: predicted vs true RUL scatter.",
        "- `result_lobo_error_over_life`: absolute error over life fraction per bearing.",
        "- `result_tail_limitation`: honest late-life tail negative result.",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result_figure_index.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    set_style()
    pred, per_run, summary = load_lobo()
    fig_honest_vs_reported(summary)
    fig_trajectories(pred, per_run)
    fig_per_bearing(per_run, summary)
    fig_eol(pred)
    fig_scatter(pred)
    fig_error_over_life(pred)
    fig_tail_limitation()
    write_index(summary)
    print(f"wrote honest result figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
