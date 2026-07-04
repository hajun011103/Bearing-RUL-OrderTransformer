"""Tests for the leak-free evaluation logic that makes the reported numbers honest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from phm_pipeline.losses import official_score_numpy  # noqa: E402
from phm_pipeline.training import apply_temporal_postprocess, select_temporal_smoothing  # noqa: E402
import run_tail_validation  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["A"] * 5,
            "mid_time_s": [0.0, 600.0, 1200.0, 1800.0, 2400.0],
            "rul_s": [3000.0, 2400.0, 1800.0, 1200.0, 600.0],
            "predicted_rul_s": [3300.0, 2100.0, 2000.0, 1000.0, 900.0],
        }
    )


def _score_of(frame: pd.DataFrame, method, quantile, blend) -> float:
    if method == "none":
        pred = frame["predicted_rul_s"].to_numpy(dtype=float)
    else:
        pred = apply_temporal_postprocess(
            frame,
            method=method,
            group_column="run_id",
            time_column="mid_time_s",
            prediction_column="predicted_rul_s",
            eol_quantile=quantile,
            blend=blend,
            floor_s=1.0,
        ).to_numpy(dtype=float)
    return float(np.mean(official_score_numpy(frame["rul_s"].to_numpy(dtype=float), pred)))


def test_select_temporal_smoothing_returns_true_argmax() -> None:
    frame = _frame()
    candidates = [("none", None, None), ("eol_quantile", 0.5, 1.0), ("eol_quantile", 0.9, 1.0)]
    scores = {cand: _score_of(frame, *cand) for cand in candidates}
    best_manual = max(scores, key=scores.get)

    choice, score = select_temporal_smoothing(
        frame, candidates,
        group_column="run_id", time_column="mid_time_s", target_column="rul_s", floor_s=1.0,
    )
    assert (choice["method"], choice["quantile"], choice["blend"]) == best_manual
    assert abs(score - scores[best_manual]) < 1e-9


def test_select_temporal_smoothing_none_is_identity() -> None:
    frame = _frame()
    choice, score = select_temporal_smoothing(
        frame, [("none", None, None)],
        group_column="run_id", time_column="mid_time_s", target_column="rul_s", floor_s=1.0,
    )
    assert choice["method"] == "none"
    expected = float(
        np.mean(official_score_numpy(frame["rul_s"].to_numpy(dtype=float),
                                     frame["predicted_rul_s"].to_numpy(dtype=float)))
    )
    assert abs(score - expected) < 1e-9


def test_select_temporal_smoothing_rejects_empty_candidates() -> None:
    try:
        select_temporal_smoothing(
            _frame(), [],
            group_column="run_id", time_column="mid_time_s", target_column="rul_s",
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty candidate list")


def test_tail_split_partitions_each_run_with_no_leakage() -> None:
    args = argparse.Namespace(group_column="run_id", time_column="mid_time_s", context_length=4, min_context=2)
    n = 40
    df = pd.concat(
        [
            pd.DataFrame(
                {"run_id": [r] * n, "mid_time_s": np.arange(n) * 600.0, "segment_id": np.arange(n)}
            )
            for r in ("A", "B")
        ],
        ignore_index=True,
    )
    head = run_tail_validation.chronological_head(df, args, keep_fraction=0.75)
    tail = run_tail_validation.chronological_tail(df, args, tail_fraction=0.25)

    for run in ("A", "B"):
        early = set(head[head["run_id"] == run]["segment_id"])
        late = set(tail[tail["run_id"] == run]["segment_id"])
        assert early.isdisjoint(late)          # no row is both trained on and tested
        assert early | late == set(range(n))   # together they cover the run
        assert max(early) < min(late)          # the tail is strictly the late life
        assert len(late) == 10                 # 25% of 40
