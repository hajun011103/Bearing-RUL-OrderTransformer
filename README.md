# PHM Korea 2026 — Order-Domain Transformer for Bearing RUL

> **Speed-robust bearing remaining-useful-life prediction: rotational order tracking, a time-gap-aware Transformer, and causal end-of-life smoothing — evaluated without test-set leakage.**

[![PHM Korea 2026](https://img.shields.io/badge/PHM_Korea-2026-1f6feb)](https://www.phm.or.kr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
[![CI](https://github.com/hajun011103/PHMKorea/actions/workflows/ci.yml/badge.svg)](https://github.com/hajun011103/PHMKorea/actions/workflows/ci.yml)

This repository contains the code, the honest leak-free evaluation, and the figures
for the PHM Korea 2026 poster **"Bearing Remaining Useful Life Prediction Using an
Order-Domain Transformer and Causal End-of-Life Smoothing"** (Jang, Jang, Kim, Lim
& Choi, SUNY Korea).

<p align="center">
  <img src="figures/results/pipeline_overview.png" width="100%"><br>
  <em>The method: 60 s vibration acquisitions under varying speed &rarr; order tracking (fault lines fixed in shaft order) &rarr; a time-gap-aware Transformer over the segment history &rarr; causal end-of-life quantile smoothing. Regenerate with <code>python scripts/make_overview_figure.py</code>.</em>
</p>

## Contents

- [Overview](#overview)
- [Why the order domain](#why-the-order-domain)
- [Results](#results)
- [Quickstart](#quickstart)
- [Data](#data)
- [Repository Layout](#repository-layout)
- [Poster & Abstract](#poster--abstract)
- [How to Cite](#how-to-cite)
- [License](#license)

## Overview

Predicting the remaining useful life (RUL) of a bearing from vibration is hard
under *varying operating speed* for two reasons: speed changes smear defect-related
spectral features, and segment-level predictions are temporally unstable. This
project addresses both with a compact, physically motivated pipeline:

1. **Order tracking** — each vibration acquisition is resampled onto the shaft-angle
   grid, so bearing-defect components (BPFO / BPFI / BSF / FTF and harmonics) stay at
   fixed *orders* regardless of RPM drift ([`order_spectrum`](src/phm_pipeline/features.py)).
2. **Time-gap-aware Transformer** — a Transformer encoder over segment-level
   order-domain features that encodes the true elapsed time and the acquisition gaps
   between segments, and predicts RUL causally ([`RULTransformer`](src/phm_pipeline/model.py)).
3. **Causal end-of-life (EOL) smoothing** — each RUL trajectory is post-processed with
   a causal running quantile of the *predicted* end-of-life, using only information up
   to the current time step, so it is safe for online use
   ([`apply_temporal_postprocess`](src/phm_pipeline/training.py)).

## Why the order domain

When the shaft speed changes during an acquisition, a fixed-Hz FFT smears each
bearing-defect line into a broad hump, because the defect's frequency moves with RPM.
Resampling onto shaft angle pins every defect to a fixed *order*, so the fault lines
stay sharp and comparable across segments recorded at different speeds.

<p align="center">
  <img src="figures/results/order_vs_time_demo.png" width="78%"><br>
  <em>Same synthetic signal with the shaft sweeping 1500 &rarr; 2400 rpm. Top: the fixed-Hz FFT smears the 1&times; shaft and BPFO (3.58&times;) lines into broad humps. Bottom: the order-domain spectrum keeps them as sharp peaks. Built with the repository's own <code>order_spectrum</code>.</em>
</p>

## Results

The headline evaluation is **leave-one-bearing-out (LOBO)** on the four PHM Korea
run-to-failure bearings, scored with the official competition score (higher is
better, max 1.0). Every hyper-parameter that touches a held-out bearing is chosen
*without looking at it*: models are early-stopped on a tail split of the training
bearings, and the EOL-smoothing hyper-parameters are selected by an **inner LOBO**
over the training bearings ([`scripts/run_lobo.py`](scripts/run_lobo.py)).

| Evaluation (KSPHM, 4 bearings) | Score ↑ | MAE (ks) ↓ | Leak-free |
|:--|--:|--:|:--:|
| LOBO — raw (no smoothing) | 0.441 | 16.9 | ✅ |
| **LOBO — honest (leak-free)** | **0.462** | 15.9 | ✅ |
| LOBO — oracle ceiling (smoothing tuned on test folds) | 0.466 | 16.5 | ❌ |
| _previously reported (two stacked leaks)_ | _0.670_ | — | ❌ |

> **On the honest number.** An earlier version reported **0.670**. That value
> stacked two optimistic leaks — the smoothing quantile was tuned on the test folds,
> *and* each model was early-stopped on the very bearing it was scored on. Removing
> both drops the score to **0.462**. We keep the honest number as the headline and
> report the oracle ceiling and the old value transparently. The full accounting is in
> [`docs/experiments.md`](docs/experiments.md).

<p align="center">
  <img src="figures/results/result_honest_vs_reported.png" width="62%"><br>
  <em>Removing test-set leakage: 0.670 &rarr; 0.462. Most of the drop is the early-stopping leak (raw score 0.555 &rarr; 0.441); the leak-free smoothing (0.462) is already within 0.004 of the oracle ceiling (0.466).</em>
</p>

<p align="center">
  <img src="figures/results/result_lobo_scorecards.png" width="100%"><br>
  <em>Honest LOBO by held-out bearing. With only four bearings the variance is high — from ~0.10 (Train3, poorly generalized) to ~0.67 (Train1).</em>
</p>

<p align="center">
  <img src="figures/results/result_lobo_rul_trajectories.png" width="100%"><br>
  <em>Honest LOBO RUL trajectories: raw Transformer (thin), causal-EOL-smoothed (thick), and true RUL (dashed). Smoothing stabilizes the trajectory without peeking into the future.</em>
</p>

> **Honest negative results.** A secondary *late-life tail* test — predict the last
> 25% of each bearing from its early life, with the scale and smoothing selected on an
> inner split — scores only **0.086** (100% over-predicted). A model that has never
> seen a near-death bearing systematically over-estimates remaining life, which is
> exactly why leave-one-**bearing**-out (the model sees complete lives of the *other*
> bearings) is the meaningful protocol. An external **PRONOSTIA / FEMTO** domain-shift
> check reaches ~0.38 and is a robustness probe, not a benchmark. See
> [`docs/experiments.md`](docs/experiments.md).

## Quickstart

Requires **Python 3.11**. The extracted order-domain feature table is committed, so
the modeling runs on CPU in minutes without the multi-GB raw signals.

```bash
# 1. Install
python -m pip install -e .          # or: pip install -r requirements.txt
#    (conda: conda env create -f environment.yml && conda activate phm-korea-rul)

# 2. Honest leave-one-bearing-out (writes artifacts/runs/lobo_order_domain_nested/)
python scripts/run_lobo.py

# 3. Honest late-life tail diagnostic
python scripts/run_tail_validation.py

# 4. Regenerate the figures
python scripts/make_overview_figure.py
python scripts/make_result_figures.py
```

`run_lobo.py` prints the raw / honest / oracle table and writes
`nested_lobo_summary.json` plus the out-of-fold predictions.

## Data

The raw bearing archives are **not** in this repository (each vibration ZIP is
multi-GB, and the challenge data is not redistributable). Only the small extracted
order-domain feature table under `artifacts/features/` is shipped. To rebuild it from
the raw data:

```bash
python scripts/extract_features.py \
  --data-root data/Train \
  --output artifacts/features/train_full.parquet

python scripts/make_order_domain_features.py \
  --source artifacts/features/train_full.parquet \
  --output artifacts/features/train_full_order_domain.parquet \
  --feature-mode order
```

**Where to get the data.** The PHM Korea challenge bearing data comes from the
**KIMM Data Platform** (participant/registration access — not redistributable), and
the external check uses the public **NASA FEMTO / PRONOSTIA** dataset. Exact,
fetch-verified download links and access terms — plus public variable-speed
substitutes (XJTU-SY, U. Ottawa, KAIST) — are in
[`docs/data.md`](docs/data.md).

## Repository Layout

- `src/phm_pipeline/`: the core library (the abstract method only) — `data.py`
  (TDMS/ZIP + operation loading), `features.py` (order tracking + fault-band
  features), `model.py` (`RULTransformer`), `losses.py` (official score + asymmetric
  loss), `training.py` (datasets, training loop, causal smoothing, export).
- `scripts/`: CLI entry points — feature extraction, `run_lobo.py`,
  `run_tail_validation.py`, figure generation, external PRONOSTIA check.
- `tests/`: unit tests (`pytest`).
- `docs/`: [`data.md`](docs/data.md) (data sources) and
  [`experiments.md`](docs/experiments.md) (design decisions + negative results).
- `artifacts/features/`: the committed order-domain feature table.
- `figures/results/`: figures regenerated from the honest runs.

Exploratory branches (wavelet / FNO / GRU encoders, DMD/SINDy dynamics features and
augmentation, learned calibration, a physics loss, adaptive envelope and
condition/equivalent-age features) were removed to keep the codebase focused on the
abstract method; the rationale is recorded in [`docs/experiments.md`](docs/experiments.md).

## Poster & Abstract

The abstract is in [`HajunJang_PHMKorea2026_abstract.pdf`](HajunJang_PHMKorea2026_abstract.pdf).
The submitted poster figures under `figures/poster/` are the as-presented artifact
(they embed the earlier 0.670 numbers) and are kept out of version control; the result
figures in this repository are regenerated from the honest, leak-free runs.

## How to Cite

If you use this code or its results, please cite the PHM Korea 2026 paper:

> Hajun Jang, Sanha Jang, Andrew H. Kim, Hansol Lim, and Jongseong Brad Choi,
> "Bearing Remaining Useful Life Prediction Using an Order-Domain Transformer and
> Causal End-of-Life Smoothing," *PHM Korea 2026 Conference* (Korean Society for
> Prognostics and Health Management), Republic of Korea, 2026.

A machine-readable citation is in [`CITATION.cff`](CITATION.cff). (The exact
proceedings page and DBpia record are left as a `# TODO` until publication.)

### Acknowledgment

This work was supported by the National Research Foundation of Korea (NRF) grant
funded by the Korea government (MSIT) (No. RS-2025-05515607).

## License

Released under the [MIT License](LICENSE). The license covers the source code only;
the bearing datasets remain under their original terms.
