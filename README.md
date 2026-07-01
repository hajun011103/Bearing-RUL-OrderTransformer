# Order-Domain Transformer + Causal End-of-Life Smoothing for Bearing RUL

Code for the PHM Korea 2026 poster:

> **Bearing Remaining Useful Life Prediction Using an Order-Domain Transformer
> and Causal End-of-Life Smoothing** — Hajun Jang, Sanha Jang, Andrew H. Kim,
> Hansol Lim, Jongseong Brad Choi (SUNY Korea).

Predicting bearing remaining useful life (RUL) from vibration under *varying
speed* is hard for two reasons: speed changes smear defect-related spectral
features, and segment-level predictions are temporally unstable. This project
addresses both with a compact, physically motivated pipeline:

1. **Order tracking** — resample each vibration acquisition onto the shaft-angle
   grid so bearing-defect components (BPFO/BPFI/BSF/FTF and harmonics) stay at
   fixed *orders* regardless of RPM drift.
2. **Time-gap-aware Transformer** — a Transformer encoder over segment-level
   order-domain features that encodes true elapsed time and the acquisition gaps
   between segments, and predicts RUL causally.
3. **Causal end-of-life (EOL) smoothing** — post-process each RUL trajectory
   using a causal running quantile of the *predicted* end-of-life, using only
   information up to the current time step (safe for online use).

## Results (honest, leak-free)

The headline evaluation is **leave-one-bearing-out (LOBO)** on the four PHM Korea
run-to-failure bearings, scored with the official competition score (higher is
better, max 1.0). Every hyper-parameter that touches a held-out bearing is
selected *without* looking at it: models are early-stopped on a tail split of the
training bearings, and the EOL-smoothing hyper-parameters are chosen by an inner
LOBO over the training bearings (see [`scripts/run_lobo.py`](scripts/run_lobo.py)).

| variant | what it measures | score | MAE (s) |
| --- | --- | --- | --- |
| raw (no smoothing) | honest models, no post-processing | 0.441 | 16,875 |
| **honest (leak-free)** | **inner-selected causal EOL smoothing** | **0.462** | 15,925 |
| oracle (ceiling) | smoothing tuned on the test folds | 0.466 | 16,531 |

> ⚠️ An earlier version reported **0.670**. That number stacked two optimistic
> leaks — the smoothing quantile was tuned on the test folds, and each model was
> early-stopped on the bearing it was scored on. Removing both drops the score to
> **0.462**. We keep the honest number as the headline and report the oracle
> ceiling and the old value transparently. See
> [`docs/experiments.md`](docs/experiments.md) for the full accounting.

With only four bearings, LOBO variance is high: the honest per-fold scores range
from ~0.10 (Train3) to ~0.67 (Train1). A secondary late-life "tail" test and an
external PRONOSTIA/FEMTO check are reported honestly (including their failure
modes) in [`docs/experiments.md`](docs/experiments.md).

## Reproduce

The extracted order-domain feature table is committed
(`artifacts/features/train_full_order_domain.parquet`, a few hundred rows), so the
modeling runs on CPU in minutes without the multi-GB raw signals.

```bash
# 1. environment
pip install -e .          # or: pip install -r requirements.txt

# 2. honest leave-one-bearing-out (writes artifacts/runs/lobo_order_domain_nested/)
python scripts/run_lobo.py

# 3. honest late-life tail diagnostic
python scripts/run_tail_validation.py
```

`run_lobo.py` prints the raw / honest / oracle table and writes
`nested_lobo_summary.json` plus the out-of-fold predictions.

### Regenerating features from raw data

The raw archives are **not** in the repo (see [`docs/data.md`](docs/data.md) for
how to obtain them and where to place them). With `data/Train/` populated:

```bash
python scripts/extract_features.py \
  --data-root data/Train \
  --output artifacts/features/train_full.parquet

python scripts/make_order_domain_features.py \
  --source artifacts/features/train_full.parquet \
  --output artifacts/features/train_full_order_domain.parquet \
  --feature-mode order
```

### External PRONOSTIA / FEMTO check

A domain-shift diagnostic on the NASA FEMTO/PRONOSTIA dataset
([`scripts/run_pronostia_order_test.py`](scripts/run_pronostia_order_test.py))
trains the same pipeline on `Learning_set` and evaluates on the validation set.
PRONOSTIA has fixed per-condition speeds and very short-life validation bearings,
so treat it as a robustness probe, not a benchmark — the order-domain pipeline
needs dataset-specific tuning to be competitive there.

## Repository layout

```
src/phm_pipeline/     core library (the abstract method only)
  data.py             TDMS/ZIP + operation-CSV loading, segment indexing
  features.py         order tracking, order/envelope-order fault-band features
  model.py            RULTransformer (time-gap-aware Transformer encoder)
  losses.py           official score + asymmetric conservative loss
  training.py         datasets, training loop, causal EOL smoothing, export
scripts/              CLI entry points (feature extraction, run_lobo, tail, …)
tests/                unit tests (pytest)
docs/                 data.md, experiments.md (design decisions + negative results)
artifacts/features/   the committed order-domain feature table
figures/results/      result figures regenerated from the honest runs
```

Result figures under `figures/results/` are regenerated from the honest runs by
`scripts/make_result_figures.py`. The submitted poster figures
(`figures/poster/`) are the as-presented artifact (they embed the earlier 0.670
numbers) and are kept out of version control.

Exploratory branches (wavelet / FNO / GRU encoders, DMD/SINDy dynamics features
and augmentation, learned calibration, a physics loss, adaptive envelope and
condition/equivalent-age features) were removed to keep the codebase focused on
the abstract method; the rationale and results are recorded in
[`docs/experiments.md`](docs/experiments.md).

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

## Acknowledgment

This work was supported by the National Research Foundation of Korea (NRF) grant
funded by the Korea government (MSIT) (RS-2025-05515607).

## License

Code is released under the [MIT License](LICENSE). The license covers the source
code only; the bearing datasets remain under their original terms.
