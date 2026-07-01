# Experiments, design decisions, and negative results

This project explored many ideas before settling on the compact method in the
abstract: **order-domain features → a time-gap-aware Transformer → causal
end-of-life (EOL) quantile smoothing**. To keep the public codebase focused, the
exploratory branches were removed from the source tree, but the *reasoning* is
recorded here so the choices are reproducible and reviewable.

Numbers below are the official competition score (higher is better, max 1.0),
evaluated leave-one-bearing-out (LOBO) unless noted. Where a comparison artifact
was retained it is cited; where only the code path survived development it is
marked *(exploratory, not benchmarked in the final comparison)* rather than
given an invented number.

---

## 1. Feature representation — kept order-domain

**Why tried.** Under varying shaft speed, fixed-frequency FFT bins smear
bearing-defect lines (BPFO/BPFI/BSF/FTF), because a defect's frequency moves
with RPM. Resampling the signal onto the shaft-angle grid ("order tracking")
pins each defect to a fixed *order* regardless of speed. See
[`order_spectrum`](../src/phm_pipeline/features.py).

**What was compared** (`artifacts/runs/order_domain_comparison.csv`,
`artifacts/runs/lobo_validation_comparison.csv`):

| feature set | scale | score |
| --- | --- | --- |
| order-domain only, raw LOBO | 1.0 | 0.555 |
| all features (time+order+op), raw LOBO | 1.0 | 0.572 |
| order-domain + log target | tail, 1.0 | 0.219 (over-predicts) |
| order-domain + log target | tail, 0.37 | 0.449 |
| time+order mixed + log target | tail, 0.37 | 0.464 |

**Decision.** On the *raw* score the full and mixed feature sets were marginally
higher, but the order-domain representation (a) is directly motivated by the
speed-robustness the abstract is about, (b) is compact, and (c) reaches the best
final score once combined with causal smoothing (Section 3). The mixed/full sets
were dropped to keep the story and the input space clean.

## 2. Prediction-scale multiplier — rejected as a leaky shortcut

**Why tried.** A `log1p` target model at scale 1.0 systematically *over-predicts*
RUL early in life (over-prediction rate ≈ 0.93, score ≈ 0.18–0.22). Multiplying
predictions by a constant < 1 sharply cuts over-prediction:
`artifacts/runs/order_domain_scale_sweep.csv` peaks near scale 0.37–0.41
(score ≈ 0.449).

**Decision — removed from the honest path.** The best scale was chosen by
sweeping on the very set being reported, i.e. it tunes on test. The headline
LOBO now uses **scale = 1.0** and controls over-prediction through the
asymmetric loss ([`AsymmetricRULLoss`](../src/phm_pipeline/losses.py)) plus
causal smoothing instead. `optimize_val_scale` / `--prediction-scale` remain in
the code only for the secondary per-run-tail diagnostic, where the scale is now
selected on an inner split, never on the reported tail
([`scripts/run_tail_validation.py`](../scripts/run_tail_validation.py)).

**How large was the leak?** With the scale and smoothing selected honestly on an
inner split, the late-life tail score collapses to **0.086** (100% of the tail
is over-predicted), versus ~0.47 when the scale was tuned on the tail itself.
The model, trained only on the earliest ~60% of each bearing, has never seen a
near-death vibration signature and systematically over-estimates remaining life
at true end-of-life. This is exactly why leave-one-**bearing**-out (where the
model sees complete run-to-failure lives of the *other* bearings) is the more
meaningful protocol, and why the tail experiment is kept only as an honest
negative result.

## 3. Temporal post-processing — decay rejected, EOL-quantile kept

**Why tried.** Segment-level predictions are jittery over a bearing's life. Two
causal smoothers were implemented ([`apply_temporal_postprocess`](../src/phm_pipeline/training.py)):

- **decay** — force RUL to fall by at least the elapsed time (monotone, ≥ 1 s/s).
- **eol_quantile** — track a causal running quantile of the *predicted* EOL
  (`time + predicted_RUL`) and re-derive RUL from it.

**What was compared** (`artifacts/runs/lobo_order_domain_temporal_sweep.csv`):

| method | params | LOBO score |
| --- | --- | --- |
| raw (no smoothing) | — | 0.555 |
| decay | blend 1.0, slack 0 | 0.437 |
| eol_quantile | q 0.75, blend 0.5 | 0.593 |
| eol_quantile | q 0.9, blend 1.0 | 0.670 *(oracle, tuned on test — see below)* |

**Decision.** Strict decay *hurt* — forcing a 1 s/s monotone drop is too rigid
for a model that starts optimistic. EOL-quantile smoothing helped and was
adopted. **Important caveat:** the q = 0.9 that gives 0.670 was selected on the
pooled OOF (the test folds), so it is an optimistic ceiling.

## 3a. The honest LOBO number

There were in fact *two* leaks stacked in the old 0.670: (1) the smoothing
hyper-parameters were tuned on the pooled OOF, and (2) each fold's model was
early-stopped on the very bearing it was scored on (`val_runs=[held-out]`).
[`scripts/run_lobo.py`](../scripts/run_lobo.py) removes both — it early-stops on
a tail split of the *training* bearings and selects smoothing with an inner LOBO
over the training bearings — and reports three numbers on the same OOF:

| variant | how smoothing is chosen | LOBO score |
| --- | --- | --- |
| previously reported | q=0.9 tuned on test **+** early-stop on test | 0.670 |
| raw (honest models, no smoothing) | — | 0.441 |
| **honest (leak-free)** | inner LOBO over training bearings | **0.462** |
| oracle (ceiling) | tuned on the pooled OOF | 0.466 |

Most of the old gap was the early-stopping leak (raw dropped 0.555 → 0.441 once
the model is no longer early-stopped on the held-out bearing). With honest
models the leak-free smoothing (0.462) is already within 0.004 of the oracle
ceiling (0.466), so the smoothing benefit that remains is small but real. Per
fold the honest smoothing helped Train1 (0.57→0.67), Train2 and Train3 slightly,
and mildly hurt Train4 — and Train3 is very poorly predicted (~0.10), showing
how high the bearing-to-bearing variance is with only four runs.

## 4. Alternative architectures — kept the plain Transformer

Several encoders were implemented and later removed:

- **Wavelet / PINNsFormer encoder** — learnable sine/cosine ("wavelet")
  activations, motivated by physics-informed transformers for smoother temporal
  extrapolation.
- **Fourier Neural Operator (FNO)** — low-mode spectral mixing over the elapsed
  time axis, treating order features as a channel field.
- **GRU autoregressive** (± FNO residual) — a recurrent baseline for the
  irregular segment history.
- **FiLM operating-condition conditioning** — feature-wise modulation from
  torque/RPM/temperature channels.

*(exploratory, not benchmarked in the final comparison — head-to-head artifacts
were pruned.)* **Decision.** None gave a clear, robust win over the plain
time-gap-aware Transformer (`RULTransformer`) that already encodes continuous
elapsed time and acquisition gaps. The simpler model was kept; the extra
architectures added surface area and tuning burden without a demonstrated gain.

## 5. Causal degradation-dynamics features (DMD / SINDy) — dropped

**Why tried.** Fit a local Koopman/DMD operator and a sparse `dy/dt = f(y)`
(SINDy) model on the trailing health-indicator history to encode *how fast* the
bearing is degrading, plus a `reconstruct_blind_trajectory` helper to estimate
health inside the ~9-minute blind gaps between acquisitions.

**Decision.** These `dmd_*` / `sindy_*` / blind-trajectory columns are **not** in
the order-domain feature table used by the abstract (confirmed: the table
contains only metadata, operation covariates, and per-channel order/envelope-order
features). They added many fragile columns fit on very short histories without a
clear score benefit, so they were removed from the pipeline.

## 6. Governing-dynamics data augmentation — dropped

**Why tried.** Densify the blind gaps with training-only pseudo rows that
interpolate the observed state toward a Koopman/DMD + SINDy one-step forecast, to
give the sequence model more temporal resolution.

**Decision.** Causal but noisy: the operator forecasts inject model bias and the
"denoise/confidence-gate" machinery needed to tame them was complex. It was off
by default in every reported run and was removed.

## 7. Learned RUL calibration and a PINO physics loss — dropped

**Why tried.** (a) A trainable positive affine calibration in physical RUL
seconds; (b) a self-supervised PINO-style term encouraging RUL to fall ~1 s per
elapsed second and stay monotone inside a context window.

**Decision.** Both were optional and off in the headline runs. The asymmetric
score loss plus causal smoothing already covered the same goals (conservative,
temporally consistent predictions), so the extra objectives/knobs were removed.

## 8. Adaptive envelope band and condition/equivalent-age features — dropped

**Why tried.** (a) Spectral-kurtosis band selection to place the envelope
demodulation band adaptively; (b) condition-compensated and "equivalent-age"
features (cumulative revolutions, load-weighted age, condition-regressed
residuals) to normalize for operating point.

**Decision.** Neither is present in the abstract's order-domain table. They
increased feature count and compute for no demonstrated LOBO gain and were
removed.

---

### Summary of what remained

| component | status |
| --- | --- |
| Order-domain (+ envelope-order) features | **kept** |
| Time-gap-aware Transformer (`RULTransformer`) | **kept** |
| Asymmetric official-score loss | **kept** |
| Causal EOL-quantile smoothing (leak-free selection) | **kept** |
| Prediction-scale sweep, decay smoothing | rejected (leaky / too rigid) |
| Wavelet / FNO / GRU / FiLM encoders | removed (no robust win) |
| DMD/SINDy features, dynamics augmentation, blind-trajectory | removed |
| Learned calibration, PINO physics loss | removed |
| Adaptive envelope, condition/equivalent-age features | removed |
