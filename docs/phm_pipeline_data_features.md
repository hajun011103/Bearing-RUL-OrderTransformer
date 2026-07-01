# PHM Pipeline API reference

Core modules under `src/phm_pipeline/`. Only the abstract method is kept; see
[experiments.md](experiments.md) for what was removed and why.

## Data (`data.py`)

- `discover_train_runs(data_root)` pairs `Train*_Operation.csv` with `Train*_Vibration.zip`.
- `read_operation_csv(path)` normalizes operation columns to `time_s`, `torque_nm`, `rpm`, `temp_front_c`, `temp_rear_c`.
- `build_segment_index(run)` creates segment timestamps, interpolated operation covariates, and training RUL labels.
- `VibrationZipReader(zip_path)` streams TDMS members directly from ZIP archives and returns arrays shaped `(channels, samples)`.

## Features (`features.py`)

- `order_spectrum(signal, rpm, ...)` resamples vibration into the angular (order) domain before FFT, reducing spectral smearing from RPM drift.
- `order_band_features(...)` / `spectral_shape_features(...)` summarize energy around BPFO/BPFI/BSF/FTF orders and harmonics, plus generic spectrum descriptors.
- `extract_segment_features(vibration, rpm, config)` returns time-domain, order-domain, and envelope-order features per channel.
- `extract_all_features(data_root, config, ...)` builds the full train feature table.

## Modeling (`model.py`, `losses.py`, `training.py`)

- `RULTransformer(ModelConfig)` consumes padded feature windows plus true elapsed timestamps and a validity mask; encodes elapsed time and acquisition gaps.
- `model_config_from_dict(payload)` rebuilds a `ModelConfig` from a checkpoint, ignoring unknown keys (so older checkpoints still load).
- `AsymmetricRULLoss` optimizes the negative log official score with extra over-prediction and conservative-bias penalties.
- `RULTrainer(TrainingConfig).fit()` trains with early stopping on validation score.
- `apply_temporal_postprocess(...)` is the single source of truth for causal smoothing (`eol_quantile` and `decay`).
- `export_predictions(...)` writes row-level predicted RUL from a saved checkpoint.
