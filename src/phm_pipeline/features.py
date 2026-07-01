"""RPM-aware order-domain vibration features for bearing RUL prediction.

Exploratory feature families (DMD/SINDy degradation dynamics, blind-trajectory
reconstruction, spectral-kurtosis adaptive envelope band, condition-compensated
and equivalent-age features) were tried during development and removed; see
``docs/experiments.md`` for the rationale.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import (
    RunFiles,
    VibrationZipReader,
    build_segment_index,
    discover_train_runs,
    read_operation_csv,
)


def _safe_import_scipy_signal():
    try:
        from scipy import signal

        return signal
    except Exception:
        return None


def robust_signal_stats(x: np.ndarray) -> dict[str, float]:
    """Compute time-domain statistics that are useful for bearing degradation."""

    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: np.nan for k in _STAT_KEYS}
    centered = x - np.mean(x)
    abs_x = np.abs(centered)
    rms = float(np.sqrt(np.mean(centered**2)))
    std = float(np.std(centered))
    p2p = float(np.ptp(centered))
    peak = float(np.max(abs_x))
    mean_abs = float(np.mean(abs_x))
    q25, q50, q75, q95, q99 = np.percentile(abs_x, [25, 50, 75, 95, 99])
    eps = 1e-12
    skew = float(np.mean((centered / (std + eps)) ** 3))
    kurt = float(np.mean((centered / (std + eps)) ** 4))
    impulse = peak / (mean_abs + eps)
    crest = peak / (rms + eps)
    clearance = peak / (np.mean(np.sqrt(abs_x + eps)) ** 2 + eps)
    shape = rms / (mean_abs + eps)
    return {
        "mean": float(np.mean(x)),
        "std": std,
        "rms": rms,
        "p2p": p2p,
        "skew": skew,
        "kurtosis": kurt,
        "crest_factor": float(crest),
        "impulse_factor": float(impulse),
        "clearance_factor": float(clearance),
        "shape_factor": float(shape),
        "abs_q25": float(q25),
        "abs_q50": float(q50),
        "abs_q75": float(q75),
        "abs_q95": float(q95),
        "abs_q99": float(q99),
    }


_STAT_KEYS = [
    "mean",
    "std",
    "rms",
    "p2p",
    "skew",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "abs_q25",
    "abs_q50",
    "abs_q75",
    "abs_q95",
    "abs_q99",
]


def rpm_trace(
    rpm: float | np.ndarray,
    n_samples: int,
    sample_rate_hz: float,
) -> np.ndarray:
    """Return a sample-level RPM trace from a scalar or lower-rate vector."""

    if np.isscalar(rpm):
        return np.full(n_samples, float(rpm), dtype=np.float64)
    rpm_arr = np.asarray(rpm, dtype=np.float64)
    if rpm_arr.size == n_samples:
        return rpm_arr
    src_t = np.linspace(0.0, n_samples / sample_rate_hz, rpm_arr.size)
    dst_t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    return np.interp(dst_t, src_t, rpm_arr)


def order_spectrum(
    x: np.ndarray,
    rpm: float | np.ndarray,
    *,
    sample_rate_hz: float,
    samples_per_revolution: int,
    max_order: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an angular-domain spectrum.

    Standard FFT assumes a stationary shaft speed and smears fault lines when
    RPM drifts. Here the signal is resampled onto a uniform shaft-angle grid and
    the frequency axis becomes order, i.e. cycles per shaft revolution.
    """

    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.array([], dtype=float), np.array([], dtype=float)
    trace = np.maximum(rpm_trace(rpm, x.size, sample_rate_hz), 1.0)
    rotations = np.cumsum(trace / 60.0) / sample_rate_hz
    rotations -= rotations[0]
    total_revs = float(rotations[-1])
    if not np.isfinite(total_revs) or total_revs <= 1.0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_angle = max(256, int(total_revs * samples_per_revolution))
    uniform_revs = np.linspace(0.0, total_revs, n_angle, endpoint=False)
    angular = np.interp(uniform_revs, rotations, x - np.mean(x))
    window = np.hanning(angular.size)
    spectrum = np.fft.rfft(angular * window)
    scale = 2.0 / max(window.sum(), 1.0)
    amplitude = np.abs(spectrum) * scale
    orders = np.fft.rfftfreq(angular.size, d=1.0 / samples_per_revolution)
    keep = orders <= max_order
    return orders[keep], amplitude[keep]


def envelope_signal(
    x: np.ndarray,
    *,
    sample_rate_hz: float,
    band_hz: tuple[float, float],
) -> np.ndarray:
    """Return the high-frequency resonance envelope."""

    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    signal = _safe_import_scipy_signal()
    if signal is None:
        smooth = np.convolve(x, np.ones(128) / 128.0, mode="same")
        return np.abs(x - smooth)

    nyq = 0.5 * sample_rate_hz
    low, high = band_hz
    low = max(low / nyq, 1e-4)
    high = min(high / nyq, 0.999)
    if not 0.0 < low < high < 1.0:
        return np.abs(signal.hilbert(x))
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, x)
    return np.abs(signal.hilbert(filtered))


def order_band_features(
    orders: np.ndarray,
    amplitude: np.ndarray,
    fault_orders: dict[str, float],
    *,
    width: float,
    harmonics: int,
    prefix: str,
) -> dict[str, float]:
    """Summarize energy around BPFI/BPFO/BSF/FTF orders and harmonics."""

    out: dict[str, float] = {}
    if orders.size == 0:
        for name in fault_orders:
            for h in range(1, harmonics + 1):
                out[f"{prefix}_{name}_h{h}_amp"] = np.nan
                out[f"{prefix}_{name}_h{h}_energy"] = np.nan
        return out

    total_energy = float(np.sum(amplitude**2) + 1e-12)
    for name, base_order in fault_orders.items():
        harmonic_amps = []
        harmonic_energy = []
        for h in range(1, harmonics + 1):
            center = h * base_order
            mask = np.abs(orders - center) <= width
            if not np.any(mask):
                amp = 0.0
                energy = 0.0
            else:
                amp = float(np.max(amplitude[mask]))
                energy = float(np.sum(amplitude[mask] ** 2) / total_energy)
            out[f"{prefix}_{name}_h{h}_amp"] = amp
            out[f"{prefix}_{name}_h{h}_energy"] = energy
            harmonic_amps.append(amp)
            harmonic_energy.append(energy)
        out[f"{prefix}_{name}_amp_sum"] = float(np.sum(harmonic_amps))
        out[f"{prefix}_{name}_energy_sum"] = float(np.sum(harmonic_energy))
    return out


def spectral_shape_features(
    orders: np.ndarray,
    amplitude: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    """Generic order-spectrum descriptors for unknown or compound faults."""

    if orders.size == 0 or np.sum(amplitude) <= 0:
        return {
            f"{prefix}_centroid": np.nan,
            f"{prefix}_spread": np.nan,
            f"{prefix}_entropy": np.nan,
            f"{prefix}_peak_order": np.nan,
            f"{prefix}_peak_amp": np.nan,
        }
    weights = amplitude / (np.sum(amplitude) + 1e-12)
    centroid = float(np.sum(orders * weights))
    spread = float(np.sqrt(np.sum(((orders - centroid) ** 2) * weights)))
    entropy = float(-np.sum(weights * np.log(weights + 1e-12)) / math.log(weights.size))
    peak_idx = int(np.argmax(amplitude))
    return {
        f"{prefix}_centroid": centroid,
        f"{prefix}_spread": spread,
        f"{prefix}_entropy": entropy,
        f"{prefix}_peak_order": float(orders[peak_idx]),
        f"{prefix}_peak_amp": float(amplitude[peak_idx]),
    }


def aggregate_domain_fault_features(
    segment_features: dict[str, float],
    *,
    n_channels: int,
    fault_orders: dict[str, float],
    domain_prefixes: tuple[str, ...],
) -> dict[str, float]:
    """Aggregate per-channel fault-band features into robust segment descriptors."""

    out: dict[str, float] = {}
    ratio_pairs = (("bpfi", "bpfo"), ("bpfo", "ftf"), ("bsf", "bpfo"), ("bpfi", "bsf"))
    spectral_stats = ("peak_amp", "peak_order", "entropy", "centroid", "spread")

    for domain in domain_prefixes:
        for stat in spectral_stats:
            values = np.asarray(
                [segment_features.get(f"ch{i}_{domain}_{stat}", np.nan) for i in range(1, n_channels + 1)],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                out[f"agg_{domain}_{stat}_mean"] = np.nan
                out[f"agg_{domain}_{stat}_max"] = np.nan
            else:
                out[f"agg_{domain}_{stat}_mean"] = float(np.mean(finite))
                out[f"agg_{domain}_{stat}_max"] = float(np.max(finite))

        totals: dict[str, dict[str, float]] = {
            "amp_sum_mean": {},
            "amp_sum_max": {},
            "energy_sum_mean": {},
            "energy_sum_max": {},
        }
        for fault in fault_orders:
            for metric in ("amp_sum", "energy_sum"):
                values = np.asarray(
                    [
                        segment_features.get(f"ch{i}_{domain}_{fault}_{metric}", np.nan)
                        for i in range(1, n_channels + 1)
                    ],
                    dtype=float,
                )
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    mean_value = np.nan
                    max_value = np.nan
                else:
                    mean_value = float(np.mean(finite))
                    max_value = float(np.max(finite))
                out[f"agg_{domain}_{fault}_{metric}_mean"] = mean_value
                out[f"agg_{domain}_{fault}_{metric}_max"] = max_value
                out[f"agg_{domain}_{fault}_{metric}_max_over_mean"] = float(
                    max_value / (abs(mean_value) + 1e-12)
                ) if np.isfinite(mean_value) and np.isfinite(max_value) else np.nan
                totals[f"{metric}_mean"][fault] = mean_value
                totals[f"{metric}_max"][fault] = max_value

        for metric_key, by_fault in totals.items():
            finite_total = float(
                np.sum([max(value, 0.0) for value in by_fault.values() if np.isfinite(value)])
            )
            suffix = f"{metric_key}_share"
            for fault, value in by_fault.items():
                out[f"agg_{domain}_{fault}_{suffix}"] = (
                    float(max(value, 0.0) / max(finite_total, 1e-12))
                    if np.isfinite(value)
                    else np.nan
                )

        for left, right in ratio_pairs:
            for metric_key, by_fault in totals.items():
                left_value = by_fault.get(left, np.nan)
                right_value = by_fault.get(right, np.nan)
                out[f"agg_{domain}_{left}_to_{right}_{metric_key}_ratio"] = (
                    float(left_value / (right_value + 1e-12))
                    if np.isfinite(left_value) and np.isfinite(right_value)
                    else np.nan
                )
    return out


def extract_segment_features(
    vibration: np.ndarray,
    *,
    rpm: float | np.ndarray,
    config: FeatureConfig,
) -> dict[str, float]:
    """Extract all per-segment features from a ``(channels, samples)`` array."""

    fault_orders = config.geometry.resolved_fault_orders()
    features: dict[str, float] = {}
    dec = max(int(config.analysis_decimate), 1)
    fs = config.sample_rate_hz / dec
    vib = vibration[:, ::dec] if dec > 1 else vibration
    domain_prefixes: tuple[str, ...] = ("order", "env_order")

    for ch_idx, signal_x in enumerate(vib, start=1):
        prefix = f"ch{ch_idx}"
        for key, value in robust_signal_stats(signal_x).items():
            features[f"{prefix}_{key}"] = value

        orders, amp = order_spectrum(
            signal_x,
            rpm,
            sample_rate_hz=fs,
            samples_per_revolution=config.samples_per_revolution,
            max_order=config.max_order,
        )
        features.update(spectral_shape_features(orders, amp, prefix=f"{prefix}_order"))
        features.update(
            order_band_features(
                orders,
                amp,
                fault_orders,
                width=config.order_band_width,
                harmonics=config.order_harmonics,
                prefix=f"{prefix}_order",
            )
        )

        env = envelope_signal(signal_x, sample_rate_hz=fs, band_hz=config.envelope_band_hz)
        env_orders, env_amp = order_spectrum(
            env,
            rpm,
            sample_rate_hz=fs,
            samples_per_revolution=config.samples_per_revolution,
            max_order=config.max_order,
        )
        features.update(
            spectral_shape_features(env_orders, env_amp, prefix=f"{prefix}_env_order")
        )
        features.update(
            order_band_features(
                env_orders,
                env_amp,
                fault_orders,
                width=config.order_band_width,
                harmonics=config.order_harmonics,
                prefix=f"{prefix}_env_order",
            )
        )

    rms_cols = [features[f"ch{i}_rms"] for i in range(1, vib.shape[0] + 1)]
    kurt_cols = [features[f"ch{i}_kurtosis"] for i in range(1, vib.shape[0] + 1)]
    features["agg_rms_mean"] = float(np.mean(rms_cols))
    features["agg_rms_max"] = float(np.max(rms_cols))
    features["agg_kurtosis_mean"] = float(np.mean(kurt_cols))
    features["agg_kurtosis_max"] = float(np.max(kurt_cols))
    features.update(
        aggregate_domain_fault_features(
            features,
            n_channels=vib.shape[0],
            fault_orders=fault_orders,
            domain_prefixes=domain_prefixes,
        )
    )
    return features


def extract_run_features(
    run: RunFiles,
    config: FeatureConfig,
    *,
    limit_segments: int | None = None,
    channels: list[int] | None = None,
) -> pd.DataFrame:
    """Extract a complete segment-level feature table for one run."""

    operation = read_operation_csv(run.operation_csv)
    index = build_segment_index(
        run,
        segment_seconds=config.segment_seconds,
        acquisition_period_seconds=config.acquisition_period_seconds,
    )
    if limit_segments is not None:
        index = index.head(limit_segments).copy()

    rows = []
    with VibrationZipReader(run.vibration_zip) as reader:
        total = len(index)
        for n, record in enumerate(index.itertuples(index=False), start=1):
            if n == 1 or n == total or n % 10 == 0:
                print(f"[features] {run.run_id}: segment {n}/{total}", flush=True)
            vibration = reader.read_segment(record.member, channels=channels)
            rpm = segment_rpm_profile(
                operation,
                float(record.start_time_s),
                float(record.end_time_s),
            )
            feats = extract_segment_features(vibration, rpm=rpm, config=config)
            base = record._asdict()
            base.pop("zip_path", None)
            base.update(feats)
            rows.append(base)

    return pd.DataFrame(rows)


def segment_rpm_profile(
    operation: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Return low-rate RPM samples spanning one vibration acquisition."""

    mask = (operation["time_s"] >= start_s) & (operation["time_s"] <= end_s)
    rpm_values = operation.loc[mask, "rpm"].to_numpy(dtype=float)
    rpm_values = rpm_values[np.isfinite(rpm_values)]
    if rpm_values.size >= 2:
        return rpm_values
    sample_times = np.linspace(start_s, end_s, 7)
    source_t = operation["time_s"].to_numpy(dtype=float)
    source_rpm = operation["rpm"].to_numpy(dtype=float)
    valid = np.isfinite(source_t) & np.isfinite(source_rpm)
    if valid.sum() < 2:
        return np.array([float(np.nanmean(source_rpm))], dtype=float)
    return np.interp(sample_times, source_t[valid], source_rpm[valid])


def extract_all_features(
    data_root: str | Path,
    config: FeatureConfig,
    *,
    limit_segments: int | None = None,
    channels: list[int] | None = None,
) -> pd.DataFrame:
    """Extract features for every discovered training run."""

    frames = []
    for run in discover_train_runs(data_root):
        print(f"[features] extracting {run.run_id}", flush=True)
        frames.append(
            extract_run_features(
                run,
                config,
                limit_segments=limit_segments,
                channels=channels,
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df.attrs["feature_config"] = asdict(config)
    return df


def write_features(df: pd.DataFrame, output: str | Path) -> None:
    """Persist features as parquet when possible, otherwise CSV."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        df.to_csv(output, index=False)
    else:
        df.to_parquet(output, index=False)
