"""Dataset discovery, operation alignment, and TDMS ZIP loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import io
import re
import struct
import zipfile

import numpy as np
import pandas as pd


_RUN_RE = re.compile(r"Train(?P<run>\d+)")


@dataclass(frozen=True, slots=True)
class RunFiles:
    """Paths belonging to one bearing run."""

    run_id: str
    operation_csv: Path
    vibration_zip: Path


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """Metadata for one 60-second vibration acquisition."""

    run_id: str
    segment_id: int
    zip_path: Path
    member: str
    start_time_s: float
    mid_time_s: float
    end_time_s: float


def discover_train_runs(data_root: str | Path) -> list[RunFiles]:
    """Find Train*_Operation.csv and Train*_Vibration.zip pairs."""

    root = Path(data_root)
    operation = {}
    vibration = {}
    for path in root.glob("Train*_Operation.csv"):
        match = _RUN_RE.search(path.name)
        if match:
            operation[f"Train{match.group('run')}"] = path
    for path in root.glob("Train*_Vibration.zip"):
        match = _RUN_RE.search(path.name)
        if match:
            vibration[f"Train{match.group('run')}"] = path

    runs = []
    for run_id in sorted(operation, key=lambda x: int(x.replace("Train", ""))):
        if run_id not in vibration:
            raise FileNotFoundError(f"Missing vibration archive for {run_id}")
        runs.append(RunFiles(run_id, operation[run_id], vibration[run_id]))
    if not runs:
        raise FileNotFoundError(f"No Train*_Operation.csv files found under {root}")
    return runs


def _clean_operation_columns(columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in columns:
        key = col.strip().lower()
        if key.startswith("time"):
            mapping[col] = "time_s"
        elif "torque" in key:
            mapping[col] = "torque_nm"
        elif "speed" in key or "rpm" in key:
            mapping[col] = "rpm"
        elif "front" in key:
            mapping[col] = "temp_front_c"
        elif "rear" in key:
            mapping[col] = "temp_rear_c"
        elif not key or key.startswith("unnamed"):
            mapping[col] = "drop_empty"
    return mapping


def read_operation_csv(path: str | Path) -> pd.DataFrame:
    """Read an operation CSV and normalize column names."""

    df = pd.read_csv(path, encoding="latin1")
    df = df.rename(columns=_clean_operation_columns(list(df.columns)))
    if "drop_empty" in df.columns:
        df = df.drop(columns=["drop_empty"])
    required = {"time_s", "torque_nm", "rpm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_s", "rpm"]).sort_values("time_s")
    df = df.drop_duplicates("time_s", keep="last").reset_index(drop=True)
    return df


def interpolate_operation(operation: pd.DataFrame, times_s: np.ndarray) -> pd.DataFrame:
    """Interpolate low-rate operation channels at arbitrary timestamps."""

    times = operation["time_s"].to_numpy(dtype=float)
    out = {"time_s": times_s.astype(float)}
    for col in operation.columns:
        if col == "time_s":
            continue
        values = operation[col].to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 2:
            out[col] = np.full_like(times_s, np.nan, dtype=float)
        else:
            out[col] = np.interp(times_s, times[mask], values[mask])
    return pd.DataFrame(out)


def operation_window_summary(
    operation: pd.DataFrame,
    start_s: float,
    end_s: float,
) -> dict[str, float]:
    """Summarize operation covariates over one active vibration window."""

    mask = (operation["time_s"] >= start_s) & (operation["time_s"] <= end_s)
    window = operation.loc[mask]
    if window.empty:
        sample_times = np.array([start_s, 0.5 * (start_s + end_s), end_s], dtype=float)
        window = interpolate_operation(operation, sample_times)

    out: dict[str, float] = {}
    time = window["time_s"].to_numpy(dtype=float)
    rel_time = time - float(time[0]) if time.size else np.array([0.0])
    for col in operation.columns:
        if col == "time_s":
            continue
        raw_values = window[col].to_numpy(dtype=float)
        finite = np.isfinite(raw_values)
        values = raw_values[finite]
        rel = rel_time[finite] if rel_time.size == raw_values.size else rel_time[: values.size]
        if values.size == 0:
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan
            out[f"{col}_min"] = np.nan
            out[f"{col}_max"] = np.nan
            out[f"{col}_slope"] = 0.0
            continue
        out[f"{col}_mean"] = float(np.mean(values))
        out[f"{col}_std"] = float(np.std(values))
        out[f"{col}_min"] = float(np.min(values))
        out[f"{col}_max"] = float(np.max(values))
        if values.size >= 2 and rel.size >= values.size:
            slope = np.polyfit(rel[: values.size], values, deg=1)[0]
            out[f"{col}_slope"] = float(slope)
        else:
            out[f"{col}_slope"] = 0.0
    return out


class VibrationZipReader:
    """Read vibration TDMS members directly from a ZIP archive."""

    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        self._zip = zipfile.ZipFile(self.zip_path)

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "VibrationZipReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def members(self) -> list[str]:
        names = [n for n in self._zip.namelist() if n.lower().endswith(".tdms")]

        def key(name: str) -> int:
            match = re.search(r"(\d+)\.tdms$", name)
            return int(match.group(1)) if match else 10**9

        return sorted(names, key=key)

    def read_segment(
        self,
        member: str,
        channels: list[int] | None = None,
        dtype: np.dtype | str = np.float32,
    ) -> np.ndarray:
        """Read one segment as an array shaped ``(channels, samples)``."""

        raw = self._zip.read(member)
        try:
            data = _read_simple_tdms(raw).astype(dtype, copy=False)
            if channels is not None:
                data = data[np.asarray(channels, dtype=int)]
            return data
        except Exception:
            return self._read_with_nptdms(raw, channels=channels, dtype=dtype)

    @staticmethod
    def _read_with_nptdms(
        raw: bytes,
        channels: list[int] | None,
        dtype: np.dtype | str,
    ) -> np.ndarray:
        from nptdms import TdmsFile

        tdms = TdmsFile.read(io.BytesIO(raw))
        group = tdms.groups()[0]
        arrays = []
        for idx, channel in enumerate(group.channels()):
            if channels is not None and idx not in channels:
                continue
            arrays.append(channel[:].astype(dtype, copy=False))
        if not arrays:
            raise ValueError("No TDMS channels were selected")
        return np.stack(arrays, axis=0)


def _read_simple_tdms(raw: bytes) -> np.ndarray:
    """Read the simple contiguous-float TDMS flavor used by this dataset."""

    if raw[:4] != b"TDSm":
        raise ValueError("Not a TDMS file")
    _, toc_mask, _version, _next_segment_offset, raw_data_offset = struct.unpack(
        "<4sIIQQ", raw[:28]
    )
    metadata_start = 28
    raw_start = metadata_start + int(raw_data_offset)
    offset = metadata_start
    object_count = struct.unpack_from("<I", raw, offset)[0]
    offset += 4

    channels: list[tuple[str, int, int]] = []
    for _ in range(object_count):
        path_len = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        path = raw[offset : offset + path_len].decode(errors="replace")
        offset += path_len
        raw_index = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if raw_index != 0xFFFFFFFF:
            data_type, dimension, values = struct.unpack_from("<IIQ", raw, offset)
            offset += 16
            if dimension != 1:
                raise ValueError(f"Unsupported TDMS dimension {dimension}")
            channels.append((path, data_type, int(values)))
        property_count = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        for _ in range(property_count):
            name_len = struct.unpack_from("<I", raw, offset)[0]
            offset += 4 + name_len
            property_type = struct.unpack_from("<I", raw, offset)[0]
            offset += 4 + _tdms_property_size(raw, offset, property_type)

    if not channels:
        raise ValueError("No numeric TDMS channels found")
    data_types = {data_type for _, data_type, _ in channels}
    sample_counts = {count for _, _, count in channels}
    if len(data_types) != 1 or len(sample_counts) != 1:
        raise ValueError("Mixed TDMS channel layouts are not supported by fallback")

    data_type = next(iter(data_types))
    sample_count = next(iter(sample_counts))
    dtype = _TDMS_NUMERIC_DTYPES.get(data_type)
    if dtype is None:
        raise ValueError(f"Unsupported TDMS numeric type code {data_type}")

    n_channels = len(channels)
    total = n_channels * sample_count
    arr = np.frombuffer(raw, dtype=dtype, count=total, offset=raw_start)
    interleaved = bool(toc_mask & 0x20)
    if interleaved:
        arr = arr.reshape(sample_count, n_channels).T
    else:
        arr = arr.reshape(n_channels, sample_count)
    return np.asarray(arr)


_TDMS_NUMERIC_DTYPES = {
    # TDMS numeric type codes: 1=I8, 2=I16, 3=I32, 4=I64, 5=U8, ... 9=float32,
    # 10=float64. Code 1 is a 1-byte signed integer, so it maps to "<i1".
    1: np.dtype("<i1"),
    2: np.dtype("<i2"),
    3: np.dtype("<i4"),
    4: np.dtype("<i8"),
    5: np.dtype("<u1"),
    6: np.dtype("<u2"),
    7: np.dtype("<u4"),
    8: np.dtype("<u8"),
    9: np.dtype("<f4"),
    10: np.dtype("<f8"),
}


def _tdms_property_size(raw: bytes, offset: int, property_type: int) -> int:
    dtype = _TDMS_NUMERIC_DTYPES.get(property_type)
    if dtype is not None:
        return dtype.itemsize
    if property_type == 0x20:
        size = struct.unpack_from("<I", raw, offset)[0]
        return 4 + size
    if property_type == 0x44:
        return 16
    raise ValueError(f"Unsupported TDMS property type {property_type}")


def build_segment_index(
    run: RunFiles,
    *,
    segment_seconds: float = 60.0,
    acquisition_period_seconds: float = 600.0,
) -> pd.DataFrame:
    """Build a segment-level index with operation covariates and RUL labels."""

    operation = read_operation_csv(run.operation_csv)
    run_end_s = float(operation["time_s"].max())
    with VibrationZipReader(run.vibration_zip) as reader:
        members = reader.members()

    records = []
    for member in members:
        match = re.search(r"(\d+)\.tdms$", member)
        if not match:
            continue
        segment_id = int(match.group(1))
        start_s = (segment_id - 1) * acquisition_period_seconds
        end_s = start_s + segment_seconds
        mid_s = start_s + 0.5 * segment_seconds
        records.append(
            SegmentRecord(
                run_id=run.run_id,
                segment_id=segment_id,
                zip_path=run.vibration_zip,
                member=member,
                start_time_s=start_s,
                mid_time_s=mid_s,
                end_time_s=end_s,
            )
        )

    index = pd.DataFrame([asdict(r) for r in records])
    op = interpolate_operation(operation, index["mid_time_s"].to_numpy(dtype=float))
    op = op.drop(columns=["time_s"]).add_prefix("op_")
    win = pd.DataFrame(
        [
            operation_window_summary(operation, r.start_time_s, r.end_time_s)
            for r in records
        ]
    ).add_prefix("opwin_")
    index = pd.concat(
        [index.reset_index(drop=True), op.reset_index(drop=True), win.reset_index(drop=True)],
        axis=1,
    )
    index["run_end_s"] = run_end_s
    index["rul_s"] = np.maximum(run_end_s - index["mid_time_s"], 1.0)
    index["rul_segments"] = np.maximum(index["segment_id"].max() - index["segment_id"], 0)
    index["life_fraction"] = index["mid_time_s"] / max(run_end_s, 1.0)
    return index
