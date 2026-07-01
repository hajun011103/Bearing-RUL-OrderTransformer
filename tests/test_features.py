from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm_pipeline.config import FeatureConfig  # noqa: E402
from phm_pipeline.features import (  # noqa: E402
    extract_segment_features,
    order_spectrum,
)


class FeatureModuleTests(unittest.TestCase):
    def test_order_spectrum_tracks_shaft_order(self) -> None:
        sample_rate = 1024.0
        rpm = 900.0
        shaft_hz = rpm / 60.0
        t = np.arange(int(sample_rate * 4.0)) / sample_rate
        signal = np.sin(2.0 * math.pi * 2.0 * shaft_hz * t)
        orders, amp = order_spectrum(
            signal,
            rpm,
            sample_rate_hz=sample_rate,
            samples_per_revolution=128,
            max_order=8.0,
        )
        peak_order = float(orders[np.argmax(amp)])
        self.assertAlmostEqual(peak_order, 2.0, delta=0.05)

    def test_segment_features_include_order_and_envelope_bands(self) -> None:
        sample_rate = 4096.0
        t = np.arange(int(sample_rate * 2.0)) / sample_rate
        carrier = np.sin(2.0 * math.pi * 600.0 * t)
        mod = 1.0 + 0.4 * np.sin(2.0 * math.pi * 30.0 * t)
        signal = (carrier * mod).astype(np.float32)
        config = FeatureConfig(
            sample_rate_hz=sample_rate,
            analysis_decimate=1,
            envelope_band_hz=(200.0, 1200.0),
        )
        features = extract_segment_features(signal[None, :], rpm=900.0, config=config)

        self.assertIn("ch1_rms", features)
        self.assertIn("ch1_order_bpfo_amp_sum", features)
        self.assertIn("ch1_env_order_bpfo_amp_sum", features)
        self.assertIn("agg_order_bpfo_energy_sum_max", features)
        self.assertIn("agg_order_bpfi_to_bpfo_energy_sum_max_ratio", features)
        self.assertGreater(features["agg_rms_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
