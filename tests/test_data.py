from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm_pipeline.data import (  # noqa: E402
    VibrationZipReader,
    build_segment_index,
    discover_train_runs,
    read_operation_csv,
)


class DataModuleTests(unittest.TestCase):
    def test_operation_csv_and_segment_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "Train1_Operation.csv").write_text(
                "\n".join(
                    [
                        "Time[sec],  Torque[Nm],  Motor speed[rpm],  TC SP Front[C],  TC SP Rear[C],",
                        "0,-4.1,735,10.2,10.3,",
                        "10,-4.2,740,10.4,10.5,",
                        "600,-5.0,800,12.0,12.2,",
                    ]
                ),
                encoding="utf-8",
            )
            with zipfile.ZipFile(root / "Train1_Vibration.zip", "w") as archive:
                archive.writestr("Train1_Vibration/000002.tdms", b"TDSm")
                archive.writestr("Train1_Vibration/000001.tdms", b"TDSm")

            runs = discover_train_runs(root)
            self.assertEqual(runs[0].run_id, "Train1")
            operation = read_operation_csv(runs[0].operation_csv)
            self.assertEqual(list(operation.columns), ["time_s", "torque_nm", "rpm", "temp_front_c", "temp_rear_c"])

            index = build_segment_index(runs[0])
            self.assertEqual(index["segment_id"].tolist(), [1, 2])
            self.assertIn("op_rpm", index)
            self.assertIn("rul_s", index)

    def test_vibration_zip_members_are_numeric_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / "Train9_Vibration.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("Train9_Vibration/000010.tdms", b"TDSm")
                archive.writestr("Train9_Vibration/000002.tdms", b"TDSm")
                archive.writestr("Train9_Vibration/000001.tdms", b"TDSm")

            with VibrationZipReader(zip_path) as reader:
                self.assertEqual(
                    reader.members(),
                    [
                        "Train9_Vibration/000001.tdms",
                        "Train9_Vibration/000002.tdms",
                        "Train9_Vibration/000010.tdms",
                    ],
                )


if __name__ == "__main__":
    unittest.main()

