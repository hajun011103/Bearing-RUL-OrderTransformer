# Data

The raw vibration archives are **not** stored in this repository. Each bearing
run is a multi-gigabyte TDMS archive (the four training archives total roughly
8 GB, and the external FEMTO/PRONOSTIA set adds several GB more), which is far
above GitHub's per-file and repository limits. The `data/` directory is
git-ignored; only the small extracted feature table under `artifacts/features/`
is shipped so the modeling code can be reproduced without the raw signals.

## Expected layout

The code discovers files by name, so place them exactly like this:

```
data/
  Train/
    Train1_Operation.csv
    Train1_Vibration.zip        # TDMS acquisitions, one per member
    Train2_Operation.csv
    Train2_Vibration.zip
    Train3_Operation.csv
    Train3_Vibration.zip
    Train4_Operation.csv
    Train4_Vibration.zip
  external/
    femto/
      FEMTOBearingDataSet/
        Training_set.zip         # contains Learning_set/BearingX_Y/acc_*.csv
        Validation_Set.zip       # contains Full_Test_Set/BearingX_Y/acc_*.csv
        Test_set.zip
```

## KSPHM / PHM Korea bearing run-to-failure data

The four `TrainN` runs come from the PHM Korea bearing degradation challenge.
Each run is one bearing driven to failure:

- `TrainN_Operation.csv` — low-rate operating log. Columns are normalized by
  `read_operation_csv` to `time_s`, `torque_nm`, `rpm`, `temp_front_c`,
  `temp_rear_c` (the raw headers are Korean/English mixed and read as latin-1).
- `TrainN_Vibration.zip` — a ZIP of TDMS members. Each member is one 60-second
  vibration acquisition sampled at 25.6 kHz across the accelerometer channels,
  recorded once every 600 s (a 60 s active window followed by a ~540 s rest).
  Member index `k` maps to acquisition start time `(k-1) * 600 s`.

> **Where to download:** obtain the archives from the official challenge
> distribution and drop them into `data/Train/`. Replace this line with the
> exact source URL/DOI you are permitted to share. Do not re-host the raw data
> here — check the challenge's data-usage terms before redistributing.

## External FEMTO / PRONOSTIA data

The domain-shift check in [`scripts/run_pronostia_order_test.py`](../scripts/run_pronostia_order_test.py)
uses the NASA FEMTO / PRONOSTIA accelerated bearing life-test dataset
(IEEE PHM 2012 Data Challenge). Download `Training_set.zip` / `Validation_Set.zip`
from the public NASA Prognostics Data Repository / FEMTO distribution and place
them under `data/external/femto/FEMTOBearingDataSet/`.

## Regenerating the feature table

With the raw `data/Train` archives in place, rebuild the order-domain feature
table that the models consume:

```bash
python scripts/extract_features.py \
  --data-root data/Train \
  --output artifacts/features/train_full.parquet

python scripts/make_order_domain_features.py \
  --source artifacts/features/train_full.parquet \
  --output artifacts/features/train_full_order_domain.parquet \
  --feature-mode order
```

The committed `artifacts/features/train_full_order_domain.parquet` is the output
of these two steps, so the modeling and evaluation scripts run without the raw
signals.
