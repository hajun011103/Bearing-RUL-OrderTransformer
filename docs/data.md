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

**Where to download.** The data is distributed through the **KIMM Data Platform**
(Korea Institute of Machinery & Materials) as the KSPHM–KIMM Machine Data
Challenge dataset:

- Platform (primary): <https://kimmdata.kimm.re.kr>
- KIMM press release confirming the bearing-degradation data release:
  <https://www.kimm.re.kr/sub0504/view/id/21037>
- Challenge organizer (KSPHM, 한국PHM학회): <https://www.phm.or.kr/event/schedule.php>
- Data contact: `smartdp@kimm.re.kr`

There is **no anonymous public direct-download link** — the dataset pages are
rendered client-side and access requires a platform account and/or challenge
registration; the archives were distributed to registered participants. **Do not
re-host the raw KIMM data in this repository**: no open license is stated, so it
is provided under the platform terms / competition rules. If you hold the files
as a participant, place them in `data/Train/` and cite the KIMM Data Platform.

## External FEMTO / PRONOSTIA data

The domain-shift check in [`scripts/run_pronostia_order_test.py`](../scripts/run_pronostia_order_test.py)
uses the NASA FEMTO / PRONOSTIA accelerated bearing life-test dataset
(IEEE PHM 2012 Data Challenge). It is a public benchmark:

- Direct download (~1.16 GB zip, no registration):
  <https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip>
  (the `+` characters are URL-encoded spaces)
- Authoritative index — NASA PCoE data repository, entry #10 "FEMTO Bearing":
  <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Mirror: <https://data.phmsociety.org/nasa/>

Unzip so that `Training_set.zip` / `Validation_Set.zip` land under
`data/external/femto/FEMTOBearingDataSet/`. No explicit open license is posted, so
link the NASA source and cite Nectoux et al. (2012) rather than re-hosting the
zip. A community challenge-split mirror also exists:
<https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset>.

## Public variable-speed substitutes

If you cannot access the KIMM data, the closest **variable-condition
run-to-failure** benchmark is **XJTU-SY**
(<https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets>; 15 bearings, 25.6 kHz,
3 speed/load conditions; cite Wang et al. 2020). Two openly re-hostable
(CC BY 4.0) options are the University of Ottawa time-varying-speed set
(<https://data.mendeley.com/datasets/v43hmbwxpm/2>; diagnosis, no RUL labels) and
the KAIST run-to-failure set
(<https://data.mendeley.com/datasets/5hcdd3tdvb/6>; constant speed, so only a
partial match).

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
