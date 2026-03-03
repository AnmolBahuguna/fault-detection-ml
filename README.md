# Machine Fault Detection (ML alrIEEEna'26)

## 1-minute quickstart (judge run)

1) Put `TRAIN.csv` and `TEST.csv` next to `fault_detection_solution.py`.

2) Install minimal deps:

```bash
pip install -r requirements.txt
```

3) Run (fast smoke mode is default via `.env.example`):

```bash
python fault_detection_solution.py
```

4) Outputs:

- `FINAL.csv` (submission)
- `final_model.pkl` (reproducibility snapshot)
- `run_summary.json` (run settings + OOF metrics + runtime)

Runtime expectations (rough, CPU):

- `FAST_SMOKE_TEST=true`: typically minutes
- `FAST_SMOKE_TEST=false`: can take significantly longer depending on optional models and hardware

## Abstract

This repository provides a research-oriented, end-to-end solution for **binary machine fault detection** using an embedded monitoring dataset with **47 numerical input features** (`F01`–`F47`).

The pipeline is designed for strong generalization on hidden evaluation data and includes:

- feature engineering via row-wise distributional statistics
- multiple strong tabular models (tree ensembles + linear baseline)
- class imbalance handling
- probability-to-label conversion via **threshold tuning for F1-score**
- generation of a submission file `FINAL.csv` in the required format (`ID`, `CLASS`)

The main entry-point is a single script: `fault_detection_solution.py`.

## Key deliverables (what judges should look at)

- `fault_detection_solution.py` — single-run end-to-end pipeline
- `FINAL.csv` — submission file in required format (`ID`, `CLASS`) and correct test order
- `run_summary.json` — automatically saved run metadata (seeds/folds/toggles/best threshold/OOF F1/runtime)

If you run the full stack and enable optional modules:

- `optuna_logs/` — Optuna trials + saved studies
- `shap_importance.png` / `shap_importance.csv` — interpretability artifacts

## 1. Problem understanding

Given 47 automatically generated detector features (`F01`–`F47`), the task is to predict whether the device is operating normally or in a faulty state.

- **Class 0**: Normal
- **Class 1**: Faulty

This is a supervised binary classification problem.

## 2. Dataset specification

### 2.1 Files

- `TRAIN.csv`
  - columns: `F01..F47` and `Class`
  - may optionally contain `ID` (the script will ignore it if present)
- `TEST.csv`
  - columns: `F01..F47` and **must** contain `ID`

### 2.2 Schema assumptions

- All `Fxx` features are numeric.
- `Class` exists only in `TRAIN.csv`.
- `ID` exists in `TEST.csv` and is used to create the submission.

## 3. Methodology (end-to-end pipeline)

The script follows a competitive tabular ML flow:

1) load data
2) engineer features
3) multi-seed stratified K-fold OOF training (robust multi-seed averaging)
4) base models + stacking meta-learner
5) optional adversarial validation (shift detection)
6) optional OOF-based blending weight optimization
7) optional isotonic calibration
8) threshold tuning for F1-score
9) generate `FINAL.csv` for `TEST.csv`

## 4. Feature engineering

All feature engineering is computed **row-wise**, so it can be applied consistently to both train and test without leakage.

In addition to mean/std/range/skew/kurtosis and group stats, the script adds a small set of domain-style “signal summary” features:

- energy proxies: `l1_energy`, `l2_energy`
- quantiles + dispersion: `q10_all`, `q25_all`, `q75_all`, `q90_all`, `iqr_all`
- sign structure: `frac_pos`, `frac_neg`
- simple outlier counters: `n_outlier_gt2`, `n_outlier_gt3`

### 4.1 Global row statistics

Computed across `F01..F47`:

- `mean_all`
- `std_all`
- `max_all`
- `min_all`
- `range_all`
- `skew_all`
- `kurt_all`

### 4.2 Group statistics

Sensors are partitioned into three contiguous groups:

- Group A: `F01`–`F16`
- Group B: `F17`–`F32`
- Group C: `F33`–`F47`

For each group the same 7 distributional statistics are computed.

### 4.3 Cross-group ratios

To capture subsystem imbalance, ratio features are added, for example:

- `mean_A / (mean_B + 1e-5)`
- `std_A / (std_C + 1e-5)`

### 4.4 Pairwise interactions (selected raw features)

The script creates multiply/divide interaction terms for a small set of raw features (e.g. `F01`, `F10`, `F08`, `F09`, `F06`, `F07`).

## 5. Modeling

### 5.1 Base models

The script trains an OOF backbone with the following estimators (availability depends on installed libraries):

- `RandomForestClassifier` (always available)
- `XGBClassifier` (optional)
- `LGBMClassifier` (optional)
- `TabNetClassifier` (optional)

### 5.2 Tree ensembles

The script can use the following base estimators:

- `RandomForestClassifier` (always available via scikit-learn)
- `XGBClassifier` (optional; requires `xgboost`)
- `LGBMClassifier` (optional; requires `lightgbm`)

Class imbalance is addressed via built-in weighting strategies (for example `class_weight='balanced'` in sklearn and `scale_pos_weight` in XGBoost).

### 5.2 Stacking ensemble

For additional performance and robustness, a stacking meta-learner (`LogisticRegression`) is trained on OOF predictions from the available base estimators.

If `xgboost` or `lightgbm` are not installed, the ensemble automatically adapts and trains with the remaining estimators.

## 6. Evaluation & threshold tuning

### 6.1 Validation protocol

The main validation signal comes from **out-of-fold (OOF)** predictions created by multi-seed stratified K-fold training.

### 6.2 F1-optimized threshold tuning

Instead of using a fixed decision threshold of `0.5`, the script:

- generates out-of-fold probabilities using stratified K-fold cross-validation
- sweeps candidate thresholds
- selects the threshold that maximizes **F1-score**

The selected threshold is applied to the final test probabilities to produce binary labels.

## 7. Robustness safeguards

To make the pipeline stable across environments and datasets:

- `ID` is ignored in training if present.
- `TEST.csv` must contain an `ID` column (hard check).
- `inf/-inf` values are replaced with `NaN`.
- Remaining missing values are filled using **training medians**.

## 8. How to run

### 8.1 Install

```bash
pip install -r requirements.txt
```

For the full research stack (optional, enables XGBoost/LightGBM/Optuna/SHAP and plotting utilities):

```bash
pip install -r requirements-full.txt
```

### 8.2 Data placement

Place `TRAIN.csv` and `TEST.csv` in the same folder as `fault_detection_solution.py`.

### 8.3 Optional environment variables

You can override dataset paths using:

- `TRAIN_PATH` (default: `TRAIN.csv`)
- `TEST_PATH` (default: `TEST.csv`)

Example (Windows PowerShell):

```bash
$env:TRAIN_PATH="C:\path\to\TRAIN.csv"
$env:TEST_PATH="C:\path\to\TEST.csv"
python fault_detection_solution.py
```

Additional environment variables supported by the script (see `.env.example` for defaults):

- `FAST_SMOKE_TEST` (true/false)
  - `true`: fast sanity run (fewer seeds/folds, smaller RF, adversarial validation off by default)
  - `false`: full run (multi-seed + full CV; slower)
- `SEEDS` (comma-separated integers)
- `N_SPLITS` (integer)
- `ENABLE_ADVERSARIAL_VALIDATION` (true/false)
- `ENABLE_CALIBRATION` (true/false)
- `ENABLE_PSEUDO_LABELING_TOP1` (true/false)
- `ENABLE_WEIGHT_OPTIMIZATION` (true/false)
- `WEIGHT_SEARCH_ITERS` (integer; random search iterations)
- `ENABLE_CV_STABILITY_REPORT` (true/false)
- `ENABLE_OPTUNA` (true/false; enables Optuna tuning logs if Optuna + model libs installed)
- `ENABLE_SHAP` (true/false)
- `ENABLE_SHAP_REFINEMENT` (true/false)
- `SHAP_TOP_N` (integer)
- `PSEUDO_POS_TH` / `PSEUDO_NEG_TH` (floats; used only if pseudo labeling enabled)
- `XGB_WEIGHT`, `LGBM_WEIGHT`, `RF_WEIGHT`, `TABNET_WEIGHT`, `META_WEIGHT` (floats)

Note: by default, the script reads these values from the process environment. If you use a `.env` file locally, load it into your shell before running.

Recommended run modes:

- **Judge verification**: keep `FAST_SMOKE_TEST=true` (fast end-to-end) and run once.
- **Leaderboard / final submission**: set `FAST_SMOKE_TEST=false`, install `requirements-full.txt`, and optionally enable `ENABLE_WEIGHT_OPTIMIZATION`, `ENABLE_OPTUNA`, and `ENABLE_SHAP`.

### 8.4 Run

```bash
python fault_detection_solution.py
```

## Repository structure

```text
FAULT DETECTION/
├── fault_detection_solution.py
├── README.md
├── readme.txt
├── requirements.txt
├── requirements-full.txt
├── .env.example
└── .gitignore
```

Notes:

- `TRAIN.csv` and `TEST.csv` are local dataset files and should not be committed.
- `FINAL.csv`, `final_model.pkl` and generated plots/reports are treated as local artifacts (see `.gitignore`).

## 9. Submission file (FINAL.csv)

The script produces `FINAL.csv` in the same folder.

### 9.1 Required format

- exactly **2 columns**: `ID`, `CLASS`
- number of rows equals `TEST.csv`
- row order matches `TEST.csv` exactly
- `CLASS` contains only `0` or `1`

Example:

```text
ID,CLASS
1,1
2,0
3,0
```

### 9.2 Built-in checks

The script asserts:

- row count matches `TEST.csv`
- output labels are valid (`0/1`)
- `ID` order matches input test order

## 10. Optional components

The following switches exist in `fault_detection_solution.py` and are disabled/enabled as configured in the script:

- `ENABLE_PSEUDO_LABELING_TOP1` (1-iteration pseudo labeling using 0.95/0.05 confidence)
- `ENABLE_ADVERSARIAL_VALIDATION` (train vs test shift detection; drops top shift features if AUC > 0.7)
- `ENABLE_CALIBRATION` (IsotonicRegression probability calibration)

Additional advanced components (implemented; optional):

- `ENABLE_WEIGHT_OPTIMIZATION` (OOF-based random search for blend weights)
- `ENABLE_CV_STABILITY_REPORT` (reports mean/std F1 across folds for each base model)
- `ENABLE_OPTUNA` (Optuna tuning for XGBoost/LightGBM; saves logs under `optuna_logs/`)
- `ENABLE_SHAP` (saves `shap_importance.png` and `shap_importance.csv` when deps available)
- `ENABLE_SHAP_REFINEMENT` (re-runs training using top `SHAP_TOP_N` features)

The script also supports a fast-vs-full mode:

- `FAST_SMOKE_TEST=true` is intended for quick verification that the pipeline runs end-to-end.
- For best leaderboard performance, use `FAST_SMOKE_TEST=false` and install the full stack from `requirements-full.txt`.

Additional hooks/flags are present for experimentation:

- `ENABLE_SHAP` (requires SHAP + a tree model; not enabled by default)
- `ENABLE_EDA` (EDA artifact generation; not enabled by default)

## 11. Generated artifacts

After a successful run:

- `FINAL.csv`
- `final_model.pkl`
- `run_summary.json`

If enabled and dependencies are present:

- `optuna_logs/` (Optuna trial logs + studies)
- `shap_importance.png`
- `shap_importance.csv`

Notes:

- `run_summary.json` is the easiest “judge-proof” artifact to show what configuration was used.
- `final_model.pkl` stores meta-learner + feature drop list + calibration object (if used) + optimized blend info (if used).

## 12. Reproducibility

- Set via `RANDOM_STATE` in `fault_detection_solution.py`.
- The script prints dataset shapes, class distribution, and validation summary.

## 13. Limitations (honest notes)

- This repository is optimized for the competition metric (F1-score) rather than a specific real-world cost function.
- Training and threshold tuning can be CPU-heavy depending on available libraries and hardware.
- The feature engineering is implemented in code, not packaged as a single serialized pipeline object.

## 14. Troubleshooting

- **`ModuleNotFoundError: No module named 'xgboost'` / `'lightgbm'`**
  - install the package(s) or run without them (the script will skip them).
- **`KeyError: TEST.csv must contain an 'ID' column`**
  - ensure `TEST.csv` includes `ID` exactly.
- **Run time is high**
  - stacking + cross-validation threshold tuning can take time on CPU.
