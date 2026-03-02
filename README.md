# Machine Fault Detection (ML alrIEEEna'26)

## Abstract

This repository provides a research-oriented, end-to-end solution for **binary machine fault detection** using an embedded monitoring dataset with **47 numerical input features** (`F01`–`F47`).

The pipeline is designed for strong generalization on hidden evaluation data and includes:

- feature engineering via row-wise distributional statistics
- multiple strong tabular models (tree ensembles + linear baseline)
- class imbalance handling
- probability-to-label conversion via **threshold tuning for F1-score**
- generation of a submission file `FINAL.csv` in the required format (`ID`, `CLASS`)

The main entry-point is a single script: `fault_detection_solution.py`.

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

The script follows a standard tabular ML flow:

1) load data
2) engineer features
3) split train/validation with stratification
4) train baseline + ensemble models
5) estimate probabilities and tune a classification threshold for F1-score
6) fit the selected model on full training data
7) generate `FINAL.csv` for `TEST.csv`

## 4. Feature engineering

All feature engineering is computed **row-wise**, so it can be applied consistently to both train and test without leakage.

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

### 5.1 Baseline

- **Logistic Regression** is used as a sanity-check baseline (with scaling).

### 5.2 Tree ensembles

The script can use the following base estimators:

- `RandomForestClassifier` (always available via scikit-learn)
- `XGBClassifier` (optional; requires `xgboost`)
- `LGBMClassifier` (optional; requires `lightgbm`)

Class imbalance is addressed via built-in weighting strategies (for example `class_weight='balanced'` in sklearn and `scale_pos_weight` in XGBoost).

### 5.3 Stacking ensemble

For additional performance and robustness, a `StackingClassifier` is trained using the available base estimators, with a `LogisticRegression` meta-learner.

If `xgboost` or `lightgbm` are not installed, the ensemble automatically adapts and trains with the remaining estimators.

## 6. Evaluation & threshold tuning

### 6.1 Validation split

A stratified train/validation split is used for quick feedback.

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

### 8.4 Run

```bash
python fault_detection_solution.py
```

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

The following features exist in code but are disabled by default:

- `ENABLE_OPTUNA` (hyperparameter tuning; slower)
- `ENABLE_PSEUDO_LABELING` (optional; can hurt if misused)
- `ENABLE_SHAP` (explainability; requires SHAP + XGBoost)

## 11. Generated artifacts

After a successful run:

- `FINAL.csv`
- `final_model.pkl`
- `scaler.pkl`

If SHAP is enabled and dependencies are present:

- `shap_importance.png`

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
