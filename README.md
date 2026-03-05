 # Machine Fault Detection — ML alrIEEEna’26

 ## Overview

 This repository presents a structured, competition-ready solution for **binary machine fault detection** using a tabular sensor dataset consisting of **47 numerical features** (`F01`–`F47`).

 The implementation is designed for:

 - strong generalization on hidden evaluation data
 - reproducibility across environments
 - modular experimentation (feature engineering, model stacking, calibration, blending)
 - metric-aligned optimization (**F1-score**)

 The complete pipeline is implemented in a single entry point:

 - `fault_detection_solution.py`

 ## Quick Start (Judge Execution)

 ### 1. Dataset Placement

 Place the following files in the same directory as the script:

- `TRAIN.csv`
- `TEST.csv`

### 2. Install Dependencies

**Option 1: Minimal stack (always works)**
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 joblib>=1.3.0
```

**Option 2: Use requirements file**
```bash
pip install -r requirements.txt  # Minimal core dependencies
pip install -r requirements-full.txt  # Full research stack
```

**Option 3: Full research stack (recommended for best performance)**
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 joblib>=1.3.0 \
xgboost>=1.7.0 lightgbm>=4.0.0 catboost>=1.2.0 optuna>=3.0.0 \
shap>=0.42.0 imbalanced-learn>=0.11.0 matplotlib>=3.7.0 seaborn>=0.12.0 \
python-dotenv>=1.0.0 torch>=2.0.0 pytorch-tabnet>=4.1.0
```

### 3. Execute

```bash
python fault_detection_solution.py
```

### 4. Generated Outputs

After execution, the following artifacts are produced:

- `FINAL.csv` — submission file (`ID`, `CLASS`)
- `final_model.pkl` — serialized model snapshot
- `run_summary.json` — configuration, validation metrics, runtime details

### Runtime Expectations (CPU)

**For Judges (FAST_SMOKE_TEST=true - default)**:
- Runtime: ~2-5 minutes
- Uses 1 seed, 3 folds, reduced RF estimators
- Produces valid submission quickly

**For Full Performance (FAST_SMOKE_TEST=false)**:
- Runtime: ~30-60 minutes depending on hardware
- Uses 4 seeds, 5 folds, full model capacity
- Recommended for final leaderboard submission

**Note**: FAST_SMOKE_TEST defaults to `true` for quick judge execution. Set to `false` for maximum performance.

 ## Problem Definition

 Given 47 automatically generated sensor features (`F01`–`F47`), the objective is to classify machine state:

 - **Class 0** — Normal operation
 - **Class 1** — Fault condition

 This is a supervised binary classification problem evaluated using **Accuracy and F1 Scores**. Rankings are strictly determined by these metrics on a hidden split.

 ## Dataset Specification

 ### Input Files

 #### `TRAIN.csv`

 - Columns: `F01`–`F47`, `Class`
 - Optional: `ID` (ignored if present)

 #### `TEST.csv`

 - Columns: `F01`–`F47`
 - Required: `ID` (used for submission generation)

 ### Assumptions

 - All feature columns are numeric.
 - Target column `Class` exists only in training data.
 - Test file order must be preserved in submission.

 ## Methodology

 The solution follows a structured tabular ML pipeline:

 1) Data loading and schema validation
 2) Deterministic feature engineering
 3) Multi-seed stratified K-fold cross-validation
 4) Base model training
 5) Stacking meta-learning
 6) Optional adversarial validation
 7) Optional blend weight optimization
 8) Optional probability calibration
 9) F1-optimized threshold selection
 10) Submission file generation

 The design emphasizes robustness, reproducibility, and competition-grade evaluation discipline.

 ## Feature Engineering

 All engineered features are computed **row-wise** to prevent leakage and ensure consistent application to train and test sets.

 ### Global Distributional Statistics

 Computed across all 47 features:

 - `mean_all`
 - `std_all`
 - `max_all`
 - `min_all`
 - `range_all`
 - `skew_all`
 - `kurt_all`

 ### Group-Based Statistics

 Sensors are partitioned into contiguous groups:

 - Group A: `F01`–`F16`
 - Group B: `F17`–`F32`
 - Group C: `F33`–`F47`

 For each group, identical distributional statistics are computed to capture subsystem behavior.

 ### Cross-Group Ratios

 To model subsystem imbalance:

 - mean ratios (e.g., `A/B`)
 - standard deviation ratios (e.g., `A/C`)

 Small constants are added for numerical stability.

 ### Signal Summary Features

 Additional engineered signals:

 - Energy proxies: `l1_energy`, `l2_energy`
 - Quantiles: `q10_all`, `q25_all`, `q75_all`, `q90_all`
 - Dispersion: `iqr_all`
 - Sign structure: `frac_pos`, `frac_neg`
 - Outlier counters: `n_outlier_gt2`, `n_outlier_gt3`

 ### Selected Pairwise Interactions

Multiplicative and ratio interactions are generated for a controlled subset of raw features to enhance nonlinear expressiveness without excessive dimensionality growth.

## Modeling Strategy

### Base Models

Depending on installed libraries, the following estimators are used:

- `RandomForestClassifier` (baseline; always available)
- `XGBClassifier` (optional; with early stopping)
- `LGBMClassifier` (optional; with early stopping)
- `CatBoostClassifier` (optional; with early stopping)
- `MLPClassifier` (optional; neural network with early stopping)
- `TabNetClassifier` (optional)

Class imbalance is handled via:

- `class_weight='balanced'` (scikit-learn)
- `scale_pos_weight` (XGBoost)
- `is_unbalance=True` (LightGBM)
- `scale_pos_weight` (CatBoost)
- `class_weight='balanced'` (MLP)
- equivalent strategies in TabNet

### Stacking Ensemble

Out-of-fold predictions from base learners are used to train a meta-learner:

- `LogisticRegression`

 The ensemble adapts dynamically if optional libraries are not installed.

 ## Validation & Metric Optimization

 ### Cross-Validation Protocol

 - Multi-seed
 - Stratified K-fold
 - Out-of-fold (OOF) probability aggregation

 OOF predictions form the primary validation signal.

 ### Threshold Optimization

 Rather than using a fixed `0.5` cutoff:

 - Candidate thresholds are swept over OOF probabilities
 - The threshold maximizing F1-score is selected
 - The selected threshold is applied to test probabilities

 This ensures direct optimization toward the competition metric.

 ## Robustness Mechanisms

 - Strict schema validation (`TEST.csv` must contain `ID`)
 - `ID` excluded from training features
 - `inf/-inf` replaced with `NaN`
 - Missing values imputed using training medians
 - Automatic adaptation when optional libraries are unavailable

 ## Environment Configuration

The script supports runtime configuration via environment variables.

Key flags:

- `FAST_SMOKE_TEST` (default: true) - Quick execution for judges
- `SEEDS`
- `N_SPLITS`
- `ENABLE_ADVERSARIAL_VALIDATION`
- `ENABLE_CALIBRATION`
- `ENABLE_WEIGHT_OPTIMIZATION`
- `ENABLE_OPTUNA`
- `ENABLE_SHAP`
- `ENABLE_SHAP_REFINEMENT`
- `ENABLE_PSEUDO_LABELING_TOP1`
- `ENABLE_FEATURE_IMPORTANCE_FILTER`
- `FEATURE_IMPORTANCE_TOP_N`
- `ENABLE_PLOTS`

Dataset path overrides:

- `TRAIN_PATH`
- `TEST_PATH`

Example (PowerShell):

```bash
$env:TRAIN_PATH="C:\path\TRAIN.csv"
$env:TEST_PATH="C:\path\TEST.csv"
python fault_detection_solution.py
```

 ## Submission File Specification

 The script generates:

 - `FINAL.csv`

 ### Required Format

 - Exactly 2 columns: `ID`, `CLASS`
 - Same number of rows as `TEST.csv`
 - Row order strictly preserved
 - `CLASS` ∈ {0, 1}

 ### Built-In Validations

 - Row count assertion
 - Label validity check
 - ID order verification

 ## Optional Advanced Components

When enabled and dependencies are available:

- Adversarial validation (train–test shift detection)
- Pseudo-labeling (high-confidence augmentation)
- Isotonic probability calibration
- OOF-based blend weight optimization
- Optuna hyperparameter tuning (logs stored in `optuna_logs/`)
- SHAP-based feature importance artifacts
- SHAP-driven feature refinement
- CV stability reporting
- Feature importance filtering (RF-based selection)
- Automated plots (confusion matrix, ROC, feature importance)
- CatBoost integration with early stopping
- MLP neural network ensemble diversity

 ## Generated Artifacts

 Core:

- `FINAL.csv`
- `final_model.pkl`
- `run_summary.json`

Optional:

- `optuna_logs/`
- `shap_importance.png`
- `shap_importance.csv`
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

`run_summary.json` serves as a reproducibility record containing:

- fold configuration
- seed configuration
- best threshold
- OOF F1-score
- runtime

 ## Reproducibility

 - Controlled via `RANDOM_STATE`
 - Deterministic CV splits
 - Configuration snapshot saved automatically

 ## Limitations

 - Optimized strictly for F1-score rather than domain-specific cost functions
 - Full-stack execution can be computationally intensive on CPU
 - Feature engineering is implemented procedurally (not serialized as a standalone pipeline object)

 ## Troubleshooting

 ### Missing optional libraries

 **`ModuleNotFoundError: No module named 'xgboost'` / `'lightgbm'`**

 Install via:

 ```bash
 pip install -r requirements-full.txt
 ```

 or run with available base models.

### Missing CatBoost

**`ModuleNotFoundError: No module named 'catboost'`**

Install via:

```bash
pip install catboost
```

CatBoost unavailable hone par pipeline automatically remaining models ke saath chalta rehta hai.

 ### Missing ID column

 **`KeyError: TEST.csv must contain an 'ID' column`**

 Ensure `TEST.csv` includes `ID` exactly (case-sensitive).

 ### High runtime

 Full CV + stacking + threshold optimization is CPU-bound. Use:

 - `FAST_SMOKE_TEST=true`

 for quick verification runs.

 ## Created by

 Created by **Team Null Syndicate**

 - Member 1: **Anmol Bahuguna**
 - Member 2: **Ayush Semwal**
