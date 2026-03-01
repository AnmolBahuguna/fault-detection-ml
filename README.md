# ML alrIEEEna'26 — Machine Fault Detection (Binary Classification)
**IEEE Student Branch | Graphic Era Hill University, Dehradun**

Research-oriented solution write-up for **Machine Fault Detection** using 47 continuous sensor features. The approach follows standard best practices for tabular binary classification: stratified validation, feature engineering driven by distributional statistics, and tree-based ensembles with careful probability post-processing for F1 optimization.

Submission Deadline: **5 March 2026**  \
Grand Finale: **20 March 2026**

---

## 1. Problem Statement
A manufacturing/industrial environment contains embedded devices continuously monitored by **47 sensors**. Each device cycle produces **47 numerical readings** (`F01`–`F47`).

Your task is to build an ML model that classifies each sample as:
- **Class 0**: Normal
- **Class 1**: Faulty

### 1.1 Formal Definition

| Parameter | Detail |
|---|---|
| Task Type | Binary Classification (Supervised) |
| Input | 47 continuous numerical features (`F01`–`F47`) |
| Output | `Class` (0 = Normal, 1 = Faulty) |
| Evaluation | **F1 Score (Primary)** + Accuracy (Secondary) |
| Train Samples | 43,776 |
| Test Samples | 10,944 |
| Missing Values | 0 (no imputation needed) |
| Class Distribution | ~60.5% Normal, ~39.5% Faulty |

### 1.2 Why F1 Score (Not Just Accuracy)
If you predict everything as **Normal**, you get ~60.5% accuracy but miss all faults.

**F1 Score** = harmonic mean of **Precision** and **Recall**. This project emphasizes **generalization** by optimizing decisions against F1 (rather than relying on accuracy).

### 1.3 Key Constraints
- **Generalization**: avoid overfitting to validation.
- **Tie-breaker**: code quality, structure, and optimization matter.
- **Imbalance**: mild 60/40 imbalance; **SMOTE is not recommended**.
- **Noise features**: some sensors may be weak; feature engineering + selection helps.

---

## 1.4 Evaluation Protocol (Recommended)
To obtain a reliable estimate of generalization performance and reduce threshold overfitting, prefer:
- **Stratified K-Fold CV** (e.g., 5 folds) for model selection.
- Compute **out-of-fold probabilities** for threshold tuning.
- Report both:
  - **F1 score** (primary)
  - **Precision/Recall** (to understand failure modes)

Note: single train/validation splits are convenient but can produce optimistic estimates and unstable thresholds.

---

## 2. Dataset Analysis

### 2.1 Structure Overview

| Property | `TRAIN.csv` | `TEST.csv` |
|---|---:|---:|
| Rows | 43,776 | 10,944 |
| Columns | 48 (47 features + `Class`) | 48 (47 features + `ID`) |
| Types | float64 (continuous) | float64 (continuous) |
| Missing Values | 0 | 0 |

### 2.2 Class Distribution (Train)

| Class | Label | Percentage |
|---:|---|---:|
| 0 | Normal | ~60.5% |
| 1 | Faulty | ~39.5% |

Insight: mild imbalance (~1.53:1). Prefer:
- `scale_pos_weight` in XGBoost
- `class_weight` in sklearn models
- `is_unbalance` / class weights in LightGBM

### 2.3 Feature Characteristics
- All 47 inputs are continuous floats (no encoding needed).
- Ranges can vary; scaling is mainly useful for linear / NN models.
- Outliers may exist; robust strategies can help (optional).

---

## 3. Complete Solution Architecture

### 3.1 High-Level Pipeline

| Phase | Step | Tool/Method | Output |
|---:|---|---|---|
| 1 | Data Loading & Basic EDA | Pandas | Clean DataFrames |
| 2 | Feature Engineering | Pandas + NumPy | ~96 engineered features |
| 3 | Preprocessing | Stratified split / CV | Leak-free train/val |
| 4 | Baseline | Logistic Regression | Initial F1 benchmark |
| 5 | Base Models | XGBoost + LightGBM + RF | Strong individual models |
| 6 | Tuning | Optuna TPE | F1-optimized configs |
| 7 | Ensemble | StackingClassifier | Best combined model |
| 8 | Calibration | CalibratedClassifierCV | Better probabilities |
| 9 | Pseudo Labeling | High-confidence test samples | Expanded training |
| 10 | Threshold Tuning | F1 sweep | Best decision boundary |
| 11 | Explainability | SHAP | Feature importance |
| 12 | Submission | CSV export | `FINAL.csv` |

### 3.2 Three-Tier Model Architecture

| Tier | Models | Purpose |
|---:|---|---|
| 1 | XGBoost + LightGBM + Random Forest | Diverse strong learners |
| 2 | Calibration (Platt scaling) | Probability reliability |
| 3 | Stacking meta-learner (LogReg) | Learns which model to trust |

---

## 4. Feature Engineering (Highest Impact)
Raw `F01`–`F47` capture individual sensors. Engineered features capture **relationships** and **anomaly signatures**.

Implementation notes:
- All engineered features are computed **row-wise**, so they are safe to apply identically to train/test without needing any fitted statistics.
- Avoid target leakage by ensuring that any feature selection (e.g., SHAP pruning) is performed only using **training folds**.

### 4.1 Global Row-wise Statistics (7 features)

| Feature | Formula (per row) | Why it helps |
|---|---|---|
| `mean_all` | mean(`F01:F47`) | average health |
| `std_all` | std(`F01:F47`) | spread/anomaly |
| `max_all` | max(`F01:F47`) | spike faults |
| `min_all` | min(`F01:F47`) | drop faults |
| `range_all` | `max_all - min_all` | magnitude |
| `skew_all` | skew(`F01:F47`) | asymmetry |
| `kurt_all` | kurtosis(`F01:F47`) | extreme events |

### 4.2 Sensor Group Statistics (21 features)
Split sensors into 3 groups:
- **Group A**: `F01`–`F16`
- **Group B**: `F17`–`F32`
- **Group C**: `F33`–`F47`

For each group compute the same 7 stats:
`mean_{G}`, `std_{G}`, `max_{G}`, `min_{G}`, `range_{G}`, `skew_{G}`, `kurt_{G}`.

### 4.3 Cross-Group Ratio Features (6 features)
Ratios highlight subsystem imbalance:
- `mean_A / (mean_B + 1e-5)`
- `mean_B / (mean_C + 1e-5)`
- `mean_A / (mean_C + 1e-5)`
- `std_A / (std_B + 1e-5)`
- `std_B / (std_C + 1e-5)`
- `std_A / (std_C + 1e-5)`

### 4.4 Top Pairwise Sensor Interactions (~15 features)
After initial model + SHAP, take top 5–6 raw features and create interaction terms:
- multiply pairs: `F_topi * F_topj`
- ratio pairs: `F_topi / (F_topj + 1e-5)`

Total: `47` raw + `7` global + `21` group + `6` ratios + `~15` interactions ≈ **96 features**.

---

## 5. Tech Stack

### 5.1 One-command Install
```bash
pip install -r requirements.txt
```

### 5.2 Libraries

| Library | Purpose | Priority |
|---|---|---|
| pandas | load/clean/explore | must |
| numpy | feature engineering | must |
| scikit-learn | CV, metrics, stacking, calibration | must |
| xgboost | strong tabular model | must |
| lightgbm | fast boosting alternative | must |
| optuna | Bayesian tuning (TPE) | must |
| shap | explainability | must |
| matplotlib, seaborn | plotting | good |
| joblib | model persistence | must |

---

## 6. Step-by-Step Execution Plan

### Phase 1: Environment & Data Loading
1. Install dependencies
2. Load `TRAIN.csv` and `TEST.csv`
3. Verify shape, dtypes, missing values, class distribution

### Phase 2: Feature Engineering
4. Add global row stats
5. Add group stats (A/B/C)
6. Add cross-group ratios
7. (Optional) Add SHAP-driven interaction features

### Phase 3: Preprocessing
8. Split train/val with stratification (or stratified CV)
9. Avoid SMOTE for this 60/40 setting; prefer class weights

### Phase 4–6: Models + Optuna
10. Baseline Logistic Regression
11. Train XGBoost / LightGBM / Random Forest
12. Run Optuna (e.g., 100 trials) optimizing F1
13. Compare models and keep best configs

### Phase 7–10: Ensemble + Calibration + Pseudo Labeling + Threshold
14. StackingClassifier (meta = Logistic Regression)
15. Calibrate probabilities before pseudo labeling
16. Pseudo label **only high-confidence** test predictions (e.g., >0.90 / <0.10)
17. Tune decision threshold to maximize validation F1

### Phase 11–12: SHAP + Submission
18. Produce SHAP importance plot
19. Generate `FINAL.csv` with exactly 2 columns: `ID`, `Class`

---

## 7. Expected F1 by Approach (Reference)

| Approach | Expected F1 |
|---|---:|
| Basic RF, no FE | ~0.80–0.83 |
| Standard XGBoost | ~0.85–0.88 |
| Tuned XGB/LGBM | ~0.88–0.91 |
| Stacking + FE | ~0.91–0.93 |
| Full pipeline (Stack + PL + threshold + SHAP) | ~0.93–0.96 |

---

## 8. Competitive Differentiators

| Technique | Common baseline | This repository |
|---|---|---|
| Feature Engineering | raw 47 only | 96+ stats/ratios/interactions |
| Imbalance handling | SMOTE / ignore | weights (`scale_pos_weight`, `class_weight`) |
| Tuning | manual | Optuna TPE (F1-optimized) |
| Ensemble | voting | stacking meta-learner |
| Decision boundary | 0.5 | threshold optimized for F1 |
| Explainability | none | SHAP plots + pruning |
| Submission quality | messy script | modular + saved model |

---

## 9. Modeling Details

### 9.1 Baseline
- **Logistic Regression** is used as a sanity check baseline.
- Use `class_weight='balanced'` if needed.

### 9.2 Tree Ensembles (Primary)
- **XGBoost** / **LightGBM** are strong defaults for dense tabular features.
- Handle imbalance via:
  - `scale_pos_weight` (XGBoost)
  - `is_unbalance=True` or explicit class weights (LightGBM)

### 9.3 Stacking
- A **StackingClassifier** can improve robustness by combining diverse base learners.
- Prefer generating meta-features from **out-of-fold** predictions to minimize leakage.

---

## 10. Probability Post-processing

### 10.1 Calibration
Tree ensembles can output poorly calibrated probabilities.
- Use **Platt scaling** (`CalibratedClassifierCV(method='sigmoid')`) to improve probability reliability.
- Calibrate on a validation fold (or via CV), not on the final test set.

### 10.2 Threshold Tuning (F1)
The default decision threshold `0.5` is not guaranteed to maximize F1.
- Sweep thresholds on validation **probabilities**.
- Select `threshold*` that maximizes validation F1.
- For stability, tune thresholds using **OOF probabilities** from stratified CV.

---

## 11. Pseudo Labeling (Optional)
Pseudo labeling can help when test distribution is similar to training distribution, but it can also degrade performance if noisy.

Recommended safeguards:
- Only accept **high-confidence** predictions (e.g., `p>0.90` or `p<0.10`).
- Use calibrated probabilities when selecting pseudo labels.
- Always validate that pseudo labeling improves F1 on a held-out fold before using it for final training.

---

## 12. Repository Structure

```text
FAULT DETECTION/
├── fault_detection_solution.py
├── requirements.txt
├── README.md
├── FINAL.csv               (generated)
├── final_model.pkl         (generated)
├── scaler.pkl              (generated)
└── shap_importance.png     (generated)
```

---

## 10. How to Run (This Repo)

1. Put `TRAIN.csv` and `TEST.csv` in the same folder as `fault_detection_solution.py`

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run
```bash
python fault_detection_solution.py
```

Output:
- `FINAL.csv` (ready to submit)
- `final_model.pkl`, `scaler.pkl`
- `shap_importance.png` (if SHAP runs successfully)

---

## 11. Reproducibility
Recommendations for consistent results:
- Fix `random_state` for splits and models.
- Log:
  - dataset shapes
  - class distribution
  - chosen hyperparameters
  - selected threshold
- Keep library versions pinned (update `requirements.txt` if you change environments).

## 13. Final Submission Checklist

### `FINAL.csv` requirements
- Exactly **2 columns**: `ID`, `Class`
- Exactly **10,944 rows**
- `Class` contains only **0** and **1**
- Row order matches `TEST.csv` `ID` order

### Tie-breaker advantage
If F1 scores tie, evaluations may consider:
- architecture elegance
- optimization choices
- reproducibility
- explainability artifacts

---

**IEEE Student Branch | Graphic Era Hill University, Dehradun | March 2026**
