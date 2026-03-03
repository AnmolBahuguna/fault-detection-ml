# ============================================================
#  ML alrIEEEna'26 — Machine Fault Detection
#  Complete Solution Script | IEEE SB GEHU
#  Goal: Maximize F1 Score on hidden test data
#  Output: FINAL.csv (ID -> Class predictions)
# ============================================================

# ── STEP 0: INSTALL (run this once in terminal) ──────────────
# pip install pandas numpy scikit-learn xgboost lightgbm optuna shap imbalanced-learn matplotlib seaborn

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_TRIALS_XGB = 100
N_TRIALS_LGBM = 100
ENABLE_OPTUNA = False
ENABLE_PSEUDO_LABELING = False
ENABLE_SHAP = False

# ── STEP 1: LOAD DATA ────────────────────────────────────────
print("="*60)
print("STEP 1: Loading Data")
print("="*60)

TRAIN_PATH = os.getenv('TRAIN_PATH', 'TRAIN.csv')
TEST_PATH = os.getenv('TEST_PATH', 'TEST.csv')

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")
print(f"Missing (train): {train.isnull().sum().sum()}")
print(f"Missing (test) : {test.isnull().sum().sum()}")
print(f"\nClass Distribution:")
print(train['Class'].value_counts())
print(f"Class 0 %: {train['Class'].value_counts(normalize=True)[0]*100:.1f}%")
print(f"Class 1 %: {train['Class'].value_counts(normalize=True)[1]*100:.1f}%")

# ── STEP 2: FEATURE ENGINEERING ──────────────────────────────
print("\n" + "="*60)
print("STEP 2: Feature Engineering")
print("="*60)

FEATURES = [f'F{str(i).zfill(2)}' for i in range(1, 48)]  # F01 to F47

def engineer_features(df):
    df = df.copy()
    feats = FEATURES

    # --- Global Row-wise Stats (7 features) ---
    df['mean_all']  = df[feats].mean(axis=1)
    df['std_all']   = df[feats].std(axis=1)
    df['max_all']   = df[feats].max(axis=1)
    df['min_all']   = df[feats].min(axis=1)
    df['range_all'] = df['max_all'] - df['min_all']
    df['skew_all']  = df[feats].skew(axis=1)
    df['kurt_all']  = df[feats].kurtosis(axis=1)

    # --- Group-level Stats (3 groups x 7 stats = 21 features) ---
    groupA = [f for f in feats if int(f[1:]) <= 16]   # F01-F16
    groupB = [f for f in feats if 17 <= int(f[1:]) <= 32]  # F17-F32
    groupC = [f for f in feats if int(f[1:]) >= 33]   # F33-F47

    for name, grp in [('A', groupA), ('B', groupB), ('C', groupC)]:
        df[f'mean_{name}']  = df[grp].mean(axis=1)
        df[f'std_{name}']   = df[grp].std(axis=1)
        df[f'max_{name}']   = df[grp].max(axis=1)
        df[f'min_{name}']   = df[grp].min(axis=1)
        df[f'range_{name}'] = df[f'max_{name}'] - df[f'min_{name}']
        df[f'skew_{name}']  = df[grp].skew(axis=1)
        df[f'kurt_{name}']  = df[grp].kurtosis(axis=1)

    # --- Cross-Group Ratios (6 features) ---
    df['ratio_A_B'] = df['mean_A'] / (df['mean_B'] + 1e-5)
    df['ratio_B_C'] = df['mean_B'] / (df['mean_C'] + 1e-5)
    df['ratio_A_C'] = df['mean_A'] / (df['mean_C'] + 1e-5)
    df['std_ratio_A_B'] = df['std_A'] / (df['std_B'] + 1e-5)
    df['std_ratio_B_C'] = df['std_B'] / (df['std_C'] + 1e-5)
    df['std_ratio_A_C'] = df['std_A'] / (df['std_C'] + 1e-5)

    # --- Top Pairwise Interactions (based on domain knowledge) ---
    top_feats = ['F01', 'F10', 'F08', 'F09', 'F06', 'F07']
    for i in range(len(top_feats)):
        for j in range(i+1, len(top_feats)):
            f1, f2 = top_feats[i], top_feats[j]
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
            df[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1e-5)

    return df

train_eng = engineer_features(train)
test_eng  = engineer_features(test)

print(f"Features before engineering : {len(FEATURES)}")
print(f"Features after engineering  : {train_eng.shape[1] - 2}")  # minus Class and any ID

# ── STEP 3: PREPROCESSING ────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Preprocessing")
print("="*60)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# Separate features and target
drop_cols = ['Class']
X = train_eng.drop(columns=drop_cols)
y = train_eng['Class']

# Test features (drop ID for prediction, save ID for submission)
test_ids = test_eng['ID'].values
X_test   = test_eng.drop(columns=['ID'])

# Align columns (train and test must have same feature cols)
X_test = X_test[X.columns]

print(f"X shape      : {X.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y shape      : {y.shape}")

# Train-Validation split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain set : {X_train.shape[0]} rows")
print(f"Val set   : {X_val.shape[0]} rows")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# class imbalance ratio for scale_pos_weight
class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight : {class_ratio:.3f}")

# ── STEP 4: BASELINE MODEL ───────────────────────────────────
print("\n" + "="*60)
print("STEP 4: Baseline — Logistic Regression")
print("="*60)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr.fit(X_train_sc, y_train)
lr_preds = lr.predict(X_val_sc)
baseline_f1 = f1_score(y_val, lr_preds)
print(f"Baseline F1 Score : {baseline_f1:.4f}")

# ── STEP 5: XGBoost ──────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: XGBoost")
print("="*60)

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
xgb.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100)

xgb_preds = xgb.predict(X_val)
xgb_f1 = f1_score(y_val, xgb_preds)
print(f"\nXGBoost F1 Score : {xgb_f1:.4f}")

# ── STEP 6: LightGBM ─────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: LightGBM")
print("="*60)

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         callbacks=[])

lgbm_preds = lgbm.predict(X_val)
lgbm_f1 = f1_score(y_val, lgbm_preds)
print(f"LightGBM F1 Score : {lgbm_f1:.4f}")

# ── STEP 7: Random Forest ────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: Random Forest")
print("="*60)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_val)
rf_f1 = f1_score(y_val, rf_preds)
print(f"Random Forest F1 Score : {rf_f1:.4f}")

# ── STEP 8: OPTUNA HYPERPARAMETER TUNING ─────────────────────
print("\n" + "="*60)
print("STEP 8: Optuna Tuning (XGBoost)")
print("="*60)

if ENABLE_OPTUNA:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

def xgb_objective(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 300, 1000),
        'max_depth'       : trial.suggest_int('max_depth', 3, 10),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
        'scale_pos_weight': class_ratio,
        'use_label_encoder': False,
        'eval_metric'     : 'logloss',
        'random_state'    : RANDOM_STATE,
        'n_jobs'          : -1
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)

if ENABLE_OPTUNA:
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS_XGB, show_progress_bar=True)

    print(f"\nBest XGBoost F1 (Optuna) : {study_xgb.best_value:.4f}")
    print(f"Best params : {study_xgb.best_params}")

    # Train final XGBoost with best params
    best_xgb = XGBClassifier(**study_xgb.best_params,
                             scale_pos_weight=class_ratio,
                             use_label_encoder=False,
                             eval_metric='logloss',
                             random_state=RANDOM_STATE, n_jobs=-1)
    best_xgb.fit(X_train, y_train)
else:
    best_xgb = xgb

# ── STEP 9: LGBM OPTUNA TUNING ───────────────────────────────
print("\n" + "="*60)
print("STEP 9: Optuna Tuning (LightGBM)")
print("="*60)

def lgbm_objective(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 300, 1000),
        'max_depth'       : trial.suggest_int('max_depth', 3, 10),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
        'is_unbalance'    : True,
        'random_state'    : RANDOM_STATE,
        'n_jobs'          : -1,
        'verbose'         : -1
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds)

if ENABLE_OPTUNA:
    study_lgbm = optuna.create_study(direction='maximize')
    study_lgbm.optimize(lgbm_objective, n_trials=N_TRIALS_LGBM, show_progress_bar=True)

    print(f"\nBest LightGBM F1 (Optuna) : {study_lgbm.best_value:.4f}")

    best_lgbm = LGBMClassifier(**study_lgbm.best_params,
                                is_unbalance=True,
                                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    best_lgbm.fit(X_train, y_train)
else:
    best_lgbm = lgbm

# ── STEP 10: STACKING ENSEMBLE ───────────────────────────────
print("\n" + "="*60)
print("STEP 10: Stacking Ensemble")
print("="*60)

from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
        ('rf', rf)
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    cv=5,
    passthrough=False,
    n_jobs=-1
)
stacking.fit(X_train, y_train)
stack_preds = stacking.predict(X_val)
stack_f1 = f1_score(y_val, stack_preds)
print(f"Stacking Ensemble F1 Score : {stack_f1:.4f}")

# ── STEP 11: THRESHOLD TUNING ────────────────────────────────
print("\n" + "="*60)
print("STEP 11: Threshold Tuning")
print("="*60)

# Use best model for probability prediction
val_probs = stacking.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_f1 = 0.0

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
for thresh in thresholds:
    preds_thresh = (val_probs >= thresh).astype(int)
    f1 = f1_score(y_val, preds_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"Optimal Threshold : {best_threshold:.2f}")
print(f"Best F1 @ Threshold : {best_f1:.4f}")

# ── STEP 12: PSEUDO LABELING ─────────────────────────────────
print("\n" + "="*60)
print("STEP 12: Pseudo Labeling")
print("="*60)

from sklearn.calibration import CalibratedClassifierCV

if ENABLE_PSEUDO_LABELING:
    calibrated_xgb = CalibratedClassifierCV(best_xgb, cv=5, method='sigmoid')
    calibrated_xgb.fit(X_train, y_train)

    test_probs = calibrated_xgb.predict_proba(X_test)[:, 1]

    high_conf_mask = (test_probs > 0.90) | (test_probs < 0.10)
    X_pseudo = X_test[high_conf_mask]
    y_pseudo = (test_probs[high_conf_mask] >= 0.5).astype(int)

    print(f"High-confidence test samples : {high_conf_mask.sum()} / {len(test_probs)}")

    if high_conf_mask.sum() > 100:
        X_expanded = pd.concat([X_train, X_pseudo], ignore_index=True)
        y_expanded = pd.concat([y_train, pd.Series(y_pseudo)], ignore_index=True)

        stacking_pl = StackingClassifier(
            estimators=[
                ('xgb', best_xgb),
                ('lgbm', best_lgbm),
                ('rf', rf)
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=5, n_jobs=-1
        )
        stacking_pl.fit(X_expanded, y_expanded)
        pl_val_preds = stacking_pl.predict(X_val)
        pl_f1 = f1_score(y_val, pl_val_preds)
        print(f"F1 after Pseudo Labeling : {pl_f1:.4f}")

        final_model = stacking_pl if pl_f1 > stack_f1 else stacking
        print(f"Using {'Pseudo Labeled' if pl_f1 > stack_f1 else 'Original Stacking'} model")
    else:
        final_model = stacking
        print("Not enough high-confidence samples — using original stacking model")
else:
    final_model = stacking

# ── STEP 13: SHAP ANALYSIS ───────────────────────────────────
print("\n" + "="*60)
print("STEP 13: SHAP Feature Importance")
print("="*60)

if ENABLE_SHAP:
    try:
        import shap
        import matplotlib.pyplot as plt

        # Use best XGBoost for SHAP (fastest)
        explainer = shap.TreeExplainer(best_xgb)
        shap_values = explainer.shap_values(X_val[:500])  # sample 500 rows for speed

        # Save SHAP plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_val[:500], plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("SHAP plot saved as: shap_importance.png")
    except Exception as e:
        print(f"SHAP skipped: {e}")
else:
    print("SHAP skipped: ENABLE_SHAP=False")

# ── STEP 14: FINAL PREDICTIONS ───────────────────────────────
print("\n" + "="*60)
print("STEP 14: Generating Final Predictions")
print("="*60)

# Get probabilities from final model
final_probs = final_model.predict_proba(X_test)[:, 1]

# Apply optimal threshold
final_preds = (final_probs >= best_threshold).astype(int)

print(f"Predictions distribution:")
print(f"  Class 0 (Normal) : {(final_preds == 0).sum()} ({(final_preds == 0).mean()*100:.1f}%)")
print(f"  Class 1 (Faulty) : {(final_preds == 1).sum()} ({(final_preds == 1).mean()*100:.1f}%)")

# ── STEP 15: SAVE FINAL.csv ──────────────────────────────────
print("\n" + "="*60)
print("STEP 15: Saving FINAL.csv")
print("="*60)

submission = pd.DataFrame({
    'ID'   : test_ids,
    'Class': final_preds
})

submission.to_csv('FINAL.csv', index=False)

# Verification
print(f"\n✅ FINAL.csv saved successfully!")
print(f"   Rows    : {len(submission)} (expected: 10944)")
print(f"   Columns : {submission.columns.tolist()}")
print(f"   Sample  :")
print(submission.head(10).to_string(index=False))

assert len(submission) == len(test), "ERROR: Row count mismatch!"
assert set(submission['Class'].unique()).issubset({0, 1}), "ERROR: Invalid class values!"
assert list(submission['ID']) == list(test_ids), "ERROR: ID order mismatch!"
print("\n✅ All checks passed! FINAL.csv is ready to submit.")

# ── STEP 16: SAVE MODEL ──────────────────────────────────────
import joblib
joblib.dump(final_model, 'final_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Model saved as: final_model.pkl")

# ── FINAL SCORE SUMMARY ──────────────────────────────────────
print("\n" + "="*60)
print("FINAL SCORE SUMMARY (Validation Set)")
print("="*60)
print(f"  Logistic Regression (Baseline) : {baseline_f1:.4f}")
print(f"  XGBoost                        : {xgb_f1:.4f}")
print(f"  LightGBM                       : {lgbm_f1:.4f}")
print(f"  Random Forest                  : {rf_f1:.4f}")
print(f"  Stacking Ensemble              : {stack_f1:.4f}")
print(f"  Best Threshold Applied         : {best_threshold:.2f}")
print(f"  Final F1 (threshold tuned)     : {best_f1:.4f}")
print("="*60)
print("\nFiles to push on GitHub:")
print("  ✅ fault_detection_solution.py")
print("  ✅ FINAL.csv")
print("  ✅ final_model.pkl")
print("  ✅ shap_importance.png")
print("  ✅ requirements.txt")
print("  ✅ README.md")