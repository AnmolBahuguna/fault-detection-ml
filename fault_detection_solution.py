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

FAST_SMOKE_TEST = True

SEEDS = [42] if FAST_SMOKE_TEST else [42, 43, 44, 45]
N_SPLITS = 3 if FAST_SMOKE_TEST else 5

ENABLE_OPTUNA = False
ENABLE_PSEUDO_LABELING_TOP1 = False
ENABLE_ADVERSARIAL_VALIDATION = False if FAST_SMOKE_TEST else True
ENABLE_CALIBRATION = True
ENABLE_SHAP = False
ENABLE_EDA = False

XGB_WEIGHT = 0.35
LGBM_WEIGHT = 0.30
RF_WEIGHT = 0.15
TABNET_WEIGHT = 0.10
META_WEIGHT = 0.10

PSEUDO_POS_TH = 0.95
PSEUDO_NEG_TH = 0.05

FEATURES = [f'F{str(i).zfill(2)}' for i in range(1, 48)]

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
import joblib

from typing import Optional


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feats = FEATURES
    df['mean_all'] = df[feats].mean(axis=1)
    df['std_all'] = df[feats].std(axis=1)
    df['max_all'] = df[feats].max(axis=1)
    df['min_all'] = df[feats].min(axis=1)
    df['range_all'] = df['max_all'] - df['min_all']
    df['skew_all'] = df[feats].skew(axis=1)
    df['kurt_all'] = df[feats].kurtosis(axis=1)

    groupA = [f for f in feats if int(f[1:]) <= 16]
    groupB = [f for f in feats if 17 <= int(f[1:]) <= 32]
    groupC = [f for f in feats if int(f[1:]) >= 33]

    for name, grp in [('A', groupA), ('B', groupB), ('C', groupC)]:
        df[f'mean_{name}'] = df[grp].mean(axis=1)
        df[f'std_{name}'] = df[grp].std(axis=1)
        df[f'max_{name}'] = df[grp].max(axis=1)
        df[f'min_{name}'] = df[grp].min(axis=1)
        df[f'range_{name}'] = df[f'max_{name}'] - df[f'min_{name}']
        df[f'skew_{name}'] = df[grp].skew(axis=1)
        df[f'kurt_{name}'] = df[grp].kurtosis(axis=1)

    df['ratio_A_B'] = df['mean_A'] / (df['mean_B'] + 1e-5)
    df['ratio_B_C'] = df['mean_B'] / (df['mean_C'] + 1e-5)
    df['ratio_A_C'] = df['mean_A'] / (df['mean_C'] + 1e-5)
    df['std_ratio_A_B'] = df['std_A'] / (df['std_B'] + 1e-5)
    df['std_ratio_B_C'] = df['std_B'] / (df['std_C'] + 1e-5)
    df['std_ratio_A_C'] = df['std_A'] / (df['std_C'] + 1e-5)

    top_feats = ['F01', 'F10', 'F08', 'F09', 'F06', 'F07']
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            f1, f2 = top_feats[i], top_feats[j]
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
            df[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + 1e-5)

    return df


def prepare_matrices(train: pd.DataFrame, test: pd.DataFrame):
    if 'ID' not in test.columns:
        raise KeyError("TEST.csv must contain an 'ID' column for submission.")

    train_eng = engineer_features(train).replace([np.inf, -np.inf], np.nan)
    test_eng = engineer_features(test).replace([np.inf, -np.inf], np.nan)

    drop_cols = ['Class']
    if 'ID' in train_eng.columns:
        drop_cols.append('ID')
    X = train_eng.drop(columns=drop_cols)
    y = train_eng['Class'].astype(int)

    test_ids = test_eng['ID'].values
    X_test = test_eng.drop(columns=['ID'])
    X_test = X_test[X.columns]

    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X_test = X_test.fillna(med)

    return X, y, X_test, test_ids


def adversarial_validation(X: pd.DataFrame, X_test: pd.DataFrame, random_state: int):
    X_all = pd.concat([X, X_test], axis=0, ignore_index=True)
    y_all = np.concatenate([np.zeros(len(X), dtype=int), np.ones(len(X_test), dtype=int)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof = np.zeros(len(X_all))

    model = RandomForestClassifier(
        n_estimators=80 if FAST_SMOKE_TEST else 300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    for tr, va in skf.split(X_all, y_all):
        m = clone(model)
        m.fit(X_all.iloc[tr], y_all[tr])
        oof[va] = m.predict_proba(X_all.iloc[va])[:, 1]

    auc = roc_auc_score(y_all, oof)
    m.fit(X_all, y_all)
    importances = pd.Series(m.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    return float(auc), importances


def get_xgb(class_ratio: float, seed: int):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_alpha=0.0,
            reg_lambda=1.0,
            scale_pos_weight=class_ratio,
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1
        )
    except Exception:
        return None


def get_lgbm(seed: int):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            reg_alpha=0.0,
            reg_lambda=0.0,
            is_unbalance=True,
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        )
    except Exception:
        return None


def get_tabnet(seed: int):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier(seed=seed, verbose=0)
    except Exception:
        return None


def fit_predict_model(model, X_tr, y_tr, X_va, X_te):
    if model is None:
        return None, None
    model.fit(X_tr, y_tr)
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    return p_va, p_te


def oof_multi_seed(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, seeds, n_splits: int):
    class_ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    model_keys = ['xgb', 'lgbm', 'rf', 'tabnet']
    oof = {k: np.zeros(len(X)) for k in model_keys}
    test = {k: np.zeros(len(X_test)) for k in model_keys}
    counts = {k: 0 for k in model_keys}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            rf = RandomForestClassifier(
                n_estimators=150 if FAST_SMOKE_TEST else 600,
                max_depth=None,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1
            )
            xgb = get_xgb(class_ratio, seed)
            lgbm = get_lgbm(seed)
            tabnet = get_tabnet(seed)

            for key, model in [('rf', rf), ('xgb', xgb), ('lgbm', lgbm), ('tabnet', tabnet)]:
                p_va, p_te = fit_predict_model(model, X_tr, y_tr, X_va, X_test)
                if p_va is None:
                    continue
                oof[key][va_idx] += p_va
                test[key] += p_te
                counts[key] += 1

    for k in model_keys:
        if counts[k] > 0:
            oof[k] /= counts[k]
            test[k] /= counts[k]
        else:
            oof[k] = None
            test[k] = None

    return oof, test


def stack_meta(oof_dict, y: pd.Series, test_dict):
    keys = [k for k, v in oof_dict.items() if v is not None]
    X_meta = np.vstack([oof_dict[k] for k in keys]).T
    X_meta_test = np.vstack([test_dict[k] for k in keys]).T

    meta = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE)
    meta.fit(X_meta, y.values)
    oof_meta = meta.predict_proba(X_meta)[:, 1]
    test_meta = meta.predict_proba(X_meta_test)[:, 1]
    return meta, keys, oof_meta, test_meta


def weighted_ensemble(probs: dict, meta_prob: Optional[np.ndarray]):
    parts = []
    weights = []

    if probs.get('xgb') is not None:
        parts.append(probs['xgb'])
        weights.append(XGB_WEIGHT)
    if probs.get('lgbm') is not None:
        parts.append(probs['lgbm'])
        weights.append(LGBM_WEIGHT)
    if probs.get('rf') is not None:
        parts.append(probs['rf'])
        weights.append(RF_WEIGHT)
    if probs.get('tabnet') is not None:
        parts.append(probs['tabnet'])
        weights.append(TABNET_WEIGHT)
    if meta_prob is not None:
        parts.append(meta_prob)
        weights.append(META_WEIGHT)

    w = np.array(weights, dtype=float)
    w = w / w.sum()
    out = np.zeros_like(parts[0], dtype=float)
    for p, ww in zip(parts, w):
        out += ww * p
    return out


def tune_threshold(y_true: np.ndarray, prob: np.ndarray):
    best_t = 0.5
    best = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best:
            best = f1
            best_t = float(t)
    return best_t, float(best)


def main():
    print("=" * 60)
    print("TOP1 PIPELINE: Loading Data")
    print("=" * 60)

    TRAIN_PATH = os.getenv('TRAIN_PATH', 'TRAIN.csv')
    TEST_PATH = os.getenv('TEST_PATH', 'TEST.csv')
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")

    X, y, X_test, test_ids = prepare_matrices(train, test)

    print(f"X shape      : {X.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y distribution:\n{y.value_counts()}")

    drop_features = []
    if ENABLE_ADVERSARIAL_VALIDATION:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Adversarial Validation")
        print("=" * 60)
        try:
            auc, importances = adversarial_validation(X, X_test, RANDOM_STATE)
            print(f"Adversarial AUC (train vs test) : {auc:.4f}")
            if auc > 0.7:
                drop_features = importances.head(10).index.tolist()
                print(f"Dropping shift-sensitive features (top 10): {drop_features}")
            else:
                print("No strong shift detected; keeping all features")
        except Exception as e:
            print(f"Adversarial validation skipped: {e}")

    if drop_features:
        X = X.drop(columns=drop_features)
        X_test = X_test.drop(columns=drop_features)

    print("\n" + "=" * 60)
    print("TOP1 PIPELINE: Multi-seed OOF backbone")
    print("=" * 60)
    oof_base, test_base = oof_multi_seed(X, y, X_test, SEEDS, N_SPLITS)

    available = [k for k, v in oof_base.items() if v is not None]
    print(f"Available base models: {available}")

    meta_model, meta_keys, oof_meta, test_meta = stack_meta(oof_base, y, test_base)
    print(f"Meta-learner trained on: {meta_keys}")

    oof_ens = weighted_ensemble(oof_base, oof_meta)
    test_ens = weighted_ensemble(test_base, test_meta)

    if ENABLE_CALIBRATION:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Isotonic Calibration")
        print("=" * 60)
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(oof_ens, y.values)
            oof_ens_cal = iso.transform(oof_ens)
            test_ens_cal = iso.transform(test_ens)
        except Exception as e:
            print(f"Calibration skipped: {e}")
            oof_ens_cal = oof_ens
            test_ens_cal = test_ens
            iso = None
    else:
        oof_ens_cal = oof_ens
        test_ens_cal = test_ens
        iso = None

    best_threshold, best_f1 = tune_threshold(y.values, oof_ens_cal)
    print("\n" + "=" * 60)
    print("TOP1 PIPELINE: Threshold selection")
    print("=" * 60)
    print(f"Best threshold : {best_threshold:.2f}")
    print(f"Best OOF F1    : {best_f1:.4f}")

    if ENABLE_PSEUDO_LABELING_TOP1:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Pseudo Labeling (1 iteration)")
        print("=" * 60)
        pseudo_mask = (test_ens_cal >= PSEUDO_POS_TH) | (test_ens_cal <= PSEUDO_NEG_TH)
        if pseudo_mask.sum() > 0:
            X_pseudo = X_test.loc[pseudo_mask]
            y_pseudo = (test_ens_cal[pseudo_mask] >= 0.5).astype(int)
            X_aug = pd.concat([X, X_pseudo], axis=0, ignore_index=True)
            y_aug = pd.concat([y, pd.Series(y_pseudo)], axis=0, ignore_index=True)
            oof_base, test_base = oof_multi_seed(X_aug, y_aug, X_test, SEEDS, N_SPLITS)
            meta_model, meta_keys, oof_meta, test_meta = stack_meta(oof_base, y_aug, test_base)
            oof_ens = weighted_ensemble(oof_base, oof_meta)
            test_ens = weighted_ensemble(test_base, test_meta)
            if iso is not None:
                oof_ens_cal = iso.transform(oof_ens)
                test_ens_cal = iso.transform(test_ens)
            else:
                oof_ens_cal = oof_ens
                test_ens_cal = test_ens
            best_threshold, best_f1 = tune_threshold(y_aug.values, oof_ens_cal)
            print(f"Pseudo labeling used samples: {int(pseudo_mask.sum())}")
            print(f"New best threshold : {best_threshold:.2f}")
            print(f"New best OOF F1    : {best_f1:.4f}")
        else:
            print("No high-confidence pseudo samples found")

    final_preds = (test_ens_cal >= best_threshold).astype(int)
    submission = pd.DataFrame({'ID': test_ids, 'CLASS': final_preds})
    submission.to_csv('FINAL.csv', index=False)

    assert len(submission) == len(test), "ERROR: Row count mismatch!"
    assert set(submission['CLASS'].unique()).issubset({0, 1}), "ERROR: Invalid class values!"
    assert list(submission['ID']) == list(test_ids), "ERROR: ID order mismatch!"

    joblib.dump(
        {
            'meta_model': meta_model,
            'meta_keys': meta_keys,
            'drop_features': drop_features,
            'isotonic': iso,
            'threshold': best_threshold,
            'seeds': SEEDS,
            'n_splits': N_SPLITS,
            'weights': {
                'xgb': XGB_WEIGHT,
                'lgbm': LGBM_WEIGHT,
                'rf': RF_WEIGHT,
                'tabnet': TABNET_WEIGHT,
                'meta': META_WEIGHT
            }
        },
        'final_model.pkl'
    )

    print("\n" + "=" * 60)
    print("TOP1 PIPELINE: Outputs")
    print("=" * 60)
    print("FINAL.csv saved")
    print("final_model.pkl saved")
    print(submission.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
