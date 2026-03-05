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
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return int(str(v).strip())


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return float(str(v).strip())


def _env_int_list(name: str, default: list[int]) -> list[int]:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    parts = [p.strip() for p in str(v).split(",") if p.strip()]
    return [int(p) for p in parts]


FAST_SMOKE_TEST = _env_bool("FAST_SMOKE_TEST", False)

SEEDS = _env_int_list("SEEDS", [42] if FAST_SMOKE_TEST else [42, 43, 44, 45])
N_SPLITS = _env_int("N_SPLITS", 3 if FAST_SMOKE_TEST else 5)

ENABLE_OPTUNA = _env_bool("ENABLE_OPTUNA", False)
ENABLE_PSEUDO_LABELING_TOP1 = _env_bool("ENABLE_PSEUDO_LABELING_TOP1", False)
ENABLE_ADVERSARIAL_VALIDATION = _env_bool(
    "ENABLE_ADVERSARIAL_VALIDATION",
    False if FAST_SMOKE_TEST else True
)
ENABLE_CALIBRATION = _env_bool("ENABLE_CALIBRATION", True)
ENABLE_SHAP = _env_bool("ENABLE_SHAP", False)
ENABLE_SHAP_REFINEMENT = _env_bool("ENABLE_SHAP_REFINEMENT", False)
SHAP_TOP_N = _env_int("SHAP_TOP_N", 40)
ENABLE_WEIGHT_OPTIMIZATION = _env_bool("ENABLE_WEIGHT_OPTIMIZATION", True)
WEIGHT_SEARCH_ITERS = _env_int("WEIGHT_SEARCH_ITERS", 250)
ENABLE_CV_STABILITY_REPORT = _env_bool("ENABLE_CV_STABILITY_REPORT", True)
ENABLE_FEATURE_IMPORTANCE_FILTER = _env_bool("ENABLE_FEATURE_IMPORTANCE_FILTER", True)
FEATURE_IMPORTANCE_TOP_N = _env_int("FEATURE_IMPORTANCE_TOP_N", 60)
ENABLE_PLOTS = _env_bool("ENABLE_PLOTS", True)
ENABLE_EDA = False

XGB_WEIGHT = _env_float("XGB_WEIGHT", 0.28)
LGBM_WEIGHT = _env_float("LGBM_WEIGHT", 0.25)
CAT_WEIGHT = _env_float("CAT_WEIGHT", 0.25)
RF_WEIGHT = _env_float("RF_WEIGHT", 0.10)
TABNET_WEIGHT = _env_float("TABNET_WEIGHT", 0.05)
META_WEIGHT = _env_float("META_WEIGHT", 0.07)

PSEUDO_POS_TH = _env_float("PSEUDO_POS_TH", 0.95)
PSEUDO_NEG_TH = _env_float("PSEUDO_NEG_TH", 0.05)

FEATURES = [f'F{str(i).zfill(2)}' for i in range(1, 48)]

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
import joblib

from typing import Optional


def engineer_features(df: pd.DataFrame, reference_feats=None):
    df = df.copy()
    feats = FEATURES if reference_feats is None else reference_feats
    df['mean_all'] = df[feats].mean(axis=1)
    df['std_all'] = df[feats].std(axis=1)
    df['max_all'] = df[feats].max(axis=1)
    df['min_all'] = df[feats].min(axis=1)
    df['range_all'] = df['max_all'] - df['min_all']
    df['skew_all'] = df[feats].skew(axis=1)
    df['kurt_all'] = df[feats].kurtosis(axis=1)

    groupA = feats[:16]    # F01–F16
    groupB = feats[16:32]  # F17–F32
    groupC = feats[32:]    # F33–F47

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

    vals = df[feats].to_numpy(dtype=float)
    abs_vals = np.abs(vals)
    df['l1_energy'] = abs_vals.sum(axis=1)
    df['l2_energy'] = np.sqrt((vals ** 2).sum(axis=1))
    df['frac_pos'] = (vals > 0).mean(axis=1)
    df['frac_neg'] = (vals < 0).mean(axis=1)
    q10 = np.quantile(vals, 0.10, axis=1)
    q25 = np.quantile(vals, 0.25, axis=1)
    q75 = np.quantile(vals, 0.75, axis=1)
    q90 = np.quantile(vals, 0.90, axis=1)
    df['q10_all'] = q10
    df['q25_all'] = q25
    df['q75_all'] = q75
    df['q90_all'] = q90
    df['iqr_all'] = q75 - q25
    row_mean = vals.mean(axis=1)
    row_std = vals.std(axis=1) + 1e-8
    z = (vals - row_mean[:, None]) / row_std[:, None]
    df['n_outlier_gt3'] = (np.abs(z) > 3.0).sum(axis=1)
    df['n_outlier_gt2'] = (np.abs(z) > 2.0).sum(axis=1)

    # Use fixed top features for consistency
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
    if 'Class' not in train.columns:
        raise KeyError("TRAIN.csv must contain a 'Class' column.")

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


def filter_features_by_importance(X, y, X_test, top_n, seed):
    print(f"\nFeature importance filter: keeping top {top_n} of {X.shape[1]} features")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight='balanced',
        random_state=seed, n_jobs=-1
    )
    rf.fit(X, y)
    imp      = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_cols = imp.head(top_n).index.tolist()
    print(f"  Dropped {X.shape[1] - len(top_cols)} low-importance features")
    return X[top_cols], X_test[top_cols], imp


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


def get_catboost(class_ratio: float, seed: int):
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            colsample_bylevel=0.8,
            min_data_in_leaf=20,
            scale_pos_weight=class_ratio,
            eval_metric='F1',
            early_stopping_rounds=50,
            random_seed=seed,
            verbose=0,
            thread_count=-1,
            use_best_model=True,
        )
    except Exception:
        return None


def get_tabnet(seed: int):
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        return TabNetClassifier(seed=seed, verbose=0)
    except Exception:
        return None


def _fit_xgb_optuna(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    class_ratio: float,
    seed: int,
    fold: int,
    n_trials: int,
):
    try:
        import optuna
        from xgboost import XGBClassifier
    except Exception:
        return get_xgb(class_ratio, seed)

    def objective(trial: 'optuna.Trial'):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=class_ratio,
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        _, f1 = tune_threshold(y_va.values, p)
        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    best = study.best_params
    model = XGBClassifier(
        **best,
        scale_pos_weight=class_ratio,
        eval_metric='logloss',
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    Path('optuna_logs').mkdir(parents=True, exist_ok=True)
    pd.DataFrame([t.params | {'value': t.value} for t in study.trials]).to_csv(
        os.path.join('optuna_logs', f'optuna_xgb_seed{seed}_fold{fold}.csv'),
        index=False,
    )
    joblib.dump(study, os.path.join('optuna_logs', f'optuna_xgb_seed{seed}_fold{fold}.pkl'))
    return model


def _fit_lgbm_optuna(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    seed: int,
    fold: int,
    n_trials: int,
):
    try:
        import optuna
        from lightgbm import LGBMClassifier
    except Exception:
        return get_lgbm(seed)

    def objective(trial: 'optuna.Trial'):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        }
        model = LGBMClassifier(
            **params,
            is_unbalance=True,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        _, f1 = tune_threshold(y_va.values, p)
        return f1

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)

    best = study.best_params
    model = LGBMClassifier(
        **best,
        is_unbalance=True,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)

    Path('optuna_logs').mkdir(parents=True, exist_ok=True)
    pd.DataFrame([t.params | {'value': t.value} for t in study.trials]).to_csv(
        os.path.join('optuna_logs', f'optuna_lgbm_seed{seed}_fold{fold}.csv'),
        index=False,
    )
    joblib.dump(study, os.path.join('optuna_logs', f'optuna_lgbm_seed{seed}_fold{fold}.pkl'))
    return model


def fit_predict_model(model, X_tr, y_tr, X_va, y_va, X_te, already_fitted: bool = False):
    if model is None:
        return None, None
    if not already_fitted:
        model_type = type(model).__name__

        if model_type == 'XGBClassifier':
            try:
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except Exception:
                model.set_params(early_stopping_rounds=None)
                model.fit(X_tr, y_tr)

        elif model_type == 'LGBMClassifier':
            try:
                from lightgbm import early_stopping, log_evaluation
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)]
                )
            except Exception:
                model.fit(X_tr, y_tr)

        elif model_type == 'CatBoostClassifier':
            try:
                from catboost import Pool
                model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), verbose=False)
            except Exception:
                model.set_params(early_stopping_rounds=None)
                model.fit(X_tr, y_tr)

        else:
            model.fit(X_tr, y_tr)

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    return p_va, p_te


def oof_multi_seed(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, seeds, n_splits: int):
    class_ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    model_keys = ['xgb', 'lgbm', 'cat', 'rf', 'tabnet']
    oof = {k: np.zeros(len(X)) for k in model_keys}
    oof_counts = {k: np.zeros(len(X), dtype=int) for k in model_keys}
    test = {k: np.zeros(len(X_test)) for k in model_keys}
    counts = {k: 0 for k in model_keys}
    fold_scores = {k: [] for k in model_keys}

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
            if ENABLE_OPTUNA and not FAST_SMOKE_TEST:
                xgb = _fit_xgb_optuna(X_tr, y_tr, X_va, y_va, class_ratio, seed, fold=fold, n_trials=40)
                lgbm = _fit_lgbm_optuna(X_tr, y_tr, X_va, y_va, seed, fold=fold, n_trials=40)
            else:
                xgb = get_xgb(class_ratio, seed)
                lgbm = get_lgbm(seed)
            cat = get_catboost(class_ratio, seed)
            tabnet = get_tabnet(seed)

            for key, model in [('rf', rf), ('xgb', xgb), ('lgbm', lgbm), ('cat', cat), ('tabnet', tabnet)]:
                already_fitted = bool(ENABLE_OPTUNA and (key in {'xgb', 'lgbm'}) and (not FAST_SMOKE_TEST))
                p_va, p_te = fit_predict_model(model, X_tr, y_tr, X_va, y_va, X_test, already_fitted=already_fitted)
                if p_va is None:
                    continue
                oof[key][va_idx] += p_va
                oof_counts[key][va_idx] += 1
                test[key] += p_te
                counts[key] += 1
                t, _ = tune_threshold(y_va.values, p_va)
                f1 = f1_score(y_va.values, (p_va >= t).astype(int))
                fold_scores[key].append(float(f1))

    for k in model_keys:
        if counts[k] > 0:
            mask = oof_counts[k] > 0
            oof[k][mask] /= oof_counts[k][mask]
            test[k] /= counts[k]
        else:
            oof[k] = None
            test[k] = None

    stability = {}
    for k in model_keys:
        scores = fold_scores.get(k, [])
        if len(scores) == 0:
            continue
        stability[k] = {
            'n_folds_total': int(len(scores)),
            'f1_mean': float(np.mean(scores)),
            'f1_std': float(np.std(scores)),
            'f1_min': float(np.min(scores)),
            'f1_max': float(np.max(scores))
        }

    return oof, test, stability


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
    if probs.get('cat') is not None:
        parts.append(probs['cat'])
        weights.append(CAT_WEIGHT)
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


def optimize_blend_weights(oof_dict: dict, meta_oof: Optional[np.ndarray], y: pd.Series, n_iters: int):
    keys = []
    parts = []
    if oof_dict.get('xgb') is not None:
        keys.append('xgb')
        parts.append(oof_dict['xgb'])
    if oof_dict.get('lgbm') is not None:
        keys.append('lgbm')
        parts.append(oof_dict['lgbm'])
    if oof_dict.get('cat') is not None:
        keys.append('cat')
        parts.append(oof_dict['cat'])
    if oof_dict.get('rf') is not None:
        keys.append('rf')
        parts.append(oof_dict['rf'])
    if oof_dict.get('tabnet') is not None:
        keys.append('tabnet')
        parts.append(oof_dict['tabnet'])
    if meta_oof is not None:
        keys.append('meta')
        parts.append(meta_oof)

    if len(parts) <= 1:
        return None

    P = np.vstack(parts).T
    rng = np.random.RandomState(RANDOM_STATE)
    best_w = None
    best_f1 = -1.0
    best_t = 0.5

    for _ in range(max(int(n_iters), 1)):
        w = rng.dirichlet(np.ones(P.shape[1]))
        prob = P @ w
        t, f1 = tune_threshold(y.values, prob)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
            best_w = w

    return {
        'keys': keys,
        'weights': [float(x) for x in best_w],
        'best_threshold': best_t,
        'best_oof_f1': best_f1
    }


def _save_shap_importance(model, X: pd.DataFrame, out_png: str, out_csv: str):
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return False

    try:
        X_sample = X.sample(n=min(2000, len(X)), random_state=RANDOM_STATE)
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        vals = np.abs(shap_values.values)
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        imp = pd.Series(vals.mean(axis=0), index=X_sample.columns).sort_values(ascending=False)
        imp.to_csv(out_csv)
        plt.figure(figsize=(10, 6))
        imp.head(30).iloc[::-1].plot(kind='barh')
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return True
    except Exception:
        return False


def tune_threshold(y_true: np.ndarray, prob: np.ndarray):
    best_t = 0.5
    best = -1.0
    for t in np.linspace(0.01, 0.99, 199):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best:
            best = f1
            best_t = float(t)
    return best_t, float(best)


def generate_plots(y_true, oof_prob, threshold, feature_imp_series=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, classification_report, confusion_matrix
    except Exception:
        print("matplotlib not available, skipping plots")
        return

    oof_pred = (oof_prob >= threshold).astype(int)

    # Confusion Matrix
    cm  = confusion_matrix(y_true, oof_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Fault (1)']).plot(ax=ax, colorbar=False)
    ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("confusion_matrix.png saved")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, oof_prob)
    auc_val     = roc_auc_score(y_true, oof_prob)
    fig, ax     = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc_val:.4f}', color='steelblue', lw=2)
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (OOF)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.close()
    print("roc_curve.png saved")

    # Feature Importance
    if feature_imp_series is not None:
        fig, ax = plt.subplots(figsize=(10, 7))
        feature_imp_series.head(30).iloc[::-1].plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Top 30 Feature Importances (RF-based)')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        plt.close()
        print("feature_importance.png saved")

    # Classification Report
    print("\nClassification Report (OOF):")
    print(classification_report(y_true, oof_pred, target_names=['Normal', 'Fault']))


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except Exception:
        pass

    t0 = time.time()
    print("=" * 60)
    print("TOP1 PIPELINE: Loading Data")
    print("=" * 60)

    TRAIN_PATH = os.getenv('TRAIN_PATH', 'TRAIN.csv')
    TEST_PATH = os.getenv('TEST_PATH', 'TEST.csv')
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("\n" + "=" * 60)
    print("TOP1 PIPELINE: Run configuration")
    print("=" * 60)
    print(f"TRAIN_PATH: {TRAIN_PATH}")
    print(f"TEST_PATH : {TEST_PATH}")
    print(f"FAST_SMOKE_TEST: {FAST_SMOKE_TEST}")
    print(f"SEEDS: {SEEDS}")
    print(f"N_SPLITS: {N_SPLITS}")
    print(f"ENABLE_ADVERSARIAL_VALIDATION: {ENABLE_ADVERSARIAL_VALIDATION}")
    print(f"ENABLE_CALIBRATION: {ENABLE_CALIBRATION}")
    print(f"ENABLE_PSEUDO_LABELING_TOP1: {ENABLE_PSEUDO_LABELING_TOP1}")
    print(f"ENABLE_WEIGHT_OPTIMIZATION: {ENABLE_WEIGHT_OPTIMIZATION}")
    print(f"ENABLE_CV_STABILITY_REPORT: {ENABLE_CV_STABILITY_REPORT}")
    print(f"ENABLE_OPTUNA: {ENABLE_OPTUNA}")
    print(f"ENABLE_SHAP: {ENABLE_SHAP}")
    print(f"ENABLE_SHAP_REFINEMENT: {ENABLE_SHAP_REFINEMENT}")

    print(f"Train shape : {train.shape}")
    print(f"Test shape  : {test.shape}")

    X, y, X_test, test_ids = prepare_matrices(train, test)

    print(f"X shape      : {X.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y distribution:\n{y.value_counts()}")

    drop_features = []
    adv_auc = None
    if ENABLE_ADVERSARIAL_VALIDATION:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Adversarial Validation")
        print("=" * 60)
        try:
            auc, importances = adversarial_validation(X, X_test, RANDOM_STATE)
            adv_auc = float(auc)
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

    feature_imp_series = None
    if ENABLE_FEATURE_IMPORTANCE_FILTER:
        print("\n" + "="*60)
        print("TOP1 PIPELINE: Feature Importance Filtering")
        print("="*60)
        X, X_test, feature_imp_series = filter_features_by_importance(
            X, y, X_test, FEATURE_IMPORTANCE_TOP_N, RANDOM_STATE
        )

    print("\n" + "=" * 60)
    print("TOP1 PIPELINE: Multi-seed OOF backbone")
    print("=" * 60)
    oof_base, test_base, stability = oof_multi_seed(X, y, X_test, SEEDS, N_SPLITS)

    available = [k for k, v in oof_base.items() if v is not None]
    print(f"Available base models: {available}")

    meta_model, meta_keys, oof_meta, test_meta = stack_meta(oof_base, y, test_base)
    print(f"Meta-learner trained on: {meta_keys}")

    if ENABLE_CV_STABILITY_REPORT and stability:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Cross-fold stability report")
        print("=" * 60)
        for k, st in stability.items():
            print(f"{k}: folds={st['n_folds_total']} f1_mean={st['f1_mean']:.4f} f1_std={st['f1_std']:.4f} min={st['f1_min']:.4f} max={st['f1_max']:.4f}")

    oof_ens = weighted_ensemble(oof_base, oof_meta)
    test_ens = weighted_ensemble(test_base, test_meta)

    blend_opt = None
    if ENABLE_WEIGHT_OPTIMIZATION and not FAST_SMOKE_TEST:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Weight optimization (OOF blending)")
        print("=" * 60)
        try:
            blend_opt = optimize_blend_weights(oof_base, oof_meta, y, n_iters=WEIGHT_SEARCH_ITERS)
            if blend_opt is not None:
                print(f"Optimized keys: {blend_opt['keys']}")
                print(f"Optimized weights: {[round(w, 4) for w in blend_opt['weights']]}")
                print(f"Optimized OOF best F1: {blend_opt['best_oof_f1']:.4f} @ t={blend_opt['best_threshold']:.2f}")
                P_oof = np.vstack([
                    (oof_base[k] if k != 'meta' else oof_meta) for k in blend_opt['keys']
                ]).T
                P_te = np.vstack([
                    (test_base[k] if k != 'meta' else test_meta) for k in blend_opt['keys']
                ]).T
                w = np.array(blend_opt['weights'], dtype=float)
                oof_ens = P_oof @ w
                test_ens = P_te @ w
        except Exception as e:
            print(f"Weight optimization skipped: {e}")

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

    if ENABLE_PLOTS:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Generating Plots")
        print("=" * 60)
        generate_plots(y.values, oof_ens_cal, best_threshold, feature_imp_series)

    if ENABLE_PSEUDO_LABELING_TOP1:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: Pseudo Labeling (1 iteration)")
        print("=" * 60)
        pseudo_mask = (test_ens_cal >= PSEUDO_POS_TH) | (test_ens_cal <= PSEUDO_NEG_TH)
        if pseudo_mask.sum() > 0:
            X_pseudo = X_test.loc[pseudo_mask]
            y_pseudo = np.where(test_ens_cal[pseudo_mask] >= PSEUDO_POS_TH, 1,
                         np.where(test_ens_cal[pseudo_mask] <= PSEUDO_NEG_TH, 0, np.nan)).astype(int)
            X_aug = pd.concat([X, X_pseudo], axis=0, ignore_index=True)
            y_aug = pd.concat([y, pd.Series(y_pseudo)], axis=0, ignore_index=True)
            oof_base, test_base, stability = oof_multi_seed(X_aug, y_aug, X_test, SEEDS, N_SPLITS)
            meta_model, meta_keys, oof_meta, test_meta = stack_meta(oof_base, y_aug, test_base)
            oof_ens = weighted_ensemble(oof_base, oof_meta)
            test_ens = weighted_ensemble(test_base, test_meta)
            if ENABLE_CALIBRATION:
                try:
                    iso = IsotonicRegression(out_of_bounds='clip')
                    iso.fit(oof_ens, y_aug.values)
                    oof_ens_cal = iso.transform(oof_ens)
                    test_ens_cal = iso.transform(test_ens)
                except Exception:
                    oof_ens_cal = oof_ens
                    test_ens_cal = test_ens
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
            'stability': stability,
            'blend_optimized': blend_opt,
            'weights': {
                'xgb': XGB_WEIGHT,
                'lgbm': LGBM_WEIGHT,
                'cat': CAT_WEIGHT,
                'rf': RF_WEIGHT,
                'tabnet': TABNET_WEIGHT,
                'meta': META_WEIGHT
            }
        },
        'final_model.pkl'
    )

    if ENABLE_SHAP:
        print("\n" + "=" * 60)
        print("TOP1 PIPELINE: SHAP importance")
        print("=" * 60)
        shap_model = None
        try:
            shap_model = get_lgbm(RANDOM_STATE)
            if shap_model is None:
                shap_model = get_xgb(1.0, RANDOM_STATE)
            if shap_model is None:
                shap_model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
            shap_model.fit(X, y)
            ok = _save_shap_importance(shap_model, X, 'shap_importance.png', 'shap_importance.csv')
            if ok:
                print("shap_importance.png saved")
                print("shap_importance.csv saved")
        except Exception as e:
            print(f"SHAP skipped: {e}")

    if ENABLE_SHAP_REFINEMENT and ENABLE_SHAP:
        try:
            imp = pd.read_csv('shap_importance.csv', index_col=0).iloc[:, 0]
            top = imp.head(int(SHAP_TOP_N)).index.tolist()
            if len(top) >= 5:
                print("\n" + "=" * 60)
                print("TOP1 PIPELINE: SHAP-driven feature refinement")
                print("=" * 60)
                X_ref = X[top]
                X_test_ref = X_test[top]
                oof_base, test_base, stability = oof_multi_seed(X_ref, y, X_test_ref, SEEDS, N_SPLITS)
                meta_model, meta_keys, oof_meta, test_meta = stack_meta(oof_base, y, test_base)
                oof_ens = weighted_ensemble(oof_base, oof_meta)
                test_ens = weighted_ensemble(test_base, test_meta)
                if ENABLE_CALIBRATION:
                    try:
                        iso = IsotonicRegression(out_of_bounds='clip')
                        iso.fit(oof_ens, y.values)
                        oof_ens_cal = iso.transform(oof_ens)
                        test_ens_cal = iso.transform(test_ens)
                    except Exception:
                        iso = None
                        oof_ens_cal = oof_ens
                        test_ens_cal = test_ens
                else:
                    iso = None
                    oof_ens_cal = oof_ens
                    test_ens_cal = test_ens

                best_threshold, best_f1 = tune_threshold(y.values, oof_ens_cal)
                final_preds = (test_ens_cal >= best_threshold).astype(int)
                submission = pd.DataFrame({'ID': test_ids, 'CLASS': final_preds})
                submission.to_csv('FINAL.csv', index=False)
                print(f"Refined run best OOF F1: {best_f1:.4f} @ t={best_threshold:.2f}")
        except Exception as e:
            print(f"SHAP refinement skipped: {e}")

    summary = {
        'train_path': TRAIN_PATH,
        'test_path': TEST_PATH,
        'train_shape': [int(train.shape[0]), int(train.shape[1])],
        'test_shape': [int(test.shape[0]), int(test.shape[1])],
        'x_shape': [int(X.shape[0]), int(X.shape[1])],
        'x_test_shape': [int(X_test.shape[0]), int(X_test.shape[1])],
        'fast_smoke_test': bool(FAST_SMOKE_TEST),
        'seeds': [int(s) for s in SEEDS],
        'n_splits': int(N_SPLITS),
        'enable_adversarial_validation': bool(ENABLE_ADVERSARIAL_VALIDATION),
        'adversarial_auc': adv_auc,
        'drop_features': drop_features,
        'available_base_models': available,
        'meta_keys': meta_keys,
        'enable_calibration': bool(ENABLE_CALIBRATION),
        'enable_pseudo_labeling_top1': bool(ENABLE_PSEUDO_LABELING_TOP1),
        'enable_optuna': bool(ENABLE_OPTUNA),
        'enable_weight_optimization': bool(ENABLE_WEIGHT_OPTIMIZATION),
        'enable_cv_stability_report': bool(ENABLE_CV_STABILITY_REPORT),
        'enable_shap': bool(ENABLE_SHAP),
        'enable_shap_refinement': bool(ENABLE_SHAP_REFINEMENT),
        'shap_top_n': int(SHAP_TOP_N),
        'enable_feature_importance_filter': bool(ENABLE_FEATURE_IMPORTANCE_FILTER),
        'feature_importance_top_n': int(FEATURE_IMPORTANCE_TOP_N),
        'enable_plots': bool(ENABLE_PLOTS),
        'pseudo_pos_th': float(PSEUDO_POS_TH),
        'pseudo_neg_th': float(PSEUDO_NEG_TH),
        'best_threshold': float(best_threshold),
        'best_oof_f1': float(best_f1),
        'stability': stability,
        'blend_optimized': blend_opt,
        'weights': {
            'xgb': float(XGB_WEIGHT),
            'lgbm': float(LGBM_WEIGHT),
            'cat': float(CAT_WEIGHT),
            'rf': float(RF_WEIGHT),
            'tabnet': float(TABNET_WEIGHT),
            'meta': float(META_WEIGHT)
        },
        'runtime_seconds': float(time.time() - t0)
    }
    with open('run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TOP1 PIPELINE: Outputs")
    print("=" * 60)
    print("FINAL.csv saved")
    print("final_model.pkl saved")
    print("run_summary.json saved")
    print(submission.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
