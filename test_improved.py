# ============================================================
#  ML alrIEEEna'26 — Machine Fault Detection (IMPROVED VERSION)
#  Complete Solution Script | IEEE SB GEHU
#  Goal: Maximize F1 Score on hidden test data
#  Output: FINAL.csv (ID -> Class predictions)
#  Improvements: Enhanced stability, error handling, validation
# ============================================================

# ── STEP 0: INSTALL (run this once in terminal) ──────────────
# pip install pandas numpy scikit-learn xgboost lightgbm optuna shap imbalanced-learn matplotlib seaborn

import pandas as pd
import numpy as np
import os
import warnings
import logging
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Configuration constants
RANDOM_STATE = 42
N_TRIALS_XGB = 100
N_TRIALS_LGBM = 100
ENABLE_OPTUNA = False
ENABLE_PSEUDO_LABELING = False
ENABLE_SHAP = False

# File paths with validation
TRAIN_PATH = os.getenv('TRAIN_PATH', 'TRAIN.csv')
TEST_PATH = os.getenv('TEST_PATH', 'TEST.csv')
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class ModelTrainingError(Exception):
    """Custom exception for model training errors"""
    pass

def validate_file_exists(filepath: str) -> bool:
    """Check if file exists and is readable"""
    try:
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return False
        if not path.is_file():
            logger.error(f"Path is not a file: {filepath}")
            return False
        if os.access(filepath, os.R_OK):
            return True
        else:
            logger.error(f"File is not readable: {filepath}")
            return False
    except Exception as e:
        logger.error(f"Error validating file {filepath}: {e}")
        return False

def validate_data_structure(df: pd.DataFrame, is_train: bool = True) -> bool:
    """Validate data structure and content"""
    try:
        expected_features = [f'F{str(i).zfill(2)}' for i in range(1, 48)]
        
        if is_train:
            required_cols = expected_features + ['Class']
            if 'Class' not in df.columns:
                logger.error("Train data missing 'Class' column")
                return False
        else:
            required_cols = ['ID'] + expected_features
            if 'ID' not in df.columns:
                logger.error("Test data missing 'ID' column")
                return False
        
        # Check for required columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        
        # Check for empty data
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        # Check for NaN values in critical columns
        if is_train:
            if df['Class'].isnull().any():
                logger.error("Train data contains NaN values in Class column")
                return False
        else:
            if df['ID'].isnull().any():
                logger.error("Test data contains NaN values in ID column")
                return False
        
        # Validate Class values (for train data)
        if is_train:
            unique_classes = set(df['Class'].unique())
            if not unique_classes.issubset({0, 1}):
                logger.error(f"Invalid class values found: {unique_classes}")
                return False
        
        logger.info(f"Data validation passed for {'train' if is_train else 'test'} data")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data structure: {e}")
        return False

def safe_load_data(filepath: str, is_train: bool = True) -> pd.DataFrame:
    """Safely load data with comprehensive error handling"""
    try:
        logger.info(f"Loading {'train' if is_train else 'test'} data from: {filepath}")
        
        # Validate file exists
        if not validate_file_exists(filepath):
            raise DataValidationError(f"File validation failed: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        
        # Validate data structure
        if not validate_data_structure(df, is_train):
            raise DataValidationError("Data structure validation failed")
        
        # Log basic statistics
        if is_train:
            class_dist = df['Class'].value_counts()
            logger.info(f"Class distribution: {class_dist.to_dict()}")
            logger.info(f"Class balance - 0: {class_dist[0]/len(df)*100:.1f}%, 1: {class_dist[1]/len(df)*100:.1f}%")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in the dataset")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise DataValidationError(f"Failed to load data: {e}")

def safe_feature_engineering(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Safe feature engineering with error handling"""
    try:
        logger.info("Starting feature engineering...")
        df = df.copy()
        feats = [f'F{str(i).zfill(2)}' for i in range(1, 48)]
        
        # Validate all features exist
        missing_features = set(feats) - set(df.columns)
        if missing_features:
            raise DataValidationError(f"Missing features for engineering: {missing_features}")
        
        # --- Global Row-wise Stats (7 features) ---
        logger.info("Computing global statistics...")
        df['mean_all'] = df[feats].mean(axis=1)
        df['std_all'] = df[feats].std(axis=1)
        df['max_all'] = df[feats].max(axis=1)
        df['min_all'] = df[feats].min(axis=1)
        df['range_all'] = df['max_all'] - df['min_all']
        df['skew_all'] = df[feats].skew(axis=1)
        df['kurt_all'] = df[feats].kurtosis(axis=1)
        
        # --- Group-level Stats (3 groups x 7 stats = 21 features) ---
        logger.info("Computing group-level statistics...")
        groupA = [f for f in feats if int(f[1:]) <= 16]   # F01-F16
        groupB = [f for f in feats if 17 <= int(f[1:]) <= 32]  # F17-F32
        groupC = [f for f in feats if int(f[1:]) >= 33]   # F33-F47
        
        for name, grp in [('A', groupA), ('B', groupB), ('C', groupC)]:
            df[f'mean_{name}'] = df[grp].mean(axis=1)
            df[f'std_{name}'] = df[grp].std(axis=1)
            df[f'max_{name}'] = df[grp].max(axis=1)
            df[f'min_{name}'] = df[grp].min(axis=1)
            df[f'range_{name}'] = df[f'max_{name}'] - df[f'min_{name}']
            df[f'skew_{name}'] = df[grp].skew(axis=1)
            df[f'kurt_{name}'] = df[grp].kurtosis(axis=1)
        
        # --- Cross-Group Ratios (6 features) ---
        logger.info("Computing cross-group ratios...")
        epsilon = 1e-5  # Small value to prevent division by zero
        df['ratio_A_B'] = df['mean_A'] / (df['mean_B'] + epsilon)
        df['ratio_B_C'] = df['mean_B'] / (df['mean_C'] + epsilon)
        df['ratio_A_C'] = df['mean_A'] / (df['mean_C'] + epsilon)
        df['std_ratio_A_B'] = df['std_A'] / (df['std_B'] + epsilon)
        df['std_ratio_B_C'] = df['std_B'] / (df['std_C'] + epsilon)
        df['std_ratio_A_C'] = df['std_A'] / (df['std_C'] + epsilon)
        
        # --- Top Pairwise Interactions ---
        logger.info("Computing pairwise interactions...")
        top_feats = ['F01', 'F10', 'F08', 'F09', 'F06', 'F07']
        for i in range(len(top_feats)):
            for j in range(i+1, len(top_feats)):
                f1, f2 = top_feats[i], top_feats[j]
                df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
                df[f'{f1}_div_{f2}'] = df[f1] / (df[f2] + epsilon)
        
        # Check for infinite values
        inf_counts = np.isinf(df).sum()
        if inf_counts.sum() > 0:
            logger.warning(f"Found infinite values: {inf_counts[inf_counts > 0].to_dict()}")
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Check for NaN values created during engineering
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"Found NaN values after engineering: {nan_counts[nan_counts > 0].to_dict()}")
            # Fill NaN values with median for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise DataValidationError(f"Feature engineering failed: {e}")

def safe_preprocessing(train_eng: pd.DataFrame, test_eng: pd.DataFrame) -> Tuple:
    """Safe preprocessing with validation"""
    try:
        logger.info("Starting preprocessing...")
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Separate features and target
        drop_cols = ['Class'] if 'Class' in train_eng.columns else []
        X = train_eng.drop(columns=drop_cols)
        y = train_eng['Class'] if 'Class' in train_eng.columns else None
        
        # Test features
        if 'ID' in test_eng.columns:
            test_ids = test_eng['ID'].values
            X_test = test_eng.drop(columns=['ID'])
        else:
            raise DataValidationError("Test data missing 'ID' column")
        
        # Align columns
        missing_cols = set(X.columns) - set(X_test.columns)
        if missing_cols:
            logger.error(f"Test data missing columns: {missing_cols}")
            raise DataValidationError("Column mismatch between train and test")
        
        X_test = X_test[X.columns]
        
        # Validate shapes
        if X.shape[0] == 0 or X_test.shape[0] == 0:
            raise DataValidationError("Empty feature matrices")
        
        # Train-Validation split (stratified)
        if y is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            logger.info(f"Train set: {X_train.shape[0]} rows, Val set: {X_val.shape[0]} rows")
            
            # Calculate class imbalance ratio
            class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
            logger.info(f"Class imbalance ratio (0/1): {class_ratio:.3f}")
        else:
            raise DataValidationError("No target variable found")
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)
        
        # Check for scaling issues
        if np.isnan(X_train_sc).any() or np.isnan(X_val_sc).any() or np.isnan(X_test_sc).any():
            raise DataValidationError("NaN values found after scaling")
        
        if np.isinf(X_train_sc).any() or np.isinf(X_val_sc).any() or np.isinf(X_test_sc).any():
            raise DataValidationError("Infinite values found after scaling")
        
        logger.info("Preprocessing completed successfully")
        return (X_train, X_val, X_test, y_train, y_val, test_ids, scaler, class_ratio)
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise DataValidationError(f"Preprocessing failed: {e}")

def safe_model_training(model, X_train, y_train, X_val, y_val, model_name: str) -> Tuple:
    """Safe model training with error handling and overfitting detection"""
    try:
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        # Fit model
        if hasattr(model, 'fit'):
            if 'eval_set' in model.fit.__code__.co_varnames:
                # Model supports early stopping
                try:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                except TypeError:
                    # Some models don't accept verbose parameter
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            else:
                model.fit(X_train, y_train)
        else:
            raise ModelTrainingError(f"Model {model_name} doesn't have fit method")
        
        training_time = time.time() - start_time
        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        
        # Make predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        # Validate predictions
        if len(val_preds) != len(y_val):
            raise ModelTrainingError(f"Prediction length mismatch in {model_name}")
        
        unique_preds = set(val_preds)
        if not unique_preds.issubset({0, 1}):
            logger.warning(f"Unexpected prediction values in {model_name}: {unique_preds}")
        
        # Calculate overfitting metrics
        train_metrics = calculate_metrics(y_train, train_preds)
        val_metrics = calculate_metrics(y_val, val_preds)
        
        # Overfitting detection
        overfitting_score = train_metrics['f1'] - val_metrics['f1']
        if overfitting_score > 0.1:
            logger.warning(f"{model_name} shows signs of overfitting (Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f})")
        elif overfitting_score < -0.05:
            logger.warning(f"{model_name} shows signs of underfitting (Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f})")
        
        return model, val_preds, train_metrics, val_metrics
        
    except Exception as e:
        logger.error(f"Error training {model_name}: {e}")
        raise ModelTrainingError(f"Model training failed for {model_name}: {e}")

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Calculate classification metrics safely"""
    try:
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
        
        metrics = {
            'f1': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }
        
        logger.info(f"Metrics - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                   f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}

def safe_threshold_tuning(model, X_val, y_val) -> Tuple[float, float]:
    """Safe threshold tuning"""
    try:
        logger.info("Tuning prediction threshold...")
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            val_probs = model.predict_proba(X_val)[:, 1]
        else:
            logger.warning("Model doesn't support predict_proba, using default threshold")
            return 0.5, 0.0
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        thresholds = np.arange(0.3, 0.71, 0.05)
        for thresh in thresholds:
            preds_thresh = (val_probs >= thresh).astype(int)
            f1 = f1_score(y_val, preds_thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        logger.info(f"Optimal threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")
        return best_threshold, best_f1
        
    except Exception as e:
        logger.error(f"Error in threshold tuning: {e}")
        return 0.5, 0.0

def test_edge_cases(model, X_train, y_train, X_val, y_val, model_name: str) -> Dict[str, Any]:
    """Test model on edge cases"""
    try:
        logger.info(f"Testing {model_name} on edge cases...")
        edge_results = {}
        
        # Test 1: Empty prediction scenario
        try:
            # Create a small test set with all zeros
            X_zeros = np.zeros((10, X_train.shape[1]))
            y_zeros = np.zeros(10)
            edge_preds = model.predict(X_zeros)
            edge_results['zeros_test'] = {
                'passed': len(set(edge_preds)) <= 2,
                'unique_preds': set(edge_preds)
            }
        except Exception as e:
            edge_results['zeros_test'] = {'passed': False, 'error': str(e)}
        
        # Test 2: Single sample prediction
        try:
            single_sample = X_train[:1]
            single_pred = model.predict(single_sample)
            edge_results['single_sample'] = {
                'passed': len(single_pred) == 1,
                'prediction': single_pred[0] if len(single_pred) > 0 else None
            }
        except Exception as e:
            edge_results['single_sample'] = {'passed': False, 'error': str(e)}
        
        # Test 3: Large values prediction
        try:
            X_large = X_train * 1000
            large_preds = model.predict(X_large)
            edge_results['large_values'] = {
                'passed': len(set(large_preds)) <= 2,
                'unique_preds': set(large_preds)
            }
        except Exception as e:
            edge_results['large_values'] = {'passed': False, 'error': str(e)}
        
        # Test 4: Prediction probabilities consistency
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_val)
                prob_sum_check = np.allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
                prob_range_check = np.all((probs >= 0) & (probs <= 1))
                edge_results['probabilities'] = {
                    'passed': prob_sum_check and prob_range_check,
                    'sums_to_one': prob_sum_check,
                    'in_range_0_1': prob_range_check
                }
            else:
                edge_results['probabilities'] = {'passed': True, 'note': 'No predict_proba method'}
        except Exception as e:
            edge_results['probabilities'] = {'passed': False, 'error': str(e)}
        
        # Log edge case results
        for test_name, result in edge_results.items():
            if result.get('passed', False):
                logger.info(f"Edge case {test_name}: PASSED")
            else:
                logger.warning(f"Edge case {test_name}: FAILED - {result}")
        
        return edge_results
        
    except Exception as e:
        logger.error(f"Error in edge case testing: {e}")
        return {'error': str(e)}

def cross_validate_model(model, X, y, model_name: str, cv_folds: int = 5) -> Dict[str, float]:
    """Perform cross-validation for robust evaluation"""
    try:
        logger.info(f"Performing {cv_folds}-fold cross-validation for {model_name}...")
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone model to avoid fitting on same data
            from sklearn.base import clone
            model_fold = clone(model)
            
            # Train on fold
            if hasattr(model_fold, 'fit'):
                model_fold.fit(X_fold_train, y_fold_train)
                fold_preds = model_fold.predict(X_fold_val)
                fold_f1 = f1_score(y_fold_val, fold_preds)
                cv_scores.append(fold_f1)
                logger.info(f"Fold {fold + 1}: F1 = {fold_f1:.4f}")
            else:
                logger.warning(f"Model {model_name} doesn't support cloning for CV")
                return {'cv_f1_mean': 0.0, 'cv_f1_std': 0.0}
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        logger.info(f"Cross-validation results - Mean F1: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return {
            'cv_f1_mean': cv_mean,
            'cv_f1_std': cv_std,
            'cv_scores': cv_scores
        }
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        return {'cv_f1_mean': 0.0, 'cv_f1_std': 0.0, 'error': str(e)}

def safe_save_predictions(predictions, test_ids, output_path: str) -> bool:
    """Save predictions with comprehensive validation"""
    try:
        logger.info(f"Saving predictions to {output_path}")
        
        # Validate inputs
        if len(predictions) != len(test_ids):
            logger.error("Length mismatch between predictions and test IDs")
            return False
        
        if not set(predictions).issubset({0, 1}):
            logger.error(f"Invalid prediction values: {set(predictions)}")
            return False
        
        # Additional validation checks
        pred_array = np.array(predictions)
        
        # Check for NaN or infinite values
        if np.isnan(pred_array).any():
            logger.error("Predictions contain NaN values")
            return False
        
        if np.isinf(pred_array).any():
            logger.error("Predictions contain infinite values")
            return False
        
        # Check prediction distribution
        class_dist = pd.Series(predictions).value_counts()
        if len(class_dist) == 1:
            logger.warning(f"Model predicts only one class: {class_dist.to_dict()}")
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'ID': test_ids,
            'Class': predictions
        })
        
        # Validate DataFrame
        if submission.isnull().any().any():
            logger.error("Submission DataFrame contains null values")
            return False
        
        # Save to file
        submission.to_csv(output_path, index=False)
        
        # Verify file was saved correctly
        if not os.path.exists(output_path):
            logger.error("Output file was not created")
            return False
        
        # Verify content
        saved_df = pd.read_csv(output_path)
        if len(saved_df) != len(submission):
            logger.error("Saved file has incorrect number of rows")
            return False
        
        if not saved_df['ID'].equals(submission['ID']):
            logger.error("ID column mismatch in saved file")
            return False
        
        if not saved_df['Class'].equals(submission['Class']):
            logger.error("Class column mismatch in saved file")
            return False
        
        logger.info(f"Prediction distribution: {class_dist.to_dict()}")
        logger.info(f"Successfully saved {len(submission)} predictions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        return False

def main():
    """Main pipeline execution"""
    try:
        logger.info("="*60)
        logger.info("STARTING IMPROVED ML PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load Data
        logger.info("STEP 1: Loading Data")
        train = safe_load_data(TRAIN_PATH, is_train=True)
        test = safe_load_data(TEST_PATH, is_train=False)
        
        # Step 2: Feature Engineering
        logger.info("STEP 2: Feature Engineering")
        train_eng = safe_feature_engineering(train, is_train=True)
        test_eng = safe_feature_engineering(test, is_train=False)
        
        # Step 3: Preprocessing
        logger.info("STEP 3: Preprocessing")
        X_train, X_val, X_test, y_train, y_val, test_ids, scaler, class_ratio = safe_preprocessing(train_eng, test_eng)
        
        # Step 4: Model Training
        logger.info("STEP 4: Model Training")
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from sklearn.metrics import f1_score
        
        models = {}
        model_scores = {}
        
        # Baseline: Logistic Regression
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        lr_model, lr_preds, lr_train_metrics, lr_val_metrics = safe_model_training(lr, X_train, y_train, X_val, y_val, "Logistic Regression")
        models['lr'] = lr_model
        model_scores['lr'] = lr_val_metrics
        
        # XGBoost
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
            n_jobs=-1,
            verbosity=0
        )
        xgb_model, xgb_preds, xgb_train_metrics, xgb_val_metrics = safe_model_training(xgb, X_train, y_train, X_val, y_val, "XGBoost")
        models['xgb'] = xgb_model
        model_scores['xgb'] = xgb_val_metrics
        
        # LightGBM
        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            silent=True
        )
        lgbm_model, lgbm_preds, lgbm_train_metrics, lgbm_val_metrics = safe_model_training(lgbm, X_train, y_train, X_val, y_val, "LightGBM")
        models['lgbm'] = lgbm_model
        model_scores['lgbm'] = lgbm_val_metrics
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf_model, rf_preds, rf_train_metrics, rf_val_metrics = safe_model_training(rf, X_train, y_train, X_val, y_val, "Random Forest")
        models['rf'] = rf_model
        model_scores['rf'] = rf_val_metrics
        
        # Step 5: Stacking Ensemble
        logger.info("STEP 5: Stacking Ensemble")
        from sklearn.ensemble import StackingClassifier
        
        # Select best performing models for stacking
        best_models = sorted(model_scores.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        logger.info(f"Top 3 models for stacking: {[name for name, _ in best_models]}")
        
        stacking = StackingClassifier(
            estimators=[
                (name, models[name]) for name, _ in best_models
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=5,
            passthrough=False,
            n_jobs=-1
        )
        
        stacking_model, stack_preds, stack_train_metrics, stack_val_metrics = safe_model_training(stacking, X_train, y_train, X_val, y_val, "Stacking Ensemble")
        model_scores['stacking'] = stack_val_metrics
        
        # Step 6: Threshold Tuning
        logger.info("STEP 6: Threshold Tuning")
        best_threshold, best_f1 = safe_threshold_tuning(stacking_model, X_val, y_val)
        
        # Step 7: Final Predictions
        logger.info("STEP 7: Final Predictions")
        final_probs = stacking_model.predict_proba(X_test)[:, 1]
        final_preds = (final_probs >= best_threshold).astype(int)
        
        # Step 8: Cross-Validation and Edge Case Testing
        logger.info("STEP 8: Cross-Validation and Edge Case Testing")
        
        # Perform cross-validation on best model
        cv_results = cross_validate_model(stacking_model, pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), "Stacking Ensemble")
        
        # Test edge cases on best model
        edge_results = test_edge_cases(stacking_model, X_train, y_train, X_val, y_val, "Stacking Ensemble")
        
        # Step 9: Save Results
        logger.info("STEP 9: Saving Results")
        output_path = OUTPUT_DIR / 'FINAL.csv'
        success = safe_save_predictions(final_preds, test_ids, str(output_path))
        
        if not success:
            raise RuntimeError("Failed to save predictions")
        
        # Save models
        import joblib
        joblib.dump(stacking_model, OUTPUT_DIR / 'final_model.pkl')
        joblib.dump(scaler, OUTPUT_DIR / 'scaler.pkl')
        logger.info("Models saved successfully")
        
        # Final Summary
        logger.info("="*60)
        logger.info("COMPREHENSIVE VALIDATION SUMMARY")
        logger.info("="*60)
        
        # Model performance summary
        logger.info("MODEL PERFORMANCE (VALIDATION SET):")
        for name, scores in model_scores.items():
            logger.info(f"  {name.upper():15} - F1: {scores['f1']:.4f}, Acc: {scores['accuracy']:.4f}, Prec: {scores['precision']:.4f}, Rec: {scores['recall']:.4f}")
        
        # Cross-validation results
        logger.info(f"\nCROSS-VALIDATION RESULTS:")
        logger.info(f"  CV Mean F1: {cv_results.get('cv_f1_mean', 0.0):.4f} ± {cv_results.get('cv_f1_std', 0.0):.4f}")
        if 'cv_scores' in cv_results:
            logger.info(f"  CV Scores: {[f'{s:.4f}' for s in cv_results['cv_scores']]}")
        
        # Overfitting analysis
        logger.info(f"\nOVERFITTING ANALYSIS:")
        logger.info(f"  Best Threshold: {best_threshold:.2f}")
        logger.info(f"  Final F1 Score: {best_f1:.4f}")
        
        # Edge case results
        logger.info(f"\nEDGE CASE TESTING:")
        for test_name, result in edge_results.items():
            if test_name != 'error':
                status = "PASSED" if result.get('passed', False) else "FAILED"
                logger.info(f"  {test_name}: {status}")
        
        # Stability assessment
        cv_std = cv_results.get('cv_f1_std', 1.0)
        stability_score = "HIGH" if cv_std < 0.02 else "MEDIUM" if cv_std < 0.05 else "LOW"
        logger.info(f"\nMODEL STABILITY: {stability_score} (CV Std: {cv_std:.4f})")
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
