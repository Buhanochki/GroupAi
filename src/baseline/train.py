"""
Enhanced training script with temporal cross-validation and ensemble support.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import ParameterGrid

from . import config, constants
from .evaluate import calculate_stage2_metrics, ndcg_at_k
from .features import add_aggregate_features, add_advanced_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


class TemporalCrossValidator:
    """Temporal cross-validation for time-series data."""
    
    def __init__(self, n_splits: int = 3):
        self.n_splits = n_splits
        
    def split(self, df: pd.DataFrame, timestamp_col: str = constants.COL_TIMESTAMP):
        """Generate temporal splits."""
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        dates = df_sorted[timestamp_col].unique()
        
        # Create split points
        split_indices = np.linspace(0, len(dates), self.n_splits + 1, dtype=int)[1:-1]
        split_dates = [dates[idx] for idx in split_indices]
        
        for split_date in split_dates:
            train_mask = df_sorted[timestamp_col] <= split_date
            val_mask = df_sorted[timestamp_col] > split_date
            
            train_indices = df_sorted[train_mask].index
            val_indices = df_sorted[val_mask].index
            
            yield train_indices, val_indices


def prepare_features(
    train_split: pd.DataFrame, 
    val_split: pd.DataFrame, 
    full_train_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Prepare features with proper temporal handling."""
    
    # Compute aggregates on train split only
    print("Computing aggregate features...")
    train_with_agg = add_aggregate_features(train_split.copy(), train_split)
    train_with_agg = add_advanced_aggregate_features(train_with_agg, train_split)
    
    val_with_agg = add_aggregate_features(val_split.copy(), train_split)  # Use train for aggregates!
    val_with_agg = add_advanced_aggregate_features(val_with_agg, train_split)
    
    # Handle missing values
    print("Handling missing values...")
    train_final = handle_missing_values(train_with_agg, train_split)
    val_final = handle_missing_values(val_with_agg, train_split)
    
    # Define features
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
        constants.COL_HAS_READ,
        constants.COL_TARGET,
    ]
    
    features = [col for col in train_final.columns if col not in exclude_cols]
    
    # Exclude object columns
    non_feature_object_cols = train_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]
    
    # Select only numerical and categorical features we care about
    final_features = []
    for feature in features:
        if (feature in config.CAT_FEATURES or 
            feature in config.NUMERICAL_FEATURES or
            feature.startswith('tfidf_') or 
            feature.startswith('bert_')):
            final_features.append(feature)
    
    print(f"Selected {len(final_features)} features for training")
    
    X_train = train_final[final_features].copy()
    y_train = train_final[config.TARGET]
    X_val = val_final[final_features].copy() 
    y_val = val_final[config.TARGET]
    
    # Optimize memory
    float64_cols = X_train.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        X_train[float64_cols] = X_train[float64_cols].astype("float32")
        X_val[float64_cols] = X_val[float64_cols].astype("float32")
    
    return X_train, X_val, y_train, y_val, final_features


def evaluate_ndcg(model, X_val, y_val, val_split: pd.DataFrame) -> float:
    """Evaluate model using NDCG@20 metric."""
    
    # Get predictions
    val_proba = model.predict_proba(X_val)  # [p_cold, p_planned, p_read]
    
    # Calculate ranking scores (weighted sum)
    ranking_scores = val_proba[:, 1] * 1.0 + val_proba[:, 2] * 2.0
    
    # Create temporary submission format for evaluation
    val_split = val_split.copy()
    val_split['prediction_score'] = ranking_scores
    
    # For each user, rank their books and calculate NDCG
    user_ndcgs = []
    
    for user_id in val_split[constants.COL_USER_ID].unique():
        user_books = val_split[val_split[constants.COL_USER_ID] == user_id]
        
        if len(user_books) == 0:
            continue
            
        # Sort by prediction score
        user_books_sorted = user_books.sort_values('prediction_score', ascending=False)
        
        # Get true relevance scores
        relevance_scores = user_books_sorted[config.TARGET].tolist()
        
        # Calculate NDCG@20
        user_ndcg = ndcg_at_k(relevance_scores, k=min(20, len(relevance_scores)))
        user_ndcgs.append(user_ndcg)
    
    return np.mean(user_ndcgs) if user_ndcgs else 0.0


def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series, 
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_params: Dict,
    model_name: str = "model"
) -> lgb.LGBMClassifier:
    """Train a single LightGBM model with enhanced callbacks."""
    
    print(f"\nTraining {model_name}...")
    
    model = lgb.LGBMClassifier(**model_params)
    
    # Enhanced callbacks
    callbacks = [
        lgb.early_stopping(
            stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose=True,
            first_metric_only=False
        ),
        lgb.log_evaluation(period=50),
    ]
    
    # Identify categorical features
    categorical_features = [
        f for f in X_train.columns if X_train[f].dtype.name == "category"
    ]
    categorical_feature_indices = [
        X_train.columns.get_loc(f) for f in categorical_features if f in X_train.columns
    ]
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=config.LGB_FIT_PARAMS["eval_metric"],
        callbacks=callbacks,
        categorical_feature=categorical_feature_indices if categorical_feature_indices else 'auto',
    )
    
    # Safe evaluation
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    
    accuracy, loss, class_dist, class_proba_mean = calculate_safe_metrics(
        y_val, val_preds, val_proba
    )
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Log Loss: {loss:.4f}")
    print(f"  Predicted class distribution: {class_dist.to_dict()}")
    
    return model

def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    features: List[str]
) -> Dict[str, lgb.LGBMClassifier]:
    """Train ensemble of models with different parameters."""
    
    ensemble = {}
    
    for model_name, model_params in config.ENSEMBLE_MODELS:
        model = train_single_model(
            X_train, y_train, X_val, y_val, model_params, model_name
        )
        ensemble[model_name] = model
        
        # Evaluate individual model
        val_preds = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        
        accuracy = accuracy_score(y_val, val_preds)
        
        # Fix for log_loss: explicitly specify labels
        try:
            loss = log_loss(y_val, val_proba, labels=[0, 1, 2])
        except ValueError:
            # If some classes are missing, use available classes
            available_classes = sorted(y_val.unique())
            loss = log_loss(y_val, val_proba, labels=available_classes)
        
        # Calculate class distribution
        class_dist = pd.Series(val_preds).value_counts().sort_index()
        
        print(f"{model_name} - Accuracy: {accuracy:.4f}, Log Loss: {loss:.4f}")
        print(f"  Predicted class distribution: {class_dist.to_dict()}")
    
    return ensemble


def train() -> None:
    """Enhanced training pipeline with temporal CV and ensemble support."""
    
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train set
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Perform temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError("Temporal split validation failed!")

    print("✅ Temporal split validation passed")

    # Prepare features
    X_train, X_val, y_train, y_val, features = prepare_features(
        train_split, val_split, train_set
    )

    print(f"Final training data shape: {X_train.shape}")
    print(f"Final validation data shape: {X_val.shape}")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train ensemble
    print("\n" + "="*50)
    print("TRAINING ENSEMBLE")
    print("="*50)
    
    ensemble = train_ensemble(X_train, y_train, X_val, y_val, features)

    # Evaluate ensemble
    print("\n" + "="*50)
    print("ENSEMBLE EVALUATION")
    print("="*50)
    
    best_ndcg = 0
    best_model_name = None
    
    for model_name, model in ensemble.items():
        # Calculate NDCG
        model_ndcg = evaluate_ndcg(model, X_val, y_val, val_split)
        print(f"{model_name} - NDCG@20: {model_ndcg:.4f}")
        
        if model_ndcg > best_ndcg:
            best_ndcg = model_ndcg
            best_model_name = model_name
    
    print(f"\nBest model: {best_model_name} with NDCG@20: {best_ndcg:.4f}")

    # Save best model and ensemble
    print("\nSaving models...")
    
    # Save best model as main model
    best_model = ensemble[best_model_name]
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    best_model.booster_.save_model(str(model_path))
    print(f"Best model saved to {model_path}")

    # Save ensemble
    for model_name, model in ensemble.items():
        ensemble_path = config.MODEL_DIR / f"ensemble_{model_name}.txt"
        model.booster_.save_model(str(ensemble_path))
    
    # Save feature list
    features_path = config.MODEL_DIR / "features_list.json"
    with open(features_path, "w") as f:
        json.dump(features, f)
    print(f"Feature list saved to {features_path}")

    # Save ensemble info
    ensemble_info = {
        "best_model": best_model_name,
        "best_ndcg": best_ndcg,
        "models": list(ensemble.keys()),
        "features_count": len(features)
    }
    
    ensemble_info_path = config.MODEL_DIR / "ensemble_info.json"
    with open(ensemble_info_path, "w") as f:
        json.dump(ensemble_info, f, indent=2)
    
    print(f"Ensemble info saved to {ensemble_info_path}")
    print("\n🎉 Training complete!")

def calculate_safe_metrics(y_true, y_pred, y_proba):
    """Calculate metrics safely when some classes are missing."""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle log_loss with possible missing classes
    try:
        loss = log_loss(y_true, y_proba, labels=[0, 1, 2])
    except ValueError:
        available_classes = sorted(y_true.unique())
        loss = log_loss(y_true, y_proba, labels=available_classes)
    
    # Class distribution
    class_dist = pd.Series(y_pred).value_counts().sort_index()
    class_proba_mean = y_proba.mean(axis=0)
    
    return accuracy, loss, class_dist, class_proba_mean


if __name__ == "__main__":
    train()