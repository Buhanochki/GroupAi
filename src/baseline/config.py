"""
Enhanced configuration file with optimized parameters for NDCG@20.
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# --- PARAMETERS ---
N_SPLITS = 3  # For temporal cross-validation
RANDOM_STATE = 42
TARGET = constants.COL_RELEVANCE  # Multiclass target: 0=cold, 1=planned, 2=read

# --- TEMPORAL SPLIT CONFIG ---
TEMPORAL_SPLIT_RATIO = 0.8

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 100  # Increased for better convergence
MODEL_FILENAME = "lgb_model.txt"

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 1000  # Increased for better text representation
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9  # Slightly more aggressive
TFIDF_NGRAM_RANGE = (1, 3)  # Include trigrams

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 16  # Increased for efficiency
BERT_MAX_LENGTH = 256  # Reduced for faster processing
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
BERT_GPU_MEMORY_FRACTION = 0.8  # Increased for better utilization

# --- FEATURES ---
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]

# New temporal and affinity features
NUMERICAL_FEATURES = [
    # User temporal features
    'user_tenure_days', 'user_total_interactions', 'user_read_ratio',
    'user_daily_interaction_rate', 'user_conversion_rate', 'user_engagement_score',
    
    # Book temporal features  
    'book_recent_popularity', 'book_historical_popularity', 'book_age_days',
    'book_popularity_trend', 'book_conversion_rate',
    
    # Affinity features
    'genre_match_score', 'author_interaction_count', 'author_read_ratio',
    
    # Existing features
    constants.F_USER_MEAN_RATING, constants.F_USER_RATINGS_COUNT,
    constants.F_BOOK_MEAN_RATING, constants.F_BOOK_RATINGS_COUNT,
    constants.F_AUTHOR_MEAN_RATING, constants.F_BOOK_GENRES_COUNT,
    constants.COL_AVG_RATING, 'book_genre_diversity'
]

# --- OPTIMIZED MODEL PARAMETERS ---
# Optimized for multiclass classification with NDCG focus
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 3000,  # Increased for better performance
    "learning_rate": 0.01,  # Optimal balance
    "num_leaves": 255,  # Increased for complex patterns
    "max_depth": -1,  # Unlimited depth
    "min_child_samples": 20,
    "min_child_weight": 0.001,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_split_gain": 0.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
    # Memory optimization
    "max_bin": 255,
    "force_row_wise": True,
    "feature_fraction": 0.9,  # Slightly higher
    "bagging_fraction": 0.9,  # Slightly higher
    "bagging_freq": 5,
}

# Alternative parameters for ensemble
LGB_PARAMS_ALT1 = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": 8,
    "min_child_samples": 30,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE + 1,  # Different seed for diversity
}

LGB_PARAMS_ALT2 = {
    "objective": "multiclass", 
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 2500,
    "learning_rate": 0.02,
    "num_leaves": 191,
    "max_depth": -1,
    "min_child_samples": 15,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE + 2,
}

# Ensemble configuration
ENSEMBLE_MODELS = [
    ("main", LGB_PARAMS),
    ("alt1", LGB_PARAMS_ALT1), 
    ("alt2", LGB_PARAMS_ALT2)
]

# Fit parameters
LGB_FIT_PARAMS = {
    "eval_metric": "multi_logloss",
    "callbacks": [],
}

# Ranking strategy
RANKING_STRATEGY = "hierarchical"  # Options: "hierarchical", "two_stage", "direct"
HIERARCHICAL_MARGINS = {
    "read_vs_planned": 1.5,    # Minimum gap between read and planned
    "planned_vs_cold": 1.0,    # Minimum gap between planned and cold
}