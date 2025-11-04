"""
Inference script to generate predictions for the test set.

Computes aggregate features on all train data and applies them to test set,
then generates predictions using the trained model and ranks candidates for each user.
"""

import lightgbm as lgb
import pandas as pd

from . import config, constants
from .data_processing import expand_candidates, load_and_merge_data
from .features import add_aggregate_features, handle_missing_values


def predict() -> None:
    """Generates and saves ranked predictions for the test set.

    This script:
    1. Loads targets.csv and candidates.csv
    2. Expands candidates into (user_id, book_id) pairs
    3. Computes aggregate features on all train data
    4. Generates probabilities using the trained model
    5. Ranks candidates for each user and selects top-K (K = min(20, num_candidates))
    6. Saves submission.csv in format: user_id,book_id_list

    Note: Data must be prepared first using prepare_data.py, and model must be trained
    using train.py
    """
    # Load targets and candidates
    print("Loading targets and candidates...")
    targets_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.TARGETS_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )
    candidates_df = pd.read_csv(
        config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME,
        dtype={constants.COL_USER_ID: "int32"},
    )

    print(f"Targets: {len(targets_df):,} users")
    print(f"Candidates: {len(candidates_df):,} users")

    # Expand candidates into pairs
    print("\nExpanding candidates...")
    candidates_pairs_df = expand_candidates(candidates_df)

    # Load prepared data for base features
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    # Get train data for computing aggregates
    train_df = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Load metadata for candidates
    print("Loading metadata...")
    _, _, _, book_genres_df, descriptions_df = load_and_merge_data()
    # We need users and books data separately
    user_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.USER_DATA_FILENAME)
    book_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME)

    # Merge metadata with candidates
    print("Merging metadata with candidates...")
    candidates_with_meta = candidates_pairs_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    candidates_with_meta = candidates_with_meta.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")

    # Add base features from prepared data (genres, text features)
    # We'll match by book_id to get TF-IDF and BERT features
    book_features = featured_df[[constants.COL_BOOK_ID]].drop_duplicates()
    # Get all feature columns except metadata and source columns
    feature_cols = [
        col
        for col in featured_df.columns
        if col
        not in [
            constants.COL_USER_ID,
            constants.COL_BOOK_ID,
            constants.COL_SOURCE,
            constants.COL_TIMESTAMP,
            constants.COL_HAS_READ,
            constants.COL_TARGET,
            constants.COL_PREDICTION,
            constants.COL_GENDER,
            constants.COL_AGE,
            constants.COL_AUTHOR_ID,
            constants.COL_PUBLICATION_YEAR,
            constants.COL_LANGUAGE,
            constants.COL_PUBLISHER,
            constants.COL_AVG_RATING,
        ]
        and not col.startswith("tfidf_")
        and not col.startswith("bert_")
    ]

    # Add genre count and text features
    # Get a representative row for each book (just take first occurrence)
    book_features_df = featured_df[[constants.COL_BOOK_ID] + feature_cols].drop_duplicates(
        subset=[constants.COL_BOOK_ID]
    )

    # Merge book features
    candidates_with_meta = candidates_with_meta.merge(
        book_features_df, on=constants.COL_BOOK_ID, how="left", suffixes=("", "_from_prep")
    )

    # Get TF-IDF and BERT features from prepared data
    tfidf_cols = [col for col in featured_df.columns if col.startswith("tfidf_")]
    bert_cols = [col for col in featured_df.columns if col.startswith("bert_")]
    text_feature_cols = tfidf_cols + bert_cols

    if text_feature_cols:
        book_text_features = featured_df[[constants.COL_BOOK_ID] + text_feature_cols].drop_duplicates(
            subset=[constants.COL_BOOK_ID]
        )
        candidates_with_meta = candidates_with_meta.merge(
            book_text_features, on=constants.COL_BOOK_ID, how="left"
        )

    # Compute aggregate features on ALL train data
    print("\nComputing aggregate features on all train data...")
    candidates_with_agg = add_aggregate_features(candidates_with_meta.copy(), train_df)

    # Handle missing values
    print("Handling missing values...")
    candidates_final = handle_missing_values(candidates_with_agg, train_df)

    # Define features (same as in train.py)
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
    ]
    features = [col for col in candidates_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = candidates_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_test = candidates_final[features]
    print(f"Prediction features: {len(features)}")

    # Load trained model
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " "Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"\nLoading model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))

    # Generate probabilities
    print("Generating predictions...")
    test_proba = model.predict(X_test)

    # Add predictions to candidates dataframe
    candidates_final["prediction"] = test_proba

    # Rank candidates for each user and select top-K
    print("\nRanking candidates for each user...")
    submission_rows = []

    for user_id in targets_df[constants.COL_USER_ID]:
        user_candidates = candidates_final[candidates_final[constants.COL_USER_ID] == user_id].copy()

        if len(user_candidates) == 0:
            # No candidates for this user - empty list
            book_id_list = ""
        else:
            # Sort by prediction probability (descending)
            user_candidates = user_candidates.sort_values("prediction", ascending=False)

            # Select top-K, where K = min(20, num_candidates)
            k = min(constants.MAX_RANKING_LENGTH, len(user_candidates))
            top_books = user_candidates.head(k)

            # Create comma-separated string of book_ids
            book_id_list = ",".join([str(int(book_id)) for book_id in top_books[constants.COL_BOOK_ID]])

        submission_rows.append({constants.COL_USER_ID: user_id, constants.COL_BOOK_ID_LIST: book_id_list})

    # Create submission DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    # Save submission
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file created at: {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

    # Print statistics
    non_empty = submission_df[submission_df[constants.COL_BOOK_ID_LIST] != ""].shape[0]
    print(f"Users with recommendations: {non_empty}/{len(submission_df)}")


if __name__ == "__main__":
    predict()

