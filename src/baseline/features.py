"""
Enhanced feature engineering script with temporal and advanced features.
"""

import time
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants


def add_temporal_user_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds advanced temporal features based on user interaction patterns.
    
    Args:
        df: Main DataFrame to add features to
        train_df: Training data for calculating temporal patterns
        
    Returns:
        DataFrame with temporal features added
    """
    print("Adding temporal user features...")
    
    # Ensure timestamp is datetime
    train_df = train_df.copy()
    train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])
    
    # User activity features
    user_activity = train_df.groupby(constants.COL_USER_ID).agg({
        constants.COL_TIMESTAMP: ['min', 'max', 'count'],
        constants.COL_HAS_READ: 'mean'
    }).reset_index()
    
    user_activity.columns = [
        constants.COL_USER_ID,
        'user_first_interaction',
        'user_last_interaction', 
        'user_total_interactions',
        'user_read_ratio'
    ]
    
    # Calculate user tenure in days
    reference_date = train_df[constants.COL_TIMESTAMP].max()
    user_activity['user_tenure_days'] = (
        reference_date - user_activity['user_last_interaction']
    ).dt.days
    
    user_activity['user_activity_days'] = (
        user_activity['user_last_interaction'] - user_activity['user_first_interaction']
    ).dt.days
    
    # Interaction frequency
    user_activity['user_daily_interaction_rate'] = (
        user_activity['user_total_interactions'] / 
        np.maximum(user_activity['user_activity_days'], 1)
    )
    
    # Merge with main dataframe
    df = df.merge(
        user_activity[[
            constants.COL_USER_ID,
            'user_tenure_days',
            'user_total_interactions', 
            'user_read_ratio',
            'user_daily_interaction_rate'
        ]], 
        on=constants.COL_USER_ID, 
        how='left'
    )
    
    return df


def add_book_temporal_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds temporal features for books based on popularity trends.
    
    Args:
        df: Main DataFrame to add features to
        train_df: Training data for calculating book trends
        
    Returns:
        DataFrame with book temporal features added
    """
    print("Adding book temporal features...")
    
    train_df = train_df.copy()
    train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])
    
    # Recent popularity (last 30 days)
    recent_cutoff = train_df[constants.COL_TIMESTAMP].max() - pd.Timedelta(days=30)
    recent_interactions = train_df[train_df[constants.COL_TIMESTAMP] > recent_cutoff]
    
    book_recent_popularity = recent_interactions.groupby(
        constants.COL_BOOK_ID
    ).size().reset_index(name='book_recent_popularity')
    
    # Book age based on first interaction
    book_first_interaction = train_df.groupby(constants.COL_BOOK_ID)[
        constants.COL_TIMESTAMP
    ].min().reset_index(name='book_first_seen')
    
    reference_date = train_df[constants.COL_TIMESTAMP].max()
    book_first_interaction['book_age_days'] = (
        reference_date - book_first_interaction['book_first_seen']
    ).dt.days
    
    # Popularity trend (recent vs historical)
    book_historical_popularity = train_df.groupby(
        constants.COL_BOOK_ID
    ).size().reset_index(name='book_historical_popularity')
    
    book_popularity = book_historical_popularity.merge(
        book_recent_popularity, on=constants.COL_BOOK_ID, how='left'
    ).merge(book_first_interaction, on=constants.COL_BOOK_ID, how='left')
    
    book_popularity['book_recent_popularity'] = book_popularity['book_recent_popularity'].fillna(0)
    book_popularity['book_popularity_trend'] = (
        book_popularity['book_recent_popularity'] / 
        np.maximum(book_popularity['book_historical_popularity'], 1)
    )
    
    # Merge with main dataframe
    df = df.merge(
        book_popularity[[
            constants.COL_BOOK_ID,
            'book_recent_popularity',
            'book_historical_popularity',
            'book_age_days',
            'book_popularity_trend'
        ]], 
        on=constants.COL_BOOK_ID, 
        how='left'
    )
    
    return df


def add_user_book_affinity_features(
    df: pd.DataFrame, 
    train_df: pd.DataFrame, 
    book_genres_df: pd.DataFrame
) -> pd.DataFrame:
    """Adds affinity features between users and books.
    
    Args:
        df: Main DataFrame to add features to
        train_df: Training data for calculating affinities
        book_genres_df: Book-genre relationships
        
    Returns:
        DataFrame with affinity features added
    """
    print("Adding user-book affinity features...")
    
    # User preferred genres
    user_genre_interactions = train_df.merge(
        book_genres_df, on=constants.COL_BOOK_ID, how='inner'
    )
    
    user_genre_preferences = user_genre_interactions.groupby([
        constants.COL_USER_ID, constants.COL_GENRE_ID
    ]).agg({
        constants.COL_HAS_READ: ['count', 'mean']
    }).reset_index()
    
    user_genre_preferences.columns = [
        constants.COL_USER_ID, constants.COL_GENRE_ID,
        'genre_interaction_count', 'genre_read_ratio'
    ]
    
    # Get top 3 genres per user
    user_top_genres = user_genre_preferences.sort_values(
        ['genre_interaction_count', 'genre_read_ratio'], ascending=[False, False]
    ).groupby(constants.COL_USER_ID).head(3)
    
    # Create genre preference sets per user
    user_preferred_genres = user_top_genres.groupby(constants.COL_USER_ID)[
        constants.COL_GENRE_ID
    ].apply(set).reset_index(name='preferred_genres')
    
    # Book genres
    book_genres = book_genres_df.groupby(constants.COL_BOOK_ID)[
        constants.COL_GENRE_ID
    ].apply(set).reset_index(name='book_genres')
    
    # Merge to calculate matches
    df_with_genres = df.merge(
        user_preferred_genres, on=constants.COL_USER_ID, how='left'
    ).merge(book_genres, on=constants.COL_BOOK_ID, how='left')
    
    # Calculate genre match score
    def calculate_genre_match(row):
        user_genres = row.get('preferred_genres', set())
        book_genres_set = row.get('book_genres', set())
        
        if not user_genres or not book_genres_set:
            return 0.0
            
        intersection = user_genres.intersection(book_genres_set)
        return len(intersection) / len(user_genres)
    
    df_with_genres['genre_match_score'] = df_with_genres.apply(calculate_genre_match, axis=1)
    
    # Keep only the match score in final dataframe
    df['genre_match_score'] = df_with_genres['genre_match_score']
    
    # Author affinity
    user_author_interactions = train_df.groupby([
        constants.COL_USER_ID, constants.COL_AUTHOR_ID
    ]).agg({
        constants.COL_HAS_READ: ['count', 'mean']
    }).reset_index()
    
    user_author_interactions.columns = [
        constants.COL_USER_ID, constants.COL_AUTHOR_ID,
        'author_interaction_count', 'author_read_ratio'
    ]
    
    # Merge author affinity
    df = df.merge(
        user_author_interactions[[
            constants.COL_USER_ID, constants.COL_AUTHOR_ID,
            'author_interaction_count', 'author_read_ratio'
        ]], 
        on=[constants.COL_USER_ID, constants.COL_AUTHOR_ID], 
        how='left'
    )
    
    # Fill missing values
    df['author_interaction_count'] = df['author_interaction_count'].fillna(0)
    df['author_read_ratio'] = df['author_read_ratio'].fillna(0)
    
    return df


def add_advanced_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds advanced aggregate features with temporal considerations.
    
    Args:
        df: Main DataFrame to add features to
        train_df: Training data for calculations
        
    Returns:
        DataFrame with advanced aggregate features
    """
    print("Adding advanced aggregate features...")
    
    # User conversion rate (planned -> read)
    user_planned_books = train_df[train_df[constants.COL_HAS_READ] == 0].groupby(
        constants.COL_USER_ID
    ).size().reset_index(name='user_planned_count')
    
    user_read_books = train_df[train_df[constants.COL_HAS_READ] == 1].groupby(
        constants.COL_USER_ID
    ).size().reset_index(name='user_read_count')
    
    user_conversion = user_planned_books.merge(user_read_books, on=constants.COL_USER_ID, how='left')
    user_conversion['user_read_count'] = user_conversion['user_read_count'].fillna(0)
    user_conversion['user_conversion_rate'] = (
        user_conversion['user_read_count'] / 
        np.maximum(user_conversion['user_planned_count'], 1)
    )
    
    # Book conversion rate
    book_planned = train_df[train_df[constants.COL_HAS_READ] == 0].groupby(
        constants.COL_BOOK_ID
    ).size().reset_index(name='book_planned_count')
    
    book_read = train_df[train_df[constants.COL_HAS_READ] == 1].groupby(
        constants.COL_BOOK_ID
    ).size().reset_index(name='book_read_count')
    
    book_conversion = book_planned.merge(book_read, on=constants.COL_BOOK_ID, how='left')
    book_conversion['book_read_count'] = book_conversion['book_read_count'].fillna(0)
    book_conversion['book_conversion_rate'] = (
        book_conversion['book_read_count'] / 
        np.maximum(book_conversion['book_planned_count'], 1)
    )
    
    # User engagement level based on interaction patterns
    user_engagement = train_df.groupby(constants.COL_USER_ID).agg({
        constants.COL_TIMESTAMP: ['min', 'max', 'nunique'],
        constants.COL_BOOK_ID: 'nunique',
        constants.COL_HAS_READ: 'mean'
    }).reset_index()
    
    user_engagement.columns = [
        constants.COL_USER_ID,
        'first_interaction', 'last_interaction', 'active_days',
        'unique_books', 'read_ratio'
    ]
    
    user_engagement['user_engagement_score'] = (
        user_engagement['active_days'] * 0.3 +
        user_engagement['unique_books'] * 0.4 + 
        user_engagement['read_ratio'] * 0.3
    )
    
    # Merge all advanced aggregates
    df = df.merge(
        user_conversion[[constants.COL_USER_ID, 'user_conversion_rate']], 
        on=constants.COL_USER_ID, how='left'
    )
    df = df.merge(
        book_conversion[[constants.COL_BOOK_ID, 'book_conversion_rate']], 
        on=constants.COL_BOOK_ID, how='left'
    )
    df = df.merge(
        user_engagement[[constants.COL_USER_ID, 'user_engagement_score']], 
        on=constants.COL_USER_ID, how='left'
    )
    
    # Fill missing values
    df['user_conversion_rate'] = df['user_conversion_rate'].fillna(0)
    df['book_conversion_rate'] = df['book_conversion_rate'].fillna(0)
    df['user_engagement_score'] = df['user_engagement_score'].fillna(0)
    
    return df


def add_interaction_feature(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced interaction feature with confidence scoring."""
    print("Adding enhanced interaction feature...")

    # Create set of (user_id, book_id) pairs from train data
    interaction_pairs = set(
        zip(train_df[constants.COL_USER_ID], train_df[constants.COL_BOOK_ID], strict=False)
    )

    # Create feature: 1 if pair exists in train, 0 otherwise
    df[constants.F_USER_BOOK_INTERACTION] = df.apply(
        lambda row: 1
        if (row[constants.COL_USER_ID], row[constants.COL_BOOK_ID]) in interaction_pairs
        else 0,
        axis=1,
    ).astype("int8")

    interaction_count = df[constants.F_USER_BOOK_INTERACTION].sum()
    print(f"  - Interactions found: {interaction_count:,} / {len(df):,} ({100 * interaction_count / len(df):.1f}%)")
    return df


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced aggregate features with additional metrics."""
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count", "std"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
        "user_rating_std"
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count", "std"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
        "book_rating_std"
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING, "author_ratings_count"]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")
    
    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced genre features."""
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    
    # Add genre diversity (if multiple genres exist)
    genre_diversity = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].nunique().reset_index()
    genre_diversity.columns = [constants.COL_BOOK_ID, 'book_genre_diversity']
    
    df = df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(genre_diversity, on=constants.COL_BOOK_ID, how="left")
    
    return df


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Keep existing TF-IDF implementation."""
    print("Adding text features (TF-IDF)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set
    train_books = train_df[constants.COL_BOOK_ID].unique()

    # Extract descriptions for training books only
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Check if vectorizer already exists (for prediction)
    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fit vectorizer on training descriptions only
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        # Save vectorizer for use in prediction
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    # Transform all book descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Create a mapping book_id -> description
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    # Get descriptions for books in df (in the same order)
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert sparse matrix to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    # Concatenate TF-IDF features with main DataFrame
    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf


def add_bert_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Keep existing BERT implementation."""
    print("Adding text features (BERT embeddings)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings (this may take a while)...")
        print(f"Using device: {config.BERT_DEVICE}")

        # Limit GPU memory usage to prevent OOM errors
        if config.BERT_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.BERT_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.BERT_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(config.BERT_DEVICE)
        model.eval()

        # Prepare descriptions: get unique book_id -> description mapping
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

        # Get unique books and their descriptions
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        # Initialize embeddings dictionary
        embeddings_dict = {}

        # Process descriptions in batches
        num_batches = (len(descriptions) + config.BERT_BATCH_SIZE - 1) // config.BERT_BATCH_SIZE

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing BERT batches", unit="batch"):
                start_idx = batch_idx * config.BERT_BATCH_SIZE
                end_idx = min(start_idx + config.BERT_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(config.BERT_DEVICE) for k, v in encoded.items()}

                # Get model outputs
                outputs = model(**encoded)

                # Mean pooling: average over sequence length dimension
                attention_mask = encoded["attention_mask"]
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

                # Sum embeddings, weighted by attention mask
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

                # Mean pooling
                mean_pooled = sum_embeddings / sum_mask

                # Convert to numpy and store
                batch_embeddings = mean_pooled.cpu().numpy()

                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                # Small pause between batches to let GPU cool down and prevent overheating
                if config.BERT_DEVICE == "cuda":
                    time.sleep(0.2)

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved BERT embeddings to {embeddings_path}")

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Create DataFrame with BERT features
    bert_feature_names = [f"bert_{i}" for i in range(config.BERT_EMBEDDING_DIM)]
    bert_df = pd.DataFrame(embeddings_array, columns=bert_feature_names, index=df.index)

    # Concatenate BERT features with main DataFrame
    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(bert_feature_names)} BERT features.")
    return df_with_bert


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced missing value handling for new features."""
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill temporal features
    temporal_features = [
        'user_tenure_days', 'user_total_interactions', 'user_read_ratio',
        'user_daily_interaction_rate', 'book_recent_popularity',
        'book_historical_popularity', 'book_age_days', 'book_popularity_trend',
        'user_conversion_rate', 'book_conversion_rate', 'user_engagement_score'
    ]
    
    for feature in temporal_features:
        if feature in df.columns:
            if 'rate' in feature or 'ratio' in feature:
                df[feature] = df[feature].fillna(0)
            else:
                df[feature] = df[feature].fillna(df[feature].median())

    # Fill affinity features
    affinity_features = ['genre_match_score', 'author_interaction_count', 'author_read_ratio']
    for feature in affinity_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)

    # Fill existing aggregate features
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)
    if 'book_genre_diversity' in df.columns:
        df['book_genre_diversity'] = df['book_genre_diversity'].fillna(1)

    # Fill TF-IDF features with 0
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df


def create_features(
    df: pd.DataFrame,
    book_genres_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    include_aggregates: bool = False,
    include_bert: bool = True,
) -> pd.DataFrame:
    """Enhanced feature engineering pipeline with temporal and affinity features."""
    print("Starting enhanced feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Add interaction feature first
    df = add_interaction_feature(df, train_df)

    # Add temporal features
    df = add_temporal_user_features(df, train_df)
    df = add_book_temporal_features(df, train_df)
    
    # Add affinity features
    df = add_user_book_affinity_features(df, train_df, book_genres_df)

    # Aggregate features (computed separately during training to avoid data leakage)
    if include_aggregates:
        df = add_aggregate_features(df, train_df)
        df = add_advanced_aggregate_features(df, train_df)

    # Existing features
    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    
    if include_bert:
        df = add_bert_features(df, train_df, descriptions_df)
    else:
        print("BERT features disabled (include_bert=False)")
        
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"Enhanced feature engineering complete. Total features: {len(df.columns)}")
    return df