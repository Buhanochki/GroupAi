"""
Enhanced data preparation script with new features.
"""

from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features


def prepare_data() -> None:
    """Enhanced data preparation with temporal and affinity features."""
    
    print("=" * 60)
    print("Enhanced Data Preparation Pipeline")
    print("=" * 60)

    # Load and merge raw data
    merged_df, targets_df, candidates_df, book_genres_df, descriptions_df = load_and_merge_data()

    # Apply enhanced feature engineering
    featured_df = create_features(
        merged_df, 
        book_genres_df, 
        descriptions_df, 
        include_aggregates=False,  # Aggregates computed during training
        include_bert=True          # Enable BERT for better performance
    )

    # Ensure processed directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Define the output path
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    # Save processed data as parquet for efficiency
    print(f"\nSaving processed data to {processed_path}...")
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    print("Processed data saved successfully!")

    # Print enhanced statistics
    train_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    total_features = len(featured_df.columns)
    
    # Count feature types
    numerical_features = len([col for col in featured_df.columns 
                            if featured_df[col].dtype in ['int64', 'float64', 'float32', 'int32']])
    categorical_features = len([col for col in featured_df.columns 
                               if featured_df[col].dtype.name == 'category'])
    text_features = len([col for col in featured_df.columns 
                        if col.startswith('tfidf_') or col.startswith('bert_')])

    print("\nEnhanced data preparation complete!")
    print(f"  - Train rows: {train_rows:,}")
    print(f"  - Total features: {total_features}")
    print(f"    * Numerical: {numerical_features}")
    print(f"    * Categorical: {categorical_features}") 
    print(f"    * Text: {text_features}")
    print(f"  - Output file: {processed_path}")


if __name__ == "__main__":
    prepare_data()
