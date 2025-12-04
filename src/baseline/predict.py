"""
Enhanced prediction script with hierarchical ranking and ensemble support.
"""

import json
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, constants
from .data_processing import expand_candidates, load_and_merge_data
from .features import add_aggregate_features, add_advanced_aggregate_features, handle_missing_values


class HierarchicalRanker:
    """Advanced ranking strategies for NDCG@20 optimization."""
    
    def __init__(self, strategy: str = "hierarchical"):
        self.strategy = strategy
        
    def calculate_ranking_scores(
        self, 
        probabilities: np.ndarray, 
        user_ids: np.ndarray = None,
        book_ids: np.ndarray = None
    ) -> np.ndarray:
        """Calculate ranking scores using hierarchical strategy."""
        
        if self.strategy == "hierarchical":
            return self._hierarchical_ranking(probabilities)
        elif self.strategy == "two_stage":
            return self._two_stage_ranking(probabilities)
        elif self.strategy == "direct":
            return self._direct_ranking(probabilities)
        else:
            return self._hierarchical_ranking(probabilities)
    
    def _hierarchical_ranking(self, probabilities: np.ndarray) -> np.ndarray:
        """Hierarchical ranking with guaranteed class separation."""
        # probabilities shape: (n_samples, 3) [p_cold, p_planned, p_read]
        
        n_samples = probabilities.shape[0]
        ranking_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            p_cold, p_planned, p_read = probabilities[i]
            
            # Base score from weighted probabilities
            base_score = p_planned * 1.0 + p_read * 2.0
            
            # Add hierarchical margins to ensure proper ordering
            if p_read > 0.3:  # High confidence read
                final_score = base_score + 3.0 + np.random.uniform(0, 0.01)
            elif p_read > 0.1:  # Medium confidence read  
                final_score = base_score + 2.5 + np.random.uniform(0, 0.01)
            elif p_planned > 0.3:  # High confidence planned
                final_score = base_score + 1.5 + np.random.uniform(0, 0.01)
            elif p_planned > 0.1:  # Medium confidence planned
                final_score = base_score + 1.0 + np.random.uniform(0, 0.01)
            else:  # Cold candidate
                final_score = base_score + np.random.uniform(0, 0.001)
                
            ranking_scores[i] = final_score
            
        return ranking_scores
    
    def _two_stage_ranking(self, probabilities: np.ndarray) -> np.ndarray:
        """Two-stage ranking: classify then rank within classes."""
        n_samples = probabilities.shape[0]
        ranking_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            p_cold, p_planned, p_read = probabilities[i]
            
            # Stage 1: Determine primary class
            predicted_class = np.argmax([p_cold, p_planned, p_read])
            
            # Stage 2: Rank within class with class probability as tie-breaker
            if predicted_class == 2:  # Read
                ranking_scores[i] = 4.0 + p_read * 2.0 + np.random.uniform(0, 0.01)
            elif predicted_class == 1:  # Planned
                ranking_scores[i] = 2.0 + p_planned * 2.0 + np.random.uniform(0, 0.01)
            else:  # Cold
                ranking_scores[i] = p_cold * 1.0 + np.random.uniform(0, 0.001)
                
        return ranking_scores
    
    def _direct_ranking(self, probabilities: np.ndarray) -> np.ndarray:
        """Direct ranking using weighted probabilities."""
        # Simple weighted sum: p_planned * 1 + p_read * 2
        return probabilities[:, 1] * 1.0 + probabilities[:, 2] * 2.0


class EnsemblePredictor:
    """Ensemble prediction with model weighting."""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.models = {}
        self.weights = {}
        
    def load_ensemble(self) -> None:
        """Load ensemble models and their weights."""
        ensemble_info_path = self.model_dir / "ensemble_info.json"
        if not ensemble_info_path.exists():
            raise FileNotFoundError("Ensemble info not found. Please train models first.")
            
        with open(ensemble_info_path, 'r') as f:
            ensemble_info = json.load(f)
            
        # Load all ensemble models
        for model_name in ensemble_info['models']:
            model_path = self.model_dir / f"ensemble_{model_name}.txt"
            if model_path.exists():
                self.models[model_name] = lgb.Booster(model_file=str(model_path))
                # Simple equal weighting for now
                self.weights[model_name] = 1.0
                
        if not self.models:
            raise ValueError("No ensemble models found!")
            
        print(f"Loaded {len(self.models)} ensemble models")
        
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions with feature validation."""
        
        # Проверяем количество признаков
        expected_features = self.models[list(self.models.keys())[0]].num_feature()
        actual_features = X.shape[1]
        
        if expected_features != actual_features:
            print(f"WARNING: Feature count mismatch. Expected: {expected_features}, Got: {actual_features}")
            print(f"Attempting to fix by using only first {expected_features} features...")  # Исправлено: добавлен f
            # Берем только первые expected_features признаков
            X = X.iloc[:, :expected_features]
        
        all_predictions = []
        
        for model_name, model in self.models.items():
            try:
                # LightGBM booster returns raw scores, need to convert to probabilities
                raw_scores = model.predict(X, predict_disable_shape_check=True)
                raw_scores = np.array(raw_scores)
                
                # Ensure correct shape: (n_samples, 3)
                if raw_scores.ndim == 1:
                    raw_scores = raw_scores.reshape(-1, 3)
                    
                # Convert to probabilities using softmax
                exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                
                all_predictions.append(probabilities)
                
            except Exception as e:
                print(f"Error in model {model_name}: {e}")
                continue
                
        if not all_predictions:
            raise ValueError("All ensemble models failed!")
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(all_predictions[0])
        total_weight = sum(self.weights.values())
        
        for i, (model_name, weight) in enumerate(self.weights.items()):
            if i < len(all_predictions):
                ensemble_proba += all_predictions[i] * (weight / total_weight)
                
        return ensemble_proba


def prepare_candidate_features(
    candidates_pairs_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    train_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for candidate prediction with exact feature matching."""
    
    print("Loading metadata...")
    _, _, _, book_genres_df, descriptions_df = load_and_merge_data()
    user_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.USER_DATA_FILENAME)
    book_data_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME)

    # Merge metadata with candidates
    print("Merging metadata with candidates...")
    candidates_with_meta = candidates_pairs_df.merge(user_data_df, on=constants.COL_USER_ID, how="left")
    book_data_df = book_data_df.drop_duplicates(subset=[constants.COL_BOOK_ID])
    candidates_with_meta = candidates_with_meta.merge(book_data_df, on=constants.COL_BOOK_ID, how="left")

    # Add base features from prepared data - ТОЛЬКО те, что были в обучении
    print("Adding base features...")
    
    # Загружаем список признаков из обучения
    features_path = config.MODEL_DIR / "features_list.json"
    if not features_path.exists():
        raise FileNotFoundError("Feature list not found. Please train models first.")
        
    with open(features_path, "r") as f:
        training_features = json.load(f)
    
    print(f"Training used {len(training_features)} features")
    
    # Получаем ВСЕ возможные признаки из featured_df (кроме служебных колонок)
    exclude_cols = [
        constants.COL_USER_ID,
        constants.COL_BOOK_ID,
        constants.COL_SOURCE,
        constants.COL_TIMESTAMP,
        constants.COL_HAS_READ,
        constants.COL_TARGET,
        constants.COL_PREDICTION,
    ]
    
    all_possible_features = [
        col for col in featured_df.columns 
        if col not in exclude_cols and col in training_features
    ]
    
    print(f"Using {len(all_possible_features)} features from prepared data")

    # Получаем уникальные признаки для книг
    book_features_df = featured_df[[constants.COL_BOOK_ID] + all_possible_features].drop_duplicates(
        subset=[constants.COL_BOOK_ID]
    )

    # Удаляем дублирующиеся колонки перед мерджем
    cols_to_drop = [col for col in all_possible_features if col in candidates_with_meta.columns]
    if cols_to_drop:
        candidates_with_meta = candidates_with_meta.drop(columns=cols_to_drop)

    # Мерджим ТОЛЬКО нужные признаки
    candidates_with_meta = candidates_with_meta.merge(
        book_features_df, on=constants.COL_BOOK_ID, how="left"
    )

    # Compute aggregate features on ALL train data
    print("Computing aggregate features...")
    candidates_with_agg = add_aggregate_features(candidates_with_meta.copy(), train_df)
    candidates_with_agg = add_advanced_aggregate_features(candidates_with_agg, train_df)

    # Handle missing values
    print("Handling missing values...")
    candidates_final = handle_missing_values(candidates_with_agg, train_df)

    # Убеждаемся, что у нас ТОЧНО те же признаки, что и при обучении
    print("Ensuring exact feature match with training...")
    
    # Проверяем, какие признаки отсутствуют
    missing_features = [f for f in training_features if f not in candidates_final.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features missing, adding defaults")
        for feat in missing_features:
            if feat in train_df.columns:
                if train_df[feat].dtype.name == "category":
                    default_val = train_df[feat].cat.categories[0] if len(train_df[feat].cat.categories) > 0 else 0
                    candidates_final[feat] = pd.Categorical(
                        [default_val] * len(candidates_final), 
                        categories=train_df[feat].cat.categories, 
                        ordered=False
                    )
                else:
                    candidates_final[feat] = train_df[feat].iloc[0] if len(train_df) > 0 else 0
            else:
                candidates_final[feat] = 0

    # Удаляем лишние признаки, которых не было в обучении
    extra_features = [col for col in candidates_final.columns if col not in training_features and col not in [constants.COL_USER_ID, constants.COL_BOOK_ID]]
    if extra_features:
        print(f"Removing {len(extra_features)} extra features not used in training")
        candidates_final = candidates_final.drop(columns=extra_features)

    # Убеждаемся, что порядок признаков совпадает
    final_features = [f for f in training_features if f in candidates_final.columns]
    
    print(f"Final feature count: {len(final_features)} (must match training: {len(training_features)})")
    
    if len(final_features) != len(training_features):
        print(f"WARNING: Feature count mismatch! Training: {len(training_features)}, Prediction: {len(final_features)}")
        # Используем только общие признаки
        common_features = list(set(training_features) & set(candidates_final.columns))
        print(f"Using {len(common_features)} common features")
        final_features = common_features

    # Обработка категориальных признаков
    for col in final_features:
        if col in featured_df.columns and featured_df[col].dtype.name == "category":
            train_categories = list(featured_df[col].cat.categories)
            
            # Конвертируем в строку
            candidates_final[col] = candidates_final[col].astype(str)
            
            # Заменяем значения не из тренировочных категорий
            valid_mask = candidates_final[col].isin([str(cat) for cat in train_categories])
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                print(f"Warning: {invalid_count} values in {col} not in training categories")
                candidates_final.loc[~valid_mask, col] = str(train_categories[0]) if train_categories else "0"
            
            # Конвертируем обратно в категориальный с тренировочными категориями
            candidates_final[col] = pd.Categorical(
                candidates_final[col], 
                categories=[str(cat) for cat in train_categories],
                ordered=False
            )

    return candidates_final[final_features + [constants.COL_USER_ID, constants.COL_BOOK_ID]], final_features

def generate_ranked_predictions(
    candidates_final: pd.DataFrame,
    features: List[str],
    ranking_strategy: str = "hierarchical"
) -> pd.DataFrame:
    """Generate ranked predictions using ensemble and hierarchical ranking."""
    
    # Сбрасываем индекс чтобы избежать проблем
    candidates_final = candidates_final.reset_index(drop=True)
    X_test = candidates_final[features].copy()
    
    # Проверяем количество признаков
    print(f"Prediction data shape: {X_test.shape}")
    
    # Try ensemble first, fall back to single model
    ensemble_predictor = EnsemblePredictor(config.MODEL_DIR)
    
    try:
        ensemble_predictor.load_ensemble()
        print("Using ensemble prediction...")
        test_proba_all = ensemble_predictor.predict_ensemble(X_test)
    except Exception as e:
        print(f"Ensemble failed: {e}. Using single model...")
        # Fall back to single model
        model_path = config.MODEL_DIR / config.MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError("No trained model found!")
            
        model = lgb.Booster(model_file=str(model_path))
        
        # Проверяем количество признаков для single model
        expected_features = model.num_feature()
        actual_features = X_test.shape[1]
        
        if expected_features != actual_features:
            print(f"Single model feature mismatch. Expected: {expected_features}, Got: {actual_features}")
            # Используем только нужное количество признаков
            if actual_features > expected_features:
                X_test = X_test.iloc[:, :expected_features]
            else:
                # Если признаков меньше, добавляем нули
                missing = expected_features - actual_features
                print(f"Adding {missing} zero features")
                for i in range(missing):
                    X_test[f'missing_feature_{i}'] = 0
        
        raw_scores = model.predict(X_test, predict_disable_shape_check=True)
        raw_scores = np.array(raw_scores)
        
        if raw_scores.ndim == 1:
            raw_scores = raw_scores.reshape(-1, 3)
            
        # Convert to probabilities
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        test_proba_all = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Use hierarchical ranking
    ranker = HierarchicalRanker(strategy=ranking_strategy)
    ranking_scores = ranker.calculate_ranking_scores(test_proba_all)
    
    # Add predictions to candidates dataframe
    candidates_final = candidates_final.copy()
    candidates_final["prediction_score"] = ranking_scores
    
    # Also store class probabilities for analysis
    candidates_final["p_cold"] = test_proba_all[:, 0]
    candidates_final["p_planned"] = test_proba_all[:, 1] 
    candidates_final["p_read"] = test_proba_all[:, 2]
    
    return candidates_final


def create_submission(
    candidates_final: pd.DataFrame, 
    targets_df: pd.DataFrame
) -> pd.DataFrame:
    """Create submission file with ranked predictions."""
    
    print("Creating ranked submissions...")
    
    # Создаем полностью чистую копию данных
    candidates_clean = candidates_final.copy()
    
    # Восстанавливаем нормальную структуру данных
    print("Fixing data structure...")
    
    # Способ 1: Если данные стали многомерными, преобразуем их обратно в одномерные
    if hasattr(candidates_clean[constants.COL_USER_ID], 'ndim') and candidates_clean[constants.COL_USER_ID].ndim > 1:
        print("Converting multidimensional user_id to 1D...")
        # Если это DataFrame, берем первый столбец
        if isinstance(candidates_clean[constants.COL_USER_ID], pd.DataFrame):
            candidates_clean[constants.COL_USER_ID] = candidates_clean[constants.COL_USER_ID].iloc[:, 0]
        # Если это массив с неправильной формой, reshape
        elif hasattr(candidates_clean[constants.COL_USER_ID], 'values'):
            candidates_clean[constants.COL_USER_ID] = pd.Series(candidates_clean[constants.COL_USER_ID].values.ravel())
    
    if hasattr(candidates_clean[constants.COL_BOOK_ID], 'ndim') and candidates_clean[constants.COL_BOOK_ID].ndim > 1:
        print("Converting multidimensional book_id to 1D...")
        if isinstance(candidates_clean[constants.COL_BOOK_ID], pd.DataFrame):
            candidates_clean[constants.COL_BOOK_ID] = candidates_clean[constants.COL_BOOK_ID].iloc[:, 0]
        elif hasattr(candidates_clean[constants.COL_BOOK_ID], 'values'):
            candidates_clean[constants.COL_BOOK_ID] = pd.Series(candidates_clean[constants.COL_BOOK_ID].values.ravel())
    
    # Способ 2: Создаем полностью новый DataFrame с правильной структурой
    try:
        # Пробуем извлечь данные разными способами
        if isinstance(candidates_clean[constants.COL_USER_ID], pd.Series):
            user_ids = candidates_clean[constants.COL_USER_ID]
        else:
            user_ids = pd.Series(candidates_clean[constants.COL_USER_ID].values.ravel())
            
        if isinstance(candidates_clean[constants.COL_BOOK_ID], pd.Series):
            book_ids = candidates_clean[constants.COL_BOOK_ID]
        else:
            book_ids = pd.Series(candidates_clean[constants.COL_BOOK_ID].values.ravel())
            
        prediction_scores = candidates_clean["prediction_score"]
        
        # Создаем чистый DataFrame
        clean_data = pd.DataFrame({
            constants.COL_USER_ID: user_ids,
            constants.COL_BOOK_ID: book_ids,
            "prediction_score": prediction_scores
        })
        
    except Exception as e:
        print(f"Error creating clean data: {e}")
        # Резервный способ: используем только нужные колонки
        clean_data = candidates_clean[[constants.COL_USER_ID, constants.COL_BOOK_ID, "prediction_score"]].copy()
        clean_data = clean_data.reset_index(drop=True)
    
    # Убеждаемся что типы данных правильные
    clean_data[constants.COL_USER_ID] = pd.to_numeric(clean_data[constants.COL_USER_ID], errors='coerce').fillna(0).astype('int32')
    clean_data[constants.COL_BOOK_ID] = pd.to_numeric(clean_data[constants.COL_BOOK_ID], errors='coerce').fillna(0).astype('int32')
    clean_data["prediction_score"] = pd.to_numeric(clean_data["prediction_score"], errors='coerce').fillna(0)
    
    # Удаляем строки с некорректными ID
    initial_count = len(clean_data)
    clean_data = clean_data[
        (clean_data[constants.COL_USER_ID] > 0) & 
        (clean_data[constants.COL_BOOK_ID] > 0)
    ].copy()
    
    if len(clean_data) < initial_count:
        print(f"Removed {initial_count - len(clean_data)} rows with invalid user_id or book_id")
    
    # Удаляем дубликаты
    duplicates = clean_data.duplicated(subset=[constants.COL_USER_ID, constants.COL_BOOK_ID])
    if duplicates.any():
        print(f"Removing {duplicates.sum()} duplicate (user_id, book_id) pairs...")
        clean_data = clean_data[~duplicates]
    
    print(f"Clean data ready: {len(clean_data)} rows")
    
    submission_rows = []

    # Простая и надежная обработка каждого пользователя
    for user_id in targets_df[constants.COL_USER_ID].unique():
        try:
            # Простая фильтрация
            user_books = clean_data[clean_data[constants.COL_USER_ID] == user_id]
            
            if len(user_books) == 0:
                book_id_list = ""
                if len(submission_rows) < 5:  # Логируем только первых несколько
                    print(f"User {user_id}: no candidates found")
            else:
                # Сортируем и выбираем топ
                user_books_sorted = user_books.sort_values("prediction_score", ascending=False)
                k = min(constants.MAX_RANKING_LENGTH, len(user_books_sorted))
                top_books = user_books_sorted.head(k)
                
                # Получаем book_id
                book_ids = top_books[constants.COL_BOOK_ID].astype(int).tolist()
                book_id_list = ",".join(map(str, book_ids))
                
                if len(submission_rows) < 5:  # Логируем только первых несколько
                    print(f"User {user_id}: {len(book_ids)} recommendations")

            submission_rows.append({
                constants.COL_USER_ID: user_id, 
                constants.COL_BOOK_ID_LIST: book_id_list
            })
            
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            submission_rows.append({
                constants.COL_USER_ID: user_id,
                constants.COL_BOOK_ID_LIST: ""
            })

    submission_df = pd.DataFrame(submission_rows)
    
    # Проверяем покрытие
    missing_users = set(targets_df[constants.COL_USER_ID]) - set(submission_df[constants.COL_USER_ID])
    if missing_users:
        print(f"Adding {len(missing_users)} missing users with empty recommendations")
        for user_id in missing_users:
            submission_df = pd.concat([
                submission_df,
                pd.DataFrame([{
                    constants.COL_USER_ID: user_id,
                    constants.COL_BOOK_ID_LIST: ""
                }])
            ], ignore_index=True)
    
    print(f"✅ Submission created: {len(submission_df)} users")
    return submission_df


def predict() -> None:
    """Enhanced prediction pipeline with ensemble and hierarchical ranking."""
    
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

    # Проверяем на дубликаты user_id в targets
    duplicate_targets = targets_df.duplicated(subset=[constants.COL_USER_ID])
    if duplicate_targets.any():
        print(f"Warning: Found {duplicate_targets.sum()} duplicate user_ids in targets, removing them...")
        targets_df = targets_df.drop_duplicates(subset=[constants.COL_USER_ID])

    # Проверяем на дубликаты user_id в candidates
    duplicate_candidates = candidates_df.duplicated(subset=[constants.COL_USER_ID])
    if duplicate_candidates.any():
        print(f"Warning: Found {duplicate_candidates.sum()} duplicate user_ids in candidates, removing them...")
        candidates_df = candidates_df.drop_duplicates(subset=[constants.COL_USER_ID])

    # Остальной код без изменений...
    # Expand candidates into pairs
    print("\nExpanding candidates...")
    candidates_pairs_df = expand_candidates(candidates_df)

    # ... остальной код функции predict
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

    # Prepare candidate features
    candidates_final, features = prepare_candidate_features(
        candidates_pairs_df, featured_df, train_df
    )

    print(f"Final prediction features: {len(features)}")
    print(f"Candidates prepared: {len(candidates_final):,} rows")

    # Generate predictions with hierarchical ranking
    candidates_with_predictions = generate_ranked_predictions(
        candidates_final, 
        features,
        ranking_strategy=config.RANKING_STRATEGY
    )

    # Create submission
    submission_df = create_submission(candidates_with_predictions, targets_df)

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    # Save submission
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file created at: {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

    # Print detailed statistics
    non_empty = submission_df[submission_df[constants.COL_BOOK_ID_LIST] != ""].shape[0]
    avg_recommendations = submission_df[constants.COL_BOOK_ID_LIST].apply(
        lambda x: len(x.split(',')) if x else 0
    ).mean()
    
    print(f"Users with recommendations: {non_empty}/{len(submission_df)}")
    print(f"Average recommendations per user: {avg_recommendations:.2f}")
    
    # Print prediction distribution
    pred_stats = candidates_with_predictions.agg({
        'p_cold': ['mean', 'std'],
        'p_planned': ['mean', 'std'], 
        'p_read': ['mean', 'std'],
        'prediction_score': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\nPrediction Statistics:")
    print(pred_stats)


if __name__ == "__main__":
    predict()