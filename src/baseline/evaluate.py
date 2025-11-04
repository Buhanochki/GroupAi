import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


def validate_submission_format(df: pd.DataFrame, solution_df: pd.DataFrame):
    """Проверяет основные требования к формату файла решения."""
    if df.empty:
        raise ValueError("Файл с решением пуст.")

    required_cols = {"user_id", "book_id_list"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"В файле решения отсутствуют необходимые колонки. Ожидаются: {required_cols}")

    # Проверяем количество пользователей (предупреждение вместо ошибки)
    expected_users = len(solution_df['user_id'].unique())
    submitted_users = df['user_id'].nunique()
    if submitted_users != expected_users:
        warnings.warn(
            f"Количество уникальных пользователей в решении ({submitted_users}) "
            f"не совпадает с ожидаемым ({expected_users}). "
            f"Отсутствующие пользователи получат нулевой score.",
            UserWarning
        )

def validate_solution_format(df: pd.DataFrame):
    """Проверяет формат файла с эталонными значениями."""
    if df.empty:
        raise ValueError("Файл с эталонными значениями пуст.")

    required_cols = {"user_id", "book_id_list_read", "book_id_list_planned", "stage"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"В файле с эталонными значениями отсутствуют необходимые колонки. Ожидаются: {required_cols}")


def dcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    scores_array = np.asarray(relevance_scores, dtype=float)[:k]
    if scores_array.size:
        return float(np.sum(scores_array / np.log2(np.arange(2, scores_array.size + 2))))
    return 0.0

def ndcg_at_k(relevance_scores: List[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k с многоуровневой релевантностью.

    Args:
        relevance_scores: Список баллов релевантности для каждой позиции (0, 1, или 2)
        k: Количество позиций для оценки

    Returns:
        NDCG@k в диапазоне [0.0, 1.0]
    """
    if len(relevance_scores) == 0:
        return 0.0

    # Берем первые k позиций
    top_k_scores = relevance_scores[:k]

    # Если все релевантности равны 0, возвращаем 0
    if sum(top_k_scores) == 0:
        return 0.0

    # Рассчитываем DCG для предсказанного ранжирования
    calculated_dcg = dcg_at_k(top_k_scores, k=k)

    # Идеальный DCG: сортируем все релевантности по убыванию
    ideal_scores = sorted(top_k_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k=k)

    return calculated_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def recall_at_k(y_true: Set[int], y_pred: List[int], k: int) -> float:
    """Recall at k."""
    # Edge case: если ground truth пустой (пользователь ничего не прочитал)
    if len(y_true) == 0:
        # Если submission тоже пустой → метрика = 1.0 (правильно предсказали отсутствие)
        # Если submission не пустой → метрика = 0.0 (неправильно предложили книги)
        return 1.0 if len(y_pred) == 0 else 0.0

    intersection = len(y_true.intersection(set(y_pred[:k])))
    return intersection / len(y_true)

def average_precision_at_k(y_true: Set[int], y_pred: List[int], k: int) -> float:
    """
    Average Precision at k (MAP@k).

    Эта метрика одновременно оценивает:
    1. Полноту (нашли ли все прочитанные книги)
    2. Качество ранжирования (насколько высоко они расположены)
    3. Точность (наказывает за нерелевантные элементы в начале списка)

    Args:
        y_true: Множество релевантных книг (прочитанные)
        y_pred: Ранжированный список предсказанных книг
        k: Количество позиций для оценки (до 20)

    Returns:
        Average Precision@k в диапазоне [0.0, 1.0]
    """
    # Edge case: если ground truth пустой (пользователь ничего не прочитал)
    if len(y_true) == 0:
        # Если submission тоже пустой → метрика = 1.0 (правильно предсказали отсутствие)
        # Если submission не пустой → метрика = 0.0 (неправильно предложили книги)
        return 1.0 if len(y_pred) == 0 else 0.0

    # Берем первые k позиций
    top_k = y_pred[:k]

    if len(top_k) == 0:
        return 0.0

    # Считаем Precision на каждой позиции, где найдена релевантная книга
    hits = 0  # Количество найденных релевантных книг на текущий момент
    ap_sum = 0.0  # Сумма Precisions для каждой найденной релевантной книги

    for i, item in enumerate(top_k, start=1):
        if item in y_true:
            hits += 1
            # Precision@i = (количество найденных релевантных) / (текущая позиция)
            precision_at_i = hits / i
            ap_sum += precision_at_i

    # Нормализуем на количество всех релевантных книг
    # Это гарантирует, что метрика учитывает полноту: если мы нашли не все книги, метрика будет ниже
    if len(y_true) == 0:
        return 0.0

    return ap_sum / len(y_true)

def calculate_stage2_metrics(submission: pd.DataFrame, solution: pd.DataFrame) -> Dict[str, float]:
    """
    Вычисляет метрику для задачи 2Б: NDCG@20 (Normalized Discounted Cumulative Gain at 20).

    Метрика использует трехуровневую релевантность:
    - 2 балла: книга прочитана (book_id_list_read)
    - 1 балл: книга добавлена в планы (book_id_list_planned)
    - 0 баллов: "холодный" кандидат (не было взаимодействия)

    NDCG наказывает за неправильный порядок: прочитанные книги должны быть выше запланированных,
    а запланированные выше "холодных" кандидатов.
    """
    K = 20

    # Подготовка ground truth: парсим book_id_list_read и book_id_list_planned
    def parse_book_id_list(book_list_str):
        """Парсит строку book_id_list (разделитель - запятая) в множество book_id."""
        if pd.isna(book_list_str) or book_list_str == '':
            return set()
        return {int(x.strip()) for x in str(book_list_str).split(',') if x.strip()}

    # Группируем по user_id и берем первую строку (все строки для одного user_id одинаковые)
    solution_grouped = solution.groupby("user_id").agg({
        "book_id_list_read": lambda x: parse_book_id_list(x.iloc[0]) if len(x) > 0 else set(),
        "book_id_list_planned": lambda x: parse_book_id_list(x.iloc[0]) if len(x) > 0 else set()
    })

    # Объединение с submission
    merged_df = submission.merge(solution_grouped, on="user_id", how="left")

    # Обработка пользователей, которые есть в submission, но нет в solution
    merged_df['book_id_list_read'] = merged_df['book_id_list_read'].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    merged_df['book_id_list_planned'] = merged_df['book_id_list_planned'].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    if merged_df.empty:
        return {"Score": 0.0, "NDCG@20": 0.0}

    # Преобразование строки с book_id_list в список int (разделитель - запятая)
    def parse_prediction_list(book_list_str):
        """Парсит строку book_id_list (разделитель - запятая) в список book_id."""
        if pd.isna(book_list_str) or book_list_str == '':
            return []
        return [int(x.strip()) for x in str(book_list_str).split(',') if x.strip()]

    merged_df["y_pred"] = merged_df["book_id_list"].apply(parse_prediction_list)

    # Расчет relevance scores для каждого пользователя
    def calculate_relevance_scores(row):
        """Рассчитывает список баллов релевантности для предсказанного списка."""
        y_pred = row["y_pred"]
        books_read = row["book_id_list_read"]
        books_planned = row["book_id_list_planned"]

        relevance = []
        for book_id in y_pred:
            if book_id in books_read:
                relevance.append(2)  # Прочитано - высшая ценность
            elif book_id in books_planned:
                relevance.append(1)  # Запланировано - средняя ценность
            else:
                relevance.append(0)  # "Холодный" кандидат - нет ценности

        return relevance

    merged_df["relevance_scores"] = merged_df.apply(calculate_relevance_scores, axis=1)

    # Расчет NDCG@20 для каждого пользователя
    merged_df[f"ndcg@{K}"] = merged_df["relevance_scores"].apply(
        lambda scores: ndcg_at_k(scores, k=K)
    )

    # Усреднение по всем пользователям
    mean_ndcg = merged_df[f"ndcg@{K}"].mean()

    # Итоговый балл равен NDCG@20
    return {"Score": mean_ndcg, f"NDCG@{K}": mean_ndcg}


def main() -> Dict[str, float]:
    """Главная функция для выполнения оценки."""
    parser = argparse.ArgumentParser(description="Evaluate submission file against solution")
    parser.add_argument(
        "--submission",
        type=str,
        default="submission.csv",
        help="Path to submission file (default: submission.csv)",
    )
    parser.add_argument(
        "--solution",
        type=str,
        default="solution.csv",
        help="Path to solution file (default: solution.csv)",
    )
    args = parser.parse_args()

    # Convert to Path objects for better error handling
    submission_path = Path(args.submission)
    solution_path = Path(args.solution)

    if not submission_path.exists():
        print(f"Ошибка: Не найден файл {submission_path}")
        sys.exit(1)

    if not solution_path.exists():
        print(f"Ошибка: Не найден файл {solution_path}")
        sys.exit(1)

    try:
        submission = pd.read_csv(submission_path)
        solution = pd.read_csv(solution_path)
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл {e.filename}")
        sys.exit(1)

    # --- Валидация ---
    validate_solution_format(solution)
    validate_submission_format(submission, solution)

    # --- Разделение на public/private ---
    solution_public = solution[solution["stage"] == "public"].copy()
    solution_private = solution[solution["stage"] == "private"].copy()

    # --- Расчет метрик ---
    public_metrics = calculate_stage2_metrics(submission, solution_public)
    private_metrics = calculate_stage2_metrics(submission, solution_private)

    print("--- Public ---")
    for metric_name, value in public_metrics.items():
        print(f"{metric_name}: {value:.6f}")

    print("\n--- Private ---")
    for metric_name, value in private_metrics.items():
        print(f"{metric_name}: {value:.6f}")

    return {"public_score": public_metrics["Score"], "private_score": private_metrics["Score"]}


if __name__ == "__main__":
    main()
