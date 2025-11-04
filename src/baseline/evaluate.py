"""
Тесты для stage2_evaluate.py - оценка метрик для задачи ранжирования (Stage 2).

Покрывает:
1. Корректность расчета NDCG@20 и Recall@20
2. Обработку списков book_id (разделенных пробелами)
3. Дубликаты в списках рекомендаций
4. Списки разной длины (< 20, = 20, > 20)
5. Отсутствующие пользователи в submission
6. Разделение public/private
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Добавляем путь к src для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.stage2_evaluate import (calculate_stage2_metrics, dcg_at_k, ndcg_at_k, recall_at_k,
                                     validate_solution_format, validate_submission_format)

# ============================================================================
# ФИКСТУРЫ
# ============================================================================
# Фикстуры загружаются из tests/conftest.py
# Используют данные из data/test_fixtures/stage2/


# ============================================================================
# ТЕСТ 1: КОРРЕКТНОСТЬ БАЗОВЫХ ФУНКЦИЙ МЕТРИК
# ============================================================================

class TestMetricFunctions:
    """Тесты базовых функций расчета метрик."""

    def test_dcg_at_k_perfect(self):
        """Тест 1.1: DCG для идеального ранжирования."""
        relevance = [1, 1, 1, 1, 1]  # Все релевантные
        dcg = dcg_at_k(relevance, k=5)

        # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(5) + 1/log2(6)
        expected = 1.0 + 0.631 + 0.5 + 0.431 + 0.387
        assert dcg == pytest.approx(expected, abs=0.01)

    def test_dcg_at_k_empty(self):
        """Тест 1.2: DCG для пустого списка."""
        assert dcg_at_k([], k=5) == 0.0

    def test_ndcg_perfect(self):
        """Тест 1.3: NDCG для идеального ранжирования."""
        y_true = {101, 102, 103}
        y_pred = [101, 102, 103, 999, 888]  # Первые 3 - все релевантные

        ndcg = ndcg_at_k(y_true, y_pred, k=20)
        assert ndcg == pytest.approx(1.0, abs=1e-6)

    def test_ndcg_worst(self):
        """Тест 1.4: NDCG для наихудшего ранжирования."""
        y_true = {101, 102, 103}
        y_pred = [999, 888, 777, 666, 555]  # Ни одного релевантного

        ndcg = ndcg_at_k(y_true, y_pred, k=20)
        assert ndcg == pytest.approx(0.0, abs=1e-6)

    def test_ndcg_partial(self):
        """Тест 1.5: NDCG для частичного попадания."""
        y_true = {101, 102, 103}
        y_pred = [101, 999, 102, 888, 777]  # 2 из 3 релевантных

        ndcg = ndcg_at_k(y_true, y_pred, k=20)
        assert 0.0 < ndcg < 1.0  # Должно быть между 0 и 1

    def test_recall_perfect(self):
        """Тест 1.6: Recall для идеального ранжирования."""
        y_true = {101, 102, 103}
        y_pred = [101, 102, 103, 999, 888]

        recall = recall_at_k(y_true, y_pred, k=20)
        assert recall == pytest.approx(1.0, abs=1e-6)

    def test_recall_worst(self):
        """Тест 1.7: Recall для наихудшего ранжирования."""
        y_true = {101, 102, 103}
        y_pred = [999, 888, 777, 666, 555]

        recall = recall_at_k(y_true, y_pred, k=20)
        assert recall == pytest.approx(0.0, abs=1e-6)

    def test_recall_partial(self):
        """Тест 1.8: Recall для частичного попадания."""
        y_true = {101, 102, 103, 104}  # 4 релевантных
        y_pred = [101, 999, 102, 888, 777]  # Нашли 2 из 4

        recall = recall_at_k(y_true, y_pred, k=20)
        assert recall == pytest.approx(0.5, abs=1e-6)  # 2/4 = 0.5

    def test_recall_with_k_limit(self):
        """Тест 1.9: Recall с ограничением по k."""
        y_true = {101, 102, 103}
        y_pred = [999, 888, 101, 102, 103]  # Релевантные на позициях 3, 4, 5

        recall_5 = recall_at_k(y_true, y_pred, k=5)
        recall_2 = recall_at_k(y_true, y_pred, k=2)

        assert recall_5 == pytest.approx(1.0, abs=1e-6)  # Все 3 в топ-5
        assert recall_2 == pytest.approx(0.0, abs=1e-6)  # Ни одного в топ-2


# ============================================================================
# ТЕСТ 2: КОРРЕКТНОСТЬ РАСЧЕТА КОМПЛЕКСНЫХ МЕТРИК
# ============================================================================

class TestMetricsCalculation:
    """Тесты комплексного расчета метрик для submission."""

    def test_perfect_recommendations(self, stage2_valid_solution, stage2_perfect_submission):
        """Тест 2.1: Идеальные рекомендации."""
        metrics = calculate_stage2_metrics(stage2_perfect_submission, stage2_valid_solution)

        # Для идеального случая NDCG и Recall должны быть близки к 1.0
        assert metrics['NDCG@20'] == pytest.approx(1.0, abs=0.01)
        assert metrics['Recall@20'] == pytest.approx(1.0, abs=0.01)
        assert metrics['Score'] == pytest.approx(1.0, abs=0.01)

    def test_worst_recommendations(self, stage2_valid_solution, stage2_worst_submission):
        """Тест 2.2: Наихудшие рекомендации (ни одного попадания)."""
        metrics = calculate_stage2_metrics(stage2_worst_submission, stage2_valid_solution)

        assert metrics['NDCG@20'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['Recall@20'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['Score'] == pytest.approx(0.0, abs=1e-6)

    def test_partial_recommendations(self, stage2_valid_solution, stage2_partial_submission):
        """Тест 2.3: Частичные попадания."""
        metrics = calculate_stage2_metrics(stage2_partial_submission, stage2_valid_solution)

        # Метрики должны быть между 0 и 1
        assert 0.0 < metrics['NDCG@20'] < 1.0
        assert 0.0 < metrics['Recall@20'] < 1.0
        assert 0.0 < metrics['Score'] < 1.0

    def test_empty_submission(self, stage2_valid_solution):
        """Тест 2.4: Пустой submission."""
        empty_submission = pd.DataFrame(columns=['user_id', 'book_id_list'])

        metrics = calculate_stage2_metrics(empty_submission, stage2_valid_solution)

        assert metrics['Score'] == 0.0
        assert metrics['NDCG@20'] == 0.0
        assert metrics['Recall@20'] == 0.0


# ============================================================================
# ТЕСТ 3: ОБРАБОТКА СПИСКОВ BOOK_ID
# ============================================================================

class TestBookIdLists:
    """Тесты обработки списков book_id."""

    def test_parsing_space_separated(self, stage2_valid_solution):
        """Тест 3.1: Парсинг списка, разделенного пробелами."""
        submission = pd.DataFrame({
            'user_id': [1],
            'book_id_list': ['101 102 103 104 105']
        })

        # Проверяем, что парсинг работает корректно (не должно быть ошибок)
        metrics = calculate_stage2_metrics(submission, stage2_valid_solution)
        assert 'NDCG@20' in metrics

    def test_single_recommendation(self, stage2_valid_solution, stage2_single_recommendation_submission):
        """Тест 3.2: Один единственный book_id в списке."""
        metrics = calculate_stage2_metrics(stage2_single_recommendation_submission, stage2_valid_solution)

        # Recall должен быть низким (1 из многих)
        assert 0.0 < metrics['Recall@20'] < 0.5

    def test_list_length_less_than_20(self, stage2_valid_solution, stage2_various_lengths_submission):
        """Тест 3.3: Список длиной < 20."""
        # Не должно быть ошибок
        metrics = calculate_stage2_metrics(stage2_various_lengths_submission, stage2_valid_solution)
        assert 'Score' in metrics

    def test_list_length_exactly_20(self, stage2_valid_solution):
        """Тест 3.4: Список длиной ровно 20."""
        books_20 = ' '.join(str(i) for i in range(100, 120))
        submission = pd.DataFrame({
            'user_id': [1, 2, 3],
            'book_id_list': [books_20, books_20, books_20]
        })

        metrics = calculate_stage2_metrics(submission, stage2_valid_solution)
        assert 'Score' in metrics

    def test_list_length_more_than_20(self, stage2_valid_solution):
        """Тест 3.5: Список длиной > 20 (должны учитываться только первые 20)."""
        # Создаем список из 30 элементов
        books_30 = ' '.join(str(i) for i in range(100, 130))
        submission = pd.DataFrame({
            'user_id': [1, 2, 3],
            'book_id_list': [books_30, books_30, books_30]
        })

        # Метрики должны считаться только по первым 20
        metrics = calculate_stage2_metrics(submission, stage2_valid_solution)
        assert 'Score' in metrics


# ============================================================================
# ТЕСТ 4: ДУБЛИКАТЫ В СПИСКАХ
# ============================================================================

class TestDuplicatesInLists:
    """Тесты обработки дубликатов в списках рекомендаций."""

    def test_duplicates_in_recommendations(self, stage2_valid_solution, stage2_duplicates_submission):
        """Тест 4.1: Дубликаты в списке рекомендаций."""
        # Дубликаты могут повлиять на метрики, но не должны вызывать ошибки
        metrics = calculate_stage2_metrics(stage2_duplicates_submission, stage2_valid_solution)
        assert 'Score' in metrics

        # При дубликатах Recall может быть ниже ожидаемого
        assert 0.0 <= metrics['Recall@20'] <= 1.0

    def test_all_same_book_id(self, stage2_valid_solution, stage2_all_same_book_submission):
        """Тест 4.2: Все рекомендации - один и тот же book_id."""
        metrics = calculate_stage2_metrics(stage2_all_same_book_submission, stage2_valid_solution)

        # Recall и NDCG должны быть очень низкими
        assert metrics['Recall@20'] < 0.5


# ============================================================================
# ТЕСТ 5: ОТСУТСТВУЮЩИЕ ПОЛЬЗОВАТЕЛИ
# ============================================================================

class TestMissingUsers:
    """Тесты обработки отсутствующих пользователей."""

    def test_user_in_solution_not_in_submission(self, stage2_valid_solution):
        """Тест 5.1: Пользователь есть в solution, но нет в submission."""
        # Submission без user_id=3
        incomplete_submission = pd.DataFrame({
            'user_id': [1, 2],
            'book_id_list': ['101 102 103', '201 202']
        })

        # Должен давать нулевой вклад для отсутствующего пользователя
        # но не вызывать ошибку
        metrics = calculate_stage2_metrics(incomplete_submission, stage2_valid_solution)
        assert 'Score' in metrics

        # Метрики считаются только по пользователям из submission
        # Если предсказания для них идеальны, Score может быть 1.0
        assert 0.0 <= metrics['Score'] <= 1.0

    def test_user_in_submission_not_in_solution(self, stage2_valid_solution):
        """Тест 5.2: Пользователь есть в submission, но нет в solution."""
        # Добавляем пользователя, которого нет в solution
        extended_submission = pd.DataFrame({
            'user_id': [1, 2, 3, 999],  # 999 нет в solution
            'book_id_list': [
                '101 102 103',
                '201 202',
                '301 302 303 304',
                '999 998 997'  # Рекомендации для несуществующего пользователя
            ]
        })

        # Лишний пользователь должен игнорироваться или давать 0
        metrics = calculate_stage2_metrics(extended_submission, stage2_valid_solution)
        assert 'Score' in metrics


# ============================================================================
# ТЕСТ 6: ВАЛИДАЦИЯ ФОРМАТА
# ============================================================================

class TestValidation:
    """Тесты валидации формата файлов."""

    def test_valid_solution_format(self, stage2_valid_solution):
        """Тест 6.1: Корректный solution проходит валидацию."""
        validate_solution_format(stage2_valid_solution)  # Не должно быть исключений

    def test_empty_solution(self):
        """Тест 6.2: Пустой solution."""
        empty = pd.DataFrame(columns=['user_id', 'book_id', 'stage'])

        with pytest.raises(ValueError, match="пуст"):
            validate_solution_format(empty)

    def test_missing_columns_solution(self):
        """Тест 6.3: Отсутствие колонок в solution."""
        invalid = pd.DataFrame({
            'user_id': [1, 2],
            'book_id': [101, 102]
            # stage отсутствует!
        })

        with pytest.raises(ValueError, match="отсутствуют необходимые колонки"):
            validate_solution_format(invalid)

    def test_valid_submission_format(self, stage2_valid_solution):
        """Тест 6.4: Корректный submission проходит валидацию."""
        valid_submission = pd.DataFrame({
            'user_id': [1, 2],
            'book_id_list': ['101 102 103', '201 202']
        })

        validate_submission_format(valid_submission, stage2_valid_solution)

    def test_empty_submission(self, stage2_valid_solution):
        """Тест 6.5: Пустой submission."""
        empty = pd.DataFrame(columns=['user_id', 'book_id_list'])

        with pytest.raises(ValueError, match="пуст"):
            validate_submission_format(empty, stage2_valid_solution)

    def test_missing_column_submission(self, stage2_valid_solution):
        """Тест 6.6: Отсутствие колонки book_id_list."""
        invalid = pd.DataFrame({
            'user_id': [1, 2]
            # book_id_list отсутствует!
        })

        with pytest.raises(ValueError, match="отсутствуют необходимые колонки"):
            validate_submission_format(invalid, stage2_valid_solution)

    def test_wrong_user_count(self, stage2_valid_solution):
        """Тест 6.7: Неверное количество уникальных пользователей выдает предупреждение."""
        # В solution 3 уникальных пользователя (1, 2, 3)
        wrong_count = pd.DataFrame({
            'user_id': [1, 2],  # Только 2 пользователя
            'book_id_list': ['101 102', '201 202']
        })

        # Теперь это warning, а не error
        with pytest.warns(UserWarning, match="Количество уникальных пользователей"):
            validate_submission_format(wrong_count, stage2_valid_solution)


# ============================================================================
# ТЕСТ 7: РАЗДЕЛЕНИЕ PUBLIC/PRIVATE
# ============================================================================

class TestPublicPrivateSplit:
    """Тесты корректности разделения на public и private."""

    def test_separate_calculation(self, stage2_valid_solution, stage2_perfect_submission):
        """Тест 7.1: Public и private считаются независимо."""
        solution_public = stage2_valid_solution[stage2_valid_solution['stage'] == 'public']
        public_metrics = calculate_stage2_metrics(stage2_perfect_submission, solution_public)

        solution_private = stage2_valid_solution[stage2_valid_solution['stage'] == 'private']
        private_metrics = calculate_stage2_metrics(stage2_perfect_submission, solution_private)

        # Метрики могут быть ниже, если не все пользователи из solution есть в каждой части
        # Проверяем, что метрики разумные и считаются корректно
        assert 0.0 <= public_metrics['Score'] <= 1.0
        assert 0.0 <= private_metrics['Score'] <= 1.0
        assert 'NDCG@20' in public_metrics
        assert 'NDCG@20' in private_metrics

    def test_different_quality_public_private(self):
        """Тест 7.2: Разное качество для public и private."""
        solution = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2],
            'book_id': [101, 102, 103, 201, 202, 203],
            'stage': ['public', 'public', 'public', 'private', 'private', 'private']
        })

        # Хорошие рекомендации для user=1 (public), плохие для user=2 (private)
        submission = pd.DataFrame({
            'user_id': [1, 2],
            'book_id_list': [
                '101 102 103',  # Идеально для public
                '999 998 997'   # Плохо для private
            ]
        })

        solution_public = solution[solution['stage'] == 'public']
        public_metrics = calculate_stage2_metrics(submission, solution_public)

        solution_private = solution[solution['stage'] == 'private']
        private_metrics = calculate_stage2_metrics(submission, solution_private)

        # Public должен быть лучше private
        assert public_metrics['Score'] > private_metrics['Score']
        # Public должен быть хорошим (идеальные предсказания)
        assert public_metrics['Score'] >= 0.5
        # Private должен быть плохим (нет попаданий)
        assert private_metrics['Score'] <= 0.5


# ============================================================================
# ТЕСТ 8: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Тесты граничных случаев."""

    def test_single_user_single_book(self):
        """Тест 8.1: Один пользователь, одна книга."""
        solution = pd.DataFrame({
            'user_id': [1],
            'book_id': [101],
            'stage': ['public']
        })

        submission = pd.DataFrame({
            'user_id': [1],
            'book_id_list': ['101']
        })

        metrics = calculate_stage2_metrics(submission, solution)

        # Должно быть идеальным
        assert metrics['NDCG@20'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['Recall@20'] == pytest.approx(1.0, abs=1e-6)

    def test_user_with_many_relevant_items(self):
        """Тест 8.2: Пользователь с большим количеством релевантных книг."""
        # 25 релевантных книг для одного пользователя
        solution = pd.DataFrame({
            'user_id': [1] * 25,
            'book_id': list(range(100, 125)),
            'stage': ['public'] * 25
        })

        # Рекомендуем первые 20
        books_20 = ' '.join(str(i) for i in range(100, 120))
        submission = pd.DataFrame({
            'user_id': [1],
            'book_id_list': [books_20]
        })

        metrics = calculate_stage2_metrics(submission, solution)

        # Recall = 20/25 = 0.8
        assert metrics['Recall@20'] == pytest.approx(0.8, abs=0.01)


        # Recall = 20/25 = 0.8
        assert metrics['Recall@20'] == pytest.approx(0.8, abs=0.01)

