# src/analysis/stationarity.py
# Функции для проверки стационарности временных рядов с использованием тестов ADFиd KPSs.

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Tuple, Dict, Optional
import numpy as np


def check_stationarity_adf(series: pd.Series, significance_level: float = 0.05, regression: str = 'c') -> Tuple[bool, float]:
    """
    Выполняет расширенный тест Дики-Фуллера (ADF) на стационарность.

    Нулевая гипотеза (H0): Ряд имеет единичный корень (нестационарный).
    Альтернативная гипотеза (H1): Ряд не имеет единичного корня (стационарный).

    Args:
        series: Данные временного ряда.
        significance_level: Пороговое значение для p-value.
        regression: Тип регрессии ('c', 'ct', 'ctt', 'n').
                    'c' - только константа (по умолчанию)
                    'ct' - константа и тренд
                    'ctt' - константа, линейный и квадратичный тренд
                    'n' - без константы, без тренда

    Returns:
        Кортеж (is_stationary, p_value). is_stationary равно True, если H0 отвергается.
    """
    print(
        f"Выполнение ADF теста для ряда: {series.name} (Регрессия: {regression})")
    try:
        result = adfuller(series.dropna(), regression=regression)
        p_value = result[1]
        is_stationary = p_value < significance_level
        print(f"Результаты ADF теста для {series.name}:")
        print(f"  Статистика теста: {result[0]:.4f}")
        print(f"  P-значение: {p_value:.4f}")
        print(f"  Использовано лагов: {result[2]}")
        print(f"  Стационарность (p < {significance_level}): {is_stationary}")
        return is_stationary, p_value
    except Exception as e:
        print(f"Error during ADF test for {series.name}: {e}")
        return False, 1.0  # Assume non-stationary on error


def check_stationarity_kpss(series: pd.Series, significance_level: float = 0.05, regression: str = 'c') -> Tuple[bool, float]:
    """
    Выполняет тест Квятковского-Филлипса-Шмидта-Шина (KPSS) на стационарность.

    Нулевая гипотеза (H0): Ряд является стационарным относительно тренда (или уровня, если regression='c').
    Альтернативная гипотеза (H1): Ряд имеет единичный корень (нестационарный).

    Args:
        series: Данные временного ряда.
        significance_level: Пороговое значение для p-value.
        regression: Тип регрессии ('c', 'ct').
                    'c' - тест на стационарность уровня (по умолчанию)
                    'ct' - тест на стационарность тренда

    Returns:
        Кортеж (is_stationary, p_value). is_stationary равно True, если H0 НЕ отвергается.
        Примечание: Интерпретация противоположна тесту ADF.
    """
    # Pre-check for constant series (zero variance)
    if series.dropna().var() < 1e-10:  # Use a small threshold for floating point
        print(
            f"KPSS тест пропущен для {series.name}: Дисперсия ряда практически нулевая (постоянное значение). Предполагаем стационарность.")
        return True, 1.0

    print(
        f"Выполнение KPSS теста для ряда: {series.name} (Регрессия: {regression})")
    try:
        result = kpss(series.dropna(), regression=regression,
                      nlags='auto')
        p_value = result[1]
        is_stationary = p_value >= significance_level
        print(f"Результаты KPSS теста для {series.name}:")
        print(f"  Статистика теста: {result[0]:.4f}")
        print(f"  P-значение: {p_value:.4f} (Примечание: p-значения интерполированы и могут быть ограничены 0.01илиr 0.1)")
        print(f"  Использовано лагов: {result[2]}")
        print(f"  Стационарность (p >= {significance_level}): {is_stationary}")
        return is_stationary, p_value
    except Exception as e:
        print(f"Error during KPSS test for {series.name}: {e}")
        return False, 0.0  # Assume non-stationary on error (low p-value)


def apply_differencing(data, order: int = 1):
    """
    Применяет дифференцирование к ряду или датафрейму.
    
    Args:
        data: Данные временного ряда (pd.Series или pd.DataFrame)
        order: Порядок дифференцирования
        
    Returns:
        Дифференцированные данные (того же типа, что и входные)
    """
    if order <= 0:
        return data
    if isinstance(data, pd.Series):
        print(f"Применение дифференцирования порядка {order} к ряду: {data.name}")
        return data.diff(order).dropna()
    else:  # DataFrame
        print(f"Применение дифференцирования порядка {order} к датафрейму с колонками: {list(data.columns)}")
        return data.diff(order).dropna()


def check_stationarity_on_dataframe(df: pd.DataFrame, adf_level: float = 0.05, kpss_level: float = 0.05) -> Dict[str, Dict[str, Tuple[bool, float]]]:
    """Выполняет тесты ADF и KPSS для всех столбцов DataFrame."""
    results = {}
    for col in df.columns:
        print(f"\n--- Проверка стационарности для: {col} ---")
        adf_stat, adf_p = check_stationarity_adf(
            df[col], significance_level=adf_level)
        kpss_stat, kpss_p = check_stationarity_kpss(
            df[col], significance_level=kpss_level)
        results[col] = {
            'ADF': (adf_stat, adf_p),
            'KPSS': (kpss_stat, kpss_p)
        }
    return results


if __name__ == '__main__':
    # Example usage (for testing purposes)
    print("\nТестирование функций проверки стационарности...")

    # Создание тестовых данных
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    # Нестационарные данные (случайное блуждание)
    non_stationary_data = np.random.randn(100).cumsum()
    # Стационарные данные
    stationary_data = np.random.randn(100)

    df_test = pd.DataFrame({
        'Stationary': stationary_data,
        'NonStationary': non_stationary_data
    }, index=idx)

    print("Тестовый датафрейм:")
    print(df_test.head(5))

    # Тест на DataFrame
    stationarity_results = check_stationarity_on_dataframe(df_test)
    print("\nСводка результатов тестов стационарности:")
    print(stationarity_results)

    # Тест дифференцирования
    diff_series = apply_differencing(df_test['NonStationary'], order=1)
    print("\nНестационарный ряд после дифференцирования 1-го порядка:")
    print(diff_series.head(5))

    print("\n--- Проверка стационарности для дифференцированного ряда ---")
    adf_stat_diff, adf_p_diff = check_stationarity_adf(diff_series)
    kpss_stat_diff, kpss_p_diff = check_stationarity_kpss(diff_series)
    print(
        f"ADF для дифференцированного ряда: Стационарность={adf_stat_diff}, p-значение={adf_p_diff:.4f}")
    print(
        f"KPSS для дифференцированного ряда: Стационарность={kpss_stat_diff}, p-значение={kpss_p_diff:.4f}")

    print("\nТестирование стационарности завершено.")
