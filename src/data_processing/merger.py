# src/data_processing/merger.py
# Функции для объединения предварительно обработанных данных временных рядов.

import pandas as pd
from typing import Tuple
import numpy as np


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, how: str = 'inner') -> pd.DataFrame:
    """
    Объединяет два DataFrame на основе их индексов (предполагается, что они основаны на времени).

    Args:
        df1: Первый DataFrame.
        df2: Второй DataFrame.
        how: Тип объединения для выполнения ('inner', 'outer', 'left', 'right').
             'inner' рекомендуется для сохранения только перекрывающихся временных периодов.

    Returns:
        Объединенный DataFrame.
    """
    print(f"Объединение датафреймов с использованием метода: {how}")
    if not isinstance(df1.index, (pd.PeriodIndex, pd.DatetimeIndex)) or \
       not isinstance(df2.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        print("Предупреждение: Один или оба DataFrame не имеют временного индекса. Объединение может быть некорректным.")

    try:
        # Убедитесь, что индексы совместимы (например, оба PeriodIndex или оба DatetimeIndex)
        # Если один PeriodIndex, а другой DatetimeIndex, преобразуйте один для объединения
        if isinstance(df1.index, pd.PeriodIndex) and isinstance(df2.index, pd.DatetimeIndex):
            df2.index = df2.index.to_period(df1.index.freq)
        elif isinstance(df2.index, pd.PeriodIndex) and isinstance(df1.index, pd.DatetimeIndex):
            df1.index = df1.index.to_period(df2.index.freq)

        merged_df = pd.merge(df1, df2, left_index=True,
                             right_index=True, how=how)

        # Проверка на дублирующиеся имена столбцов после объединения (если у dfs изначально были одинаковые имена столбцов)
        if merged_df.columns.duplicated().any():
            print("Предупреждение: Объединенный DataFrame содержит дублирующиеся имена столбцов. Рассмотрите возможность переименования столбцов перед объединением.")

        print("DataFrame успешно объединены.")
        return merged_df
    except Exception as e:
        print(f"Ошибка объединения DataFrame: {e}")
        return pd.DataFrame()  # Возвращает пустой DataFrame в случае ошибки


def check_completeness(df: pd.DataFrame) -> None:
    """
    Проверяет объединенный DataFrame на наличие пропущенных значений и временных пробелов.
    """
    print("Проверка полноты объединенных данных...")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Обнаружены пропущенные значения:")
        print(missing_values[missing_values > 0])
    else:
        print("Пропущенные значения не обнаружены.")

    if isinstance(df.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        # Проверка на пропуски во временном индексе
        expected_index = pd.period_range(start=df.index.min(), end=df.index.max(), freq=df.index.freq) if isinstance(
            df.index, pd.PeriodIndex) else pd.date_range(start=df.index.min(), end=df.index.max(), freq=df.index.freq)
        if len(df.index) != len(expected_index):
            print(
                f"Предупреждение: Обнаружены пропуски во временном ряду. Ожидается {len(expected_index)} периодов, найдено {len(df.index)}.")
            # При необходимости определите отсутствующие периоды/даты
            missing_periods = expected_index.difference(df.index)
            print(f"Пропущенные периоды/даты: {missing_periods}")
        else:
            print("Индекс временного ряда непрерывный (нет пропусков).")
    else:
        print("Индекс не основан на времени, пропуск проверки на пропуски.")

if __name__ == '__main__':
    # Пример использования (в целях тестирования)
    print("\nТестирование функций объединения данных...")

    # Создание примера предварительно обработанных данных
    idx1 = pd.period_range(start='2020-01', periods=12, freq='M')
    df_temp_processed = pd.DataFrame(
        {'Temperature_Norm': np.random.randn(12)}, index=idx1)

    # Перекрывающиеся, но с разными началом/концом
    idx2 = pd.period_range(start='2020-03', periods=12, freq='M')
    df_secondary_processed = pd.DataFrame(
        {'Secondary_Norm': np.random.rand(12)}, index=idx2)

    print("\nПример обработанных данных о температуре:")
    print(df_temp_processed)
    print("\nПример обработанных вторичных данных:")
    print(df_secondary_processed)

    # Тест внутреннего объединения
    merged_inner = merge_dataframes(
        df_temp_processed, df_secondary_processed, how='inner')
    print("\nОбъединенный DataFrame (Inner):")
    print(merged_inner)
    if not merged_inner.empty:
        check_completeness(merged_inner)

    # Тест внешнего объединения
    merged_outer = merge_dataframes(
        df_temp_processed, df_secondary_processed, how='outer')
    print("\nОбъединенный DataFrame (Outer):")
    print(merged_outer)
    if not merged_outer.empty:
        check_completeness(merged_outer)  # Ожидаем здесь пропущенные значения

    print("\nТест объединения данных завершен.")
