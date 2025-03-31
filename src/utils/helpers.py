# src/utils/helpers.py
# Общие вспомогательные функции для проекта.

import pandas as pd
import numpy as np
import time
from functools import wraps

def check_data_consistency(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Выполняет базовые проверки согласованности между двумя dataframes,
    обычно перед слиянием.
    """
    print("Выполнение базовых проверок согласованности данных...")
    # Проверка типов индексов
    if type(df1.index) != type(df2.index):
        print(f"Предупреждение: Типы индексов различаются - {type(df1.index)} vs {type(df2.index)}")
    # Проверка частоты индексов, если применимо
    if isinstance(df1.index, (pd.DatetimeIndex, pd.PeriodIndex)) and \
       isinstance(df2.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if getattr(df1.index, 'freq', None) != getattr(df2.index, 'freq', None):
             print(f"Предупреждение: Частоты индексов различаются - {getattr(df1.index, 'freq', None)} vs {getattr(df2.index, 'freq', None)}")
    # Проверка на наличие перекрывающихся имен столбцов (исключая индекс)
    common_cols = df1.columns.intersection(df2.columns)
    if not common_cols.empty:
        print(f"Предупреждение: Найдены общие столбцы: {common_cols.tolist()}. Рекомендуется переименовать перед слиянием.")

    print("Проверки согласованности завершены.")


def timeit(func):
    """Декоратор для измерения времени выполнения функции."""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Функция {func.__name__} выполнялась {total_time:.4f} секунд')
        return result
    return timeit_wrapper


# Пример потенциальной вспомогательной функции для проверки преобразования данных
def ensure_series_positive(series: pd.Series, series_name: str = "Series") -> bool:
    """Проверяет, являются ли все значения в Series положительными."""
    if (series <= 0).any():
        print(f"Предупреждение: {series_name} содержит неположительные значения.")
        return False
    return True


if __name__ == '__main__':
    print("\nТестирование вспомогательных функций...")

    # Тест декоратора timeit
    @timeit
    def sample_function(duration):
        print(f"Запуск образца функции на {duration} секунд...")
        time.sleep(duration)
        print("Образец функции завершен.")
        return "Done"

    sample_function(0.5)

    # Тест проверки согласованности
    idx1 = pd.period_range(start='2020-01', periods=5, freq='M')
    df_a = pd.DataFrame({'ValueA': range(5), 'Common': range(5)}, index=idx1)
    idx2 = pd.date_range(start='2020-01-01', periods=5, freq='MS') # Different index type/freq
    df_b = pd.DataFrame({'ValueB': range(5,10), 'Common': range(5,10)}, index=idx2)

    print("\nТестирование проверки согласованности данных:")
    check_data_consistency(df_a, df_b)

    # Тест ensure_series_positive
    print("\nTesting ensure_series_positive:")
    positive_series = pd.Series([1, 2, 3])
    mixed_series = pd.Series([1, 0, -2])
    print(f"Проверка положительного ряда: {ensure_series_positive(positive_series, 'Positive Series')}")
    print(f"Проверка смешанного ряда: {ensure_series_positive(mixed_series, 'Mixed Series')}")


    print("\nТестирование вспомогательных функций завершено.")