# src/data_processing/cleaner.py
# Функции для очистки и преобразования загруженных данных.

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def unify_timestamps(df: pd.DataFrame, date_col: str, target_format: str = '%Y-%m') -> pd.DataFrame:
    """
    Унифицирует временные метки в месячный формат (YYYY-MM) и устанавливает его в качестве индекса.

    Args:
        df: Входной DataFrame.
        date_col: Имя столбца, содержащего информацию о дате/времени.
        target_format: Целевой строковый формат для месячного периода.

    Returns:
        DataFrame с PeriodIndex ('YYYY-MM').
    """
    print(f"Унификация временных меток для столбца: {date_col}")
    try:
        # Сначала преобразуйте в объекты datetime для обработки различных форматов ввода
        df[date_col] = pd.to_datetime(df[date_col])
        # Преобразование в месячный период и установка в качестве индекса
        df['Month'] = df[date_col].dt.to_period('M')
        df = df.set_index('Month')
        # При необходимости удалите исходный столбец даты
        # df = df.drop(columns=[date_col])
        print("Временные метки унифицированы и установлены в качестве индекса.")
        return df
    except KeyError:
        print(f"Ошибка: столбец даты '{date_col}' не найден.")
        return df
    except Exception as e:
        print(f"Ошибка унификации временных меток: {e}")
        return df


def normalize_data(series: pd.Series, method: Optional[str] = 'z-score') -> pd.Series:
    """
    Нормализует ряд данных с использованием указанного метода.

    Args:
        series: Входной ряд данных.
        method: Метод нормализации ('z-score', 'log', или None).

    Returns:
        Нормализованный ряд данных.
    """
    print(f"Нормализация ряда с использованием метода: {method}")
    if method == 'z-score':
        # Обработка потенциального нулевого стандартного отклонения
        if series.std() == 0:
            print(
                f"Предупреждение: стандартное отклонение равно нулю для ряда {series.name}. Возвращается исходный ряд.")
            return series
        return pd.Series(stats.zscore(series), index=series.index, name=series.name)
    elif method == 'log':
        # Добавьте небольшую константу для обработки нулевых или отрицательных значений, если это необходимо
        if (series <= 0).any():
            print(
                f"Предупреждение: ряд {series.name} содержит неположительные значения. Добавление 1 перед логарифмическим преобразованием.")
            # Ensure we don't modify the original series if it's used elsewhere
            series_adjusted = series.copy()
            # Замените неположительные значения на 1 для log(1)=0
            series_adjusted[series_adjusted <= 0] = 1
            return np.log(series_adjusted)
            # Альтернатива: добавьте небольшой эпсилон, например, np.log(series + 1e-9)
            # Альтернатива: вернуть NaN для неположительных значений, если это уместно
        else:
            return np.log(series)
    elif method is None:
        print("Нормализация не применена.")
        return series
    else:
        print(
            f"Предупреждение: неизвестный метод нормализации '{method}'. Возвращается исходный ряд.")
        return series


def aggregate_monthly(df: pd.DataFrame, value_col: str, agg_func: str = 'mean') -> pd.Series:
    """
    Агрегирует данные до месячной частоты с использованием указанной функции.
    Предполагается, что df имеет PeriodIndex ('M').

    Args:
        df: Входной DataFrame с месячным PeriodIndex.
        value_col: Имя столбца, содержащего значения для агрегации.
        agg_func: Функция агрегации ('mean', 'sum', 'median' и т.д.).

    Returns:
        Series с месячными агрегированными значениями.
    """
    print(f"Агрегация столбца '{value_col}' помесячно с использованием '{agg_func}'")
    
    # Убедитесь, что индекс является DatetimeIndex для передискретизации
    df_copy = df.copy() # Работайте с копией, чтобы избежать изменения исходного df
    if isinstance(df_copy.index, pd.PeriodIndex):
        print("Преобразование PeriodIndex в DatetimeIndex для агрегации.")
        try:
            df_copy.index = df_copy.index.to_timestamp()
        except Exception as e:
            print(f"Не удалось преобразовать PeriodIndex в DatetimeIndex: {e}")
            return pd.Series(dtype=float)
    elif not isinstance(df_copy.index, pd.DatetimeIndex):
        print("Попытка преобразовать индекс в DatetimeIndex для агрегации.")
        try:
            df_copy.index = pd.to_datetime(df_copy.index)
        except Exception as e:
            print(f"Не удалось преобразовать индекс в DatetimeIndex: {e}")
            return pd.Series(dtype=float)
    # else: индекс уже DatetimeIndex, продолжить

    try:
        # Передискретизация непосредственно с использованием DatetimeIndex
        # Используйте 'ME' для частоты окончания месяца, так как 'M' устарел
        aggregated_series = df_copy[value_col].resample('ME').agg(agg_func)
        print("Агрегация завершена.")
        return aggregated_series
    except KeyError:
        print(f"Ошибка: столбец значения '{value_col}' не найден.")
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"Ошибка во время агрегации: {e}")
        return pd.Series(dtype=float)


if __name__ == '__main__':
    # Пример использования (в целях тестирования)
    print("\nТестирование функций очистки данных...")

    # Создать образец данных
    dates = pd.date_range(start='2020-01-01', periods=60, freq='D')
    data = {'Date': dates, 'Value': np.random.rand(60) * 100}
    sample_df = pd.DataFrame(data)
    print("\nИсходный пример DataFrame:")
    print(sample_df.head())

    # Тест унификации временных меток
    # Используйте копию, чтобы избежать изменения оригинала
    unified_df = unify_timestamps(sample_df.copy(), 'Date')
    print("\nDataFrame после унификации временных меток:")
    print(unified_df.head())
    print(unified_df.index)

    # Тест агрегации
    if not unified_df.empty and 'Value' in unified_df.columns:
        aggregated_series = aggregate_monthly(
            unified_df, 'Value', agg_func='mean')
        print("\nАгрегированный ряд (среднее значение):")
        print(aggregated_series)

        aggregated_sum = aggregate_monthly(unified_df, 'Value', agg_func='sum')
        print("\nАгрегированный ряд (сумма):")
        print(aggregated_sum)
    else:
        print("\nПропуск теста агрегации из-за проблем на предыдущих шагах или отсутствия столбца 'Value'.")

    # Тест нормализации
    if 'aggregated_series' in locals() and not aggregated_series.empty:
        normalized_z = normalize_data(aggregated_series, method='z-score')
        print("\nНормализованный ряд (Z-оценка):")
        print(normalized_z)

        # Тест логарифмической нормализации (обработка потенциальных неположительных значений)
        sample_log_series = pd.Series(
            [10, 20, 0, 40, -5], index=pd.period_range(start='2020-01', periods=5, freq='M'))
        normalized_log = normalize_data(sample_log_series, method='log')
        print("\nНормализованный ряд (Log - с обработкой неположительных значений):")
        print(normalized_log)

        normalized_none = normalize_data(aggregated_series, method=None)
        print("\nНормализованный ряд (None):")
        print(normalized_none)
    else:
        print("\nПропуск теста нормализации из-за проблем с агрегацией.")

    print("\nТест очистки данных завершен.")
