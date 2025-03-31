# src/data_processing/loader.py
# Функции для загрузки необработанных наборов данных.

import pandas as pd
from typing import Tuple

TEMP_COL_MAP = {
    'Год': 'Year',
    'Месяц': 'Month',
    'День': 'Day',
    'Средняя температура воздуха': 'Temperature',
    'Количество осадков': 'Precipitation'
}

MORTALITY_COL_MAP = {
    'StateRegistrationOfDeath': 'Mortality'
}


def load_temperature_data(filepath: str) -> pd.DataFrame:
    """Загружает данные о температуре из текстового файла."""
    print(f"Загрузка данных о температуре из: {filepath}")
    try:
        df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')
        df = df.rename(columns=TEMP_COL_MAP)
        # Объединение Year, Month, Day в объект datetime
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        # Выбор релевантных столбцов
        df = df[['Date', 'Temperature', 'Precipitation']]
        print("Данные о температуре успешно загружены.")
        return df
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по адресу {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ошибка загрузки данных о температуре: {e}")
        return pd.DataFrame()


def load_secondary_data(filepath: str) -> pd.DataFrame:
    """Загружает вторичный набор данных (ДТП или Смертность) из CSV файла."""
    print(f"Загрузка вторичных данных из: {filepath}")
    try:
        if 'mortality' in filepath:
            df = pd.read_csv(filepath, delimiter=',', encoding='utf-8')
            # Объединение Year и Month name в объект datetime (начало месяца)
            # Преобразование названий месяцев, если они не числовые (например, 'January')
            # Предполагается, что названия месяцев на английском языке
            try:
                df['Date'] = pd.to_datetime(df['Year'].astype(
                    str) + '-' + df['Month'], format='%Y-%B')
            except ValueError:
                # Запасной вариант, если месяц числовой или в другом формате
                df['Date'] = pd.to_datetime(df['Year'].astype(
                    str) + '-' + df['Month'].astype(str))

            df = df.rename(columns=MORTALITY_COL_MAP)
            # Выбор релевантных столбцов
            df = df[['Date', 'Mortality']]  # Add other columns if needed
            print("Данные о смертности успешно загружены.")
            return df
        elif 'dtp' in filepath:
            # Укажите кодировку И индекс строки заголовка
            # Использовать parse_dates непосредственно в read_csv
            date_col_name = 'Дата(месяц,год)'
            try:
                df = pd.read_csv(
                    filepath,
                    delimiter=';',
                    encoding='utf-8',
                    header=0,
                    parse_dates=[date_col_name],  # Укажите столбец для анализа
                    date_format='%m.%Y'          # Укажите формат для парсера
                )
                # Переименование проанализированного столбца даты в 'Date' для согласованности
                df = df.rename(columns={date_col_name: 'Date'})
            except ValueError as e:
                # Если прямая обработка не удалась, вернуться к ручной обработке (предыдущая попытка)
                print(
                    f"Прямой анализ даты не удался ({e}), попытка ручного анализа...")
                df = pd.read_csv(filepath, delimiter=';',
                                 encoding='utf-8', header=0)
                df['Date'] = pd.to_datetime(df[date_col_name], format='%m.%Y')

            # Разобрать 'Дата(месяц,год)', который находится в формате MM.YYYY
            # df['Date'] = pd.to_datetime(df['Дата(месяц,год)'], format='%m.%Y') # Теперь обрабатывается parse_dates
            # Переименование столбцов, если необходимо (предполагая, что 'ДТП' является целью)
            df = df.rename(
                columns={'ДТП': 'DTP', 'Погибло': 'Deaths', 'Ранено': 'Injured'})
            # Выбор релевантных столбцов
            df = df[['Date', 'DTP', 'Deaths', 'Injured']]
            print("Данные ДТП успешно загружены.")
            return df
        else:
            print(
                f"Ошибка: Неизвестный тип файла вторичных данных для пути: {filepath}")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по адресу {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ошибка загрузки вторичных данных: {e}")
        return pd.DataFrame()


def load_all_data(temp_path: str, secondary_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает как данные о температуре, так и вторичные наборы данных."""
    df_temp = load_temperature_data(temp_path)
    df_secondary = load_secondary_data(secondary_path)
    return df_temp, df_secondary


if __name__ == '__main__':
    # Пример использования (в целях тестирования)
    import sys
    sys.path.append('..')  # Add parent directory to path to import config
    import config

    # Использовать абсолютные пути для тестирования, если скрипт запускается непосредственно из data_processing
    # base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # temp_path_test = os.path.join(base_path, config.TEMP_DATA_PATH)
    # secondary_path_test = os.path.join(base_path, config.SECONDARY_DATA_PATH)

    print("Тестирование функций загрузки данных...")
    # Использовать пути непосредственно из config, предполагая, что скрипт/блокнот запускается из корня проекта
    df_temp, df_secondary = load_all_data(
        config.TEMP_DATA_PATH, config.SECONDARY_DATA_PATH)

    if not df_temp.empty:
        print("\nЗаголовок данных о температуре:")
        print(df_temp.head())
        print("\nИнформация о данных о температуре:")
        df_temp.info()

    if not df_secondary.empty:
        print("\nЗаголовок вторичных данных:")
        print(df_secondary.head())
        print("\nИнформация о вторичных данных:")
        df_secondary.info()

    print("\nТест загрузки данных завершен.")
