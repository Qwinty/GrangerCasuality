# src/data_processing/loader.py
# Functions for loading the raw datasets.

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
    """Loads the temperature data from a text file."""
    print(f"Loading temperature data from: {filepath}")
    try:
        df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')
        df = df.rename(columns=TEMP_COL_MAP)
        # Combine Year, Month, Day into a datetime object
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        # Select relevant columns
        df = df[['Date', 'Temperature', 'Precipitation']]
        print("Temperature data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading temperature data: {e}")
        return pd.DataFrame()


def load_secondary_data(filepath: str) -> pd.DataFrame:
    """Loads the secondary dataset (DTP or Mortality) from a CSV file."""
    print(f"Loading secondary data from: {filepath}")
    try:
        if 'mortality' in filepath:
            df = pd.read_csv(filepath, delimiter=',', encoding='utf-8')
            # Combine Year and Month name into a datetime object (start of month)
            # Convert month names if they are not numeric (e.g., 'January')
            # Assuming month names are in English for this example
            try:
                df['Date'] = pd.to_datetime(df['Year'].astype(
                    str) + '-' + df['Month'], format='%Y-%B')
            except ValueError:
                # Fallback if month is numeric or different format
                df['Date'] = pd.to_datetime(df['Year'].astype(
                    str) + '-' + df['Month'].astype(str))

            df = df.rename(columns=MORTALITY_COL_MAP)
            # Select relevant columns
            df = df[['Date', 'Mortality']]  # Add other columns if needed
            print("Mortality data loaded successfully.")
            return df
        elif 'dtp' in filepath:
            # Specify encoding AND header row index
            # Use parse_dates directly in read_csv
            date_col_name = 'Дата(месяц,год)'
            try:
                df = pd.read_csv(
                    filepath,
                    delimiter=';',
                    encoding='utf-8',
                    header=0,
                    parse_dates=[date_col_name],  # Specify column to parse
                    date_format='%m.%Y'          # Specify format for the parser
                )
                # Rename the parsed date column to 'Date' for consistency
                df = df.rename(columns={date_col_name: 'Date'})
            except ValueError as e:
                # If direct parsing fails, fall back to manual parsing (previous attempt)
                print(
                    f"Direct date parsing failed ({e}), attempting manual parsing...")
                df = pd.read_csv(filepath, delimiter=';',
                                 encoding='utf-8', header=0)
                df['Date'] = pd.to_datetime(df[date_col_name], format='%m.%Y')

            # Parse 'Дата(месяц,год)' which is in MM.YYYY format
            # df['Date'] = pd.to_datetime(df['Дата(месяц,год)'], format='%m.%Y') # Now handled by parse_dates
            # Rename columns if needed (assuming 'ДТП' is the target)
            df = df.rename(
                columns={'ДТП': 'DTP', 'Погибло': 'Deaths', 'Ранено': 'Injured'})
            # Select relevant columns
            df = df[['Date', 'DTP', 'Deaths', 'Injured']]
            print("DTP data loaded successfully.")
            return df
        else:
            print(
                f"Error: Unknown secondary data file type for path: {filepath}")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading secondary data: {e}")
        return pd.DataFrame()


def load_all_data(temp_path: str, secondary_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads both temperature and secondary datasets."""
    df_temp = load_temperature_data(temp_path)
    df_secondary = load_secondary_data(secondary_path)
    return df_temp, df_secondary


if __name__ == '__main__':
    # Example usage (for testing purposes)
    import sys
    sys.path.append('..')  # Add parent directory to path to import config
    import config

    # Use absolute paths for testing if running script directly from data_processing
    # base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # temp_path_test = os.path.join(base_path, config.TEMP_DATA_PATH)
    # secondary_path_test = os.path.join(base_path, config.SECONDARY_DATA_PATH)

    print("Testing data loading functions...")
    # Use paths directly from config, assuming script/notebook is run from project root
    df_temp, df_secondary = load_all_data(
        config.TEMP_DATA_PATH, config.SECONDARY_DATA_PATH)

    if not df_temp.empty:
        print("\nTemperature Data Head:")
        print(df_temp.head())
        print("\nTemperature Data Info:")
        df_temp.info()

    if not df_secondary.empty:
        print("\nSecondary Data Head:")
        print(df_secondary.head())
        print("\nSecondary Data Info:")
        df_secondary.info()

    print("\nData loading test finished.")
