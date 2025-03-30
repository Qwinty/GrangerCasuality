# src/data_processing/merger.py
# Functions for merging the preprocessed time series data.

import pandas as pd
from typing import Tuple
import numpy as np


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, how: str = 'inner') -> pd.DataFrame:
    """
    Merges two DataFrames based on their indices (assumed to be time-based).

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        how: Type of merge to be performed ('inner', 'outer', 'left', 'right').
             'inner' is recommended to keep only overlapping time periods.

    Returns:
        Merged DataFrame.
    """
    print(f"Merging dataframes using method: {how}")
    if not isinstance(df1.index, (pd.PeriodIndex, pd.DatetimeIndex)) or \
       not isinstance(df2.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        print("Warning: One or both DataFrames do not have a time-based index. Merge might be incorrect.")

    try:
        # Ensure indices are compatible (e.g., both PeriodIndex or both DatetimeIndex)
        # If one is PeriodIndex and other is DatetimeIndex, convert one for merging
        if isinstance(df1.index, pd.PeriodIndex) and isinstance(df2.index, pd.DatetimeIndex):
            df2.index = df2.index.to_period(df1.index.freq)
        elif isinstance(df2.index, pd.PeriodIndex) and isinstance(df1.index, pd.DatetimeIndex):
            df1.index = df1.index.to_period(df2.index.freq)

        merged_df = pd.merge(df1, df2, left_index=True,
                             right_index=True, how=how)

        # Check for duplicate column names after merge (if dfs had same column names initially)
        if merged_df.columns.duplicated().any():
            print("Warning: Merged DataFrame contains duplicate column names. Consider renaming columns before merging.")

        print("DataFrames merged successfully.")
        return merged_df
    except Exception as e:
        print(f"Error merging DataFrames: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def check_completeness(df: pd.DataFrame) -> None:
    """
    Проверяет объединенный DataFrame на наличие пропущенных значений и временных пробелов.
    """
    print("Checking merged data completeness...")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")

    if isinstance(df.index, (pd.PeriodIndex, pd.DatetimeIndex)):
        # Check for gaps in the time index
        expected_index = pd.period_range(start=df.index.min(), end=df.index.max(), freq=df.index.freq) if isinstance(
            df.index, pd.PeriodIndex) else pd.date_range(start=df.index.min(), end=df.index.max(), freq=df.index.freq)
        if len(df.index) != len(expected_index):
            print(
                f"Warning: Time series gaps detected. Expected {len(expected_index)} periods, found {len(df.index)}.")
            # Optionally, identify the missing periods/dates
            missing_periods = expected_index.difference(df.index)
            print(f"Missing periods/dates: {missing_periods}")
        else:
            print("Time series index is continuous (no gaps).")
    else:
        print("Index is not time-based, skipping gap check.")


if __name__ == '__main__':
    # Example usage (for testing purposes)
    print("\nTesting data merging functions...")

    # Create sample preprocessed data
    idx1 = pd.period_range(start='2020-01', periods=12, freq='M')
    df_temp_processed = pd.DataFrame(
        {'Temperature_Norm': np.random.randn(12)}, index=idx1)

    # Overlapping but different start/end
    idx2 = pd.period_range(start='2020-03', periods=12, freq='M')
    df_secondary_processed = pd.DataFrame(
        {'Secondary_Norm': np.random.rand(12)}, index=idx2)

    print("\nSample Processed Temperature Data:")
    print(df_temp_processed)
    print("\nSample Processed Secondary Data:")
    print(df_secondary_processed)

    # Test inner merge
    merged_inner = merge_dataframes(
        df_temp_processed, df_secondary_processed, how='inner')
    print("\nMerged DataFrame (Inner):")
    print(merged_inner)
    if not merged_inner.empty:
        check_completeness(merged_inner)

    # Test outer merge
    merged_outer = merge_dataframes(
        df_temp_processed, df_secondary_processed, how='outer')
    print("\nMerged DataFrame (Outer):")
    print(merged_outer)
    if not merged_outer.empty:
        check_completeness(merged_outer)  # Expect missing values here

    print("\nData merging test finished.")
