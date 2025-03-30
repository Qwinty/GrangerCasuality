# src/utils/helpers.py
# General utility functions for the project.

import pandas as pd
import numpy as np
import time
from functools import wraps

def check_data_consistency(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Performs basic consistency checks between two dataframes,
    typically before merging.
    """
    print("Performing basic data consistency checks...")
    # Check index types
    if type(df1.index) != type(df2.index):
        print(f"Warning: Index types differ - {type(df1.index)} vs {type(df2.index)}")
    # Check index frequency if applicable
    if isinstance(df1.index, (pd.DatetimeIndex, pd.PeriodIndex)) and \
       isinstance(df2.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        if getattr(df1.index, 'freq', None) != getattr(df2.index, 'freq', None):
             print(f"Warning: Index frequencies differ - {getattr(df1.index, 'freq', None)} vs {getattr(df2.index, 'freq', None)}")
    # Check for overlapping column names (excluding index)
    common_cols = df1.columns.intersection(df2.columns)
    if not common_cols.empty:
        print(f"Warning: Common columns found: {common_cols.tolist()}. Consider renaming before merge.")

    print("Consistency checks finished.")


def timeit(func):
    """Decorator to time function execution."""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


# Example of a potential helper for data transformation validation
def ensure_series_positive(series: pd.Series, series_name: str = "Series") -> bool:
    """Checks if all values in a Series are positive."""
    if (series <= 0).any():
        print(f"Warning: {series_name} contains non-positive values.")
        return False
    return True


if __name__ == '__main__':
    print("\nTesting helper functions...")

    # Test timeit decorator
    @timeit
    def sample_function(duration):
        print(f"Running sample function for {duration} seconds...")
        time.sleep(duration)
        print("Sample function finished.")
        return "Done"

    sample_function(0.5)

    # Test consistency check
    idx1 = pd.period_range(start='2020-01', periods=5, freq='M')
    df_a = pd.DataFrame({'ValueA': range(5), 'Common': range(5)}, index=idx1)
    idx2 = pd.date_range(start='2020-01-01', periods=5, freq='MS') # Different index type/freq
    df_b = pd.DataFrame({'ValueB': range(5,10), 'Common': range(5,10)}, index=idx2)

    print("\nTesting data consistency check:")
    check_data_consistency(df_a, df_b)

    # Test ensure_series_positive
    print("\nTesting ensure_series_positive:")
    positive_series = pd.Series([1, 2, 3])
    mixed_series = pd.Series([1, 0, -2])
    print(f"Positive series check: {ensure_series_positive(positive_series, 'Positive Series')}")
    print(f"Mixed series check: {ensure_series_positive(mixed_series, 'Mixed Series')}")


    print("\nHelper functions test finished.")