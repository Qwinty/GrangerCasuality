# src/data_processing/cleaner.py
# Functions for cleaning and transforming the loaded data.

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def unify_timestamps(df: pd.DataFrame, date_col: str, target_format: str = '%Y-%m') -> pd.DataFrame:
    """
    Unifies timestamps to a monthly format (YYYY-MM) and sets it as the index.

    Args:
        df: Input DataFrame.
        date_col: Name of the column containing date/time information.
        target_format: The target string format for the monthly period.

    Returns:
        DataFrame with a PeriodIndex ('YYYY-MM').
    """
    print(f"Unifying timestamps for column: {date_col}")
    try:
        # Convert to datetime objects first to handle various input formats
        df[date_col] = pd.to_datetime(df[date_col])
        # Convert to monthly period and set as index
        df['Month'] = df[date_col].dt.to_period('M')
        df = df.set_index('Month')
        # Optionally, drop the original date column if no longer needed
        # df = df.drop(columns=[date_col])
        print("Timestamps unified and set as index.")
        return df
    except KeyError:
        print(f"Error: Date column '{date_col}' not found.")
        return df
    except Exception as e:
        print(f"Error unifying timestamps: {e}")
        return df


def normalize_data(series: pd.Series, method: Optional[str] = 'z-score') -> pd.Series:
    """
    Normalizes a data series using the specified method.

    Args:
        series: Input data series.
        method: Normalization method ('z-score', 'log', or None).

    Returns:
        Normalized data series.
    """
    print(f"Normalizing series using method: {method}")
    if method == 'z-score':
        # Handle potential zero standard deviation
        if series.std() == 0:
            print(
                f"Warning: Standard deviation is zero for series {series.name}. Returning original series.")
            return series
        return pd.Series(stats.zscore(series), index=series.index, name=series.name)
    elif method == 'log':
        # Add a small constant to handle zero or negative values if necessary
        if (series <= 0).any():
            print(
                f"Warning: Series {series.name} contains non-positive values. Adding 1 before log transformation.")
            # Ensure we don't modify the original series if it's used elsewhere
            series_adjusted = series.copy()
            # Replace non-positive with 1 for log(1)=0
            series_adjusted[series_adjusted <= 0] = 1
            return np.log(series_adjusted)
            # Alternative: Add a small epsilon, e.g., np.log(series + 1e-9)
            # Alternative: Return NaN for non-positive values if appropriate
        else:
            return np.log(series)
    elif method is None:
        print("No normalization applied.")
        return series
    else:
        print(
            f"Warning: Unknown normalization method '{method}'. Returning original series.")
        return series


def aggregate_monthly(df: pd.DataFrame, value_col: str, agg_func: str = 'mean') -> pd.Series:
    """
    Aggregates data to a monthly frequency using the specified function.
    Assumes df has a PeriodIndex ('M').

    Args:
        df: Input DataFrame with a monthly PeriodIndex.
        value_col: Name of the column containing the values to aggregate.
        agg_func: Aggregation function ('mean', 'sum', 'median', etc.).

    Returns:
        Series with monthly aggregated values.
    """
    print(f"Aggregating column '{value_col}' monthly using '{agg_func}'")
    
    # Ensure the index is DatetimeIndex for resampling
    df_copy = df.copy() # Work on a copy to avoid modifying original df
    if isinstance(df_copy.index, pd.PeriodIndex):
        print("Converting PeriodIndex to DatetimeIndex for aggregation.")
        try:
            df_copy.index = df_copy.index.to_timestamp()
        except Exception as e:
            print(f"Failed to convert PeriodIndex to DatetimeIndex: {e}")
            return pd.Series(dtype=float)
    elif not isinstance(df_copy.index, pd.DatetimeIndex):
        print("Attempting to convert index to DatetimeIndex for aggregation.")
        try:
            df_copy.index = pd.to_datetime(df_copy.index)
        except Exception as e:
            print(f"Failed to convert index to DatetimeIndex: {e}")
            return pd.Series(dtype=float)
    # else: index is already DatetimeIndex, proceed

    try:
        # Resample directly using DatetimeIndex
        # Use 'ME' for Month End frequency as 'M' is deprecated
        aggregated_series = df_copy[value_col].resample('ME').agg(agg_func)
        print("Aggregation complete.")
        return aggregated_series
    except KeyError:
        print(f"Error: Value column '{value_col}' not found.")
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return pd.Series(dtype=float)


if __name__ == '__main__':
    # Example usage (for testing purposes)
    print("\nTesting data cleaning functions...")

    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=60, freq='D')
    data = {'Date': dates, 'Value': np.random.rand(60) * 100}
    sample_df = pd.DataFrame(data)
    print("\nOriginal Sample DataFrame:")
    print(sample_df.head())

    # Test timestamp unification
    # Use copy to avoid modifying original
    unified_df = unify_timestamps(sample_df.copy(), 'Date')
    print("\nDataFrame after timestamp unification:")
    print(unified_df.head())
    print(unified_df.index)

    # Test aggregation
    if not unified_df.empty and 'Value' in unified_df.columns:
        aggregated_series = aggregate_monthly(
            unified_df, 'Value', agg_func='mean')
        print("\nAggregated Series (Mean):")
        print(aggregated_series)

        aggregated_sum = aggregate_monthly(unified_df, 'Value', agg_func='sum')
        print("\nAggregated Series (Sum):")
        print(aggregated_sum)
    else:
        print("\nSkipping aggregation test due to issues in previous steps or missing 'Value' column.")

    # Test normalization
    if 'aggregated_series' in locals() and not aggregated_series.empty:
        normalized_z = normalize_data(aggregated_series, method='z-score')
        print("\nNormalized Series (Z-score):")
        print(normalized_z)

        # Test log normalization (handle potential non-positive values)
        sample_log_series = pd.Series(
            [10, 20, 0, 40, -5], index=pd.period_range(start='2020-01', periods=5, freq='M'))
        normalized_log = normalize_data(sample_log_series, method='log')
        print("\nNormalized Series (Log - with non-positive handling):")
        print(normalized_log)

        normalized_none = normalize_data(aggregated_series, method=None)
        print("\nNormalized Series (None):")
        print(normalized_none)
    else:
        print("\nSkipping normalization test due to issues in aggregation.")

    print("\nData cleaning test finished.")
