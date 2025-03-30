# src/analysis/stationarity.py
# Functions for checking time series stationarity using ADF and KPSS tests.

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Tuple, Dict, Optional
import numpy as np


def check_stationarity_adf(series: pd.Series, significance_level: float = 0.05, regression: str = 'c') -> Tuple[bool, float]:
    """
    Performs the Augmented Dickey-Fuller (ADF) test for stationarity.

    Null Hypothesis (H0): The series has a unit root (non-stationary).
    Alternative Hypothesis (H1): The series does not have a unit root (stationary).

    Args:
        series: Time series data.
        significance_level: Threshold for p-value.
        regression: Type of regression ('c', 'ct', 'ctt', 'n').
                    'c' - constant only (default)
                    'ct' - constant and trend
                    'ctt' - constant, linear and quadratic trend
                    'n' - no constant, no trend

    Returns:
        Tuple (is_stationary, p_value). is_stationary is True if H0 is rejected.
    """
    print(
        f"Performing ADF test on series: {series.name} (Regression: {regression})")
    try:
        # Drop NA before test
        result = adfuller(series.dropna(), regression=regression)
        p_value = result[1]
        is_stationary = p_value < significance_level
        print(f"ADF Test Results for {series.name}:")
        print(f"  Test Statistic: {result[0]:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Lags Used: {result[2]}")
        print(f"  Is Stationary (p < {significance_level}): {is_stationary}")
        return is_stationary, p_value
    except Exception as e:
        print(f"Error during ADF test for {series.name}: {e}")
        return False, 1.0  # Assume non-stationary on error


def check_stationarity_kpss(series: pd.Series, significance_level: float = 0.05, regression: str = 'c') -> Tuple[bool, float]:
    """
    Performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.

    Null Hypothesis (H0): The series is trend-stationary (or level-stationary if regression='c').
    Alternative Hypothesis (H1): The series has a unit root (non-stationary).

    Args:
        series: Time series data.
        significance_level: Threshold for p-value.
        regression: Type of regression ('c', 'ct').
                    'c' - test is for level stationarity (default)
                    'ct' - test is for trend stationarity

    Returns:
        Tuple (is_stationary, p_value). is_stationary is True if H0 is NOT rejected.
        Note: Interpretation is opposite to ADF test.
    """
    # Pre-check for constant series (zero variance)
    if series.dropna().var() < 1e-10:  # Use a small threshold for floating point
        print(
            f"KPSS test skipped for {series.name}: Series variance is effectively zero (constant value). Assuming stationary.")
        return True, 1.0

    print(
        f"Performing KPSS test on series: {series.name} (Regression: {regression})")
    try:
        # Note: nlags='auto' is generally recommended
        result = kpss(series.dropna(), regression=regression,
                      nlags='auto')  # Drop NA
        p_value = result[1]
        # KPSS interpretation: If p-value is LOW, reject H0 (series is non-stationary)
        is_stationary = p_value >= significance_level
        print(f"KPSS Test Results for {series.name}:")
        print(f"  Test Statistic: {result[0]:.4f}")
        print(
            f"  P-value: {p_value:.4f} (Note: p-values are interpolated and may be capped at 0.01 or 0.1)")
        print(f"  Lags Used: {result[2]}")
        print(f"  Is Stationary (p >= {significance_level}): {is_stationary}")
        return is_stationary, p_value
    except Exception as e:
        print(f"Error during KPSS test for {series.name}: {e}")
        return False, 0.0  # Assume non-stationary on error (low p-value)


def apply_differencing(series: pd.Series, order: int = 1) -> pd.Series:
    """Applies differencing to a series."""
    if order <= 0:
        return series
    print(f"Applying differencing of order {order} to series: {series.name}")
    return series.diff(order).dropna()


def check_stationarity_on_dataframe(df: pd.DataFrame, adf_level: float = 0.05, kpss_level: float = 0.05) -> Dict[str, Dict[str, Tuple[bool, float]]]:
    """Runs ADF and KPSS tests on all columns of a DataFrame."""
    results = {}
    for col in df.columns:
        print(f"\n--- Checking Stationarity for: {col} ---")
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
    print("\nTesting stationarity functions...")

    # Create sample data
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    # Non-stationary (random walk)
    non_stationary_data = np.random.randn(100).cumsum()
    # Stationary data
    stationary_data = np.random.randn(100)

    df_test = pd.DataFrame({
        'Stationary': stationary_data,
        'NonStationary': non_stationary_data
    }, index=idx)

    print("\nSample Test DataFrame:")
    print(df_test.head())

    # Test on DataFrame
    stationarity_results = check_stationarity_on_dataframe(df_test)
    print("\nStationarity Test Results Summary:")
    print(stationarity_results)

    # Test differencing
    diff_series = apply_differencing(df_test['NonStationary'], order=1)
    print("\nNon-Stationary Series after 1st order differencing:")
    print(diff_series.head())

    print("\n--- Checking Stationarity for Differenced Series ---")
    adf_stat_diff, adf_p_diff = check_stationarity_adf(diff_series)
    kpss_stat_diff, kpss_p_diff = check_stationarity_kpss(diff_series)
    print(
        f"Differenced ADF: Stationary={adf_stat_diff}, p-value={adf_p_diff:.4f}")
    print(
        f"Differenced KPSS: Stationary={kpss_stat_diff}, p-value={kpss_p_diff:.4f}")

    print("\nStationarity test finished.")
