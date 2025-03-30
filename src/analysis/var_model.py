# src/analysis/var_model.py
# Functions for fitting VAR models and selecting optimal lag order.

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from typing import Tuple, List, Optional, Dict


def select_optimal_lag(data: pd.DataFrame, max_lags: int, criteria: List[str] = ['aic', 'bic']) -> Dict[str, int]:
    """
    Selects the optimal lag order for a VAR model based on information criteria.

    Args:
        data: DataFrame containing the time series (assumed stationary).
        max_lags: Maximum number of lags to test.
        criteria: List of information criteria to use ('aic', 'bic', 'hqic', 'fpe').

    Returns:
        Dictionary mapping each criterion to the selected optimal lag order.
    """
    print(
        f"Selecting optimal lag order (max_lags={max_lags}) using criteria: {criteria}")
    if data.isnull().values.any():
        print("Warning: Data contains NaN values. Fitting VAR might fail or yield incorrect results. Consider imputation or dropping NaNs.")
        # Optionally drop NaNs here if appropriate for the analysis
        # data = data.dropna()

    try:
        model = VAR(data)
        # Note: statsmodels select_order might print results directly.
        # We capture the results programmatically.
        lag_selection_results = model.select_order(maxlags=max_lags)
        print("\nLag Selection Results Summary:")
        print(lag_selection_results.summary())

        optimal_lags = {}
        for criterion in criteria:
            if criterion in lag_selection_results.selected_orders:
                optimal_lags[criterion] = lag_selection_results.selected_orders[criterion]
            else:
                print(
                    f"Warning: Criterion '{criterion}' not found in selection results.")
                optimal_lags[criterion] = -1  # Indicate not found

        print("\nSelected Optimal Lags:")
        for crit, lag in optimal_lags.items():
            print(f"  {crit.upper()}: {lag}")

        # Return both the selected lags and the full summary object
        return {
            **optimal_lags, # Unpack the optimal lags (e.g., 'aic': 1, 'bic': 1)
            'summary': lag_selection_results # Add the summary object
        }

    except Exception as e:
        print(f"Error during lag selection: {e}")
        # Return -1 for all criteria on error
        return {crit: -1 for crit in criteria}


def fit_var_model(data: pd.DataFrame, lag_order: int) -> Optional[VARResults]:
    """
    Fits a VAR model to the data with the specified lag order.

    Args:
        data: DataFrame containing the time series (assumed stationary).
        lag_order: The number of lags to include in the model.

    Returns:
        Fitted VARResults object, or None if fitting fails.
    """
    print(f"Fitting VAR model with lag order: {lag_order}")
    if data.isnull().values.any():
        print("Warning: Data contains NaN values. Fitting VAR might fail. Consider imputation or dropping NaNs.")
        # data = data.dropna()

    if lag_order < 0:
        print("Error: Invalid lag order provided (must be >= 0).")
        return None
    # VAR(0) model is just intercepts, handle separately or disallow?
    # For Granger/IRF, lag > 0 is needed. Let's return None for lag=0 too.
    if lag_order == 0:
        print("Error: Lag order 0 is not suitable for Granger causality/IRF. Returning None.")
        return None

    try:
        model = VAR(data)
        results = model.fit(lag_order)
        print("\nVAR Model Fit Summary:")
        print(results.summary())
        return results
    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        return None


def check_model_stability(results: VARResults) -> bool:
    """
    Checks if the fitted VAR model is stable.
    (All roots of the characteristic polynomial lie outside the unit circle).

    Args:
        results: Fitted VARResults object.

    Returns:
        True if the model is stable, False otherwise.
    """
    print("Checking VAR model stability...")
    try:
        roots = results.roots
        # Stability condition: |root| < 1
        is_stable = np.all(np.abs(roots) < 1)
        # Note: Some sources define stability as roots *outside* the unit circle.
        # Statsmodels VAR `is_stable()` method checks if roots are *inside* or *on* the unit circle.
        # Let's use the statsmodels built-in check for consistency.
        # verbose=True prints the roots check
        is_stable_sm = results.is_stable(verbose=True)

        print(
            f"\nModel Stability Check (statsmodels): {'Stable' if is_stable_sm else 'Unstable'}")
        # You can also check manually:
        # max_root_magnitude = np.max(np.abs(roots)) if len(roots) > 0 else 0
        # print(f"  Maximum root magnitude: {max_root_magnitude:.4f}")
        # is_stable_manual = max_root_magnitude < 1
        # print(f"  Model Stable (Manual Check |root|<1): {is_stable_manual}")

        return is_stable_sm
    except Exception as e:
        print(f"Error checking model stability: {e}")
        return False


if __name__ == '__main__':
    # Example usage (for testing purposes)
    print("\nTesting VAR model functions...")

    # Create sample stationary data
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100)
    # Create a second series related to the first with a lag
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + \
        np.random.randn(100) * 0.5
    df_var_test = pd.DataFrame({'Var1': data1, 'Var2': data2}, index=idx)

    print("\nSample DataFrame for VAR:")
    print(df_var_test.head())

    # Select optimal lag
    max_lags_test = 5
    criteria_test = ['aic', 'bic']
    optimal_lags_result = select_optimal_lag(
        df_var_test, max_lags=max_lags_test, criteria=criteria_test)

    # Fit VAR model (using BIC lag, for example)
    chosen_lag = optimal_lags_result.get('bic', -1)
    if chosen_lag > 0:  # Ensure a valid lag was found
        var_results = fit_var_model(df_var_test, lag_order=chosen_lag)

        # Check stability
        if var_results:
            is_stable = check_model_stability(var_results)
        else:
            print("VAR model fitting failed, skipping stability check.")
    elif chosen_lag == 0:
        print("Optimal lag is 0, VAR model is not applicable in the standard sense. Consider VECM or other models.")
    else:
        print("Optimal lag selection failed or returned invalid lag.")

    print("\nVAR model test finished.")
