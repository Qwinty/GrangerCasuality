# src/analysis/granger.py
# Functions for performing Granger causality tests on VAR results.

import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Dict, Any, Optional, Tuple
import numpy as np


def perform_granger_causality_test(results: VARResults, max_lag: int, significance_level: float = 0.05) -> Optional[Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    Performs Granger causality tests for all variable pairs in a fitted VAR model.

    Args:
        results: Fitted VARResults object from statsmodels.
        max_lag: The maximum lag order to test for causality (should typically be the
                 lag order of the fitted VAR model).
        significance_level: Threshold for p-value to determine significance.

    Returns:
        A dictionary where keys are tuples (caused_variable, causing_variable)
        and values are dictionaries containing test results (p-value, F-statistic,
        degrees of freedom, and significance). Returns None if an error occurs.
    """
    print(
        f"\nPerforming Granger Causality Tests (max_lag={max_lag}, alpha={significance_level})...")
    if max_lag <= 0:
        print("Error: max_lag must be a positive integer.")
        return None

    variables = results.names
    data = pd.DataFrame(results.model.y, columns=variables) 

    test_results = {}

    for caused_var in variables:
        for causing_var in variables:
            if caused_var == causing_var:
                continue  # Skip testing variable on itself

            print(f"  Testing: {causing_var} Granger-causes {caused_var}?")

            # Select the relevant columns for the bivariate test
            test_data = data[[caused_var, causing_var]]

            try:
                # statsmodels grangercausalitytests expects a DataFrame/array
                # It performs tests for lags 1 to max_lag
                # We are interested in the results for the specified max_lag
                gc_results = grangercausalitytests(
                    test_data, [max_lag], verbose=False)

                # Extract results for the specified max_lag
                # The results dictionary is keyed by the lag number
                # [0] accesses the test results dict
                lag_result = gc_results[max_lag][0]

                f_test_stat = lag_result['ssr_ftest'][0]  # F-statistic
                p_value = lag_result['ssr_ftest'][1]     # p-value
                # Numerator degrees of freedom (lag)
                df_num = lag_result['ssr_ftest'][2]
                # Denominator degrees of freedom
                df_den = lag_result['ssr_ftest'][3]

                # Also consider 'params_ftest' which tests joint significance of lagged causing variable coeffs
                f_params_stat = lag_result['params_ftest'][0]
                p_params_value = lag_result['params_ftest'][1]

                significant = p_value < significance_level
                significant_params = p_params_value < significance_level

                test_results[(caused_var, causing_var)] = {
                    'ssr_F': f_test_stat,
                    'ssr_p_value': p_value,
                    'ssr_significant': significant,
                    'params_F': f_params_stat,
                    'params_p_value': p_params_value,
                    'params_significant': significant_params,
                    'lag': max_lag,
                    'df_num': df_num,
                    'df_den': df_den
                }
                print(
                    f"    ssr_ftest: p-value={p_value:.4f} ({'Significant' if significant else 'Not Significant'})")
                print(
                    f"    params_ftest: p-value={p_params_value:.4f} ({'Significant' if significant_params else 'Not Significant'})")

            except Exception as e:
                print(f"    Error testing {causing_var} -> {caused_var}: {e}")
                test_results[(caused_var, causing_var)] = {'error': str(e)}

    return test_results


def summarize_granger_results(results_dict: Dict[Tuple[str, str], Dict[str, Any]]) -> pd.DataFrame:
    """Formats the Granger causality test results into a readable DataFrame."""
    summary_list = []
    for (caused, causing), results in results_dict.items():
        if 'error' in results:
            summary_list.append({
                'Effect': f"{causing} -> {caused}",
                'Lag': results.get('lag', 'N/A'),
                'SSR_p_value': 'Error',
                'SSR_Significant': 'Error',
                'Params_p_value': 'Error',
                'Params_Significant': 'Error',
                'Details': results['error']
            })
        else:
            summary_list.append({
                'Effect': f"{causing} -> {caused}",
                'Lag': results['lag'],
                'SSR_p_value': f"{results['ssr_p_value']:.4f}",
                'SSR_Significant': results['ssr_significant'],
                'Params_p_value': f"{results['params_p_value']:.4f}",
                'Params_Significant': results['params_significant'],
                'Details': f"F={results['ssr_F']:.2f}, df=({results['df_num']:.0f}, {results['df_den']:.0f})"
            })
    return pd.DataFrame(summary_list)


if __name__ == '__main__':
    # Example usage (requires a fitted VAR model)
    print("\nTesting Granger causality functions...")
    # Re-use VAR fitting example from var_model.py (or mock results)

    # --- Mocking VARResults ---
    # This part would typically use the actual results from fit_var_model
    from statsmodels.tsa.vector_ar.var_model import VAR
    idx = pd.period_range(start='2020-01', periods=100, freq='M')
    data1 = np.random.randn(100).cumsum()  # Non-stationary
    data2 = 0.5 * pd.Series(data1).shift(1).fillna(0) + \
        np.random.randn(100) * 0.5
    # Make data stationary for VAR/Granger
    df_granger_test = pd.DataFrame(
        {'Var1': np.diff(data1), 'Var2': np.diff(data2)}, index=idx[1:])

    print("\nSample DataFrame for Granger Test (Differenced):")
    print(df_granger_test.head())

    var_lag = 2  # Assume optimal lag is 2 for this example
    try:
        model = VAR(df_granger_test)
        mock_results = model.fit(var_lag)
        print("\nFitted Mock VAR Model Summary:")
        print(mock_results.summary())

        # Perform Granger test
        granger_test_results = perform_granger_causality_test(
            mock_results, max_lag=var_lag, significance_level=0.05)

        if granger_test_results:
            print("\nGranger Causality Test Raw Results:")
            # print(granger_test_results) # Can be verbose
            summary_df = summarize_granger_results(granger_test_results)
            print("\nGranger Causality Test Summary Table:")
            print(summary_df)
        else:
            print("Granger causality testing failed.")

    except Exception as e:
        print(f"\nError during VAR fitting or Granger testing in example: {e}")

    print("\nGranger causality test finished.")
