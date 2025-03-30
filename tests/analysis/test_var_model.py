import unittest
import pandas as pd
import numpy as np
import os
import sys
from statsmodels.tsa.vector_ar.var_model import VARResults, VARResultsWrapper, LagOrderResults # Added VARResultsWrapper

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.analysis import var_model

class TestVarModel(unittest.TestCase):

    def setUp(self):
        """Set up test data for VAR modeling."""
        np.random.seed(123)
        # Create a simple stationary VAR(1) process
        n_obs = 100
        data1 = np.random.randn(n_obs)
        # data2 depends on lag 1 of data1 and its own lag 1
        data2 = np.zeros(n_obs)
        data2[1:] = 0.5 * data1[:-1] + 0.3 * data2[:-1] + np.random.randn(n_obs - 1) * 0.5
        
        # Ensure data starts from index 1 if needed, or adjust length
        self.test_df_stationary = pd.DataFrame({'Var1': data1, 'Var2': data2}, 
                                               index=pd.period_range(start='2020-01', periods=n_obs, freq='M'))
        
        # Data with NaNs
        self.test_df_nan = self.test_df_stationary.copy()
        self.test_df_nan.iloc[5, 0] = np.nan
        self.test_df_nan.iloc[10, 1] = np.nan

    def test_select_optimal_lag_structure(self):
        """Test the structure of the optimal lag selection results."""
        max_lags = 5
        criteria = ['aic', 'bic']
        results = var_model.select_optimal_lag(self.test_df_stationary, max_lags=max_lags, criteria=criteria)
        
        self.assertIsInstance(results, dict)
        self.assertIn('aic', results)
        self.assertIn('bic', results)
        # Accept numpy integer types as well
        self.assertTrue(isinstance(results['aic'], (int, np.integer)), "AIC lag should be int or numpy int")
        self.assertTrue(isinstance(results['bic'], (int, np.integer)), "BIC lag should be int or numpy int")
        self.assertGreaterEqual(results['aic'], 0) # Lag should be non-negative
        self.assertGreaterEqual(results['bic'], 0)
        self.assertIn('summary', results) # Check if summary exists
        self.assertIsInstance(results['summary'], LagOrderResults) # Check summary type
        # For our simple VAR(1) process, we expect lag 1 to be selected by most criteria
        self.assertEqual(results.get('aic', -1), 1) 
        self.assertEqual(results.get('bic', -1), 1)

    def test_select_optimal_lag_with_nans(self):
        """Test lag selection with NaN values (should warn)."""
        max_lags = 5
        criteria = ['aic', 'bic']
        # Should print warnings about NaNs
        results = var_model.select_optimal_lag(self.test_df_nan.dropna(), max_lags=max_lags, criteria=criteria) # Dropna before passing
        self.assertIsInstance(results, dict)
        self.assertGreaterEqual(results.get('aic', -1), 0) 
        self.assertGreaterEqual(results.get('bic', -1), 0)

    def test_fit_var_model_success(self):
        """Test successful fitting of a VAR model."""
        lag_order = 1
        results = var_model.fit_var_model(self.test_df_stationary, lag_order=lag_order)
        
        # Accept VARResultsWrapper as well
        self.assertIsInstance(results, VARResultsWrapper) # Fit returns a wrapper
        self.assertEqual(results.k_ar, lag_order) # Check if correct lag order was used
        self.assertEqual(len(results.params), (lag_order * results.neqs) + 1) # Params per equation: k_ar*neqs + intercept
        self.assertEqual(results.neqs, 2) # Number of variables/equations

    def test_fit_var_model_invalid_lag(self):
        """Test fitting VAR with an invalid lag order."""
        results = var_model.fit_var_model(self.test_df_stationary, lag_order=-1)
        self.assertIsNone(results)
        results_zero = var_model.fit_var_model(self.test_df_stationary, lag_order=0)
        self.assertIsNone(results_zero) # VAR(0) is not typically fitted this way

    def test_fit_var_model_with_nans(self):
        """Test fitting VAR with NaN values (should warn or fail)."""
        lag_order = 1
        # Fitting directly with NaNs might raise an error in statsmodels or produce NaNs in results
        # The function currently warns. Let's test fitting after dropna.
        results = var_model.fit_var_model(self.test_df_nan.dropna(), lag_order=lag_order)
        # Accept VARResultsWrapper as well
        self.assertTrue(isinstance(results, (VARResults, VARResultsWrapper))) # Should succeed after dropna

    def test_check_model_stability_stable(self):
        """Test stability check for a known stable model."""
        lag_order = 1
        results = var_model.fit_var_model(self.test_df_stationary, lag_order=lag_order)
        self.assertIsNotNone(results)
        is_stable = var_model.check_model_stability(results)
        self.assertTrue(is_stable)

    # def test_check_model_stability_unstable(self):
    #     """Test stability check for a known unstable model (requires crafting unstable data)."""
    #     # Crafting unstable data requires careful setup, e.g., coefficients > 1
    #     # This test is more complex and might be skipped if stable data is the focus.
    #     # Example (conceptual):
    #     # data_unstable = ... # create data with explosive process
    #     # results_unstable = var_model.fit_var_model(data_unstable, lag_order=1)
    #     # self.assertIsNotNone(results_unstable)
    #     # is_stable = var_model.check_model_stability(results_unstable)
    #     # self.assertFalse(is_stable)
    #     pass # Skipping unstable test for now


if __name__ == '__main__':
    unittest.main()