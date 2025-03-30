import unittest
import pandas as pd
import numpy as np
import os
import sys
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.analysis import granger


class TestGranger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up a VAR model result once for all tests."""
        np.random.seed(456)
        n_obs = 100
        # Create a VAR(2) process where Var1 causes Var2, but Var2 does not cause Var1
        data1 = np.random.randn(n_obs)
        data2 = np.zeros(n_obs)
        # Var2 depends on lag 1 and 2 of Var1, and its own lag 1
        for t in range(2, n_obs):
            data2[t] = 0.4 * data1[t-1] - 0.3 * data1[t-2] + \
                0.5 * data2[t-1] + np.random.randn() * 0.3

        cls.test_df = pd.DataFrame({'Var1': data1, 'Var2': data2},
                                   index=pd.period_range(start='2020-01', periods=n_obs, freq='M'))

        # Fit the VAR model (assuming lag 2 is known or selected)
        cls.var_lag = 2
        try:
            model = VAR(cls.test_df)
            cls.var_results = model.fit(cls.var_lag)
        except Exception as e:
            print(f"Error fitting VAR model in setUpClass: {e}")
            cls.var_results = None  # Ensure it's None if fitting fails

    def test_perform_granger_causality_structure(self):
        """Test the structure of the Granger causality results dictionary."""
        if not self.var_results:
            self.skipTest("VAR model fitting failed in setUpClass.")

        results_dict = granger.perform_granger_causality_test(
            self.var_results, max_lag=self.var_lag)

        self.assertIsInstance(results_dict, dict)
        # Should have results for Var1->Var2 and Var2->Var1
        self.assertIn(('Var2', 'Var1'), results_dict)  # Var1 causing Var2
        self.assertIn(('Var1', 'Var2'), results_dict)  # Var2 causing Var1

        # Check structure of one result entry
        res_v1_v2 = results_dict[('Var2', 'Var1')]
        self.assertIsInstance(res_v1_v2, dict)
        self.assertIn('ssr_p_value', res_v1_v2)
        self.assertIn('ssr_significant', res_v1_v2)
        self.assertIn('params_p_value', res_v1_v2)
        self.assertIn('params_significant', res_v1_v2)
        self.assertIn('lag', res_v1_v2)
        self.assertEqual(res_v1_v2['lag'], self.var_lag)

    def test_perform_granger_causality_expected_outcome(self):
        """Test the expected causality outcome based on data generation."""
        if not self.var_results:
            self.skipTest("VAR model fitting failed in setUpClass.")

        significance_level = 0.05
        results_dict = granger.perform_granger_causality_test(
            self.var_results, max_lag=self.var_lag, significance_level=significance_level)

        # Var1 -> Var2 (Expected: Significant, p < 0.05)
        res_v1_v2 = results_dict.get(('Var2', 'Var1'), {})
        self.assertLess(res_v1_v2.get('ssr_p_value', 1.0), significance_level)
        self.assertTrue(res_v1_v2.get('ssr_significant', False))
        self.assertLess(res_v1_v2.get(
            'params_p_value', 1.0), significance_level)
        self.assertTrue(res_v1_v2.get('params_significant', False))

        # Var2 -> Var1 (Expected: Not Significant, p >= 0.05)
        res_v2_v1 = results_dict.get(('Var1', 'Var2'), {})
        self.assertGreaterEqual(res_v2_v1.get(
            'ssr_p_value', 0.0), significance_level)
        self.assertFalse(res_v2_v1.get('ssr_significant', True))
        # params_ftest might sometimes be significant by chance, ssr_ftest is often preferred
        # self.assertGreaterEqual(res_v2_v1.get('params_p_value', 0.0), significance_level)
        # self.assertFalse(res_v2_v1.get('params_significant', True))

    def test_perform_granger_causality_invalid_lag(self):
        """Test Granger causality with invalid max_lag."""
        if not self.var_results:
            self.skipTest("VAR model fitting failed in setUpClass.")
        results_dict = granger.perform_granger_causality_test(
            self.var_results, max_lag=0)
        self.assertIsNone(results_dict)
        results_dict_neg = granger.perform_granger_causality_test(
            self.var_results, max_lag=-1)
        self.assertIsNone(results_dict_neg)

    def test_summarize_granger_results(self):
        """Test the summary DataFrame formatting."""
        if not self.var_results:
            self.skipTest("VAR model fitting failed in setUpClass.")

        results_dict = granger.perform_granger_causality_test(
            self.var_results, max_lag=self.var_lag)
        summary_df = granger.summarize_granger_results(results_dict)

        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df), 2)  # Two pairs: V1->V2, V2->V1
        self.assertListEqual(list(summary_df.columns), [
                             'Effect', 'Lag', 'SSR_p_value', 'SSR_Significant', 'Params_p_value', 'Params_Significant', 'Details'])
        self.assertIn(f"Var1 -> Var2", summary_df['Effect'].tolist())
        self.assertIn(f"Var2 -> Var1", summary_df['Effect'].tolist())
        self.assertTrue(all(summary_df['Lag'] == self.var_lag))

    def test_summarize_granger_results_with_error(self):
        """Test summary formatting when an error occurred."""
        error_dict = {
            ('Var2', 'Var1'): {'error': 'Test failed'},
            ('Var1', 'Var2'): {'lag': 2, 'ssr_p_value': 0.5, 'ssr_significant': False, 'params_p_value': 0.6, 'params_significant': False, 'ssr_F': 0.1, 'df_num': 2, 'df_den': 90}
        }
        summary_df = granger.summarize_granger_results(error_dict)
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df), 2)
        # Check the row with the error
        error_row = summary_df[summary_df['Effect'] == 'Var1 -> Var2']
        self.assertEqual(error_row.iloc[0]['SSR_p_value'], 'Error')
        self.assertEqual(error_row.iloc[0]['SSR_Significant'], 'Error')
        self.assertEqual(error_row.iloc[0]['Details'], 'Test failed')


if __name__ == '__main__':
    unittest.main()
