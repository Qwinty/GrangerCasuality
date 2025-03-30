import unittest
import pandas as pd
import numpy as np
import os
import sys

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.analysis import stationarity

class TestStationarity(unittest.TestCase):

    def setUp(self):
        """Set up test series."""
        np.random.seed(42) # for reproducible results
        # Stationary series (white noise)
        self.stationary_series = pd.Series(np.random.randn(100), name='Stationary')
        # Non-stationary series (random walk)
        self.non_stationary_series = pd.Series(np.random.randn(100).cumsum(), name='NonStationary')
        # Trend-stationary series (linear trend + noise)
        self.trend_stationary_series = pd.Series(np.arange(100) * 0.5 + np.random.randn(100), name='TrendStationary')
        # Constant series
        self.constant_series = pd.Series([5.0] * 100, name='Constant')

        self.test_df = pd.DataFrame({
            'Stationary': self.stationary_series,
            'NonStationary': self.non_stationary_series,
            'TrendStationary': self.trend_stationary_series,
            'Constant': self.constant_series
        })

    # --- ADF Tests ---
    def test_adf_stationary(self):
        """Test ADF on a stationary series."""
        is_stationary, p_value = stationarity.check_stationarity_adf(self.stationary_series)
        self.assertTrue(is_stationary)
        self.assertLess(p_value, 0.05)

    def test_adf_non_stationary(self):
        """Test ADF on a non-stationary (random walk) series."""
        is_stationary, p_value = stationarity.check_stationarity_adf(self.non_stationary_series)
        self.assertFalse(is_stationary)
        self.assertGreaterEqual(p_value, 0.05)

    def test_adf_trend_stationary(self):
        """Test ADF on a trend-stationary series (should appear non-stationary without trend term)."""
        # Test with constant only ('c') - should likely fail to reject H0 (non-stationary)
        is_stationary_c, p_value_c = stationarity.check_stationarity_adf(self.trend_stationary_series, regression='c')
        self.assertFalse(is_stationary_c)

        # Test with constant and trend ('ct') - should reject H0 (stationary around trend)
        is_stationary_ct, p_value_ct = stationarity.check_stationarity_adf(self.trend_stationary_series, regression='ct')
        self.assertTrue(is_stationary_ct)
        self.assertLess(p_value_ct, 0.05)

    # --- KPSS Tests ---
    def test_kpss_stationary(self):
        """Test KPSS on a stationary series."""
        # Test for level stationarity ('c')
        is_stationary_c, p_value_c = stationarity.check_stationarity_kpss(self.stationary_series, regression='c')
        self.assertTrue(is_stationary_c)
        self.assertGreaterEqual(p_value_c, 0.05)

    def test_kpss_non_stationary(self):
        """Test KPSS on a non-stationary (random walk) series."""
        # Should reject H0 (level stationarity)
        is_stationary_c, p_value_c = stationarity.check_stationarity_kpss(self.non_stationary_series, regression='c')
        self.assertFalse(is_stationary_c)
        self.assertLess(p_value_c, 0.05)

    def test_kpss_trend_stationary(self):
        """Test KPSS on a trend-stationary series."""
        # Test for level stationarity ('c') - should reject H0
        is_stationary_c, p_value_c = stationarity.check_stationarity_kpss(self.trend_stationary_series, regression='c')
        self.assertFalse(is_stationary_c)
        self.assertLess(p_value_c, 0.05)

        # Test for trend stationarity ('ct') - should NOT reject H0
        is_stationary_ct, p_value_ct = stationarity.check_stationarity_kpss(self.trend_stationary_series, regression='ct')
        self.assertTrue(is_stationary_ct)
        self.assertGreaterEqual(p_value_ct, 0.05)

    def test_kpss_constant(self):
        """Test KPSS on a constant series (should handle zero variance)."""
        # Should return True (stationary) and print a warning
        is_stationary, p_value = stationarity.check_stationarity_kpss(self.constant_series)
        self.assertTrue(is_stationary)

    # --- Differencing Test ---
    def test_apply_differencing(self):
        """Test differencing function."""
        # First order difference
        diff1 = stationarity.apply_differencing(self.non_stationary_series, order=1)
        self.assertEqual(len(diff1), len(self.non_stationary_series) - 1)
        # A random walk differenced once should be stationary (white noise)
        is_stationary_adf, _ = stationarity.check_stationarity_adf(diff1)
        is_stationary_kpss, _ = stationarity.check_stationarity_kpss(diff1)
        self.assertTrue(is_stationary_adf)
        self.assertTrue(is_stationary_kpss)

        # Second order difference
        diff2 = stationarity.apply_differencing(self.non_stationary_series, order=2)
        self.assertEqual(len(diff2), len(self.non_stationary_series) - 2)

        # Zero order difference (should return original)
        diff0 = stationarity.apply_differencing(self.non_stationary_series, order=0)
        pd.testing.assert_series_equal(diff0, self.non_stationary_series)

    # --- DataFrame Test ---
    def test_check_stationarity_on_dataframe(self):
        """Test running checks on a full DataFrame."""
        results = stationarity.check_stationarity_on_dataframe(self.test_df)
        self.assertIsInstance(results, dict)
        self.assertIn('Stationary', results)
        self.assertIn('NonStationary', results)
        self.assertIn('TrendStationary', results)
        self.assertIn('Constant', results)

        # Check specific expected outcomes based on individual tests
        self.assertTrue(results['Stationary']['ADF'][0])    # Stationary ADF=True
        self.assertTrue(results['Stationary']['KPSS'][0])   # Stationary KPSS=True
        self.assertFalse(results['NonStationary']['ADF'][0]) # NonStat ADF=False
        self.assertFalse(results['NonStationary']['KPSS'][0])# NonStat KPSS=False (rejects H0)
        # Constant series results might vary slightly depending on implementation details
        # but ADF might fail, KPSS should pass (or be skipped)
        self.assertTrue(results['Constant']['KPSS'][0]) # KPSS should handle constant

if __name__ == '__main__':
    unittest.main()