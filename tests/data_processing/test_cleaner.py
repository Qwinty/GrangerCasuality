from src.data_processing import cleaner
import unittest
import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class TestCleaner(unittest.TestCase):

    def setUp(self):
        """Set up test data."""
        self.dates = pd.date_range(
            start='2022-01-01', periods=65, freq='D')  # ~2 months
        self.values = np.random.rand(65) * 10 + 5
        self.sample_df = pd.DataFrame(
            {'Date': self.dates, 'Value': self.values})

        self.series_for_norm = pd.Series([10, 20, 30, 40, 50], name='NormTest')
        self.series_with_zero = pd.Series([10, 0, 30, 0, 50], name='ZeroTest')
        self.series_with_neg = pd.Series([10, -5, 30, -1, 50], name='NegTest')
        self.series_constant = pd.Series([5, 5, 5, 5, 5], name='ConstantTest')

    def test_unify_timestamps_success(self):
        """Test successful timestamp unification."""
        df = cleaner.unify_timestamps(self.sample_df.copy(), date_col='Date')
        self.assertIsInstance(df.index, pd.PeriodIndex)
        self.assertEqual(df.index.freqstr, 'M')
        self.assertEqual(len(df), 65)  # Length remains same, index is added
        # Original date column is kept by default
        self.assertTrue('Date' in df.columns)
        # Check if index values are correct periods
        self.assertEqual(df.index[0], pd.Period('2022-01', freq='M'))
        self.assertEqual(df.index[31], pd.Period(
            '2022-02', freq='M'))  # 31st day is Feb 1st
        self.assertEqual(df.index[64], pd.Period(
            '2022-03', freq='M'))  # 65th day is Mar 6th

    def test_unify_timestamps_key_error(self):
        """Test timestamp unification with incorrect date column."""
        df = cleaner.unify_timestamps(
            self.sample_df.copy(), date_col='WrongDateCol')
        # Should return the original df and print an error (check logs/stdout)
        pd.testing.assert_frame_equal(df, self.sample_df)
        # Assert index is not PeriodIndex
        self.assertNotIsInstance(df.index, pd.PeriodIndex)

    def test_normalize_data_zscore(self):
        """Test Z-score normalization."""
        normalized = cleaner.normalize_data(
            self.series_for_norm, method='z-score')
        # Compare directly with scipy.stats.zscore result
        expected_values = stats.zscore(self.series_for_norm)
        expected = pd.Series(expected_values, name=self.series_for_norm.name)
        pd.testing.assert_series_equal(normalized, expected)
        self.assertAlmostEqual(normalized.mean(), 0.0, places=6)
        # Note: scipy.stats.zscore uses ddof=0, so std might not be exactly 1 if original data std used ddof=1
        # Let's check the std dev calculated with ddof=0
        self.assertAlmostEqual(np.std(normalized), 1.0, places=6)
 # Use np.std with default ddof=0

    def test_normalize_data_zscore_zero_std(self):
        """Test Z-score normalization with zero standard deviation."""
        normalized = cleaner.normalize_data(
            self.series_constant, method='z-score')
        # Should return the original series and print a warning
        pd.testing.assert_series_equal(normalized, self.series_constant)

    def test_normalize_data_log(self):
        """Test log normalization for positive values."""
        normalized = cleaner.normalize_data(self.series_for_norm, method='log')
        expected = np.log(self.series_for_norm)
        pd.testing.assert_series_equal(normalized, expected)

    def test_normalize_data_log_non_positive(self):
        """Test log normalization handling of zero/negative values."""
        # Test with zero
        normalized_zero = cleaner.normalize_data(
            self.series_with_zero, method='log')
        # Replaces 0 with 1 for log(1)=0
        expected_zero = np.log(self.series_with_zero.replace(0, 1))
        pd.testing.assert_series_equal(normalized_zero, expected_zero)

        # Test with negative
        normalized_neg = cleaner.normalize_data(
            self.series_with_neg, method='log')
        expected_neg = np.log(self.series_with_neg.mask(
            self.series_with_neg <= 0, 1))  # Replaces <=0 with 1
        pd.testing.assert_series_equal(normalized_neg, expected_neg)

    def test_normalize_data_none(self):
        """Test applying no normalization."""
        normalized = cleaner.normalize_data(self.series_for_norm, method=None)
        pd.testing.assert_series_equal(normalized, self.series_for_norm)

    def test_normalize_data_unknown(self):
        """Test applying an unknown normalization method."""
        normalized = cleaner.normalize_data(
            self.series_for_norm, method='unknown_method')
        # Should return original series and print warning
        pd.testing.assert_series_equal(normalized, self.series_for_norm)

    def test_aggregate_monthly_success(self):
        """Test monthly aggregation (mean and sum)."""
        df_unified = cleaner.unify_timestamps(
            self.sample_df.copy(), date_col='Date')

        # Test mean aggregation
        aggregated_mean = cleaner.aggregate_monthly(
            df_unified, value_col='Value', agg_func='mean')
        self.assertIsInstance(aggregated_mean, pd.Series)
        self.assertFalse(aggregated_mean.empty)  # Check not empty on success
        # Expect DatetimeIndex with Month End frequency after resampling
        self.assertIsInstance(aggregated_mean.index, pd.DatetimeIndex)
        self.assertEqual(aggregated_mean.index.freqstr, 'ME')

        self.assertEqual(len(aggregated_mean), 3)  # Jan, Feb, Mar
        # Check the actual datetime values (should be month ends)
        self.assertEqual(aggregated_mean.index[0], pd.Timestamp('2022-01-31'))
        self.assertEqual(aggregated_mean.index[1], pd.Timestamp('2022-02-28'))
        self.assertEqual(aggregated_mean.index[2], pd.Timestamp('2022-03-31'))
        # Check calculation for January (first 31 days)
        expected_jan_mean = self.sample_df['Value'].iloc[:31].mean()
        # Access the value using the correct DatetimeIndex
        self.assertAlmostEqual(
            aggregated_mean.loc[pd.Timestamp('2022-01-31')], expected_jan_mean)

        # Test sum aggregation
        aggregated_sum = cleaner.aggregate_monthly(
            df_unified, value_col='Value', agg_func='sum')
        self.assertIsInstance(aggregated_sum, pd.Series)
        self.assertFalse(aggregated_sum.empty)  # Check not empty on success
        self.assertEqual(len(aggregated_sum), 3)
        expected_jan_sum = self.sample_df['Value'].iloc[:31].sum()
        # Access the value using the correct DatetimeIndex
        self.assertAlmostEqual(aggregated_sum.loc[pd.Timestamp('2022-01-31')], expected_jan_sum)

    def test_aggregate_monthly_no_periodindex(self):
        """Test aggregation when index is DatetimeIndex (should be converted)."""
        df_datetime_index = self.sample_df.set_index('Date')
        aggregated_mean = cleaner.aggregate_monthly(
            df_datetime_index, value_col='Value', agg_func='mean')
        self.assertIsInstance(aggregated_mean, pd.Series)
        self.assertFalse(aggregated_mean.empty)  # Check not empty on success
        # Check if the output index is DatetimeIndex after resampling
        self.assertIsInstance(aggregated_mean.index, pd.DatetimeIndex)
        # Check for Month End frequency
        self.assertEqual(aggregated_mean.index.freqstr, 'ME')
        self.assertEqual(len(aggregated_mean), 3)

    def test_aggregate_monthly_key_error(self):
        """Test aggregation with incorrect value column."""
        df_unified = cleaner.unify_timestamps(
            self.sample_df.copy(), date_col='Date')
        aggregated = cleaner.aggregate_monthly(
            df_unified, value_col='WrongValueCol', agg_func='mean')
        self.assertIsInstance(aggregated, pd.Series)
        self.assertTrue(aggregated.empty)


if __name__ == '__main__':
    unittest.main()
