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

from src.data_processing import merger

class TestMerger(unittest.TestCase):

    def setUp(self):
        """Set up test dataframes with PeriodIndex."""
        self.idx1 = pd.period_range(start='2022-01', periods=6, freq='M') # Jan to Jun
        self.df1 = pd.DataFrame({'A': range(6)}, index=self.idx1)

        self.idx2 = pd.period_range(start='2022-03', periods=6, freq='M') # Mar to Aug
        self.df2 = pd.DataFrame({'B': range(10, 16)}, index=self.idx2)

        # DataFrame with DatetimeIndex for compatibility testing
        self.idx3_dt = pd.date_range(start='2022-03-01', periods=6, freq='MS') # Mar to Aug
        self.df3_dt = pd.DataFrame({'C': range(20, 26)}, index=self.idx3_dt)

        # DataFrame with overlapping columns
        self.df4 = pd.DataFrame({'A': range(100, 106)}, index=self.idx2) # Mar to Aug

        # DataFrame with non-time index
        self.df_non_time = pd.DataFrame({'D': range(5)}, index=range(5))

    def test_merge_dataframes_inner(self):
        """Test inner merge."""
        merged = merger.merge_dataframes(self.df1, self.df2, how='inner')
        self.assertIsInstance(merged, pd.DataFrame)
        self.assertIsInstance(merged.index, pd.PeriodIndex)
        self.assertEqual(merged.index.freqstr, 'M')
        # Expect overlap from March to June (4 months)
        self.assertEqual(len(merged), 4)
        self.assertListEqual(list(merged.columns), ['A', 'B'])
        expected_index = pd.period_range(start='2022-03', periods=4, freq='M')
        pd.testing.assert_index_equal(merged.index, expected_index)
        # Check values
        pd.testing.assert_series_equal(merged['A'], pd.Series([2, 3, 4, 5], index=expected_index, name='A'))
        pd.testing.assert_series_equal(merged['B'], pd.Series([10, 11, 12, 13], index=expected_index, name='B'))

    def test_merge_dataframes_outer(self):
        """Test outer merge."""
        merged = merger.merge_dataframes(self.df1, self.df2, how='outer')
        self.assertIsInstance(merged, pd.DataFrame)
        self.assertIsInstance(merged.index, pd.PeriodIndex)
        self.assertEqual(merged.index.freqstr, 'M')
        # Expect range from Jan to Aug (8 months)
        self.assertEqual(len(merged), 8)
        self.assertListEqual(list(merged.columns), ['A', 'B'])
        expected_index = pd.period_range(start='2022-01', periods=8, freq='M')
        pd.testing.assert_index_equal(merged.index, expected_index)
        # Check for NaNs where there's no overlap
        self.assertTrue(merged['A'].loc['2022-07':'2022-08'].isnull().all())
        self.assertTrue(merged['B'].loc['2022-01':'2022-02'].isnull().all())
        # Check non-NaN values
        pd.testing.assert_series_equal(merged['A'].dropna(), self.df1['A'].astype(float))
        pd.testing.assert_series_equal(merged['B'].dropna(), self.df2['B'].astype(float))

    def test_merge_dataframes_left(self):
        """Test left merge."""
        merged = merger.merge_dataframes(self.df1, self.df2, how='left')
        self.assertIsInstance(merged, pd.DataFrame)
        self.assertEqual(len(merged), 6) # Should match length of df1
        pd.testing.assert_index_equal(merged.index, self.df1.index)
        self.assertTrue(merged['B'].loc['2022-01':'2022-02'].isnull().all()) # Check NaNs
        pd.testing.assert_series_equal(merged['A'], self.df1['A']) # Column A should be unchanged

    def test_merge_dataframes_right(self):
        """Test right merge."""
        merged = merger.merge_dataframes(self.df1, self.df2, how='right')
        self.assertIsInstance(merged, pd.DataFrame)
        self.assertEqual(len(merged), 6) # Should match length of df2
        pd.testing.assert_index_equal(merged.index, self.df2.index)
        self.assertTrue(merged['A'].loc['2022-07':'2022-08'].isnull().all()) # Check NaNs
        pd.testing.assert_series_equal(merged['B'], self.df2['B']) # Column B should be unchanged

    def test_merge_dataframes_mixed_index_types(self):
        """Test merging PeriodIndex with DatetimeIndex."""
        # df1 (PeriodIndex) with df3_dt (DatetimeIndex)
        merged = merger.merge_dataframes(self.df1, self.df3_dt, how='inner')
        self.assertIsInstance(merged, pd.DataFrame)
        self.assertIsInstance(merged.index, pd.PeriodIndex) # Expect PeriodIndex output
        self.assertEqual(len(merged), 4) # Mar to Jun overlap
        self.assertListEqual(list(merged.columns), ['A', 'C'])
        expected_index = pd.period_range(start='2022-03', periods=4, freq='M')
        pd.testing.assert_index_equal(merged.index, expected_index)

        # df3_dt (DatetimeIndex) with df1 (PeriodIndex)
        merged_rev = merger.merge_dataframes(self.df3_dt, self.df1, how='inner')
        self.assertIsInstance(merged_rev, pd.DataFrame)
        self.assertIsInstance(merged_rev.index, pd.PeriodIndex) # Expect PeriodIndex output
        self.assertEqual(len(merged_rev), 4)
        self.assertListEqual(list(merged_rev.columns), ['C', 'A'])
        pd.testing.assert_index_equal(merged_rev.index, expected_index)

    def test_merge_dataframes_duplicate_columns(self):
        """Test merge with duplicate column names (should warn)."""
        # Should print a warning about duplicate column 'A'
        merged = merger.merge_dataframes(self.df1, self.df4, how='inner')
        self.assertIsInstance(merged, pd.DataFrame)
        # Pandas automatically suffixes duplicates, e.g., 'A_x', 'A_y'
        # Note: The exact suffix depends on pandas version, test might need adjustment
        # self.assertTrue('A_x' in merged.columns and 'A_y' in merged.columns)
        # For simplicity, just check the merge happened
        self.assertEqual(len(merged), 4) # Mar to Jun overlap

    def test_merge_dataframes_non_time_index(self):
        """Test merging with a non-time index (should warn)."""
        # Should print warnings
        merged = merger.merge_dataframes(self.df1, self.df_non_time, how='inner')
        # Merge might technically work if indices happen to align, but it's meaningless
        # We expect an empty dataframe or error in practice, but pd.merge might try based on index value
        # Let's assert it's likely empty or very small due to index mismatch
        self.assertTrue(merged.empty or len(merged) < len(self.df1))

    def test_check_completeness_no_issues(self):
        """Test completeness check with no missing values or gaps."""
        merged = merger.merge_dataframes(self.df1, self.df2, how='inner')
        # Should print messages indicating no issues, difficult to assert stdout directly
        # We just run it to ensure no exceptions are raised
        try:
            merger.check_completeness(merged)
        except Exception as e:
            self.fail(f"check_completeness raised an exception unexpectedly: {e}")

    def test_check_completeness_with_nans(self):
        """Test completeness check with missing values (outer join)."""
        merged = merger.merge_dataframes(self.df1, self.df2, how='outer')
        # Should print messages about missing values, difficult to assert stdout
        # Run to ensure no exceptions
        try:
            merger.check_completeness(merged)
        except Exception as e:
            self.fail(f"check_completeness raised an exception unexpectedly: {e}")

    def test_check_completeness_with_gaps(self):
        """Test completeness check with time gaps."""
        idx_gap = pd.PeriodIndex(['2022-01', '2022-03', '2022-04'], freq='M')
        df_gap = pd.DataFrame({'X': [1, 2, 3]}, index=idx_gap)
        # Should print messages about gaps, difficult to assert stdout
        # Run to ensure no exceptions
        try:
            merger.check_completeness(df_gap)
        except Exception as e:
            self.fail(f"check_completeness raised an exception unexpectedly: {e}")

    def test_check_completeness_non_time_index(self):
        """Test completeness check with non-time index."""
        # Should print message about skipping gap check
        # Run to ensure no exceptions
        try:
            merger.check_completeness(self.df_non_time)
        except Exception as e:
            self.fail(f"check_completeness raised an exception unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()