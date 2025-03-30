import unittest
import pandas as pd
import numpy as np
import time
import os
import sys
from io import StringIO

# Ensure the src directory is in the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.utils import helpers

# Dummy function for testing decorator
@helpers.timeit
def dummy_timed_function(duration):
    """A simple function that sleeps for a given duration."""
    time.sleep(duration)
    return "Slept for " + str(duration)

class TestHelpers(unittest.TestCase):

    def setUp(self):
        """Set up data for consistency checks."""
        self.idx1_period = pd.period_range(start='2022-01', periods=5, freq='M')
        self.df1_period = pd.DataFrame({'A': range(5), 'Common': range(5)}, index=self.idx1_period)

        self.idx2_period = pd.period_range(start='2022-01', periods=5, freq='M')
        self.df2_period = pd.DataFrame({'B': range(10, 15)}, index=self.idx2_period)

        self.idx3_datetime = pd.date_range(start='2022-01-01', periods=5, freq='MS')
        self.df3_datetime = pd.DataFrame({'C': range(20, 25)}, index=self.idx3_datetime)

        self.idx4_mismatch_freq = pd.period_range(start='2022-01', periods=5, freq='Q')
        self.df4_mismatch_freq = pd.DataFrame({'D': range(30, 35)}, index=self.idx4_mismatch_freq)

        self.df5_common_col = pd.DataFrame({'Common': range(100, 105), 'E': range(5)}, index=self.idx1_period)

        self.positive_series = pd.Series([1, 5, 10.5], name='Positive')
        self.zero_series = pd.Series([1, 0, 5], name='Zero')
        self.negative_series = pd.Series([1, -2, 5], name='Negative')

    def test_check_data_consistency_no_issues(self):
        """Test consistency check with compatible dataframes."""
        # Redirect stdout to capture print warnings (if any)
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            helpers.check_data_consistency(self.df1_period, self.df2_period)
            sys.stdout = sys.__stdout__ # Restore stdout
            output = captured_output.getvalue()
            self.assertNotIn("Warning:", output) # Expect no warnings
        except Exception as e:
            sys.stdout = sys.__stdout__
            self.fail(f"check_data_consistency raised an exception unexpectedly: {e}")

    def test_check_data_consistency_diff_index_type(self):
        """Test consistency check with different index types."""
        captured_output = StringIO()
        sys.stdout = captured_output
        helpers.check_data_consistency(self.df1_period, self.df3_datetime)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("Warning: Index types differ", output)

    def test_check_data_consistency_diff_index_freq(self):
        """Test consistency check with different index frequencies."""
        captured_output = StringIO()
        sys.stdout = captured_output
        helpers.check_data_consistency(self.df1_period, self.df4_mismatch_freq)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("Warning: Index frequencies differ", output)

    def test_check_data_consistency_common_columns(self):
        """Test consistency check with common columns."""
        captured_output = StringIO()
        sys.stdout = captured_output
        helpers.check_data_consistency(self.df1_period, self.df5_common_col)
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("Warning: Common columns found: ['Common']", output)

    def test_timeit_decorator(self):
        """Test the timeit decorator functionality."""
        sleep_duration = 0.1
        # Redirect stdout to capture print
        captured_output = StringIO()
        sys.stdout = captured_output
        
        result = dummy_timed_function(sleep_duration)
        
        sys.stdout = sys.__stdout__ # Restore stdout
        output = captured_output.getvalue()

        self.assertEqual(result, f"Slept for {sleep_duration}")
        self.assertIn(f"Function dummy_timed_function Took", output)
        self.assertIn("seconds", output)
        # Check if the reported time is roughly correct (allowing some buffer)
        try:
            # Extract the time value printed by the decorator
            time_str = output.split('Took ')[1].split(' seconds')[0]
            reported_time = float(time_str)
            self.assertGreaterEqual(reported_time, sleep_duration)
            self.assertLess(reported_time, sleep_duration + 0.1) # Allow 100ms overhead
        except (IndexError, ValueError):
            self.fail("Could not parse time from timeit decorator output.")


    def test_ensure_series_positive_true(self):
        """Test ensure_series_positive with all positive values."""
        self.assertTrue(helpers.ensure_series_positive(self.positive_series))

    def test_ensure_series_positive_with_zero(self):
        """Test ensure_series_positive with a zero value."""
        # Should print a warning
        captured_output = StringIO()
        sys.stdout = captured_output
        result = helpers.ensure_series_positive(self.zero_series, series_name="Zero Series")
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertFalse(result)
        self.assertIn("Warning: Zero Series contains non-positive values.", output)

    def test_ensure_series_positive_with_negative(self):
        """Test ensure_series_positive with a negative value."""
         # Should print a warning
        captured_output = StringIO()
        sys.stdout = captured_output
        result = helpers.ensure_series_positive(self.negative_series, series_name="Negative Series")
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertFalse(result)
        self.assertIn("Warning: Negative Series contains non-positive values.", output)


if __name__ == '__main__':
    unittest.main()