import unittest
import pandas as pd
import os
import sys
from datetime import datetime

# Ensure the src directory is in the Python path
# This allows importing modules from src like 'data_processing.loader'
# Adjust the path based on where the test script is run from
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.data_processing import loader

class TestLoader(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.fixture_dir = os.path.join(project_root, 'tests', 'fixtures')
        self.dummy_temp_path = os.path.join(self.fixture_dir, 'dummy_temp.csv')
        self.dummy_mortality_path = os.path.join(self.fixture_dir, 'dummy_mortality.csv')
        self.dummy_dtp_path = os.path.join(self.fixture_dir, 'dummy_dtp.csv')
        self.non_existent_path = os.path.join(self.fixture_dir, 'non_existent.csv')

        # Expected structure after loading dummy_temp.csv
        self.expected_temp_dates = pd.to_datetime([
            '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-15',
            '2022-02-01', '2022-02-02', '2022-02-15', '2022-03-01'
        ])
        self.expected_temp_values = [-1.7, -6.1, -7.6, -1.8, -1.3, -3.0, -0.3, -2.2]

        # Expected structure after loading dummy_mortality.csv
        self.expected_mortality_dates = pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'])
        self.expected_mortality_values = [1000, 950, 1050, 980]

        # Expected structure after loading dummy_dtp.csv
        self.expected_dtp_dates = pd.to_datetime(['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'])
        self.expected_dtp_values = [537, 1038, 1682, 2321]


    def test_load_temperature_data_success(self):
        """Test successful loading of temperature data."""
        df = loader.load_temperature_data(self.dummy_temp_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertListEqual(list(df.columns), ['Date', 'Temperature', 'Precipitation'])
        self.assertEqual(len(df), len(self.expected_temp_dates))
        # Compare values directly, ignore index name difference
        pd.testing.assert_index_equal(pd.Index(df['Date']), pd.Index(self.expected_temp_dates), check_names=False)
        pd.testing.assert_series_equal(df['Temperature'], pd.Series(self.expected_temp_values, name='Temperature'), check_dtype=False)

    def test_load_temperature_data_file_not_found(self):
        """Test loading non-existent temperature file."""
        df = loader.load_temperature_data(self.non_existent_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_load_secondary_data_mortality_success(self):
        """Test successful loading of mortality data."""
        df = loader.load_secondary_data(self.dummy_mortality_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertListEqual(list(df.columns), ['Date', 'Mortality'])
        self.assertEqual(len(df), len(self.expected_mortality_dates))
        # Compare values directly, ignore index name difference
        pd.testing.assert_index_equal(pd.Index(df['Date']), pd.Index(self.expected_mortality_dates), check_names=False)
        pd.testing.assert_series_equal(df['Mortality'], pd.Series(self.expected_mortality_values, name='Mortality'), check_dtype=False)

    def test_load_secondary_data_dtp_success(self):
        """Test successful loading of DTP data."""
        df = loader.load_secondary_data(self.dummy_dtp_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertListEqual(list(df.columns), ['Date', 'DTP', 'Deaths', 'Injured'])
        self.assertEqual(len(df), len(self.expected_dtp_dates))
        # Compare values directly, ignore index name difference
        pd.testing.assert_index_equal(pd.Index(df['Date']), pd.Index(self.expected_dtp_dates), check_names=False)
        pd.testing.assert_series_equal(df['DTP'], pd.Series(self.expected_dtp_values, name='DTP'), check_dtype=False)

    def test_load_secondary_data_file_not_found(self):
        """Test loading non-existent secondary file."""
        df = loader.load_secondary_data(self.non_existent_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_load_secondary_data_unknown_type(self):
        """Test loading an unknown secondary file type (using temp file path)."""
        df = loader.load_secondary_data(self.dummy_temp_path) # Pass temp path to trigger unknown type
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_load_all_data(self):
        """Test loading both datasets together."""
        df_temp, df_secondary = loader.load_all_data(self.dummy_temp_path, self.dummy_mortality_path)
        self.assertIsInstance(df_temp, pd.DataFrame)
        self.assertFalse(df_temp.empty)
        self.assertIsInstance(df_secondary, pd.DataFrame)
        self.assertFalse(df_secondary.empty)
        self.assertEqual(len(df_temp), 8)
        self.assertEqual(len(df_secondary), 4)

if __name__ == '__main__':
    unittest.main()