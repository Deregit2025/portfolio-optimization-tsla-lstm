import unittest
from src import data_loader
from pathlib import Path
import pandas as pd

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Path to sample CSV for testing
        self.sample_file = Path("data/raw/TSLA.csv")

    def test_load_csv_file_exists(self):
        """Test that the file exists and can be loaded"""
        df = data_loader.load_csv(self.sample_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0, "DataFrame should not be empty")

    def test_columns_present(self):
        """Test that expected columns exist"""
        df = data_loader.load_csv(self.sample_file)
        expected_cols = ['Price', 'Close', 'High', 'Low', 'Open', 'Volume']
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_index_is_datetime(self):
        """Test that the index is datetime"""
        df = data_loader.load_csv(self.sample_file)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df.index))

if __name__ == "__main__":
    unittest.main()
