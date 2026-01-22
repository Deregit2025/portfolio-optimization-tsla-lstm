import unittest
from src import preprocessing
from pathlib import Path
import pandas as pd

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_file = Path("data/raw/TSLA.csv")
        self.df = preprocessing.load_and_clean(self.sample_file)

    def test_returns_column_created(self):
        """Test that 'Return' column is created"""
        df = preprocessing.calculate_returns(self.df)
        self.assertIn('Return', df.columns)
        self.assertFalse(df['Return'].isnull().all(), "'Return' column should not be all NaN")

    def test_rolling_volatility_created(self):
        """Test that rolling volatility column is created"""
        df = preprocessing.rolling_volatility(self.df)
        self.assertIn('RollingVol', df.columns)
        self.assertFalse(df['RollingVol'].isnull().all(), "'RollingVol' should not be all NaN")

    def test_stationarity_check_runs(self):
        """Test that stationarity function runs without error"""
        try:
            preprocessing.check_stationarity(df=self.df['Close'], ticker='TSLA')
        except Exception as e:
            self.fail(f"check_stationarity raised Exception unexpectedly: {e}")

    def test_var_and_sharpe_calculation(self):
        """Test VaR and Sharpe ratio calculation"""
        var = preprocessing.calculate_var(self.df, alpha=0.05)
        sharpe = preprocessing.calculate_sharpe_ratio(self.df)
        self.assertIsInstance(var, float)
        self.assertIsInstance(sharpe, float)

if __name__ == "__main__":
    unittest.main()
