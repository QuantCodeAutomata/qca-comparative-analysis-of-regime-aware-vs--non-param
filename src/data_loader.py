"""
Data loading and feature engineering module.

This module handles:
- Loading daily adjusted close prices from Yahoo Finance
- Computing log returns with strict causality
- Computing rolling volatility (60-day) and rolling mean (20-day)
- Ensuring proper alignment and handling of missing data
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, List
import warnings


class DataLoader:
    """
    Load and prepare financial data for asset allocation strategies.
    
    Implements strict causality: features at time t-1 are used for decisions at time t.
    """
    
    def __init__(
        self,
        tickers: Dict[str, str],
        start_date: str = "2005-01-01",
        end_date: str = "2026-12-31",
        vol_window: int = 60,
        mean_window: int = 20
    ):
        """
        Initialize data loader.
        
        Parameters
        ----------
        tickers : Dict[str, str]
            Mapping of asset names to ticker symbols
            e.g., {'SPX': 'SPY', 'BOND': 'AGG', 'GOLD': 'GLD', 'OIL': 'USO', 'USD': 'UUP'}
        start_date : str
            Start date for data download (YYYY-MM-DD)
        end_date : str
            End date for data download (YYYY-MM-DD)
        vol_window : int
            Rolling window for volatility calculation (default: 60 days)
        mean_window : int
            Rolling window for mean calculation (default: 20 days)
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.vol_window = vol_window
        self.mean_window = mean_window
        self.asset_names = list(tickers.keys())
        self.n_assets = len(self.asset_names)
        
    def load_prices(self) -> pd.DataFrame:
        """
        Load adjusted close prices for all tickers.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and asset names as columns
        """
        print(f"Loading price data for {len(self.tickers)} assets...")
        
        prices_list = []
        for asset_name, ticker in self.tickers.items():
            try:
                data = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False
                )
                if len(data) == 0:
                    warnings.warn(f"No data returned for {ticker}")
                    continue
                    
                if 'Adj Close' in data.columns:
                    price_series = data['Adj Close']
                else:
                    price_series = data['Close']
                
                # Ensure it's a Series
                if isinstance(price_series, pd.DataFrame):
                    price_series = price_series.iloc[:, 0]
                
                price_series.name = asset_name
                prices_list.append(price_series)
                print(f"  {asset_name} ({ticker}): {len(price_series)} days")
            except Exception as e:
                warnings.warn(f"Failed to load {ticker}: {e}")
                
        # Combine into single DataFrame
        prices = pd.concat(prices_list, axis=1)
        
        # Forward fill missing values (holidays, non-trading days)
        prices = prices.fillna(method='ffill')
        
        # Drop any remaining NaN rows
        prices = prices.dropna()
        
        print(f"Total aligned trading days: {len(prices)}")
        return prices
    
    def compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily log returns.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
            
        Returns
        -------
        pd.DataFrame
            Log returns: r_t = log(P_t) - log(P_{t-1})
        """
        log_prices = np.log(prices)
        returns = log_prices.diff()
        return returns.iloc[1:]  # Drop first NaN row
    
    def compute_rolling_volatility(
        self,
        returns: pd.DataFrame,
        window: int = None
    ) -> pd.DataFrame:
        """
        Compute rolling volatility (standard deviation of returns).
        
        Parameters
        ----------
        returns : pd.DataFrame
            Log returns
        window : int, optional
            Rolling window size (default: self.vol_window)
            
        Returns
        -------
        pd.DataFrame
            Rolling volatility for each asset
        """
        if window is None:
            window = self.vol_window
            
        volatility = returns.rolling(window=window, min_periods=window).std()
        return volatility
    
    def compute_rolling_mean(
        self,
        returns: pd.DataFrame,
        window: int = None
    ) -> pd.DataFrame:
        """
        Compute rolling mean of returns.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Log returns
        window : int, optional
            Rolling window size (default: self.mean_window)
            
        Returns
        -------
        pd.DataFrame
            Rolling mean for each asset
        """
        if window is None:
            window = self.mean_window
            
        rolling_mean = returns.rolling(window=window, min_periods=window).mean()
        return rolling_mean
    
    def prepare_features(
        self,
        prices: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare all features with strict causality enforcement.
        
        The feature vector x_{t-1} = [r_{t-1}; σ_{t-1}; m_{t-1}] is used
        for making decisions at time t.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (returns, volatility, rolling_mean, features)
            - returns: Daily log returns (r_t)
            - volatility: 60-day rolling volatility (σ_t)
            - rolling_mean: 20-day rolling mean (m_t)
            - features: Combined feature matrix [r_t; σ_t; m_t] of size (T, 3N)
        """
        # Compute returns
        returns = self.compute_log_returns(prices)
        
        # Compute rolling statistics
        volatility = self.compute_rolling_volatility(returns)
        rolling_mean = self.compute_rolling_mean(returns)
        
        # Combine features: [returns, volatility, rolling_mean]
        # Each has shape (T, N), so combined is (T, 3N)
        features = pd.concat([returns, volatility, rolling_mean], axis=1)
        
        # Drop rows with NaN (from rolling windows)
        features = features.dropna()
        
        # Ensure all DataFrames are aligned
        common_index = features.index
        returns = returns.loc[common_index]
        volatility = volatility.loc[common_index]
        rolling_mean = rolling_mean.loc[common_index]
        
        print(f"Features prepared: {len(features)} days, {features.shape[1]} features (3 × {self.n_assets} assets)")
        
        return returns, volatility, rolling_mean, features
    
    def load_and_prepare(self) -> Dict:
        """
        Load prices and prepare all features.
        
        Returns
        -------
        Dict
            Dictionary containing:
            - 'prices': Raw price data
            - 'returns': Daily log returns
            - 'volatility': Rolling volatility
            - 'rolling_mean': Rolling mean
            - 'features': Combined feature matrix
            - 'asset_names': List of asset names
        """
        prices = self.load_prices()
        returns, volatility, rolling_mean, features = self.prepare_features(prices)
        
        return {
            'prices': prices,
            'returns': returns,
            'volatility': volatility,
            'rolling_mean': rolling_mean,
            'features': features,
            'asset_names': self.asset_names
        }


def get_default_tickers() -> Dict[str, str]:
    """
    Get default ticker mapping for the 5-asset universe.
    
    Returns
    -------
    Dict[str, str]
        Mapping of asset names to ticker symbols
    """
    return {
        'SPX': 'SPY',    # S&P 500 ETF
        'BOND': 'AGG',   # iShares Core U.S. Aggregate Bond ETF
        'GOLD': 'GLD',   # SPDR Gold Shares
        'OIL': 'USO',    # United States Oil Fund
        'USD': 'UUP'     # Invesco DB US Dollar Index Bullish Fund
    }


def split_train_test(
    data: Dict,
    oos_start_date: str
) -> Tuple[Dict, Dict]:
    """
    Split data into training (in-sample) and testing (out-of-sample) periods.
    
    Parameters
    ----------
    data : Dict
        Data dictionary from load_and_prepare()
    oos_start_date : str
        Start date for out-of-sample period (YYYY-MM-DD)
        
    Returns
    -------
    Tuple[Dict, Dict]
        (train_data, test_data) dictionaries with same structure as input
    """
    oos_start = pd.Timestamp(oos_start_date)
    
    train_data = {}
    test_data = {}
    
    for key in ['prices', 'returns', 'volatility', 'rolling_mean', 'features']:
        df = data[key]
        train_data[key] = df[df.index < oos_start]
        test_data[key] = df[df.index >= oos_start]
    
    train_data['asset_names'] = data['asset_names']
    test_data['asset_names'] = data['asset_names']
    
    print(f"Train period: {train_data['returns'].index[0]} to {train_data['returns'].index[-1]} ({len(train_data['returns'])} days)")
    print(f"Test period: {test_data['returns'].index[0]} to {test_data['returns'].index[-1]} ({len(test_data['returns'])} days)")
    
    return train_data, test_data
