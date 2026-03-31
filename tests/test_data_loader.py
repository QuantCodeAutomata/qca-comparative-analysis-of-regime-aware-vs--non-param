"""
Tests for data loading and feature engineering.
"""

import numpy as np
import pandas as pd
import pytest
from src.data_loader import DataLoader, get_default_tickers, split_train_test


def test_default_tickers():
    """Test that default tickers are correctly defined."""
    tickers = get_default_tickers()
    
    assert len(tickers) == 5
    assert 'SPX' in tickers
    assert 'BOND' in tickers
    assert 'GOLD' in tickers
    assert 'OIL' in tickers
    assert 'USD' in tickers


def test_data_loader_initialization():
    """Test DataLoader initialization."""
    tickers = get_default_tickers()
    loader = DataLoader(tickers, start_date="2020-01-01", end_date="2021-01-01")
    
    assert loader.n_assets == 5
    assert loader.vol_window == 60
    assert loader.mean_window == 20
    assert loader.asset_names == list(tickers.keys())


def test_log_returns_calculation():
    """Test log returns calculation."""
    # Create synthetic price data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'A': np.exp(np.cumsum(np.random.randn(100) * 0.01)),
        'B': np.exp(np.cumsum(np.random.randn(100) * 0.01))
    }, index=dates)
    
    tickers = {'A': 'A', 'B': 'B'}
    loader = DataLoader(tickers)
    returns = loader.compute_log_returns(prices)
    
    # Check shape
    assert len(returns) == len(prices) - 1
    assert returns.shape[1] == 2
    
    # Check no NaN in returns (except possibly first row which is dropped)
    assert not returns.isna().any().any()
    
    # Check returns are approximately correct
    manual_returns = np.log(prices).diff().iloc[1:]
    np.testing.assert_array_almost_equal(returns.values, manual_returns.values)


def test_rolling_volatility():
    """Test rolling volatility calculation."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.DataFrame({
        'A': np.random.randn(100) * 0.01,
        'B': np.random.randn(100) * 0.01
    }, index=dates)
    
    tickers = {'A': 'A', 'B': 'B'}
    loader = DataLoader(tickers, vol_window=20)
    volatility = loader.compute_rolling_volatility(returns, window=20)
    
    # Check that volatility is positive
    assert (volatility.dropna() > 0).all().all()
    
    # Check that first 19 values are NaN (window=20)
    assert volatility.iloc[:19].isna().all().all()


def test_rolling_mean():
    """Test rolling mean calculation."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.DataFrame({
        'A': np.random.randn(100) * 0.01,
        'B': np.random.randn(100) * 0.01
    }, index=dates)
    
    tickers = {'A': 'A', 'B': 'B'}
    loader = DataLoader(tickers, mean_window=10)
    rolling_mean = loader.compute_rolling_mean(returns, window=10)
    
    # Check that first 9 values are NaN (window=10)
    assert rolling_mean.iloc[:9].isna().all().all()
    
    # Check that rolling mean is computed correctly
    manual_mean = returns.rolling(window=10, min_periods=10).mean()
    np.testing.assert_array_almost_equal(
        rolling_mean.dropna().values,
        manual_mean.dropna().values
    )


def test_feature_preparation():
    """Test feature preparation with strict causality."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'A': np.exp(np.cumsum(np.random.randn(100) * 0.01)),
        'B': np.exp(np.cumsum(np.random.randn(100) * 0.01))
    }, index=dates)
    
    tickers = {'A': 'A', 'B': 'B'}
    loader = DataLoader(tickers, vol_window=20, mean_window=10)
    returns, volatility, rolling_mean, features = loader.prepare_features(prices)
    
    # Check that all outputs have same length
    assert len(returns) == len(volatility) == len(rolling_mean) == len(features)
    
    # Check feature dimensionality: 3N features (returns, vol, mean)
    assert features.shape[1] == 3 * 2  # 2 assets
    
    # Check no NaN in features
    assert not features.isna().any().any()
    
    # Check that features are aligned
    assert (features.index == returns.index).all()


def test_split_train_test():
    """Test train/test split."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = {
        'returns': pd.DataFrame(np.random.randn(100, 2), index=dates),
        'features': pd.DataFrame(np.random.randn(100, 6), index=dates),
        'prices': pd.DataFrame(np.random.randn(100, 2), index=dates),
        'volatility': pd.DataFrame(np.random.randn(100, 2), index=dates),
        'rolling_mean': pd.DataFrame(np.random.randn(100, 2), index=dates),
        'asset_names': ['A', 'B']
    }
    
    split_date = '2020-02-15'
    train_data, test_data = split_train_test(data, split_date)
    
    # Check that split is correct
    assert len(train_data['returns']) < len(data['returns'])
    assert len(test_data['returns']) < len(data['returns'])
    assert len(train_data['returns']) + len(test_data['returns']) == len(data['returns'])
    
    # Check that test data starts at split date
    assert test_data['returns'].index[0] >= pd.Timestamp(split_date)
    
    # Check that train data ends before split date
    assert train_data['returns'].index[-1] < pd.Timestamp(split_date)


def test_feature_causality():
    """Test that features respect causality (no lookahead bias)."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'A': np.exp(np.cumsum(np.random.randn(100) * 0.01))
    }, index=dates)
    
    tickers = {'A': 'A'}
    loader = DataLoader(tickers, vol_window=20, mean_window=10)
    returns, volatility, rolling_mean, features = loader.prepare_features(prices)
    
    # For any time t in features, the values should only depend on data up to t
    # This is implicitly tested by the rolling window calculations
    # We verify that the rolling statistics at time t use data ending at t
    
    # Check that volatility at time t uses returns up to t
    for i in range(20, len(returns)):
        manual_vol = returns.iloc[i-19:i+1].std().values[0]
        assert np.isclose(volatility.iloc[i].values[0], manual_vol, rtol=1e-5)


def test_empty_data_handling():
    """Test handling of edge cases."""
    tickers = {'A': 'A'}
    loader = DataLoader(tickers)
    
    # Test with very short data
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    prices = pd.DataFrame({'A': np.random.randn(10)}, index=dates)
    
    returns, volatility, rolling_mean, features = loader.prepare_features(prices)
    
    # With vol_window=60 and mean_window=20, we expect very few or no valid features
    # This should not crash
    assert len(features) >= 0
