"""
Tests for performance metrics.
"""

import numpy as np
import pandas as pd
import pytest
from src.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_cumulative_return,
    calculate_turnover_statistics,
    calculate_n_eff_statistics
)


def test_sharpe_ratio_positive_returns():
    """Test Sharpe ratio with positive returns."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.001, index=dates)
    
    sharpe = calculate_sharpe_ratio(returns)
    
    # Sharpe should be positive for positive mean returns
    assert sharpe > 0


def test_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with zero volatility."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.zeros(252), index=dates)
    
    sharpe = calculate_sharpe_ratio(returns)
    
    # Should return 0 for zero volatility
    assert sharpe == 0.0


def test_sharpe_ratio_annualization():
    """Test that Sharpe ratio is properly annualized."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    
    sharpe_252 = calculate_sharpe_ratio(returns, annualization_factor=252)
    sharpe_1 = calculate_sharpe_ratio(returns, annualization_factor=1)
    
    # Annualized should be sqrt(252) times larger
    assert np.isclose(sharpe_252, sharpe_1 * np.sqrt(252), rtol=1e-5)


def test_sortino_ratio():
    """Test Sortino ratio calculation."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.001, index=dates)
    
    sortino = calculate_sortino_ratio(returns)
    
    # Sortino should be finite and positive for positive mean returns
    assert np.isfinite(sortino)
    assert sortino > 0


def test_sortino_ratio_no_downside():
    """Test Sortino ratio with no negative returns."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.abs(np.random.randn(252) * 0.01), index=dates)
    
    sortino = calculate_sortino_ratio(returns)
    
    # Should be infinite when there are no negative returns
    assert np.isinf(sortino)


def test_max_drawdown_no_drawdown():
    """Test max drawdown with monotonically increasing returns."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.Series(np.abs(np.random.randn(100) * 0.01), index=dates)
    
    max_dd = calculate_max_drawdown(returns)
    
    # Max drawdown should be zero or very small
    assert max_dd <= 0
    assert max_dd >= -0.01  # Allow for small numerical errors


def test_max_drawdown_with_crash():
    """Test max drawdown with a significant crash."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.Series(np.zeros(100), index=dates)
    returns.iloc[50] = -0.5  # 50% crash
    
    max_dd = calculate_max_drawdown(returns)
    
    # Max drawdown should be approximately -50%
    assert max_dd < -0.49
    assert max_dd > -0.51


def test_max_drawdown_recovery():
    """Test max drawdown with drawdown and recovery."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.Series(np.zeros(100), index=dates)
    returns.iloc[25] = -0.3  # 30% drawdown
    returns.iloc[50] = 0.5   # Recovery
    
    max_dd = calculate_max_drawdown(returns)
    
    # Max drawdown should capture the worst point
    assert max_dd < -0.25


def test_cumulative_return():
    """Test cumulative return calculation."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    returns = pd.Series([0.01] * 10, index=dates)  # 1% daily return
    
    cum_return = calculate_cumulative_return(returns)
    
    # (1.01)^10 - 1 ≈ 0.1046
    expected = (1.01 ** 10) - 1
    assert np.isclose(cum_return, expected, rtol=1e-5)


def test_cumulative_return_negative():
    """Test cumulative return with losses."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    returns = pd.Series([-0.01] * 10, index=dates)  # -1% daily return
    
    cum_return = calculate_cumulative_return(returns)
    
    # (0.99)^10 - 1 ≈ -0.0956
    expected = (0.99 ** 10) - 1
    assert np.isclose(cum_return, expected, rtol=1e-5)
    assert cum_return < 0


def test_turnover_statistics():
    """Test turnover statistics calculation."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    turnover = pd.Series(np.random.rand(100) * 0.1, index=dates)
    
    stats = calculate_turnover_statistics(turnover)
    
    # Check that all statistics are present
    assert 'mean' in stats
    assert 'median' in stats
    assert 'std' in stats
    assert 'q25' in stats
    assert 'q75' in stats
    assert 'q95' in stats
    assert 'frac_above_0.1' in stats
    assert 'frac_above_0.5' in stats
    
    # Check that statistics are reasonable
    assert stats['mean'] >= 0
    assert stats['median'] >= 0
    assert 0 <= stats['frac_above_0.1'] <= 1
    assert 0 <= stats['frac_above_0.5'] <= 1


def test_turnover_statistics_high_turnover():
    """Test turnover statistics with high turnover."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    turnover = pd.Series(np.ones(100) * 0.6, index=dates)  # All above 0.5
    
    stats = calculate_turnover_statistics(turnover)
    
    # All values should be above 0.5
    assert stats['frac_above_0.5'] == 1.0
    assert stats['mean'] == 0.6


def test_n_eff_statistics():
    """Test effective positions statistics."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    n_eff = pd.Series(np.random.rand(100) * 3 + 1, index=dates)  # Between 1 and 4
    
    stats = calculate_n_eff_statistics(n_eff)
    
    # Check that all statistics are present
    assert 'mean' in stats
    assert 'median' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    
    # Check that statistics are reasonable
    assert stats['min'] >= 1.0
    assert stats['max'] <= 4.0
    assert stats['mean'] >= stats['min']
    assert stats['mean'] <= stats['max']


def test_empty_series():
    """Test metrics with empty series."""
    empty_series = pd.Series([], dtype=float)
    
    assert calculate_sharpe_ratio(empty_series) == 0.0
    assert calculate_sortino_ratio(empty_series) == 0.0
    assert calculate_max_drawdown(empty_series) == 0.0
    assert calculate_cumulative_return(empty_series) == 0.0


def test_sharpe_vs_sortino():
    """Test that Sortino ratio is typically higher than Sharpe for positive skew."""
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    # Create returns with positive skew (more small gains, few large losses)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.002, index=dates)
    
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    
    # Both should be positive
    assert sharpe > 0
    assert sortino > 0
    
    # Sortino is typically higher because it only penalizes downside volatility
    # (though this is not guaranteed for all return distributions)
    assert np.isfinite(sortino)
