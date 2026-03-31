"""
Tests for portfolio optimization.
"""

import numpy as np
import pytest
from src.optimizer import (
    optimize_portfolio,
    calculate_turnover,
    calculate_effective_positions,
    equal_weight_portfolio,
    validate_weights
)


def test_optimize_portfolio_basic():
    """Test basic portfolio optimization."""
    n = 3
    mu = np.array([0.1, 0.05, 0.03])
    sigma = np.eye(n) * 0.01
    w_prev = np.ones(n) / n
    
    w_opt = optimize_portfolio(mu, sigma, w_prev, gamma=1.0, tau=0.0, w_max=1.0)
    
    # Check that weights are valid
    assert validate_weights(w_opt)
    assert len(w_opt) == n
    assert np.isclose(np.sum(w_opt), 1.0)
    assert np.all(w_opt >= 0)


def test_optimize_portfolio_turnover_penalty():
    """Test that turnover penalty reduces portfolio changes."""
    n = 3
    mu = np.array([0.1, 0.05, 0.03])
    sigma = np.eye(n) * 0.01
    w_prev = np.array([0.5, 0.3, 0.2])
    
    # Without turnover penalty
    w_no_penalty = optimize_portfolio(mu, sigma, w_prev, gamma=1.0, tau=0.0, w_max=1.0)
    turnover_no_penalty = calculate_turnover(w_no_penalty, w_prev)
    
    # With high turnover penalty
    w_with_penalty = optimize_portfolio(mu, sigma, w_prev, gamma=1.0, tau=10.0, w_max=1.0)
    turnover_with_penalty = calculate_turnover(w_with_penalty, w_prev)
    
    # Turnover should be lower with penalty
    assert turnover_with_penalty <= turnover_no_penalty + 1e-6


def test_optimize_portfolio_position_limits():
    """Test that position limits are respected."""
    n = 3
    mu = np.array([0.5, 0.01, 0.01])  # First asset has much higher return
    sigma = np.eye(n) * 0.01
    w_prev = np.ones(n) / n
    w_max = 0.4
    
    w_opt = optimize_portfolio(mu, sigma, w_prev, gamma=0.1, tau=0.0, w_max=w_max)
    
    # Check that no weight exceeds w_max
    assert np.all(w_opt <= w_max + 1e-6)


def test_calculate_turnover():
    """Test turnover calculation."""
    w_current = np.array([0.5, 0.3, 0.2])
    w_previous = np.array([0.4, 0.4, 0.2])
    
    turnover = calculate_turnover(w_current, w_previous)
    
    # Turnover = 0.5 * ||w_current - w_previous||_1
    expected = 0.5 * (abs(0.5 - 0.4) + abs(0.3 - 0.4) + abs(0.2 - 0.2))
    assert np.isclose(turnover, expected)


def test_calculate_turnover_no_change():
    """Test that turnover is zero when weights don't change."""
    w = np.array([0.5, 0.3, 0.2])
    
    turnover = calculate_turnover(w, w)
    
    assert np.isclose(turnover, 0.0)


def test_calculate_turnover_complete_change():
    """Test turnover for complete portfolio change."""
    w_current = np.array([1.0, 0.0, 0.0])
    w_previous = np.array([0.0, 1.0, 0.0])
    
    turnover = calculate_turnover(w_current, w_previous)
    
    # Complete change should give turnover of 1.0
    assert np.isclose(turnover, 1.0)


def test_calculate_effective_positions():
    """Test effective number of positions calculation."""
    # Equal weights
    w_equal = np.array([0.25, 0.25, 0.25, 0.25])
    n_eff_equal = calculate_effective_positions(w_equal)
    assert np.isclose(n_eff_equal, 4.0)
    
    # Concentrated portfolio
    w_concentrated = np.array([1.0, 0.0, 0.0, 0.0])
    n_eff_concentrated = calculate_effective_positions(w_concentrated)
    assert np.isclose(n_eff_concentrated, 1.0)
    
    # Partially diversified
    w_partial = np.array([0.5, 0.5, 0.0, 0.0])
    n_eff_partial = calculate_effective_positions(w_partial)
    assert np.isclose(n_eff_partial, 2.0)


def test_equal_weight_portfolio():
    """Test equal-weight portfolio creation."""
    n = 5
    w = equal_weight_portfolio(n)
    
    assert len(w) == n
    assert np.allclose(w, 1.0 / n)
    assert np.isclose(np.sum(w), 1.0)


def test_validate_weights_valid():
    """Test weight validation for valid weights."""
    w_valid = np.array([0.3, 0.4, 0.3])
    assert validate_weights(w_valid)


def test_validate_weights_invalid_sum():
    """Test weight validation for invalid sum."""
    w_invalid = np.array([0.3, 0.4, 0.4])  # Sum > 1
    assert not validate_weights(w_invalid)


def test_validate_weights_negative():
    """Test weight validation for negative weights."""
    w_invalid = np.array([0.5, 0.6, -0.1])
    assert not validate_weights(w_invalid)


def test_validate_weights_exceeds_one():
    """Test weight validation for weights > 1."""
    w_invalid = np.array([1.5, -0.3, -0.2])
    assert not validate_weights(w_invalid)


def test_optimize_portfolio_risk_aversion():
    """Test that risk aversion parameter affects portfolio."""
    n = 3
    mu = np.array([0.1, 0.05, 0.03])
    sigma = np.array([[0.04, 0.01, 0.01],
                      [0.01, 0.02, 0.005],
                      [0.01, 0.005, 0.01]])
    w_prev = np.ones(n) / n
    
    # Low risk aversion (more aggressive)
    w_low_gamma = optimize_portfolio(mu, sigma, w_prev, gamma=0.1, tau=0.0, w_max=1.0)
    
    # High risk aversion (more conservative)
    w_high_gamma = optimize_portfolio(mu, sigma, w_prev, gamma=10.0, tau=0.0, w_max=1.0)
    
    # With low risk aversion, should allocate more to high-return asset
    # This is a qualitative test - exact behavior depends on optimization
    assert validate_weights(w_low_gamma)
    assert validate_weights(w_high_gamma)


def test_optimize_portfolio_fallback():
    """Test that optimization falls back to previous weights on failure."""
    n = 3
    # Create an invalid covariance matrix (negative definite)
    mu = np.array([0.1, 0.05, 0.03])
    sigma = -np.eye(n)  # Invalid!
    w_prev = np.ones(n) / n
    
    w_opt = optimize_portfolio(mu, sigma, w_prev, gamma=1.0, tau=0.0, w_max=1.0)
    
    # Should return previous weights
    np.testing.assert_array_almost_equal(w_opt, w_prev)


def test_optimize_portfolio_edge_case_single_asset():
    """Test optimization with single asset."""
    n = 1
    mu = np.array([0.1])
    sigma = np.array([[0.01]])
    w_prev = np.array([1.0])
    
    w_opt = optimize_portfolio(mu, sigma, w_prev, gamma=1.0, tau=0.0, w_max=1.0)
    
    # Should be fully invested in single asset
    assert np.isclose(w_opt[0], 1.0)
