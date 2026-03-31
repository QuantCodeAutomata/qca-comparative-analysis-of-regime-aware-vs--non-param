"""
Mean-Variance Optimization (MVO) with turnover penalty.

Implements the portfolio optimization problem:
    max_{w_t} μ_t^T w_t - γ w_t^T Σ_t w_t - τ ||w_t - w_{t-1}||_1
    subject to:
        1^T w_t = 1 (fully invested)
        w_t ≥ 0 (long-only)
        ||w_t||_∞ ≤ w_max (position limits)
"""

import numpy as np
import cvxpy as cp
from typing import Optional
import warnings


def optimize_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    gamma: float = 1.0,
    tau: float = 0.0,
    w_max: float = 1.0,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Solve the mean-variance optimization problem with turnover penalty.
    
    Parameters
    ----------
    mu : np.ndarray
        Expected return vector (N,)
    sigma : np.ndarray
        Covariance matrix (N × N)
    w_prev : np.ndarray
        Previous portfolio weights (N,)
    gamma : float
        Risk aversion parameter (default: 1.0)
    tau : float
        Turnover penalty parameter (default: 0.0)
    w_max : float
        Maximum weight per asset (default: 1.0)
    epsilon : float
        Small value added to diagonal for numerical stability
        
    Returns
    -------
    np.ndarray
        Optimal portfolio weights (N,)
    """
    n = len(mu)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma)
    w_prev = np.asarray(w_prev).flatten()
    
    # Ensure covariance is symmetric FIRST
    sigma = (sigma + sigma.T) / 2
    
    # Add small epsilon to diagonal for numerical stability
    sigma = sigma + epsilon * np.eye(n)
    
    # Final symmetry enforcement
    sigma = (sigma + sigma.T) / 2
    
    # Verify symmetry with tight tolerance
    max_asym = np.max(np.abs(sigma - sigma.T))
    if max_asym > 1e-10:
        warnings.warn(f"Covariance matrix asymmetry: {max_asym}")
        sigma = (sigma + sigma.T) / 2
    
    # Define optimization variable
    w = cp.Variable(n)
    
    # Objective: maximize μ^T w - γ w^T Σ w - τ ||w - w_prev||_1
    # Use quad_form with assume_symmetric=True
    returns = mu @ w
    risk = cp.quad_form(w, sigma, assume_PSD=True)
    turnover = cp.norm1(w - w_prev)
    
    objective = cp.Maximize(returns - gamma * risk - tau * turnover)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,           # Fully invested
        w >= 0,                    # Long-only
        w <= w_max                 # Position limits
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            warnings.warn(f"Optimization status: {problem.status}. Using previous weights.")
            return w_prev
        
        return w.value
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using previous weights.")
        return w_prev


def calculate_turnover(w_current: np.ndarray, w_previous: np.ndarray) -> float:
    """
    Calculate portfolio turnover.
    
    Turnover = 0.5 * ||w_t - w_{t-1}||_1
    
    Parameters
    ----------
    w_current : np.ndarray
        Current weights
    w_previous : np.ndarray
        Previous weights
        
    Returns
    -------
    float
        Turnover value
    """
    return 0.5 * np.sum(np.abs(w_current - w_previous))


def calculate_effective_positions(w: np.ndarray) -> float:
    """
    Calculate effective number of positions.
    
    N_eff = 1 / Σ_i (w_i)^2
    
    Parameters
    ----------
    w : np.ndarray
        Portfolio weights
        
    Returns
    -------
    float
        Effective number of positions
    """
    return 1.0 / np.sum(w ** 2)


def equal_weight_portfolio(n_assets: int) -> np.ndarray:
    """
    Create equal-weight portfolio.
    
    Parameters
    ----------
    n_assets : int
        Number of assets
        
    Returns
    -------
    np.ndarray
        Equal weights (1/N for each asset)
    """
    return np.ones(n_assets) / n_assets


def validate_weights(w: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Validate portfolio weights.
    
    Checks:
    - Sum to 1 (fully invested)
    - All non-negative (long-only)
    - All between 0 and 1
    
    Parameters
    ----------
    w : np.ndarray
        Portfolio weights
    tol : float
        Tolerance for sum constraint
        
    Returns
    -------
    bool
        True if weights are valid
    """
    if not np.allclose(np.sum(w), 1.0, atol=tol):
        return False
    if np.any(w < -tol):
        return False
    if np.any(w > 1.0 + tol):
        return False
    return True
