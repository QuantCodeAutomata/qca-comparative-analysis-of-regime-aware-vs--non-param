"""
K-Nearest Neighbors (KNN) + MVO Strategy.

Implements a non-parametric asset allocation strategy using:
- K-nearest neighbors for moment estimation
- Ledoit-Wolf shrinkage for covariance estimation
- Mean-variance optimization with turnover penalty
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from typing import Dict
import warnings

from src.optimizer import optimize_portfolio, calculate_turnover, calculate_effective_positions


class KNNStrategy:
    """
    K-Nearest Neighbors + MVO strategy for dynamic asset allocation.
    """
    
    def __init__(
        self,
        n_assets: int,
        k_neighbors: int = 20,
        gamma: float = 1.0,
        tau: float = 0.1,
        w_max: float = 0.5,
        scale_features: bool = True
    ):
        """
        Initialize KNN strategy.
        
        Parameters
        ----------
        n_assets : int
            Number of assets
        k_neighbors : int
            Number of nearest neighbors
        gamma : float
            Risk aversion parameter for MVO
        tau : float
            Turnover penalty parameter for MVO
        w_max : float
            Maximum weight per asset
        scale_features : bool
            Whether to standardize features
        """
        self.n_assets = n_assets
        self.k_neighbors = k_neighbors
        self.gamma = gamma
        self.tau = tau
        self.w_max = w_max
        self.scale_features = scale_features
        
        self.scaler = StandardScaler() if scale_features else None
    
    def find_neighbors(
        self,
        features_history: np.ndarray,
        current_features: np.ndarray
    ) -> np.ndarray:
        """
        Find K nearest neighbors of current features in historical data.
        
        Parameters
        ----------
        features_history : np.ndarray
            Historical feature matrix (T × d)
        current_features : np.ndarray
            Current feature vector (d,)
            
        Returns
        -------
        np.ndarray
            Indices of K nearest neighbors
        """
        # Scale features if requested
        if self.scale_features:
            # Fit scaler on history
            features_scaled = self.scaler.fit_transform(features_history)
            current_scaled = self.scaler.transform(current_features.reshape(1, -1))
        else:
            features_scaled = features_history
            current_scaled = current_features.reshape(1, -1)
        
        # Find K nearest neighbors using Euclidean distance
        knn = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
        knn.fit(features_scaled)
        
        distances, indices = knn.kneighbors(current_scaled)
        
        return indices[0]
    
    def estimate_moments(
        self,
        returns_history: np.ndarray,
        neighbor_indices: np.ndarray
    ) -> tuple:
        """
        Estimate expected returns and covariance from neighbors.
        
        Parameters
        ----------
        returns_history : np.ndarray
            Historical returns (T × N)
        neighbor_indices : np.ndarray
            Indices of neighbors
            
        Returns
        -------
        tuple
            (expected_returns, covariance_matrix)
        """
        # Get returns following the neighbor observations
        # Neighbor at index i -> use return at i+1
        neighbor_returns = []
        for idx in neighbor_indices:
            if idx + 1 < len(returns_history):
                neighbor_returns.append(returns_history[idx + 1])
        
        if len(neighbor_returns) == 0:
            # Fallback: use last available returns
            warnings.warn("No valid neighbor returns, using historical mean")
            mu = np.mean(returns_history, axis=0)
            sigma = np.cov(returns_history, rowvar=False)
            return mu, sigma
        
        neighbor_returns = np.array(neighbor_returns)
        
        # Estimate expected returns as mean of neighbor returns
        mu = np.mean(neighbor_returns, axis=0)
        
        # Estimate covariance with Ledoit-Wolf shrinkage
        if len(neighbor_returns) > 1:
            try:
                lw = LedoitWolf()
                sigma = lw.fit(neighbor_returns).covariance_
            except Exception as e:
                warnings.warn(f"Ledoit-Wolf failed: {e}. Using sample covariance.")
                sigma = np.cov(neighbor_returns, rowvar=False)
        else:
            # Fallback for single neighbor
            sigma = np.eye(self.n_assets) * 0.01
        
        return mu, sigma
    
    def backtest(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        oos_start_idx: int
    ) -> Dict:
        """
        Run backtest of KNN strategy.
        
        Parameters
        ----------
        features : pd.DataFrame
            Full feature matrix
        returns : pd.DataFrame
            Full return matrix
        oos_start_idx : int
            Index where out-of-sample period starts
            
        Returns
        -------
        Dict
            Backtest results containing weights, returns, turnover, etc.
        """
        features_array = features.values
        returns_array = returns.values
        dates = features.index
        
        # Initialize results storage
        n_oos = len(features) - oos_start_idx
        weights_history = np.zeros((n_oos, self.n_assets))
        portfolio_returns = np.zeros(n_oos)
        turnover_history = np.zeros(n_oos)
        n_eff_history = np.zeros(n_oos)
        
        # Initial weights (equal-weight)
        w_prev = np.ones(self.n_assets) / self.n_assets
        
        print(f"Running KNN backtest ({n_oos} days)...")
        
        for i, t in enumerate(range(oos_start_idx, len(features))):
            if i % 50 == 0:
                print(f"  Day {i+1}/{n_oos}")
            
            # Historical data up to t-1
            features_history = features_array[:t]
            returns_history = returns_array[:t]
            
            # Current features at t-1
            current_features = features_array[t - 1]
            
            # Find K nearest neighbors
            neighbor_indices = self.find_neighbors(features_history, current_features)
            
            # Estimate moments from neighbors
            mu_t, sigma_t = self.estimate_moments(returns_history, neighbor_indices)
            
            # Optimize portfolio
            w_t = optimize_portfolio(
                mu_t, sigma_t, w_prev,
                gamma=self.gamma, tau=self.tau, w_max=self.w_max
            )
            
            # Store results
            weights_history[i] = w_t
            portfolio_returns[i] = np.dot(w_t, returns_array[t])
            turnover_history[i] = calculate_turnover(w_t, w_prev)
            n_eff_history[i] = calculate_effective_positions(w_t)
            
            # Update previous weights
            w_prev = w_t
        
        print("Backtest complete!")
        
        return {
            'weights': pd.DataFrame(
                weights_history,
                index=dates[oos_start_idx:],
                columns=returns.columns
            ),
            'returns': pd.Series(
                portfolio_returns,
                index=dates[oos_start_idx:]
            ),
            'turnover': pd.Series(
                turnover_history,
                index=dates[oos_start_idx:]
            ),
            'n_eff': pd.Series(
                n_eff_history,
                index=dates[oos_start_idx:]
            )
        }
