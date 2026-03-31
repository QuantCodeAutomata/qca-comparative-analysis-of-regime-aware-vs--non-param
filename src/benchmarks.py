"""
Benchmark strategies for comparison.

Implements:
- Equal-Weight (EW): Static 1/N allocation
- SPX Buy & Hold (BH): 100% allocation to S&P 500
"""

import numpy as np
import pandas as pd
from typing import Dict


class EqualWeightBenchmark:
    """
    Equal-weight benchmark strategy.
    
    Maintains constant 1/N allocation across all assets.
    """
    
    def __init__(self, n_assets: int):
        """
        Initialize equal-weight benchmark.
        
        Parameters
        ----------
        n_assets : int
            Number of assets
        """
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets
    
    def backtest(
        self,
        returns: pd.DataFrame,
        oos_start_idx: int
    ) -> Dict:
        """
        Run backtest of equal-weight strategy.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Full return matrix
        oos_start_idx : int
            Index where out-of-sample period starts
            
        Returns
        -------
        Dict
            Backtest results
        """
        returns_array = returns.values
        dates = returns.index
        
        # Out-of-sample period
        n_oos = len(returns) - oos_start_idx
        oos_returns = returns_array[oos_start_idx:]
        
        # Compute portfolio returns
        portfolio_returns = oos_returns @ self.weights
        
        # Weights are constant
        weights_history = np.tile(self.weights, (n_oos, 1))
        
        # Turnover is zero (no rebalancing)
        turnover_history = np.zeros(n_oos)
        
        # Effective positions
        n_eff = 1.0 / np.sum(self.weights ** 2)
        n_eff_history = np.full(n_oos, n_eff)
        
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


class SPXBuyHoldBenchmark:
    """
    SPX Buy & Hold benchmark strategy.
    
    Maintains 100% allocation to S&P 500 (first asset).
    """
    
    def __init__(self, n_assets: int, spx_index: int = 0):
        """
        Initialize SPX buy & hold benchmark.
        
        Parameters
        ----------
        n_assets : int
            Number of assets
        spx_index : int
            Index of SPX asset in the asset list (default: 0)
        """
        self.n_assets = n_assets
        self.spx_index = spx_index
        self.weights = np.zeros(n_assets)
        self.weights[spx_index] = 1.0
    
    def backtest(
        self,
        returns: pd.DataFrame,
        oos_start_idx: int
    ) -> Dict:
        """
        Run backtest of SPX buy & hold strategy.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Full return matrix
        oos_start_idx : int
            Index where out-of-sample period starts
            
        Returns
        -------
        Dict
            Backtest results
        """
        returns_array = returns.values
        dates = returns.index
        
        # Out-of-sample period
        n_oos = len(returns) - oos_start_idx
        oos_returns = returns_array[oos_start_idx:]
        
        # Compute portfolio returns (just SPX returns)
        portfolio_returns = oos_returns @ self.weights
        
        # Weights are constant
        weights_history = np.tile(self.weights, (n_oos, 1))
        
        # Turnover is zero (no rebalancing)
        turnover_history = np.zeros(n_oos)
        
        # Effective positions (concentrated in one asset)
        n_eff = 1.0
        n_eff_history = np.full(n_oos, n_eff)
        
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
