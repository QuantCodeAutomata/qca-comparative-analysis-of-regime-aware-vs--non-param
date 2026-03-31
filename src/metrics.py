"""
Performance metrics calculation.

Implements standard portfolio performance metrics:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Cumulative Returns
- Turnover statistics
"""

import numpy as np
import pandas as pd
from typing import Dict


def calculate_sharpe_ratio(
    returns: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Sharpe = (mean_return / std_return) * sqrt(annualization_factor)
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    annualization_factor : int
        Number of trading days per year (default: 252)
        
    Returns
    -------
    float
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    annualization_factor: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Sortino = (mean_return / downside_std) * sqrt(annualization_factor)
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
    annualization_factor : int
        Number of trading days per year (default: 252)
        
    Returns
    -------
    float
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    mean_return = returns.mean()
    
    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = downside_returns.std()
    if downside_std == 0:
        return np.inf
    
    sortino = (mean_return / downside_std) * np.sqrt(annualization_factor)
    return sortino


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        Maximum drawdown (negative value, e.g., -0.15 for 15% drawdown)
    """
    if len(returns) == 0:
        return 0.0
    
    # Compute cumulative returns
    cumulative = (1 + returns).cumprod()
    
    # Compute running maximum
    running_max = cumulative.expanding().max()
    
    # Compute drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_dd = drawdown.min()
    
    return max_dd


def calculate_cumulative_return(returns: pd.Series) -> float:
    """
    Calculate cumulative return.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns
        
    Returns
    -------
    float
        Cumulative return
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).prod() - 1
    return cumulative


def calculate_cumulative_log_return(returns: pd.Series) -> float:
    """
    Calculate cumulative log return.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns
        
    Returns
    -------
    float
        Cumulative log return
    """
    if len(returns) == 0:
        return 0.0
    
    return returns.sum()


def calculate_turnover_statistics(turnover: pd.Series) -> Dict:
    """
    Calculate turnover statistics.
    
    Parameters
    ----------
    turnover : pd.Series
        Daily turnover values
        
    Returns
    -------
    Dict
        Dictionary of turnover statistics
    """
    if len(turnover) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'q25': 0.0,
            'q75': 0.0,
            'q95': 0.0,
            'frac_above_0.1': 0.0,
            'frac_above_0.5': 0.0
        }
    
    return {
        'mean': turnover.mean(),
        'median': turnover.median(),
        'std': turnover.std(),
        'min': turnover.min(),
        'max': turnover.max(),
        'q25': turnover.quantile(0.25),
        'q75': turnover.quantile(0.75),
        'q95': turnover.quantile(0.95),
        'frac_above_0.1': (turnover > 0.1).mean(),
        'frac_above_0.5': (turnover > 0.5).mean()
    }


def calculate_n_eff_statistics(n_eff: pd.Series) -> Dict:
    """
    Calculate effective number of positions statistics.
    
    Parameters
    ----------
    n_eff : pd.Series
        Daily effective number of positions
        
    Returns
    -------
    Dict
        Dictionary of N_eff statistics
    """
    if len(n_eff) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }
    
    return {
        'mean': n_eff.mean(),
        'median': n_eff.median(),
        'std': n_eff.std(),
        'min': n_eff.min(),
        'max': n_eff.max()
    }


def calculate_all_metrics(results: Dict) -> Dict:
    """
    Calculate all performance metrics for a strategy.
    
    Parameters
    ----------
    results : Dict
        Strategy backtest results
        
    Returns
    -------
    Dict
        Dictionary of all metrics
    """
    returns = results['returns']
    turnover = results['turnover']
    n_eff = results['n_eff']
    
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'cumulative_return': calculate_cumulative_return(returns),
        'cumulative_log_return': calculate_cumulative_log_return(returns),
        'annualized_return': returns.mean() * 252,
        'annualized_volatility': returns.std() * np.sqrt(252),
        'turnover': calculate_turnover_statistics(turnover),
        'n_eff': calculate_n_eff_statistics(n_eff)
    }
    
    return metrics


def create_performance_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create performance comparison table for all strategies.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary mapping strategy names to their results
        
    Returns
    -------
    pd.DataFrame
        Performance comparison table
    """
    rows = []
    
    for strategy_name, results in all_results.items():
        metrics = calculate_all_metrics(results)
        
        row = {
            'Strategy': strategy_name,
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Sortino Ratio': metrics['sortino_ratio'],
            'Max Drawdown (%)': metrics['max_drawdown'] * 100,
            'Cumulative Return (%)': metrics['cumulative_return'] * 100,
            'Ann. Return (%)': metrics['annualized_return'] * 100,
            'Ann. Volatility (%)': metrics['annualized_volatility'] * 100,
            'Avg Turnover': metrics['turnover']['mean'],
            'Avg N_eff': metrics['n_eff']['mean']
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Strategy')
    
    return df


def create_turnover_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create detailed turnover statistics table.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary mapping strategy names to their results
        
    Returns
    -------
    pd.DataFrame
        Turnover statistics table
    """
    rows = []
    
    for strategy_name, results in all_results.items():
        turnover = results['turnover']
        stats = calculate_turnover_statistics(turnover)
        
        row = {
            'Strategy': strategy_name,
            'Mean': stats['mean'],
            'Median': stats['median'],
            'Std': stats['std'],
            'Q25': stats['q25'],
            'Q75': stats['q75'],
            'Q95': stats['q95'],
            'Frac > 0.1': stats['frac_above_0.1'],
            'Frac > 0.5': stats['frac_above_0.5']
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Strategy')
    
    return df
