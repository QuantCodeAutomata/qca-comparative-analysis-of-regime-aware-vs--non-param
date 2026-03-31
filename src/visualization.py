"""
Visualization module for experiment results.

Creates plots for:
- Cumulative performance comparison
- Portfolio weights over time
- Turnover time series
- Effective number of positions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import os


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_cumulative_performance(
    all_results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot cumulative performance for all strategies.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary mapping strategy names to their results
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, results in all_results.items():
        returns = results['returns']
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=strategy_name, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Out-of-Sample Cumulative Performance Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative performance plot to {save_path}")
    
    plt.close()


def plot_portfolio_weights(
    results: Dict,
    strategy_name: str,
    asset_names: list,
    save_path: str = None
):
    """
    Plot portfolio weights over time (stacked area chart).
    
    Parameters
    ----------
    results : Dict
        Strategy backtest results
    strategy_name : str
        Name of the strategy
    asset_names : list
        List of asset names
    save_path : str, optional
        Path to save the plot
    """
    weights = results['weights']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.stackplot(
        weights.index,
        *[weights[asset].values for asset in weights.columns],
        labels=asset_names,
        alpha=0.8
    )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Weight', fontsize=12)
    ax.set_title(f'{strategy_name}: Portfolio Weights Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, ncol=len(asset_names))
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved weights plot to {save_path}")
    
    plt.close()


def plot_turnover(
    all_results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot turnover time series for all strategies.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary mapping strategy names to their results
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, results in all_results.items():
        turnover = results['turnover']
        if turnover.sum() > 0:  # Only plot if there's non-zero turnover
            ax.plot(turnover.index, turnover.values, label=strategy_name, alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Turnover', fontsize=12)
    ax.set_title('Portfolio Turnover Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved turnover plot to {save_path}")
    
    plt.close()


def plot_n_eff(
    all_results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot effective number of positions over time.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary mapping strategy names to their results
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, results in all_results.items():
        n_eff = results['n_eff']
        ax.plot(n_eff.index, n_eff.values, label=strategy_name, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Effective Number of Positions', fontsize=12)
    ax.set_title('Portfolio Diversification Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved N_eff plot to {save_path}")
    
    plt.close()


def plot_drawdown(
    all_results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot drawdown over time for all strategies.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary mapping strategy names to their results
    save_path : str, optional
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, results in all_results.items():
        returns = results['returns']
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.plot(drawdown.index, drawdown.values * 100, label=strategy_name, linewidth=2)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved drawdown plot to {save_path}")
    
    plt.close()


def plot_hmm_diagnostics(
    results: Dict,
    save_path: str = None
):
    """
    Plot HMM-specific diagnostics (selected K and template probabilities).
    
    Parameters
    ----------
    results : Dict
        HMM strategy backtest results
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot selected K over time
    if 'k_selected' in results:
        k_selected = results['k_selected']
        axes[0].plot(k_selected.index, k_selected.values, linewidth=2, color='steelblue')
        axes[0].set_xlabel('Date', fontsize=12)
        axes[0].set_ylabel('Number of HMM States (K)', fontsize=12)
        axes[0].set_title('Dynamic HMM Order Selection', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # Plot template probabilities over time
    if 'template_probs' in results:
        template_probs = results['template_probs']
        axes[1].stackplot(
            template_probs.index,
            *[template_probs[col].values for col in template_probs.columns],
            labels=template_probs.columns,
            alpha=0.8
        )
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel('Template Probability', fontsize=12)
        axes[1].set_title('Regime Probabilities Over Time', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=9, ncol=3)
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved HMM diagnostics plot to {save_path}")
    
    plt.close()


def create_all_plots(
    all_results: Dict[str, Dict],
    asset_names: list,
    output_dir: str = 'results'
):
    """
    Create all visualization plots.
    
    Parameters
    ----------
    all_results : Dict[str, Dict]
        Dictionary mapping strategy names to their results
    asset_names : list
        List of asset names
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Cumulative performance
    plot_cumulative_performance(
        all_results,
        save_path=os.path.join(output_dir, 'cumulative_performance.png')
    )
    
    # Drawdown
    plot_drawdown(
        all_results,
        save_path=os.path.join(output_dir, 'drawdown.png')
    )
    
    # Turnover
    plot_turnover(
        all_results,
        save_path=os.path.join(output_dir, 'turnover.png')
    )
    
    # Effective positions
    plot_n_eff(
        all_results,
        save_path=os.path.join(output_dir, 'n_eff.png')
    )
    
    # Portfolio weights for each strategy
    for strategy_name, results in all_results.items():
        safe_name = strategy_name.replace(' ', '_').replace('+', '').lower()
        plot_portfolio_weights(
            results,
            strategy_name,
            asset_names,
            save_path=os.path.join(output_dir, f'weights_{safe_name}.png')
        )
    
    # HMM diagnostics
    if 'Wasserstein HMM+MVO' in all_results:
        plot_hmm_diagnostics(
            all_results['Wasserstein HMM+MVO'],
            save_path=os.path.join(output_dir, 'hmm_diagnostics.png')
        )
    
    print("All visualizations generated!")
