# Comparative Analysis of Regime-Aware vs. Non-Parametric Dynamic Asset Allocation

This repository implements a comprehensive experiment comparing a novel dynamic asset allocation strategy based on a Wasserstein Hidden Markov Model (HMM) against a non-parametric K-Nearest Neighbors (KNN) approach and passive benchmarks.

## Overview

The experiment validates that a regime-aware allocation strategy using a causally-inferred, dynamically-sized, and identity-stabilized HMM improves implementability and downside control at a daily frequency.

## Strategies Implemented

1. **Wasserstein HMM + MVO**: Regime-aware strategy using HMM with template-based identity tracking via 2-Wasserstein distance
2. **KNN + MVO**: Non-parametric strategy using K-nearest neighbors for moment estimation
3. **Equal-Weight Benchmark**: Static equal allocation across all assets
4. **SPX Buy & Hold Benchmark**: 100% allocation to S&P 500

## Asset Universe

- SPX (S&P 500) - Proxy: SPY
- BOND (Broad Bond) - Proxy: AGG
- GOLD - Proxy: GLD
- OIL - Proxy: USO
- USD (U.S. Dollar) - Proxy: UUP

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and feature engineering
│   ├── wasserstein.py          # Wasserstein distance calculation
│   ├── hmm_strategy.py         # Wasserstein HMM + MVO strategy
│   ├── knn_strategy.py         # KNN + MVO strategy
│   ├── benchmarks.py           # Benchmark strategies
│   ├── optimizer.py            # MVO optimization with turnover penalty
│   ├── metrics.py              # Performance metrics calculation
│   └── visualization.py        # Plotting and visualization
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_wasserstein.py
│   ├── test_hmm_strategy.py
│   ├── test_knn_strategy.py
│   ├── test_optimizer.py
│   └── test_metrics.py
├── run_experiment.py           # Main experiment runner
└── results/
    └── RESULTS.md              # Experiment results and metrics
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full experiment:

```bash
python run_experiment.py
```

Results will be saved in the `results/` directory.

## Key Features

- **Strict Causality**: All features use only information available up to t-1 for decisions at time t
- **Dynamic HMM Order Selection**: Automatic selection of optimal number of states
- **Template-Based Identity Tracking**: Persistent regime identification using Wasserstein distance
- **Turnover Penalty**: Realistic transaction cost modeling in portfolio optimization
- **Comprehensive Testing**: Full test suite validating methodology adherence

## Expected Outcomes

- **Wasserstein HMM+MVO**: Sharpe ≈ 2.18, Max Drawdown ≈ -5.43%, Avg Turnover ≈ 0.0079
- **KNN+MVO**: Sharpe ≈ 1.81, Max Drawdown ≈ -12.52%, Avg Turnover ≈ 0.5665
- **Equal-Weight**: Sharpe ≈ 1.59, Max Drawdown ≈ -9.87%
- **SPX Buy & Hold**: Sharpe ≈ 1.18, Max Drawdown ≈ -14.62%

## References

This implementation strictly follows the methodology described in the reference paper on regime-aware dynamic asset allocation.
