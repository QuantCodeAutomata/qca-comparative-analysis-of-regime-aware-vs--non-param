"""
Wasserstein HMM + MVO Strategy.

Implements a regime-aware asset allocation strategy using:
- Dynamic HMM order selection
- Template-based identity tracking via 2-Wasserstein distance
- Mean-variance optimization with turnover penalty
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress HMM convergence warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='hmmlearn')

from src.wasserstein import (
    wasserstein_distance_gaussian,
    find_closest_template,
    update_template_exponential_smoothing,
    initialize_templates_kmeans
)
from src.optimizer import optimize_portfolio, calculate_turnover, calculate_effective_positions


class WassersteinHMMStrategy:
    """
    Wasserstein HMM + MVO strategy for dynamic asset allocation.
    """
    
    def __init__(
        self,
        n_assets: int,
        n_templates: int = 6,
        k_min: int = 2,
        k_max: int = 8,
        order_selection_freq: int = 5,
        validation_window: int = 60,
        complexity_penalty: float = 0.1,
        template_learning_rate: float = 0.05,
        gamma: float = 1.0,
        tau: float = 0.1,
        w_max: float = 0.5,
        random_state: int = 42
    ):
        """
        Initialize Wasserstein HMM strategy.
        
        Parameters
        ----------
        n_assets : int
            Number of assets
        n_templates : int
            Number of persistent templates G
        k_min : int
            Minimum number of HMM states
        k_max : int
            Maximum number of HMM states
        order_selection_freq : int
            Frequency of order selection (in days)
        validation_window : int
            Size of validation window for order selection
        complexity_penalty : float
            Penalty for model complexity in order selection
        template_learning_rate : float
            Learning rate η for template updates
        gamma : float
            Risk aversion parameter for MVO
        tau : float
            Turnover penalty parameter for MVO
        w_max : float
            Maximum weight per asset
        random_state : int
            Random seed
        """
        self.n_assets = n_assets
        self.n_features = 3 * n_assets  # [returns, volatility, mean]
        self.n_templates = n_templates
        self.k_min = k_min
        self.k_max = k_max
        self.order_selection_freq = order_selection_freq
        self.validation_window = validation_window
        self.complexity_penalty = complexity_penalty
        self.template_learning_rate = template_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.w_max = w_max
        self.random_state = random_state
        
        # State variables
        self.templates: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.current_k: Optional[int] = None
        self.current_hmm: Optional[hmm.GaussianHMM] = None
        
    def initialize_templates(self, features: np.ndarray):
        """
        Initialize persistent templates using K-means.
        
        Parameters
        ----------
        features : np.ndarray
            Initial feature matrix for template initialization
        """
        self.templates = initialize_templates_kmeans(
            features,
            self.n_templates,
            self.random_state
        )
        print(f"Initialized {self.n_templates} templates")
    
    def select_hmm_order(
        self,
        features_train: np.ndarray,
        features_val: np.ndarray
    ) -> int:
        """
        Select optimal HMM order using validation log-likelihood.
        
        Parameters
        ----------
        features_train : np.ndarray
            Training features
        features_val : np.ndarray
            Validation features
            
        Returns
        -------
        int
            Optimal number of states K
        """
        best_k = self.k_min
        best_score = -np.inf
        
        for k in range(self.k_min, self.k_max + 1):
            try:
                # Fit HMM with k states
                model = hmm.GaussianHMM(
                    n_components=k,
                    covariance_type='diag',  # Use diagonal for speed
                    n_iter=20,  # Reduce iterations
                    tol=1e-2,  # Looser tolerance
                    random_state=self.random_state,
                    verbose=False
                )
                model.fit(features_train)
                
                # Compute validation log-likelihood
                val_ll = model.score(features_val)
                
                # Apply complexity penalty
                score = val_ll - self.complexity_penalty * k
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                warnings.warn(f"HMM fitting failed for k={k}: {e}")
                continue
        
        return best_k
    
    def fit_hmm(self, features: np.ndarray, n_states: int) -> hmm.GaussianHMM:
        """
        Fit Gaussian HMM on features.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        n_states : int
            Number of HMM states
            
        Returns
        -------
        hmm.GaussianHMM
            Fitted HMM model
        """
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',  # Use diagonal for speed
            n_iter=20,  # Reduce iterations
            tol=1e-2,  # Looser tolerance
            random_state=self.random_state,
            verbose=False
        )
        
        try:
            model.fit(features)
        except Exception as e:
            warnings.warn(f"HMM fitting failed: {e}")
            # Return previous model if available
            if self.current_hmm is not None:
                return self.current_hmm
            raise
        
        return model
    
    def predict_component_probabilities(
        self,
        model: hmm.GaussianHMM,
        features_history: np.ndarray
    ) -> np.ndarray:
        """
        Predict component probabilities for next time step.
        
        Uses forward algorithm to compute P(z_t = k | F_{t-1}).
        
        Parameters
        ----------
        model : hmm.GaussianHMM
            Fitted HMM
        features_history : np.ndarray
            Historical features up to t-1
            
        Returns
        -------
        np.ndarray
            Component probabilities (K,)
        """
        try:
            # Get filtered state probabilities at t-1
            _, state_probs = model.score_samples(features_history)
            last_state_prob = state_probs[-1]
            
            # Check for NaN
            if np.any(np.isnan(last_state_prob)):
                # Fallback to uniform distribution
                return np.ones(model.n_components) / model.n_components
            
            # Predict next state using transition matrix
            next_state_prob = last_state_prob @ model.transmat_
            
            # Check for NaN in result
            if np.any(np.isnan(next_state_prob)):
                return np.ones(model.n_components) / model.n_components
            
            return next_state_prob
            
        except Exception as e:
            warnings.warn(f"Failed to predict component probabilities: {e}")
            # Fallback to uniform distribution
            return np.ones(model.n_components) / model.n_components
    
    def map_components_to_templates(
        self,
        model: hmm.GaussianHMM
    ) -> np.ndarray:
        """
        Map HMM components to persistent templates.
        
        For each component k, find template g with minimum Wasserstein distance.
        
        Parameters
        ----------
        model : hmm.GaussianHMM
            Fitted HMM
            
        Returns
        -------
        np.ndarray
            Mapping array g(k) of shape (K,)
        """
        k = model.n_components
        mapping = np.zeros(k, dtype=int)
        
        for component_idx in range(k):
            mu_k = model.means_[component_idx]
            sigma_k = model.covars_[component_idx]
            
            # Convert diagonal covariance to full matrix if needed
            if sigma_k.ndim == 1:
                sigma_k = np.diag(sigma_k)
            
            template_idx, _ = find_closest_template(
                mu_k, sigma_k, self.templates
            )
            mapping[component_idx] = template_idx
        
        return mapping
    
    def aggregate_template_probabilities(
        self,
        component_probs: np.ndarray,
        mapping: np.ndarray
    ) -> np.ndarray:
        """
        Aggregate component probabilities to template probabilities.
        
        p_{t,g} = Σ_{k: g(k)=g} p_{t,k}
        
        Parameters
        ----------
        component_probs : np.ndarray
            Component probabilities (K,)
        mapping : np.ndarray
            Component-to-template mapping (K,)
            
        Returns
        -------
        np.ndarray
            Template probabilities (G,)
        """
        template_probs = np.zeros(self.n_templates)
        
        for k, g in enumerate(mapping):
            template_probs[g] += component_probs[k]
        
        return template_probs
    
    def update_templates(
        self,
        model: hmm.GaussianHMM,
        mapping: np.ndarray
    ):
        """
        Update template parameters using exponential smoothing.
        
        Parameters
        ----------
        model : hmm.GaussianHMM
            Fitted HMM
        mapping : np.ndarray
            Component-to-template mapping
        """
        # For each template, compute average of assigned components
        for g in range(self.n_templates):
            assigned_components = np.where(mapping == g)[0]
            
            if len(assigned_components) > 0:
                # Average parameters of assigned components
                avg_mu = np.mean([model.means_[k] for k in assigned_components], axis=0)
                
                # Handle diagonal covariance
                covars_list = []
                for k in assigned_components:
                    cov_k = model.covars_[k]
                    if cov_k.ndim == 1:
                        cov_k = np.diag(cov_k)
                    covars_list.append(cov_k)
                avg_sigma = np.mean(covars_list, axis=0)
                
                # Update template with exponential smoothing
                mu_g, sigma_g = self.templates[g]
                updated_mu, updated_sigma = update_template_exponential_smoothing(
                    mu_g, sigma_g, avg_mu, avg_sigma, self.template_learning_rate
                )
                self.templates[g] = (updated_mu, updated_sigma)
    
    def aggregate_moments(
        self,
        template_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute expected returns and covariance for assets.
        
        μ_t = Σ_g p_{t,g} μ_g[1:N]
        Σ_t = Σ_g p_{t,g} Σ_g[1:N, 1:N]
        
        Parameters
        ----------
        template_probs : np.ndarray
            Template probabilities (G,)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (expected_returns, covariance_matrix)
        """
        mu_agg = np.zeros(self.n_assets)
        sigma_agg = np.zeros((self.n_assets, self.n_assets))
        
        for g, prob in enumerate(template_probs):
            mu_g, sigma_g = self.templates[g]
            
            # Extract asset return moments (first N elements/block)
            mu_agg += prob * mu_g[:self.n_assets]
            
            # For diagonal covariance, extract diagonal elements
            if sigma_g.ndim == 1:
                # Diagonal covariance stored as 1D array
                sigma_block = np.diag(sigma_g[:self.n_assets])
            else:
                sigma_block = sigma_g[:self.n_assets, :self.n_assets]
                # Ensure it's symmetric
                sigma_block = (sigma_block + sigma_block.T) / 2
            
            sigma_agg += prob * sigma_block
        
        # Ensure covariance is symmetric and positive definite
        sigma_agg = (sigma_agg + sigma_agg.T) / 2
        
        # Add small regularization to ensure positive definiteness
        sigma_agg = sigma_agg + 1e-6 * np.eye(self.n_assets)
        
        # Final symmetry check
        sigma_agg = (sigma_agg + sigma_agg.T) / 2
        
        return mu_agg, sigma_agg
    
    def backtest(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        oos_start_idx: int
    ) -> Dict:
        """
        Run backtest of Wasserstein HMM strategy.
        
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
        
        # Initialize templates using initial training data
        init_features = features_array[:oos_start_idx]
        self.initialize_templates(init_features)
        
        # Initialize results storage
        n_oos = len(features) - oos_start_idx
        weights_history = np.zeros((n_oos, self.n_assets))
        portfolio_returns = np.zeros(n_oos)
        turnover_history = np.zeros(n_oos)
        n_eff_history = np.zeros(n_oos)
        k_history = np.zeros(n_oos, dtype=int)
        template_probs_history = np.zeros((n_oos, self.n_templates))
        
        # Initial weights (equal-weight)
        w_prev = np.ones(self.n_assets) / self.n_assets
        
        print(f"Running Wasserstein HMM backtest ({n_oos} days)...")
        
        for i, t in enumerate(range(oos_start_idx, len(features))):
            if i % 50 == 0:
                print(f"  Day {i+1}/{n_oos}")
            
            # Historical data up to t-1
            features_history = features_array[:t]
            
            # Model order selection (periodic)
            if i % self.order_selection_freq == 0 or self.current_k is None:
                # Split into train and validation
                if len(features_history) > self.validation_window:
                    features_train = features_history[:-self.validation_window]
                    features_val = features_history[-self.validation_window:]
                    
                    self.current_k = self.select_hmm_order(features_train, features_val)
                else:
                    self.current_k = self.k_min
            
            k_history[i] = self.current_k
            
            # Fit HMM
            self.current_hmm = self.fit_hmm(features_history, self.current_k)
            
            # Predict component probabilities
            component_probs = self.predict_component_probabilities(
                self.current_hmm, features_history
            )
            
            # Map components to templates
            mapping = self.map_components_to_templates(self.current_hmm)
            
            # Aggregate to template probabilities
            template_probs = self.aggregate_template_probabilities(
                component_probs, mapping
            )
            template_probs_history[i] = template_probs
            
            # Update templates
            self.update_templates(self.current_hmm, mapping)
            
            # Aggregate moments for assets
            mu_t, sigma_t = self.aggregate_moments(template_probs)
            
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
            ),
            'k_selected': pd.Series(
                k_history,
                index=dates[oos_start_idx:]
            ),
            'template_probs': pd.DataFrame(
                template_probs_history,
                index=dates[oos_start_idx:],
                columns=[f'Template_{g}' for g in range(self.n_templates)]
            )
        }
