"""
Wasserstein distance calculation for Gaussian distributions.

Implements the 2-Wasserstein distance between multivariate Gaussian distributions,
used for template matching in the HMM strategy.
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Tuple


def wasserstein_distance_gaussian(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """
    Compute the 2-Wasserstein distance between two Gaussian distributions.
    
    The formula for the squared 2-Wasserstein distance is:
    W_2^2 = ||μ_1 - μ_2||_2^2 + Tr(Σ_1 + Σ_2 - 2(Σ_2^{1/2} Σ_1 Σ_2^{1/2})^{1/2})
    
    Parameters
    ----------
    mu1 : np.ndarray
        Mean vector of first Gaussian (shape: d)
    sigma1 : np.ndarray
        Covariance matrix of first Gaussian (shape: d × d)
    mu2 : np.ndarray
        Mean vector of second Gaussian (shape: d)
    sigma2 : np.ndarray
        Covariance matrix of second Gaussian (shape: d × d)
    epsilon : float
        Small positive value added to diagonal for numerical stability
        
    Returns
    -------
    float
        2-Wasserstein distance W_2(N(μ_1, Σ_1), N(μ_2, Σ_2))
    """
    # Ensure inputs are numpy arrays
    mu1 = np.asarray(mu1).flatten()
    mu2 = np.asarray(mu2).flatten()
    sigma1 = np.asarray(sigma1)
    sigma2 = np.asarray(sigma2)
    
    # Add small epsilon to diagonal for numerical stability
    sigma1 = sigma1 + epsilon * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + epsilon * np.eye(sigma2.shape[0])
    
    # Mean difference term: ||μ_1 - μ_2||_2^2
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    # Covariance term: Tr(Σ_1 + Σ_2 - 2(Σ_2^{1/2} Σ_1 Σ_2^{1/2})^{1/2})
    try:
        # Compute Σ_2^{1/2}
        sqrt_sigma2 = sqrtm(sigma2)
        
        # Ensure it's real (sqrtm can return complex for numerical reasons)
        if np.iscomplexobj(sqrt_sigma2):
            sqrt_sigma2 = sqrt_sigma2.real
        
        # Compute Σ_2^{1/2} Σ_1 Σ_2^{1/2}
        product = sqrt_sigma2 @ sigma1 @ sqrt_sigma2
        
        # Compute (Σ_2^{1/2} Σ_1 Σ_2^{1/2})^{1/2}
        sqrt_product = sqrtm(product)
        
        # Ensure it's real
        if np.iscomplexobj(sqrt_product):
            sqrt_product = sqrt_product.real
        
        # Compute trace term
        cov_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(sqrt_product)
        
        # Ensure non-negative (numerical errors can make it slightly negative)
        cov_term = max(0, cov_term)
        
    except Exception as e:
        # Fallback: use simpler approximation if matrix operations fail
        import warnings
        warnings.warn(f"Matrix square root failed, using Frobenius norm approximation: {e}")
        cov_term = np.linalg.norm(sigma1 - sigma2, 'fro') ** 2
    
    # Squared Wasserstein distance
    w2_squared = mean_diff + cov_term
    
    # Return Wasserstein distance (not squared)
    return np.sqrt(max(0, w2_squared))


def find_closest_template(
    mu: np.ndarray,
    sigma: np.ndarray,
    templates: list,
    epsilon: float = 1e-8
) -> Tuple[int, float]:
    """
    Find the closest template to a given Gaussian distribution.
    
    Parameters
    ----------
    mu : np.ndarray
        Mean vector of the distribution
    sigma : np.ndarray
        Covariance matrix of the distribution
    templates : list
        List of template tuples (mu_g, sigma_g)
    epsilon : float
        Numerical stability parameter
        
    Returns
    -------
    Tuple[int, float]
        (index of closest template, distance to closest template)
    """
    distances = []
    for mu_g, sigma_g in templates:
        dist = wasserstein_distance_gaussian(mu, sigma, mu_g, sigma_g, epsilon)
        distances.append(dist)
    
    closest_idx = np.argmin(distances)
    closest_dist = distances[closest_idx]
    
    return closest_idx, closest_dist


def update_template_exponential_smoothing(
    template_mu: np.ndarray,
    template_sigma: np.ndarray,
    new_mu: np.ndarray,
    new_sigma: np.ndarray,
    learning_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update template parameters using exponential smoothing.
    
    θ_new = (1 - η) * θ_old + η * θ_observed
    
    Parameters
    ----------
    template_mu : np.ndarray
        Current template mean
    template_sigma : np.ndarray
        Current template covariance
    new_mu : np.ndarray
        Observed mean to incorporate
    new_sigma : np.ndarray
        Observed covariance to incorporate
    learning_rate : float
        Learning rate η ∈ (0, 1)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (updated_mu, updated_sigma)
    """
    updated_mu = (1 - learning_rate) * template_mu + learning_rate * new_mu
    updated_sigma = (1 - learning_rate) * template_sigma + learning_rate * new_sigma
    
    # Ensure covariance remains symmetric
    updated_sigma = (updated_sigma + updated_sigma.T) / 2
    
    return updated_mu, updated_sigma


def initialize_templates_kmeans(
    features: np.ndarray,
    n_templates: int,
    random_state: int = 42
) -> list:
    """
    Initialize templates using K-means clustering.
    
    Parameters
    ----------
    features : np.ndarray
        Feature matrix (T × d)
    n_templates : int
        Number of templates to initialize
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    list
        List of template tuples (mu_g, sigma_g)
    """
    from sklearn.cluster import KMeans
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_templates, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Compute mean and covariance for each cluster
    templates = []
    for k in range(n_templates):
        cluster_data = features[labels == k]
        if len(cluster_data) > 1:
            mu_k = np.mean(cluster_data, axis=0)
            sigma_k = np.cov(cluster_data, rowvar=False)
        else:
            # Fallback for empty or single-point clusters
            mu_k = kmeans.cluster_centers_[k]
            sigma_k = np.eye(features.shape[1])
        
        templates.append((mu_k, sigma_k))
    
    return templates
