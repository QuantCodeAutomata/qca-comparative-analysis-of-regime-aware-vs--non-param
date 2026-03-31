"""
Tests for Wasserstein distance calculations.
"""

import numpy as np
import pytest
from src.wasserstein import (
    wasserstein_distance_gaussian,
    find_closest_template,
    update_template_exponential_smoothing,
    initialize_templates_kmeans
)


def test_wasserstein_distance_identical():
    """Test that Wasserstein distance is zero for identical distributions."""
    mu = np.array([0.0, 0.0])
    sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    dist = wasserstein_distance_gaussian(mu, sigma, mu, sigma)
    
    assert np.isclose(dist, 0.0, atol=1e-6)


def test_wasserstein_distance_different_means():
    """Test Wasserstein distance for distributions with different means."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([1.0, 0.0])
    sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    dist = wasserstein_distance_gaussian(mu1, sigma, mu2, sigma)
    
    # Distance should be positive
    assert dist > 0
    
    # For same covariance, distance should be at least the Euclidean distance of means
    mean_dist = np.linalg.norm(mu1 - mu2)
    assert dist >= mean_dist - 1e-6


def test_wasserstein_distance_symmetry():
    """Test that Wasserstein distance is symmetric."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([1.0, 1.0])
    sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    sigma2 = np.array([[2.0, 0.0], [0.0, 2.0]])
    
    dist1 = wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2)
    dist2 = wasserstein_distance_gaussian(mu2, sigma2, mu1, sigma1)
    
    assert np.isclose(dist1, dist2, rtol=1e-5)


def test_wasserstein_distance_positive():
    """Test that Wasserstein distance is always non-negative."""
    np.random.seed(42)
    
    for _ in range(10):
        d = np.random.randint(2, 6)
        mu1 = np.random.randn(d)
        mu2 = np.random.randn(d)
        
        # Generate random positive definite covariance matrices
        A1 = np.random.randn(d, d)
        sigma1 = A1 @ A1.T + np.eye(d)
        A2 = np.random.randn(d, d)
        sigma2 = A2 @ A2.T + np.eye(d)
        
        dist = wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2)
        
        assert dist >= 0


def test_find_closest_template():
    """Test finding closest template."""
    # Create templates
    templates = [
        (np.array([0.0, 0.0]), np.eye(2)),
        (np.array([5.0, 5.0]), np.eye(2)),
        (np.array([-5.0, -5.0]), np.eye(2))
    ]
    
    # Test point close to first template
    mu = np.array([0.1, 0.1])
    sigma = np.eye(2)
    
    idx, dist = find_closest_template(mu, sigma, templates)
    
    assert idx == 0
    assert dist >= 0


def test_update_template_exponential_smoothing():
    """Test template update with exponential smoothing."""
    template_mu = np.array([0.0, 0.0])
    template_sigma = np.eye(2)
    
    new_mu = np.array([1.0, 1.0])
    new_sigma = 2 * np.eye(2)
    
    learning_rate = 0.1
    
    updated_mu, updated_sigma = update_template_exponential_smoothing(
        template_mu, template_sigma, new_mu, new_sigma, learning_rate
    )
    
    # Check that update is between old and new
    expected_mu = 0.9 * template_mu + 0.1 * new_mu
    np.testing.assert_array_almost_equal(updated_mu, expected_mu)
    
    expected_sigma = 0.9 * template_sigma + 0.1 * new_sigma
    np.testing.assert_array_almost_equal(updated_sigma, expected_sigma)
    
    # Check that covariance is symmetric
    assert np.allclose(updated_sigma, updated_sigma.T)


def test_update_template_learning_rate_bounds():
    """Test that learning rate affects update magnitude."""
    template_mu = np.array([0.0])
    template_sigma = np.array([[1.0]])
    
    new_mu = np.array([10.0])
    new_sigma = np.array([[5.0]])
    
    # With learning rate 0, should not change
    updated_mu_0, _ = update_template_exponential_smoothing(
        template_mu, template_sigma, new_mu, new_sigma, 0.0
    )
    np.testing.assert_array_almost_equal(updated_mu_0, template_mu)
    
    # With learning rate 1, should fully update
    updated_mu_1, _ = update_template_exponential_smoothing(
        template_mu, template_sigma, new_mu, new_sigma, 1.0
    )
    np.testing.assert_array_almost_equal(updated_mu_1, new_mu)


def test_initialize_templates_kmeans():
    """Test template initialization using K-means."""
    np.random.seed(42)
    
    # Create synthetic data with 3 clusters
    n_samples = 300
    features = np.vstack([
        np.random.randn(100, 4) + np.array([0, 0, 0, 0]),
        np.random.randn(100, 4) + np.array([5, 5, 5, 5]),
        np.random.randn(100, 4) + np.array([-5, -5, -5, -5])
    ])
    
    n_templates = 3
    templates = initialize_templates_kmeans(features, n_templates, random_state=42)
    
    # Check that we get correct number of templates
    assert len(templates) == n_templates
    
    # Check that each template has mean and covariance
    for mu, sigma in templates:
        assert mu.shape == (4,)
        assert sigma.shape == (4, 4)
        
        # Check covariance is symmetric
        assert np.allclose(sigma, sigma.T)
        
        # Check covariance is positive semi-definite
        eigenvalues = np.linalg.eigvals(sigma)
        assert np.all(eigenvalues >= -1e-10)


def test_wasserstein_numerical_stability():
    """Test numerical stability with near-singular covariance matrices."""
    mu1 = np.array([0.0, 0.0])
    mu2 = np.array([0.0, 0.0])
    
    # Nearly singular covariance
    sigma1 = np.array([[1e-10, 0.0], [0.0, 1e-10]])
    sigma2 = np.array([[1e-10, 0.0], [0.0, 1e-10]])
    
    # Should not crash
    dist = wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2)
    
    assert np.isfinite(dist)
    assert dist >= 0


def test_wasserstein_high_dimensional():
    """Test Wasserstein distance in higher dimensions."""
    np.random.seed(42)
    
    d = 15  # 3 * 5 assets
    mu1 = np.random.randn(d)
    mu2 = np.random.randn(d)
    
    # Generate random positive definite covariance matrices
    A1 = np.random.randn(d, d)
    sigma1 = A1 @ A1.T + np.eye(d)
    A2 = np.random.randn(d, d)
    sigma2 = A2 @ A2.T + np.eye(d)
    
    dist = wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2)
    
    assert np.isfinite(dist)
    assert dist >= 0
