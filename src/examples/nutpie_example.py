#!/usr/bin/env python3
"""
Example: Using Nutpie for GPU-accelerated Bayesian inference.
Nutpie is a pure JAX backend that completely bypasses PyTensor.
"""

import numpy as np
import jax
import jax.numpy as jnp
import nutpie

# Verify JAX is using GPU
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

def log_likelihood(params, data):
    """Log likelihood function for linear regression."""
    coeffs, log_sigma = params[:-1], params[-1]
    sigma = jnp.exp(log_sigma)
    
    X, y = data
    mu = X @ coeffs
    return jnp.sum(jax.scipy.stats.norm.logpdf(y, mu, sigma))

def log_prior(params):
    """Log prior function."""
    coeffs, log_sigma = params[:-1], params[-1]
    
    # Normal priors on coefficients
    prior_coeffs = jnp.sum(jax.scipy.stats.norm.logpdf(coeffs, 0, 1))
    
    # Half-normal prior on sigma (log-normal on log_sigma)
    prior_sigma = jax.scipy.stats.norm.logpdf(log_sigma, 0, 1)
    
    return prior_coeffs + prior_sigma

def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)
    true_coeffs = np.array([1.5, -2.0, 0.5])
    y = X @ true_coeffs + np.random.randn(n_samples) * 0.1
    
    data = (X, y)
    
    # Define the posterior
    def log_posterior(params):
        return log_likelihood(params, data) + log_prior(params)
    
    # Initial parameters
    n_params = 4  # 3 coefficients + 1 log_sigma
    init_params = np.random.randn(n_params) * 0.1
    
    print("🚀 Sampling with Nutpie (pure JAX backend, GPU acceleration)...")
    
    # Sample using Nutpie
    sampler = nutpie.sample(
        logp=log_posterior,
        cores=4,
        chains=4,
        draws=1000,
        tune=500,
        target_accept=0.9,
        seed=42,
        progress_bar=True
    )
    
    # Extract results
    samples = sampler.to_arviz()
    
    # Print results
    print("\n📊 Posterior Summary:")
    print(f"True coefficients: {true_coeffs}")
    
    posterior_mean = samples.posterior.mean(dim=['chain', 'draw'])
    print(f"Estimated coefficients: {posterior_mean.values[:-1]}")  # exclude log_sigma
    print(f"Estimated sigma: {np.exp(posterior_mean.values[-1]):.3f}")
    
    return samples

if __name__ == "__main__":
    main() 