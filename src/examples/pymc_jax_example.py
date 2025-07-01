#!/usr/bin/env python3
"""
Example: Using PyMC with JAX backend for GPU acceleration.
Since JAX is already working with CUDA, this is the fastest way to get GPU acceleration.
"""

import numpy as np
import pymc as pm
import jax

# Verify JAX is using GPU
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Simple Bayesian linear regression
def main():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)
    true_coeffs = np.array([1.5, -2.0, 0.5])
    y = X @ true_coeffs + np.random.randn(n_samples) * 0.1
    
    # PyMC model using JAX backend
    with pm.Model() as model:
        # Priors
        coeffs = pm.Normal("coeffs", mu=0, sigma=1, shape=3)
        sigma = pm.HalfNormal("sigma", sigma=1)
        
        # Likelihood
        mu = pm.math.dot(X, coeffs)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)
        
        # Sample using JAX backend (runs on GPU!)
        print("🚀 Sampling with JAX backend (GPU acceleration)...")
        idata = pm.sample_numpyro_nuts(
            draws=1000,
            tune=500,
            chains=4,
            target_accept=0.9,
            random_seed=42
        )
    
    # Print results
    print("\n📊 Posterior Summary:")
    print(f"True coefficients: {true_coeffs}")
    print(f"Estimated coefficients: {idata.posterior.coeffs.mean(dim=['chain', 'draw']).values}")
    print(f"Estimated sigma: {idata.posterior.sigma.mean().values:.3f}")
    
    return idata

if __name__ == "__main__":
    main() 