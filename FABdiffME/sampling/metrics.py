import chex
import jax
import jax.numpy as jnp

def log_sampling_efficiency(log_weights: chex.Array) -> chex.Array:
    """
    Numerically stable computation of log of sampling efficiency epsilon.
    epsilon := 1/N (sum_i weight_i)^2 / (sum_i weight_i^2) and so working in terms of logs
    log epsilon = 2 log sum_i (log exp log weight_i) - log sum_i (exp 2 log weight_i ) - log (N)

    Args:
        log_weights: Array of shape (num_batch). log of normalized weights.
    Returns:
        Scalar log efficiency.
    """
    n_samples = log_weights.shape[0]
    first_term = 2. * jax.scipy.special.logsumexp(log_weights)
    second_term = jax.scipy.special.logsumexp(2.*log_weights)
    
    return first_term - second_term - jnp.log(n_samples)

def sampling_efficiency(weights: chex.Array) -> chex.Array:
    """
    Computation of sampling efficiency epsilon, unstable version.
    Preferrable to use log_sampling_efficiency()

    Args:
        weights: Array of shape (num_batch). Normalized weights.
    Returns:
        Scalar efficiency.
    """
    n_samples = weights.shape[0]
    first_term = jnp.sum(weights)**2
    second_term = jnp.sum(weights**2)
    
    return 1/n_samples * first_term / second_term

def log_unweighting_efficiency(log_weights: chex.Array) -> chex.Array:
    """
    Numerically stable computation of log of unweighting efficiency.
    epsilon_u := mean(weights) / max(weights) = 1/N sum_i weight_i / max(weights) and so working in terms of logs
    log epsilon_u = - log (N) + log sum_i (log exp log weight_i) - max(log weights)

    Args:
        log_weights: Array of shape (num_batch). log of normalized weights.
    Returns:
        Scalar log efficiency.
    """
    n_samples = log_weights.shape[0]
    first_term = jax.scipy.special.logsumexp(log_weights)
    second_term = jnp.max(log_weights)
    
    return first_term - second_term - jnp.log(n_samples)

def unweighting_efficiency(weights: chex.Array) -> chex.Array:
    """
    Computation of unweighting efficiency epsilon_uw, unstable version.
    Preferrable to use log_unweighting_efficiency()

    Args:
        weights: Array of shape (num_batch). Normalized weights.
    Returns:
        Scalar efficiency.
    """
    n_samples = weights.shape[0]
    first_term = jnp.sum(weights) / n_samples
    second_term = jnp.max(weights)
    
    return first_term / second_term
