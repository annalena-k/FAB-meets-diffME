from typing import Tuple
import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as jsps

from FABdiffME.targets.target_util import Target

Sample = chex.Array
Prob = chex.Array


def check_restriction_to_unit_hypercube(samples):
    mask_larger1 = jnp.prod(jnp.where(samples > 1.0, 0, 1), axis=1)
    mask_smaller0 = jnp.prod(jnp.where(samples < 0.0, 0, 1), axis=1)
    mask = mask_larger1 * mask_smaller0
    return mask


class Gaussian(Target):
    def __init__(
        self,
        dim: int,
        mu: chex.Array = None,
        cov: chex.Array = None,
        sigma: chex.Array = 0.1,
    ):
        self.dim = dim
        self.name = "gaussian"
        if mu == None:
            self.mu = jnp.array([0.75] * self.dim)
        else:
            self.mu = mu
        assert self.mu.shape[0] == self.dim
        if cov == None:
            self.cov = sigma * jnp.eye(self.dim)
        else:
            self.cov = cov
        assert self.cov.shape == (dim, dim)

    def prob(self, samples):
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        ps = jsps.multivariate_normal.pdf(samples, mean=self.mu, cov=self.cov)
        mask = check_restriction_to_unit_hypercube(samples)
        ps = jnp.where(mask, ps, 0.0)
        return ps.squeeze()

    def log_prob(self, samples):
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        log_prob = jsps.multivariate_normal.logpdf(samples, mean=self.mu, cov=self.cov)
        mask = check_restriction_to_unit_hypercube(samples)
        log_prob = jnp.where(mask, log_prob, -jnp.inf)
        # Can have numerical instabilities once log prob is very small. Manually override to prevent this.
        # This will cause the flow to ignore regions with a very small probability under the target.
        valid_log_prob = log_prob > -1e8
        log_prob = jnp.where(
            valid_log_prob, log_prob, -jnp.inf * jnp.ones_like(log_prob)
        )
        return log_prob.squeeze()

    def sample(self, key: chex.PRNGKey, n_samples: int = 1000) -> chex.Array:
        """
        Generate samples with rejection sampling.
        """
        n_batch = 10
        samples = jnp.array([])
        counter = 0
        count_n_target_eval = 0
        keys = jax.random.split(key, int(1e8))
        while len(samples) < n_samples:
            # consider n_batch data points at the same time
            # sample x ~ Uniform
            keys_i = jax.random.split(keys[counter], 3)
            x = jax.random.multivariate_normal(
                keys_i[0], mean=self.mu, cov=self.cov, shape=[n_batch]
            )
            count_n_target_eval += n_batch

            # decide whether samples lie in unit hypercube
            mask = check_restriction_to_unit_hypercube(x)
            inds = jnp.where(mask)[0]
            # save samples
            if samples.size == 0:
                samples = x[inds, :]
            else:
                samples = jnp.vstack([samples, x[inds, :]])

            if counter % 200 == 0:
                print(len(samples), "/", n_samples)
            if counter == 1e8:
                break
            counter += 1

        samples = jnp.array(samples[:n_samples, :])

        return samples, count_n_target_eval
