import chex
import jax
import jax.numpy as jnp
from FABdiffME.targets.target_util import Target


class CrossedRing(Target):
    def __init__(
        self,
        dim: int = 2,
        center: chex.Array = jnp.array([0.5, 0.5]),
        radius: float = 0.25,
        sigma_ring: float = 0.02,
        mu1: float = 0,
        sigma1: float = 0.5,
        mu2: float = 0.72,
        sigma2: float = 0.02,
    ):
        """
        Based on paper https://arxiv.org/abs/2212.06172
        """
        self.dim = dim
        self.name = "crossed_ring"
        self.center = center
        self.radius = radius
        self.sigma_ring = sigma_ring
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2

    def p_line(self, samples):
        x1 = (samples[:, 0] - samples[:, 1]) / jnp.sqrt(2)
        x2 = (samples[:, 0] + samples[:, 1]) / jnp.sqrt(2)
        arg1 = -((x1 - self.mu1) ** 2) / (2 * self.sigma1**2)
        arg2 = -((x2 - self.mu2) ** 2) / (2 * self.sigma2**2)
        p_line = jnp.exp(arg1) * jnp.exp(arg2)
        # set p=0 for values outside of unit square
        p_line = jnp.where(samples[:, 0] > 1, 0, p_line)
        p_line = jnp.where(samples[:, 0] < 0, 0, p_line)
        p_line = jnp.where(samples[:, 1] > 1, 0, p_line)
        p_line = jnp.where(samples[:, 1] < 0, 0, p_line)

        return p_line

    def p_circle(self, samples):
        arg = -(
            (
                jnp.sqrt(
                    (samples[:, 0] - self.center[0]) ** 2
                    + (samples[:, 1] - self.center[1]) ** 2
                )
                - self.radius
            )
            ** 2
        ) / (2 * self.sigma_ring**2)
        return jnp.exp(arg)

    def prob(self, samples):
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        p = 1 / 2 * (self.p_line(samples) + self.p_circle(samples))
        return p.squeeze()

    def log_prob(self, samples):
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        log_prob = jnp.log(self.p_line(samples) + self.p_circle(samples)) - jnp.log(2)
        # Can have numerical instabilities once log prob is very small. Manually override to prevent this.
        # This will cause the flow will ignore regions with less than 1e-4 probability under the target.
        valid_log_prob = log_prob > -1e4
        log_prob = jnp.where(
            valid_log_prob, log_prob, -jnp.inf * jnp.ones_like(log_prob)
        )
        return log_prob.squeeze()

    def sample(self, key: chex.PRNGKey, n_samples: int = 1000) -> chex.Array:
        """
        Generate samples with rejection sampling.
        """
        n_batch = 10
        axis_1 = axis_2 = jnp.linspace(0.0, 1.0, 300)[1:-1]
        grid_unit = jnp.dstack(jnp.meshgrid(axis_1, axis_2))
        grid_unit = grid_unit.reshape([-1, 2])

        density_dp = self.prob(grid_unit)
        max_density = jnp.max(density_dp)

        # proposal distribution: Uniform dist
        p_proposal = max_density
        samples = jnp.array([])
        counter = 0
        keys = jax.random.split(key, int(1e8))
        while len(samples) < n_samples:
            # consider n_batch data points at the same time
            # sample x ~ Uniform
            keys_i = jax.random.split(keys[counter], 3)
            x1 = jax.random.uniform(keys_i[0], shape=[n_batch])
            x2 = jax.random.uniform(keys_i[1], shape=[n_batch])
            x = jnp.array([x1, x2]).T

            # sample uniformly [0, density_x]
            y = jax.random.uniform(
                keys_i[2], shape=[n_batch], minval=0.0, maxval=p_proposal
            )
            # evaluate Dalitz plot density at x
            p_dp = self.prob(x)

            # decide whether or not to keep sample
            inds = jnp.where(y < p_dp)[0]
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

        return samples
