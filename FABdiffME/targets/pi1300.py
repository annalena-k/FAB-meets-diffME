from typing import Tuple
import chex
import jax
from jax import numpy as jnp
from jax import random as random

from FABdiffME.targets.target_util import HEP2DTarget, kallen, sqrtkallen

Arr = chex.Array
Sample = chex.Array
Prob = chex.Array
Det = chex.Array

# Scalar decay pi(1300) -> 3 pi
# Phase space definition from: https://compwa-org--129.org.readthedocs.build/report/017.html


class PI1300(HEP2DTarget):
    def __init__(
        self,
        dim: int = 2,
        MASSES: Arr = jnp.array([1.3, 0.14, 0.14, 0.14]),
        R1: Arr = jnp.array([1.0, 0.77, 0.15]),  # [c, M, GAMMA],
        R2: Arr = jnp.array([1.0, 0.77, 0.15]),  # [c, M, GAMMA],
    ):
        self.m0, self.m1, self.m2, self.m3 = MASSES
        self.R1 = R1
        self.R2 = R2
        self.dim = dim
        self.name = "pi1300"

    def matrix_element(self, s12: Arr, s23: Arr) -> Arr:
        c1, M1, gamma1 = self.R1
        c3, M3, gamma3 = self.R2
        denominator1 = M1**2 - s12 - 1j * M1 * gamma1
        me1 = c1 * M1 * gamma1 * jnp.where(denominator1 == 0, 0, 1 / denominator1)
        denominator2 = M3**2 - s23 - 1j * M3 * gamma3
        me2 = c3 * M3 * gamma3 * jnp.where(denominator2 == 0, 0, 1 / denominator2)
        return me1 + me2

    def I(self, s12: Arr, s23: Arr) -> Arr:
        I = jnp.abs(self.matrix_element(s12, s23)) ** 2
        return I

    def kibble(self, s1: Arr, s2: Arr, s3: Arr) -> Arr:
        a = kallen(self.m0**2, s1, self.m1**2)
        b = kallen(self.m0**2, s2, self.m2**2)
        c = kallen(self.m0**2, s3, self.m3**2)
        return kallen(a, b, c)

    def is_physical(self, s12: Arr, s23: Arr) -> Arr:
        MASSES = jnp.array([self.m0, self.m1, self.m2, self.m3])
        s31 = jnp.sum(MASSES**2) - s12 - s23
        physical_region = self.kibble(s12, s23, s31)
        physical_region = jnp.where(physical_region <= 0, 1.0, 0.0)
        return physical_region

    def is_physical_nan(self, s12: Arr, s23: Arr) -> Arr:
        MASSES = jnp.array([self.m0, self.m1, self.m2, self.m3])
        s31 = jnp.sum(MASSES**2) - s12 - s23
        physical_region = self.kibble(s12, s23, s31)
        physical_region = jnp.where(physical_region <= 0, 1, jnp.nan)
        return physical_region

    def get_decay_rate_nan(self, samples: Sample) -> Prob:
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        s12vs = samples[:, 0]
        s23vs = samples[:, 1]
        phys_reg = self.is_physical_nan(s12vs, s23vs)

        Is = self.I(s12vs, s23vs)
        Is_final = Is * phys_reg

        return Is_final

    def get_decay_rate(self, samples: Sample) -> Prob:
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        s12vs = samples[:, 0]
        s23vs = samples[:, 1]
        phys_reg = self.is_physical(s12vs, s23vs)

        Is = self.I(s12vs, s23vs)
        Is_final = Is * phys_reg

        return Is_final

    def get_cos_theta_from_s12(self, samples: Sample) -> Tuple[Sample, Det]:
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        s12, s23 = samples[:, 0], samples[:, 1]
        physical_region = self.is_physical(s12, s23)

        def calc_cos_theta_from_s12(s12: Arr, s23: Arr) -> Arr:
            nom = 2 * s23 * (self.m1**2 + self.m2**2 - s12) + (
                self.m0**2 - s23 - self.m1**2
            ) * (s23 + self.m2**2 - self.m3**2)
            denominator = sqrtkallen(self.m0**2, s23, self.m1**2) * sqrtkallen(
                s23, self.m2**2, self.m3**2
            )
            cos_theta = nom * jnp.where(denominator == 0, 0, 1 / denominator)
            return cos_theta

        cos_theta, dcds = jax.vmap(
            jax.value_and_grad(calc_cos_theta_from_s12, argnums=0)
        )(s12, s23)

        samples_out = jnp.dstack([cos_theta * physical_region, s23])
        samples_out = samples_out.reshape([-1, 2])
        if len(samples_out.shape) == 1:
            samples_out = jnp.expand_dims(samples_out, axis=0)

        return samples_out, dcds

    def get_s12_from_cos_theta(self, samples: Sample) -> Tuple[Sample, Det]:
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        c, s23 = samples[:, 0], samples[:, 1]

        def calc_s12_from_cos_theta(c, s23):
            s12 = (
                self.m1**2
                + self.m2**2
                + (self.m0**2 - s23 - self.m1**2)
                * (s23 + self.m2**2 - self.m3**2)
                * jnp.where(s23 == 0, 0, 1 / (2 * s23))
                - sqrtkallen(self.m0**2, s23, self.m1**2)
                * sqrtkallen(s23, self.m2**2, self.m3**2)
                * jnp.where(s23 == 0, 0, 1 / (2 * s23))
                * c
            )
            return s12

        s12, dsdc = jax.vmap(jax.value_and_grad(calc_s12_from_cos_theta, argnums=0))(
            c, s23
        )

        samples_out = jnp.dstack([s12, s23])
        samples_out = samples_out.reshape([-1, 2])
        if len(samples_out.shape) == 1:
            samples_out = jnp.expand_dims(samples_out, axis=0)

        return samples_out, dsdc

    def get_decay_rate_square_nan(self, samples: Sample) -> Prob:
        samples_square, dsdc = self.get_s12_from_cos_theta(samples)
        phys_reg = self.is_physical(samples_square[:, 0], samples_square[:, 1])
        Is = self.get_decay_rate_nan(samples_square)
        Is_final = Is * phys_reg * jnp.abs(dsdc)

        return Is_final

    def get_decay_rate_square(self, samples: Sample) -> Prob:
        samples_square, dsdc = self.get_s12_from_cos_theta(samples)
        phys_reg = self.is_physical(samples_square[:, 0], samples_square[:, 1])
        Is = self.get_decay_rate(samples_square)
        Is_final = Is * phys_reg * jnp.abs(dsdc)

        return Is_final

    def scale_samples_to_unit_interval(self, samples: Sample) -> Tuple[Sample, Det]:
        def scale_min_max(var, v_min, v_max):
            var_scaled = (var - v_min) / (v_max - v_min)
            return var_scaled

        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)

        cos_min, cos_max = jnp.full(samples[:, 0].shape, -1.0), jnp.full(
            samples[:, 0].shape, 1.0
        )
        s23_min_val = jnp.min(
            jnp.array([(self.m2 + self.m3) ** 2, (self.m0 - self.m1) ** 2])
        )
        s23_max_val = jnp.max(
            jnp.array([(self.m2 + self.m3) ** 2, (self.m0 - self.m1) ** 2])
        )
        s23_min = jnp.full(samples[:, 1].shape, s23_min_val)
        s23_max = jnp.full(samples[:, 1].shape, s23_max_val)

        cos_t_trafo, dxdcos = jax.vmap(jax.value_and_grad(scale_min_max, argnums=0))(
            samples[:, 0], cos_min, cos_max
        )
        s23_trafo, dyds23 = jax.vmap(jax.value_and_grad(scale_min_max, argnums=0))(
            samples[:, 1], s23_min, s23_max
        )
        samples_trafo = jnp.dstack([cos_t_trafo, s23_trafo]).squeeze()
        dets = jnp.dstack([dxdcos, dyds23]).squeeze()

        if len(samples_trafo.shape) == 1:
            samples_trafo = jnp.expand_dims(samples_trafo, axis=0)
        if len(dets.shape) == 1:
            dets = jnp.expand_dims(dets, axis=0)

        return samples_trafo, dets

    def scale_samples_to_original_interval(self, samples: Sample) -> Tuple[Sample, Det]:
        def inverse_scale_min_max(var, v_min, v_max):
            var_scaled = (v_max - v_min) * var + v_min
            return var_scaled

        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)

        cos_min, cos_max = jnp.full(samples[:, 0].shape, -1.0), jnp.full(
            samples[:, 0].shape, 1.0
        )
        s23_min_val = jnp.min(
            jnp.array([(self.m2 + self.m3) ** 2, (self.m0 - self.m1) ** 2])
        )
        s23_max_val = jnp.max(
            jnp.array([(self.m2 + self.m3) ** 2, (self.m0 - self.m1) ** 2])
        )
        s23_min = jnp.full(samples[:, 1].shape, s23_min_val)
        s23_max = jnp.full(samples[:, 1].shape, s23_max_val)

        cos_t_trafo, dcosdx = jax.vmap(
            jax.value_and_grad(inverse_scale_min_max, argnums=0)
        )(samples[:, 0], cos_min, cos_max)
        s23_trafo, ds23dy = jax.vmap(
            jax.value_and_grad(inverse_scale_min_max, argnums=0)
        )(samples[:, 1], s23_min, s23_max)
        samples_trafo = jnp.dstack([cos_t_trafo, s23_trafo]).squeeze()
        dets = jnp.dstack([dcosdx, ds23dy]).squeeze()

        if len(samples_trafo.shape) == 1:
            samples_trafo = jnp.expand_dims(samples_trafo, axis=0)
        if len(dets.shape) == 1:
            dets = jnp.expand_dims(dets, axis=0)

        return samples_trafo, dets

    def evaluate_DP_on_unit_interval(self, samples: Sample) -> Prob:
        """Evaluate samples from flow on original Dalitz plot and transform density back to unit interval
        to compare to log q of flow."""
        samples_square_dp, detJ = self.scale_samples_to_original_interval(samples)
        dcosdx, dsdy = detJ[:, 0], detJ[:, 1]
        samples_dp, dsdcos = self.get_s12_from_cos_theta(samples_square_dp)
        I_dp = self.get_decay_rate_nan(samples_dp)
        I_xy = I_dp * jnp.abs(dsdcos) * jnp.abs(dcosdx) * jnp.abs(dsdy)

        return I_xy.squeeze()

    def evaluate_DP_on_physical_space(self, samples: Sample) -> Tuple[Sample, Prob]:
        """Evaluate samples from flow on Dalitz plot and return transformed coordinates and DP density."""
        samples_square_dp, _ = self.scale_samples_to_original_interval(samples)
        samples_dp, dsdcos = self.get_s12_from_cos_theta(samples_square_dp)
        I_dp = self.get_decay_rate_nan(samples_dp)

        return samples_dp.squeeze(), I_dp.squeeze()

    def transform_q_to_physical_space(
        self, samples: Sample, q: Prob
    ) -> Tuple[Sample, Prob]:
        """Transform flow samples and corresponding q values to Dalitz plot space."""
        samples_square_dp, detJ = self.scale_samples_to_original_interval(samples)
        dcosdx, dsdy = detJ[:, 0], detJ[:, 1]
        samples_dp, dsdcos = self.get_s12_from_cos_theta(samples_square_dp)
        # Clip derivative to avoid divergent values in plotting of q (except 0)
        dsdcos = jnp.where(dsdcos == 0, 0, jnp.clip(dsdcos, a_min=0.3))
        q_dp = (
            q
            * jnp.where(dsdcos == 0, 0, 1 / jnp.abs(dsdcos))
            * jnp.where(dcosdx == 0, 0, 1 / jnp.abs(dcosdx))
            * jnp.where(dsdy == 0, 0, 1 / jnp.abs(dsdy))
        )

        return samples_dp.squeeze(), q_dp.squeeze()

    def transform_samples_to_physical_space(self, samples: Sample) -> Sample:
        """Transform flow samples to Dalitz plot space."""
        samples_square_dp, _ = self.scale_samples_to_original_interval(samples)
        samples_dp, _ = self.get_s12_from_cos_theta(samples_square_dp)

        return samples_dp.squeeze()

    def log_prob(self, samples: Sample) -> Prob:
        """Evaluate logprob of samples from unit interval."""
        log_probs = jnp.log(self.evaluate_DP_on_unit_interval(samples))
        return log_probs.squeeze()

    def prob_no_nan(self, samples: Sample) -> Prob:
        """Evaluate prob of samples from unit interval and replace NaN values with zero.
        Reason: baseline VEGAS cannot deal with NaN values"""
        probs = self.evaluate_DP_on_unit_interval(samples)
        probs = jnp.nan_to_num(probs, nan=0.0)
        return probs.squeeze()

    def sample(self, key: chex.PRNGKey, n_samples: int = 1000) -> dict[Sample, int]:
        """
        Generate samples with rejection sampling.
        """
        n_batch = 10
        axis_1 = axis_2 = jnp.linspace(0.0, 1.0, 300)[1:-1]
        grid_unit = jnp.dstack(jnp.meshgrid(axis_1, axis_2))
        grid_unit = grid_unit.reshape([-1, 2])

        density_dp = self.evaluate_DP_on_unit_interval(grid_unit)
        max_density = jnp.max(density_dp)

        # proposal distribution: Uniform dist
        p_proposal = max_density
        samples = jnp.array([])
        counter = 0
        count_target_evals = 0
        keys = random.split(key, int(1e8))
        while len(samples) < n_samples:
            # consider n_batch data points at the same time
            # sample x ~ Uniform
            keys_i = random.split(keys[counter], 3)
            x1 = random.uniform(keys_i[0], shape=[n_batch])
            x2 = random.uniform(keys_i[1], shape=[n_batch])
            x = jnp.array([x1, x2]).T

            # sample uniformly [0, density_x]
            y = random.uniform(
                keys_i[2], shape=[n_batch], minval=0.0, maxval=p_proposal
            )
            # evaluate Dalitz plot density at x
            p_dp = self.evaluate_DP_on_unit_interval(x)
            count_target_evals += n_batch

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

        return samples, count_target_evals
