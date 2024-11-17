from typing import Dict, Tuple
import chex
import jax
import jax.numpy as jnp
from madjax import MadJax

from FABdiffME.targets.target_util import HEP2DTarget, move_away_from_boundary

# Functions to load the the e+ e- -> mu+ mu- MadJAX matrix element

Sample = chex.Array
Prob = chex.Array


class EeToMumu(HEP2DTarget):
    def __init__(
        self,
        dim: int,
        center_of_mass_energy: float,
        model_parameters: dict = {},  # default
        epsilon_boundary: float = 1e-10,
    ):
        assert dim == 2
        self.path_madjax = "FABdiffME.targets.madjax_ee_to_mumu"
        self.dim = dim
        self.name = "ee_to_mumu"
        self.E_cm = center_of_mass_energy
        self.model_params = model_parameters
        self.epsilon_boundary = epsilon_boundary
        self.load_matrix_element()

    def load_matrix_element(self):
        self.mj = MadJax(config_name=self.path_madjax)
        assert len(self.mj.processes.items()) == 1
        for proc_name, _ in self.mj.processes.items():
            self.process_name = proc_name
        self.phase_space_generator = self.mj.phasespace_generator(
            E_cm=self.E_cm, process_name=self.process_name
        )(self.model_params)
        self.params = self.mj.parameters.calculate_full_parameters(self.model_params)
        self.me_and_jac_func = self.get_matrix_element_and_jacobian()

    def get_phase_space_generator(self):
        """Return phase space generator with empty parameters."""
        return self.phase_space_generator

    def get_matrix_element_and_jacobian(self):
        """Return function that evaluate the matrix element and the jacobian.
        Based on mj.matrix_element_and_jacobian, but the original function cannot be jit compiled, because the phase space generator
        is instantiated within mj.matrix_element_and_jacobian. mj.phasespace_generator contains an 'assert' statement which throws errors in jit.
        """

        def matrix_element_and_jacobian(random_variables):
            ps_point, jacobian = self.phase_space_generator.generateKinematics(
                self.E_cm, random_variables
            )
            process = self.mj.processes[self.process_name]()
            return process.smatrix(ps_point, self.params), jacobian

        return matrix_element_and_jacobian

    def evaluate_me_on_unit_hypercube(self, samples: Sample) -> Prob:
        """Evaluate matrix element on unit hypercube."""
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        assert samples.shape[-1] == self.dim

        if self.epsilon_boundary is not None and self.epsilon_boundary > 0.0:
            samples = move_away_from_boundary(samples, self.epsilon_boundary)

        def func(rv):
            me_val, jac_val = self.me_and_jac_func(rv)
            return me_val * jnp.abs(jac_val)

        values_me = jax.vmap(func)(samples)

        return values_me.squeeze()

    def evaluate_DP_on_physical_space(self, samples) -> Tuple[Sample, Prob]:
        """Evaluates matrix element on unit hypercube, because ee->mumu would be 1D in Dalitz plot space.
        Needed for "plot_density_comparison_and_flow_samples"
        """
        me_values = self.evaluate_me_on_unit_hypercube(samples)

        return samples, me_values

    def transform_q_to_physical_space(
        self, samples: Sample, q: Prob
    ) -> Tuple[Sample, Prob]:
        """Dummy function, no transformation to DP space, because ee->mumu would be 1D in Dalitz plot space.
        Needed for "plot_density_comparison_and_flow_samples"
        """
        return samples, q

    def transform_samples_to_physical_space(self, samples: Sample) -> Sample:
        """Dummy function, no transformation to Dalitz plot space, because ee->mumu would be 1D in Dalitz plot space.
        Needed for intermediate plot function of FAB
        """
        return samples

    def log_prob(self, samples: Sample) -> Prob:
        """Evaluate logprob of samples from unit interval."""
        log_probs = jnp.log(self.evaluate_me_on_unit_hypercube(samples))

        return log_probs

    def prob_no_nan(self, samples: Sample) -> Prob:
        """Evaluate prob of samples from unit interval and replace NaN values with zero.
        Reason: baseline VEGAS cannot deal with NaN values"""
        probs = self.evaluate_me_on_unit_hypercube(samples)
        probs = jnp.nan_to_num(probs, nan=0.0)

        return probs

    def sample(self, key: chex.PRNGKey, n_samples: int = 1000) -> Dict[Sample, int]:
        """
        Generate samples with rejection sampling.
        """
        n_batch = 10
        axis_1 = axis_2 = jnp.linspace(0.0, 1.0, 300)[1:-1]
        grid_unit = jnp.dstack(jnp.meshgrid(axis_1, axis_2))
        grid_unit = grid_unit.reshape([-1, 2])

        density_dp = self.prob_no_nan(grid_unit)
        max_density = jnp.max(density_dp)

        # proposal distribution: Uniform dist
        p_proposal = max_density
        samples = jnp.array([])
        counter = 0
        count_target_evals = 0
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
            p_dp = self.prob_no_nan(x)
            count_target_evals += n_batch

            # decide whether or not to keep sample
            inds = jnp.where(y < p_dp)[0]
            # save samples
            if samples.size == 0:
                samples = x[inds, :]
            else:
                samples = jnp.vstack([samples, x[inds, :]])

            if counter % 200 == 0:
                print(
                    len(samples),
                    "/",
                    n_samples,
                    "no. target evaluations:",
                    count_target_evals,
                )
            if counter == 1e8:
                break
            counter += 1

        samples = jnp.array(samples[:n_samples, :])

        return samples, count_target_evals
