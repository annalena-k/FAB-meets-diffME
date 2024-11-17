from typing import Callable, NamedTuple, Sequence, Tuple
import distrax
import chex
import haiku as hk
from jax import numpy as jnp

LogProb = chex.Array
Params = chex.ArrayTree
Samples = chex.Array

InitFn = Callable[[chex.PRNGKey, Samples], Params]
LogProbApplyFn = Callable[[Params, Samples], LogProb]
SampleNLogProbApplyFn = Callable[
    [Params, chex.PRNGKey, chex.Array], Tuple[Samples, LogProb]
]
SampleApplyFn = Callable[[Params, chex.PRNGKey, chex.Array], Samples]


class FlowConfig(NamedTuple):
    dim: int
    n_trafos: int
    n_bijector_params: int
    hidden_layer_sizes: Sequence[int]


class FlowRecipe(NamedTuple):
    config: FlowConfig
    dim: int
    create_model: Callable[[], distrax.Transformed]


class Flow(NamedTuple):
    config: FlowConfig
    dim: int
    init: InitFn
    log_prob_apply: LogProbApplyFn
    sample_and_log_prob_apply: SampleNLogProbApplyFn
    sample_apply: SampleApplyFn


def create_flow(recipe: FlowRecipe) -> Flow:

    def init(prng_key: chex.PRNGKey, data: Samples) -> Params:
        batch_size = data.shape[0]
        dummy_data = jnp.ones([batch_size, recipe.dim])
        params = log_prob.init(rng=prng_key, data=dummy_data)
        return params

    @hk.without_apply_rng
    @hk.transform
    def log_prob(data: chex.Array):
        """Pass data through inverse bijector and calculate log prob."""
        model = recipe.create_model()
        return model.log_prob(data)

    @hk.without_apply_rng
    @hk.transform
    def sample_n_and_log_prob(key: chex.PRNGKey, n_samples: int):
        """Sample in latent space, pass through forward bijector, and simultaneously calculate log prob."""
        model = recipe.create_model()
        return model._sample_n_and_log_prob(key, n_samples)

    def sample_apply(params: Params, key: chex.PRNGKey, shape: chex.Shape) -> Samples:
        n_samples = shape[0]
        samples, _ = sample_n_and_log_prob.apply(params, key=key, n_samples=n_samples)
        return samples

    def log_prob_apply(params: Params, data: Samples) -> LogProb:
        return log_prob.apply(params, data=data)

    def sample_and_log_prob_apply(
        params: Params, key: chex.PRNGKey, shape: chex.Shape
    ) -> Tuple[Samples, LogProb]:
        n_samples = shape[0]
        samples, log_p = sample_n_and_log_prob.apply(
            params, key=key, n_samples=n_samples
        )
        return samples, log_p

    flow = Flow(
        config=recipe.config,
        dim=recipe.dim,
        init=init,
        log_prob_apply=log_prob_apply,
        sample_and_log_prob_apply=sample_and_log_prob_apply,
        sample_apply=sample_apply,
    )
    return flow
