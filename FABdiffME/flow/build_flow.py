import chex
import distrax
import jax.numpy as jnp
import numpy as np

from FABdiffME.flow.spline_flow import FlowConfig, FlowRecipe, Flow, create_flow
from FABdiffME.utils.nets import MLP


def build_flow(flow_config: FlowConfig) -> Flow:
    flow_recipe = create_flow_recipe(flow_config)
    flow = create_flow(flow_recipe)

    return flow


def create_flow_recipe(config: FlowConfig) -> FlowRecipe:

    def create_model() -> distrax.Transformed:
        mask = np.arange(0, np.prod(config.dim), 1) % 2
        mask = mask.astype(bool)

        def create_conditioner():
            return MLP(
                [
                    config.dim,
                ],
                config.hidden_layer_sizes,
                config.n_bijector_params,
            )

        def create_bijector(params: chex.ArrayTree):
            return distrax.RationalQuadraticSpline(
                params, range_min=0.0, range_max=1.0
            )  # , boundary_slopes='identity')

        layers = []
        for _ in range(config.n_trafos):
            layer = distrax.MaskedCoupling(
                mask=mask, bijector=create_bijector, conditioner=create_conditioner()
            )
            layers.append(layer)
            mask = jnp.logical_not(mask)

        # invert flow => `forward` method called with `log_prob`
        flow = distrax.Inverse(distrax.Chain(layers))

        base_dist = distrax.Independent(
            distrax.Uniform(low=jnp.zeros(config.dim), high=jnp.ones(config.dim)),
            reinterpreted_batch_ndims=1,
        )
        return distrax.Transformed(base_dist, flow)

    recipe = FlowRecipe(config=config, dim=config.dim, create_model=create_model)
    return recipe
