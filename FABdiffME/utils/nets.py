import haiku as hk
import jax.numpy as jnp
import numpy as np


class MLP(hk.Module):
    """Simple MLP with final linear layer."""

    def __init__(self, event_shape, hidden_layers_sizes, n_bijector_params):
        super(MLP, self).__init__()
        self.hidden_layer_sizes = hidden_layers_sizes
        self.linear_size = np.prod(np.array(event_shape)) * n_bijector_params
        self.output_size = tuple(event_shape) + (n_bijector_params,)

    def __call__(self, x: jnp.ndarray):
        x = hk.nets.MLP(self.hidden_layer_sizes, activate_final=True, name="mlp")(x)
        x = hk.Linear(self.linear_size, name="lin", w_init=jnp.zeros, b_init=jnp.zeros)(
            x
        )
        x = hk.Reshape(self.output_size, preserve_dims=-1)(x)
        return x
