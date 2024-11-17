from typing import Callable, Dict, NamedTuple, Optional, Tuple
import chex
import optax
import jax
import jax.numpy as jnp

from fabjax.buffer.prioritised_buffer import PrioritisedBufferState

from FABdiffME.flow.spline_flow import Flow
from FABdiffME.targets.target_util import Target
from FABdiffME.utils.load_data import CustomDataset, NumpyLoader

Params = chex.ArrayTree
Data = chex.Array
LogWeight = chex.Array
LogProb = chex.Array
LogProbDataFn = Callable[[Params, Data], LogProb]
LogProbFn = Callable[[Params], LogProb]
ParamLogProbFn = Callable[[Params, chex.PRNGKey], LogProb]
SampleLogProbFn = Callable[[Params, chex.PRNGKey], Tuple[Data, LogProb]]


class TrainingState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    key: chex.PRNGKey


class TrainingStateWithBuffer(NamedTuple):
    params: Params
    opt_state: optax.OptState
    key: chex.PRNGKey
    buffer_state: PrioritisedBufferState


InitStateFn = Callable[[chex.PRNGKey], TrainingState]
UpdateStateFn = Callable[[TrainingState], Tuple[TrainingState, dict]]


def loss_fkld(params: Params, data: Data, log_prob_apply: LogProbDataFn):
    log_probs = log_prob_apply(params, data=data)
    loss = -jnp.mean(log_probs)
    return loss, (log_probs)


def loss_rkld(
    params: Params, key: chex.PRNGKey, log_q_fn: ParamLogProbFn, log_p_fn: LogProbFn
):
    """Calculate reverse KL divergence loss between log prob of flow and target."""
    samples, log_q = log_q_fn(params, key=key)
    log_p = log_p_fn(samples)

    loss = jnp.mean(log_q - log_p)
    std = jnp.std(log_q - log_p)

    return loss, (samples, log_q, log_p, std)


def build_fkld_init_and_step(
    flow: Flow,
    log_p_fn: LogProbFn,
    optimizer: optax.GradientTransformation,
    batch_size: int,
    data: chex.Array,
) -> Tuple[InitStateFn, UpdateStateFn]:

    dataset = CustomDataset(data=data)
    train_loader = NumpyLoader(dataset, batch_size=batch_size)

    def init(key: chex.PRNGKey) -> TrainingState:
        # Initialize optimizer
        dummy_data = jnp.ones([batch_size, flow.dim])
        params = flow.init(key, data=dummy_data)

        # Initialize optimizer
        opt_state = optimizer.init(params)

        state = TrainingState(params=params, opt_state=opt_state, key=None)
        return state

    @jax.jit
    @chex.assert_max_traces(n=4)
    def internal_step(
        data: Data, state: TrainingState
    ) -> Tuple[TrainingState, LogProb]:
        def log_prob_apply(params: Params, data: Data) -> LogProbDataFn:
            return flow.log_prob_apply(params, data=data)

        # Update params
        (loss, (log_qs)), grads = jax.value_and_grad(loss_fkld, has_aux=True)(
            state.params, data=data, log_prob_apply=log_prob_apply
        )
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainingState(params=new_params, opt_state=new_opt_state, key=None)

        return new_state, log_qs

    def step(state: TrainingState) -> Tuple[TrainingState, dict]:
        train_loss_mean = []
        train_loss_std = []
        # Loop over batches
        for data_batch in train_loader:
            # Train
            state, log_qs = internal_step(data_batch, state)
            # Save training loss and std
            train_loss_mean.append(-log_qs)
            log_ps = log_p_fn(data_batch)
            train_loss_std.append(log_ps - log_qs)

        info = {}
        loss_mean = jnp.mean(jnp.concatenate(train_loss_mean).ravel())
        loss_std = jnp.std(jnp.concatenate(train_loss_std).ravel())
        info.update(loss=loss_mean)
        info.update(std_loss=loss_std)

        return state, info

    return init, step


def build_rkld_init_and_step(
    flow: Flow, target: Target, optimizer: optax.GradientTransformation, batch_size: int
) -> Tuple[InitStateFn, UpdateStateFn]:

    def init(key: chex.PRNGKey) -> TrainingState:
        # Initialize flow
        dummy_data = jnp.ones([batch_size, flow.dim])
        key, subkey = jax.random.split(key)
        params = flow.init(subkey, data=dummy_data)

        # Initialize optimizer
        opt_state = optimizer.init(params)

        state = TrainingState(params=params, opt_state=opt_state, key=key)
        return state

    @jax.jit
    @chex.assert_max_traces(n=4)
    def step(state: TrainingState) -> Tuple[TrainingState, Dict]:

        def sample_log_q_apply(params: Params, key: chex.PRNGKey) -> LogProbDataFn:
            return flow.sample_and_log_prob_apply(
                params, key=key, shape=[batch_size, flow.dim]
            )

        def log_p_apply(samples) -> LogProbFn:
            return target.log_prob(samples)

        info = {}
        # Update params
        key, subkey = jax.random.split(state.key)
        (loss, (samples, log_q, log_p, std)), grads = jax.value_and_grad(
            loss_rkld, has_aux=True
        )(state.params, key=subkey, log_q_fn=sample_log_q_apply, log_p_fn=log_p_apply)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

        info.update(loss=loss, std_loss=std)

        new_state = TrainingState(params=new_params, opt_state=new_opt_state, key=key)

        return new_state, info

    return init, step
