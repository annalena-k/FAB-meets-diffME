from typing import Callable, Optional, Tuple
import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fabjax.sampling.resampling import log_effective_sample_size

from FABdiffME.targets.target_util import Target, LogProbTargetFn

LogProbFn = Callable[[chex.Array], chex.Array]
SampleNLogProbFn = Callable[
    [chex.Array, chex.PRNGKey, chex.Array], Tuple[chex.Array, chex.Array]
]


def evaluate_data_fkld(
    log_p_fn: LogProbTargetFn,
    log_q_fn: LogProbFn,
    data: chex.Array,
    inner_batch_size: Optional[int] = int(1e5),
) -> Tuple[float, float]:
    """Evaluate fkld of data"""

    def nll_scan_fn(carry: None, xs: chex.Array) -> Tuple[None, Tuple[float,]]:
        val_data = xs
        log_qs = log_q_fn(val_data)
        sum_log_q = jnp.sum(jnp.array(-log_qs))

        return None, (sum_log_q,)

    def std_nll_scan_fn(
        carry: None, xs: Tuple[chex.Array, float]
    ) -> Tuple[None, Tuple[float,]]:
        """Manually accumulate sum (arg - mean)*2 for std calculation."""
        val_data, mean_nll = xs
        log_qs = log_q_fn(val_data)
        log_ps = log_p_fn(val_data)
        sum_std_arg = jnp.sum(jnp.array(log_ps - log_qs - mean_nll))

        return None, (sum_std_arg,)

    # Calculate fkld
    n_samples = data.shape[0]
    dim = data.shape[1]
    if n_samples < inner_batch_size:
        log_qs = log_q_fn(data)
        log_ps = log_p_fn(data)
        mean_test_loss = jnp.mean(jnp.array(-log_qs))
        std_test_loss = jnp.std(jnp.array(log_ps - log_qs))
    else:
        # Speed up calculation for a large number of samples
        assert n_samples % inner_batch_size == 0
        n_batches = int(n_samples / inner_batch_size)
        data = data.reshape([n_batches, -1, dim])  # [N, dim] ->[n, -1, dim]
        _, (sum_log_q,) = jax.lax.scan(nll_scan_fn, init=None, xs=data)
        sum_log_q = sum_log_q.flatten()
        mean_test_loss = jnp.sum(sum_log_q) / n_samples

        arr_mean = jnp.array([mean_test_loss] * n_batches)
        _, (sum_std_arg,) = jax.lax.scan(
            std_nll_scan_fn, init=None, xs=[data, arr_mean]
        )
        sum_std_arg = sum_std_arg.flatten()
        std_test_loss = jnp.sqrt(jnp.sum(sum_std_arg) / n_samples)

    return mean_test_loss, std_test_loss


def evaluate_log_ess(
    sample_log_q_fn: SampleNLogProbFn,
    params: chex.ArrayTree,
    key: chex.PRNGKey,
    log_p_fn: LogProbTargetFn,
    n_flow_samples: int,
    inner_batch_size: Optional[int] = int(1e5),
) -> Tuple[float, chex.Array]:

    def ess_scan_fn(carry: None, xs: chex.PRNGKey) -> Tuple[None, Tuple[chex.Array,]]:
        key_in = xs
        samples, log_q_flow = sample_log_q_fn(params, key_in, (inner_batch_size,))
        log_p_flow = log_p_fn(samples)
        log_w_flow = log_p_flow - log_q_flow

        return None, (log_w_flow,)

    # Calculate ESS
    if n_flow_samples <= inner_batch_size:
        key, subkey = jax.random.split(key)
        samples, log_q_flow = sample_log_q_fn(params, subkey, (n_flow_samples,))
        log_p_flow = log_p_fn(samples)
        log_w_flow = log_p_flow - log_q_flow
        log_ess_flow = log_effective_sample_size(log_w_flow)
    else:
        n_batches = (n_flow_samples // inner_batch_size) + 1
        _, (log_w_flow,) = jax.lax.scan(
            ess_scan_fn, init=None, xs=jax.random.split(key, n_batches)
        )
        log_w_flow = log_w_flow.flatten()[:n_flow_samples]
        log_ess_flow = log_effective_sample_size(log_w_flow)

    return log_ess_flow, log_w_flow


def evaluate_flow_final(
    target: Target,
    val_data: chex.Array,
    flow,
    params: chex.ArrayTree,
    key: chex.PRNGKey,
    n_flow_samples: int,
    inner_batch_size: int,
    log_dir: Optional[str] = None,
    iter_nr: Optional[int] = None,
):
    """Evaluates flow on large validation data set and calculates ESS on large number of samples
    # 1) generate samples from flow in batches to calculate ESS
    # 2) evaluate large validation data set on flow in batches
    """
    w_threshold = 1000

    def ess_scan_fn(
        carry: None, xs: chex.PRNGKey
    ) -> Tuple[None, Tuple[chex.Array, chex.Array, chex.Array]]:
        key_in = xs
        samples, log_q_flow = flow.sample_and_log_prob_apply(
            params, key_in, (inner_batch_size,)
        )
        log_p_flow = target.log_prob(samples)
        log_w_flow = log_p_flow - log_q_flow

        return None, (log_w_flow, log_q_flow, log_p_flow)

    def nll_scan_fn(carry: None, xs: chex.Array) -> Tuple[None, Tuple[float,]]:
        val_data = xs
        log_qs = flow.log_prob_apply(params, val_data)
        sum_log_q = jnp.sum(jnp.array(-log_qs))

        return None, (sum_log_q,)

    def std_nll_scan_fn(
        carry: None, xs: Tuple[chex.Array, float]
    ) -> Tuple[None, Tuple[float,]]:
        """Manually accumulate sum (arg - mean)*2 for std calculation."""
        val_data, mean_nll = xs
        log_qs = flow.log_prob_apply(params, val_data)
        log_ps = target.log_prob(val_data)
        sum_std_arg = jnp.sum(jnp.array(log_ps - log_qs - mean_nll))

        return None, (sum_std_arg,)

    def fkld_scan_fn(carry: None, xs: chex.Array) -> Tuple[None, Tuple[float,]]:
        val_data = xs
        log_qs = flow.log_prob_apply(params, val_data)
        log_ps = target.log_prob(val_data)
        sum_fkld = jnp.sum(jnp.array(log_ps)) - jnp.sum(jnp.array(log_qs))

        return None, (sum_fkld,)

    # Calculate ESS
    n_batches = (n_flow_samples // inner_batch_size) + 1
    _, (log_w_flow, log_q, log_p) = jax.lax.scan(
        ess_scan_fn, init=None, xs=jax.random.split(key, n_batches)
    )
    log_w_flow = log_w_flow.flatten()[:n_flow_samples]
    log_q = log_q.flatten()[:n_flow_samples]
    log_p = log_p.flatten()[:n_flow_samples]

    integral_estimate = jnp.mean(jnp.exp(log_w_flow))
    integral_std = jnp.std(jnp.exp(log_w_flow))

    log_ess_flow = log_effective_sample_size(log_w_flow)
    eval_info = {}
    eval_info.update(
        ess=jnp.exp(log_ess_flow), integral=integral_estimate, integral_std=integral_std
    )

    # NLL of target samples
    n_target_samples = val_data.shape[0]
    assert n_target_samples % inner_batch_size == 0
    n_batches_eval = int(n_target_samples / inner_batch_size)
    val_data = val_data.reshape(
        [n_batches_eval, -1, target.dim]
    )  # [N, dim] ->[n, -1, dim]
    _, (sum_log_q,) = jax.lax.scan(nll_scan_fn, init=None, xs=val_data)
    sum_log_q = sum_log_q.flatten()
    mean_val_loss = jnp.sum(sum_log_q) / n_target_samples

    arr_mean = jnp.array([mean_val_loss] * n_batches_eval)
    _, (sum_std_arg,) = jax.lax.scan(
        std_nll_scan_fn, init=None, xs=[val_data, arr_mean]
    )
    sum_std_arg = sum_std_arg.flatten()
    std_val_loss = jnp.sqrt(jnp.sum(sum_std_arg) / n_target_samples)

    eval_info.update(val_loss=mean_val_loss, std_val_loss=std_val_loss)

    # fKLD of target samples
    _, (sum_fkld,) = jax.lax.scan(fkld_scan_fn, init=None, xs=val_data)
    sum_fkld = sum_fkld.flatten()
    mean_fkld = jnp.sum(sum_fkld) / n_target_samples

    eval_info.update(fkld=mean_fkld)

    return eval_info
