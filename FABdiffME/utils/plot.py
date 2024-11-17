from typing import Callable, Tuple, Optional
import chex
import matplotlib.pyplot as plt
import itertools
import jax.numpy as jnp
import numpy as np
import pandas as pd
import haiku as hk

from FABdiffME.flow.spline_flow import Flow
from FABdiffME.targets.target_util import HEP2DTarget

LogProbFn = Callable[[chex.Array], chex.Array]


def get_default_colors():
    def normalize_rgb(rgb: Tuple):
        return tuple([x / 255.0 for x in rgb])

    # accessible color scheme based on 10.48550/arXiv.2107.02270
    c_vegas = normalize_rgb((148, 164, 162))  # gray
    c_fkld = normalize_rgb((131, 45, 182))  # purple
    c_rkld = normalize_rgb((189, 31, 1))  # red
    c_fab_nb = normalize_rgb((255, 169, 14))  # orange
    c_fab_wb = normalize_rgb((231, 99, 0))  # dark orange
    c_fab_hmc_nb = normalize_rgb((146, 218, 221))  # light blue
    c_fab_hmc_wb = normalize_rgb((63, 144, 218))  # blue
    c_fab_rhmc_nb = normalize_rgb((113, 117, 129))  # dark gray
    c_fab_rhmc_wb = normalize_rgb((185, 172, 112))  # tan
    # (169, 107, 89) # brown
    c_train_types = {
        "vegas": c_vegas,
        "fkld": c_fkld,
        "rkld": c_rkld,
        "fab_no_buffer": c_fab_nb,
        "fab_buffer": c_fab_wb,
        "fab_no_buffer_mh": c_fab_nb,
        "fab_buffer_mh": c_fab_wb,
        "fab_no_buffer_hmc": c_fab_hmc_nb,
        "fab_buffer_hmc": c_fab_hmc_wb,
        "fab_no_buffer_rhmc": c_fab_rhmc_nb,
        "fab_buffer_rhmc": c_fab_rhmc_wb,
    }

    c_uniform = "black"
    c_rejection = "darkred"
    c_madgraph = "black"  # normalize_rgb((200, 73, 169)) # pink
    c_data = {"uniform": c_uniform, "rej": c_rejection, "madgraph": c_madgraph}

    c_train_loss = normalize_rgb((87, 144, 252))  # blue
    c_val_loss = normalize_rgb((248, 156, 32))  # red
    c_ess = normalize_rgb((228, 37, 54))  # red
    c_ess_ais = normalize_rgb((150, 74, 139))  # purple
    c_metrics = {
        "train_loss": c_train_loss,
        "val_loss": c_val_loss,
        "ess": c_ess,
        "ess_ais": c_ess_ais,
    }

    return c_train_types, c_data, c_metrics


def plot_grid_2D(
    log_p_func: LogProbFn,
    bound_lower: float,
    bound_upper: float,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
):
    """Plot the grid of a 2D prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    n_points = 300
    x_points_dim1 = np.linspace(bound_lower, bound_upper, n_points)
    x_points_dim2 = np.linspace(bound_lower, bound_upper, n_points)
    x_points = jnp.array(list(itertools.product(x_points_dim1, x_points_dim2)))
    probs = jnp.exp(log_p_func(x_points))
    probs = jnp.clip(probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    p = probs.reshape(n_points, n_points)
    pcm = ax.scatter(x1, x2, c=p, s=0.5)
    fig.colorbar(pcm, ax=ax)


def plot_dalitz_plot_grid_2D(
    log_p_func: LogProbFn,
    target: HEP2DTarget,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    low_quality=False,
):
    """Plot the Dalitz plot grid of a 2D prob function."""
    if ax is None:
        fig, ax = plt.subplots(1)
    if low_quality:
        n_points = 100
        size = 2
    else:
        n_points = 300
        size = 0.5
    x_points = np.linspace(0, 1, n_points)
    x_points = jnp.array(list(itertools.product(x_points, x_points)))
    probs = jnp.exp(log_p_func(x_points))
    x_points, probs = target.transform_q_to_physical_space(x_points, probs)
    probs = jnp.clip(probs, a_min=-1000, a_max=None)
    x1 = x_points[:, 0].reshape(n_points, n_points)
    x2 = x_points[:, 1].reshape(n_points, n_points)
    p = probs.reshape(n_points, n_points)
    if target.name == "lambdac":
        pcm = ax.scatter(x2, x1, c=p, s=size)
    else:
        pcm = ax.scatter(x1, x2, c=p, s=size)

    cbar = fig.colorbar(pcm, ax=ax)
    if target.name == "lambdac":
        cbar.formatter.set_powerlimits((0, 0))


def plot_marginal_pair(
    samples: chex.Array,
    ax: Optional[plt.Axes] = None,
    marginal_dims: Tuple[int, int] = (0, 1),
    bounds: Tuple[int, int] = (0, 1),
    alpha: float = 0.8,
    c: str = "w",
):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = jnp.clip(samples, bounds[0], bounds[1])
    ax.plot(
        samples[:, marginal_dims[0]],
        samples[:, marginal_dims[1]],
        "o",
        alpha=alpha,
        c=c,
        markersize=2.0,
    )


def plot_samples(
    samples: chex.Array, ax: Optional[plt.Axes] = None, alpha: float = 0.8, c: str = "w"
):
    """Plot samples."""
    if not ax:
        fig, ax = plt.subplots(1)
    ax.plot(samples[:, 0], samples[:, 1], "o", alpha=alpha, c=c, markersize=1.0)


def plot_progress_during_training(
    target: HEP2DTarget,
    grid: chex.Array,
    log_q_grid: chex.Array,
    samples: chex.Array,
    log_q_samples: chex.Array,
    fig: Optional[plt.Figure] = None,
    axs: Optional[Tuple[plt.Axes]] = None,
):
    """Plot distribution of target and flow as well as samples used for training on Dalitz Plot.
    This function maps grid and samples to physical space and scales log_prob values accordingly.
        grid: on [0,1]^2
        log_q_grid: flow evaluated on grid
        samples: from flow on [0,1]^2
        log_q_samples: flow evaluated on samples
    """

    if axs is None:
        fig, axs = plt.subplots(1, 2, sharey="row")

    # target
    grid_p, p_vals_grid = target.evaluate_DP_on_physical_space(grid)
    samples_p, p_vals_samples = target.evaluate_DP_on_physical_space(samples)

    min0 = np.nanmin(np.array([np.nanmin(p_vals_samples), np.nanmin(p_vals_samples)]))
    max0 = np.nanmax(np.array([np.nanmax(p_vals_samples), np.nanmax(p_vals_samples)]))
    if max0 > 1e8:
        max0 = 10

    pcm0 = axs[0].scatter(
        grid_p[:, 0], grid_p[:, 1], c=p_vals_grid, s=0.1, vmin=min0, vmax=max0
    )
    axs[0].scatter(
        samples_p[:, 0],
        samples_p[:, 1],
        c=p_vals_samples,
        s=5,
        edgecolors="black",
        linewidths=0.5,
        vmin=min0,
        vmax=max0,
    )
    cbar = fig.colorbar(pcm0, ax=axs[0])
    if target.name == "lambdac":
        cbar.formatter.set_powerlimits((0, 0))
    axs[0].set_title(r"True $p$ with samples", fontsize=10)

    # flow
    grid_q, q_vals_grid = target.transform_q_to_physical_space(
        grid, jnp.exp(log_q_grid)
    )
    samples_q_dp, q_vals_samples = target.transform_q_to_physical_space(
        samples, jnp.exp(log_q_samples)
    )

    min1 = np.min(np.array([jnp.min(q_vals_samples), np.min(q_vals_grid)]))
    if target.name == "lambdac":
        max1 = 8.0
    else:
        max1 = np.max(np.array([jnp.max(q_vals_samples), np.max(q_vals_grid)]))
    if max1 > 1e8:
        max1 = 10

    pcm1 = axs[1].scatter(
        grid_q[:, 0], grid_q[:, 1], c=q_vals_grid, s=0.1, vmin=min1, vmax=max1
    )
    axs[1].scatter(
        samples_q_dp[:, 0],
        samples_q_dp[:, 1],
        c=q_vals_samples,
        s=5,
        edgecolors="black",
        linewidths=0.5,
        vmin=min1,
        vmax=max1,
    )
    cbar = fig.colorbar(pcm1, ax=axs[1])
    if target.name == "lambdac":
        cbar.formatter.set_powerlimits((0, 0))
    axs[1].set_title(r"Learned $q$ with samples", fontsize=10)


def plot_loss_curves(
    losses: chex.Array,
    val_losses: chex.Array,
    mean_test_loss: float = None,
    ax: Optional[plt.Axes] = None,
    x: Optional[chex.Array] = None,
):
    """Plot the loss curves for training, validation, and test data set without stds"""
    if ax is None:
        fig, ax = plt.subplots(1)
    if x is None:
        x = jnp.arange(0, len(losses), 1)
    ax.plot(x, losses, "-o", c="tab:blue", markersize=4, label="training", zorder=0)
    ax.plot(
        x, val_losses, "-o", c="tab:red", markersize=4, label="validation", zorder=1
    )
    if mean_test_loss:
        ax.scatter(x[-1], mean_test_loss, c="darkred", s=15, label="test", zorder=2)
    ax.legend()


def plot_curve_std(
    vals: chex.Array,
    std_vals: chex.Array,
    loss_label: str,
    ax: Optional[plt.Axes] = None,
    x: Optional[chex.Array] = None,
    color: str = "tab:blue",
    zorder: Optional[int] = 0,
):
    """Plot loss curve with std."""
    if ax is None:
        fig, ax = plt.subplots(1)
    if x is None:
        x = jnp.arange(0, len(vals), 1)
    std_upper = vals + std_vals
    std_lower = vals - std_vals
    ax.fill_between(x, std_upper, std_lower, alpha=0.4, color=color, zorder=zorder)
    ax.plot(x, vals, "-o", label=loss_label, c=color, markersize=4, zorder=zorder)


def plot_curve(
    vals: chex.Array,
    ax: Optional[plt.Axes] = None,
    x: Optional[chex.Array] = None,
    val_color: str = "tab:blue",
    label: str = "None",
    zorder: Optional[int] = 0,
):
    """Plot curve without std."""
    if ax is None:
        fig, ax = plt.subplots(1)
    if x is None:
        x = jnp.arange(0, len(vals), 1)
    ax.plot(x, vals, "-o", c=val_color, markersize=4, label=label, zorder=zorder)


def plot_smoothed_curve(
    vals: chex.Array,
    downsampling_factor: int = 50,
    ax: Optional[plt.Axes] = None,
    val_color: str = "tab:blue",
):
    """Plot smoothed curve without std."""
    if ax is None:
        fig, ax = plt.subplots(1)
    x = jnp.arange(0, len(vals), 1)
    x_sub = x[::downsampling_factor]
    vals_smooth = (
        pd.DataFrame(vals)
        .rolling(downsampling_factor, min_periods=1)
        .mean()[::downsampling_factor]
    )
    ax.plot(x_sub, jnp.array(vals_smooth[0]), "-o", c=val_color, markersize=4)


def plot_density_comparison_and_flow_samples(
    target: HEP2DTarget,
    flow: Flow,
    params: hk.Params,
    key: chex.PRNGKey,
    plot_batch_size: int = 1000,
    fig: Optional[plt.Figure] = None,
    axs: Optional[Tuple[plt.Axes]] = None,
):
    """Plot distribution of target and flow as well as samples from flow."""
    if axs is None:
        fig, axs = plt.subplots(1, 3, sharey="row")
    # Create grid
    unit_axis = jnp.linspace(0, 1, 300)[1:-1]  # x, y
    grid_unit = jnp.dstack(jnp.meshgrid(unit_axis, unit_axis))
    grid_re = grid_unit.reshape([-1, 2])

    # Evaluate target
    grid_p, p_vals_grid = target.evaluate_DP_on_physical_space(grid_re)
    # Evaluate flow and transform to physical space
    logq_grid_unit = flow.log_prob_apply(params, grid_re)
    grid_q, q_vals_grid = target.transform_q_to_physical_space(
        grid_re, jnp.exp(logq_grid_unit)
    )
    # Generate samples from flow and transform to physical space
    fsamples, logq_fsamples = flow.sample_and_log_prob_apply(
        params, key=key, shape=[plot_batch_size, flow.dim]
    )
    flow_samples, q_vals_samples = target.transform_q_to_physical_space(
        fsamples, jnp.exp(logq_fsamples)
    )

    pcm0 = axs[0].scatter(grid_p[:, 0], grid_p[:, 1], c=p_vals_grid, s=0.1)
    axs[0].set_title(r"True $p$", fontsize=10)
    cbar = fig.colorbar(pcm0, ax=axs[0])
    if target.name == "lambdac":
        cbar.formatter.set_powerlimits((0, 0))

    pcm1 = axs[1].scatter(grid_q[:, 0], grid_q[:, 1], c=q_vals_grid, s=0.1)
    axs[1].set_title(r"Learned $q_{\theta}$", fontsize=10)
    fig.colorbar(pcm1, ax=axs[1])

    pcm2 = axs[2].scatter(
        flow_samples[:, 0], flow_samples[:, 1], c=q_vals_samples, s=10
    )
    axs[2].set_title(r"Samples from $q_{\theta}$", fontsize=10)
    fig.colorbar(pcm2, ax=axs[2])
