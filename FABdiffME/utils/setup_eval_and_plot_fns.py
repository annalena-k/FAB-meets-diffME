from typing import Dict, Tuple
import chex
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from FABdiffME.flow.evaluate_flow import evaluate_data_fkld, evaluate_log_ess
from FABdiffME.flow.spline_flow import Flow
from FABdiffME.targets.target_util import Target
from FABdiffME.train.init_and_step_state import TrainingState
from FABdiffME.train.generic_training_loop import (
    PlotFn,
    EvalFn,
    FinalEvalAndPlotFn,
)
from FABdiffME.sampling.metrics import integral_estimate
from FABdiffME.utils.plot import *
from FABdiffME.utils.config import Dirs


def setup_intermediate_plot_fn(
    flow: Flow, target: Target, plot_batch_size: int, log_dir: str
) -> PlotFn:

    if target.dim == 2:
        unit_axis = np.linspace(0, 1, 300)[1:-1]  # x, y
        grid_unit = np.dstack(jnp.meshgrid(unit_axis, unit_axis))
        grid_re = grid_unit.reshape([-1, 2])

    def get_data_for_plotting(
        state: TrainingState, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        samples, log_q_samples = flow.sample_and_log_prob_apply(
            state.params, key=key, shape=[plot_batch_size, flow.dim]
        )
        if target.dim == 2:
            log_q_grid = flow.log_prob_apply(state.params, grid_re)
            return samples, log_q_samples, log_q_grid
        else:
            return samples, log_q_samples

    def plot(state: TrainingState, key: chex.PRNGKey, epoch: int):
        if target.dim == 2:
            fig, axs = plt.subplots(1, 2, figsize=(7, 3))
            samples, log_q_samples, log_q_grid = get_data_for_plotting(state, key)
            plot_progress_during_training(
                target, grid_re, log_q_grid, samples, log_q_samples, fig=fig, axs=axs
            )
            if target.name == "pi1800":
                axs[0].set_xlabel(r"$m_{12}^2 ~ [GeV^2]$")
                axs[0].set_ylabel(r"$m_{23}^2 ~ [GeV^2]$")
                axs[1].set_xlabel(r"$m_{12}^2 ~ [GeV^2]$")
            elif target.name == "lambdac":
                axs[0].set_xlabel(r"$m_{23}^2 = m_{pK}^2 ~ [GeV^2]$")
                axs[0].set_ylabel(r"$m_{31}^2 = m_{K\pi}^2 ~ [GeV^2]$")
                axs[1].set_xlabel(r"$m_{23}^2 = m_{pK}^2 ~ [GeV^2]$")
            elif target.name == "ee_to_mumu":
                axs[0].set_xlabel(r"$x$")
                axs[0].set_ylabel(r"$y$")
                axs[1].set_xlabel(r"$x$")
            else:
                print(
                    f"target.name={target.name} not in [pi1800, lambdac, ee_to_mumu]."
                )
            plt.tight_layout()
            plt.savefig(
                f"{log_dir}/train_progression_{epoch}.png", dpi=200, bbox_inches="tight"
            )
            plt.close(fig)
        else:
            samples, log_q_samples = get_data_for_plotting(state, key)

    return plot


def setup_eval_fn(
    flow: Flow, target: Target, data: chex.Array, n_samples_ESS: int
) -> EvalFn:

    def eval_fn(state: TrainingState, prng_key: chex.PRNGKey) -> Dict:
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.params, x)

        log_p_fn_jit = jax.jit(target.log_prob)
        # Validation loss
        val_loss_mean, val_loss_std = evaluate_data_fkld(log_p_fn_jit, log_q_fn, data)

        # Effective samples size
        log_ess, log_weights = evaluate_log_ess(
            flow.sample_and_log_prob_apply,
            state.params,
            prng_key,
            log_p_fn_jit,
            n_samples_ESS,
        )
        # Integral estimation
        integral, integral_std = integral_estimate(log_weights)

        info = {}
        info.update(
            ess=float(jnp.exp(log_ess)),
            integral=float(integral),
            std_integral=float(integral_std),
            val_loss=float(val_loss_mean),
            std_val_loss=float(val_loss_std),
        )

        return info

    return eval_fn


def setup_final_eval_and_plot_fn(
    flow: Flow,
    target: Target,
    dirs: Dirs,
    data: chex.Array,
    plot_batch_size: int,
    eval_and_plot_scale: int,
) -> FinalEvalAndPlotFn:

    def final_eval_and_plot_fn(
        state: TrainingState, info: Dict, key: chex.PRNGKey, eval_and_plot_iter: list
    ):
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow.log_prob_apply(state.params, x)

        iteration = info["iteration"][-1]
        final_iteration = eval_and_plot_iter[-1]
        # Evaluate test data only in final iteration
        test_info = {}
        if iteration == final_iteration:
            # Evaluate test data
            mean_test_loss, std_test_loss = evaluate_data_fkld(
                target.log_prob, log_q_fn, data
            )
            test_info.update(test_loss=mean_test_loss, std_test_loss=std_test_loss)

        # Plot loss curves without stds
        loss = info["loss"]
        std_loss = info["std_loss"]
        val_loss = info["val_loss"]
        if len(loss) > len(val_loss) or len(eval_and_plot_iter) != len(loss):
            is_relevant_iter = eval_and_plot_iter <= len(loss) - 1
            eval_and_plot_iter = eval_and_plot_iter[is_relevant_iter]
            loss = np.take(np.array(loss), eval_and_plot_iter)
            std_loss = np.take(np.array(std_loss), eval_and_plot_iter)

        if len(loss) == len(val_loss) == len(eval_and_plot_iter):
            fig, axs = plt.subplots(1, 1, figsize=(4, 3))
            if test_info == {}:
                plot_loss_curves(
                    jnp.array(loss),
                    jnp.array(val_loss),
                    None,
                    axs,
                    x=eval_and_plot_iter,
                )
            else:
                plot_loss_curves(
                    jnp.array(loss),
                    jnp.array(val_loss),
                    test_info["test_loss"],
                    axs,
                    x=eval_and_plot_iter,
                )
            # ticks = jnp.arange(0, len(info["loss"]), 2)
            # labels = jnp.arange(0, len(info["loss"]), 2)
            # axs.set_xticks(ticks, labels)
            axs.set_xlabel("Epochs")
            axs.set_ylabel("Loss")
            axs.set_axisbelow(True)
            if eval_and_plot_scale == "log":
                axs.set_xscale("log")
            axs.grid()
            plt.tight_layout()
            plt.savefig(f"{dirs.plot_dir}/losses.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            print("Plotting of loss curves failed.")
            print(
                "Different sizes of loss:",
                len(loss),
                "val_loss:",
                len(val_loss),
                "iter:",
                len(eval_and_plot_iter),
            )

        # Plot ESS
        ess = info["ess"]
        if len(eval_and_plot_iter) == len(ess):
            print("eval_iter", len(eval_and_plot_iter), "ess", len(ess))
            fig, axs = plt.subplots(1, 1, figsize=(4, 3))
            try:
                plot_curve(ess, ax=axs, x=eval_and_plot_iter)
                axs.set_title(
                    r"Effective Sample Size of $10^5$ samples from $q_{\theta}$",
                    fontsize=10,
                )
                axs.set_xlabel("Iterations")
                axs.set_ylabel(r"$ESS$")
                if eval_and_plot_scale == "log":
                    axs.set_xscale("log")
                axs.grid()
                # axs.set_ylim([-0.04, 1.04])
                plt.tight_layout()
                plt.savefig(f"{dirs.plot_dir}/ess.png", dpi=200, bbox_inches="tight")
            except:
                print("Plotting of ESS failed.")
                print(
                    "Different sizes of iter:",
                    len(eval_and_plot_iter),
                    "and ess",
                    len(ess),
                )
            plt.close(fig)
        else:
            print("Plotting of ESS failed.")
            print(
                "Different sizes of iter:", len(eval_and_plot_iter), "and ess", len(ess)
            )

        # Plot evaluation metrics and ESS
        std_val_loss = info["std_val_loss"]
        if (
            len(loss)
            == len(std_loss)
            == len(eval_and_plot_iter)
            == len(val_loss)
            == len(std_val_loss)
        ):
            fig, axs = plt.subplots(1, 2, figsize=(7, 3))
            label_train = r"training loss ($10^6$ samples)"
            plot_curve_std(
                jnp.array(loss),
                jnp.array(std_loss),
                ax=axs[0],
                loss_label=label_train,
                x=eval_and_plot_iter,
                color="tab:blue",
            )
            label_val = r"val loss ($10^4$ samples)"
            plot_curve_std(
                jnp.array(val_loss),
                jnp.array(std_val_loss),
                ax=axs[0],
                loss_label=label_val,
                x=eval_and_plot_iter,
                color="tab:red",
            )
            if iteration == final_iteration:
                label_test = r"test loss ($10^4$ samples)"
                axs[0].errorbar(
                    len(info["loss"]) - 1,
                    test_info["test_loss"],
                    yerr=test_info["std_test_loss"],
                    c="darkred",
                    marker="o",
                    markersize=4,
                    label=label_test,
                )
            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Loss")
            label_title = r"fKLD of $10^6$ samples from $p$"
            axs[0].set_title(label_title, fontsize=10)
            if target.name == "pi1800":
                axs[0].set_ylim([-1.04, 1.04])
            elif target.name == "lambdac":
                axs[0].set_ylim([-0.75, 1.5])
            axs[0].grid()
            axs[0].legend()
            axs[0].legend(labelspacing=0)

            plot_curve(info["ess"], ax=axs[1], x=eval_and_plot_iter)
            # axs[1].set_ylim([-0.04, 1.04])
            axs[1].set_title(
                r"Effective Sample Size of $10^5$ samples from $q_{\theta}$",
                fontsize=10,
            )
            axs[1].set_xlabel("Epochs")
            axs[1].set_ylabel(r"$ESS$")
            axs[1].grid()

            plt.tight_layout()
            plt.savefig(f"{dirs.plot_dir}/metrics.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        else:
            print("Joint plotting of loss curves and ESS failed.")
            print(
                "Different sizes of loss:",
                len(loss),
                "std_loss:",
                len(std_loss),
                "val_loss:",
                len(val_loss),
                "std_val_loss:",
                len(std_val_loss),
                "iter:",
                len(eval_and_plot_iter),
            )

        if target.dim == 2:
            # Plot density comparison of target and flow as well as samples from flow
            fig, axs = plt.subplots(1, 3, sharey="row", figsize=(10, 3))
            plot_density_comparison_and_flow_samples(
                target,
                flow,
                state.params,
                key,
                fig=fig,
                axs=axs,
                plot_batch_size=plot_batch_size,
            )
            if target.name == "pi1800":
                axs[0].set_xlabel(r"$m_{12}^2 ~ [GeV^2]$")
                axs[1].set_xlabel(r"$m_{12}^2 ~ [GeV^2]$")
                axs[2].set_xlabel(r"$m_{12}^2 ~ [GeV^2]$")
                axs[0].set_ylabel(r"$m_{23}^2 ~ [GeV^2]$")
                axs[0].xaxis.set_ticks(jnp.arange(0.2, 1.4, 0.2))
                axs[1].xaxis.set_ticks(jnp.arange(0.2, 1.4, 0.2))
                axs[2].xaxis.set_ticks(jnp.arange(0.2, 1.4, 0.2))
                axs[2].yaxis.set_ticks(jnp.arange(0.2, 1.4, 0.2))
            elif target.name == "lambdac":
                axs[0].set_xlabel(r"$m_{23}^2 = m_{pK}^2 ~ [GeV^2]$")
                axs[1].set_xlabel(r"$m_{23}^2 = m_{pK}^2 ~ [GeV^2]$")
                axs[2].set_xlabel(r"$m_{23}^2 = m_{pK}^2 ~ [GeV^2]$")
                axs[0].set_ylabel(r"$m_{31}^2 = m_{K\pi}^2 ~ [GeV^2]$")
            elif target.name == "ee_to_mumu":
                axs[0].set_xlabel(r"$x$")
                axs[1].set_xlabel(r"$x$")
                axs[2].set_xlabel(r"$x$")
                axs[0].set_ylabel(r"$y$")
            else:
                print(
                    f"target.name={target.name} not in [pi1800, lambdac, ee_to_mumu]."
                )

            plt.tight_layout()
            plt.savefig(
                f"{dirs.plot_dir}/dalitz_plot_result.png", dpi=200, bbox_inches="tight"
            )
            plt.close(fig)

        return test_info

    return final_eval_and_plot_fn
