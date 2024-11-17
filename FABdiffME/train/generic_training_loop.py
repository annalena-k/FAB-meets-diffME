from typing import Callable, Dict, NamedTuple, Optional, Tuple
import chex
import jax
import optax
import pickle
import numpy as np
import os
import time
from tqdm.autonotebook import tqdm

from fabjax.utils.checkpoints import get_latest_checkpoint
from fabjax.utils.jax_util import get_leading_axis_tree
from fabjax.utils.loggers import Logger

from FABdiffME.train.init_and_step_state import (
    TrainingState,
    InitStateFn,
    UpdateStateFn,
)
from FABdiffME.utils.loggers import ListLogger

PlotFn = Callable[[TrainingState, chex.PRNGKey, int], None]
EvalFn = Callable[[TrainingState, chex.PRNGKey], dict]
FinalEvalAndPlotFn = Callable[[TrainingState, dict, chex.PRNGKey, list], dict]


class TrainConfig(NamedTuple):
    seed: int
    init_state: InitStateFn
    update_state: UpdateStateFn
    n_iterations: int
    use_lr_schedule: bool
    lr: optax.Schedule
    eval_fn: EvalFn
    plot_fn: PlotFn
    final_eval_and_plot_fn: FinalEvalAndPlotFn
    save: bool
    n_eval_and_plot: int
    eval_and_plot_scale: str
    checkpoints_dir: str
    n_checkpoints: int
    use_64_bit: bool = True
    logger: Logger = ListLogger(save_period=1)
    resume: bool = False
    runtime_limit: Optional[float] = None


def train(config: TrainConfig) -> Tuple[TrainingState, Dict]:
    if config.runtime_limit:
        start_time = time.time()

    if config.use_64_bit:
        jax.config.update("jax_enable_x64", True)

    # Initialize flow
    key = jax.random.PRNGKey(config.seed)
    key, subkey = jax.random.split(key)
    state = config.init_state(subkey)

    # Get iterations for checkpointing
    checkpoint_iter_np = np.linspace(
        0, config.n_iterations - 1, config.n_checkpoints, dtype="int"
    )
    checkpoint_iter = list(checkpoint_iter_np)
    print("checkpoints", checkpoint_iter[:10], len(checkpoint_iter))
    print("resume", config.resume)
    print("tlimit", config.runtime_limit)

    # Get iterations for logging (i.e. evaluation and plotting)
    if config.eval_and_plot_scale == "log":
        eval_and_plot_iter = list(
            np.unique(
                np.geomspace(
                    1, config.n_iterations - 1, num=config.n_eval_and_plot, dtype="int"
                )
            )
        )
        eval_and_plot_iter.append(0)
        count = 0
        while len(eval_and_plot_iter) < config.n_eval_and_plot:
            rand_int = np.random.randint(0, config.n_iterations - 1)
            if rand_int not in eval_and_plot_iter:
                eval_and_plot_iter.append(rand_int)
            if count > 1e5:
                print(
                    "Not possible to generate log iterator for evaluation and plotting with n=",
                    config.n_eval_and_plot,
                )
            count += 1
    elif config.eval_and_plot_scale == "linear":
        eval_and_plot_iter = list(
            np.linspace(0, config.n_iterations - 1, config.n_eval_and_plot, dtype="int")
        )
    else:
        print(
            f"eval_and_plot_scale={config.eval_and_plot_scale} not in [linear, log], defaulting to linear"
        )
        eval_and_plot_iter = list(
            np.linspace(0, config.n_iterations - 1, config.n_eval_and_plot, dtype="int")
        )
    eval_and_plot_iter = np.sort(eval_and_plot_iter)

    start_iter = 0
    if config.resume:
        # Load latest checkpoint
        latest_cp = get_latest_checkpoint(config.checkpoints_dir, key="state_")
        if latest_cp:
            try:
                start_iter = int(latest_cp[-12:-4]) + 1
                with open(latest_cp, "rb") as f:
                    state = pickle.load(f)
            except:
                start_iter = (
                    int(latest_cp[-(4 + len(str(config.n_iterations))) : -4]) + 1
                )
                with open(latest_cp, "rb") as f:
                    state = pickle.load(f)
            print(f"Loaded checkpoint {latest_cp}")
            if len(jax.devices()) > 1:
                state = jax.pmap(
                    lambda key_: state.__class__(state.params, state.opt_state, key_)
                )(jax.random.split(key, len(jax.devices())))
        else:
            print("No checkpoint found, starting training from scratch")

    key, subkey = jax.random.split(key)
    pbar = tqdm(range(start_iter, config.n_iterations))
    iteration = start_iter
    for iteration in pbar:
        # Train
        state, info = config.update_state(state)

        # Write training info to logger.history
        # Check for scalar info -- usually if last batch info is active
        leading_info_shape = get_leading_axis_tree(info, 1)
        if len(leading_info_shape) == 0 or leading_info_shape == (1,):
            info.update(iteration=iteration)
            config.logger.write(info)
        else:
            for batch_idx in range(leading_info_shape[0]):
                batch_info = jax.tree_map(lambda x: x[batch_idx], info)
                batch_info.update(iteration=iteration)
                config.logger.write(batch_info)

        # Evaluate and Plot
        if config.save is True and iteration in eval_and_plot_iter:
            # Evaluate
            if config.eval_fn is not None:
                key, subkey = jax.random.split(key)
                eval_info = config.eval_fn(state, subkey)
                if config.use_lr_schedule:
                    total_steps = state.opt_state.total_steps
                    current_lr = config.lr(total_steps)
                    eval_info.update(lr=current_lr)
                eval_info.update(eval_and_plot_iter=iteration)
                config.logger.write(eval_info)

            # Plot
            if config.plot_fn is not None:
                key, subkey = jax.random.split(key)
                config.plot_fn(state, subkey, iteration)

                # Save permanent checkpoint
                eval_state_str = "evalstate_%0{}i.pkl".format(
                    len(str(config.n_iterations))
                )
                checkpoint_path = os.path.join(
                    config.checkpoints_dir, eval_state_str % iteration
                )
                with open(checkpoint_path, "wb") as f:
                    if len(jax.devices()) > 1:
                        state_first = jax.tree_map(lambda x: x[0], state)
                        pickle.dump(state_first, f)
                    else:
                        pickle.dump(state, f)

        if config.save is True and iteration in checkpoint_iter:
            print("Ckpt iteration:", iteration)
            # Get previous checkpoint
            # previous_cp = get_latest_checkpoint(config.checkpoints_dir, key="state_")
            # Save new checkpoint
            if len(str(config.n_iterations)) > 8:
                state_str = "state_%0{}i.pkl".format(len(str(config.n_iterations)))
                checkpoint_path = os.path.join(
                    config.checkpoints_dir, state_str % iteration
                )
            else:
                checkpoint_path = os.path.join(
                    config.checkpoints_dir, "state_%08i.pkl" % iteration
                )
            with open(checkpoint_path, "wb") as f:
                if len(jax.devices()) > 1:
                    state_first = jax.tree_map(lambda x: x[0], state)
                    pickle.dump(state_first, f)
                else:
                    pickle.dump(state, f)
            # Delete previous checkpoint
            # if previous_cp:
            #    os.remove(previous_cp)
            # Check whether job hit the runtime limit
            if (
                config.runtime_limit
                and iteration > 0
                and np.any(checkpoint_iter_np > iteration)
            ):
                next_checkpoint_iter = np.min(
                    checkpoint_iter_np[checkpoint_iter_np > iteration]
                )
                time_diff = (time.time() - start_time) / 3600
                if (
                    time_diff
                    * (next_checkpoint_iter - start_iter)
                    / (iteration - start_iter)
                    > config.runtime_limit
                ):
                    pbar.write("Reached break condition")
                    break

    pbar.write("Saving stuff after training")
    if config.save is True and config.final_eval_and_plot_fn is not None:
        # Final evaluation
        key, subkey = jax.random.split(key)
        test_info = config.final_eval_and_plot_fn(
            state, config.logger.history, subkey, eval_and_plot_iter
        )
        if test_info is not {}:
            config.logger.write(test_info)

    # Save params in final iteration
    if iteration == config.n_iterations - 1:
        path_params = os.path.join(config.checkpoints_dir, "params.pkl")
        with open(path_params, "wb") as f:
            try:
                pickle.dump(state.params, f)
            except:
                pickle.dump(state.flow_params, f)

    config.logger.close()
