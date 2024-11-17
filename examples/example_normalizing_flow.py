from typing import Dict
import argparse
from functools import partial
import os
from pathlib import Path
import pickle

from fabjax.utils.optimize import get_optimizer, OptimizerConfig
from fabjax.utils.checkpoints import get_latest_checkpoint
from fabjax.train import (
    build_fab_no_buffer_init_step_fns,
    build_fab_with_buffer_init_step_fns,
)
from fabjax.buffer.prioritised_buffer import build_prioritised_buffer
from fabjax.sampling import build_smc, build_blackjax_hmc, build_metropolis
from fabjax.sampling import default_point_is_valid_fn, point_is_valid_if_in_bounds_fn

from FABdiffME.flow.build_flow import build_flow
from FABdiffME.flow.spline_flow import FlowConfig
from FABdiffME.targets.target_util import load_target
from FABdiffME.train.init_and_step_state import (
    build_fkld_init_and_step,
    build_rkld_init_and_step,
)
from FABdiffME.train.generic_training_loop import (
    TrainConfig,
    train,
)
from FABdiffME.utils.config import get_config, setup_directories
from FABdiffME.utils.setup_eval_and_plot_fns import *
from FABdiffME.utils.loggers import ListLogger
from FABdiffME.utils.load_data import load_data

# Train normalizing flow with
# (1) forward Kullback-Leibler divergence
# (2) reverse Kullback-Leibler divergence
# (3) Flow Annealed Importance Sampling Bootstrap


def setup_train_config(
    config: Dict, dirs: Dict, resume: bool, runtime_limit: float
) -> TrainConfig:

    # Load parameters for target
    n_data_val = config["target"]["n_samples_val"]
    n_data_test = config["target"]["n_samples_test"]
    n_samples_per_file = config["target"]["n_samples_per_file"]
    data_dir = config["target"]["data_dir"]
    generator = config["target"]["generator"]
    # Load parameters for training
    lr = config["training"]["lr"]
    batch_size = config["training"]["batch_size"]
    seed = config["training"]["seed"]
    plot_batch_size = config["training"]["plot_batch_size"]
    n_samples_ESS = config["training"]["n_samples_ESS"]
    n_eval_and_plot = config["training"]["n_eval_and_plot"]
    eval_and_plot_scale = config["training"]["eval_and_plot_scale"]
    n_checkpoints = config["training"]["n_checkpoints"]
    use_64_bit = config["training"]["use_64_bit"]

    # Setup target.
    dim = config["target"]["dim"]
    name = config["target"]["name"]
    if name == "pi1300" or name == "lambdac":
        target = load_target(name=name, dim=dim)
    elif name == "ee_to_mumu" or name == "ee_to_ttbar" or name == "ee_to_ttbar_wb":
        com_energy = config["target"]["center_of_mass_energy"]
        model_params = dict(config["target"]["model_parameters"])
        epsilon_boundary = config["target"]["epsilon_boundary"]
        target = load_target(
            name=name,
            dim=dim,
            center_of_mass_energy=com_energy,
            model_parameters=model_params,
            epsilon_boundary=epsilon_boundary,
        )
    elif name == "gaussian":
        target = load_target(name=name, dim=dim)
    else:
        print(
            f"{name} not within [gaussian, pi1300, lambdac, ee_to_mumu, ee_to_ttbar, ee_to_ttbar_wb]."
        )

    # Setup flow
    flow_config = FlowConfig(
        dim=dim,
        n_trafos=config["model"]["n_trafos"],
        n_bijector_params=config["model"]["n_bijector_params"],
        hidden_layer_sizes=config["model"]["hidden_layer_sizes"],
    )
    flow = build_flow(flow_config)

    # Setup optimizer
    use_lr_schedule = config["training"]["use_schedule"]
    if use_lr_schedule:
        n_iter_total = config["training"]["n_iter_total"]
        n_iter_warmup = config["training"]["n_iter_warmup"]
        optimizer_config = OptimizerConfig(
            init_lr=config["training"]["lr"],
            use_schedule=config["training"]["use_schedule"],
            n_iter_total=n_iter_total,
            n_iter_warmup=n_iter_warmup,
            peak_lr=config["training"]["peak_lr"],
            end_lr=config["training"]["end_lr"],
            dynamic_grad_ignore_and_clip=config["training"][
                "dynamic_grad_ignore_and_clip"
            ],
        )
    else:
        optimizer_config = OptimizerConfig(
            init_lr=config["training"]["lr"],
            dynamic_grad_ignore_and_clip=config["training"][
                "dynamic_grad_ignore_and_clip"
            ],
        )
    optimizer, lr = get_optimizer(optimizer_config)

    if config["training"]["type"] == "fkld":
        # Load training data
        train_data = load_data(
            data_dir, "train", generator, target, n_data_val, n_samples_per_file
        )
    # Load validation & test data and setup eval & plot functions
    if n_data_val > 0 and n_samples_ESS > 0:
        val_data = load_data(
            data_dir, "val", generator, target, n_data_val, n_samples_per_file
        )
        eval_fn = setup_eval_fn(flow, target, val_data, n_samples_ESS)
    else:
        eval_fn = None
    if n_data_test > 0 and plot_batch_size > 0:
        test_data = load_data(
            data_dir, "test", generator, target, n_data_test, n_samples_per_file
        )
        final_eval_and_plot_fn = setup_final_eval_and_plot_fn(
            flow, target, dirs, test_data, plot_batch_size, eval_and_plot_scale
        )
    else:
        final_eval_and_plot_fn = None
    # Setup function to evaluate and plot intermediate training results
    if plot_batch_size > 0:
        plot_fn = setup_intermediate_plot_fn(
            flow, target, plot_batch_size=plot_batch_size, log_dir=dirs.log_dir
        )
    else:
        plot_fn = None

    # Load function to initialize and update training state
    if config["training"]["type"] == "fkld":
        epochs = config["training"]["epochs"]
        if use_lr_schedule:
            assert n_iter_total == epochs and n_iter_warmup <= epochs
        assert n_eval_and_plot <= epochs
        n_iterations = epochs
        init, step = build_fkld_init_and_step(
            flow, target.log_prob, optimizer, batch_size, train_data
        )
        assert n_eval_and_plot <= epochs
    elif config["training"]["type"] == "rkld":
        n_iterations = config["training"]["n_iterations"]
        init, step = build_rkld_init_and_step(flow, target, optimizer, batch_size)
    elif config["training"]["type"] == "fab":
        n_iterations = config["training"]["n_iterations"]
        alpha = config["training"]["alpha"]
        use_kl_loss = False  # Include additional reverse KL loss.
        # Buffer.
        with_buffer = config["buffer"]["with_buffer"]
        buffer_max_length = batch_size * 100
        buffer_min_length = batch_size * 10
        n_updates_per_smc_forward_pass = config["buffer"][
            "n_updates_per_smc_forward_pass"
        ]
        w_adjust_clip = config["buffer"]["w_adjust_clip"]

        # SMC.
        use_resampling = config["smc"]["use_resampling"]
        use_hmc = config["smc"]["use_hmc"]
        hmc_n_outer_steps = config["smc"]["hmc_n_outer_steps"]
        hmc_init_step_size = config["smc"]["hmc_init_step_size"]
        metro_n_outer_steps = config["smc"]["metro_n_outer_steps"]
        hmc_n_inner_steps = config["smc"]["hmc_n_inner_steps"]
        metro_init_step_size = config["smc"]["metro_init_step_size"]
        point_is_valid_fn = config["smc"]["point_is_valid_fn"]

        target_p_accept = config["smc"]["target_p_accept"]
        n_intermediate_distributions = config["smc"]["n_intermediate_distributions"]
        spacing_type = config["smc"]["spacing_type"]

        # Setup smc.
        if use_hmc:
            tune_step_size = True
            transition_operator = build_blackjax_hmc(
                dim=dim,
                n_outer_steps=hmc_n_outer_steps,
                init_step_size=hmc_init_step_size,
                target_p_accept=target_p_accept,
                adapt_step_size=tune_step_size,
                n_inner_steps=hmc_n_inner_steps,
            )
        else:
            tune_step_size = False
            transition_operator = build_metropolis(
                dim,
                metro_n_outer_steps,
                metro_init_step_size,
                target_p_accept=target_p_accept,
                tune_step_size=tune_step_size,
            )

        if point_is_valid_fn["type"] == "in_bounds":
            point_is_valid_fn = partial(
                point_is_valid_if_in_bounds_fn,
                min_bounds=point_is_valid_fn["min"],
                max_bounds=point_is_valid_fn["max"],
            )
        else:
            point_is_valid_fn = default_point_is_valid_fn

        smc = build_smc(
            transition_operator=transition_operator,
            n_intermediate_distributions=n_intermediate_distributions,
            spacing_type=spacing_type,
            alpha=alpha,
            use_resampling=use_resampling,
            point_is_valid_fn=point_is_valid_fn,
        )

        # Initialize buffer & Load function to initialize and update training state
        if with_buffer:
            buffer = build_prioritised_buffer(
                dim,
                max_length=buffer_max_length,
                min_length_to_sample=buffer_min_length,
            )
            assert n_updates_per_smc_forward_pass is not None
            init, step = build_fab_with_buffer_init_step_fns(
                flow=flow,
                log_p_fn=target.log_prob,
                smc=smc,
                optimizer=optimizer,
                batch_size=batch_size,
                buffer=buffer,
                n_updates_per_smc_forward_pass=n_updates_per_smc_forward_pass,
                w_adjust_clip=w_adjust_clip,
                use_reverse_kl_loss=use_kl_loss,
            )
        else:
            buffer = None
            n_updates_per_smc_forward_pass = None
            init, step = build_fab_no_buffer_init_step_fns(
                flow,
                log_p_fn=target.log_prob,
                smc=smc,
                optimizer=optimizer,
                batch_size=batch_size,
            )

    # Logging
    save = config["training"]["save"]
    save_root = dirs.log_dir + "/history.pkl"
    if resume:
        # Load previous history and provide it to logger
        history_filename = os.path.join(dirs.log_dir, "history.pkl")
        with open(history_filename, "rb") as f:
            history = pickle.load(f)
        # Get number of last checkpoint
        latest_cp = get_latest_checkpoint(dirs.ckpt_dir, key="state_")
        if latest_cp:
            final_iter = int(latest_cp[-12:-4])
            # Remove all values that were saved after the last checkpoint
            # Assumption: checkpoint is always saved with evaluation
            for key in history.keys():
                if len(history[key]) > final_iter:
                    history[key] = history[key][: final_iter + 1]
                    print(f"discarded some history values of {key}")

        logger = ListLogger(save, save_root, history=history)
    else:
        logger = ListLogger(save, save_root)

    # Build training config
    train_config = TrainConfig(
        seed=seed,
        init_state=init,
        update_state=step,
        n_iterations=n_iterations,
        use_lr_schedule=use_lr_schedule,
        lr=lr,
        eval_fn=eval_fn,
        plot_fn=plot_fn,
        final_eval_and_plot_fn=final_eval_and_plot_fn,
        save=save,
        n_eval_and_plot=n_eval_and_plot,
        eval_and_plot_scale=eval_and_plot_scale,
        checkpoints_dir=dirs.ckpt_dir,
        n_checkpoints=n_checkpoints,
        use_64_bit=use_64_bit,
        logger=logger,
        resume=resume,
        runtime_limit=runtime_limit,
    )

    return train_config


if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(description="Train a normalizing flow")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file specifying model "
        "architecture and training procedure",
    )
    parser.add_argument(
        "--mename", type=str, default=None, help="name of matrix element"
    )
    parser.add_argument(
        "--traintype",
        type=str,
        default=None,
        help="training type, e.g. fkld, rkld, fab",
    )
    parser.add_argument("--tlimit", type=str, default=None, help="time limit")
    parser.add_argument(
        "--resume",
        type=str,
        default=False,
        help="Whether or not to load latest checkpoint and resume training",
    )
    args, _ = parser.parse_known_args()

    # Load config
    config = get_config(args.config)
    # Check if config file arguments are consistens with folder names
    if args.mename != None:
        assert config["target"]["name"] == args.mename
        assert args.mename in config["target"]["data_dir"]
        assert args.mename in config["training"]["save_root"]
    if args.traintype != None:
        assert config["training"]["type"] == args.traintype

    config_file_path = Path(args.config)
    folder_path = config_file_path.parent.absolute()

    # Make directories & copy config file to save_root folder
    root = os.path.abspath(config["training"]["save_root"])
    save = config["training"]["save"]
    resume = args.resume
    resume = bool(int(args.resume))
    dirs = setup_directories(
        root=root,
        folder_path=folder_path,
        config_file_name=args.config,
        save=save,
        resume=resume,
    )
    if args.tlimit is not None:
        tlim = float(args.tlimit)
    else:
        tlim = args.tlimit
    train_config = setup_train_config(
        config=config, dirs=dirs, resume=resume, runtime_limit=tlim
    )

    # Train flow
    train(train_config)
