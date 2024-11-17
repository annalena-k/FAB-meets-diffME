from typing import NamedTuple
import argparse
from datetime import timedelta
import jax
import jax.numpy as jnp
import numpy as np
import os
import pickle
import time
import vegas
from pathlib import Path

from fabjax.utils.loggers import Logger
from fabjax.sampling.resampling import log_effective_sample_size

from FABdiffME.utils.config import get_config, setup_directories
from FABdiffME.utils.loggers import ListLogger
from FABdiffME.targets.target_util import load_target, ProbTargetFn
from FABdiffME.sampling.metrics import log_sampling_efficiency, log_unweighting_efficiency

class VegasConfig(NamedTuple):
    n_itn_warmup: int
    neval_warmup: int
    n_itn: int
    neval: int
    alpha: float
    maxinc_axis: int
    prob_fn: ProbTargetFn
    dim: int
    ckpt_dir: str
    logger: Logger=ListLogger(save_period=1)


def setup_target_prob_jitted(config_target: dict):
    name = config_target["name"]
    if name == "pi1800" or name == "lambdac":
        target = load_target(name=name, dim=config_target["dim"])
    elif name == "ee_to_mumu" or name == "ee_to_ttbar_wb" or name == "ee_to_ttbar":
        target = load_target(
            name=name,
            dim=config_target["dim"], 
            center_of_mass_energy=config_target.get("center_of_mass_energy", 1000), # GeV 
            model_parameters=config_target.get("model_parameters", {}),
            epsilon_boundary=config_target.get("epsilon_boundary", 1e-8)
            )
    else:
        print(f"{name} not within [pi1800, lambdac, ee_to_mumu, ee_to_ttbar_wb, ee_to_ttbar].")

    prob_jit = jax.jit(target.prob_no_nan)

    return prob_jit


def setup_config_vegas(config: dict, dirs: dict) -> VegasConfig:
    # Setup target
    probs_jit = setup_target_prob_jitted(config_target=config['target'])

    # Logging
    save_root = dirs.log_dir + "/history.pkl"
    logger = ListLogger(save, save_root)

    vegas_config = VegasConfig(
        n_itn_warmup = config['vegas'].get('iterations_warmup', 10),
        neval_warmup=config['vegas'].get('neval_warmup', 1e3),
        n_itn=config['vegas'].get('iterations', 10),
        neval=config['vegas'].get('neval', 2e5),
        alpha=config['vegas'].get('alpha', 0.5),
        maxinc_axis=config['vegas'].get('maxinc_axis', 64),
        prob_fn=probs_jit,
        dim=config['target']['dim'],
        logger=logger,
        ckpt_dir = dirs.ckpt_dir
        )
    return vegas_config


def run_vegas(config: VegasConfig):

    info = {}
    # Optimize one working integ
    vegas_start = time.perf_counter()
    # Setup VEGAS integrator
    integ = vegas.Integrator([[0., 1.]]*config.dim)
    if config.n_itn_warmup > 0 and config.neval_warmup > 0:
        # Step 1 -- Adapt to f; discard results
        integ(config.prob_fn, nitn=config.n_itn_warmup, neval=config.neval_warmup)
    # Step 2 -- Run VEGAS; collect results
    result = integ(
        config.prob_fn, 
        nitn=config.n_itn, 
        neval=config.neval, 
        alpha=config.alpha, 
        maxinc_axis=config.maxinc_axis
        )
    print(result.summary())
    info.update(vegas_result_summary=result.summary())

    diff = time.perf_counter() - vegas_start
    vegas_time = timedelta(seconds=diff)
    print("VEGAS run took:", vegas_time)
    info.update(time=vegas_time)
    
    # Save number of function evaluation and integral value
    info.update(
        vegas_sum_neval=result.sum_neval,
        integral_vegas=result.itn_results[-1].mean,
        std_integral_vegas=result.itn_results[-1].sdev,
        )
    
    # Generate samples
    samples_v, weights_v = [], []
    for x, w in integ.random_batch():
        # Copy array over due to bug in vegas (work-around by Peter Lepage)
        x = np.array(x)
        w = np.array(w)
        samples_v.append(x)
        weights_v.append(w * config.prob_fn(jnp.array(x)))
    samples = jnp.vstack(samples_v)
    weights = jnp.hstack(weights_v)
    # Calculate ESS
    n_eff = np.exp(log_effective_sample_size(log_weights=jnp.log(weights)))
    # Calculate efficiencies
    epsilon = np.exp(log_sampling_efficiency(log_weights=jnp.log(weights)))
    epsilon_uw = np.exp(log_unweighting_efficiency(log_weights=jnp.log(weights)))
    # Estimate integral with weights
    integral_weights = np.sum(weights)
    # Calculate unnormalized standard deviation
    sum_squared_x = np.sum(weights**2)
    unnormalized_variance = sum_squared_x - (integral_weights**2 / len(weights))
    unnormalized_std_dev_integral_weights = np.sqrt(unnormalized_variance)
    info.update(
        ess=n_eff,
        epsilon=epsilon,
        epsilon_uw=epsilon_uw,
        integral_weights=integral_weights,
        std_integral_weights=unnormalized_std_dev_integral_weights,
        n_samples_used_in_estimation=len(weights),
        )
    print("-------RESULTS-------")
    print("VEGAS integral:", result.itn_results[-1].mean, "+/-", result.itn_results[-1].sdev, "based on", result.sum_neval, "function evals.")
    print("Integral based on", len(weights),"weights:", integral_weights, "+/-", unnormalized_std_dev_integral_weights)
    print("ESS:", info['ess'], "epsilon:", epsilon, "epsilon_uw:", epsilon_uw)

    # Save integ and final result
    filename_integ = f'{config.ckpt_dir}/integ.pkl'
    with open(filename_integ, 'wb') as f:
        pickle.dump(integ, file=f)
    filename_res = f'{config.ckpt_dir}/final_result.pkl'
    with open(filename_res, 'wb') as f:
        pickle.dump(result, file=f)

    # Save samples and weights
    filename_samples = f'{config.ckpt_dir}/samples.pkl'
    data = {
        "samples": samples,
        "weights": weights
    }
    with open(filename_samples, 'wb') as f:
        pickle.dump(data, file=f)
    print(f"Saved samples to {filename_samples}")

    config.logger.write(info)
    config.logger.close()


if __name__ == '__main__':
    # Parse input arguments
    config_file_name = 'config_vegas.yaml'
    folder_path = './paper_experiments/2d/lambdac/vegas/0'
    parser = argparse.ArgumentParser(description='Run VEGAS optimization')
    parser.add_argument('--config', type=str, default=f'{folder_path}/{config_file_name}',
                        help='Path to config file specifying VEGAS parameters')
    args = parser.parse_args()

    # Load config
    config = get_config(args.config)

    config_file_path = Path(args.config)
    folder_path = config_file_path.parent.absolute()

    # Make directories & copy config file to save_root folder
    root = os.path.abspath(config['vegas']['save_root'])
    save = config['vegas']['save']
    resume = False
    dirs = setup_directories(root, folder_path, config_file_name, save, resume)

    # Run VEGAS
    vegas_config = setup_config_vegas(config, dirs)
    run_vegas(vegas_config)
