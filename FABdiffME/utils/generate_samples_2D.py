import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import vegas

from FABdiffME.targets.target_util import load_target

# File to generate samples from a specific matrix element (me)


def generate_samples_2D(
    n_samples: int,
    n_samples_per_file: int,
    begin_file: int,
    name: str,
    data_type: str,
    data_folder: str,
    dim: int,
    use_vegas: bool,
    center_of_mass_energy: float = None,
    model_parameters: dict = None,
    n_itn: int = None,  # = 10
    n_eval1: int = None,  # = 5e4
    n_eval2: int = None,  # = 4e4
):
    assert dim == 2
    # Parameters
    n_files = int(n_samples / n_samples_per_file)

    if (name == "pi1300" or name == "lambdac") and dim == 2:
        target = load_target(name=name, path_to_target=None)
    elif name == "ee_to_mumu":
        target = load_target(
            name=name,
            dim=dim,
            center_of_mass_energy=center_of_mass_energy,
            model_parameters=model_parameters,
        )
    elif name == "gaussian":
        target = load_target(name=name, dim=dim)
    else:
        print(f"{name} not within [pi1300, lambdac, ee_to_mumu, gaussian].")

    # Different keys for training, validation & test data
    if data_type == "train":
        key = random.PRNGKey(1)
        # Different keys for each batch of generated training data
        if begin_file != 0:
            key = random.PRNGKey(begin_file)
    elif data_type == "val":
        key = random.PRNGKey(2)
    else:
        key = random.PRNGKey(3)

    # Create folder for storing the samples
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    if use_vegas:
        # Check if vegas integrator already exists
        vegas_filename = data_folder + f"/vegas_integ.pkl"
        if os.path.exists(vegas_filename):
            with open(vegas_filename, "rb") as f:
                result, integ, neval_total = pickle.load(f)
        # Run vegas if it does not exist
        else:
            good_result = True
            while good_result:
                prob_fn = jax.jit(target.prob_no_nan)
                #  Setup VEGAS integrator
                integ = vegas.Integrator([[0.0, 1.0]] * target.dim)

                # Run VEGAS
                # 1) Adapt to prob_n, discard results
                result1 = integ(prob_fn, nitn=n_itn, neval=n_eval1)
                # 2) Integ has adapted to prob_fn, keep results
                result2 = integ(prob_fn, nitn=n_itn, neval=n_eval2)
                neval_total = result1.sum_neval + result2.sum_neval
                # Check if result converged
                if result2.Q > 0.1:
                    good_result = False
                print(result2.summary())

            # Save result
            with open(vegas_filename, "wb") as f:
                pickle.dump([result2, integ, neval_total], f)

    key, subkey = random.split(key)
    full_no_target_evals = []
    for i in range(n_files):
        # Generate data if file does not exist already
        if use_vegas:
            filename = (
                data_folder
                + f"/{data_type}_data_vegas_{n_samples_per_file}_{begin_file + i}.npy"
            )
        else:
            filename = (
                data_folder
                + f"/{data_type}_data_unit_{n_samples_per_file}_{begin_file + i}.npy"
            )
        if not os.path.exists(filename):
            print(i)
            if use_vegas:
                no_target_evals = 0
                samples = integ.map.random(n_samples_per_file)
                no_target_evals += neval_total
            else:
                key, subkey = random.split(key)
                samples, no_target_evals = target.sample(subkey, n_samples_per_file)
            full_no_target_evals.append(no_target_evals)
            # save data
            with open(filename, "wb") as f:
                jnp.save(f, samples)
        else:
            print("File ", filename, " already exists, not overwriting it.")
    n_target_evals = np.array(full_no_target_evals)
    all_evals = np.sum(n_target_evals)
    print("Number of target evaluations:", all_evals, "for", n_samples, "samples")
    # save number of target evaluations
    if use_vegas:
        filename = (
            data_folder
            + f"/{data_type}_number_of_target_evals_vegas_{begin_file}_{begin_file + n_files - 1}.txt"
        )
    else:
        filename = (
            data_folder
            + f"/{data_type}_number_of_target_evals_{begin_file}_{begin_file + n_files - 1}.txt"
        )
    np.savetxt(filename, n_target_evals)

    print("Done")
