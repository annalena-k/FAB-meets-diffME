import jax.numpy as jnp

from FABdiffME.targets.madjax_target import MadjaxTarget
from FABdiffME.targets.target_util import load_target


def test_ee_to_ttbar_wb_target():
    dim = 8
    name = "ee_to_ttbar_wb"
    center_of_mass_energy = 1000  # [GeV]
    model_parameters = {}
    epsilon = 1e-8
    target = MadjaxTarget(
        dim=dim,
        name=name,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters=model_parameters,
        epsilon_boundary=epsilon,
    )
    eval_target = target.log_prob(jnp.array([0.5] * dim))

    target_load = load_target(
        name=name,
        dim=dim,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters=model_parameters,
        epsilon_boundary=epsilon,
    )
    eval_target_loaded = target_load.log_prob(jnp.array([0.5] * dim))

    assert eval_target == eval_target_loaded


def test_ee_to_ttbar_target():
    dim = 14
    name = "ee_to_ttbar"
    center_of_mass_energy = 1000  # [GeV]
    model_parameters = {}
    epsilon = 1e-8
    target = MadjaxTarget(
        dim=dim,
        name=name,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters=model_parameters,
        epsilon_boundary=epsilon,
    )
    eval_target = target.log_prob(jnp.array([0.5] * dim))

    target_load = load_target(
        name=name,
        dim=dim,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters=model_parameters,
        epsilon_boundary=epsilon,
    )
    eval_target_loaded = target_load.log_prob(jnp.array([0.5] * dim))

    assert eval_target == eval_target_loaded
