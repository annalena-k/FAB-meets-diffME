import jax.numpy as jnp

from FABdiffME.targets.crossed_ring import CrossedRing
from FABdiffME.targets.gaussian import Gaussian
from FABdiffME.targets.pi1300 import PI1300
from FABdiffME.targets.lambdac import Lambdac
from FABdiffME.targets.ee_to_mumu import EeToMumu
from FABdiffME.targets.target_util import load_target

# Tests for madjax target need to be in separate file, because some internal functions for ee_to_mumu do not
# automatically get overwritten when loading another madjax matrix element directly afterwards.


def test_load_gaussian():
    dim = 2
    target = load_target(name="gaussian", dim=dim)
    v1 = target.log_prob(jnp.array([0.5] * dim))
    target = Gaussian(dim=dim)
    v2 = target.log_prob(jnp.array([0.5] * dim))

    assert v1 == v2


def test_load_crossed_ring():
    dim = 2
    target = load_target(name="crossed_ring", dim=dim)
    v1 = target.log_prob(jnp.array([0.5] * dim))
    target = CrossedRing(dim=dim)
    v2 = target.log_prob(jnp.array([0.5] * dim))

    assert v1 == v2


def test_load_pi1300():
    dim = 2
    target = load_target(name="pi1300", dim=dim)
    v1 = target.log_prob(jnp.array([0.5] * dim))
    target = PI1300()
    v2 = target.log_prob(jnp.array([0.5] * dim))

    assert v1 == v2


def test_load_lambdac():
    dim = 2
    target = load_target(name="lambdac", dim=dim)
    v1 = target.log_prob(jnp.array([0.5] * dim))
    target = Lambdac()
    v2 = target.log_prob(jnp.array([0.5] * dim))

    assert v1 == v2


def test_load_ee_to_mumu():
    dim = 2
    center_of_mass_energy = 1000  # [GeV]
    model_parameters = {}

    target = load_target(
        name="ee_to_mumu",
        dim=dim,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters=model_parameters,
    )
    v1 = target.log_prob(jnp.array([0.5] * dim))

    target = EeToMumu(
        dim=dim,
        center_of_mass_energy=center_of_mass_energy,
        model_parameters=model_parameters,
    )
    v2 = target.log_prob(jnp.array([0.5] * dim))

    assert v1 == v2
