from typing import Callable, Tuple
import jax
import jax.numpy as jnp
import chex
import madjax
import pylhe
import gzip
import pickle

LogProbTargetFn = Callable[[chex.Array], chex.Array]
ProbTargetFn = Callable[[chex.Array], chex.Array]
SampleTargetFn = Callable[[int], chex.Array]


class Target:
    dim: int
    name: str
    log_prob: LogProbTargetFn
    prob: ProbTargetFn
    samples: SampleTargetFn


class HEP2DTarget(Target):
    evaluate_DP_on_physical_space: Callable[[chex.Array], Tuple[chex.Array, chex.Array]]
    transform_q_to_physical_space: Callable[
        [chex.Array, chex.Array], Tuple[chex.Array, chex.Array]
    ]
    transform_samples_to_physical_space: Callable[[chex.Array], chex.Array]


def kallen(x, y, z):
    return x**2 + y**2 + z**2 - 2 * x * y - 2 * y * z - 2 * z * x


def sqrtkallen(x, y, z):
    kall = kallen(x, y, z)
    return jnp.sqrt(kall)


def move_away_from_boundary(samples, epsilon):
    return jnp.minimum(
        jnp.maximum(samples, jnp.ones_like(samples) * epsilon),
        jnp.ones_like(samples) * (1.0 - epsilon),
    )


def load_target(
    name: str,
    path_to_target: str = None,
    dim: int = None,
    center_of_mass_energy: float = None,
    model_parameters: dict = None,
    epsilon_boundary: float = None,
) -> Target:
    """Load specific target
    name: str, one of ["pi1800", "lambdac", "ee_to_mumu"]
    path_to_target: str
            for lambdac: path is folder path, e.g. diffME/targets/lambdac_model.pkl
            for ee_to_mumu: path is module path, e.g. diffME.targets.madjax_ee_to_mumu
    dim: int, dimensionality of matrix element
    center_of_mass_energy: float, specifies center of mass energy of scattering
    model_parameters: dict, ={} for default parameters, see ee_to_mumu_test.ipynb for non-standard example
    """
    if name == "pi1800":
        assert dim == None or dim == 2
        from diffME.targets.pi1300 import PI1800

        target = PI1800()
    elif name == "lambdac":
        # path is folder path, e.g. diffME/targets/lambdac_model.pkl
        assert dim == None or dim == 2
        from diffME.targets.lambdac import Lambdac

        target = Lambdac(path_to_ComPWA_matrix_element=path_to_target)
    elif name == "ee_to_mumu":
        # path is module path, e.g. diffME.targets.madjax_ee_to_mumu
        assert dim is not None and center_of_mass_energy is not None, (
            model_parameters is not None
        )
        from diffME.targets.ee_to_mumu import EeToMumu

        target = EeToMumu(
            path_to_target,
            dim,
            center_of_mass_energy,
            model_parameters,
            epsilon_boundary,
        )
    elif name == "ee_to_ttbar" or name == "ee_to_ttbar_wb":
        # path is module path, e.g. diffME.targets.madjax_ee_to_ttbar
        assert dim is not None and center_of_mass_energy is not None, (
            model_parameters is not None
        )
        from diffME.targets.madjax_target import MadjaxTarget

        target = MadjaxTarget(
            path_to_target,
            dim,
            name,
            center_of_mass_energy,
            model_parameters,
            epsilon_boundary,
        )
    elif name == "gaussian":
        assert dim is not None
        from diffME.targets.gaussian import Gaussian

        target = Gaussian(dim)
    else:
        print("Type of matrix element not part of [pi1800, lambdac, ee_to_mumu]")
        target = None
    return target


def read_madgraph_phasespace_points(filename: str, target: Target, n_events: int):
    """Read lhe (=Les Houches Event) file and convert events to phase space points.
    filename: either .lhe or .lhe.gz
    target: Target that has the method 'get_phase_space_generator'
    n_events: number of events to load
    """

    def extract_incoming_and_outgoing_particles(event):
        particles = []
        for p in event.particles:
            if (
                p.status == -1 or p.status == 1
            ):  # -1: incoming, 1: outgoing & stable particles
                particles.append([p.e, p.px, p.py, p.pz])
        return jnp.array(particles)

    def get_n_lhe_events(lhe_event_generator, n_events: int):
        events = []
        for _ in range(n_events):
            try:
                event = lhe_event_generator.__next__()
            except:
                print(f"No More Events. Total number: {len(events)}")
                return jnp.array(events)

            relevant_particles = extract_incoming_and_outgoing_particles(event)
            events.append(relevant_particles)

        return jnp.array(events)

    def get_phase_space_points_from_events(events: chex.Array):
        momenta_vec = [madjax.phasespace.vectors.Vector(p) for p in events]
        ps_points, _ = target.phase_space_generator.invertKinematics(
            target.E_cm, momenta_vec
        )
        return jnp.array(ps_points)

    get_phase_space_points_from_event_vec = jax.vmap(
        get_phase_space_points_from_events, (0)
    )

    lhe_events = pylhe.read_lhe_with_attributes(filename)
    events = get_n_lhe_events(lhe_events, n_events)
    if events.shape == (0,):
        raise TypeError(f"Momenta array is empty with shape {events.shape}.")
    ps_points = get_phase_space_points_from_event_vec(events)

    return ps_points


def read_preprocssed_phasespace_points(filename: str, n_events: int):
    """Read pkl file that contains preprocessed phasespace points based on examples/preprocess_lhe_files.py.
    Input:
        filename: path and filename of .pkl
        n_events: number of events to load
    Output:
        phase space points of shape [n_events, dim]
    """
    with gzip.open(filename, "rb") as f:
        moms = pickle.load(f)
    assert (
        len(moms) >= n_events
    ), f"File {filename} contains less than the requested {n_events} samples."
    return moms[:n_events]
