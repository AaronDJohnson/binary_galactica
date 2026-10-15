"""Invalid source domains fail clearly without breaking JAX transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass.response import polarization_tensor


# One case per guard: finite parameters, positive f0, and the cosine bound,
# which is checked for both the sky (1) and inclination (4) cosines.
@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (0, 0.0, "f0 must be positive"),
        (1, 1.01, "cos"),
        (4, -1.01, "cos"),
        (3, np.nan, "params must be finite"),
    ],
)
def test_eager_native_source_rejects_invalid_domains(
    glass_params, index, value, message
):
    bad = glass_params.copy()
    bad[index] = value
    with pytest.raises(ValueError, match=message):
        bg.ucb_waveform(bad, T=7_884_000.0, BW=32)


# T, t0 and q share one scalar check (finite, then scalar shape, then
# positive when required); its three messages are covered on T, and t0 and q
# each keep one case for their wiring. q adds its own integrality guard.
@pytest.mark.parametrize(
    ("duration", "message"),
    [
        (0.0, "T must be positive"),
        (np.nan, "T must be finite"),
        ([1024.0], "T must be a scalar"),
    ],
)
def test_eager_duration_is_finite_positive_scalar(glass_params, duration, message):
    with pytest.raises(ValueError, match=message):
        bg.ucb_waveform(glass_params, T=duration, BW=32)


def test_eager_epoch_is_finite_scalar(glass_params):
    with pytest.raises(ValueError, match="t0 must be finite"):
        bg.ucb_waveform(glass_params, T=7_884_000.0, BW=32, t0=np.inf)


@pytest.mark.parametrize(
    ("carrier", "message"),
    [(7884.5, "q must be an integral"), ([7884.0], "q must be a scalar")],
)
def test_eager_carrier_override_is_finite_integral_scalar(
    glass_params, carrier, message
):
    with pytest.raises(ValueError, match=message):
        bg.ucb_waveform(glass_params, T=7_884_000.0, BW=32, q=carrier)


def test_eager_cached_positions_must_be_finite(glass_params):
    positions = np.asarray(bg.orbit_cache(7_884_000.0, 0.0, 32)).copy()
    positions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="positions.*finite"):
        bg.ucb_waveform(glass_params, T=7_884_000.0, BW=32, positions=positions)


def test_traced_cached_positions_must_be_finite(glass_params):
    positions = np.asarray(bg.orbit_cache(7_884_000.0, 0.0, 32)).copy()
    positions[0, 0, 0] = np.inf
    evaluate = jax.jit(
        lambda orbit: bg.ucb_waveform(
            glass_params, T=7_884_000.0, BW=32, positions=orbit
        )
    )
    for output in evaluate(positions).values():
        assert np.all(np.isnan(np.asarray(output)))


def test_pole_guard_does_not_project_out_of_domain_sky_to_a_finite_response():
    # The guard tests costh**2, so one sign covers both.
    for value in polarization_tensor(1.01, 0.7):
        assert not np.all(np.isfinite(np.asarray(value)))


def test_traced_invalid_native_sources_return_nan(glass_params):
    """Each traced domain condition alone must poison only its own row.

    Rows: valid, sky cosine > 1, inclination cosine < -1, negative frequency.
    One batch shares a single compiled program.
    """
    sources = np.tile(glass_params, (4, 1))
    for row, (index, value) in enumerate([(1, 1.01), (4, -1.01), (0, -7884.0)], 1):
        sources[row, index] = value
    evaluate = jax.jit(jax.vmap(lambda p: bg.ucb_waveform(p, T=7_884_000.0, BW=32)))
    result = evaluate(jnp.asarray(sources))
    for output in result.values():
        assert np.all(np.isfinite(np.asarray(output[0])))
        assert np.all(np.isnan(np.asarray(output[1:])))


@pytest.mark.parametrize("kwargs", [{"T": -1.0}, {"t0": np.inf}, {"q": 7884.5}])
def test_traced_invalid_grid_values_return_nan(glass_params, kwargs):
    arguments = {"T": 7_884_000.0, "t0": 0.0, "q": 7884.0, **kwargs}
    evaluate = jax.jit(
        lambda duration, epoch, carrier: bg.ucb_waveform(
            glass_params, T=duration, BW=32, t0=epoch, q=carrier
        )
    )
    result = evaluate(arguments["T"], arguments["t0"], arguments["q"])
    for output in result.values():
        assert np.all(np.isnan(np.asarray(output)))


def test_cosine_endpoints_remain_valid(glass_params):
    # The bounds compare |cos| <= 1, so one sign per cosine covers both.
    p = glass_params.copy()
    p[[1, 4]] = [1.0, -1.0]
    result = bg.ucb_waveform(p, T=7_884_000.0, BW=32)
    for output in result.values():
        assert np.all(np.isfinite(np.asarray(output)))


# One case per guard and message; amp and f0 share a message but are separate
# conditions.
@pytest.mark.parametrize(
    "field,value,message",
    [
        ("amp", 0.0, "amp and f0 must be positive"),
        ("f0", -1e-3, "amp and f0 must be positive"),
        ("lam", np.inf, "theta must be finite"),
        ("iota", np.pi + 0.1, r"iota must lie in \[0, pi\]"),
        ("beta", -np.pi / 2 - 0.1, r"beta must lie in \[-pi/2, pi/2\]"),
    ],
)
def test_eager_physical_sources_reject_invalid_domains(params, field, value, message):
    with pytest.raises(ValueError, match=message):
        bg.tdi_waveform(params._replace(**{field: value}), T=7_884_000.0, BW=32)


def test_physical_wrapper_validates_observation_grid(params):
    with pytest.raises(ValueError, match="T must be"):
        bg.tdi_waveform(params, T=-1.0, BW=32)


# Two cases reach all four closed endpoints of iota and beta.
@pytest.mark.parametrize(
    "inclination,latitude", [(0.0, -np.pi / 2), (np.pi, np.pi / 2)]
)
def test_physical_angle_endpoints_remain_valid(params, inclination, latitude):
    p = params._replace(iota=inclination, beta=latitude)
    for output in bg.tdi_waveform(p, T=7_884_000.0, BW=32).values():
        assert np.all(np.isfinite(np.asarray(output)))
