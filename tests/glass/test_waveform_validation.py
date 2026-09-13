"""Invalid source domains fail clearly without breaking JAX transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass.response import polarization_tensor


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (0, 0.0, "positive"),
        (0, -1.0, "positive"),
        (0, np.inf, "finite"),
        (1, 1.01, "cos"),
        (1, -1.01, "cos"),
        (4, 1.01, "cos"),
        (4, -1.01, "cos"),
        (3, np.nan, "finite"),
        (7, np.inf, "finite"),
    ],
)
def test_eager_native_source_rejects_invalid_domains(
    glass_params, index, value, message
):
    bad = glass_params.copy()
    bad[index] = value
    with pytest.raises(ValueError, match=message):
        bg.ucb_waveform(bad, T=7_884_000.0, BW=32)


@pytest.mark.parametrize("duration", [0.0, -1.0, np.inf, np.nan, [1024.0]])
def test_eager_duration_is_finite_positive_scalar(glass_params, duration):
    with pytest.raises(ValueError, match="T must be"):
        bg.ucb_waveform(glass_params, T=duration, BW=32)


@pytest.mark.parametrize("epoch", [np.inf, np.nan, [0.0]])
def test_eager_epoch_is_finite_scalar(glass_params, epoch):
    with pytest.raises(ValueError, match="t0 must be"):
        bg.ucb_waveform(glass_params, T=7_884_000.0, BW=32, t0=epoch)


@pytest.mark.parametrize("carrier", [7884.5, np.inf, np.nan, [7884.0]])
def test_eager_carrier_override_is_finite_integral_scalar(glass_params, carrier):
    with pytest.raises(ValueError, match="q must be"):
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


@pytest.mark.parametrize("costh", [-1.01, 1.01])
def test_pole_guard_does_not_project_out_of_domain_sky_to_a_finite_response(costh):
    for value in polarization_tensor(costh, 0.7):
        assert not np.all(np.isfinite(np.asarray(value)))


@pytest.mark.parametrize("index,value", [(1, 1.01), (4, -1.01), (0, -7884.0)])
def test_traced_invalid_native_sources_return_nan(glass_params, index, value):
    bad = glass_params.copy()
    bad[index] = value
    sources = jnp.asarray(np.stack([glass_params, bad]))
    evaluate = jax.jit(jax.vmap(lambda p: bg.ucb_waveform(p, T=7_884_000.0, BW=32)))
    result = evaluate(sources)
    for output in result.values():
        assert np.all(np.isfinite(np.asarray(output[0])))
        assert np.all(np.isnan(np.asarray(output[1])))


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


@pytest.mark.parametrize("sky", [-1.0, 1.0])
@pytest.mark.parametrize("inclination", [-1.0, 1.0])
def test_cosine_endpoints_remain_valid(glass_params, sky, inclination):
    p = glass_params.copy()
    p[[1, 4]] = [sky, inclination]
    result = bg.ucb_waveform(p, T=7_884_000.0, BW=32)
    for output in result.values():
        assert np.all(np.isfinite(np.asarray(output)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("amp", 0.0),
        ("amp", -1e-22),
        ("f0", 0.0),
        ("f0", -1e-3),
        ("lam", np.inf),
        ("iota", -0.1),
        ("iota", np.pi + 0.1),
        ("beta", -np.pi / 2 - 0.1),
        ("beta", np.pi / 2 + 0.1),
    ],
)
def test_eager_physical_sources_reject_invalid_domains(params, field, value):
    with pytest.raises(ValueError):
        bg.tdi_waveform(params._replace(**{field: value}), T=7_884_000.0, BW=32)


def test_physical_wrapper_validates_observation_grid(params):
    with pytest.raises(ValueError, match="T must be"):
        bg.tdi_waveform(params, T=-1.0, BW=32)


@pytest.mark.parametrize("inclination", [0.0, np.pi])
@pytest.mark.parametrize("latitude", [-np.pi / 2, np.pi / 2])
def test_physical_angle_endpoints_remain_valid(params, inclination, latitude):
    p = params._replace(iota=inclination, beta=latitude)
    for output in bg.tdi_waveform(p, T=7_884_000.0, BW=32).values():
        assert np.all(np.isfinite(np.asarray(output)))


def test_invalid_traced_physical_angles_return_nan(params):
    bad = params._replace(beta=np.pi).to_array()
    evaluate = jax.jit(lambda p: bg.tdi_waveform(p, T=7_884_000.0, BW=32))
    for output in evaluate(bad).values():
        assert np.all(np.isnan(np.asarray(output)))
