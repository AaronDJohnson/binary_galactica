"""Low-level kernels compute in the precision of their inputs.

The float64 entry points require x64, but the kernels beneath them take their
working dtype from the parameter or time array. Float32 inputs must stay in
float32 and complex64 even though this suite enables x64, where NumPy float64
constants and default-dtype constructors would silently promote them. The
float64 side is pinned bitwise by ``test_float64_golden.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass import _precision
from binary_galactica.glass import waveform as glass_waveform
from binary_galactica.glass.constants import PI2, YEAR
from binary_galactica.glass.onfly import _sinc, reference_phase_shift, tdi_response
from binary_galactica.glass.orbit import (
    analytic_orbits,
    orbital_phase,
    positions_from_phase,
    slow_time_grid,
)
from binary_galactica.glass.response import polarization_tensor
from binary_galactica.glass.sparse import interpolate_uniform
from binary_galactica.glass.tdi import xyz_to_ae
from tests.conftest import double_precision_values, nested_jaxprs

DURATION = np.float32(1.2e8)
EPOCH = np.float32(2.0e7)
BW = 32
#: A 10 mHz source with both frequency derivatives, native coordinates.
SOURCE = np.array(
    [1.2e6, -0.3, 5.1, np.log(1e-22), -0.8, 1.2, 2.1, -130.0, 9.0], dtype=np.float32
)
#: Response times from before the epoch to beyond the observation.
TIMES = (EPOCH + DURATION * np.linspace(-0.1, 1.1, 13)).astype(np.float32)
#: Host positions rounded from float64 orbits, as a float32 caller supplies them.
POSITIONS = np.asarray(analytic_orbits(TIMES.astype(np.float64)), dtype=np.float32)
CACHE = np.asarray(bg.orbit_cache(float(DURATION), float(EPOCH), BW), dtype=np.float32)
THETA = np.array([1e-22, 1e-2, 1e-15, 0.7, 1.1, 0.4, 4.1, -0.3], dtype=np.float32)
_RNG = np.random.default_rng(3)
SAMPLES = (_RNG.normal(size=(17, 3)) + 1j * _RNG.normal(size=(17, 3))).astype(
    np.complex64
)
COORDINATES = np.concatenate(([0.0, 1.0], _RNG.uniform(size=40))).astype(np.float32)
XYZ = (_RNG.normal(size=(3, 8)) + 1j * _RNG.normal(size=(3, 8))).astype(np.complex64)
#: Host float64 inputs, which kernels must cast instead of promoting to float64.
HOST_DURATION, HOST_EPOCH = np.float64(DURATION), np.float64(EPOCH)
HOST_POSITIONS, HOST_CACHE = POSITIONS.astype(np.float64), CACHE.astype(np.float64)

#: Each kernel as a function of float32 arrays, and those arrays.
KERNELS = {
    "tdi_response": (tdi_response, (SOURCE, TIMES, DURATION)),
    "tdi_response, reference phase": (
        lambda p, t, duration: tdi_response(p, t, duration, reference_phase=True),
        (SOURCE, TIMES, DURATION),
    ),
    "tdi_response, seven parameters": (tdi_response, (SOURCE[:7], TIMES, DURATION)),
    "tdi_response, eight parameters and positions": (
        tdi_response,
        (SOURCE[:8], TIMES, DURATION, POSITIONS),
    ),
    "tdi_response, reference phase and positions": (
        lambda p, t, duration, x: tdi_response(p, t, duration, x, reference_phase=True),
        (SOURCE, TIMES, DURATION, POSITIONS),
    ),
    "tdi_response, host float64 duration and positions": (
        lambda p, t: tdi_response(p, t, HOST_DURATION, HOST_POSITIONS),
        (SOURCE, TIMES),
    ),
    "reference_phase_shift": (reference_phase_shift, (SOURCE, TIMES, DURATION)),
    "reference_phase_shift, seven parameters and positions": (
        reference_phase_shift,
        (SOURCE[:7], TIMES, DURATION, POSITIONS),
    ),
    "reference_phase_shift, host float64 duration and positions": (
        lambda p, t: reference_phase_shift(p, t, HOST_DURATION, HOST_POSITIONS),
        (SOURCE, TIMES),
    ),
    "_sinc": (_sinc, (np.linspace(-3.0, 3.0, 41, dtype=np.float32),)),
    "interpolate_uniform": (interpolate_uniform, (SAMPLES, COORDINATES)),
    "interpolate_uniform, real samples": (
        interpolate_uniform,
        (SAMPLES.real.copy(), COORDINATES),
    ),
    "orbital_phase": (orbital_phase, (TIMES,)),
    "positions_from_phase": (
        positions_from_phase,
        (np.linspace(0.0, 7.0, 5, dtype=np.float32),),
    ),
    "analytic_orbits": (analytic_orbits, (TIMES,)),
    "analytic_orbits, kappa_0": (
        lambda t: analytic_orbits(t, kappa_0=0.3),
        (TIMES,),
    ),
    "analytic_orbits, host float64 kappa_0": (
        lambda t: analytic_orbits(t, kappa_0=np.float64(0.3)),
        (TIMES,),
    ),
    "slow_time_grid": (
        lambda duration, epoch: slow_time_grid(duration, epoch, BW),
        (DURATION, EPOCH),
    ),
    "polarization_tensor": (polarization_tensor, (SOURCE[1], SOURCE[2])),
    "xyz_to_ae": (xyz_to_ae, tuple(XYZ)),
    "to_glass_vector": (bg.to_glass_vector, (THETA, DURATION)),
    "to_glass_vector, host float64 duration": (
        lambda theta: bg.to_glass_vector(theta, HOST_DURATION),
        (THETA,),
    ),
    "to_glass_vector, Python fields beside a batched amplitude": (
        lambda amp, duration: bg.to_glass_vector(
            bg.GBParams(amp, 1e-2, 1e-15, 0.7, 1.1, 0.4, 4.1, -0.3), duration
        ),
        (np.array([1e-22, 2e-22, 5e-23], dtype=np.float32), DURATION),
    ),
    "_cached_positions": (
        glass_waveform._cached_positions,
        (CACHE, TIMES[1:-1], DURATION, EPOCH),
    ),
    "_cached_positions, host float64 cache and clock": (
        lambda t: glass_waveform._cached_positions(
            HOST_CACHE, t, HOST_DURATION, HOST_EPOCH
        ),
        (TIMES[1:-1],),
    ),
    "_response_envelope": (
        lambda p, duration, epoch: glass_waveform._response_envelope(
            p, duration, BW, epoch, None, 16
        ),
        (SOURCE, DURATION, EPOCH),
    ),
    "_response_envelope, cached positions": (
        lambda p, duration, epoch, cache: glass_waveform._response_envelope(
            p, duration, BW, epoch, cache, 16
        ),
        (SOURCE, DURATION, EPOCH, CACHE),
    ),
    "_response_envelope, automatic samples": (
        lambda p, duration, epoch: glass_waveform._response_envelope(
            p, duration, BW, epoch, None, None
        ),
        (SOURCE, DURATION, EPOCH),
    ),
    "_response_envelope, host float64 clock and cache": (
        lambda p: glass_waveform._response_envelope(
            p, HOST_DURATION, BW, HOST_EPOCH, HOST_CACHE, None
        ),
        (SOURCE,),
    ),
}

#: Kernels that contract with einsum or the interpolation-window convolution.
CONTRACTING = (
    "tdi_response",
    "tdi_response, reference phase and positions",
    "reference_phase_shift",
    "interpolate_uniform",
    "_response_envelope, automatic samples",
)

HIGHEST = jax.lax.Precision.HIGHEST


def _upcast(value):
    """Return the float64 or complex128 copy of a float32 input."""
    value = np.asarray(value)
    return value.astype(np.complex128 if np.iscomplexobj(value) else np.float64)


def _contractions(jaxpr):
    """Return every dot_general and convolution, including in inner jaxprs."""
    return [
        equation
        for inner in nested_jaxprs(jaxpr)
        for equation in inner.eqns
        if equation.primitive.name in ("dot_general", "conv_general_dilated")
    ]


@pytest.mark.parametrize("name", list(KERNELS))
def test_float32_inputs_give_float32_outputs(name):
    function, args = KERNELS[name]
    leaves = jax.tree_util.tree_leaves(function(*args))
    assert leaves
    for leaf in leaves:
        assert leaf.dtype in (jnp.float32, jnp.complex64), f"{name}: {leaf.dtype}"


@pytest.mark.parametrize("name", list(KERNELS))
def test_float32_jaxprs_contain_no_double_precision(name):
    function, args = KERNELS[name]
    assert double_precision_values(jax.make_jaxpr(function)(*args)) == []


@pytest.mark.parametrize("name", list(KERNELS))
def test_float32_follows_float64_at_identical_inputs(name):
    """Both precisions evaluate one function; this is not the accuracy target."""
    function, args = KERNELS[name]
    single = jax.tree_util.tree_leaves(function(*args))
    double = jax.tree_util.tree_leaves(function(*map(_upcast, args)))
    for low, high in zip(single, double, strict=True):
        assert high.dtype in (jnp.float64, jnp.complex128)
        # Measure in float64: squared strains near 1e-44 underflow float32.
        high = np.asarray(high)
        error = np.max(np.abs(np.asarray(low, dtype=high.dtype) - high))
        assert error <= 1e-4 * np.max(np.abs(high)), name


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", CONTRACTING)
def test_kernel_contractions_use_highest_precision(name, dtype):
    function, args = KERNELS[name]
    if dtype == "float64":
        args = tuple(map(_upcast, args))
    contractions = _contractions(jax.make_jaxpr(function)(*args))
    assert contractions
    for equation in contractions:
        assert equation.params["precision"] == (HIGHEST, HIGHEST), equation


@pytest.mark.parametrize("backend", ["onfly", "fourier"])
def test_float64_waveform_contractions_use_highest_precision(backend):
    """CPU float64 ignores the setting; the golden outputs confirm the bits."""

    def waveform(params):
        return bg.ucb_waveform(params, T=1.2e8, BW=BW, backend=backend)

    contractions = _contractions(jax.make_jaxpr(waveform)(_upcast(SOURCE)))
    assert contractions
    for equation in contractions:
        assert equation.params["precision"] == (HIGHEST, HIGHEST), equation


def test_float64_orbit_cache_keeps_float64_for_a_float32_clock():
    """The guarded float64 entry point does not follow float32 inputs down."""
    cache = bg.orbit_cache(DURATION, EPOCH, BW)
    assert cache.dtype == jnp.float64
    np.testing.assert_array_equal(
        cache, bg.orbit_cache(float(DURATION), float(EPOCH), BW)
    )


@pytest.mark.parametrize("kappa_0", [0.0, 0.3])
def test_float64_orbital_phase_is_the_glass_expression(kappa_0):
    times = _upcast(TIMES)
    alpha = PI2 * jnp.asarray(times) / YEAR + kappa_0
    phase = orbital_phase(times, kappa_0=kappa_0)
    assert phase.dtype == jnp.float64
    np.testing.assert_array_equal(phase, alpha)
    np.testing.assert_array_equal(
        analytic_orbits(times, kappa_0=kappa_0), positions_from_phase(alpha)
    )


@pytest.mark.parametrize(
    ("values", "real"),
    [
        ((np.float32(1.0),), np.float32),
        ((np.ones(2, dtype=np.float32), 1.0, 2), np.float32),
        ((np.complex64(1j),), np.float32),
        ((np.float16(1.0),), np.float32),
        ((np.float32(1.0), np.float64(1.0)), np.float64),
        ((1.0,), np.float64),
        ((np.arange(3),), np.float64),
        (([[1.0, 2.0]],), np.float64),
        (([1.0, np.float32(2.0)],), np.float32),
        ((jnp.ones(2, dtype=jnp.complex128),), np.float64),
    ],
)
def test_working_dtypes(values, real):
    complex_ = np.complex64 if real == np.float32 else np.complex128
    assert _precision.real_dtype(*values) == real
    assert _precision.complex_dtype(*values) == complex_


def test_cast_keeps_python_scalars_weakly_typed():
    assert type(_precision.cast(3.0, jnp.float32)) is float
    assert type(_precision.cast(3, jnp.float32)) is int
    assert _precision.cast(np.float64(3.0), jnp.float32).dtype == jnp.float32
    assert _precision.cast(jnp.ones(2), jnp.float32).dtype == jnp.float32


def test_double_precision_check_reports_promotion():
    """The jaxpr check sees promotion in nested jaxprs and allows Python scalars."""
    x = np.ones(3, dtype=np.float32)
    weak = jax.make_jaxpr(lambda x: jnp.where(x > 0, x, 0.0))(x)
    assert double_precision_values(weak) == []
    constant = jax.make_jaxpr(lambda x: x * np.float64(2.0))(x)
    assert double_precision_values(constant)

    def branch(y):
        return (y * np.ones(3)).astype(np.float32)

    nested = jax.make_jaxpr(lambda x: jax.lax.cond(x[0] > 0, branch, jnp.sin, x))(x)
    assert double_precision_values(nested)
    function, args = KERNELS["tdi_response"]
    assert double_precision_values(jax.make_jaxpr(function)(*map(_upcast, args)))
