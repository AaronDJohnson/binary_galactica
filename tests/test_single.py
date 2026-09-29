"""The float32 namespace: dtypes, accuracy, split helpers and invalid inputs.

The suite enables x64, so these tests also check that nothing promotes to
float64. One subprocess repeats the smoke checks with x64 disabled. Errors
against float64 are computed in float64, since raw strains square below
float32's range.
"""

from __future__ import annotations

import json
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from binary_galactica.glass import _twofloat as tf
from binary_galactica.glass import single as single_impl
from binary_galactica.glass.constants import PI2, YEAR
from binary_galactica.glass.waveform import (
    _AUTO_RESPONSE_SAMPLES,
    _duration_boundaries,
)
from tests.conftest import double_precision_values
from tests.test_precision import run_without_x64

INT32_MIN = np.iinfo(np.int32).min
#: A quarter year observed ten years after the reference epoch, at 20 mHz:
#: every float32 reformulation matters here.
T32 = np.float32(0.25 * YEAR)
EPOCH32 = np.float32(10 * YEAR)
Q_REF = math.floor(0.02 * float(T32))
BW = 128
SAMPLES = 32
_END = float(EPOCH32) + float(T32)
#: Offset native coordinates; the derivatives drift 3 and -1 bins by the end.
PARAMS = np.array(
    [
        0.63,
        np.sin(0.1),
        4.1,
        np.log(2e-22),
        np.cos(1.1),
        0.4,
        -0.7,
        3 * float(T32) / _END,
        -2 * float(T32) ** 2 / _END**2,
    ],
    dtype=np.float32,
)
#: The same source in float64 at exactly these inputs.
NATIVE = PARAMS.astype(np.float64)
NATIVE[0] += Q_REF
TIMES = (float(EPOCH32) + float(T32) * np.linspace(0.0, 1.0, 33)).astype(np.float32)
CACHE = np.asarray(bg.orbit_cache(float(T32), float(EPOCH32), BW), dtype=np.float32)
THETA = np.array(
    [1e-22, 0.63 / float(T32), PARAMS[7] / float(T32) ** 2, 0.7, 1.1, 0.4, 4.1, 0.1],
    dtype=np.float32,
)
WINDOW: dict[str, Any] = {"bin_lo": Q_REF - 90, "n_bins": 180}
COMMON: dict[str, Any] = {"T": T32, "t0": EPOCH32, "response_samples": SAMPLES}


def errors(single, double):
    """Relative L2 error and flat-weighted mismatch, in float64."""
    a = np.asarray(single, dtype=np.complex128).ravel()
    b = np.asarray(double, dtype=np.complex128).ravel()
    overlap = np.vdot(a, b).real / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    return np.linalg.norm(a - b) / np.linalg.norm(b), 1 - overlap


def channels(result):
    """Stack A and E."""
    return np.stack([np.asarray(result["A"]), np.asarray(result["E"])])


def replaced(index, value, params=PARAMS):
    """Return float32 ``params`` with one entry replaced."""
    params = params.copy()
    params[index] = value
    return params


def waveform(params=PARAMS, q_ref=Q_REF, **options):
    """Evaluate the float32 waveform at the shared observation."""
    return bg32.ucb_waveform(params, q_ref=q_ref, BW=BW, **{**COMMON, **options})


#: Each float32 entry point as a function of (params, q_ref).
ENTRY_POINTS = {
    "ucb_waveform": lambda p, q: waveform(p, q),
    "ucb_waveform, phase format and cache": lambda p, q: waveform(
        p, q, fmt="phase", positions=CACHE
    ),
    "ucb_waveform, carrier override": lambda p, q: waveform(p, q, q=q + 1),
    "ucb_waveform, automatic grid": lambda p, q: waveform(p, q, response_samples=None),
    "template_on_window": lambda p, q: bg32.template_on_window(
        p, q_ref=q, BW=BW, **WINDOW, **COMMON
    ),
    "tdi_response": lambda p, q: bg32.tdi_response(p, TIMES, q_ref=q, T=T32),
    "tdi_waveform": lambda p, q: bg32.tdi_waveform(THETA, q_ref=q, BW=BW, **COMMON),
}


def dtypes(result):
    """Return the output dtypes by name."""
    if isinstance(result, dict):
        return {name: np.asarray(value).dtype for name, value in result.items()}
    return {"window or response": np.asarray(result).dtype}


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_entry_points_return_single_precision(name):
    result = ENTRY_POINTS[name](PARAMS, Q_REF)
    for key, dtype in dtypes(result).items():
        expected = np.int32 if key in ("qmin", "freq_bin_indices") else np.complex64
        assert dtype == expected, key
    values = result.values() if isinstance(result, dict) else [result]
    assert all(np.all(np.isfinite(np.asarray(value))) for value in values)


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_traced_entry_points_contain_no_double_precision(name):
    jaxpr = jax.make_jaxpr(ENTRY_POINTS[name])(PARAMS, np.int32(Q_REF))
    assert double_precision_values(jaxpr) == []


@pytest.mark.parametrize("fmt", bg.FORMATS)
@pytest.mark.parametrize("cached", [False, True], ids=["analytic", "cache"])
def test_waveform_matches_float64(fmt, cached):
    single = waveform(fmt=fmt, positions=CACHE if cached else None)
    double = bg.ucb_waveform(
        NATIVE,
        T=float(T32),
        BW=BW,
        t0=float(EPOCH32),
        fmt=fmt,
        positions=CACHE.astype(np.float64) if cached else None,
        q=float(Q_REF),
        response_samples=SAMPLES,
    )
    assert int(single["qmin"]) == int(double["qmin"])
    np.testing.assert_array_equal(
        single["freq_bin_indices"], np.asarray(double["freq_bin_indices"])
    )
    relative_l2, mismatch = errors(channels(single), channels(double))
    assert relative_l2 < 1e-4
    assert mismatch < 1e-8


@pytest.fixture(scope="module")
def reference():
    """Float64 waveform and response at the shared inputs."""
    result = bg.ucb_waveform(
        NATIVE,
        T=float(T32),
        BW=BW,
        t0=float(EPOCH32),
        q=float(Q_REF),
        response_samples=SAMPLES,
    )
    response = bg.tdi_response(NATIVE, TIMES.astype(np.float64), T=float(T32))
    return channels(result), response


@pytest.mark.parametrize("name", single_impl.REFORMULATIONS)
def test_each_reformulation_reduces_the_error(reference, name):
    """At this corner, disabling any reformulation at least doubles the error."""
    waveform64, response64 = reference

    def measure():
        response = bg32.tdi_response(PARAMS, TIMES, q_ref=Q_REF, T=T32)
        return (
            errors(channels(waveform()), waveform64)[0],
            errors(response, response64)[0],
        )

    waveform_error, response_error = measure()
    with single_impl._without(name):
        disabled_waveform, disabled_response = measure()
    assert disabled_waveform > 2 * waveform_error
    if name == "orbit":
        assert disabled_response > 2 * response_error


def test_window_matches_float64():
    single = bg32.template_on_window(PARAMS, q_ref=Q_REF, BW=BW, **WINDOW, **COMMON)
    double = bg.template_on_window(
        NATIVE,
        T=float(T32),
        BW=BW,
        t0=float(EPOCH32),
        response_samples=SAMPLES,
        **WINDOW,
    )
    assert single.shape == (2, WINDOW["n_bins"])
    relative_l2, mismatch = errors(single, double)
    assert relative_l2 < 1e-4
    assert mismatch < 1e-8


@pytest.mark.parametrize("given", [False, True], ids=["analytic", "positions"])
def test_arbitrary_time_response_matches_float64(given):
    positions = np.asarray(bg.analytic_orbits(TIMES.astype(np.float64)))
    positions = positions.astype(np.float32) if given else None
    single = bg32.tdi_response(PARAMS, TIMES, q_ref=Q_REF, T=T32, positions=positions)
    double = bg.tdi_response(
        NATIVE,
        TIMES.astype(np.float64),
        T=float(T32),
        positions=None if positions is None else positions.astype(np.float64),
    )
    relative_l2, mismatch = errors(single, double)
    assert relative_l2 < 1e-4
    assert mismatch < 1e-8


def test_physical_waveform_matches_float64():
    single = bg32.tdi_waveform(THETA, q_ref=Q_REF, BW=BW, **COMMON)
    theta = THETA.astype(np.float64)
    theta[1] += Q_REF / float(T32)
    double = bg.tdi_waveform(
        theta, T=float(T32), BW=BW, t0=float(EPOCH32), response_samples=SAMPLES
    )
    relative_l2, mismatch = errors(channels(single), channels(double))
    assert relative_l2 < 1e-4
    assert mismatch < 1e-8


def test_amplitude_rescales_through_the_log_amplitude():
    """Subtracting ``log(s)`` scales channels by ``1/s``, clear of underflow."""
    scale = 1e-20
    rescaled = channels(waveform(replaced(3, PARAMS[3] - np.float32(np.log(scale)))))
    assert np.max(np.abs(rescaled)) > 1e-3
    relative_l2, _ = errors(
        rescaled.astype(np.complex128) * scale, channels(waveform())
    )
    assert relative_l2 < 1e-5


def test_jit_and_vmap_over_reference_bins():
    batch = jax.vmap(lambda p, q: waveform(p, q))(
        np.stack([PARAMS, PARAMS]), np.array([Q_REF, Q_REF + 5], dtype=np.int32)
    )
    np.testing.assert_array_equal(batch["qmin"], [Q_REF - BW // 2, Q_REF + 5 - BW // 2])
    compiled = jax.jit(lambda p, q: waveform(p, q)["A"])(PARAMS, np.int32(Q_REF))
    relative_l2, _ = errors(compiled, waveform()["A"])
    assert relative_l2 < 1e-5


def test_automatic_grid_matches_float64_at_identical_durations():
    """Thresholds sit at 0.25-8 years, where float32 durations round across them."""
    counts = _AUTO_RESPONSE_SAMPLES

    def count(duration, dtype):
        boundaries = _duration_boundaries(np.dtype(dtype))
        return counts[int(jnp.sum(boundaries < jnp.asarray(duration, dtype)))]

    for years in (0.25, 0.5, 1, 2, 4, 8):
        duration = np.float32(years * YEAR)
        for _ in range(3):
            duration = np.nextafter(duration, np.float32(0))
        for _ in range(7):
            assert count(duration, np.float32) == count(float(duration), np.float64)
            duration = np.nextafter(duration, np.float32(np.inf))


def test_default_carrier_uses_the_floor_of_the_offset():
    result = waveform(replaced(0, PARAMS[0] + 3), Q_REF - 3)
    assert int(result["qmin"]) == Q_REF - BW // 2
    # Only the rounding of the shifted offset differs.
    relative_l2, _ = errors(channels(result), channels(waveform()))
    assert relative_l2 < 1e-4


@pytest.mark.parametrize(
    "native",
    [
        np.array([1234567.1, 0.2, 1.0, -50.0, 0.1, 0.3, 0.4]),
        np.array(
            [
                [5.5, 0.2, 1.0, -50.0, 0.1, 0.3, 0.4, 1.0, 0.1],
                [98765.3, -0.9, 5.0, -49.0, 0.7, 1.0, 2.0, -3.0, 0.2],
            ]
        ),
    ],
    ids=["one", "catalogue"],
)
def test_split_native_round_trips(native):
    q_ref, params = bg32.split_native(native, 1e7)
    assert params.dtype == np.float32
    assert params.shape == native.shape
    if native.ndim == 1:
        assert isinstance(q_ref, int)
    else:
        assert q_ref.dtype == np.int32 and q_ref.shape == native.shape[:-1]
    offset = params[..., 0]
    assert np.all((offset >= 0) & (offset < 1))
    error = np.abs(q_ref + offset.astype(np.float64) - native[..., 0])
    assert np.all(error <= np.spacing(offset))
    np.testing.assert_array_equal(params[..., 1:], native[..., 1:].astype(np.float32))


def test_split_native_keeps_offsets_below_one():
    native = np.array([7.0 - 2.0**-30, 0.2, 1.0, -50.0, 0.1, 0.3, 0.4])
    q_ref, params = bg32.split_native(native, 1e7)
    assert q_ref == 6
    assert params[0] == np.nextafter(np.float32(1), np.float32(0))


def test_split_physical_uses_the_float32_duration():
    duration = 4 * YEAR
    duration32 = float(np.float32(duration))
    assert duration32 != duration
    theta = np.array([1e-22, 0.0123456789, 1e-17, 0.7, 1.1, 0.4, 4.1, -0.3])
    q_ref, split = bg32.split_physical(bg.GBParams(*theta), duration)
    assert q_ref == math.floor(theta[1] * duration32)
    assert 0 <= split[1] * duration32 < 1
    frequency = q_ref / duration32 + float(split[1])
    ulp = np.spacing(split[1])
    assert abs(frequency - theta[1]) <= ulp
    np.testing.assert_array_equal(np.delete(split, 1), np.delete(theta, 1).astype("f4"))


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: bg32.split_native([0.0, 0, 0, 0, 0, 0, 0], 1.0), "positive"),
        (lambda: bg32.split_native([1.0, 2, 0, 0, 0, 0, 0], 1.0), "cosines"),
        (lambda: bg32.split_native([1.0] * 6, 1.0), "trailing length"),
        (lambda: bg32.split_native([1.0, 0, 0, 0, 0, 0, 0], -1.0), "T must"),
        (lambda: bg32.split_physical([1e-22, 1e-3, 0, 0, 4, 0, 0, 0], 1.0), "iota"),
        (lambda: bg32.split_physical([1e-22, 3e9, 0, 0, 1, 0, 0, 0], 1.0), "2\\*\\*31"),
    ],
)
def test_split_helpers_reject_invalid_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: waveform(q_ref=1.0 * Q_REF), "q_ref must be an integer"),
        (lambda: waveform(q_ref=True), "q_ref must be an integer"),
        (lambda: waveform(q_ref=2**31 - 10), "int32 range"),
        (lambda: waveform(q_ref=np.array([Q_REF])), "scalar"),
        (lambda: waveform(q=Q_REF + 0.5), "q must be an integer"),
        (lambda: waveform(q=Q_REF + 2**25), "2\\*\\*24"),
        (lambda: waveform(replaced(0, -0.5), 0), "positive"),
        (lambda: waveform(replaced(0, 2.0**25)), "2\\*\\*24"),
        (lambda: waveform(replaced(4, 1.5)), "cosines"),
        (lambda: waveform(T=np.float64(1e-50)), "T must be positive"),
        (lambda: waveform(t0=1e39), "t0 must be finite"),
        (lambda: waveform(fmt="bogus"), "format"),
        (
            lambda: bg32.tdi_response(PARAMS, np.array([]), q_ref=Q_REF, T=T32),
            "nonempty",
        ),
        (
            lambda: bg32.template_on_window(
                PARAMS, q_ref=Q_REF, BW=BW, bin_lo=2**31 - 5, n_bins=10, **COMMON
            ),
            "int32 range",
        ),
        (
            lambda: bg32.tdi_waveform(
                replaced(4, 4.0, THETA), q_ref=Q_REF, BW=BW, **COMMON
            ),
            "iota",
        ),
    ],
)
def test_eager_invalid_inputs_raise(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_traced_reference_bins_must_be_int32():
    with pytest.raises(ValueError, match="int32 scalar"):
        jax.jit(lambda q: waveform(q_ref=q)["A"])(np.int64(Q_REF))


INT32_MAX = int(np.iinfo(np.int32).max)


@pytest.mark.parametrize(
    "params, q_ref, source_invalid",
    [
        (replaced(1, 1.5), Q_REF, True),
        (replaced(0, -1.5), 1, True),
        (replaced(0, np.nan), Q_REF, True),
        (PARAMS, INT32_MAX - BW + 1, False),
        (replaced(0, 150.5), INT32_MAX - 200, False),
    ],
    ids=["sky cosine", "negative frequency", "nan", "reference bin", "carrier"],
)
def test_traced_invalid_inputs_give_nan_and_int32_min(params, q_ref, source_invalid):
    result = jax.jit(lambda p, q: waveform(p, q))(params, np.int32(q_ref))
    assert result["qmin"] == INT32_MIN
    assert np.all(np.asarray(result["freq_bin_indices"]) == INT32_MIN)
    assert all(np.all(np.isnan(result[name])) for name in "XYZAE")
    window = jax.jit(
        lambda p, q: bg32.template_on_window(p, q_ref=q, BW=BW, **WINDOW, **COMMON)
    )(params, np.int32(q_ref))
    assert np.all(np.isnan(window))
    response = jax.jit(lambda p, q: bg32.tdi_response(p, TIMES, q_ref=q, T=T32))(
        params, np.int32(q_ref)
    )
    # The arbitrary-time response has no bins to overflow.
    assert np.all(np.isnan(response)) == source_invalid


def compiled(function, *args):
    """Evaluate in one XLA computation, where this CPU fuses ``a*b + c``."""
    return jax.jit(function)(*args)


def dispatched(function, *args):
    """Evaluate op by op, rounding every product before it is added."""
    with jax.disable_jit():
        return function(*args)


def test_this_backend_contracts_multiply_add():
    """Documents why pairs avoid rounded products: compiled code fuses them."""
    rng = np.random.default_rng(2)
    a, b = rng.uniform(1, 2, (2, 1000)).astype(np.float32)
    c = -(a * b)
    fused = (a.astype(np.float64) * b + c).astype(np.float32)
    result = jax.jit(lambda a, b, c: a * b + c)(a, b, c)
    unfused = dispatched(lambda a, b, c: a * b + c, a, b, c)
    assert np.all(np.asarray(unfused) == 0)
    if jax.default_backend() == "cpu" and np.any(np.asarray(result) != 0):
        np.testing.assert_array_equal(result, fused)


@pytest.mark.parametrize("evaluate", [compiled, dispatched])
def test_orbital_phase_reduction_on_this_backend(evaluate):
    """Paired reductions keep ``PI2*t/YEAR`` to float32 rounding of the angle.

    Plain float32 loses about 1e-5 rad at twenty years. The pairs must hold
    whether or not the backend contracts multiply-adds.
    """

    def error(phase, times):
        exact = PI2 * np.asarray(times, dtype=np.float64) / YEAR
        difference = np.asarray(phase, dtype=np.float64) - exact
        return np.max(np.abs(np.remainder(difference + np.pi, PI2) - np.pi))

    times = np.linspace(0.0, 20 * YEAR, 4097).astype(np.float32)
    phase = evaluate(single_impl.orbital_phase, times)
    assert phase.dtype == np.float32
    assert error(phase, times) < 4e-7
    assert error(PI2 * jnp.asarray(times) / np.float32(YEAR), times) > 3e-6

    grid = evaluate(
        lambda duration, epoch: single_impl.grid_orbital_phase(duration, epoch, 96),
        T32,
        EPOCH32,
    )
    assert error(grid, float(EPOCH32) + float(T32) * np.arange(96) / 96) < 4e-7


@pytest.mark.parametrize("evaluate", [compiled, dispatched])
def test_pair_products_and_quotients_are_accurate(evaluate):
    rng = np.random.default_rng(5)
    x = rng.uniform(1.0, 1e9, 1000).astype(np.float32)
    y = rng.uniform(1e-9, 1.0, 1000).astype(np.float32)
    for function, exact in (
        (tf.product, x.astype(np.float64) * y),
        (tf.divide, x.astype(np.float64) / y),
    ):
        hi, lo = evaluate(function, x, y)
        total = np.asarray(hi, np.float64) + np.asarray(lo, np.float64)
        assert np.max(np.abs(total / exact - 1)) < 2**-44


def test_epoch_expansion_reproduces_the_float64_phase():
    """The re-expanded phase is the float64 formula at ``u = (t0 + T*v)/T``."""
    residual = tf.pair(np.float32(PARAMS[0]))
    coefficients = jax.jit(single_impl.epoch_coefficients)(
        residual, PARAMS, T32, EPOCH32
    )
    c0, c1, c2, c3 = (float(c) for c in coefficients)
    v = np.arange(BW) / BW
    single = c0 + v * (c1 + v * (c2 + v * c3))
    r, first, second = (float(PARAMS[i]) for i in (0, 7, 8))
    u = (float(EPOCH32) + float(T32) * v) / float(T32)
    double = r * u + 0.5 * first * u**2 + second * u**3 / 6
    difference = np.remainder(single - double + 0.5, 1.0) - 0.5
    assert np.max(np.abs(difference)) < 1e-6


def test_reformulation_switch_is_scoped():
    assert single_impl._DISABLED == frozenset()
    with single_impl._without("orbit", "epoch"):
        assert single_impl._DISABLED == {"orbit", "epoch"}
    assert single_impl._DISABLED == frozenset()
    with pytest.raises(ValueError, match="unknown"), single_impl._without("bogus"):
        pass


def test_entry_points_without_x64():
    """A fresh process with x64 disabled runs every float32 function."""
    code = f"""
import json

import jax
import numpy as np

assert not jax.config.read("jax_enable_x64")
from binary_galactica import single as bg32

T = np.float32({float(T32)!r})
q_ref, params = bg32.split_native(
    np.array({NATIVE.tolist()!r}), {float(T32)!r}
)
q_phys, theta = bg32.split_physical(
    np.array([1e-22, 0.02, 1e-16, 0.7, 1.1, 0.4, 4.1, 0.1]), T
)
common = dict(T=T, BW=64, t0=np.float32(1e8), response_samples=16)
results = {{
    "ucb_waveform": bg32.ucb_waveform(params, q_ref=q_ref, **common),
    "tdi_waveform": bg32.tdi_waveform(theta, q_ref=q_phys, **common),
    "template_on_window": bg32.template_on_window(
        params, q_ref=q_ref, bin_lo=q_ref - 40, n_bins=80, **common
    ),
    "tdi_response": bg32.tdi_response(
        params, np.linspace(0.0, 1e8, 9), q_ref=q_ref, T=T
    ),
}}
report = {{
    "split": [str(params.dtype), str(theta.dtype), type(q_ref).__name__],
    "x64": jax.config.read("jax_enable_x64"),
}}
for name, result in results.items():
    values = result if isinstance(result, dict) else {{"value": result}}
    report[name] = {{
        key: [str(np.asarray(v).dtype), bool(np.all(np.isfinite(v)))]
        for key, v in values.items()
    }}
print(json.dumps(report))
"""
    report = json.loads(run_without_x64(code))
    assert report.pop("x64") is False
    assert report.pop("split") == ["float32", "float32", "int"]
    for name, outputs in report.items():
        for key, (dtype, finite) in outputs.items():
            expected = "int32" if key in ("qmin", "freq_bin_indices") else "complex64"
            assert (dtype, finite) == (expected, True), (name, key)
