"""Invalid inputs: eager ``ValueError`` and traced NaN with ``INT32_MIN`` bins.

Concrete invalid arguments raise before compiling. The same values supplied
through tracers cannot raise, so floating outputs become NaN and int32 bin
metadata ``INT32_MIN``; valid traced inputs stay finite.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from tests.single.support import INT32_MAX, INT32_MIN, physical, source

CASE = source(1.0, 1e-2, 10.0, (1, -1))
THETA, _ = physical(CASE)
TIMES = np.linspace(0.0, 6e8, 9, dtype=np.float32)
WINDOW = {"bin_lo": CASE.carrier - 60, "n_bins": 120}


def replaced(values, index, value):
    """Return a copy of float32 ``values`` with one entry replaced."""
    values = np.array(values, dtype=np.float32)
    values[index] = value
    return values


def waveform(params=CASE.params, q_ref=CASE.q_ref, **options):
    """Call the float32 waveform with ``CASE``'s observation."""
    return bg32.ucb_waveform(params, q_ref=q_ref, **{**CASE.observation(), **options})


def window(params=CASE.params, q_ref=CASE.q_ref, **options):
    """Call the float32 window with ``CASE``'s observation."""
    return bg32.template_on_window(
        params, q_ref=q_ref, **{**WINDOW, **CASE.observation(), **options}
    )


def response(params=CASE.params, q_ref=CASE.q_ref, times=TIMES, **options):
    """Call the float32 arbitrary-time response."""
    return bg32.tdi_response(
        params, times, q_ref=q_ref, **{"T": CASE.duration, **options}
    )


def adapter(theta=THETA, q_ref=CASE.q_ref, **options):
    """Call the float32 physical adapter with ``CASE``'s observation."""
    return bg32.tdi_waveform(theta, q_ref=q_ref, **{**CASE.observation(), **options})


NATIVE64 = np.array([1e5 + 0.3, 0.2, 1.0, -50.0, 0.1, 0.3, 0.4])
THETA64 = np.array([1e-22, 1e-3, 1e-18, 0.7, 1.1, 0.4, 4.1, 0.1])

EAGER = {
    # Source coordinates, shared by every native entry point.
    "params length": (lambda: waveform(CASE.params[:6]), "length 7, 8, or 9"),
    "params rank": (lambda: window(CASE.params[None]), "one-dimensional"),
    "params nan": (lambda: response(replaced(CASE.params, 2, np.nan)), "finite"),
    "params beyond float32": (
        lambda: waveform(np.where(np.arange(9) == 3, 1e39, CASE.params.astype(float))),
        "finite",
    ),
    "sky cosine": (lambda: waveform(replaced(CASE.params, 1, 1.01)), "cosines"),
    "inclination cosine": (
        lambda: response(replaced(CASE.params, 4, -1.5)),
        "cosines",
    ),
    "offset beyond 2**24": (
        lambda: window(replaced(CASE.params, 0, 2.0**24)),
        "2\\*\\*24",
    ),
    "frequency zero": (lambda: waveform(replaced(CASE.params, 0, 0.0), 0), "positive"),
    "frequency negative": (
        lambda: response(replaced(CASE.params, 0, 3.5), -10),
        "positive",
    ),
    # Reference bins and carrier overrides.
    "q_ref float": (lambda: waveform(q_ref=float(CASE.q_ref)), "q_ref must be an int"),
    "q_ref bool": (lambda: response(q_ref=True), "q_ref must be an integer"),
    "q_ref array": (lambda: window(q_ref=np.array([CASE.q_ref])), "scalar"),
    "q_ref above int32": (lambda: response(q_ref=2**31), "int32 range"),
    "q_ref band above int32": (
        lambda: waveform(q_ref=INT32_MAX - CASE.bandwidth + 1),
        "int32 range",
    ),
    "q_ref band below int32": (
        lambda: adapter(q_ref=INT32_MIN + CASE.bandwidth - 1),
        "int32 range",
    ),
    "q fractional": (lambda: waveform(q=CASE.carrier + 0.5), "q must be an integer"),
    "q far from the source": (
        lambda: waveform(q=CASE.carrier + 2**24),
        "within 2\\*\\*24",
    ),
    "carrier band beyond int32": (
        lambda: window(replaced(CASE.params, 0, 150.5), INT32_MAX - CASE.bandwidth),
        "int32 range",
    ),
    "response carrier beyond int32": (
        lambda: response(replaced(CASE.params, 0, 300.5), INT32_MAX - 100),
        "int32 range",
    ),
    # Observation.
    "T zero": (lambda: waveform(T=0.0), "T must be positive"),
    "T underflows": (lambda: response(T=1e-50), "T must be positive"),
    "T nan": (lambda: window(T=np.nan), "T must be finite"),
    "t0 overflows": (lambda: adapter(t0=1e39), "t0 must be finite"),
    "BW odd": (lambda: waveform(BW=33), "BW must be a positive even"),
    "BW bool": (lambda: window(BW=True), "BW must be a positive even"),
    "response_samples": (lambda: adapter(response_samples=3), "at least 4"),
    "format": (lambda: window(fmt="bogus"), "format must be one of"),
    "cache shape": (
        lambda: waveform(positions=np.zeros((3, CASE.bandwidth + 2, 3))),
        "positions must have shape",
    ),
    "cache nan": (
        lambda: adapter(positions=np.full((3, CASE.bandwidth, 3), np.nan)),
        "positions must be finite",
    ),
    # Windows.
    "window n_bins": (lambda: window(n_bins=-1), "n_bins"),
    "window bin_lo": (lambda: window(bin_lo=1.5), "bin_lo"),
    "window beyond int32": (lambda: window(bin_lo=INT32_MAX - 10), "int32 range"),
    # Arbitrary-time response.
    "times empty": (lambda: response(times=np.array([])), "nonempty"),
    "times rank": (lambda: response(times=TIMES[None]), "one-dimensional"),
    "times nan": (lambda: response(times=replaced(TIMES, 3, np.nan)), "finite"),
    "response positions shape": (
        lambda: response(positions=np.zeros((3, TIMES.size + 1, 3))),
        "positions must have shape",
    ),
    # Physical parameters.
    "theta shape": (lambda: adapter(THETA[:7]), "shape \\(8,\\)"),
    "amp": (lambda: adapter(replaced(THETA, 0, 0.0)), "amp must be positive"),
    "iota": (lambda: adapter(replaced(THETA, 4, 3.2)), "iota"),
    "beta": (lambda: adapter(replaced(THETA, 7, -1.6)), "beta"),
    "physical offset beyond 2**24": (
        lambda: adapter(replaced(THETA, 1, 2.0**25 / float(CASE.duration))),
        "2\\*\\*24",
    ),
    "physical frequency negative": (
        lambda: adapter(replaced(THETA, 1, -3.5 / float(CASE.duration)), 2),
        "positive",
    ),
    # Host split helpers.
    "split_native length": (
        lambda: bg32.split_native(NATIVE64[:6], 1e7),
        "trailing length",
    ),
    "split_native frequency": (
        lambda: bg32.split_native(replaced(NATIVE64, 0, -1.0), 1e7),
        "positive",
    ),
    "split_native cosine": (
        lambda: bg32.split_native(replaced(NATIVE64, 4, 1.5), 1e7),
        "cosines",
    ),
    "split_native nan": (
        lambda: bg32.split_native(np.where(np.arange(7) == 5, np.nan, NATIVE64), 1e7),
        "finite",
    ),
    "split_native overflow": (
        lambda: bg32.split_native(np.where(np.arange(7) == 3, 1e39, NATIVE64), 1e7),
        "float32",
    ),
    "split_native int32": (
        lambda: bg32.split_native(np.where(np.arange(7) == 0, 2.0**31, NATIVE64), 1e7),
        "2\\*\\*31",
    ),
    "split_native duration": (lambda: bg32.split_native(NATIVE64, 0.0), "T must"),
    "split_physical length": (
        lambda: bg32.split_physical(THETA64[:7], 1e7),
        "trailing length 8",
    ),
    "split_physical amp": (
        lambda: bg32.split_physical(np.where(np.arange(8) == 0, 0.0, THETA64), 1e7),
        "positive",
    ),
    "split_physical beta": (
        lambda: bg32.split_physical(np.where(np.arange(8) == 7, 2.0, THETA64), 1e7),
        "beta",
    ),
    "split_physical duration": (
        lambda: bg32.split_physical(THETA64, np.inf),
        "T must",
    ),
    "split_physical float32 duration": (
        lambda: bg32.split_physical(THETA64, 1e39),
        "T must",
    ),
}


@pytest.mark.parametrize("name", list(EAGER))
def test_eager_invalid_inputs_raise(name):
    call, message = EAGER[name]
    with pytest.raises(ValueError, match=message):
        call()


def test_traced_reference_bins_must_be_int32():
    """With x64 on, a Python int traced by JIT is int64 and could wrap."""
    with pytest.raises(ValueError, match="int32 scalar"):
        jax.jit(lambda q: waveform(q_ref=q)["A"])(CASE.q_ref)
    with pytest.raises(ValueError, match="int32 scalar"):
        jax.jit(lambda q: waveform(q=q)["A"])(np.int64(CASE.carrier))


def assert_invalid(result):
    """Require NaN in every floating output and ``INT32_MIN`` in every int32 one."""
    values = result if isinstance(result, dict) else {"value": result}
    for key, value in values.items():
        value = np.asarray(value)
        if value.dtype == np.int32:
            assert np.all(value == INT32_MIN), key
        else:
            assert value.dtype == np.complex64, key
            assert np.all(np.isnan(value)), key


def assert_valid(result):
    """Require finite floating outputs and no ``INT32_MIN`` bin."""
    values = result if isinstance(result, dict) else {"value": result}
    for key, value in values.items():
        value = np.asarray(value)
        if value.dtype == np.int32:
            assert np.all(value != INT32_MIN), key
        else:
            assert np.all(np.isfinite(value)), key


CACHE = np.asarray(
    bg.orbit_cache(float(CASE.duration), float(CASE.epoch), CASE.bandwidth)
).astype(np.float32)
STATIC = {"BW": CASE.bandwidth, "response_samples": CASE.samples}


def arguments(**changes):
    """Return traced arguments for ``CASE``, with ``changes`` applied."""
    values = {
        "params": CASE.params,
        "q_ref": CASE.q_ref,
        "duration": CASE.duration,
        "epoch": CASE.epoch,
        "positions": CACHE,
    }
    values.update(changes)
    return {**values, "q_ref": np.int32(values["q_ref"])}


@jax.jit
def traced_waveform(params, q_ref, duration, epoch, positions):
    """Evaluate the waveform with a cache, every argument traced."""
    return bg32.ucb_waveform(
        params, q_ref=q_ref, T=duration, t0=epoch, positions=positions, **STATIC
    )


@jax.jit
def traced_window(params, q_ref, duration, epoch, positions):
    """Evaluate the window with a cache, every argument traced."""
    return bg32.template_on_window(
        params,
        q_ref=q_ref,
        T=duration,
        t0=epoch,
        positions=positions,
        **WINDOW,
        **STATIC,
    )


@jax.jit
def traced_response(params, q_ref, duration, epoch, positions):
    """Evaluate the response at ``epoch + TIMES`` on the analytic orbit."""
    del positions
    return bg32.tdi_response(params, TIMES + epoch, q_ref=q_ref, T=duration)


#: Invalid traced waveform inputs; each must invalidate every output.
TRACED_INVALID = {
    "params nan": {"params": replaced(CASE.params, 5, np.nan)},
    "sky cosine": {"params": replaced(CASE.params, 1, 1.5)},
    "inclination cosine": {"params": replaced(CASE.params, 4, -1.01)},
    "negative frequency": {"params": replaced(CASE.params, 0, -2.5), "q_ref": 1},
    "zero frequency": {"params": replaced(CASE.params, 0, 0.0), "q_ref": 0},
    "offset beyond 2**24": {"params": replaced(CASE.params, 0, 2.0**24)},
    "T negative": {"duration": np.float32(-1e7)},
    "T nan": {"duration": np.float32(np.nan)},
    "t0 infinite": {"epoch": np.float32(np.inf)},
    "cache nan": {
        "positions": np.full((3, CASE.bandwidth, 3), np.nan, dtype=np.float32)
    },
}
#: Invalid because bins leave int32, as ``(changes, response_valid)``. The
#: arbitrary-time response has no band, but needs a positive int32 carrier.
TRACED_BAND_INVALID = {
    "reference bin near the top": ({"q_ref": INT32_MAX - CASE.bandwidth + 1}, True),
    "carrier near the top": (
        {"params": replaced(CASE.params, 0, 150.5), "q_ref": INT32_MAX - 200},
        True,
    ),
    "carrier beyond int32": (
        {"params": replaced(CASE.params, 0, 300.5), "q_ref": INT32_MAX - 100},
        False,
    ),
    "carrier wraps": (
        {
            "params": replaced(CASE.params, 0, 2.0**24 - 1),
            "q_ref": INT32_MAX - CASE.bandwidth,
        },
        False,
    ),
    "reference bin near the bottom": (
        {"q_ref": INT32_MIN + CASE.bandwidth - 1},
        False,
    ),
}


@pytest.mark.parametrize("name", list(TRACED_INVALID))
def test_traced_invalid_inputs_invalidate_every_output(name):
    changes = TRACED_INVALID[name]
    assert_invalid(traced_waveform(**arguments(**changes)))
    assert_invalid(traced_window(**arguments(**changes)))
    if "positions" not in changes:
        assert_invalid(traced_response(**arguments(**changes)))


@pytest.mark.parametrize("name", list(TRACED_BAND_INVALID))
def test_traced_bins_outside_int32_invalidate_the_band(name):
    changes, response_valid = TRACED_BAND_INVALID[name]
    assert_invalid(traced_waveform(**arguments(**changes)))
    assert_invalid(traced_window(**arguments(**changes)))
    (assert_valid if response_valid else assert_invalid)(
        traced_response(**arguments(**changes))
    )


def test_traced_valid_inputs_stay_finite():
    """The controls: the same compiled functions on valid inputs."""
    assert_valid(traced_waveform(**arguments()))
    assert_valid(traced_window(**arguments()))
    assert_valid(traced_response(**arguments()))
    edge = arguments(q_ref=INT32_MAX - CASE.bandwidth)
    assert_valid(traced_waveform(**edge))


@pytest.mark.parametrize(
    "q, valid",
    [(CASE.carrier + 3, True), (CASE.carrier + 2**24, False), (INT32_MAX, False)],
    ids=["near", "far", "band beyond int32"],
)
def test_traced_carrier_overrides(q, valid):
    result = jax.jit(lambda q: waveform(q=q))(np.int32(q))
    (assert_valid if valid else assert_invalid)(result)


@pytest.mark.parametrize(
    "index, value",
    [(0, 0.0), (1, np.nan), (4, 3.3), (7, 1.6)],
    ids=["amp", "nan", "iota", "beta"],
)
def test_traced_invalid_physical_parameters(index, value):
    compiled = jax.jit(lambda theta, q_ref: adapter(theta, q_ref))
    assert_invalid(compiled(replaced(THETA, index, value), np.int32(CASE.q_ref)))
    assert_valid(compiled(THETA, np.int32(CASE.q_ref)))


def test_vmap_masks_only_the_invalid_member():
    params = np.stack([CASE.params, replaced(CASE.params, 1, 2.0), CASE.params])
    q_refs = np.array([CASE.q_ref, CASE.q_ref, INT32_MAX], dtype=np.int32)
    batch = jax.vmap(lambda p, q: waveform(p, q))(params, q_refs)
    members = [{k: v[i] for k, v in batch.items()} for i in range(3)]
    assert_valid(members[0])
    assert_invalid(members[1])
    assert_invalid(members[2])
    assert int(members[0]["qmin"]) == CASE.carrier - CASE.bandwidth // 2
