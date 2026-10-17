"""Public entries choose the automatic response grid before compiling.

With ``response_samples=None`` the compiled kernel can select one of seven
response counts from the duration with ``jax.lax.switch``. Counts below
``BW`` interpolate, each in its own branch; every count of at least ``BW``
evaluates the response directly and shares one. When ``T`` is traced, every
branch is compiled and, under differentiation, JAX keeps the intermediate
values of all of them. Each public entry therefore chooses the count on the
host whenever ``T`` is concrete, exactly as the compiled selection would, and
warns once when it cannot, unless ``BW`` is at most the smallest count and
there is nothing to select.
"""

from __future__ import annotations

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import detector
from binary_galactica import single as bg32
from binary_galactica.glass import compat
from binary_galactica.glass import single as single_impl
from binary_galactica.glass import waveform as glass_waveform
from binary_galactica.glass.constants import YEAR
from tests.conftest import nested_jaxprs

DURATION = 7_884_000.0
#: Above the smallest automatic count, so that a traced duration keeps a
#: switch: 32 samples interpolate and larger counts evaluate directly.
BW = 64
NATIVE = bg.params_from_native(
    f0=1e-3,
    fdot=1e-18,
    amp=7.16e-22,
    phi0=0.5,
    colat=1.78,
    lon=0.0,
    psi=0.7,
    cos_iota=0.3,
    T=DURATION,
)
PHYSICAL = np.asarray(bg.from_glass_vector(NATIVE, DURATION).to_array())
Q_REF, NATIVE32 = bg32.split_native(NATIVE, DURATION)
Q_PHYSICAL, PHYSICAL32 = bg32.split_physical(PHYSICAL, DURATION)
WINDOW = {"bin_lo": Q_REF - BW // 2, "n_bins": BW}

#: Every public function taking ``response_samples``, as a function of ``T``
#: and, optionally, the bandwidth.
ENTRIES = {
    "ucb_waveform": lambda duration, bandwidth=BW, **kw: bg.ucb_waveform(
        NATIVE, T=duration, BW=bandwidth, **kw
    ),
    "template_on_window": lambda duration, bandwidth=BW, **kw: bg.template_on_window(
        NATIVE, T=duration, BW=bandwidth, **WINDOW, **kw
    ),
    "tdi_waveform": lambda duration, bandwidth=BW, **kw: bg.tdi_waveform(
        PHYSICAL, T=duration, BW=bandwidth, **kw
    ),
    "single.ucb_waveform": lambda duration, bandwidth=BW, **kw: bg32.ucb_waveform(
        NATIVE32, q_ref=Q_REF, T=duration, BW=bandwidth, **kw
    ),
    "single.template_on_window": lambda duration, bandwidth=BW, **kw: (
        bg32.template_on_window(
            NATIVE32, q_ref=Q_REF, T=duration, BW=bandwidth, **WINDOW, **kw
        )
    ),
    "single.tdi_waveform": lambda duration, bandwidth=BW, **kw: bg32.tdi_waveform(
        PHYSICAL32, q_ref=Q_PHYSICAL, T=duration, BW=bandwidth, **kw
    ),
}
#: Each entry's compiled function, which receives the resolved count, and the
#: working dtype of its duration.
COMPILED = {
    "ucb_waveform": (glass_waveform, "_ucb_waveform", np.float64),
    "template_on_window": (compat, "_template_on_window", np.float64),
    "tdi_waveform": (detector, "_tdi_waveform", np.float64),
    "single.ucb_waveform": (single_impl, "_ucb_waveform", np.float32),
    "single.template_on_window": (single_impl, "_template_on_window", np.float32),
    "single.tdi_waveform": (single_impl, "_tdi_waveform", np.float32),
}
#: The float64 entries that also take the legacy backend, with no switch.
FOURIER = ("ucb_waveform", "template_on_window", "tdi_waveform")


def primitives(jaxpr):
    """Return the names of every primitive in a jaxpr and its inner jaxprs."""
    return {eqn.primitive.name for inner in nested_jaxprs(jaxpr) for eqn in inner.eqns}


def grid_warnings(record):
    """Return the recorded warnings about the automatic response grid."""
    return [w for w in record if issubclass(w.category, bg.ResponseSamplesWarning)]


def neighbours(boundary, steps=2):
    """Return ``boundary`` and its ``steps`` nearest values on either side."""
    values = [boundary]
    below = above = boundary
    for _ in range(steps):
        below = np.nextafter(below, below.dtype.type(0))
        above = np.nextafter(above, above.dtype.type(np.inf))
        values = [below, *values, above]
    return values


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_host_count_matches_the_compiled_selection(dtype):
    """At every threshold and its nearest neighbours, in the graph's dtype."""
    real = np.dtype(dtype)
    counts = glass_waveform._AUTO_RESPONSE_SAMPLES
    compiled = jax.jit(lambda duration: glass_waveform._automatic_index(duration, real))
    boundaries = glass_waveform._duration_boundaries(real)
    assert boundaries.dtype == real
    durations = [d for b in boundaries for d in neighbours(b)]
    durations += [real.type(1.0), real.type(1e12), real.type(-1.0), real.type(np.nan)]
    seen = set()
    for duration in durations:
        index = int(compiled(jnp.asarray(duration, dtype=real)))
        host = glass_waveform._automatic_samples(duration, real)
        assert host == counts[index], (real, duration)
        seen.add(host)
    assert seen == set(counts)


def test_both_precisions_agree_at_identical_durations():
    """Float32 thresholds are rounded down, so equal durations give equal counts."""
    exact = glass_waveform._duration_boundaries(np.dtype(np.float64))
    for boundary in exact:
        for duration in neighbours(boundary):
            rounded = np.float32(duration)
            single = glass_waveform._automatic_samples(rounded, np.dtype(np.float32))
            double = glass_waveform._automatic_samples(rounded, np.dtype(np.float64))
            assert single == double, duration


def test_float64_thresholds_keep_their_bits():
    """Host thresholds equal those the kernel formed with ``jnp`` before 0.7."""
    boundaries = glass_waveform._duration_boundaries(np.dtype(np.float64))
    assert boundaries.tobytes() == (YEAR * 2.0 ** np.arange(-2, 4)).tobytes()
    counts = jnp.asarray((32, 64, 128, 256, 512, 1024), dtype=jnp.float64)
    traced = jax.jit(lambda: YEAR * counts / 128)()
    assert np.asarray(traced).tobytes() == boundaries.tobytes()


@pytest.mark.parametrize("name", list(ENTRIES))
def test_public_entries_compile_the_selected_count(name, monkeypatch):
    """Each entry passes the count the compiled switch selects for its dtype.

    Durations at and 0.01 s below each float64 threshold round above the
    float32 threshold, so there the float32 entries must choose the next count
    from the rounded duration while the float64 entries keep the lower one.
    """
    module, function, dtype = COMPILED[name]
    real = np.dtype(dtype)
    passed = []
    monkeypatch.setattr(
        module, function, lambda *a, **kw: passed.append(kw["response_samples"])
    )
    compiled = jax.jit(lambda duration: glass_waveform._automatic_index(duration, real))
    counts = glass_waveform._AUTO_RESPONSE_SAMPLES
    crossed = 0
    for boundary in glass_waveform._duration_boundaries(np.dtype(np.float64)):
        for duration in (boundary - 0.01, float(boundary), boundary + 0.01):
            ENTRIES[name](duration)
            expected = counts[int(compiled(jnp.asarray(duration, dtype=real)))]
            assert passed.pop() == expected, (name, duration)
            crossed += expected != glass_waveform._automatic_samples(
                duration, np.dtype(np.float64)
            )
    assert crossed == (12 if real == np.float32 else 0)


@pytest.mark.parametrize("name", list(ENTRIES))
def test_concrete_duration_compiles_one_grid(name):
    """A host, NumPy or concrete JAX duration leaves no switch to compile."""
    entry = ENTRIES[name]
    for duration in (DURATION, np.float64(DURATION), jnp.asarray(DURATION)):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            jaxpr = jax.make_jaxpr(partial(entry, duration))()
        assert not grid_warnings(record), name
        assert "cond" not in primitives(jaxpr), name


@pytest.mark.parametrize("name", list(ENTRIES))
def test_traced_duration_warns_once_and_keeps_the_switch(name):
    entry = ENTRIES[name]
    with pytest.warns(bg.ResponseSamplesWarning, match="response_samples") as record:
        jaxpr = jax.make_jaxpr(entry)(DURATION)
    found = grid_warnings(record)
    assert len(found) == 1, [str(w.message) for w in found]
    # stacklevel points at the caller's line, here the lambda in ENTRIES.
    assert found[0].filename == __file__
    assert "cond" in primitives(jaxpr)


#: The same public functions with every argument but ``T`` bound, so that a
#: JAX transform, not a function in this file, calls them.
BARE = {
    "ucb_waveform": partial(bg.ucb_waveform, NATIVE, BW=BW),
    "template_on_window": partial(bg.template_on_window, NATIVE, BW=BW, **WINDOW),
    "tdi_waveform": partial(bg.tdi_waveform, PHYSICAL, BW=BW),
    "single.ucb_waveform": partial(bg32.ucb_waveform, NATIVE32, q_ref=Q_REF, BW=BW),
    "single.template_on_window": partial(
        bg32.template_on_window, NATIVE32, q_ref=Q_REF, BW=BW, **WINDOW
    ),
    "single.tdi_waveform": partial(
        bg32.tdi_waveform, PHYSICAL32, q_ref=Q_PHYSICAL, BW=BW
    ),
}


@pytest.mark.parametrize("name", list(BARE))
def test_warning_names_the_line_that_applies_a_transform_to_an_entry(name):
    """JAX's frames lie between the transform and the entry; skip them too."""
    assert set(BARE) == set(ENTRIES)
    with pytest.warns(bg.ResponseSamplesWarning) as record:
        jax.make_jaxpr(BARE[name])(T=DURATION)
    found = grid_warnings(record)
    assert len(found) == 1
    assert found[0].filename == __file__


@pytest.mark.parametrize("bandwidth", [2, 32])
@pytest.mark.parametrize("name", list(ENTRIES))
def test_traced_duration_is_silent_when_every_count_is_direct(name, bandwidth):
    """With ``BW`` at most 32, every automatic count evaluates directly.

    The grid then does not depend on ``T``: no switch is compiled, and there
    is no cost to warn about.
    """
    entry = ENTRIES[name]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        jaxpr = jax.make_jaxpr(lambda duration: entry(duration, bandwidth=bandwidth))(
            DURATION
        )
    assert not grid_warnings(record), name
    assert "cond" not in primitives(jaxpr), name


@pytest.mark.parametrize("name", list(ENTRIES))
def test_explicit_count_with_traced_duration_is_silent(name):
    entry = ENTRIES[name]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        jaxpr = jax.make_jaxpr(lambda duration: entry(duration, response_samples=32))(
            DURATION
        )
    assert not grid_warnings(record)
    assert "cond" not in primitives(jaxpr)


@pytest.mark.parametrize("name", FOURIER)
def test_legacy_backend_never_warns(name):
    """The Fourier backend has no response grid to choose."""
    entry = ENTRIES[name]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        jax.make_jaxpr(lambda duration: entry(duration, backend="fourier"))(DURATION)
    assert not grid_warnings(record)


def test_closing_over_the_duration_matches_the_traced_switch():
    """Both sides of the quarter-year threshold, through the public call.

    Adjacent automatic counts differ here by about 1e-10 of the channel
    norm, far above float64 rounding, so agreement to 1e-13 shows that the
    host chose the grid the compiled switch selects.
    """
    threshold = glass_waveform._duration_boundaries(np.dtype(np.float64))[0]
    switched = jax.jit(lambda duration: bg.ucb_waveform(NATIVE, T=duration, BW=64))
    with pytest.warns(bg.ResponseSamplesWarning):
        switched(threshold)
    for duration in (threshold, np.nextafter(threshold, np.inf)):
        host = bg.ucb_waveform(NATIVE, T=duration, BW=64)
        traced = switched(duration)
        for channel in ("X", "A", "E"):
            norm = np.linalg.norm(host[channel])
            difference = np.linalg.norm(host[channel] - traced[channel])
            assert difference <= 1e-13 * norm, (duration, channel)
    below = bg.ucb_waveform(NATIVE, T=threshold, BW=64, response_samples=64)
    host = bg.ucb_waveform(NATIVE, T=threshold, BW=64)
    norm = np.linalg.norm(host["X"])
    assert np.linalg.norm(host["X"] - below["X"]) > 1e-12 * norm
