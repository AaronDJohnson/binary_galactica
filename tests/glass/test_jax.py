"""The JAX contract of the default waveform: jit, vmap, and AD at the poles.

These tests exercise ``ucb_waveform`` with its default ``backend="onfly"``:
the delayed TDI response sampled on a sparse grid, interpolated, re-phased,
and Fourier transformed. The generator must compose with JAX transforms, so
these are behavioural requirements, not nice-to-haves. In particular a ``NaN``
in a derivative is fatal to a sampler that never sees the forward value.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica.glass as gbonfly


def _p(injection, **over):
    d = {k: v for k, v in injection.items() if k != "T"}
    d.update(over)
    return gbonfly.params_from_native(T=injection["T"], **d)


def test_vmap_over_sources(injection):
    """Batching must not mix sources; the format is a static branch after it."""
    T, BW, fmt = injection["T"], 64, "sangria"
    orb = gbonfly.orbit_cache(T, 0.0, BW)
    batch = jnp.stack([_p(injection, f0=1e-3 + 1e-7 * i) for i in range(8)])
    fn = jax.vmap(lambda p: gbonfly.ucb_waveform(p, T=T, BW=BW, fmt=fmt, positions=orb))
    out = fn(batch)
    assert out["A"].shape == (8, BW)
    # Each source individually must reproduce its row. atol=0: these
    # amplitudes are ~1e-22, so the default absolute tolerance would pass
    # regardless of what the batched call returned.
    #
    # The batched and scalar calls are separate XLA programs, so they are
    # not guaranteed to be bit-identical: XLA may reassociate the batched
    # response, interpolation and FFT reductions. On the on-fly default this
    # case agrees to ~1e-15 relative; the bound only has to separate rounding
    # from a wrong row, and a vmap'd sampler need not reproduce an unbatched
    # one bit for bit.
    one = gbonfly.ucb_waveform(batch[3], T=T, BW=BW, fmt=fmt, positions=orb)
    np.testing.assert_array_equal(out["freq_bin_indices"][3], one["freq_bin_indices"])
    assert out["qmin"][3] == one["qmin"]
    for channel in ("X", "Y", "Z", "A", "E"):
        a, b = np.asarray(out[channel][3]), np.asarray(one[channel])
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-10


def test_jit_cache_reuse_across_parameters(injection):
    """Changing source values must reuse the compiled waveform kernel."""
    from binary_galactica.glass.waveform import _ucb_waveform

    T, BW = injection["T"], 64
    orb = gbonfly.orbit_cache(T, 0.0, BW)
    a = gbonfly.ucb_waveform(_p(injection), T=T, BW=BW, positions=orb)
    n = _ucb_waveform._cache_size()
    for i in range(5):
        gbonfly.ucb_waveform(_p(injection, phi0=0.1 * i), T=T, BW=BW, positions=orb)
    assert _ucb_waveform._cache_size() == n
    assert np.all(np.isfinite(np.asarray(a["A"])))


def test_derivatives_finite_at_the_ecliptic_poles():
    """``sqrt(1-costh^2)`` must not poison AD at ``costh = +-1``.

    Its derivative there is ``-costh/0``, and in forward mode the resulting
    ``0/0`` contaminates *every* tangent direction -- ``jacfwd`` came back
    all-NaN, including the columns for ``phi0`` and ``ln(amp)`` whose
    derivatives are perfectly well defined at the pole. The forward value is
    finite throughout, so only a derivative test catches it. Both poles share
    one compiled Jacobian.

    Forward mode only needs the guard's outer ``where``. Reverse mode also
    needs the inner one (otherwise ``0 * inf``); that is checked on the
    polarization tensor itself, far cheaper than a waveform gradient.
    """
    from binary_galactica.glass.response import polarization_tensor

    # The pole is reachable from any colatitude within ~1e-8 rad of it.
    assert np.cos(1e-8) == 1.0
    T, BW = 7884000.0, 64
    orb = gbonfly.orbit_cache(T, 0.0, BW)

    def channel(q):
        return gbonfly.ucb_waveform(q, T=T, BW=BW, positions=orb)["A"]

    for costh in (1.0, -1.0):
        p = jnp.array([7884.0, costh, 0.0, jnp.log(1e-22), 0.3, 0.7, 0.5, 0.0])
        assert np.all(np.isfinite(np.asarray(channel(p))))
        J = np.asarray(jax.jacfwd(lambda q: channel(q).real)(p))
        assert np.all(np.isfinite(J)), (
            f"jacfwd has {np.isnan(J).sum()}/{J.size} non-finite entries "
            f"at costh={costh}"
        )

    def tensors(c):
        return jnp.concatenate([jnp.ravel(t) for t in polarization_tensor(c, 0.7)])

    for costh in (1.0, -1.0):
        rows = np.asarray(jax.jacrev(tensors)(costh))
        assert np.all(np.isfinite(rows)), f"jacrev not finite at costh={costh}"


def test_pole_guard_leaves_the_forward_value_alone():
    """The guard must not perturb sinth anywhere it was already fine."""
    from binary_galactica.glass.response import polarization_tensor

    for costh in (-0.999, -0.3, 0.0, 0.3, 0.999):
        _, _, k = polarization_tensor(costh, 0.7)
        assert float(k[2]) == pytest.approx(-costh, rel=0, abs=0)
        sinth = np.sqrt(1.0 - costh * costh)
        assert float(k[0]) == pytest.approx(-sinth * np.cos(0.7), rel=1e-15)
    _, _, k = polarization_tensor(1.0, 0.7)
    assert float(k[0]) == 0.0 and float(k[1]) == 0.0
