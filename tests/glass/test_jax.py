"""The JAX contract: float64, jit, vmap, grad, and no NaNs at the edges.

The point of the port is that the generator composes with JAX transforms, so
these are behavioural requirements, not nice-to-haves. In particular a
``NaN`` in a reverse-mode gradient is fatal to a sampler that never sees the
forward value.
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


def test_is_float64(injection, glass_params):
    T = injection["T"]
    w = gbonfly.ucb_waveform(
        glass_params, T=T, BW=64, positions=gbonfly.orbit_cache(T, 0.0, 64)
    )
    assert w["A"].dtype == jnp.complex128
    assert gbonfly.orbit_cache(T, 0.0, 64).dtype == jnp.float64


@pytest.mark.parametrize("fmt", ["sangria", "phase", "frequency"])
def test_vmap_over_sources(injection, fmt):
    T, BW = injection["T"], 64
    orb = gbonfly.orbit_cache(T, 0.0, BW)
    batch = jnp.stack([_p(injection, f0=1e-3 + 1e-7 * i) for i in range(8)])
    fn = jax.vmap(lambda p: gbonfly.ucb_waveform(p, T=T, BW=BW, fmt=fmt, positions=orb))
    out = fn(batch)
    assert out["A"].shape == (8, BW)
    # Each source individually must reproduce its row. atol=0: these
    # amplitudes are ~1e-22, so the default absolute tolerance would pass
    # regardless of what the batched call returned.
    #
    # The batched and scalar paths are NOT bit-identical: XLA reassociates
    # the batched FFT, which costs ~1e-12 relative. That is the same order
    # as this package's agreement with GLASS itself, and four orders below
    # anything the fast-slow approximation controls -- but it does mean a
    # vmap'd sampler will not reproduce an unbatched one bit for bit.
    one = gbonfly.ucb_waveform(batch[3], T=T, BW=BW, fmt=fmt, positions=orb)
    np.testing.assert_array_equal(out["freq_bin_indices"][3], one["freq_bin_indices"])
    assert out["qmin"][3] == one["qmin"]
    for channel in ("X", "Y", "Z", "A", "E"):
        a, b = np.asarray(out[channel][3]), np.asarray(one[channel])
        assert np.abs(a - b).max() / np.abs(b).max() < 1e-10


def test_grad_is_finite(injection):
    """Reverse-mode through the whole generator, including the guarded sinc."""
    T, BW = injection["T"], 64
    orb = gbonfly.orbit_cache(T, 0.0, BW)

    def power(p):
        w = gbonfly.ucb_waveform(p, T=T, BW=BW, positions=orb)
        return jnp.sum(jnp.abs(w["A"]) ** 2 + jnp.abs(w["E"]) ** 2)

    g = jax.grad(power)(_p(injection))
    assert np.all(np.isfinite(np.asarray(g))), g
    assert np.any(np.asarray(g) != 0.0)


def test_phase_gradient_matches_normalized_complex_finite_difference(injection):
    """Phase rotates a complex coefficient even though its power is constant."""
    T, BW = injection["T"], 64
    orb = gbonfly.orbit_cache(T, 0.0, BW)
    p0 = jnp.asarray(_p(injection))
    base = gbonfly.ucb_waveform(p0, T=T, BW=BW, positions=orb)["A"]
    index = int(jnp.argmax(jnp.abs(base)))
    scale = float(jnp.abs(base[index]))
    assert np.isfinite(scale) and scale > 0.0

    def quadratures(phase):
        p = p0.at[6].set(phase)
        value = gbonfly.ucb_waveform(p, T=T, BW=BW, positions=orb)["A"][index]
        return jnp.array([value.real, value.imag]) / scale

    autodiff = np.asarray(jax.jacrev(quadratures)(p0[6]))
    h = 2e-5
    difference = np.asarray(quadratures(p0[6] + h) - quadratures(p0[6] - h)) / (2 * h)
    norm = np.linalg.norm(difference)
    assert 0.99 < norm < 1.01  # The normalized phase derivative has unit magnitude.
    assert np.linalg.norm(autodiff - difference) / norm < 1e-6


def test_sinc_guard_at_zero_argument():
    """An arm exactly along the line of sight gives ``arg1 == 0``.

    GLASS computes ``sin(0)/0`` there and returns NaN. We return 1 and stay
    differentiable, which is the only difference in behaviour.
    """
    from binary_galactica.glass.waveform import _sinc

    z = jnp.array([0.0, 1e-12, 1e-3, 1.0])
    v = _sinc(z)
    assert np.all(np.isfinite(np.asarray(v)))
    assert v[0] == pytest.approx(1.0)
    assert float(v[3]) == pytest.approx(np.sin(1.0) / 1.0, rel=1e-12)
    d = jax.grad(lambda x: _sinc(x).sum())(z)
    assert np.all(np.isfinite(np.asarray(d)))


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


def test_odd_bandwidth_rejected(injection):
    with pytest.raises(ValueError, match="even"):
        gbonfly.ucb_waveform(_p(injection), T=injection["T"], BW=63)


@pytest.mark.parametrize("costh", [1.0, -1.0, np.cos(1e-8)])
def test_derivatives_finite_at_the_ecliptic_poles(costh):
    """``sqrt(1-costh^2)`` must not poison AD at ``costh = +-1``.

    Its derivative there is ``-costh/0``, and in forward mode the resulting
    ``0/0`` contaminates *every* tangent direction -- ``jacfwd`` came back
    all-NaN, including the columns for ``phi0`` and ``ln(amp)`` whose
    derivatives are perfectly well defined at the pole. The forward value is
    finite throughout, so only a derivative test catches it.

    ``cos(1e-8) == 1.0`` exactly in float64, so this is reachable from any
    colatitude within ~1e-8 rad of the pole, not just from the pole itself.
    """
    T, BW = 7884000.0, 64
    orb = gbonfly.orbit_cache(T, 0.0, BW)
    p = jnp.array([7884.0, costh, 0.0, jnp.log(1e-22), 0.3, 0.7, 0.5, 0.0])

    def power(q):
        w = gbonfly.ucb_waveform(q, T=T, BW=BW, positions=orb)
        return jnp.sum(jnp.abs(w["A"]) ** 2)

    assert np.isfinite(float(power(p)))

    g = np.asarray(jax.grad(power)(p))
    assert np.all(np.isfinite(g)), f"grad not finite at costh={costh}: {g}"

    J = np.asarray(
        jax.jacfwd(
            lambda q: gbonfly.ucb_waveform(q, T=T, BW=BW, positions=orb)["A"].real
        )(p)
    )
    assert np.all(np.isfinite(J)), (
        f"jacfwd has {np.isnan(J).sum()}/{J.size} non-finite entries"
    )

    H = np.asarray(jax.hessian(power)(p))
    assert np.all(np.isfinite(H))


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
