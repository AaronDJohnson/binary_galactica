"""The model composes with JAX's transformations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from binary_galactica import NUM_PARAMS, PARAMETER_NAMES, polarizations


def test_vmap_over_a_catalogue_matches_a_loop(params, t):
    """A (K, 8) catalogue maps to K waveforms, one per source."""
    offsets = jnp.linspace(0.0, 1e-4, 5)
    catalogue = jnp.stack(
        [params._replace(f0=params.f0 + d).to_array() for d in offsets]
    )

    batched_hp, batched_hc = jax.vmap(polarizations, in_axes=(0, None))(catalogue, t)
    assert batched_hp.shape == (len(offsets), len(t))

    for i, theta in enumerate(catalogue):
        hp, hc = polarizations(theta, t)
        assert np.allclose(batched_hp[i], hp, rtol=1e-15, atol=0)
        assert np.allclose(batched_hc[i], hc, rtol=1e-15, atol=0)


def _finite_difference_jacobian(theta, t, rel_step=1e-6):
    """Central differences with a per-parameter relative step."""
    theta = np.asarray(theta, dtype=np.float64)
    jac_hp = np.empty((len(t), NUM_PARAMS))
    jac_hc = np.empty((len(t), NUM_PARAMS))
    for i in range(NUM_PARAMS):
        step = rel_step * abs(theta[i]) if theta[i] != 0.0 else rel_step
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        hp_p, hc_p = polarizations(jnp.asarray(plus), t)
        hp_m, hc_m = polarizations(jnp.asarray(minus), t)
        jac_hp[:, i] = (np.asarray(hp_p) - np.asarray(hp_m)) / (2.0 * step)
        jac_hc[:, i] = (np.asarray(hc_p) - np.asarray(hc_m)) / (2.0 * step)
    return jac_hp, jac_hc


def test_jacobian_matches_finite_differences(params):
    """Jacfwd reproduces central differences on all eight parameters.

    Uses a strongly chirping source over a short span so that every parameter
    -- fdot in particular -- is well conditioned for differencing.
    """
    p = params._replace(fdot=1e-11)
    theta = p.to_array()
    t = jnp.linspace(0.0, 1.0e5, 129)

    jac_hp, jac_hc = jax.jacfwd(polarizations)(theta, t)
    fd_hp, fd_hc = _finite_difference_jacobian(theta, t)

    for exact, approx, name in ((jac_hp, fd_hp, "hp"), (jac_hc, fd_hc, "hc")):
        exact = np.asarray(exact)
        scale = np.abs(exact).max(axis=0)
        for i, param in enumerate(PARAMETER_NAMES):
            tol = 1e-5 * max(scale[i], np.abs(approx[:, i]).max(), 1e-300)
            assert np.allclose(exact[:, i], approx[:, i], rtol=0, atol=tol), (
                f"d{name}/d{param} disagrees with finite differences"
            )


def test_params_pytree_flows_through_grad(params, t):
    """GBParams is a pytree, so grad returns a GBParams of derivatives."""

    def squared_norm(p):
        hp, hc = polarizations(p, t)
        return jnp.sum(hp**2 + hc**2)

    grad = jax.grad(squared_norm)(params)
    assert grad._fields == PARAMETER_NAMES
    assert float(grad.lam) == 0.0
    assert float(grad.beta) == 0.0
