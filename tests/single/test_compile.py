"""The float32 phase is computed once, outside the channel kernels.

``grid_cycles`` sums the restored phase polynomial in float32 pairs, about 600
dependent operations. Fused into each channel kernel, it made LLVM take 4-9 s
to compile every float32 waveform with a ``fddot`` entry instead of ~0.3 s.
A fence (an optimization barrier with portable batching and derivative rules)
on its output keeps it in its own kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from binary_galactica import single as bg32
from binary_galactica.glass import _twofloat as tf
from binary_galactica.glass import single as gs

from .support import source


def test_grid_cycles_is_fenced():
    case = source(0.25, 3e-3)
    obs = case.observation()
    params = jnp.asarray(case.params)
    residual = tf.pair(params[0] - jnp.floor(params[0]))
    jaxpr = jax.make_jaxpr(
        lambda p: gs.grid_cycles(residual, p, obs["T"], obs["t0"], obs["BW"])
    )(params)
    assert "optimization_barrier" in str(jaxpr)


def test_float32_waveform_with_fddot_keeps_the_fence():
    case = source(0.25, 3e-3)
    assert case.params.shape == (9,)
    jaxpr = jax.make_jaxpr(
        lambda p: bg32.ucb_waveform(p, q_ref=case.q_ref, **case.observation())["A"]
    )(np.asarray(case.params))
    assert "optimization_barrier" in str(jaxpr)


def test_fence_is_the_identity_under_every_transform():
    """Holds on JAX versions whose barrier lacks batching or derivative rules."""
    x = jnp.linspace(-1.0, 2.0, 5, dtype=jnp.float32)
    batch = jnp.stack([x, 2 * x])
    np.testing.assert_array_equal(jax.jit(gs._fence)(x), x)
    np.testing.assert_array_equal(jax.vmap(gs._fence)(batch), batch)
    nested = batch[None]
    np.testing.assert_array_equal(jax.vmap(jax.vmap(gs._fence))(nested), nested)
    identity = np.eye(5, dtype=np.float32)
    np.testing.assert_array_equal(jax.jacfwd(gs._fence)(x), identity)
    np.testing.assert_array_equal(jax.jacrev(gs._fence)(x), identity)
    np.testing.assert_array_equal(
        jax.vmap(jax.jacfwd(gs._fence))(batch), np.stack([identity, identity])
    )
    square = jax.grad(lambda v: jnp.sum(gs._fence(v**2)))
    np.testing.assert_array_equal(square(x), 2 * x)
    np.testing.assert_array_equal(
        jax.hessian(lambda v: jnp.sum(gs._fence(v**2)))(x), 2 * identity
    )
