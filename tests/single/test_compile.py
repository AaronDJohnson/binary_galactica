"""The float32 phase is computed once, outside the channel kernels.

``grid_cycles`` sums the restored phase polynomial in float32 pairs, about 600
dependent operations. Fused into each channel kernel, it made LLVM take 4-9 s
to compile every float32 waveform with a ``fddot`` entry instead of ~0.3 s.
An optimization barrier on its output keeps it in its own kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from binary_galactica import single as bg32
from binary_galactica.glass import _twofloat as tf
from binary_galactica.glass import single as gs

from .support import source


def test_grid_cycles_ends_in_an_optimization_barrier():
    case = source(0.25, 3e-3)
    obs = case.observation()
    params = jnp.asarray(case.params)
    residual = tf.pair(params[0] - jnp.floor(params[0]))
    jaxpr = jax.make_jaxpr(
        lambda p: gs.grid_cycles(residual, p, obs["T"], obs["t0"], obs["BW"])
    )(params).jaxpr
    assert jaxpr.eqns[-1].primitive.name == "optimization_barrier"


def test_float32_waveform_with_fddot_keeps_the_barrier():
    case = source(0.25, 3e-3)
    assert case.params.shape == (9,)
    jaxpr = jax.make_jaxpr(
        lambda p: bg32.ucb_waveform(p, q_ref=case.q_ref, **case.observation())["A"]
    )(np.asarray(case.params))
    assert "optimization_barrier" in str(jaxpr)
