"""The parameter container is a faithful, order-preserving view of the vector."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica import NUM_PARAMS, PARAMETER_NAMES, GBParams


def test_canonical_order_is_the_documented_one():
    assert PARAMETER_NAMES == (
        "amp",
        "f0",
        "fdot",
        "phi0",
        "iota",
        "psi",
        "lam",
        "beta",
    )
    assert NUM_PARAMS == 8
    assert GBParams._fields == PARAMETER_NAMES


def test_array_roundtrip(params):
    assert GBParams.from_array(params.to_array()) == pytest.approx(tuple(params))


def test_array_slots_follow_the_canonical_order(params):
    theta = params.to_array()
    for i, name in enumerate(PARAMETER_NAMES):
        assert float(theta[i]) == pytest.approx(float(getattr(params, name)))


def test_batched_roundtrip(params):
    catalogue = jnp.stack([params.to_array()] * 4)
    restored = GBParams.from_array(catalogue)
    assert restored.amp.shape == (4,)
    assert np.allclose(restored.to_array(), catalogue)


def test_scalars_broadcast_against_arrays():
    p = GBParams(
        amp=jnp.array([1e-22, 2e-22]),
        f0=1e-3,
        fdot=0.0,
        phi0=0.0,
        iota=0.0,
        psi=0.0,
        lam=0.0,
        beta=0.0,
    )
    assert p.to_array().shape == (2, NUM_PARAMS)


def test_from_array_passes_through_existing_params(params):
    assert GBParams.from_array(params) is params


@pytest.mark.parametrize("shape", [(7,), (9,), (4, 3), ()])
def test_wrong_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="trailing axis"):
        GBParams.from_array(jnp.zeros(shape))


def test_is_a_pytree(params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    assert len(leaves) == NUM_PARAMS
    assert jax.tree_util.tree_unflatten(treedef, leaves) == params
