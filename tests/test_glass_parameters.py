"""Physical/native parameter conversion fixes signs, scales, and coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from binary_galactica import GBParams
from binary_galactica import parameters as parameter_module


def test_physical_vector_has_explicit_glass_conventions():
    p = GBParams(
        amp=0.125,
        f0=0.125,
        fdot=0.03125,
        phi0=0.75,
        iota=np.pi / 3,
        psi=0.2,
        lam=1.2,
        beta=np.pi / 6,
    )
    expected = [0.5, 0.5, 1.2, np.log(0.25), 0.5, 0.2, -0.75, 0.5]
    got = parameter_module.to_glass_vector(p, 4.0)
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-15)


def test_native_vector_has_explicit_physical_conventions():
    native = jnp.array([0.5, 0.5, 1.2, np.log(0.25), 0.5, 0.2, -0.75, 0.5])
    got = parameter_module.from_glass_vector(native, 4.0)
    expected = [0.125, 0.125, 0.03125, 0.75, np.pi / 3, 0.2, 1.2, np.pi / 6]
    np.testing.assert_allclose(got.to_array(), expected, rtol=1e-14, atol=1e-15)


def test_conversion_preserves_catalogue_and_is_traceable(params):
    catalogue = jnp.stack([params.to_array(), params._replace(phi0=-0.2).to_array()])
    encode = jax.jit(parameter_module.to_glass_vector)
    decode = jax.jit(parameter_module.from_glass_vector)
    encoded = encode(catalogue, 128.0)
    assert encoded.shape == (2, 8)
    np.testing.assert_allclose(
        decode(encoded, 128.0).to_array(), catalogue, rtol=1e-14, atol=1e-30
    )


def test_conversion_jacobian_carries_physical_scales(params):
    duration = 128.0
    jac = np.asarray(
        jax.jacfwd(parameter_module.to_glass_vector)(params.to_array(), duration)
    )
    expected = np.zeros((8, 8))
    expected[0, 1] = duration
    expected[1, 7] = np.cos(params.beta)
    expected[2, 6] = 1.0
    expected[3, 0] = 1.0 / params.amp
    expected[4, 4] = -np.sin(params.iota)
    expected[5, 5] = 1.0
    expected[6, 3] = -1.0
    expected[7, 2] = duration**2
    np.testing.assert_allclose(jac, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("shape", [(), (7,), (9,), (3, 9)])
def test_inverse_does_not_silently_drop_frequency_derivatives(shape):
    with pytest.raises(ValueError, match="8"):
        parameter_module.from_glass_vector(jnp.zeros(shape), 1.0)
