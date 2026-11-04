"""Host validation avoids device transfers and preserves traced input packing."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass import _validation


class _JaxArrayOnly:
    """Represent an array through JAX's protocol without NumPy conversion."""

    def __init__(self, value):
        self.value = value

    def __jax_array__(self):
        return self.value


# A scalar and a nested list/tuple: _contains_jax's two host paths.
@pytest.mark.parametrize(
    "value", [2, [[2.0, 3.0], (4.0, 5.0)]], ids=["integer", "nested-sequence"]
)
def test_host_validation_does_not_transfer_to_device(value):
    with jax.transfer_guard("disallow"):
        result = _validation.finite_array(value, "value")
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.asarray(value))


@pytest.mark.parametrize(
    "value", [np.float64(2.0), [2.0, 3.0]], ids=["float64-scalar", "sequence"]
)
def test_float32_validation_casts_on_the_host(value):
    with jax.transfer_guard("disallow"):
        result = _validation.finite_array(value, "value", dtype=np.float32)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    traced = jax.jit(lambda x: _validation.finite_array(x, "value", dtype=np.float32))(
        jnp.asarray(value, dtype=jnp.float64)
    )
    assert traced.dtype == np.float32


@pytest.mark.parametrize("value", [np.inf, [1.0, np.nan]], ids=["inf", "nan-in-list"])
def test_invalid_host_values_raise_before_any_device_transfer(value):
    with jax.transfer_guard("disallow"), pytest.raises(ValueError, match="finite"):
        _validation.finite_array(value, "value")


@pytest.mark.parametrize(
    "pack", [lambda x: x, lambda x: [(x, x + 1)]], ids=["array", "nested-tuple"]
)
def test_jax_input_packing_stays_on_jax(pack):
    result = _validation.finite_array(pack(jnp.array(2.0)), "value")
    assert isinstance(result, jax.Array)
    assert result.dtype == jnp.float64


@pytest.mark.parametrize(
    "transform,pack,expected",
    [
        (lambda f: f, lambda x: [(x, x)], [[2.0, 2.0]]),
        (jax.jit, lambda x: x, 2.0),
    ],
    ids=["eager-nested-wrapper", "jit-wrapper"],
)
def test_jax_array_protocol_preserves_eager_and_traced_values(
    transform, pack, expected
):
    def convert(value):
        return _validation.finite_array(pack(_JaxArrayOnly(value)), "value")

    result = transform(convert)(jnp.array(2.0, dtype=jnp.float32))
    assert isinstance(result, jax.Array)
    assert result.dtype == jnp.float64
    np.testing.assert_array_equal(result, expected)


def test_jax_array_protocol_preserves_gradients():
    def squared(value):
        array = _validation.finite_array(_JaxArrayOnly(value), "value")
        return jnp.sum(array**2)

    np.testing.assert_array_equal(
        jax.jit(jax.grad(squared))(jnp.array([2.0, 3.0])), [4, 6]
    )


def test_jax_array_protocol_rejects_nonfinite_eager_values():
    with pytest.raises(ValueError, match="finite"):
        _validation.finite_array(_JaxArrayOnly(jnp.array([1.0, np.inf])), "value")


def test_nested_tracer_packing_supports_jit_grad_and_vmap():
    def pack(value):
        return _validation.finite_array(
            [[value, value + 1], (value + 2, value + 3)], "value"
        )

    np.testing.assert_array_equal(jax.jit(pack)(2.0), [[2, 3], [4, 5]])
    derivative = jax.jit(jax.grad(lambda x: jnp.sum(pack(x) ** 2)))(2.0)
    assert float(derivative) == 28.0
    np.testing.assert_array_equal(
        jax.jit(jax.vmap(pack))(jnp.array([1.0, 2.0])),
        [[[1, 2], [3, 4]], [[2, 3], [4, 5]]],
    )


@pytest.mark.parametrize(
    "field,value", [("amp", 0.0), ("beta", np.pi), ("lam", np.inf)]
)
def test_invalid_host_params_raise_before_any_device_transfer(params, field, value):
    invalid = params._replace(**{field: value})
    with jax.transfer_guard("disallow"), pytest.raises(ValueError):
        bg.tdi_waveform(invalid, T=7_884_000.0, BW=32)


@pytest.mark.parametrize(
    "representation",
    [
        lambda p: list(p),
        lambda p: jnp.array(p),
        lambda p: p,
        lambda p: p._replace(lam=jnp.array(p.lam)),
    ],
    # Host sequence, JAX array, GBParams of scalars (packed by validation) and
    # GBParams with an array field (packed by to_array). The NumPy array is
    # the reference itself.
    ids=["list", "jax", "params-python", "params-mixed"],
)
def test_physical_input_representations_produce_same_waveform(params, representation):
    expected = bg.tdi_waveform(np.array(params), T=7_884_000.0, BW=32)
    actual = bg.tdi_waveform(representation(params), T=7_884_000.0, BW=32)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize("array", [np.array, jnp.array], ids=["numpy", "jax"])
def test_params_array_fields_retain_broadcast_shape_validation(params, array):
    batched = params._replace(lam=array([1.0, 2.0]))
    with pytest.raises(ValueError, match=r"theta must have shape.*\(2, 8\)"):
        bg.tdi_waveform(batched, T=7_884_000.0, BW=32)


def test_named_params_support_jit_grad_and_vmap(params):
    def channels(source, duration):
        # A fixed response grid: the automatic one compiles every branch of
        # its switch on the traced duration, which test_onfly covers.
        waveform = bg.tdi_waveform(source, T=duration, BW=32, response_samples=32)
        return waveform["A"] / 1e-22

    duration = 7_884_000.0
    expected = channels(np.array(params), duration)
    np.testing.assert_allclose(
        jax.jit(channels)(params, duration), expected, rtol=1e-12, atol=0
    )
    derivative = jax.jit(
        jax.grad(lambda p, duration: jnp.sum(jnp.abs(channels(p, duration)) ** 2))
    )(params, duration)
    assert isinstance(derivative, bg.GBParams)
    assert np.all(np.isfinite(derivative))
    assert float(derivative.lam) != 0
    assert float(derivative.beta) != 0

    batch = bg.GBParams(*(jnp.array([x, x]) for x in params))
    batch = batch._replace(lam=jnp.array([params.lam, 2.3]))
    result = jax.jit(jax.vmap(channels, in_axes=(0, None)))(batch, duration)
    np.testing.assert_allclose(result[0], expected, rtol=1e-12, atol=0)
    assert not np.allclose(result[0], result[1], rtol=1e-4, atol=0)


def test_nested_traced_physical_source_returns_nan_for_invalid_values(params):
    def evaluate(latitude):
        # Packing occurs inside the transform, so only one field is traced.
        source = list(params)
        source[7] = latitude
        return bg.tdi_waveform(source, T=7_884_000.0, BW=32, response_samples=32)

    result = jax.jit(jax.vmap(evaluate))(jnp.array([params.beta, np.pi]))
    for value in result.values():
        assert np.all(np.isfinite(value[0]))
        assert np.all(np.isnan(value[1]))
