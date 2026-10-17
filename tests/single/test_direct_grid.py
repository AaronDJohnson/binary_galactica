"""Float32 batches on a directly evaluated response grid, against float64.

Four years select 512 automatic response samples, or 1024 from the float32
duration, which rounds above the threshold. With a 128- or 512-bin band
either is at least ``BW``, so both precisions evaluate the response directly
at the FFT times instead of interpolating. A batch of 32 sources
must meet the float32 target, relative L2 1e-4 and mismatch 1e-8 for A and
E, and 1e-3 for every column of their forward Jacobian, as elsewhere. The
gradient of a scaled inner product, the reverse-mode quantity of a
likelihood, must follow from that Jacobian bound. Measured worst over both
batches: relative L2 8.6e-6, mismatch 3.7e-11 and Jacobian column 9.2e-6,
as when the 512 samples were interpolated (9.1e-6, 4.1e-11 and 1.0e-5).
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from binary_galactica.glass import waveform as glass_waveform
from tests.single.support import assert_accurate, errors, source

#: Frequencies whose band, from :func:`binary_galactica.ucb_bw` at four
#: years, fits each bandwidth.
FREQUENCIES = {
    128: (1e-4, 1.5e-4, 2e-4, 3e-4, 4.5e-4, 6e-4, 8e-4, 1e-3),
    512: (1e-4, 3e-4, 7e-4, 1e-3, 1.5e-3, 2.2e-3, 3.3e-3, 5e-3),
}
EPOCHS = (0.0, 10.0)
SIGNS = ((1, 1), (-1, -1))


def batch(bandwidth):
    """Return 32 four-year sources sharing one duration and bandwidth."""
    cases = [
        source(4.0, frequency, epoch, signs, bandwidth=bandwidth)
        for frequency, epoch, signs in itertools.product(
            FREQUENCIES[bandwidth], EPOCHS, SIGNS
        )
    ]
    assert len(cases) == 32
    assert len({case.duration for case in cases}) == 1
    return cases


def stacked(cases):
    """Return the batch as float32 and float64 arrays for ``jax.vmap``."""
    return {
        "params": np.stack([case.params for case in cases]),
        "q_ref": np.array([case.q_ref for case in cases], dtype=np.int32),
        "t0": np.array([case.epoch for case in cases], dtype=np.float32),
        "native": np.stack([case.native for case in cases]),
        "carrier": np.array([case.carrier for case in cases], dtype=np.float64),
    }


@pytest.mark.slow
@pytest.mark.parametrize("bandwidth", [128, 512])
def test_four_year_batch_meets_the_float32_target(bandwidth):
    cases = batch(bandwidth)
    duration = cases[0].duration
    for real in (np.float32, np.float64):
        assert glass_waveform._automatic_samples(duration, real) >= bandwidth
    arrays = stacked(cases)

    def single(params, q_ref, t0):
        result = bg32.ucb_waveform(params, q_ref=q_ref, T=duration, t0=t0, BW=bandwidth)
        return jnp.stack((result["A"], result["E"]))

    def double(native, carrier, t0):
        result = bg.ucb_waveform(
            native, q=carrier, T=float(duration), t0=t0, BW=bandwidth
        )
        return jnp.stack((result["A"], result["E"]))

    single_args = (arrays["params"], arrays["q_ref"], arrays["t0"])
    double_args = (arrays["native"], arrays["carrier"], arrays["t0"].astype(float))
    values = jax.jit(jax.vmap(single))(*single_args)
    reference = np.asarray(jax.jit(jax.vmap(double))(*double_args))
    assert values.dtype == np.complex64
    for index in range(len(cases)):
        assert_accurate(values[index], reference[index], f"source {index}")

    jacobian = jax.jit(jax.vmap(jax.jacfwd(single)))(*single_args)
    expected = np.asarray(jax.jit(jax.vmap(jax.jacfwd(double)))(*double_args))
    for index in range(len(cases)):
        for column in range(9):
            relative_l2, _ = errors(
                jacobian[index, ..., column], expected[index, ..., column]
            )
            assert relative_l2 <= 1e-3, (index, column, relative_l2)

    # A likelihood's data term: Re<d, s*h> with s scaling each source to order
    # one, and d the float64 waveform of a displaced source.
    scale = 1 / np.max(np.abs(reference), axis=(1, 2))
    displaced = arrays["native"].copy()
    displaced[:, 0] += 0.2
    displaced[:, 6] += 0.3
    data = (
        np.asarray(jax.jit(jax.vmap(double))(displaced, *double_args[1:]))
        * scale[:, None, None]
    )

    def overlap(function, scale, data):
        def term(*args):
            return jnp.sum(jnp.real(jnp.conj(data) * scale * function(*args)))

        return term

    gradient = jax.jit(
        jax.vmap(
            lambda s, d, *args: jax.grad(overlap(single, s, d))(*args),
        )
    )(scale.astype(np.float32), data.astype(np.complex64), *single_args)
    expected_gradient = np.asarray(
        jax.jit(
            jax.vmap(
                lambda s, d, *args: jax.grad(overlap(double, s, d))(*args),
            )
        )(scale, data, *double_args)
    )
    for index in range(len(cases)):
        # |Re<d, s*dJ>| <= |d| * s * |dJ|, and |dJ| <= 1e-3 * |J| per column.
        bound = (
            1e-3
            * np.linalg.norm(data[index])
            * scale[index]
            * np.linalg.norm(expected[index].reshape(-1, 9), axis=0)
        )
        difference = np.abs(
            np.asarray(gradient[index], dtype=np.float64) - expected_gradient[index]
        )
        assert np.all(difference <= bound), (index, difference / bound)
