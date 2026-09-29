"""Float32 functions under ``jax.jacfwd``, ``jax.jit`` and ``jax.vmap``.

Forward Jacobians of A/E with respect to the offset native coordinates must
agree with float64 to relative L2 1e-3 in every parameter column. Compiled
and batched evaluations must match eager float32 calls, and float64 at
identical inputs.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica import single as bg32
from tests.single.support import (
    Source,
    assert_accurate,
    channels,
    errors,
    physical,
    source,
)

JACOBIAN_SOURCES = {
    "quarter year at 30 mHz, late": source(0.25, 3e-2, 10.0, (-1, 1)),
    "one year at 10 mHz, late": source(1.0, 1e-2, 10.0, (1, -1)),
    "four years at 1 mHz": source(4.0, 1e-3, 0.0, (1, 1)),
}


def channels_of(result):
    """Return A and E without converting traced values."""
    return result["A"], result["E"]


@pytest.mark.parametrize("fmt", ["sangria", "phase"])
@pytest.mark.parametrize("name", list(JACOBIAN_SOURCES))
def test_forward_jacobian_matches_float64(name, fmt):
    case = JACOBIAN_SOURCES[name]

    def single(params):
        result = bg32.ucb_waveform(
            params, q_ref=case.q_ref, fmt=fmt, **case.observation()
        )
        return jnp.stack(channels_of(result))

    def double(native):
        result = bg.ucb_waveform(
            native, q=float(case.carrier), fmt=fmt, **case.double_observation()
        )
        return jnp.stack(channels_of(result))

    jacobian = jax.jacfwd(single)(case.params)
    reference = np.asarray(jax.jacfwd(double)(case.native))
    assert jacobian.dtype == np.complex64
    assert jacobian.shape == reference.shape == (2, case.bandwidth, 9)
    for column in range(9):
        relative_l2, _ = errors(jacobian[..., column], reference[..., column])
        assert relative_l2 <= 1e-3, f"column {column}: {relative_l2:.3g}"
    compiled = jax.jit(jax.jacfwd(single))(case.params)
    relative_l2, _ = errors(compiled, reference)
    assert relative_l2 <= 1e-3


def test_physical_jacobian_matches_float64():
    """The physical adapter's Jacobian, with the frequency offset in hertz."""
    case = JACOBIAN_SOURCES["one year at 10 mHz, late"]
    theta, double_theta = physical(case)

    def single(values):
        result = bg32.tdi_waveform(values, q_ref=case.q_ref, **case.observation())
        return jnp.stack(channels_of(result))

    def double(values):
        result = bg.tdi_waveform(values, **case.double_observation())
        return jnp.stack(channels_of(result))

    jacobian = jax.jacfwd(single)(theta)
    reference = np.asarray(jax.jacfwd(double)(double_theta))
    assert jacobian.dtype == np.complex64
    for column in range(8):
        relative_l2, _ = errors(jacobian[..., column], reference[..., column])
        assert relative_l2 <= 1e-3, f"column {column}: {relative_l2:.3g}"


CASE = JACOBIAN_SOURCES["one year at 10 mHz, late"]
STATIC = {"BW": CASE.bandwidth, "response_samples": CASE.samples}


def test_jit_with_every_array_argument_traced():
    """Reference bin, duration, epoch, positions and override all traced."""
    cache = np.asarray(
        bg.orbit_cache(float(CASE.duration), float(CASE.epoch), CASE.bandwidth)
    ).astype(np.float32)

    @jax.jit
    def compiled(params, q_ref, duration, epoch, positions, q):
        return bg32.ucb_waveform(
            params,
            q_ref=q_ref,
            T=duration,
            t0=epoch,
            positions=positions,
            q=q,
            **STATIC,
        )

    q = np.int32(CASE.carrier + 1)
    result = compiled(
        CASE.params, np.int32(CASE.q_ref), CASE.duration, CASE.epoch, cache, q
    )
    eager = CASE.single(positions=cache, q=q)
    assert result["qmin"].dtype == np.int32
    assert int(result["qmin"]) == int(eager["qmin"])
    assert errors(channels(result), channels(eager))[0] < 1e-5
    double = CASE.double(positions=cache.astype(np.float64), q=float(q))
    assert_accurate(channels(result), channels(double))


def test_jit_of_the_other_entry_points():
    times = np.linspace(-1e6, 6.2e8, 17, dtype=np.float32)
    window = {"bin_lo": CASE.carrier - 50, "n_bins": 100}
    theta, _ = physical(CASE)

    @jax.jit
    def compiled(params, theta, q_ref, duration, epoch):
        observation = {"T": duration, "t0": epoch, **STATIC}
        return (
            bg32.template_on_window(params, q_ref=q_ref, **window, **observation),
            bg32.tdi_response(params, times, q_ref=q_ref, T=duration),
            bg32.tdi_waveform(theta, q_ref=q_ref, **observation)["A"],
        )

    traced = compiled(
        CASE.params, theta, np.int32(CASE.q_ref), CASE.duration, CASE.epoch
    )
    eager = (
        bg32.template_on_window(
            CASE.params, q_ref=CASE.q_ref, **window, **CASE.observation()
        ),
        bg32.tdi_response(CASE.params, times, q_ref=CASE.q_ref, T=CASE.duration),
        bg32.tdi_waveform(theta, q_ref=CASE.q_ref, **CASE.observation())["A"],
    )
    for value, expected in zip(traced, eager, strict=True):
        assert value.dtype == np.complex64
        assert errors(value, expected)[0] < 1e-5


def batch_sources() -> list[Source]:
    """Return three sources with ``CASE``'s reference bin and observation."""
    variants = []
    for offset, first, sky in ((0.1, 1.0, 0.3), (0.55, -0.5, -0.8), (0.97, 0.2, 0.0)):
        params = CASE.params.copy()
        params[[0, 7, 1]] = offset, first * CASE.params[7], sky
        variants.append(dataclasses.replace(CASE, params=params))
    return variants


def test_vmap_over_params():
    cases = batch_sources()

    def evaluate(params):
        return bg32.ucb_waveform(params, q_ref=CASE.q_ref, **CASE.observation())

    batch = jax.vmap(evaluate)(np.stack([c.params for c in cases]))
    for index, case in enumerate(cases):
        member = {key: value[index] for key, value in batch.items()}
        assert int(member["qmin"]) == case.carrier - case.bandwidth // 2
        assert errors(channels(member), channels(case.single()))[0] < 1e-5
        assert_accurate(channels(member), channels(case.double()), f"source {index}")


@pytest.mark.parametrize("jit", [False, True], ids=["vmap", "jit of vmap"])
def test_vmap_over_reference_bins(jit):
    """One offset vector against several reference bins: different sources."""
    shifts = np.array([0, 7, -1000, 123_456], dtype=np.int32)
    q_refs = np.int32(CASE.q_ref) + shifts

    def evaluate(q_ref):
        return bg32.ucb_waveform(CASE.params, q_ref=q_ref, **CASE.observation())

    batched = jax.vmap(evaluate)
    batch = (jax.jit(batched) if jit else batched)(q_refs)
    assert batch["qmin"].dtype == np.int32
    for index, q_ref in enumerate(q_refs):
        case = dataclasses.replace(CASE, q_ref=int(q_ref))
        member = {key: value[index] for key, value in batch.items()}
        assert int(member["qmin"]) == case.carrier - case.bandwidth // 2
        np.testing.assert_array_equal(
            member["freq_bin_indices"],
            case.carrier - case.bandwidth // 2 + np.arange(case.bandwidth),
        )
        assert_accurate(channels(member), channels(case.double()), f"q_ref {q_ref}")


def test_vmap_over_params_and_reference_bins_in_every_entry_point():
    cases = [
        dataclasses.replace(case, q_ref=case.q_ref + shift)
        for case, shift in zip(batch_sources(), (0, 3, -40), strict=True)
    ]
    params = np.stack([c.params for c in cases])
    q_refs = np.array([c.q_ref for c in cases], dtype=np.int32)
    times = np.linspace(0.0, 6e8, 9, dtype=np.float32)
    window = {"bin_lo": CASE.carrier - 200, "n_bins": 400}
    observation = CASE.observation()

    def evaluate(p, q):
        return (
            bg32.ucb_waveform(p, q_ref=q, **observation)["A"],
            bg32.template_on_window(p, q_ref=q, **window, **observation),
            bg32.tdi_response(p, times, q_ref=q, T=CASE.duration),
        )

    batch = jax.vmap(evaluate)(params, q_refs)
    for index, case in enumerate(cases):
        eager = (
            case.single()["A"],
            bg32.template_on_window(
                case.params, q_ref=case.q_ref, **window, **observation
            ),
            bg32.tdi_response(case.params, times, q_ref=case.q_ref, T=CASE.duration),
        )
        for value, expected in zip(batch, eager, strict=True):
            assert errors(value[index], expected)[0] < 1e-5, f"source {index}"
        double = bg.tdi_response(
            case.native, times.astype(np.float64), T=float(CASE.duration)
        )
        assert_accurate(batch[2][index], double, f"response {index}")


def test_vmap_of_the_physical_adapter():
    cases = [
        dataclasses.replace(case, q_ref=case.q_ref + shift)
        for case, shift in zip(batch_sources()[:2], (0, 9), strict=True)
    ]
    parameters = [physical(case) for case in cases]
    theta = np.stack([single for single, _ in parameters])
    q_refs = np.array([case.q_ref for case in cases], dtype=np.int32)

    def evaluate(values, q_ref):
        return bg32.tdi_waveform(values, q_ref=q_ref, **CASE.observation())

    batch = jax.vmap(evaluate)(theta, q_refs)
    for index, (_, double_theta) in enumerate(parameters):
        double = bg.tdi_waveform(double_theta, **CASE.double_observation())
        member = {key: value[index] for key, value in batch.items()}
        assert int(member["qmin"]) == int(double["qmin"])
        assert_accurate(channels(member), channels(double), f"source {index}")
