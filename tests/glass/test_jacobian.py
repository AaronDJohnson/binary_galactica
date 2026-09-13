"""Scale-aware waveform Jacobian checks across all native parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica.glass as glass


@pytest.mark.parametrize(
    "fmt,T,BW,values",
    [
        (
            "sangria",
            7_884_000.0,
            64,
            [7884.25, -0.21, 1.1, np.log(8e-22), 0.37, 0.63, 0.41, 7e-5],
        ),
        (
            "phase",
            31_536_000.0,
            32,
            [74600.25, 0.61, 4.75, np.log(3e-20), -0.08, 2.41, 1.82],
        ),
        (
            "frequency",
            126_144_000.0,
            96,
            [431970.25, -0.44, 1.57, np.log(5e-23), -0.83, 1.77, 4.39, -0.15, 0.4],
        ),
    ],
    ids=["8p-sangria-quarter-year", "7p-phase-year", "9p-frequency-four-years"],
)
def test_normalized_real_imaginary_jacobian_matches_finite_differences(
    fmt, T, BW, values
):
    """Catch wrong derivatives that merely remain finite.

    The carrier coordinate is placed a quarter-bin from its truncation
    boundary, so every central difference stays within one discrete band.
    """
    params = jnp.array(values, dtype=jnp.float64)
    positions = glass.orbit_cache(T, 12_345.0, BW)

    def raw_vector(p):
        waveform = glass.ucb_waveform(
            p,
            T=T,
            BW=BW,
            t0=12_345.0,
            fmt=fmt,
            positions=positions,
        )
        complex_vector = jnp.concatenate([waveform["A"], waveform["E"]])
        return jnp.concatenate([complex_vector.real, complex_vector.imag])

    scale = float(jnp.max(jnp.abs(raw_vector(params))))
    assert np.isfinite(scale) and scale > 0.0

    def normalized_vector(p):
        return raw_vector(p) / scale

    autodiff = np.asarray(jax.jacfwd(normalized_vector)(params))
    steps = np.full(len(values), 2e-5)
    finite_difference = np.empty_like(autodiff)
    p0 = np.asarray(params)
    for index, step in enumerate(steps):
        delta = np.zeros(len(values))
        delta[index] = step
        finite_difference[:, index] = (
            np.asarray(normalized_vector(p0 + delta))
            - np.asarray(normalized_vector(p0 - delta))
        ) / (2.0 * step)

    column_scale = np.linalg.norm(finite_difference, axis=0)
    # Every coordinate must affect the normalized waveform appreciably,
    # including fddot in the nine-parameter case. No absolute floor can hide
    # an incorrectly zero or tiny derivative at physical strain amplitudes.
    assert np.all(column_scale > 1e-3), column_scale
    relative_error = np.linalg.norm(autodiff - finite_difference, axis=0) / column_scale
    assert np.all(relative_error < 2e-5), relative_error
