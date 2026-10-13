"""The native amplitude is the physical (LDC ``Amplitude``) strain amplitude.

These checks pin the overall normalization without the pinned GLASS C
fixtures, whose fast-slow polarization factors GLASS b0751bd halved. The
first maps native vectors to physical parameters by hand, with
``amp = exp(v[3])`` as in a Sangria/LDC catalogue, and requires the physical
API to reproduce the native one. The second compares the arbitrary-time
response with the long-wavelength limit of first-generation TDI ``X/Y/Z``,
written out from the strain tensor alone.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import binary_galactica as bg
from binary_galactica.glass.constants import CLIGHT, YEAR
from binary_galactica.glass.response import polarization_tensor

CHANNELS = ("X", "Y", "Z", "A", "E")


def _native_sources(rng, duration, count, frequencies):
    """Random eight-entry native vectors, a fraction of a bin off the grid.

    Cosines stay within 0.95 of the poles so that ``arccos``/``arcsin`` round
    trips through physical angles are accurate to a few ulps.
    """
    sources = []
    for _ in range(count):
        cycles = np.floor(rng.uniform(*frequencies) * duration)
        sources.append(
            [
                cycles + rng.uniform(0.1, 0.9),
                rng.uniform(-0.95, 0.95),
                rng.uniform(0.0, 2.0 * np.pi),
                np.log(rng.uniform(1e-23, 1e-21)),
                rng.uniform(-0.95, 0.95),
                rng.uniform(0.0, np.pi),
                rng.uniform(0.0, 2.0 * np.pi),
                rng.uniform(-1.0, 1.0),
            ]
        )
    return np.array(sources)


def _arm(positions, vertex, far, frequency, strain):
    """Return ``x = 2*pi*f*L/c`` and ``n.H.n`` along the arm ``vertex -> far``."""
    separation = positions[far] - positions[vertex]
    length = np.linalg.norm(separation, axis=-1)
    unit = separation / length[:, None]
    x = 2 * np.pi * frequency * length / CLIGHT
    return x, np.einsum("ti,ij,tj->t", unit, strain, unit)


@pytest.mark.parametrize("backend", ["fourier", "onfly"])
@pytest.mark.parametrize("duration", [7_884_000.0, YEAR])
def test_native_and_physical_apis_share_the_ldc_amplitude(backend, duration):
    """``ucb_waveform(v)`` is the physical source with ``amp = exp(v[3])``.

    The physical parameters are built here, not by ``from_glass_vector``, so a
    hidden amplitude factor in either adapter direction fails. With the
    Fourier backend this is also gbonfly's convention, which reproduces
    Sangria data from catalogue values.
    """
    rng = np.random.default_rng(5)
    for v in _native_sources(rng, duration, 3, (5e-4, 1e-2)):
        physical = bg.GBParams(
            amp=np.exp(v[3]),
            f0=v[0] / duration,
            fdot=v[7] / duration**2,
            phi0=-v[6],
            iota=np.arccos(v[4]),
            psi=v[5],
            lam=v[2],
            beta=np.arcsin(v[1]),
        )
        np.testing.assert_allclose(
            bg.from_glass_vector(v, duration).to_array(),
            physical.to_array(),
            rtol=1e-14,
            atol=0,
        )
        native = bg.ucb_waveform(v, T=duration, BW=64, backend=backend)
        for theta in (physical, bg.from_glass_vector(v, duration)):
            result = bg.tdi_waveform(theta, T=duration, BW=64, backend=backend)
            np.testing.assert_array_equal(
                result["freq_bin_indices"], native["freq_bin_indices"]
            )
            for channel in CHANNELS:
                expected = np.asarray(native[channel])
                residual = np.linalg.norm(np.asarray(result[channel]) - expected)
                assert residual / np.linalg.norm(expected) < 1e-12, channel


def test_response_matches_the_long_wavelength_limit():
    """Anchor the absolute scale of ``tdi_response`` at 0.1-0.3 mHz.

    With ``x = 2*pi*f*L/c << 1`` first-generation Michelson ``X`` has the
    envelope ``2*x12*x13*|n13.H.n13 - n12.H.n12|`` in fractional frequency,
    for the complex strain tensor ``H`` with ``h = Re[H*exp(i*Phi)]`` and
    full inclination factors ``amp*(1 + cos(iota)**2)`` and
    ``2*amp*cos(iota)``. ``Y`` and ``Z`` permute the vertex. Arm lengths and
    directions come from the analytic orbit at each time, because the arms
    flex by about a percent and enter squared. Seven-entry sources carry no
    amplitude evolution. Over a year the RMS envelopes agree to 3e-4
    (first-order corrections average out); a factor-of-two normalization
    error fails by fifty times the tolerance.
    """
    duration = YEAR
    times = np.linspace(0.0, duration, 33)
    positions = np.asarray(bg.analytic_orbits(jnp.asarray(times)))
    rng = np.random.default_rng(11)
    for source in _native_sources(rng, duration, 4, (1e-4, 3e-4)):
        v = source[:7]
        response = np.asarray(
            bg.tdi_response(v, times, T=duration, positions=positions)
        )
        amp, cosi, psi = np.exp(v[3]), v[4], v[5]
        plus, cross = amp * (1 + cosi**2), 2 * amp * cosi
        e_plus, e_cross, _ = (np.asarray(e) for e in polarization_tensor(v[1], v[2]))
        strain = e_plus * (
            plus * np.cos(2 * psi) + 1j * cross * np.sin(2 * psi)
        ) + e_cross * (-plus * np.sin(2 * psi) + 1j * cross * np.cos(2 * psi))
        frequency = v[0] / duration
        for channel, (vertex, first, second) in enumerate(
            ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        ):
            x_first, h_first = _arm(positions, vertex, first, frequency, strain)
            x_second, h_second = _arm(positions, vertex, second, frequency, strain)
            expected = 2 * x_first * x_second * np.abs(h_second - h_first)
            envelope = np.linalg.norm(np.abs(response[channel]))
            ratio = envelope / np.linalg.norm(expected)
            assert abs(ratio - 1) < 1e-2, ("XYZ"[channel], ratio)
