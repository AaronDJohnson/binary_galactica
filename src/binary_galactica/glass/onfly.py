# Copyright 2019 Tyson B. Littenberg & Neil J. Cornish
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Arbitrary-time first-generation TDI, following GLASS's spline response.

The eight retarded source evaluations of ``LISA_TDI_spline`` are evaluated
analytically for a UCB, with geometry frozen at each reception time. Arms use
their actual lengths. This is the frozen-constellation unequal-arm response,
not a solution for moving spacecraft along each light path.

By default the response is exactly GLASS's ``(M-i*Mf)*exp(-i*phi(t))``.
An internal option instead removes the reference-spacecraft phase, keeping
the orbital Doppler phase outside sparse response interpolation. Neither
response has Fourier normalization applied. Both are already the physical
Sangria response for the strain amplitude ``exp(params[3])`` (LDC
``Amplitude``); the public ``tdi_response`` returns them without rescaling.
Pairing source differences before dividing by ``1 +/- k.n`` makes aligned-arm
limits finite and differentiable without clipping a physical denominator.

The pinned spline response and fast-slow response place directed links
differently in the Michelson combination. For static equal arms and a
monochromatic source this raw response equals the package's fast-slow
Sangria combination, which uses full polarization factors, *after reversing
each directed link*; it is twice the pinned 079c6a3 fast-slow combination,
whose factors GLASS b0751bd halved. Without that reversal the two responses
converge only in the long-wavelength limit.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._precision import CONTRACTION, cast, real_dtype
from ._trig import cis, cos_sinc
from .constants import CLIGHT, PI2
from .orbit import analytic_orbits
from .response import polarization_tensor


def reference_phase_shift(params, times, T, positions=None):
    """Return ``phi(t-k.x0/c)-phi(t)`` using stable polynomial differences.

    Multiplying the spacecraft-reference response by ``exp(1j*shift)`` restores
    the default source-phase-demodulated response. Inputs and shapes follow
    :func:`tdi_response`; spacecraft zero supplies the reference position.
    """
    real = real_dtype(params)
    params = jnp.asarray(params, dtype=real)
    times = jnp.asarray(times, dtype=real)
    T = cast(T, real)
    f0 = params[0] / T
    zero = jnp.zeros((), dtype=real)
    fdot = params[7] / T**2 if params.shape[0] > 7 else zero
    fddot = params[8] / T**3 if params.shape[0] > 8 else zero
    _, _, k = polarization_tensor(params[1], params[2])
    if positions is None:
        positions = analytic_orbits(times)
    positions = jnp.asarray(positions, dtype=real)
    delay = jnp.einsum("ta,a->t", positions[0] / CLIGHT, k, precision=CONTRACTION)
    return PI2 * (
        -delay * (f0 + fdot * times + 0.5 * fddot * times**2)
        + 0.5 * delay**2 * (fdot + fddot * times)
        - fddot * delay**3 / 6
    )


def tdi_response(params, times, T, positions=None, *, reference_phase=False):
    """Return source-phase-demodulated raw XYZ, shape ``(3, N)``.

    ``params`` has the usual 7/8/9 GLASS parameters, ``times`` has shape
    ``(N,)`` in seconds, and optional positions have shape ``(3,N,3)`` in
    metres. This internal kernel assumes validated inputs and computes in the
    floating dtype of ``params``, casting times, ``T`` and positions to it.
    ``phi0`` cancels exactly because the full analytic source phase is removed.
    Unlike the C spline helper, which fixes fddot=0, the nine-parameter model
    retains fddot.

    The static ``reference_phase=True`` option removes ``phi(t-k.x0/c)``
    instead, matching the reference used by GLASS's UCB spline interpolation.
    This changes only the demodulation phase; source amplitudes, frequencies
    and retarded times are the same. Use :func:`reference_phase_shift` to
    restore the default phase convention after interpolation.
    """
    real = real_dtype(params)
    params = jnp.asarray(params, dtype=real)
    times = jnp.asarray(times, dtype=real)
    T = cast(T, real)
    f0 = params[0] / T
    zero = jnp.zeros((), dtype=real)
    fdot = params[7] / T**2 if params.shape[0] > 7 else zero
    fddot = params[8] / T**3 if params.shape[0] > 8 else zero
    amp = jnp.exp(params[3])
    slope = (2 / 3) * fdot / f0
    cosi, psi = params[4], params[5]
    ep, ec, k = polarization_tensor(params[1], params[2])
    dp = 0.5 * (1 + cosi**2) * jnp.cos(2 * psi) + 1j * cosi * jnp.sin(2 * psi)
    dc = -0.5 * (1 + cosi**2) * jnp.sin(2 * psi) + 1j * cosi * jnp.cos(2 * psi)

    if positions is None:
        positions = analytic_orbits(times)
    p = jnp.asarray(positions, dtype=real) / CLIGHT
    arm = jnp.stack((p[1] - p[2], p[2] - p[0], p[0] - p[1]))
    lengths = jnp.sqrt(jnp.sum(arm**2, axis=-1))
    unit = arm / lengths[..., None]
    mu = jnp.einsum("sta,a->st", unit, k, precision=CONTRACTION)
    kr = jnp.einsum("sta,a->st", p, k, precision=CONTRACTION)
    if reference_phase:
        # Center source time and all delays on spacecraft zero before any
        # complex exponentials. This removes orbital Doppler phase directly,
        # without computing and subsequently cancelling rapid exponentials.
        reference_delay = kr[0]
        times = times - reference_delay
        kr = kr - reference_delay[None, :]

    def project(tensor):
        return jnp.einsum("sti,ij,stj->st", unit, tensor, unit, precision=CONTRACTION)

    antenna = dp * project(ep) + dc * project(ec)

    def difference(first, second, length, denominator):
        # first-second = length*denominator algebraically. Evaluate that
        # identity directly so an aligned arm never causes 0/0 or cancellation.
        delay = 0.5 * (first + second)
        delta_tau = -length * denominator
        tau = times - delay
        frequency = f0 + fdot * tau + 0.5 * fddot * tau**2
        secant_frequency = frequency + fddot * delta_tau**2 / 24
        half_phase = jnp.pi * delta_tau * secant_frequency
        # Polynomial phase differences retain precision at large epochs.
        phase = (
            PI2
            * (
                -delay * (f0 + fdot * times + 0.5 * fddot * times**2)
                + 0.5 * delay**2 * (fdot + fddot * times)
                - fddot * delay**3 / 6
            )
            + 0.25 * jnp.pi * (fdot + fddot * tau) * delta_tau**2
        )
        # One cosine-sine pair serves both factors of the half phase.
        cos_half, sinc_half = cos_sinc(half_phase)
        return (
            amp
            * cis(phase)
            * (
                -slope * length * cos_half
                - 2j
                * (1 + slope * tau)
                * jnp.pi
                * length
                * secant_frequency
                * sinc_half
            )
        )

    # Carry all three cyclic channels and all eight divided differences on
    # batch axes. This keeps the source expression a single graph under JVP,
    # Hessians and JIT instead of expanding it 24 times in Python.
    b, c = jnp.array([1, 2, 0]), jnp.array([2, 0, 1])
    lb, lc = lengths[b], lengths[c]
    krb, krc = kr[b], kr[c]
    delays = jnp.stack(
        (
            kr + 2 * lb + 2 * lc,
            krb + 2 * lb + lc,
            krc + lb + 2 * lc,
            kr + 2 * lb,
            kr + 2 * lc,
            krc + lb,
            krb + lc,
            kr,
        )
    )
    first = delays[jnp.array([0, 3, 0, 4, 1, 6, 2, 5])]
    second = delays[jnp.array([2, 5, 1, 6, 3, 7, 4, 7])]
    pair_lengths = jnp.stack((lb, lb, lc, lc, lc, lc, lb, lb))
    mub, muc = mu[b], mu[c]
    minus_b, plus_b = 1 - mub, 1 + mub
    minus_c, plus_c = 1 - muc, 1 + muc
    denominators = jnp.stack(
        (minus_b, minus_b, plus_c, plus_c, minus_c, minus_c, plus_b, plus_b)
    )
    values = difference(first, second, pair_lengths, denominators)
    # Each channel is antenna[b]*(v0 - v1 + v6 - v7) +
    # antenna[c]*(-v2 + v3 - v4 + v5). Weighting every difference by its
    # signed antenna factor makes that one reduction over the batch axis.
    # Its rounding differs from differencing pairs first by about 2e-14 of
    # the channel at 30 uHz, where the pairs nearly cancel, falling as 1/f.
    ab, ac = antenna[b], antenna[c]
    weights = jnp.stack((ab, -ab, -ac, ac, -ac, ac, ab, -ab))
    return jnp.sum(weights * values, axis=0)


__all__ = ["reference_phase_shift", "tdi_response"]
