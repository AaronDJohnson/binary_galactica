# Copyright 2019 Tyson B. Littenberg & Neil J. Cornish
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The GLASS fast-slow (``ucb_waveform``) UCB generator, in JAX.

A transcription of ``ucb_waveform`` in
``glass/ucb/src/glass_ucb_waveform.c``, vectorised over the slow-time grid
and over sources. The algorithm is the fast-slow / reduced-DFT generator of
Cornish & Littenberg; this module follows GLASS's structure so the method
can be cited rather than re-derived.

Structure
---------
GLASS ties the slow-time sample count to the output bandwidth: it evaluates
the six single-link responses at ``BW`` times, takes one ``BW``-point
complex FFT, and returns ``BW`` frequency bins. There is no separate
``n_red`` / ``n_window`` / ``n_slow``, and consequently **no out-of-band
mask** --- every returned bin is a bin the transform actually resolved.
Refining the slow grid and widening the band are the same knob, which is
what makes GLASS's SNR-driven bandwidth heuristic
(:func:`binary_galactica.glass.bandwidth.ucb_bandwidth`) do double duty.

Deliberate deviations from the C, all documented at the point of use:

* the single-link ``sin(arg)/arg`` is evaluated with a guarded ``sinc`` so
  that ``arg == 0`` (an arm exactly along the line of sight) returns 1
  rather than ``NaN``, and so that reverse-mode gradients are finite there;
* ``c1, s1`` are computed directly rather than by GLASS's per-bin trig
  recursion, which is a scalar-loop optimisation with no effect on the
  result (see :mod:`binary_galactica.glass.tdi`);
* everything is float64 and pure, so ``jit``/``vmap``/``grad`` compose.

The transcription targets GLASS commit
``079c6a3a6fcf380a7c107f39640fff7fc720d0f1``. Its reduced FFT uses the
literal ``1/(2*BW)`` normalization from that revision.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.custom_batching import custom_vmap
from jax.typing import ArrayLike

from . import _validation
from .constants import CLIGHT, FSTAR, PI2
from .orbit import analytic_orbits, arm_vectors, slow_time_grid
from .response import LINKS, detector_tensor, polarization_tensor
from .tdi import synthesize
from .types import WaveformResult


try:
    from jax.lax import optimization_barrier as _optimization_barrier
except ImportError:  # JAX 0.4.30 predates the public spelling.
    from jax._src.ad_checkpoint import _optimization_barrier


@custom_vmap
def _carrier_bin(scaled_frequency, duration):
    """Preserve GLASS's separate IEEE division and multiplication under JIT."""
    scaled_frequency, duration = _optimization_barrier((scaled_frequency, duration))
    frequency = _optimization_barrier(scaled_frequency / duration)
    return jnp.trunc(frequency * duration)


@_carrier_bin.def_vmap
def _carrier_bin_vmap(axis_size, in_batched, scaled_frequency, duration):
    """Apply the scalar carrier rule elementwise, including nested batches."""
    scaled_frequency, duration = (
        value if batched else jnp.broadcast_to(value, (axis_size, *value.shape))
        for value, batched in zip((scaled_frequency, duration), in_batched, strict=True)
    )
    return _carrier_bin(scaled_frequency, duration), True


def _sinc(z):
    """``sin(z)/z``, finite and differentiable at ``z = 0``.

    GLASS computes ``sin(arg1)/arg1`` directly, which is ``NaN`` when an arm
    lies exactly along the propagation direction. That is a measure-zero
    defect rather than a convention, so we guard it; the two agree
    everywhere GLASS is finite.
    """
    # Direct AD of sin(z)/z subtracts O(1/z) terms for the gradient and
    # O(1/z**2) terms for the Hessian. A machine-epsilon-sized cutoff only
    # fixes z=0; it leaves catastrophic cancellation just outside the guard.
    # Through z**10, the omitted value / gradient / Hessian terms at 0.1
    # are below 2e-22 / 2e-20 / 3e-18, respectively.
    small = jnp.abs(z) < 0.1
    zs = jnp.where(small, 1.0, z)
    # Guard the polynomial too: its inactive large-z terms must not overflow
    # and inject 0*inf into derivatives of the trigonometric branch.
    zp = jnp.where(small, z, 0.0)
    z2 = zp * zp
    series = 1.0 + z2 * (
        -1.0 / 6.0
        + z2
        * (1.0 / 120.0 + z2 * (-1.0 / 5040.0 + z2 * (1.0 / 362880.0 - z2 / 39916800.0)))
    )
    return jnp.where(small, series, jnp.sin(zs) / zs)


def params_from_native(f0, fdot, amp, phi0, colat, lon, psi, cos_iota, T, d2fdt2=None):
    """Build the GLASS UCB parameter vector from native parameters.

    GLASS samples dimensionless combinations: ``params[0] = f0*T``,
    ``params[7] = fdot*T^2``, ``params[8] = d2fdt2*T^3``, and the amplitude
    logarithmically. The sky angle is the **cosine** of the colatitude.

    Returns a **numpy** array, deliberately. Packing eight scalars with
    ``jnp`` costs eight dispatches --- measured at 0.26 ms, against 0.0007 ms
    for the numpy equivalent, and against a 0.10 ms waveform kernel. In a
    sampler that calls this once per likelihood evaluation it was two thirds
    of the total template cost, and it made the JAX generator *slower* than
    the NumPy one it replaced.

    ``jit`` and ``vmap`` accept numpy input, so this costs nothing downstream.
    If you need to build the vector from traced values, stack them yourself
    rather than routing through this helper.
    """
    p = [f0 * T, np.cos(colat), lon, np.log(amp), cos_iota, psi, phi0, fdot * T * T]
    if d2fdt2 is not None:
        p.append(d2fdt2 * T**3)
    return np.asarray(p, dtype=np.float64)


def ucb_waveform(
    params: ArrayLike,
    *,
    T: ArrayLike,
    BW: int,
    t0: ArrayLike = 0.0,
    fmt: str = "sangria",
    positions: ArrayLike | None = None,
    q: ArrayLike | None = None,
) -> WaveformResult:
    """Generate the banded TDI spectrum of one galactic binary.

    Parameters
    ----------
    params
        GLASS UCB parameter vector, length 7, 8 or 9:
        ``[f0*T, cos(theta), lambda, ln(amp), cos(iota), psi, phi0,
        fdot*T^2, d2fdt2*T^3]``. Elements 7 and 8 are optional and, when
        absent, the corresponding chirp terms are dropped exactly as
        GLASS's ``NParams > 7`` / ``> 8`` branches do.
    T
        Observation time [s]. May be traced or closed over as a constant in
        an outer JIT function; the carrier calculation preserves GLASS's
        float64 rounding in either case.
    BW
        Number of slow-time samples **and** of output frequency bins.
        Static; must be a positive even integer.
    t0
        Epoch of the first slow-time sample [s]. Traced.
    fmt
        TDI convention, see :data:`binary_galactica.glass.tdi.FORMATS`.
    positions
        Optional ``(3, BW, 3)`` precomputed spacecraft positions on this
        exact grid. The grid depends only on ``(T, t0, BW)``, so one cache
        serves every source at a given bandwidth.
    q
        Optional carrier bin, overriding GLASS's ``trunc(f0*T)``. Not needed
        for fidelity -- the default already reproduces the C -- but useful
        for pinning a band across a source population. Must be a finite
        scalar with an integral value; integral floats are accepted.

    Returns
    -------
    dict with

    ``qmin``
        absolute Fourier-bin index of the first output bin. Returned as a
        float-valued scalar, not a Python ``int``, because under ``jit`` it
        is traced; ``int(...)`` it at the boundary.
    ``freq_bin_indices``
        ``(BW,)``, the absolute bin index of every output sample. Mirrors
        the field of the same name on the generator this replaces, so a
        caller can be switched over without rewriting its bin bookkeeping.
    ``X``, ``Y``, ``Z``, ``A``, ``E``
        ``(BW,)`` complex, in GLASS's amplitude convention. See
        :func:`binary_galactica.glass.convert.to_continuous_ft` for the
        convention used by the rest of this repository.

    Notes
    -----
    Eager calls raise ``ValueError`` for nonfinite parameters, nonpositive
    ``T`` or ``f0``, cosines outside ``[-1, 1]``, nonfinite ``t0`` or cached
    positions, and invalid ``q``. Shape checks also run during tracing.
    Value checks never extract JAX tracers: under ``jit``/``vmap`` invalid
    traced domains return NaN in every output, including bin coordinates.
    Both cosine endpoints are valid; the sky-pole derivative is regularized.

    """
    _validation.require_x64()
    T, t0, positions = _validation.observation(T, BW, t0, positions)
    params = _validation.finite_array(params, "params")
    if params.ndim != 1 or params.shape[0] not in (7, 8, 9):
        raise ValueError(
            "params must be a one-dimensional array with length 7, 8, or 9; "
            f"got shape {params.shape}"
        )
    if not isinstance(params, jax.core.Tracer):
        concrete = np.asarray(params)
        if concrete[0] <= 0:
            raise ValueError("f0 must be positive")
        if np.any(np.abs(concrete[[1, 4]]) > 1):
            raise ValueError("sky and inclination cosines must lie in [-1, 1]")
        if not isinstance(T, jax.core.Tracer):
            with np.errstate(over="ignore", under="ignore"):
                f0 = concrete[0] / np.asarray(T)
            if not np.isfinite(f0) or f0 <= 0:
                raise ValueError("f0 must be finite and positive")
    if q is not None:
        q = _validation.scalar(q, "q")
        if not isinstance(q, jax.core.Tracer) and np.asarray(q) != np.trunc(q):
            raise ValueError("q must be an integral Fourier-bin index")
    return _ucb_waveform(
        params, T=T, BW=int(BW), t0=t0, fmt=fmt, positions=positions, q=q
    )


@partial(jax.jit, static_argnames=("BW", "fmt"))
def _ucb_waveform(
    params, *, T, BW: int, t0=0.0, fmt: str = "sangria", positions=None, q=None
) -> WaveformResult:
    """Evaluate validated inputs and preserve GLASS's discrete carrier rule."""
    n_params = params.shape[0]

    f0 = params[0] / T
    costh = params[1]
    phi = params[2]
    amp = jnp.exp(params[3])
    cosi = params[4]
    psi = params[5]
    phi0 = params[6]
    # GLASS: dfdt = params[7]/(T*T) only when NParams>7, else 0.
    fdot = params[7] / (T * T) if n_params > 7 else jnp.array(0.0)
    d2fdt2 = params[8] / (T**3) if n_params > 8 else jnp.array(0.0)

    # Carrier bin. GLASS: q = (long)(f0*T) -- a C cast, which TRUNCATES.
    # It does not round. The residual heterodyne (2 pi (f0 T - q)) therefore
    # spans a full turn over the record rather than half of one.
    #
    # Truncate the ROUND-TRIPPED product, exactly as GLASS does: it has
    # already formed f0 = params[0]/T, and (params[0]/T)*T is NOT params[0]
    # in float64 -- at T = 7884000 s, 5620 of the 39000 integer values of
    # params[0] come back just below the integer, and truncating params[0]
    # directly would put the whole band one bin high for every one of them.
    #
    # A constant T in an outer JIT can turn p/T into p*(1/T). Barriers
    # retain the two rounded operations even when both inputs are constants.
    # Only the discrete carrier uses barriers: its derivatives are zero,
    # and stopping gradients before the barrier avoids unsupported AD rules.
    q = (
        _carrier_bin(jax.lax.stop_gradient(params[0]), jax.lax.stop_gradient(T))
        if q is None
        else jnp.asarray(q, jnp.float64)
    )
    valid = (
        jnp.all(jnp.isfinite(params))
        & jnp.isfinite(T)
        & (T > 0)
        & jnp.isfinite(f0)
        & (f0 > 0)
        & jnp.isfinite(t0)
        & (jnp.abs(costh) <= 1)
        & (jnp.abs(cosi) <= 1)
        & jnp.isfinite(q)
        & (q == jnp.trunc(q))
    )

    # --- geometry -------------------------------------------------------
    t = slow_time_grid(T, t0, BW)
    if positions is None:
        positions = analytic_orbits(t)
    else:
        valid = valid & jnp.all(jnp.isfinite(positions))
    arms = arm_vectors(positions)
    e_plus, e_cross, k = polarization_tensor(costh, phi)
    d_plus, d_cross, kdotr = detector_tensor(arms, k, e_plus, e_cross)

    # --- polarization amplitudes ----------------------------------------
    # Current GLASS keeps the explicit halves in both inclination factors:
    # Aplus = amp*(1+cos^2(i))/2 and Across = -amp*cos(i).
    Aplus = amp * (1.0 + cosi * cosi) * 0.5
    Across = -amp * cosi
    cosps = jnp.cos(2.0 * psi)
    sinps = jnp.sin(2.0 * psi)
    DP = Aplus * cosps - 1j * Across * sinps
    DC = -Aplus * sinps - 1j * Across * cosps

    # --- per-spacecraft retarded quantities -----------------------------
    kdotx = jnp.einsum("sta,a->st", positions, k) / CLIGHT  # (3, BW)
    xi = t[None, :] - kdotx
    f_inst = f0 + fdot * xi + 0.5 * d2fdt2 * xi * xi
    fonfs = f_inst / FSTAR

    # GLASS writes PI2*(((double)q)/T) -- divide first. Python would
    # associate PI2*q/T as (PI2*q)/T, which rounds differently.
    df = PI2 * (q / T)
    arg2 = (
        PI2 * f0 * xi
        - phi0
        - df * t[None, :]
        + jnp.pi * fdot * xi * xi
        + (jnp.pi / 3.0) * d2fdt2 * xi * xi * xi
    )
    # First-order amplitude evolution; GLASS guards it behind NParams>7.
    aevol = 1.0 + (2.0 / 3.0) * (fdot / f0) * xi if n_params > 7 else jnp.ones_like(xi)

    # --- six single links ------------------------------------------------
    # A single (6, BW) expression lets XLA fuse the link calculations into
    # one vectorized body while preserving GLASS's directed-link order.
    sc = jnp.array([sc for sc, _, _ in LINKS])
    pair = jnp.array([pair for _, pair, _ in LINKS])
    sign = jnp.array([sign for _, _, sign in LINKS])[:, None]
    arg1 = 0.5 * fonfs[sc] * (1.0 + sign * kdotr[pair])
    tran1 = aevol[sc] * (d_plus[pair] * DP + d_cross[pair] * DC)
    env = 0.25 * _sinc(arg1) * tran1 * jnp.exp(-1j * (arg1 - arg2[sc]))

    # --- reduced DFT ------------------------------------------------------
    # GLASS FFTs the BW-sample complex envelope (kiss_fft, unnormalised),
    # then swaps the two halves of the output. The half-swap is an fftshift:
    # it puts the carrier bin at index BW/2.
    #
    # Current GLASS scales the unnormalised complex FFT by invBW2=1/(2*BW).
    # The half-polarization factors above are retained independently.
    d = jnp.fft.fftshift(jnp.fft.fft(env, axis=-1), axes=-1) / (2 * BW)

    # --- TDI --------------------------------------------------------------
    m = jnp.arange(BW, dtype=jnp.float64)
    qmin = q - BW // 2
    bins = qmin + m
    X, Y, Z, A, E = synthesize(d, bins / T, T, fmt)

    result = {
        "qmin": qmin,
        "freq_bin_indices": bins,
        "X": X,
        "Y": Y,
        "Z": Z,
        "A": A,
        "E": E,
    }
    return {name: jnp.where(valid, value, jnp.nan) for name, value in result.items()}


def orbit_cache(T: ArrayLike, t0: ArrayLike, BW: int) -> Array:
    """Spacecraft positions on the ``(T, t0, BW)`` slow grid.

    Hoist this out of a sampler loop: it depends on no source parameter.
    """
    _validation.require_x64()
    return analytic_orbits(slow_time_grid(T, t0, BW))


__all__ = ["orbit_cache", "params_from_native", "ucb_waveform"]
