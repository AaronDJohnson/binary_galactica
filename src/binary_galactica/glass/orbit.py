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
r"""LISA orbit, transcribed from GLASS ``analytic_orbits``.

``glass/utils/src/glass_lisa.c``, function ``analytic_orbits``. Each
spacecraft is expanded about its own guiding centre at phase
:math:`\beta_i = 2\pi i/3`, to first order in the constellation
eccentricity :math:`e = L/(2\sqrt{3}\,\mathrm{AU})`.

Three details are transcribed rather than tidied, because tidying them
changes the numbers:

* ``kappa_0`` is **not** applied. GLASS writes ``alpha = PI2*t/YEAR; //
  + kappa_0`` --- the offset is commented out, so the constellation phase
  at ``t = 0`` is fixed. We keep it fixed too, and expose ``kappa_0`` only
  as an explicit opt-in that GLASS itself never exercises.
* the :math:`\sin\beta_i` literals are asymmetric in their last digit
  (``0.866025403784439`` for SC2 against ``-0.866025403784438`` for SC3).
  Both are copied verbatim.
* ``SQ3`` is GLASS's truncated ``1.73205080757``, not ``sqrt(3)``.

The arm vectors are formed by differencing positions and dividing by the
*nominal* arm length ``L``, exactly as ``LISA_detector_tensor`` does, so
they are not quite unit vectors --- the constellation is only equilateral
to :math:`O(e^2)`. That flexing is physical and GLASS keeps it; do not
normalise it away.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import AU, LARM, PI2, RORB, SQ3, YEAR

jax.config.update("jax_enable_x64", True)

# sin/cos of beta_i = 2 pi i / 3, copied verbatim from glass_lisa.c.
_SB = (0.0, 0.866025403784439, -0.866025403784438)
_CB = (1.0, -0.5, -0.5)


def analytic_orbits(t, *, kappa_0: float = 0.0):
    """Spacecraft positions on the GLASS analytic (rigid Keplerian) orbit.

    Parameters
    ----------
    t
        Times [s], shape ``(n,)``.
    kappa_0
        Guiding-centre phase at ``t = 0`` [rad]. GLASS has this commented
        out and therefore always ``0``; a non-zero value is a deliberate
        departure from GLASS. Unlike the arm length it is a pure rotation,
        consistent on its own.

    Returns
    -------
    ``(3, n, 3)`` array indexed ``[spacecraft, time, xyz]``, in metres.

    Notes
    -----
    The arm length is deliberately **not** a parameter. GLASS derives ``L``,
    ``R = AU*e`` and ``fstar = c/(2 pi L)`` together in
    ``initialize_analytic_orbit``, so they can never disagree. Exposing
    ``arm_length`` on this one function would let a caller rescale the
    constellation while :func:`arm_vectors` still divided by the old ``L``
    and :mod:`binary_galactica.glass.tdi` still used the old ``fstar`` ---
    a silently wrong waveform. Change
    :data:`binary_galactica.glass.constants.LARM` instead; ``RORB`` and
    ``FSTAR`` derive from it.

    """
    t = jnp.asarray(t, dtype=jnp.float64)
    R = RORB

    alpha = PI2 * t / YEAR + kappa_0
    sa = jnp.sin(alpha)
    ca = jnp.cos(alpha)
    sa2, ca2, saca = sa * sa, ca * ca, sa * ca
    AUca, AUsa = AU * ca, AU * sa

    out = []
    for sb, cb in zip(_SB, _CB, strict=True):
        x = AUca + R * (saca * sb - (1.0 + sa2) * cb)
        y = AUsa + R * (saca * cb - (1.0 + ca2) * sb)
        z = -SQ3 * R * (ca * cb + sa * sb)
        out.append(jnp.stack([x, y, z], axis=-1))
    return jnp.stack(out, axis=0)


def arm_vectors(positions):
    """Directed separation vectors, GLASS ``LISA_detector_tensor`` style.

    ``r_12 = (P_2 - P_1)/L``, ``r_13 = (P_3 - P_1)/L``,
    ``r_23 = (P_3 - P_2)/L`` --- divided by the nominal
    :data:`binary_galactica.glass.constants.LARM`, so ``|r| != 1`` at the
    ``O(e^2)`` level.
    That residual is the constellation flexing; it is physical and GLASS
    keeps it, so do not normalise it away.

    Returns ``(3, n, 3)`` indexed ``[pair, time, xyz]`` with pair order
    ``(12, 13, 23)``.
    """
    P = positions
    return jnp.stack(
        [(P[1] - P[0]) / LARM, (P[2] - P[0]) / LARM, (P[2] - P[1]) / LARM],
        axis=0,
    )


def slow_time_grid(T: float, t0: float, BW: int):
    """Return the GLASS slow-time grid for ``n = 0..BW-1``.

    GLASS writes ``t = t0 + T*(double)(n-1)/(double)BW`` over ``n = 1..BW``
    (``glass_ucb_waveform.c``), which is this grid. Note the comment there:
    *"First time sample must be at t=0 for phasing"* --- the grid starts at
    ``t0`` and does **not** include ``t0 + T``.
    """
    return t0 + T * jnp.arange(BW, dtype=jnp.float64) / BW


__all__ = ["analytic_orbits", "arm_vectors", "slow_time_grid"]
