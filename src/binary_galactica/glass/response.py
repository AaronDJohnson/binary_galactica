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
"""Polarisation and detector tensors, from GLASS ``glass_lisa.c``.

Two conventions for the polarisation tensors live in GLASS:

``LISA_polarization_tensor``
    ``e+ = v v - u u``, the LDC convention, and the one
    ``ucb_waveform`` actually calls.
``LISA_polarization_tensor_njc``
    ``e+ = u u - v v``, i.e. the opposite overall sign, unused by the UCB
    path.

The sign difference is degenerate with ``phi0 -> phi0 + pi``, which is why
it can sit in the source unnoticed. We implement the LDC one and name it.
"""

from __future__ import annotations

import jax.numpy as jnp


def polarization_tensor(costh, phi):
    """GW basis tensors in the LDC convention (``LISA_polarization_tensor``).

    Parameters
    ----------
    costh
        Cosine of the ecliptic colatitude.
    phi
        Ecliptic longitude [rad].

    Returns
    -------
    ``(e_plus, e_cross, k_hat)`` with tensors ``(3, 3)`` and ``k_hat``
    ``(3,)``. ``k_hat`` is the wave **propagation** direction.

    """
    # sqrt(1 - costh^2) is exact in C, where nothing differentiates it. Under
    # AD its derivative is -costh/sqrt(1-costh^2), which at the ecliptic poles
    # (costh = +-1) is -1/0. Forward mode then propagates 0/0 = NaN into
    # EVERY tangent direction, so jacfwd and hessian come back all-NaN even
    # for parameters like phi0 whose derivative is perfectly well defined;
    # reverse mode gives +inf in the sky component and NaN in a Fisher
    # matrix built from it. The forward value is finite and correct, so no
    # forward-only check sees any of this.
    #
    # The poles are reachable, not hypothetical: in float64 every colatitude
    # within ~1e-8 rad of 0 or pi has cos(colat) == 1.0 exactly, and costh
    # = +-1 is the boundary of the standard cos-colatitude prior.
    #
    # Guarded the same way as _sinc in waveform.py: the inner `where` keeps
    # the square root away from its branch point so the derivative is never
    # formed there. sinth is exact (0) at the pole; its gradient is reported
    # as 0 rather than the true infinity, which is a deliberate truncation of
    # a measure-zero singularity. Values outside the cosine domain are not
    # poles: preserve NaN there rather than turning invalid sky coordinates
    # into plausible finite responses.
    c2 = costh * costh
    below = c2 < 1.0
    sinth = jnp.where(below, jnp.sqrt(1.0 - jnp.where(below, c2, 0.0)), 0.0)
    sinth = jnp.where(c2 <= 1.0, sinth, jnp.nan)
    cosph = jnp.cos(phi)
    sinph = jnp.sin(phi)

    u = jnp.stack([costh * cosph, costh * sinph, -sinth])
    v = jnp.stack([sinph, -cosph, jnp.zeros_like(sinph)])
    k = jnp.stack([-sinth * cosph, -sinth * sinph, -costh])

    e_plus = jnp.outer(v, v) - jnp.outer(u, u)
    e_cross = jnp.outer(u, v) + jnp.outer(v, u)
    return e_plus, e_cross, k


def detector_tensor(arms, k, e_plus, e_cross):
    """Arm projections and Doppler factors (``LISA_detector_tensor``).

    Parameters
    ----------
    arms
        ``(3, n, 3)`` directed separation vectors in pair order
        ``(12, 13, 23)``, already divided by the arm length.
    k, e_plus, e_cross
        From :func:`polarization_tensor`.

    Returns
    -------
    ``(d_plus, d_cross, kdotr)``, each ``(3, n)`` in pair order
    ``(12, 13, 23)``.

    GLASS stores the full 3x3 tables and fills the reversed entries by
    symmetry: ``dplus[j][i] = dplus[i][j]`` (the contraction is quadratic
    in the arm vector) and ``kdotr[j][i] = -kdotr[i][j]``. We keep only the
    three independent pairs and apply the sign at the call site, which is
    the same thing with a third of the arithmetic.

    """
    # These three-component contractions are faster as scalar expressions:
    # XLA can fuse them with the link response instead of materializing the
    # small matrix products selected by einsum, especially under vmap.
    x, y, z = arms[..., 0], arms[..., 1], arms[..., 2]

    def contract(e):
        return (
            x * (x * e[0, 0] + y * e[1, 0] + z * e[2, 0])
            + y * (x * e[0, 1] + y * e[1, 1] + z * e[2, 1])
            + z * (x * e[0, 2] + y * e[1, 2] + z * e[2, 2])
        )

    d_plus = contract(e_plus)
    d_cross = contract(e_cross)
    kdotr = x * k[0] + y * k[1] + z * k[2]
    return d_plus, d_cross, kdotr


#: Directed links in GLASS's ``(i, j)`` order, as
#: ``(spacecraft_index, pair_index, kdotr_sign)``.
#:
#: ``i`` is the spacecraft whose retarded time and instantaneous frequency
#: the link uses; ``pair_index`` selects the unordered arm from
#: ``(12, 13, 23)``; the sign flips ``k.r`` for the reversed direction,
#: which is GLASS's ``kdotr[j][i] = -kdotr[i][j]``.
LINKS = (
    (0, 0, +1.0),  # 1 -> 2
    (0, 1, +1.0),  # 1 -> 3
    (1, 0, -1.0),  # 2 -> 1
    (1, 2, +1.0),  # 2 -> 3
    (2, 1, -1.0),  # 3 -> 1
    (2, 2, -1.0),  # 3 -> 2
)

#: Index of each directed link within :data:`LINKS`, by GLASS's ``ij`` name.
LINK_INDEX = {"12": 0, "13": 1, "21": 2, "23": 3, "31": 4, "32": 5}

__all__ = ["LINKS", "LINK_INDEX", "detector_tensor", "polarization_tensor"]
