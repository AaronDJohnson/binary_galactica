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
r"""TDI synthesis, from GLASS ``LISA_tdi*`` in ``glass/utils/src/glass_lisa.c``.

GLASS carries three TDI conventions, selected by a string ``format``
argument to ``ucb_waveform``. All three form the *same* combination,

.. math::
    X = P\,\big[(d_{12}-d_{13})e^{-2ix} + (d_{21}-d_{31})e^{-ix}\big],

with ``Y``, ``Z`` its ``1->2->3`` cyclic images, and differ only in ``P``:

=============  ===================  =============================  ===========
``format``     GLASS function       :math:`P`                      A/E conj.
=============  ===================  =============================  ===========
``"phase"``    ``LISA_tdi``         :math:`-2i\sin(x)\sqrt{T}`      **yes**
``"frequency"````LISA_tdi_FF``      :math:`-4ix\sin(x)\sqrt{T}`     no
``"sangria"``  ``LISA_tdi_Sangria`` :math:`4x\sin(x)\sqrt{T}`       no
=============  ===================  =============================  ===========

``LISA_tdi`` and ``LISA_tdi_FF`` spell the combination out as four terms in
:math:`e^{-3ix}, e^{-2ix}, e^{-ix}, 1`; factoring
:math:`e^{-ix}-e^{+ix} = -2i\sin x` out of both pairs recovers the two-term
form above, which is what ``LISA_tdi_Sangria`` writes directly.

Here :math:`x = f/f_*` is evaluated at **each output bin's own frequency**,
:math:`f = (q + m - BW/2)/T`, exactly as in current GLASS.

Two traps preserved verbatim:

* ``LISA_tdi`` (and only it) negates the imaginary part of ``A`` and ``E``
  after forming them --- ``A[k] = -A[k]; E[k] = -E[k];``. ``X``, ``Y``,
  ``Z`` are left alone, so in that convention ``A != (Z-X)/sqrt2`` as
  complex numbers.
* the comment above ``LISA_tdi_FF``'s prefactor reads
  ``sqrt(T)*4*(f/fstar)*sin(f/fstar)`` while the code is ``sqT*fonfs2``
  with ``fonfs2 = 2*fonfs``. That is not a stale comment: the C's literal
  ``prefactor`` variable multiplies the *four-term* form, and once the
  ``-2i sin x`` is factored out of it the effective prefactor really is
  ``4 x sin(x) sqrt(T)`` in modulus. The comment describes ``|P|``, not the
  variable beside it.

GLASS also computes ``c1,s1`` by a recursion in bin index rather than
calling trig per bin (``recursive_phase_evolution`` / ``double_angle`` /
``triple_angle``). That is an optimisation for a scalar C loop with no
effect on the result; a vectorised ``exp`` is both faster here and better
conditioned, so we do not reproduce the recursion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import FSTAR, INV_SQ3, INV_SQ6, INV_SQRT2
from .response import LINK_INDEX

jax.config.update("jax_enable_x64", True)

FORMATS = ("sangria", "frequency", "phase")


def _links(d):
    idx = LINK_INDEX
    return (
        d[idx["12"]],
        d[idx["13"]],
        d[idx["21"]],
        d[idx["23"]],
        d[idx["31"]],
        d[idx["32"]],
    )


def synthesize(d, f, T: float, fmt: str = "sangria"):
    r"""Combine the six single-link spectra into TDI ``X, Y, Z, A, E``.

    Parameters
    ----------
    d
        ``(6, BW)`` complex single-link spectra in link order
        ``(12, 13, 21, 23, 31, 32)``.
    f
        ``(BW,)`` frequency of each output bin [Hz].
    T
        Observation time [s]; enters only as the :math:`\sqrt{T}` in every
        prefactor.
    fmt
        One of :data:`FORMATS`.

    Returns
    -------
    ``(X, Y, Z, A, E)``, each ``(BW,)`` complex.

    """
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}; got {fmt!r}")

    d12, d13, d21, d23, d31, d32 = _links(d)
    x = f / FSTAR
    sqT = jnp.sqrt(T)

    e1 = jnp.exp(-1j * x)
    e2 = e1 * e1

    # All three variants reduce to one structure. ``LISA_tdi`` and
    # ``LISA_tdi_FF`` write four terms,
    #     (a)e^{-3ix} + (b)e^{-2ix} - (a)e^{-ix} - (b),
    # and factoring e^{-ix} - e^{+ix} = -2i sin x out of both pairs gives
    # the same two-term form ``LISA_tdi_Sangria`` writes directly. So the
    # variants differ *only* in the prefactor below, which is worth knowing:
    # they are the same TDI combination in three unit conventions, not
    # three different combinations.
    pre = {
        "sangria": 4.0 * x * jnp.sin(x) * sqT,
        "phase": -2.0j * jnp.sin(x) * sqT,
        "frequency": -4.0j * x * jnp.sin(x) * sqT,
    }[fmt]

    X = pre * ((d12 - d13) * e2 + (d21 - d31) * e1)
    Y = pre * ((d23 - d21) * e2 + (d32 - d12) * e1)
    Z = pre * ((d31 - d32) * e2 + (d13 - d23) * e1)

    A, E = xyz_to_ae(X, Y, Z)
    if fmt == "phase":
        # LISA_tdi negates the imaginary part of A and E only.
        A = jnp.conj(A)
        E = jnp.conj(E)
    return X, Y, Z, A, E


def xyz_to_ae(X, Y, Z):
    """``XYZ2AE``: ``A = (Z-X)/sqrt2``, ``E = (X-2Y+Z)/sqrt6``.

    Both are *multiplications* by a stored reciprocal, as in the C
    (``M_SQRT1_2`` and the literal ``invSQ6``). Dividing by ``sqrt(2)``
    instead would differ in the last bit.
    """
    return (Z - X) * INV_SQRT2, (X - 2.0 * Y + Z) * INV_SQ6


def xyz_to_aet(X, Y, Z):
    """``XYZ2AET``. Note GLASS divides ``T`` by ``sqrt(3)``, not by 3.

    This choice makes ``T`` share the normalization of ``A`` and ``E``.
    """
    A, E = xyz_to_ae(X, Y, Z)
    return A, E, (X + Y + Z) * INV_SQ3


__all__ = ["FORMATS", "synthesize", "xyz_to_ae", "xyz_to_aet"]
