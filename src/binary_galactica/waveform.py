r"""The galactic-binary waveform model.

A galactic binary is a slowly inspiralling compact pair whose orbit, over a
LISA observation, is circular to excellent approximation and whose frequency
evolves so little that a linear chirp suffices.  The emitted wave is therefore
monochromatic with a quadratic phase,

.. math::

    \Phi(t) = \phi_0 + 2\pi f_0 t + \pi \dot{f} t^2,
    \qquad f(t) = f_0 + \dot{f} t,

and constant polarization amplitudes set by the inclination,

.. math::

    \mathcal{A}_+ = \mathcal{A}\,(1 + \cos^2\iota),
    \qquad
    \mathcal{A}_\times = -2\mathcal{A}\cos\iota.

In the frame of the source's own orbital angular momentum -- equivalently, at
polarization angle :math:`\psi = 0` -- the two polarizations are

.. math::

    h_+^{\rm src}(t) = \mathcal{A}_+ \cos\Phi(t),
    \qquad
    h_\times^{\rm src}(t) = \mathcal{A}_\times \sin\Phi(t),

and a rotation by the polarization angle carries them into the solar-system
barycentre frame,

.. math::

    \begin{pmatrix} h_+ \\ h_\times \end{pmatrix}
    =
    \begin{pmatrix}
        \cos 2\psi & \sin 2\psi \\
       -\sin 2\psi & \cos 2\psi
    \end{pmatrix}
    \begin{pmatrix} h_+^{\rm src} \\ h_\times^{\rm src} \end{pmatrix}.

Sky position does not appear in these source polarizations. It enters through
the detector response in :func:`binary_galactica.tdi_waveform`.

Conventions
-----------
The source polarization algebra can be expressed using complex coefficients

.. code-block:: text

    DPr =  A+ cos2psi        DPi = -Ax sin2psi
    DCr = -A+ sin2psi        DCi = -Ax cos2psi

contracted against a wave written as :math:`e^{+i\Phi}`, so that the factor
multiplying the plus (cross) detector tensor is ``DPr cos(Phi) - DPi sin(Phi)``
(``DCr cos(Phi) - DCi sin(Phi)``).  Those are exactly the expressions returned
by :func:`polarizations`; ``tests/test_conventions.py`` checks this algebra.

These source helpers use full inclination factors and a positive initial
phase. Current GLASS uses half inclination factors, the opposite initial-phase
parameter, and first-order amplitude evolution. Use
:func:`binary_galactica.to_glass_vector` to translate parameters, or
:func:`binary_galactica.tdi_waveform` for the full detector signal. Numerical
agreement of the backend with GLASS is checked separately against C-generated
reference spectra.

References
----------
Cornish & Littenberg, *Tests of Bayesian model selection techniques for
gravitational wave astronomy*, Phys. Rev. D **76**, 083006 (2007).

Littenberg, *A detection pipeline for galactic binaries in LISA data*,
Phys. Rev. D **84**, 063009 (2011).

"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from binary_galactica.glass import _validation
from binary_galactica.parameters import GBParams

__all__ = ["amplitudes", "frequency", "phase", "polarizations"]


def _parameters(theta: Array | GBParams) -> GBParams:
    """Require float64 and normalize ``theta`` to :class:`GBParams`."""
    _validation.require_x64()
    return GBParams.from_array(theta)


def phase(theta: Array | GBParams, t: Array) -> Array:
    r"""Gravitational-wave phase :math:`\Phi(t)`.

    Args:
        theta: Parameters, as a ``(8,)`` array or :class:`GBParams`.
        t: Times in seconds, measured from the epoch at which ``theta`` is
           defined.

    Returns:
        Phase in radians, with the shape of ``t``.

    """
    p = _parameters(theta)
    return p.phi0 + 2.0 * jnp.pi * p.f0 * t + jnp.pi * p.fdot * t**2


def frequency(theta: Array | GBParams, t: Array) -> Array:
    r"""Instantaneous gravitational-wave frequency :math:`f(t)`.

    This is :math:`\Phi'(t) / 2\pi` -- the derivative of :func:`phase`.

    Args:
        theta: Parameters, as a ``(8,)`` array or :class:`GBParams`.
        t: Times in seconds.

    Returns:
        Frequency in Hz, with the shape of ``t``.

    """
    p = _parameters(theta)
    return p.f0 + p.fdot * t


def amplitudes(theta: Array | GBParams) -> tuple[Array, Array]:
    r"""Constant polarization amplitudes :math:`(\mathcal{A}_+, \mathcal{A}_\times)`.

    Args:
        theta: Parameters, as a ``(8,)`` array or :class:`GBParams`.

    Returns:
        The plus and cross amplitudes.  Both are constant in time; the binary
        is circular and does not evolve appreciably over the observation.

    """
    p = _parameters(theta)
    cos_iota = jnp.cos(p.iota)
    return p.amp * (1.0 + cos_iota**2), -2.0 * p.amp * cos_iota


def polarizations(theta: Array | GBParams, t: Array) -> tuple[Array, Array]:
    r"""Gravitational-wave polarizations :math:`(h_+, h_\times)` at the barycentre.

    This is the source-only model. See :func:`binary_galactica.tdi_waveform`
    for GLASS's detector waveform, including amplitude evolution and TDI.

    Args:
        theta: Parameters, as a ``(8,)`` array or :class:`GBParams`.  The
            ``lam`` and ``beta`` entries are unused -- sky position affects the
            detector response, not the polarizations -- so the corresponding
            columns of the Jacobian are identically zero.
        t: Times in seconds, measured from the epoch at which ``theta`` is
            defined.

    Returns:
        ``(hp, hc)``, each with the shape of ``t``.

    Example:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> from binary_galactica import DAY, GBParams, polarizations
        >>> p = GBParams(amp=1e-22, f0=1e-3, fdot=1e-17, phi0=0.3,
        ...              iota=0.7, psi=1.2, lam=4.1, beta=-0.3)
        >>> t = jnp.arange(0.0, DAY, 15.0)
        >>> hp, hc = polarizations(p, t)
        >>> hp.shape == t.shape
        True

        Catalogues are handled by ``vmap`` over the leading axis of a
        ``(K, 8)`` array of parameters::

            from jax import vmap
            hp, hc = vmap(polarizations, in_axes=(0, None))(catalogue, t)

    """
    p = _parameters(theta)
    a_plus, a_cross = amplitudes(p)
    phi = phase(p, t)

    # Source frame: the polarization basis aligned with the orbital angular
    # momentum, i.e. psi = 0.
    hp_src = a_plus * jnp.cos(phi)
    hc_src = a_cross * jnp.sin(phi)

    # Rotate by the polarization angle into the barycentre frame.  The wave is
    # spin-2, so the basis rotates at twice the angle and psi has period pi.
    cos_2psi = jnp.cos(2.0 * p.psi)
    sin_2psi = jnp.sin(2.0 * p.psi)
    hp = hp_src * cos_2psi + hc_src * sin_2psi
    hc = -hp_src * sin_2psi + hc_src * cos_2psi
    return hp, hc
