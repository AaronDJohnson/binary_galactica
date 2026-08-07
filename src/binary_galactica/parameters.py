"""The eight-dimensional galactic-binary parameter space.

The parameters are carried in a fixed order,

    (amp, f0, fdot, phi0, iota, psi, lam, beta),

so that a single ``(8,)`` array is an unambiguous point in the sampling space.
:class:`GBParams` names those slots without changing the representation: it is
a :class:`~typing.NamedTuple`, hence a JAX pytree, so it may be passed through
``jit``, ``vmap`` and ``grad`` exactly like the bare array.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import jax.numpy as jnp
from jax import Array

#: Parameter names, in the canonical order used by every array in this package.
PARAMETER_NAMES: tuple[str, ...] = (
    "amp",
    "f0",
    "fdot",
    "phi0",
    "iota",
    "psi",
    "lam",
    "beta",
)

#: Dimension of the parameter space.
NUM_PARAMS: int = len(PARAMETER_NAMES)

Scalar = Union[Array, float]


class GBParams(NamedTuple):
    """Parameters of a galactic binary.

    Every parameter is defined at ``t = 0`` of the time array passed to the
    waveform functions; there is no hidden reference epoch.

    Attributes:
        amp: Dimensionless strain amplitude :math:`\\mathcal{A}`.
        f0: Gravitational-wave frequency at ``t = 0``, in Hz.  This is twice
            the orbital frequency of the binary.
        fdot: First frequency derivative :math:`\\dot{f}`, in Hz/s.
        phi0: Gravitational-wave phase at ``t = 0``, in radians.
        iota: Inclination of the orbital angular momentum to the line of
            sight, in radians.
        psi: Polarization angle, in radians.  Periodic with period
            :math:`\\pi`.
        lam: Ecliptic longitude, in radians.
        beta: Ecliptic latitude, in radians.

    Note:
        ``lam`` and ``beta`` do not affect the polarizations :math:`h_+` and
        :math:`h_\\times`.  Sky position enters only through the detector
        response -- the propagation direction and polarization basis -- which
        is deliberately outside the scope of this package.  They are carried
        here because they are part of the sampling space and are needed by the
        response code downstream.

    Example:
        >>> from binary_galactica import GBParams
        >>> p = GBParams(amp=1e-22, f0=1e-3, fdot=1e-17, phi0=0.3,
        ...              iota=0.7, psi=1.2, lam=4.1, beta=-0.3)
        >>> p.to_array().shape
        (8,)
    """

    amp: Scalar
    f0: Scalar
    fdot: Scalar
    phi0: Scalar
    iota: Scalar
    psi: Scalar
    lam: Scalar
    beta: Scalar

    @classmethod
    def from_array(cls, theta: Array | GBParams) -> GBParams:
        """Interpret a ``(..., 8)`` array as parameters.

        Passing a :class:`GBParams` returns it unchanged, so waveform
        functions accept either representation.

        Args:
            theta: Array whose trailing axis indexes :data:`PARAMETER_NAMES`,
                or an existing :class:`GBParams`.

        Returns:
            The named parameters.  Leading axes of ``theta`` are preserved on
            each field.

        Raises:
            ValueError: If the trailing axis of ``theta`` is not length 8.
        """
        if isinstance(theta, cls):
            return theta
        theta = jnp.asarray(theta)
        if theta.ndim == 0 or theta.shape[-1] != NUM_PARAMS:
            raise ValueError(
                f"expected trailing axis of length {NUM_PARAMS} "
                f"({', '.join(PARAMETER_NAMES)}), got shape {theta.shape}"
            )
        return cls(*jnp.moveaxis(theta, -1, 0))

    def to_array(self) -> Array:
        """Stack the parameters into a ``(..., 8)`` array.

        Fields are broadcast against one another first, so scalars may be
        mixed with arrays.

        Returns:
            Array whose trailing axis indexes :data:`PARAMETER_NAMES`.
        """
        fields = jnp.broadcast_arrays(*(jnp.asarray(x) for x in self))
        return jnp.stack(fields, axis=-1)


__all__ = ["NUM_PARAMS", "PARAMETER_NAMES", "GBParams"]
