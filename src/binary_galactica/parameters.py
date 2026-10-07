"""The eight-dimensional physical galactic-binary parameter space.

The parameters are carried in a fixed order,

    (amp, f0, fdot, phi0, iota, psi, lam, beta),

for the physical waveform interfaces. GLASS-native arrays use a different
ordering despite having the same shape; use :func:`to_glass_vector` and
:func:`from_glass_vector` to convert between them. :class:`GBParams` names
the physical slots without changing the representation: it is
a :class:`~typing.NamedTuple`, hence a JAX pytree, so it may be passed through
``jit``, ``vmap`` and ``grad`` exactly like the bare array.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from binary_galactica.glass._precision import real_dtype

#: Parameter names, in the canonical order used by physical parameter arrays.
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

ParameterField = Array | float


class GBParams(NamedTuple):
    r"""Parameters of a galactic binary.

    Every parameter is defined at ``t = 0`` of the time array passed to the
    waveform functions; there is no hidden reference epoch.

    Attributes:
        amp: Dimensionless strain amplitude :math:`\mathcal{A}`.
        f0: Gravitational-wave frequency at ``t = 0``, in Hz.  This is twice
            the orbital frequency of the binary.
        fdot: First frequency derivative :math:`\dot{f}`, in Hz/s.
        phi0: Gravitational-wave phase at ``t = 0``, in radians.
        iota: Inclination of the orbital angular momentum to the line of
            sight, in radians.
        psi: Polarization angle, in radians.  Periodic with period
            :math:`\pi`.
        lam: Ecliptic longitude, in radians.
        beta: Ecliptic latitude, in radians.

    Note:
        ``lam`` and ``beta`` do not affect the polarizations :math:`h_+` and
        :math:`h_\times`.  Sky position enters only through the detector
        response -- the propagation direction and polarization basis. Use
        :func:`binary_galactica.tdi_waveform` to include that response.

    Example:
        >>> from binary_galactica import GBParams
        >>> p = GBParams(amp=1e-22, f0=1e-3, fdot=1e-17, phi0=0.3,
        ...              iota=0.7, psi=1.2, lam=4.1, beta=-0.3)
        >>> p.to_array().shape
        (8,)

    """

    amp: ParameterField
    f0: ParameterField
    fdot: ParameterField
    phi0: ParameterField
    iota: ParameterField
    psi: ParameterField
    lam: ParameterField
    beta: ParameterField

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


def to_glass_vector(theta: Array | GBParams, duration: Array | float) -> Array:
    """Convert physical parameters to the eight-parameter GLASS convention.

    The last axis is ``(f0*T, sin(beta), lam, log(A_G), cos(iota), psi,
    phi0_G, fdot*T**2)``. GLASS's half inclination factors require
    ``A_G = 2*amp`` to match this package's source polarization amplitudes
    at the reference epoch. Its phase function uses ``-phi0_G``, so
    ``phi0_G = -phi0``. Both models define their parameters at time zero.

    GLASS additionally evolves the amplitude with frequency; converting
    parameters does not add that evolution to :func:`polarizations`.

    Args:
        theta: Physical ``GBParams`` or an array with trailing dimension 8.
        duration: Observation duration in seconds, positive and broadcastable
            with the parameter fields. Supports traced values under JIT.

    Returns:
        A JAX array with trailing dimension 8, in the floating dtype of
        ``theta``; ``duration`` is cast to it. All operations support AD.
        The physical amplitude must be positive for its logarithm to exist.

    """
    p = GBParams.from_array(theta)
    real = real_dtype(*p)
    p = GBParams(*(jnp.asarray(field, dtype=real) for field in p))
    duration = jnp.asarray(duration, dtype=real)
    fields = (
        p.f0 * duration,
        jnp.sin(p.beta),
        p.lam,
        jnp.log(2.0 * p.amp),
        jnp.cos(p.iota),
        p.psi,
        -p.phi0,
        p.fdot * duration**2,
    )
    return jnp.stack(jnp.broadcast_arrays(*fields), axis=-1)


def from_glass_vector(vector: Array, duration: Array | float) -> GBParams:
    """Convert an eight-parameter GLASS vector to physical parameters.

    Inclination is returned in ``[0, pi]`` and latitude in ``[-pi/2, pi/2]``;
    the cosine coordinates must lie in ``[-1, 1]``. Amplitude and initial
    phase invert :func:`to_glass_vector`. Other angles are not wrapped.

    Args:
        vector: GLASS-native JAX/NumPy array with trailing dimension 8.
            Seven- and nine-parameter vectors are deliberately rejected:
            the latter can contain a second frequency derivative that
            ``GBParams`` cannot represent.
        duration: Positive observation duration in seconds, broadcastable
            with the vector's leading dimensions.

    Returns:
        Physical parameters, with canonical inclination and latitude, in the
        floating dtype of ``vector``; ``duration`` is cast to it.

    Raises:
        ValueError: If the vector does not have trailing dimension 8.

    """
    real = real_dtype(vector)
    vector = jnp.asarray(vector, dtype=real)
    if vector.ndim == 0 or vector.shape[-1] != NUM_PARAMS:
        raise ValueError(
            f"expected GLASS trailing axis of length 8, got {vector.shape}"
        )
    (
        f0_times_duration,
        sin_beta,
        lam,
        log_glass_amplitude,
        cos_iota,
        psi,
        phi0_glass,
        fdot_times_duration_sq,
    ) = jnp.moveaxis(vector, -1, 0)
    duration = jnp.asarray(duration, dtype=real)
    return GBParams(
        amp=0.5 * jnp.exp(log_glass_amplitude),
        f0=f0_times_duration / duration,
        fdot=fdot_times_duration_sq / duration**2,
        phi0=-phi0_glass,
        iota=jnp.arccos(cos_iota),
        psi=psi,
        lam=lam,
        beta=jnp.arcsin(sin_beta),
    )


__all__ = [
    "NUM_PARAMS",
    "PARAMETER_NAMES",
    "GBParams",
    "to_glass_vector",
    "from_glass_vector",
]
