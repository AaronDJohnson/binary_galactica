"""Shared fixtures, an independent reference implementation and jaxpr checks."""

from __future__ import annotations

import numpy as np
import pytest

from binary_galactica import GBParams

#: A generic source, deliberately with no special values: every angle is
#: off-axis and every parameter enters non-trivially.
FIDUCIAL = GBParams(
    amp=1e-22,
    f0=1.2345e-3,
    fdot=3.21e-16,
    phi0=0.7,
    iota=1.1,
    psi=0.4,
    lam=4.1,
    beta=-0.3,
)


@pytest.fixture
def params() -> GBParams:
    """Return the fiducial source."""
    return FIDUCIAL


@pytest.fixture
def t() -> np.ndarray:
    """Return a short, irregular-length time array in seconds."""
    return np.linspace(0.0, 1.0e5, 257)


def reference_polarizations(
    theta: GBParams, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an independent NumPy implementation for cross-checking.

    Written a different way from the package -- a complex exponential for the
    phase and an explicit rotation matrix for the polarization angle -- so that
    agreement is evidence about the model rather than about shared code.

    The convention is the historical one of ``galactic_binary()`` in ldasoft's
    ``galactic.c``, identical in GBGPU's ``run_wave``: with ``cosps, sinps =
    cos(2 psi), sin(2 psi)``, the complex coefficients
    ``DP = Aplus*cosps - 1j*Across*sinps`` and
    ``DC = -Aplus*sinps - 1j*Across*cosps`` give ``h+ = Re(DP e^{i Phi})`` and
    ``hx = Re(DC e^{i Phi})``, which is the rotation below. Both codes carry
    the minus sign in ``Across = -2 amp cos(iota)``, which fixes the handedness
    of a face-on source, and ``Re(DC) = -Aplus sin(2 psi)`` fixes the direction
    of the psi rotation.
    """
    amp, f0, fdot, phi0, iota, psi, _lam, _beta = (float(x) for x in theta)
    times = np.asarray(times, dtype=np.float64)

    cos_iota = np.cos(iota)
    a_plus = amp * (1.0 + cos_iota**2)
    a_cross = -2.0 * amp * cos_iota

    carrier = np.exp(1j * (phi0 + 2.0 * np.pi * f0 * times + np.pi * fdot * times**2))
    source_frame = np.stack([a_plus * carrier.real, a_cross * carrier.imag])

    rotation = np.array(
        [
            [np.cos(2.0 * psi), np.sin(2.0 * psi)],
            [-np.sin(2.0 * psi), np.cos(2.0 * psi)],
        ]
    )
    hp, hc = rotation @ source_frame
    return hp, hc


#: Dtypes that a float32 computation must never produce.
DOUBLE_DTYPES = (np.dtype(np.float64), np.dtype(np.complex128))


def nested_jaxprs(jaxpr):
    """Yield a (closed) jaxpr and every jaxpr nested in its equations."""
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    yield jaxpr
    for equation in jaxpr.eqns:
        for value in equation.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    yield from nested_jaxprs(inner)


def double_precision_values(jaxpr) -> list[str]:
    """Describe the float64 and complex128 values in a jaxpr and its inner jaxprs.

    With x64 enabled, a Python scalar passed to a jit-wrapped ``jnp`` function
    such as ``where`` or ``linspace`` enters that function as a weakly typed
    float64 scalar, which is converted to the working dtype before any
    arithmetic. Those are allowed. Every other double-precision input, such as
    a NumPy float64 constant, and every equation result in double precision is
    reported.
    """

    def double(var):
        return getattr(var.aval, "dtype", None) in DOUBLE_DTYPES

    def python_scalar(var):
        return var.aval.weak_type and var.aval.shape == ()

    found = []
    for inner in nested_jaxprs(jaxpr):
        inputs = [*inner.constvars, *inner.invars]
        inputs += [var for equation in inner.eqns for var in equation.invars]
        found += [
            f"input {var.aval.str_short()}"
            for var in inputs
            if double(var) and not python_scalar(var)
        ]
        found += [
            f"{equation.primitive.name} result {var.aval.str_short()}"
            for equation in inner.eqns
            for var in equation.outvars
            if double(var)
        ]
    return found
