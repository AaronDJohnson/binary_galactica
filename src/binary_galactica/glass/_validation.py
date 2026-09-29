"""Eager boundary checks shared by the native and physical waveform APIs."""

import jax
import jax.numpy as jnp
import numpy as np


def require_x64():
    """Refuse to run a float64 entry point while JAX would compute in float32."""
    # read() gives the effective value, including jax.experimental.enable_x64
    # scopes, on every supported JAX; it never changes the setting.
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "binary_galactica's float64 functions need JAX double precision, "
            "which is disabled. Call "
            'jax.config.update("jax_enable_x64", True) at startup, before '
            "creating JAX arrays, or use binary_galactica.single for float32 "
            "waveforms."
        )


def integer(value, name, *, minimum=None):
    """Require a concrete integer index/count without truncating or accepting bools."""
    requirement = {
        None: "an integer",
        0: "a non-negative integer",
        1: "a positive integer",
    }[minimum]
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(f"{name} must be {requirement}; got {value!r}")
    return int(value)


def _contains_jax(value):
    """Detect JAX values and conversion protocols before packing sequences."""
    if isinstance(value, (jax.Array, jax.core.Tracer)) or callable(
        getattr(value, "__jax_array__", None)
    ):
        return True
    return isinstance(value, (list, tuple)) and any(_contains_jax(x) for x in value)


def finite_array(value, name, *, dtype=np.float64):
    """Validate floating inputs, keeping host values off device until the JIT call.

    Values are cast to ``dtype``: float64 for the double-precision API and
    float32 for :mod:`binary_galactica.single`. A host value too large for
    float32 becomes infinite and is rejected like any other nonfinite input.
    """
    # Packing host scalars with jnp dispatches before the compiled waveform.
    # JAX arrays and nested tracers must stay in JAX to preserve placement/AD.
    if _contains_jax(value):
        value = jnp.asarray(value, dtype=dtype)
    else:
        with np.errstate(over="ignore"):
            value = np.asarray(value, dtype=dtype)
    if not isinstance(value, jax.core.Tracer) and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def scalar(value, name, *, positive=False, dtype=np.float64):
    """Validate scalar shape during tracing, and concrete value eagerly."""
    value = finite_array(value, name, dtype=dtype)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar; got shape {value.shape}")
    if positive and not isinstance(value, jax.core.Tracer) and np.asarray(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def observation(T, BW, t0, positions, *, dtype=np.float64):
    """Check the shared observation arguments at the public boundary."""
    if (
        isinstance(BW, bool)
        or not isinstance(BW, (int, np.integer))
        or BW <= 0
        or BW % 2
    ):
        raise ValueError(f"BW must be a positive even integer; got {BW!r}")
    T = scalar(T, "T", positive=True, dtype=dtype)
    t0 = scalar(t0, "t0", dtype=dtype)
    if positions is not None:
        positions = finite_array(positions, "positions", dtype=dtype)
        if positions.shape != (3, BW, 3):
            raise ValueError(
                "positions must have shape (3, BW, 3); "
                f"got {positions.shape} for BW={BW}"
            )
    return T, t0, positions
