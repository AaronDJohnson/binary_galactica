"""Eager boundary checks shared by the native and physical waveform APIs."""

import jax
import jax.numpy as jnp
import numpy as np


def _contains_jax(value):
    """Detect JAX values and conversion protocols before packing sequences."""
    if isinstance(value, (jax.Array, jax.core.Tracer)) or callable(
        getattr(value, "__jax_array__", None)
    ):
        return True
    return isinstance(value, (list, tuple)) and any(_contains_jax(x) for x in value)


def finite_array(value, name):
    """Validate float64 inputs, keeping host values off device until the JIT call."""
    # Packing host scalars with jnp dispatches before the compiled waveform.
    # JAX arrays and nested tracers must stay in JAX to preserve placement/AD.
    array_module = jnp if _contains_jax(value) else np
    value = array_module.asarray(value, dtype=np.float64)
    if not isinstance(value, jax.core.Tracer) and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


def scalar(value, name, *, positive=False):
    """Validate scalar shape during tracing, and concrete value eagerly."""
    value = finite_array(value, name)
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar; got shape {value.shape}")
    if positive and not isinstance(value, jax.core.Tracer) and np.asarray(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def observation(T, BW, t0, positions):
    """Check the shared observation arguments at the public boundary."""
    if (
        isinstance(BW, bool)
        or not isinstance(BW, (int, np.integer))
        or BW <= 0
        or BW % 2
    ):
        raise ValueError(f"BW must be a positive even integer; got {BW!r}")
    T = scalar(T, "T", positive=True)
    t0 = scalar(t0, "t0")
    if positions is not None:
        positions = finite_array(positions, "positions")
        if positions.shape != (3, BW, 3):
            raise ValueError(
                "positions must have shape (3, BW, 3); "
                f"got {positions.shape} for BW={BW}"
            )
    return T, t0, positions
