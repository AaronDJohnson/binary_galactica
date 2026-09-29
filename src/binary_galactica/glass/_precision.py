"""Working precision for kernels that follow the dtype of their inputs.

Float64 inputs keep float64 arithmetic and results; float32 inputs compute
float32 and complex64 values throughout, even with x64 enabled. Python scalars
are weakly typed in JAX: they adopt the working precision without promoting
it. NumPy float64 values and default-dtype constructors are not, so kernels
cast them to the working dtype explicitly.
"""

import jax
import jax.numpy as jnp
import numpy as np

#: Precision of every contraction. The default lets float32 contractions use
#: TF32 on NVIDIA GPUs and reduced-precision passes on TPUs. CPU float64
#: ignores the setting, so float64 results do not change.
CONTRACTION = jax.lax.Precision.HIGHEST


def real_dtype(*values) -> np.dtype:
    """Return the real working dtype of array-like ``values``, at least float32.

    Complex values give their component dtype. Integer and boolean inputs, and
    Python scalars on their own, give JAX's default floating dtype: float64
    only when x64 is enabled.
    """
    # Nested sequences have no dtype of their own; their elements decide it.
    leaves = jax.tree_util.tree_leaves(values)
    dtype = jnp.result_type(*leaves) if leaves else jnp.result_type(float)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(float)
    return jnp.promote_types(jnp.finfo(dtype).dtype, jnp.float32)


def complex_dtype(*values) -> np.dtype:
    """Return the complex dtype with the precision of :func:`real_dtype`."""
    return jnp.promote_types(real_dtype(*values), jnp.complex64)


def cast(value, dtype):
    """Return ``value`` in ``dtype``, leaving Python scalars weakly typed.

    A Python ``int`` or ``float`` already adopts the precision of the arrays it
    meets, so it is returned unchanged; arithmetic on it, such as ``T**3``,
    stays in Python double precision.
    """
    if type(value) in (int, float):
        return value
    return jnp.asarray(value, dtype=dtype)
