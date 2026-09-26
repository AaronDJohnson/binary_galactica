"""Value-only interpolation of uniformly sampled complex TDI envelopes.

Local Cartesian polynomials preserve a smooth complex response through zeros
without constructing singular amplitudes or phases. The interpolation is linear
in its samples: source derivatives and Hessians pass through the same operation.
No time derivatives or additional response evaluations are needed.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax import Array


@lru_cache(maxsize=5)
def _coefficient_matrix(points: int) -> np.ndarray:
    """Return centered cardinal-polynomial coefficients as host constants."""
    center = points // 2 - 1
    nodes = np.arange(points, dtype=float) - center
    columns = []
    for index in range(points):
        other = np.delete(nodes, index)
        columns.append(
            np.polynomial.polynomial.polyfromroots(other)
            / np.prod(nodes[index] - other)
        )
    return np.stack(columns, axis=1)


def interpolate_uniform(values: Array, coordinates: Array) -> Array:
    """Interpolate complex samples at normalized observation coordinates.

    ``values`` has shape ``(N, channels)`` with ``N >= 4`` and represents a
    uniform grid on the closed interval ``[0, 1]``. ``coordinates`` has arbitrary
    shape with entries in that interval; the caller validates their domain.
    The result has shape ``coordinates.shape + (channels,)`` and complex128
    dtype. Fixed normalized coordinates keep stencil selection independent of
    traced observation duration, epoch, and source parameters.

    Each local polynomial uses up to eight neighboring samples, with one-sided
    stencils at observation boundaries. Four through seven samples use the
    highest degree their grid supports. Away from boundaries, stencils advance
    with the containing interval. Time derivatives need not agree exactly
    across stencil boundaries; for a fixed observation grid, source dependence
    is smooth wherever the sampled response is smooth.

    A resolved smooth response converges at order eight under grid refinement.
    This is not an error bound or a guarantee of discovering all zeros. Remove
    known fast phase evolution before interpolation, and check convergence of
    the sparse grid for the intended source. Near misses retain their complex
    residual; this function neither asserts zeros nor changes amplitude signs.
    """
    values = jnp.asarray(values, dtype=jnp.complex128)
    coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
        raise ValueError("values must have shape (N, channels), N >= 4, channels >= 1")
    count, channels = values.shape
    points = min(8, count)
    center = points // 2 - 1
    windows = jnp.arange(count - points + 1)[:, None] + jnp.arange(points)[None, :]
    samples = values[windows]
    anchor = samples[:, center, :]
    matrix = jnp.asarray(_coefficient_matrix(points), dtype=jnp.float64)
    # Subtract a sampled central value before fitting. A constant response then
    # has exactly zero higher coefficients, avoiding cancellation in the fit.
    coefficients = jnp.einsum("kj,wjc->wck", matrix, samples - anchor[:, None, :])
    coefficients = coefficients.at[..., 0].add(anchor)
    sample_coordinate = coordinates.ravel() * (count - 1)
    cell = jnp.floor(sample_coordinate).astype(jnp.int32)
    window = jnp.clip(cell - center, 0, count - points)
    local = sample_coordinate - (window + center)
    # Gather only after constructing sparse-window coefficients. Moving sample
    # arithmetic past this gather can repeat response work at every dense time.
    selected = coefficients[window]
    result = selected[..., -1]
    for degree in range(points - 2, -1, -1):
        result = result * local[:, None] + selected[..., degree]
    return result.reshape(coordinates.shape + (channels,))
