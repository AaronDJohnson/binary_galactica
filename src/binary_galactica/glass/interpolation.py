"""Derivative-based reference interpolation of complex responses.

The mainline waveform uses the value-only interpolator in :mod:`.sparse`.
This Hermite utility retains explicit time derivatives and local refinement
for numerical comparisons and responses needing those reconstruction choices.

Polar cubic Hermite interpolation preserves smooth carrier-free phase. Near
zeros its representation is singular, so those cells use complex Cartesian
Hermite interpolation. A bounded number of suspect cells receive an additional
true-response sample, located by fixed-effort complex Gauss--Newton refinement.
No root is asserted, no residual is replaced by zero, and no amplitude sign
change is inferred from a scalar projection alone.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


def _hermite(y0, y1, d0, d1, width, fraction):
    """Evaluate a cubic with endpoint values and physical-time derivatives."""
    u = fraction
    return (
        (2 * u**3 - 3 * u**2 + 1) * y0
        + (u**3 - 2 * u**2 + u) * width * d0
        + (-2 * u**3 + 3 * u**2) * y1
        + (u**3 - u**2) * width * d1
    )


def _magnitude(value):
    """Compute magnitudes for discrete decisions without singular AD at zero."""
    return jnp.abs(jax.lax.stop_gradient(value))


def _cubic_coefficients(y0, y1, d0, d1, width):
    """Construct normalized-time cubic coefficients once per cell."""
    delta = y1 - y0
    return jnp.stack(
        (
            y0,
            width * d0,
            3 * delta - width * (2 * d0 + d1),
            -2 * delta + width * (d0 + d1),
        ),
        axis=-1,
    )


def _piece_coefficients(y0, y1, d0, d1, lower, upper, cartesian):
    """Pack coefficients and metadata into one gatherable cell table."""
    width = (upper - lower)[..., None]
    # Inactive refinement rows may have no representable interior timestamp.
    # Keep their arithmetic finite before selecting the unsplit cell.
    width = jnp.where(width > 0, width, 1.0)
    cart = _cubic_coefficients(y0, y1, d0, d1, width)
    mag0, mag1 = _magnitude(y0), _magnitude(y1)
    scale = jnp.maximum(mag0, mag1)
    safe = (jnp.minimum(mag0, mag1) > 0.15 * scale) & (scale > 0)
    # Mask before abs/angle/division so exact nulls have finite higher AD.
    z0, z1 = jnp.where(safe, y0, 1.0 + 0j), jnp.where(safe, y1, 1.0 + 0j)
    dz0, dz1 = jnp.where(safe, d0, 0j), jnp.where(safe, d1, 0j)
    a0, a1 = jnp.abs(z0), jnp.abs(z1)
    logarithmic0, logarithmic1 = dz0 / z0, dz1 / z1
    # Polar coordinates develop large curvature near nulls before an endpoint
    # magnitude becomes tiny. Keep those neighboring cells Cartesian too.
    radial_change = width * jnp.maximum(
        _magnitude(logarithmic0.real), _magnitude(logarithmic1.real)
    )
    safe = safe & (radial_change < 0.25)
    phase0 = jnp.angle(z0)
    delta = jnp.angle(z1 / z0)
    # Time derivatives disambiguate full phase windings between sparse knots.
    # The integer winding choice itself remains nondifferentiable.
    predicted = 0.5 * width * (logarithmic0.imag + logarithmic1.imag)
    delta = delta + 2 * jnp.pi * jax.lax.stop_gradient(
        jnp.round((predicted - delta) / (2 * jnp.pi))
    )
    amp = _cubic_coefficients(
        a0, a1, a0 * logarithmic0.real, a1 * logarithmic1.real, width
    )
    w0, w1 = logarithmic0.imag, logarithmic1.imag
    phase = jnp.stack(
        (
            phase0,
            width * w0,
            3 * delta - width * (2 * w0 + w1),
            -2 * delta + width * (w0 + w1),
        ),
        axis=-1,
    )
    metadata = jnp.stack(
        (
            safe & ~cartesian,
            jnp.broadcast_to(lower[..., None], y0.shape),
            jnp.broadcast_to(width, y0.shape),
        ),
        axis=-1,
    )
    # One packed table keeps costly response/phase work on the sparse cell
    # axis, instead of permitting separate endpoint gathers at every query.
    return jnp.concatenate((cart, amp, phase, metadata), axis=-1)


def _evaluate_coefficients(packed, query):
    """Evaluate gathered coefficients; all expensive cell algebra is upstream."""
    u = (query[..., None] - packed[..., 13].real) / packed[..., 14].real

    def polynomial(offset):
        return (
            (packed[..., offset + 3] * u + packed[..., offset + 2]) * u
            + packed[..., offset + 1]
        ) * u + packed[..., offset]

    complex_value = polynomial(0)
    polar_value = polynomial(4).real * jnp.exp(1j * polynomial(8).real)
    return jnp.where(packed[..., 12].real != 0, polar_value, complex_value)


def interpolate_response(
    response_fn: Callable[[Array], Array],
    knots: Array,
    query: Array,
    *,
    max_refinements: int = 8,
) -> Array:
    """Interpolate a pointwise complex response with zero-safe local refinement.

    ``response_fn(times)`` accepts a flat time array and returns
    ``(len(times), channels)``. It must support JAX time differentiation and
    evaluate the actual response, not a preexisting interpolant. ``knots`` is
    a strictly increasing one-dimensional grid of at least two times; ``query``
    has arbitrary shape and lies in that closed interval. The result has shape
    ``query.shape + (channels,)``. The caller validates time domains.

    The static ``max_refinements`` caps additional cells, independently of any
    Fourier bandwidth. The worst channel in each selected cell drives six
    safeguarded Gauss--Newton steps on the normalized complex magnitude. All
    channels are evaluated at the resulting split time. Unselected suspect
    cells retain Cartesian interpolation, so capacity overflow does not cause
    polar singularities. Set the cap to zero to disable refinement.

    Derivatives propagate through response evaluations and refinement times;
    cell selection, phase winding, and singularity guards are discrete. These
    choices make derivatives piecewise smooth, not globally smooth at a
    selection boundary. The fixed work budget does not guarantee exhaustive
    zero discovery or prescribed root precision. Increase the sparse grid
    density to verify waveform and derivative convergence. Higher-order zeros,
    near misses, endpoints, and identically zero channels require no sign flip.
    """
    knots = jnp.asarray(knots, dtype=jnp.float64)
    query = jnp.asarray(query, dtype=jnp.float64)
    if knots.ndim != 1 or knots.shape[0] < 2:
        raise ValueError("knots must be a one-dimensional grid of at least two times")
    if isinstance(max_refinements, bool) or not isinstance(max_refinements, int):
        raise ValueError("max_refinements must be a nonnegative integer")
    if max_refinements < 0:
        raise ValueError("max_refinements must be a nonnegative integer")

    def evaluate(times):
        return jax.jvp(response_fn, (times,), (jnp.ones_like(times),))

    values, derivatives = evaluate(knots)
    if values.ndim != 2 or values.shape[0] != knots.shape[0]:
        raise ValueError("response_fn must return (number of times, channels)")
    lower, upper = knots[:-1], knots[1:]
    width = upper - lower
    y0, y1 = values[:-1], values[1:]
    d0, d1 = derivatives[:-1], derivatives[1:]
    scale = jnp.maximum(_magnitude(y0), _magnitude(y1))
    scale = jnp.where(scale > 0, scale, 1.0)
    fractions = jnp.array([0.25, 0.5, 0.75])
    trial = _hermite(
        y0[:, None],
        y1[:, None],
        d0[:, None],
        d1[:, None],
        width[:, None, None],
        fractions[None, :, None],
    )
    ratios = _magnitude(trial) / scale[:, None, :]
    # A persistently zero channel has no isolated root and must not consume
    # every refinement slot while other channels contain informative minima.
    dormant = (_magnitude(y0) + _magnitude(y1) + _magnitude(d0) + _magnitude(d1)) == 0
    ratios = jnp.where(dormant[:, None, :], 1.0, ratios)
    channel_scores = jnp.min(ratios, axis=1)
    scores = jnp.min(channel_scores, axis=1)
    suspect = scores < 0.35
    complex_cells = channel_scores < 0.35
    cell_index = jnp.clip(
        jnp.searchsorted(knots, query.ravel(), side="right") - 1, 0, knots.shape[0] - 2
    )
    flat_query = query.ravel()
    packed = _piece_coefficients(y0, y1, d0, d1, lower, upper, complex_cells)
    query_rows = cell_index

    count = min(max_refinements, knots.shape[0] - 1)
    if count:
        _, selected = jax.lax.top_k(-scores, count)
        channel = jnp.argmin(channel_scores[selected], axis=1)
        slot = jnp.arange(count)
        chosen_ratios = ratios[selected, :, channel]
        fraction = fractions[jnp.argmin(chosen_ratios, axis=1)]
        lo, hi, h = lower[selected], upper[selected], width[selected]
        t = lo + h * fraction
        norm = scale[selected, channel]

        def refine(_, time):
            z, dz = evaluate(time)
            z, dz = z[slot, channel] / norm, dz[slot, channel] / norm
            power = dz.real**2 + dz.imag**2
            safe_power = jnp.where(power > jnp.finfo(knots.dtype).tiny, power, 1.0)
            step = (dz.real * z.real + dz.imag * z.imag) / safe_power
            return jnp.clip(time - step, lo + 0.001 * h, hi - 0.001 * h)

        refined = jax.lax.fori_loop(0, 6, refine, t)
        # A boundary root cannot supply a well-conditioned interior split.
        interior = (refined > lo + 0.01 * h) & (refined < hi - 0.01 * h)
        refined = jnp.where(interior, refined, lo + 0.5 * h)
        split_values, split_derivatives = evaluate(refined)
        usable = suspect[selected] & (refined > lo) & (refined < hi)
        # A coarse cubic can predict a spurious minimum during smooth phase
        # rotation; the actual split sample determines whether to stay polar.
        refined_complex = (_magnitude(split_values) / scale[selected]) < 0.35
        left_packed = _piece_coefficients(
            y0[selected],
            split_values,
            d0[selected],
            split_derivatives,
            lo,
            refined,
            refined_complex,
        )
        right_packed = _piece_coefficients(
            split_values,
            y1[selected],
            split_derivatives,
            d1[selected],
            refined,
            hi,
            refined_complex,
        )
        slot_by_cell = jnp.full(scores.shape, -1, dtype=jnp.int32)
        slot_by_cell = slot_by_cell.at[selected].set(slot.astype(jnp.int32))
        query_slot = jnp.maximum(slot_by_cell[cell_index], 0)
        active = (slot_by_cell[cell_index] >= 0) & usable[query_slot]
        right = flat_query > refined[query_slot]
        query_rows = jnp.where(
            active, scores.shape[0] + query_slot + right * count, cell_index
        )
        packed = jnp.concatenate((packed, left_packed, right_packed), axis=0)
    output = _evaluate_coefficients(packed[query_rows], flat_query)
    return output.reshape(query.shape + (values.shape[1],))
