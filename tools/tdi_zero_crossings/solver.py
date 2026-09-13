"""CPU reference for complex zero refinement, independent of waveform synthesis.

Supply a smooth, preferably demodulated complex response. The callback must
evaluate the response at arbitrary times: refining an interpolant only locates
zeros of that interpolant. These routines do not assign an amplitude sign or a
phase; they return diagnostics for a caller to validate before interpolation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite, ulp
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

Method = Literal["bounded", "brentq", "toms748"]
Kind = Literal["crossing", "touch", "near_miss", "boundary", "unresolved"]
Response = Callable[[float], complex]


class _EvaluationLimitError(Exception):
    """Stop optimization stages when their shared objective budget is spent."""


@dataclass(frozen=True)
class ZeroResult:
    """A numerical candidate, including all callback costs of its refinement.

    ``kind`` describes the evidence at the requested tolerances, not a proof of
    an exact mathematical zero. ``method`` records the actual solver, including
    a bounded fallback when a projection has no sign-changing bracket.
    ``boundary`` and ``unresolved`` must not trigger amplitude sign changes.
    """

    time: float
    value: complex
    residual: float
    scale: float
    kind: Kind
    method: str
    evaluations: int
    converged: bool


def _value(response: Response, time: float) -> complex:
    """Require a finite scalar response before numerical operations."""
    raw = np.asarray(response(time))
    if raw.ndim != 0:
        raise ValueError("response must return a finite complex scalar")
    value = complex(raw)
    if not isfinite(value.real) or not isfinite(value.imag) or not isfinite(abs(value)):
        raise ValueError("response must return a finite complex scalar")
    return value


def _direction(
    evaluate: Response,
    time: float,
    lower: float,
    upper: float,
    threshold: float,
    time_atol: float,
) -> Kind:
    """Require consistent directions on two progressively smaller probe pairs.

    A simple zero has opposite complex directions on its two sides; an even
    order zero has the same direction. The 1e-3 angular-cosine tolerance is a
    conservative classification control, not the zero acceptance tolerance.
    Unstable or numerically insignificant probe directions remain unresolved.
    """
    radius = min(time - lower, upper - time)
    step = min(0.05 * (upper - lower), radius / 2)
    previous: Kind = "unresolved"
    for _ in range(8):
        if step < 8 * time_atol:
            break
        left = evaluate(time - step)
        right = evaluate(time + step)
        # Ratios of unit complex numbers avoid squaring tiny/huge amplitudes.
        if min(abs(left), abs(right)) <= 8 * threshold:
            break
        cosine = ((left / abs(left)).conjugate() * (right / abs(right))).real
        current: Kind = "unresolved"
        if cosine < -1 + 1e-3:
            current = "crossing"
        elif cosine > 1 - 1e-3:
            current = "touch"
        if current != "unresolved" and current == previous:
            return current
        previous = current
        step /= 4
    return "unresolved"


def _validate_controls(
    interval: tuple[float, float],
    method: Method,
    amplitude_atol: float,
    amplitude_rtol: float,
    time_atol: float,
    maxiter: int,
) -> tuple[float, float, float]:
    """Check numerical controls without evaluating the supplied response."""
    bounds = np.asarray(interval, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
        raise ValueError("interval must contain two finite increasing times")
    lower, upper = map(float, bounds)
    width = upper - lower
    if not isfinite(width) or width <= 0:
        raise ValueError("interval must contain two finite increasing times")
    if method not in ("bounded", "brentq", "toms748"):
        raise ValueError("method must be bounded, brentq, or toms748")
    if not isfinite(time_atol) or time_atol <= 0 or time_atol >= width:
        raise ValueError("time_atol must be positive and smaller than the interval")
    if time_atol < max(ulp(lower), ulp(upper)):
        raise ValueError("time_atol is smaller than floating-point time resolution")
    if any(not isfinite(x) or x < 0 for x in (amplitude_atol, amplitude_rtol)):
        raise ValueError("amplitude tolerances must be finite and nonnegative")
    if amplitude_atol == amplitude_rtol == 0:
        raise ValueError("at least one amplitude tolerance must be positive")
    if isinstance(maxiter, bool) or not isinstance(maxiter, (int, np.integer)):
        raise ValueError("maxiter must be a positive integer")
    if maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    return lower, upper, width


def refine_candidate(
    response: Response,
    interval: tuple[float, float],
    *,
    method: Method = "bounded",
    amplitude_atol: float = 0.0,
    amplitude_rtol: float = 1e-8,
    time_atol: float = 1e-8,
    maxiter: int = 100,
) -> ZeroResult:
    """Refine one isolated candidate and distinguish crossings from near misses.

    ``bounded`` minimizes normalized squared magnitude, then recenters a second
    solve inside the nearest sampled bracket. ``brentq`` and ``toms748`` solve
    the projection onto the endpoint complex secant, falling back to bounded
    refinement when no signed bracket exists or the complex residual fails.
    All methods
    check the full complex residual against ``amplitude_atol + amplitude_rtol
    * scale``; ``scale`` is the largest magnitude at the two endpoints and
    midpoint. A successful projection solve alone never implies a complex zero.

    Times are normalized inside the solver. ``time_atol`` is expressed in the
    callback's time units and must be attainable at both endpoints. The stock
    recentered solve reduces the minimizer's floating-point relative stopping
    floor. Callback cancellation and flat nonzero minima still limit accuracy;
    a tight time tolerance alone does not guarantee a small residual or an
    accurate physical zero. Inspect ``kind`` and
    ``converged``. A near miss is a tested nonzero candidate, not proof that the
    interval contains no other zeros; subdivision is the caller's responsibility.

    The callback must be deterministic. Results count its actual evaluations,
    including endpoint checks and direction probes; repeated times are cached.
    ``maxiter`` caps objective calls across all optimization stages, including
    calls served from the cache. Initial samples, final residual verification,
    and direction probes are extra.
    """
    lower, upper, width = _validate_controls(
        interval, method, amplitude_atol, amplitude_rtol, time_atol, maxiter
    )

    cache: dict[float, complex] = {}

    def evaluate(time: float) -> complex:
        time = float(time)
        if time not in cache:
            cache[time] = _value(response, time)
        return cache[time]

    midpoint = lower + 0.5 * width
    initial = [(t, evaluate(t)) for t in (lower, midpoint, upper)]
    scale = max(abs(value) for _, value in initial)
    threshold = amplitude_atol + amplitude_rtol * scale
    if not isfinite(threshold):
        raise ValueError("amplitude tolerance overflows at this response scale")

    def result(time: float, kind: Kind, used: str, converged: bool) -> ZeroResult:
        value = evaluate(time)
        return ZeroResult(
            time, value, abs(value), scale, kind, used, len(cache), converged
        )

    if scale == 0:
        return result(midpoint, "unresolved", "samples", False)
    for time in (lower, upper):
        if evaluate(time) == 0:
            return result(time, "boundary", "samples", True)
    # Keep exact evidence already found by the scan. A local minimizer may
    # otherwise choose a different basin even when this midpoint is a zero.
    if evaluate(midpoint) == 0:
        kind = _direction(evaluate, midpoint, lower, upper, threshold, time_atol)
        return result(midpoint, kind, "samples", True)

    def physical_time(unit_time: float) -> float:
        # In particular, lower + (upper - lower) can exceed upper by one ULP.
        if unit_time <= 0:
            return lower
        if unit_time >= 1:
            return upper
        return min(upper, max(lower, lower + width * unit_time))

    objective_calls = 0

    def normalized_value(time: float) -> complex:
        nonlocal objective_calls
        if objective_calls >= maxiter:
            raise _EvaluationLimitError
        objective_calls += 1
        normalized = evaluate(time) / scale
        if not isfinite(abs(normalized)):
            raise ValueError("response scale changes too much; subdivide the interval")
        return normalized

    def power_at(time: float) -> float:
        magnitude = abs(normalized_value(time))
        if magnitude > np.sqrt(np.finfo(float).max):
            raise ValueError("response scale changes too much; subdivide the interval")
        return magnitude * magnitude

    unit_atol = time_atol / width

    def minimize_magnitude() -> bool:
        # First locate a basin on normalized time. Its relative stopping term
        # is O(sqrt(eps)*abs(x)), even when the caller requests smaller xatol.
        if objective_calls >= maxiter:
            return False
        minimum = minimize_scalar(
            lambda u: power_at(physical_time(u)),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": unit_atol, "maxiter": maxiter - objective_calls},
        )
        if not minimum.success:
            return False
        center = min(cache, key=lambda t: abs(cache[t]))
        if cache[center] == 0:
            return True

        # The nearest tested neighbors provide a local bracket without an
        # assumed error radius. At an endpoint, use the one-sided bracket:
        # a small endpoint residual can conceal a nearby interior zero.
        sampled = sorted(cache)
        index = sampled.index(center)
        local_lower = sampled[max(0, index - 1)]
        local_upper = sampled[min(len(sampled) - 1, index + 1)]
        if local_upper - local_lower <= time_atol:
            return True
        if objective_calls >= maxiter:
            return False

        def centered_power(delta: float) -> float:
            # Evaluate a small displacement from the actual cached timestamp.
            # Avoid converting a large absolute epoch to unit time and back.
            time = min(local_upper, max(local_lower, center + width * delta))
            return power_at(time)

        polished = minimize_scalar(
            centered_power,
            bounds=((local_lower - center) / width, (local_upper - center) / width),
            method="bounded",
            options={"xatol": unit_atol, "maxiter": maxiter - objective_calls},
        )
        return bool(polished.success)

    used = method
    try:
        if method != "bounded":
            left, right = evaluate(lower) / scale, evaluate(upper) / scale
            secant = right - left
            direction = secant / abs(secant) if secant != 0 else 0j

            def projection(unit_time: float) -> float:
                return float(
                    (
                        direction.conjugate()
                        * normalized_value(physical_time(unit_time))
                    ).real
                )

            first = (direction.conjugate() * left).real
            last = (direction.conjugate() * right).real
            if (first < 0 < last) or (last < 0 < first):
                solution = root_scalar(
                    projection,
                    bracket=(0.0, 1.0),
                    method=method,
                    xtol=unit_atol,
                    maxiter=maxiter,
                )
                converged = bool(solution.converged)
                # Validate the projected solution itself. An unrelated cached
                # near-zero sample cannot certify its location or hide failure.
                if (
                    converged
                    and abs(evaluate(physical_time(solution.root))) > threshold
                ):
                    used = "bounded"
            else:
                used = "bounded"
        if used == "bounded":
            converged = minimize_magnitude()
    except _EvaluationLimitError:
        converged = False

    # Retain the best evaluated candidate; never replace a known lower
    # magnitude by the final (possibly less accurate) solver iterate.
    time = min(cache, key=lambda t: abs(cache[t]))
    if not converged:
        return result(time, "unresolved", used, False)
    value = evaluate(time)
    if abs(value) > threshold:
        return result(time, "near_miss", used, True)
    if min(time - lower, upper - time) <= 8 * time_atol:
        return result(time, "boundary", used, True)
    kind = _direction(evaluate, time, lower, upper, threshold, time_atol)
    return result(time, kind, used, True)


def scan_candidates(
    response: Response,
    times: Sequence[float],
    *,
    subdivisions: int = 4,
    method: Method = "bounded",
    amplitude_atol: float = 0.0,
    amplitude_rtol: float = 1e-8,
    time_atol: float = 1e-8,
    maxiter: int = 100,
) -> tuple[ZeroResult, ...]:
    """Subdivide a grid, retain tied minima, and refine candidate intervals.

    ``subdivisions`` equally divides each original interval, including a
    nonuniform one. It is a resolution control, not a guarantee to find every
    zero: compare denser scans when the response varies faster than the grid.
    Global endpoint minima are retained and cannot imply a sign flip without
    samples outside the observation. Results include near misses for inspection.

    Refinement evaluates the callback afresh; its per-result evaluation count
    excludes the initial scan. Wrap the callback to measure total scan cost.
    """
    times_array = np.asarray(times, dtype=float)
    if (
        times_array.ndim != 1
        or len(times_array) < 2
        or not np.all(np.isfinite(times_array))
        or not np.all(np.diff(times_array) > 0)
    ):
        raise ValueError("times must be a finite, strictly increasing 1-D grid")
    if isinstance(subdivisions, bool) or not isinstance(
        subdivisions, (int, np.integer)
    ):
        raise ValueError("subdivisions must be a positive integer")
    if subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    _validate_controls(
        (float(times_array[0]), float(times_array[-1])),
        method,
        amplitude_atol,
        amplitude_rtol,
        time_atol,
        maxiter,
    )
    grid = np.concatenate(
        [
            np.linspace(a, b, subdivisions + 1)[:-1]
            for a, b in zip(times_array[:-1], times_array[1:], strict=True)
        ]
        + [times_array[-1:]]
    )
    if not np.all(np.diff(grid) > 0):
        raise ValueError("subdivisions exceed floating-point time resolution")
    amplitudes = np.array([abs(_value(response, float(t))) for t in grid])
    if np.all(amplitudes == amplitudes[0]) and amplitudes[0] != 0:
        return ()

    # Group nonzero tied minima, but retain each exact sampled zero separately.
    # Adjacent exact zeros are evidence at distinct times, not a single valley.
    minima = (
        np.flatnonzero(
            (amplitudes[1:-1] <= amplitudes[:-2])
            & (amplitudes[1:-1] <= amplitudes[2:])
            & (amplitudes[1:-1] > 0)
        )
        + 1
    )
    groups = np.split(minima, np.flatnonzero(np.diff(minima) != 1) + 1)
    intervals = [
        (float(grid[g[0] - 1]), float(grid[g[-1] + 1])) for g in groups if len(g)
    ]
    if 0 < amplitudes[0] < amplitudes[1]:
        intervals.append((float(grid[0]), float(grid[1])))
    if 0 < amplitudes[-1] < amplitudes[-2]:
        intervals.append((float(grid[-2]), float(grid[-1])))
    if np.all(amplitudes == 0):
        intervals = [(float(grid[0]), float(grid[-1]))]
    else:
        for index in np.flatnonzero(amplitudes == 0):
            if index == 0:
                intervals.append(
                    (float(grid[0]), float(grid[0] + 0.5 * (grid[1] - grid[0])))
                )
            elif index == len(grid) - 1:
                intervals.append(
                    (float(grid[-1] - 0.5 * (grid[-1] - grid[-2])), float(grid[-1]))
                )
            else:
                time = float(grid[index])
                radius = 0.5 * min(time - grid[index - 1], grid[index + 1] - time)
                intervals.append((time - radius, time + radius))

    candidates = (
        refine_candidate(
            response,
            (start, stop),
            method=method,
            amplitude_atol=amplitude_atol,
            amplitude_rtol=amplitude_rtol,
            time_atol=time_atol,
            maxiter=maxiter,
        )
        for start, stop in intervals
    )
    return tuple(sorted(candidates, key=lambda candidate: candidate.time))
