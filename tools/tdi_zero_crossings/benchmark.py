"""Compare candidate refinement methods on reproducible synthetic responses.

Run from the repository root with ``python -m tools.tdi_zero_crossings.benchmark``.
This measures Python/SciPy candidate refinement, not the production TDI pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from .solver import refine_candidate

METHODS = ("legacy_curvature", "bounded", "brentq", "toms748")


@dataclass(frozen=True)
class Case:
    """One isolated candidate with analytic ground truth and fixed sample nodes."""

    name: str
    kind: str
    interval: tuple[float, float]
    amplitude: float
    root_fraction: float
    nodes: tuple[float, ...]
    phase: float = 0.0
    rotation: float = 0.0
    gap: float = 0.0
    envelope: float = 0.0
    rotating_only: bool = False

    @property
    def width(self) -> float:
        """Return the candidate interval width."""
        return self.interval[1] - self.interval[0]

    @property
    def true_time(self) -> float | None:
        """Return the known zero time, if this response has a zero."""
        if self.kind == "near_miss":
            return None
        return self.interval[0] + self.width * self.root_fraction

    def response(self, t: float) -> complex:
        """Evaluate a smooth, locally parameterized complex response."""
        x = (t - self.interval[0]) / self.width
        u = x - self.root_fraction
        if self.rotating_only:
            value = complex(1.0)
        elif self.kind == "touch":
            value = complex(u * u)
        else:
            value = complex(u, self.gap)
        angle = self.phase + self.rotation * u
        carrier = complex(math.cos(angle), math.sin(angle))
        return self.amplitude * math.exp(self.envelope * x) * value * carrier


class CountedResponse:
    """Count actual response calls, including residual and crossing checks."""

    def __init__(self, response: Callable[[float], complex]) -> None:
        """Wrap one case response with an independent evaluation counter."""
        self.response = response
        self.evaluations = 0

    def __call__(self, t: float) -> complex:
        """Evaluate the response and increment the callback count."""
        self.evaluations += 1
        return self.response(t)


def make_cases(seed: int = 20260913, random_cases: int = 24) -> list[Case]:
    """Build paired amplitude scales and fixed plus seeded candidate shapes."""
    rng = np.random.default_rng(seed)
    uniform = (0.0, 0.25, 0.5, 0.75, 1.0)
    nonuniform = (0.0, 0.08, 0.31, 0.68, 1.0)
    # name, kind, zero/minimum location, nodes, rotation, imaginary gap
    shapes = [
        ("simple", "crossing", 0.37, uniform, 0.0, 0.0),
        ("sampled_zero", "crossing", 0.5, uniform, 0.0, 0.0),
        ("tied_midpoint", "crossing", 0.375, uniform, 0.0, 0.0),
        ("near_left", "crossing", 0.01, uniform, 0.0, 0.0),
        ("near_right", "crossing", 0.99, uniform, 0.0, 0.0),
        ("endpoint_left", "boundary", 0.0, uniform, 0.0, 0.0),
        ("endpoint_right", "boundary", 1.0, uniform, 0.0, 0.0),
        ("tangent", "touch", 0.43, uniform, 0.0, 0.0),
        ("near_miss", "near_miss", 0.43, uniform, 0.0, 0.003),
        ("rotating", "crossing", 0.37, uniform, 2.0, 0.0),
        ("fast_rotation", "crossing", 0.37, uniform, 8.0, 0.0),
        ("nonuniform", "crossing", 0.47, nonuniform, 0.0, 0.0),
        ("nonuniform_tangent", "touch", 0.31, nonuniform, 0.0, 0.0),
        ("constant_modulus", "near_miss", 0.5, nonuniform, 8.0, 0.0),
    ]
    cases = []
    for width in (1.0, 2e-5):
        for amplitude in (1.0, 1e-22):
            for name, kind, root, nodes, rotation, gap in shapes:
                cases.append(
                    Case(
                        name=f"{name}/width={width:g}/amplitude={amplitude:g}",
                        kind=kind,
                        interval=(0.0, width),
                        amplitude=amplitude,
                        root_fraction=root,
                        nodes=nodes,
                        rotation=rotation,
                        gap=gap,
                        rotating_only=name == "constant_modulus",
                    )
                )
    for index in range(random_cases):
        kind = ("crossing", "touch", "near_miss")[index % 3]
        root = float(rng.uniform(0.02, 0.98))
        phase = float(rng.uniform(-math.pi, math.pi))
        rotation = float(rng.uniform(-2.0, 2.0))
        envelope = float(rng.uniform(-0.8, 0.8))
        width = 10.0 ** float(rng.uniform(-5.0, 1.0))
        nodes = (0.0, *sorted(float(x) for x in rng.uniform(0.03, 0.97, 3)), 1.0)
        for amplitude in (1.0, 1e-22):
            cases.append(
                Case(
                    name=f"seeded_{index:03d}/amplitude={amplitude:g}",
                    kind=kind,
                    interval=(0.0, width),
                    amplitude=amplitude,
                    root_fraction=root,
                    nodes=nodes,
                    phase=phase,
                    rotation=rotation,
                    gap=0.003 if kind == "near_miss" else 0.0,
                    envelope=envelope,
                )
            )
    return cases


def legacy_flips(
    response: Callable[[float], complex], times: Sequence[float]
) -> tuple[int, ...]:
    """Mirror strict minima, curvature tests, and final-sample suppression.

    The source assigns counts, cumulatively sums them, then copies the penultimate
    flip onto the final sample. Return effective count locations, not zero times.
    """
    amplitudes = [abs(response(t)) for t in times]
    counts = [0] * len(amplitudes)
    for i in range(1, len(amplitudes) - 1):
        left, middle, right = amplitudes[i - 1 : i + 2]
        if middle < left and middle < right:
            curvature = right + left - 2.0 * middle
            after = -right + left - 2.0 * middle
            before = -right + left + 2.0 * middle
            if abs(after / curvature) < 0.1:
                counts[i + 1] = 1
            elif abs(before / curvature) < 0.1:
                counts[i] = 1
    return tuple(i for i in range(1, len(counts) - 1) if counts[i])


def run_case(
    case: Case, method: str, amplitude_rtol: float, time_fraction_atol: float
) -> dict:
    """Time a complete solver call and record metrics against analytic truth."""
    callback = CountedResponse(case.response)
    start = time.process_time_ns()
    if method == "legacy_curvature":
        times = tuple(case.interval[0] + case.width * x for x in case.nodes)
        flips = legacy_flips(callback, times)
        result = None
    else:
        flips = ()
        result = refine_candidate(
            callback,
            case.interval,
            method=method,
            amplitude_atol=0.0,
            amplitude_rtol=amplitude_rtol,
            time_atol=case.width * time_fraction_atol,
            maxiter=100,
        )
    cpu_seconds = (time.process_time_ns() - start) * 1e-9
    predicted_crossing = bool(flips) if result is None else result.kind == "crossing"
    record = {
        "name": case.name,
        "expected_kind": case.kind,
        "kind": ("crossing" if flips else "no_crossing")
        if result is None
        else result.kind,
        "method_used": method if result is None else result.method,
        "crossing_correct": predicted_crossing == (case.kind == "crossing"),
        "classification_correct": None if result is None else result.kind == case.kind,
        "evaluations": callback.evaluations,
        "cpu_seconds": cpu_seconds,
        "time": None if result is None else result.time,
        "time_error": None,
        "time_error_fraction": None,
        "time_within_tolerance": None,
        "residual": None if result is None else result.residual,
        "scale": None if result is None else result.scale,
        "converged": None if result is None else result.converged,
        "reported_evaluations": None if result is None else result.evaluations,
        "flip_indices": list(flips) if result is None else None,
    }
    if result is not None:
        if result.evaluations != callback.evaluations:
            raise RuntimeError(
                f"{method}/{case.name}: reported {result.evaluations} response "
                f"evaluations, observed {callback.evaluations}"
            )
        if case.true_time is not None:
            record["time_error"] = abs(result.time - case.true_time)
            record["time_error_fraction"] = record["time_error"] / case.width
            record["time_within_tolerance"] = (
                record["time_error_fraction"] <= time_fraction_atol
            )
    return record


def summarize(records: list[dict]) -> dict:
    """Aggregate repeated solves without assigning root accuracy to the baseline."""
    errors = [r["time_error_fraction"] for r in records if r["time_error"] is not None]
    absolute_errors = [r["time_error"] for r in records if r["time_error"] is not None]
    localized = [r for r in records if r["time_within_tolerance"] is not None]
    classified = [r for r in records if r["classification_correct"] is not None]
    confusion = Counter((r["expected_kind"], r["kind"]) for r in records)
    expected_crossings = [r for r in records if r["expected_kind"] == "crossing"]
    others = [r for r in records if r["expected_kind"] != "crossing"]
    return {
        "runs": len(records),
        "cpu_seconds_total": sum(r["cpu_seconds"] for r in records),
        "cpu_seconds_median": statistics.median(r["cpu_seconds"] for r in records),
        "evaluations_total": sum(r["evaluations"] for r in records),
        "evaluations_mean": statistics.mean(r["evaluations"] for r in records),
        "evaluations_median": statistics.median(r["evaluations"] for r in records),
        "evaluations_max": max(r["evaluations"] for r in records),
        "classification_accuracy": None
        if not classified
        else statistics.mean(r["classification_correct"] for r in classified),
        "crossing_accuracy": statistics.mean(r["crossing_correct"] for r in records),
        "crossing_recall": statistics.mean(
            r["crossing_correct"] for r in expected_crossings
        ),
        "false_crossings": sum(not r["crossing_correct"] for r in others),
        "converged_fraction": None
        if not classified
        else statistics.mean(r["converged"] for r in classified),
        "time_error_max": max(absolute_errors) if absolute_errors else None,
        "time_error_fraction_median": statistics.median(errors) if errors else None,
        "time_error_fraction_max": max(errors) if errors else None,
        "time_accuracy_fraction": None
        if not localized
        else statistics.mean(r["time_within_tolerance"] for r in localized),
        "methods_used": dict(Counter(r["method_used"] for r in records)),
        "confusion": [
            {"expected": expected, "observed": observed, "count": count}
            for (expected, observed), count in sorted(confusion.items())
        ],
    }


def benchmark(
    *,
    seed: int = 20260913,
    random_cases: int = 24,
    repeats: int = 3,
    amplitude_rtol: float = 1e-8,
    time_fraction_atol: float = 1e-12,
) -> dict:
    """Measure identical isolated candidates, with untimed per-method warmups."""
    if repeats < 1 or random_cases < 0:
        raise ValueError("repeats must be positive and random_cases nonnegative")
    if not 0.0 < amplitude_rtol < 1.0 or not 0.0 < time_fraction_atol < 1.0:
        raise ValueError("relative tolerances must lie strictly between zero and one")
    cases = make_cases(seed, random_cases)
    records: dict[str, list[dict]] = {method: [] for method in METHODS}
    for method in METHODS:
        run_case(cases[0], method, amplitude_rtol, time_fraction_atol)
    for repeat in range(repeats):
        # Rotate method order to reduce a systematic first-run timing advantage.
        order = METHODS[repeat % len(METHODS) :] + METHODS[: repeat % len(METHODS)]
        for case in cases:
            for method in order:
                record = run_case(case, method, amplitude_rtol, time_fraction_atol)
                record["repeat"] = repeat
                records[method].append(record)
    return {
        "schema_version": 1,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "clock": "time.process_time_ns",
        },
        "configuration": {
            "seed": seed,
            "random_cases": random_cases,
            "cases": len(cases),
            "repeats": repeats,
            "amplitude_atol": 0.0,
            "amplitude_rtol": amplitude_rtol,
            "time_atol": "interval_width * time_fraction_atol",
            "time_fraction_atol": time_fraction_atol,
            "maxiter": 100,
        },
        "limitations": [
            "Synthetic scalar Python callbacks do not establish production speedup.",
            "All refiners receive the same known candidate interval; "
            "discovery is excluded.",
            "The legacy method receives five samples and detects flips only; "
            "it does not estimate zero times or distinguish touches from near misses.",
            "Legacy callback counts include sampling, although the production "
            "heuristic reads already available amplitudes. "
            "Its CPU time is not a pipeline comparison.",
            "An endpoint zero is labeled boundary and is not an interior crossing.",
            "Errors include every true zero, "
            "even misclassified or unconverged estimates.",
            "Time accuracy is the fraction of all true-zero estimates within "
            "the requested time tolerance, independently of classification.",
            "CPU times include refinement and classification; "
            "warmups and reporting are excluded.",
            "Fast rotation deliberately probes "
            "projection and local-minimum limitations.",
        ],
        "cases": [asdict(case) for case in cases],
        "summary": {method: summarize(rows) for method, rows in records.items()},
        "results": records,
    }


def main() -> None:
    """Write full benchmark JSON and print a compact, qualified comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/tdi_zero_crossings.json")
    )
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--random-cases", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--amplitude-rtol", type=float, default=1e-8)
    parser.add_argument("--time-fraction-atol", type=float, default=1e-12)
    args = parser.parse_args()
    report = benchmark(
        seed=args.seed,
        random_cases=args.random_cases,
        repeats=args.repeats,
        amplitude_rtol=args.amplitude_rtol,
        time_fraction_atol=args.time_fraction_atol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Synthetic candidates: {len(report['cases'])}; repeats: {args.repeats}")
    print(
        "method              kind accuracy  crossing accuracy  "
        "within time tol  mean calls    CPU (s)"
    )
    for method, summary in report["summary"].items():
        accuracy = summary["classification_accuracy"]
        label = "n/a" if accuracy is None else f"{accuracy:.1%}"
        time_accuracy = summary["time_accuracy_fraction"]
        time_label = "n/a" if time_accuracy is None else f"{time_accuracy:.1%}"
        print(
            f"{method:19} {label:>13} {summary['crossing_accuracy']:>18.1%} "
            f"{time_label:>16} "
            f"{summary['evaluations_mean']:>11.1f} "
            f"{summary['cpu_seconds_total']:>10.4f}"
        )
    print(
        "Legacy detects flips only; "
        "synthetic timings do not establish production speedup."
    )
    print(f"JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
