"""Regression checks for the synthetic comparison and its reported costs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .benchmark import Case, legacy_flips, make_cases, run_case, summarize


@pytest.mark.parametrize(
    ("root", "gap", "expected"),
    [
        (1.5, 0.0, ()),  # Equal adjacent minima do not satisfy strict inequality.
        (2.2, 0.0, ()),  # The final sample inherits the penultimate sample's flip.
        (1.2, 0.0, (2,)),  # An otherwise identical interior flip survives.
        (1.2, 0.1, (2,)),  # This response never vanishes but curvature accepts it.
    ],
)
def test_legacy_baseline_matches_known_extractor_edge_cases(root, gap, expected):
    """Preserve documented old behavior instead of silently improving the baseline."""
    assert (
        legacy_flips(lambda t: complex(t - root, gap), (0.0, 1.0, 2.0, 3.0)) == expected
    )


def test_baseline_near_miss_is_a_false_crossing_without_a_zero_time():
    """The report must not award a root estimate to a flip-only heuristic."""
    case = Case(
        name="known_false_flip",
        kind="near_miss",
        interval=(0.0, 3.0),
        amplitude=3.0,
        root_fraction=0.4,
        nodes=(0.0, 1 / 3, 2 / 3, 1.0),
        gap=1 / 30,
    )
    result = run_case(case, "legacy_curvature", 1e-8, 1e-10)
    assert result["kind"] == "crossing"
    assert result["flip_indices"] == [2]
    assert result["crossing_correct"] is False
    assert result["classification_correct"] is None
    assert result["time"] is None
    assert result["time_error"] is None
    assert result["time_within_tolerance"] is None
    assert result["evaluations"] == 4


def test_seeded_shapes_repeat_and_amplitude_pairs_preserve_the_response():
    """Changing a seed changes random shapes while paired scales share geometry."""
    first = make_cases(seed=314159, random_cases=3)
    repeated = make_cases(seed=314159, random_cases=3)
    changed = make_cases(seed=271828, random_cases=3)
    assert first == repeated
    fixed = [case for case in first if not case.name.startswith("seeded_")]
    assert fixed == [case for case in changed if not case.name.startswith("seeded_")]
    seeded = [case for case in first if case.name.startswith("seeded_")]
    assert seeded != [case for case in changed if case.name.startswith("seeded_")]
    assert [case.kind for case in seeded] == [
        "crossing",
        "crossing",
        "touch",
        "touch",
        "near_miss",
        "near_miss",
    ]
    for large, small in zip(seeded[::2], seeded[1::2], strict=True):
        assert (large.amplitude, small.amplitude) == (1.0, 1e-22)
        assert large.interval == small.interval
        assert large.nodes == small.nodes
        assert large.root_fraction == small.root_fraction
        time = large.interval[0] + 0.77 * large.width
        assert small.response(time) / 1e-22 == pytest.approx(large.response(time))


@pytest.mark.parametrize("method", ["bounded", "brentq", "toms748"])
def test_run_case_reports_actual_callback_cost_including_classification(method):
    """Observe a real solve independently of the benchmark's callback wrapper."""
    calls = []

    class ObservedCase(Case):
        """Record real function evaluations without replacing the solver."""

        def response(self, t):
            """Record the time and evaluate the analytic response."""
            calls.append(t)
            return super().response(t)

    case = ObservedCase(
        name="observed_crossing",
        kind="crossing",
        interval=(0.0, 1.0),
        amplitude=1e-22,
        root_fraction=0.37,
        nodes=(0.0, 0.25, 0.5, 0.75, 1.0),
        rotation=0.3,
    )
    result = run_case(case, method, 1e-8, 1e-10)
    assert result["kind"] == "crossing"
    assert result["time"] == pytest.approx(0.37, abs=1e-7)
    assert result["evaluations"] == len(calls)
    assert result["reported_evaluations"] == len(calls)
    assert result["time_within_tolerance"] is True
    assert result["evaluations"] > 3
    assert {0.0, 0.5, 1.0}.issubset(calls)
    assert any(0.0 < t < result["time"] for t in calls)
    assert any(result["time"] < t < 1.0 for t in calls)


def test_localization_accuracy_includes_unconverged_true_zero_estimates():
    """A failed solve must remain in the localization-accuracy denominator."""
    case = Case(
        name="known_crossing",
        kind="crossing",
        interval=(0.0, 1.0),
        amplitude=1.0,
        root_fraction=0.37,
        nodes=(0.0, 0.5, 1.0),
    )
    accurate = run_case(case, "bounded", 1e-8, 1e-12)
    assert accurate["time_within_tolerance"] is True
    inaccurate = {
        **accurate,
        "name": "failed_true_zero",
        "kind": "unresolved",
        "classification_correct": False,
        "crossing_correct": False,
        "converged": False,
        "time": 0.38,
        "time_error": 0.01,
        "time_error_fraction": 0.01,
        "time_within_tolerance": False,
    }
    summary = summarize([accurate, inaccurate])
    assert summary["time_accuracy_fraction"] == 0.5
    assert summary["time_error_fraction_max"] == 0.01


@pytest.mark.parametrize("time_fraction_atol", [None, 1e-9])
def test_cli_writes_costs_and_separates_flip_detection_from_root_accuracy(
    tmp_path, time_fraction_atol
):
    """Exercise the real entry point and a complete small deterministic report."""
    output = tmp_path / "results" / "benchmark.json"
    command = [
        sys.executable,
        "-m",
        "tools.tdi_zero_crossings.benchmark",
        "--output",
        str(output),
        "--seed",
        "314159",
        "--random-cases",
        "0",
        "--repeats",
        "2",
    ]
    if time_fraction_atol is not None:
        command.extend(["--time-fraction-atol", str(time_fraction_atol)])
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text())
    assert "Synthetic candidates: 56; repeats: 2" in completed.stdout
    assert "do not establish production speedup" in completed.stdout
    assert report["configuration"]["seed"] == 314159
    expected_tolerance = 1e-12 if time_fraction_atol is None else time_fraction_atol
    assert report["configuration"]["time_fraction_atol"] == expected_tolerance
    assert report["environment"]["scipy"]
    baseline = report["summary"]["legacy_curvature"]
    assert baseline["runs"] == 112
    assert baseline["evaluations_total"] == 560
    assert baseline["evaluations_mean"] == 5.0
    assert baseline["classification_accuracy"] is None
    assert baseline["time_error_max"] is None
    assert baseline["time_accuracy_fraction"] is None
    for method in ("bounded", "brentq", "toms748"):
        rows = report["results"][method]
        assert len(rows) == 112
        assert {row["repeat"] for row in rows} == {0, 1}
        assert all(row["evaluations"] == row["reported_evaluations"] for row in rows)
        assert report["summary"][method]["evaluations_total"] > 0
        assert report["summary"][method]["cpu_seconds_total"] > 0
