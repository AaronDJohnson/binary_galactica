"""Mandatory comparison with independently compiled, pinned GLASS C outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data" / "glass"
CHANNELS = ("X", "Y", "Z", "A", "E")


def load_reference_cases():
    """Return (metadata, arrays) pairs; arrays contain X/Y/Z/A/E and bins."""
    metadata = json.loads((DATA / "reference.json").read_text())
    if not metadata["cases"]:
        raise ValueError("The mandatory GLASS reference cases are missing")
    digest = hashlib.sha256((DATA / "reference.npz").read_bytes()).hexdigest()
    if digest != metadata["npz_sha256"]:
        raise ValueError("GLASS reference fixture checksum mismatch")
    with np.load(DATA / "reference.npz", allow_pickle=False) as archive:
        return [
            (case, {key: archive[f"{case['id']}_{key}"] for key in (*CHANNELS, "bins")})
            for case in metadata["cases"]
        ]


REFERENCE_CASES = load_reference_cases()


def test_reference_corpus_covers_varied_sources():
    """Keep the C comparison from shrinking back to a single orientation."""
    varied = [case for case, _ in REFERENCE_CASES if case["id"].startswith("varied_")]
    sources = {tuple(case["params"][1:7]) for case in varied}
    assert len(sources) >= 6
    for index in range(6):
        assert len({source[index] for source in sources}) >= 6
    for source in sources:
        formats = {
            case["fmt"] for case in varied if tuple(case["params"][1:7]) == source
        }
        assert formats == {"sangria", "phase", "frequency"}
    assert {len(case["params"]) for case in varied} == {7, 8, 9}
    assert {case["BW"] for case in varied} == {32, 96, 512}
    assert {case["T"] for case in varied} == {7_884_000.0, 31_536_000.0, 126_144_000.0}
    chirps = [case["params"][7] for case in varied if len(case["params"]) >= 8]
    assert min(chirps) < 0 < max(chirps)
    second_derivatives = [
        case["params"][8] for case in varied if len(case["params"]) == 9
    ]
    assert min(second_derivatives) < 0 < max(second_derivatives)


@pytest.mark.parametrize(
    "case,reference",
    REFERENCE_CASES,
    ids=[case["id"] for case, _ in REFERENCE_CASES],
)
def test_native_waveform_matches_glass_c(case, reference):
    # A missing backend or fixture is a failure, never an optional skipped test.
    from binary_galactica.glass import ucb_waveform

    actual = ucb_waveform(
        np.asarray(case["params"]),
        T=case["T"],
        BW=case["BW"],
        t0=case["t0"],
        fmt=case["fmt"],
        backend="fourier",
    )
    assert int(actual["qmin"]) == case["qmin"]
    assert int(actual["qmin"]) + case["BW"] // 2 == case["q"]
    np.testing.assert_array_equal(actual["freq_bin_indices"], reference["bins"])
    for channel in CHANNELS:
        expected = reference[channel]
        result = np.asarray(actual[channel])
        assert result.shape == expected.shape == (case["BW"],)
        assert np.all(np.isfinite(result))
        scale = np.linalg.norm(expected)
        assert scale > 0
        relative_error = np.linalg.norm(result - expected) / scale
        assert relative_error <= 1e-9, (case["id"], channel, relative_error)
