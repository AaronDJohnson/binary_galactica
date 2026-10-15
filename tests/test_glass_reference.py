"""Mandatory comparison with independently compiled, pinned GLASS C outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data" / "glass"
CHANNELS = ("X", "Y", "Z", "A", "E")
#: The Fourier backend relative to the unscaled 079c6a3 C spectra. GLASS
#: b0751bd halved Aplus and Across in the fast-slow ``ucb_waveform`` while
#: keeping ``invBW2 = 1/(2*BW)``, so the pinned C output is half the physical
#: response for the amplitude GLASS's ``amplitude()`` returns. The backend
#: restores the full factors (GLASS <= 95184e1, LDC); everything else matches.
GLASS_079C6A3_FAST_SLOW_SCALE = 2.0


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


#: The C cases compared on every run, one per regime of the 36-case corpus
#: (whose checksum load_reference_cases verifies in full). Each parameter
#: count is paired with each output format; between them they also span the
#: three durations and bandwidths, a zero and a nonzero epoch, both signs of
#: each frequency derivative, 1-10 mHz, and all six varied orientations.
COMPARED = (
    "varied_north_7p_four_years_bw96_sangria",  # nearly face-on
    "varied_south_7p_quarter_year_bw32_phase",  # t0 = 0
    "5mhz_7p_bw32_frequency",
    "carrier_roundtrip_below_integer",  # carrier rounding on either side
    "carrier_just_above_integer",  # of an integer bin
    "varied_north_8p_negative_chirp_year_bw96_phase",
    "varied_south_8p_positive_chirp_quarter_year_bw512_frequency",  # edge-on
    "varied_equatorial_9p_four_years_bw512_sangria",  # falling fddot
    "varied_oblique_9p_negative_chirp_year_bw32_phase",  # rising fddot
    "10mhz_9p_bw512_frequency",  # highest frequency
)
COMPARED_CASES = [
    (case, arrays) for case, arrays in load_reference_cases() if case["id"] in COMPARED
]


def test_compared_cases_span_the_reference_regimes():
    """Keep the C comparison from shrinking back to a single regime."""
    cases = [case for case, _ in COMPARED_CASES]
    assert sorted(case["id"] for case in cases) == sorted(COMPARED)
    formats = ("sangria", "phase", "frequency")
    assert {(len(case["params"]), case["fmt"]) for case in cases} == {
        (count, fmt) for count in (7, 8, 9) for fmt in formats
    }
    assert {case["BW"] for case in cases} == {32, 96, 512}
    assert {case["T"] for case in cases} == {7_884_000.0, 31_536_000.0, 126_144_000.0}
    epochs = [case["t0"] for case in cases]
    assert min(epochs) == 0 < max(epochs)
    for index in (7, 8):  # fdot and fddot
        values = [
            case["params"][index] for case in cases if len(case["params"]) > index
        ]
        assert min(values) < 0 < max(values)
    frequencies = [case["params"][0] / case["T"] for case in cases]
    assert min(frequencies) < 2e-3 and max(frequencies) >= 1e-2
    varied = {
        tuple(case["params"][1:7]) for case in cases if case["id"].startswith("varied_")
    }
    assert len(varied) == 6
    for index in range(6):
        assert len({source[index] for source in varied}) == 6


@pytest.mark.parametrize(
    "case,reference",
    COMPARED_CASES,
    ids=[case["id"] for case, _ in COMPARED_CASES],
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
        expected = GLASS_079C6A3_FAST_SLOW_SCALE * reference[channel]
        result = np.asarray(actual[channel])
        assert result.shape == expected.shape == (case["BW"],)
        assert np.all(np.isfinite(result))
        scale = np.linalg.norm(expected)
        assert scale > 0
        relative_error = np.linalg.norm(result - expected) / scale
        assert relative_error <= 1e-9, (case["id"], channel, relative_error)
