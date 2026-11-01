import pytest

from .support import TOOLS  # noqa: F401  (puts tools/segments on sys.path)


def test_glass_segments_and_chains():
    pytest.importorskip("pandas")
    pytest.importorskip("tables")
    import compare_glass

    if not (compare_glass.GLASS_DIR / "UCB.h5").exists():
        pytest.skip("GLASS outputs not found; set BG_GLASS_DIR")
    det, chosen = compare_glass.glass_segments(3)
    assert len(chosen) == 3
    for seg in chosen:
        rows = det[det["segment"] == seg]
        assert 8 <= len(rows) <= 20
        chain = compare_glass.glass_chain(det, rows.index[0])
        assert {"Frequency", "Amplitude"} <= set(chain.columns)
