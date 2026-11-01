from types import SimpleNamespace

import numpy as np
import pytest

from .support import SANGRIA, requires_sangria


@requires_sangria
def test_target_band():
    import runs
    import sangria

    n, dt = sangria.grid(SANGRIA)
    T = n * dt
    core = (113993, 114068)
    base = runs.catalogue_rows(SANGRIA, T=T, band=(core[0], core[1]))
    band, rows = runs.target_band(SANGRIA, T=T, core=core, span=100, extra=base)
    assert band == (core[0] - 100, core[1] + 100)
    q = rows["Frequency"] * T
    assert np.all((q >= band[0]) & (q < band[1]))
    assert np.all(np.diff(rows["Frequency"]) >= 0)
    assert len(set(rows["Name"])) == len(rows)
    assert set(base["Name"]) <= set(rows["Name"])
    _, plain = runs.target_band(SANGRIA, T=T, core=core, span=100)
    assert len(plain) <= len(rows)
    # every row not from ``extra`` meets the SNR cut
    snr = runs.analytic_snrs(rows["Frequency"], rows["Amplitude"], T)
    assert np.all(snr[~np.isin(rows["Name"], base["Name"])] >= 7.0)


def test_inner_segments():
    import runs

    segs = [SimpleNamespace(index=i, lo=10 * i, hi=10 * i + 10) for i in range(4)]
    block = SimpleNamespace(table=SimpleNamespace(segments=segs))
    assert [s.index for s in runs.inner_segments(block, (12, 28))] == [1, 2]
    with pytest.raises(ValueError):
        runs.inner_segments(block, (5, 28))
    with pytest.raises(ValueError):
        runs.inner_segments(block, (12, 35))
    with pytest.raises(ValueError):
        runs.inner_segments(SimpleNamespace(table=SimpleNamespace(segments=segs[:2])), (0, 1))
