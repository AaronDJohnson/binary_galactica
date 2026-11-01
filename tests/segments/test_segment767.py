import numpy as np
import pytest

from .support import SANGRIA, requires_sangria


@requires_sangria
def test_start_rows_match_the_npz():
    import segment767
    import sangria

    if not segment767.NPZ.exists():
        pytest.skip("Segment767 npz not found; set BG_SEGMENT767")
    n, dt = sangria.grid(SANGRIA)
    rows = segment767.start_rows(SANGRIA, n * dt)
    freqs = np.load(segment767.NPZ)["initial_sources_native"][:, 0]
    assert rows.shape[0] == 18 and len(set(rows["Name"])) == 18
    np.testing.assert_allclose(np.sort(rows["Frequency"]), np.sort(freqs), rtol=1e-12)
