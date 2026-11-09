"""The isolated-source tool folds exact symmetries and runs a short chain."""

import numpy as np
import pytest

from .support import rj_impulse


def test_fold_offsets_exposes_only_the_exact_symmetry():
    from isolated_source import fold_offsets

    d = np.zeros((2, 8))
    d[0, 5], d[0, 6] = np.pi / 2, np.pi  # exact (psi + pi/2, phi0 + pi) image
    d[1, 5] = np.pi / 2  # psi alone is not a symmetry
    out = fold_offsets(d)
    assert np.allclose(out[0], 0.0, atol=1e-12)
    assert abs(out[1, 5]) < 1e-12 and abs(abs(out[1, 6]) - np.pi) < 1e-12


@pytest.mark.slow
def test_short_isolated_run(tmp_path):
    rj_impulse()
    pytest.importorskip("h5py")
    import isolated_source

    report = isolated_source.run(turns=2, steps=200, ntemps=4, outdir=tmp_path)
    offset = np.asarray(report["mean_offset"]) / np.asarray(report["fisher_sigma"])
    assert np.all(np.abs(offset) < 6), offset
    assert np.isfinite(report["seconds_per_step"])
