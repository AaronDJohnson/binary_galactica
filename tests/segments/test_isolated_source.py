import numpy as np

from .support import rj_impulse


def test_short_isolated_run(tmp_path):
    rj_impulse()
    import isolated_source

    report = isolated_source.run(turns=2, steps=200, ntemps=4, outdir=tmp_path)
    offset = np.asarray(report["mean_offset"]) / np.asarray(report["fisher_sigma"])
    assert np.all(np.abs(offset) < 6), offset
    assert np.isfinite(report["seconds_per_step"])
