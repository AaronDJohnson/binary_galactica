import numpy as np

from binary_galactica.segments import CATALOGUE_FIELDS, ldc_to_native

from .support import SANGRIA, requires_sangria


@requires_sangria
def test_bright_verification_binaries_match():
    import conventions_gate
    import sangria

    n, dt = sangria.grid(SANGRIA)
    T, n_freq = n * dt, n // 2 + 1
    spectra = sangria.aet_spectra(*sangria.load_xyz("vgb", SANGRIA))
    rows = sangria.load_catalogue("vgb", ("Name", *CATALOGUE_FIELDS), SANGRIA)
    native = ldc_to_native(rows, T)
    names = [name.decode() for name in rows["Name"]]
    for name in ("HPLib", "SDSSJ0651", "V407Vul"):
        j = names.index(name)
        for backend in ("fourier", "onfly"):
            result = conventions_gate.match_source(native[j], spectra, T, n_freq, backend)
            assert result["match"] > 0.999, (name, backend, result)
            scale = complex(*result["scale"])
            assert abs(scale - 1.0) < 2e-3, (name, backend, result)
            assert np.isfinite(result["match"])
