import numpy as np
import pytest

from binary_galactica import GBParams
from binary_galactica.parameters import to_glass_vector
from binary_galactica.segments import CATALOGUE_FIELDS, ldc_to_native, native_to_ldc

from .support import ldc_rows

T = 31_536_000.0


def _random_rows(n, seed=1):
    rng = np.random.default_rng(seed)
    rows = np.zeros(n, dtype=[(name, np.float64) for name in CATALOGUE_FIELDS])
    rows["Amplitude"] = 10 ** rng.uniform(-23, -21, n)
    rows["EclipticLatitude"] = np.arcsin(rng.uniform(-1, 1, n))
    rows["EclipticLongitude"] = rng.uniform(0, 2 * np.pi, n)
    rows["Frequency"] = rng.uniform(1e-4, 2e-2, n)
    rows["FrequencyDerivative"] = rng.uniform(-1e-17, 1e-15, n)
    rows["Inclination"] = np.arccos(rng.uniform(-1, 1, n))
    rows["InitialPhase"] = rng.uniform(0.01, 2 * np.pi - 0.01, n)
    rows["Polarization"] = rng.uniform(0, np.pi, n)
    return rows


def test_matches_to_glass_vector():
    rows = _random_rows(50)
    theta = GBParams(
        amp=rows["Amplitude"],
        f0=rows["Frequency"],
        fdot=rows["FrequencyDerivative"],
        phi0=-rows["InitialPhase"],  # LDC InitialPhase is GLASS's phi0_G
        iota=rows["Inclination"],
        psi=rows["Polarization"],
        lam=rows["EclipticLongitude"],
        beta=rows["EclipticLatitude"],
    )
    expected = np.array(to_glass_vector(theta, T))
    expected[:, 6] = np.mod(expected[:, 6], 2 * np.pi)
    np.testing.assert_allclose(ldc_to_native(rows, T), expected, rtol=1e-14, atol=1e-15)


def test_round_trip():
    rows = _random_rows(20, seed=2)
    back = native_to_ldc(ldc_to_native(rows, T), T)
    for name in CATALOGUE_FIELDS:
        np.testing.assert_allclose(back[name], rows[name], rtol=1e-12, err_msg=name)


def test_h5py_shaped_rows_and_folds():
    rows = ldc_rows(Polarization=4.0, EclipticLongitude=7.0, InitialPhase=0.5)
    native = ldc_to_native(rows.reshape(1, 1), T)
    assert native.shape == (1, 8)
    assert native[0, 5] == pytest.approx(4.0 - np.pi)
    assert native[0, 2] == pytest.approx(7.0 - 2 * np.pi)
    assert native[0, 6] == pytest.approx(0.5)


def test_mapping_input():
    row = {name: ldc_rows()[name][0] for name in CATALOGUE_FIELDS}
    np.testing.assert_array_equal(ldc_to_native(row, T), ldc_to_native(ldc_rows(), T))


@pytest.mark.parametrize(
    "bad", [{"Amplitude": 0.0}, {"Frequency": -1e-3}, {"Inclination": np.nan}]
)
def test_invalid_rows_raise(bad):
    with pytest.raises(ValueError):
        ldc_to_native(ldc_rows(**bad), T)


def test_missing_field_raises():
    rows = ldc_rows()
    reduced = rows[[name for name in rows.dtype.names if name != "Polarization"]]
    with pytest.raises(ValueError, match="Polarization"):
        ldc_to_native(reduced, T)
