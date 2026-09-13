"""Physical constants pinned to the selected GLASS revision."""

from __future__ import annotations

import numpy as np
import pytest

from binary_galactica.glass import constants

GLASS_HEADER = {
    "SQ3": 1.73205080757,
    "PI2": 6.283185307179586,
    "CLIGHT": 299792458.0,
    "TSUN": 4.92569043916e-6,
    "PC": 3.0856775815e16,
    "YEAR": 3.15581497632e7,
    "AU": 1.49597870700e11,
    "VEARTH": 0.00010103671,
    "LARM": 2.5e9,
    "SPS": 8.321000e-23,
    "SACC": 9.000000e-30,
    "SLOC": 2.89e-24,
}


@pytest.mark.parametrize("name,value", sorted(GLASS_HEADER.items()))
def test_constant_matches_the_glass_header_literal(name, value):
    assert getattr(constants, name) == value


def test_derived_constants_follow_the_pinned_literals():
    assert np.sqrt(3.0) != constants.SQ3
    assert constants.FSTAR == constants.CLIGHT / (constants.PI2 * constants.LARM)
    assert constants.ECC == constants.LARM / (2.0 * constants.SQ3 * constants.AU)
    assert constants.RORB == constants.AU * constants.ECC
