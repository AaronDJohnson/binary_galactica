# Copyright 2019 Tyson B. Littenberg & Neil J. Cornish
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Physical constants, copied digit-for-digit from GLASS.

Every value here is a literal transcription of a ``#define`` at GLASS commit
``079c6a3a6fcf380a7c107f39640fff7fc720d0f1``, with the defining file and
line recorded. The package-local tests pin these literals and the independent
C waveform fixtures exercise the resulting response.

Sources
-------
``glass/utils/src/glass_constants.h``
``glass/utils/src/glass_lisa.h``
"""

from __future__ import annotations

import numpy as np

# --- glass_constants.h -----------------------------------------------------
SQ3 = 1.73205080757  # :31   NOT sqrt(3); GLASS truncates, and it shows.
PI2 = 6.283185307179586  # :40
CLIGHT = 299792458.0  # :54   [m/s]
TSUN = 4.92569043916e-6  # :60   [s]
PC = 3.0856775815e16  # :66   [m]
YEAR = 3.15581497632e7  # :69   [s]  sidereal year
AU = 1.49597870700e11  # :78   [m]
VEARTH = 0.00010103671  # :89   v_Earth / c

# --- glass_lisa.h ----------------------------------------------------------
LARM = 2.5e9  # :31   [m]  arm length
SPS = 8.321000e-23  # :43   position (shot) noise
SACC = 9.000000e-30  # :46   acceleration noise
SLOC = 2.89e-24  # :49   local oscillator noise

# --- derived (initialize_analytic_orbit, glass_lisa.c:~128) ----------------
FSTAR = CLIGHT / (PI2 * LARM)  # transfer frequency [Hz], ~19.09 mHz
ECC = LARM / (2.0 * SQ3 * AU)  # constellation eccentricity
RORB = AU * ECC  # orbit->R, the per-spacecraft offset scale [m]

SQRT2 = np.sqrt(2.0)
SQRT6 = np.sqrt(6.0)
INV_SQRT2 = 0.70710678118654752440  # C's M_SQRT1_2, used by XYZ2AE
INV_SQ6 = 0.408248290463863  # XYZ2AE, glass_lisa.c
INV_SQ3 = 0.577350269189626  # XYZ2AET, glass_lisa.c

#: Number of parameters in the GLASS UCB sampling vector.
UCB_MODEL_NP = 8

__all__ = [
    "AU",
    "CLIGHT",
    "ECC",
    "FSTAR",
    "INV_SQ3",
    "INV_SQ6",
    "LARM",
    "PC",
    "PI2",
    "INV_SQRT2",
    "RORB",
    "SACC",
    "SLOC",
    "SPS",
    "SQ3",
    "SQRT2",
    "SQRT6",
    "TSUN",
    "UCB_MODEL_NP",
    "VEARTH",
    "YEAR",
]
