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
r"""Convert between the raw GLASS response and a continuous Fourier transform.

The two conventions differ by one factor:

.. math::
    h^{\mathrm{cont}} = \sqrt{T}\,h^{\mathrm{GLASS}}.

GLASS's inner product carries no frequency-bin width, while the continuous
Fourier convention uses :math:`\rho^2 = 4\,\delta f\sum_j
|h_j|^2/S(f_j)` with :math:`\delta f=1/T`. Absorbing the bin width into
the template yields exactly :math:`\sqrt{T}`.

The raw waveform follows the ``1/(2*BW)`` reduced-FFT normalization at
GLASS commit ``079c6a3a6fcf380a7c107f39640fff7fc720d0f1`` with full
polarization factors, so ``exp(params[3])`` is the physical strain amplitude.
Neither the response normalization nor that amplitude convention alters this
separate unit conversion: ``sqrt(T)`` times the raw spectrum, divided by the
sample interval, matches the in-band ``rfft`` of Sangria/LDC time-domain TDI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, overload

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .types import WaveformResult


def GLASS_TO_CONTINUOUS(T: ArrayLike) -> Array:
    """Return the multiplier from raw GLASS units to continuous FT units.

    ``T`` may be traced, including when differentiating observation duration.
    """
    return jnp.sqrt(T)


#: Keys that index bins rather than carry amplitude, and so are not rescaled.
_INDEX_KEYS = frozenset({"qmin", "freq_bin_indices"})


# TypedDict is a Mapping but not a dict to type checkers. The overlap is safe:
# the same runtime dict preserves every WaveformResult field and satisfies
# the generic mapping overload's dictionary contract.
@overload
def to_continuous_ft(  # type: ignore[overload-overlap]
    waveform: WaveformResult, T: ArrayLike
) -> WaveformResult: ...


@overload
def to_continuous_ft(
    waveform: Mapping[str, ArrayLike], T: ArrayLike
) -> dict[str, ArrayLike]: ...


def to_continuous_ft(
    waveform: WaveformResult | Mapping[str, ArrayLike], T: ArrayLike
) -> WaveformResult | dict[str, ArrayLike]:
    """Rescale a native waveform result to continuous Fourier units.

    Bin-index fields are passed through unchanged. A ``WaveformResult``
    retains its named field types; partial channel mappings return a dict.
    """
    factor = GLASS_TO_CONTINUOUS(T)
    # Every WaveformResult field is an Array; expose that common value type
    # when iterating instead of TypedDict.items()'s conservative object type.
    values = cast(Mapping[str, ArrayLike], waveform)
    return {
        key: value if key in _INDEX_KEYS else factor * value
        for key, value in values.items()
    }


__all__ = ["GLASS_TO_CONTINUOUS", "to_continuous_ft"]
