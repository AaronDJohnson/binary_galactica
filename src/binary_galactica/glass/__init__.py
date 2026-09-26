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
"""GLASS-compatible galactic-binary TDI waveforms in JAX.

The default waveform constructs first-generation frozen-geometry XYZ directly
at sparse reception times, interpolates its demodulated analytic response, and
then Fourier transforms it. It follows the pinned GLASS
``LISA_spline_response`` / ``LISA_TDI_spline`` path from commit
``079c6a3a6fcf380a7c107f39640fff7fc720d0f1``. The explicit
``backend="fourier"`` option retains GLASS's earlier fast-slow / reduced-DFT
link-response construction for legacy reference comparisons.

Quick start
-----------
>>> import binary_galactica.glass
>>> T, BW = 7884000.0, 64
>>> p = binary_galactica.glass.params_from_native(
...     f0=1e-3, fdot=1e-18, amp=7.16e-22, phi0=0.5,
...     colat=1.78, lon=0.0, psi=0.7, cos_iota=0.3, T=T)
>>> w = binary_galactica.glass.ucb_waveform(p, T=T, BW=BW)
>>> w["A"].shape
(64,)

The returned spectrum is in GLASS's amplitude convention and occupies
absolute frequency bins ``qmin ... qmin + BW - 1``. Use
:func:`binary_galactica.glass.convert.to_continuous_ft` for the continuous-Fourier
convention used elsewhere in this repository.
"""

from __future__ import annotations

from .alignment import Alignment, ucb_alignment
from .bandwidth import ae_noise, analytic_snr, ucb_bandwidth, ucb_bw
from .compat import template_on_window
from .constants import FSTAR, LARM, UCB_MODEL_NP
from .convert import GLASS_TO_CONTINUOUS, to_continuous_ft
from .orbit import analytic_orbits, arm_vectors, slow_time_grid
from .prior import SNR_MIN, UCBPrior, snr_cut, ucb_fdot, ucb_prior
from .proposals import (
    ExtrinsicPriorDraw,
    FisherCache,
    FisherProposal,
    PriorDraw,
    ProposalMix,
    draw_from_fisher,
    fm_shift,
    psi_phi_jump,
    single_source_mix,
)
from .response import detector_tensor, polarization_tensor
from .tdi import FORMATS, synthesize, xyz_to_ae, xyz_to_aet
from .types import WaveformResult
from .waveform import (
    BACKENDS,
    DEFAULT_RESPONSE_SAMPLES,
    orbit_cache,
    params_from_native,
    tdi_response,
    ucb_waveform,
)

__all__ = [
    "BACKENDS",
    "DEFAULT_RESPONSE_SAMPLES",
    "tdi_response",
    "Alignment",
    "FORMATS",
    "FSTAR",
    "GLASS_TO_CONTINUOUS",
    "LARM",
    "UCB_MODEL_NP",
    "WaveformResult",
    "ae_noise",
    "analytic_orbits",
    "analytic_snr",
    "arm_vectors",
    "detector_tensor",
    "orbit_cache",
    "params_from_native",
    "ExtrinsicPriorDraw",
    "FisherCache",
    "FisherProposal",
    "PriorDraw",
    "ProposalMix",
    "SNR_MIN",
    "UCBPrior",
    "draw_from_fisher",
    "fm_shift",
    "polarization_tensor",
    "psi_phi_jump",
    "single_source_mix",
    "slow_time_grid",
    "snr_cut",
    "synthesize",
    "ucb_fdot",
    "ucb_prior",
    "template_on_window",
    "to_continuous_ft",
    "ucb_alignment",
    "ucb_bandwidth",
    "ucb_bw",
    "ucb_waveform",
    "xyz_to_ae",
    "xyz_to_aet",
]
