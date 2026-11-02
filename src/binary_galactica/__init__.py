"""Galactic-binary LISA TDI waveforms in JAX.

``ucb_waveform`` uses GLASS-native parameter vectors; ``tdi_waveform`` accepts
physical ``GBParams``. Both return frequency-domain LISA detector channels.
The original ``polarizations`` function remains a source-only interface.
"""

from __future__ import annotations

# Importing the package leaves JAX's process-global configuration alone.
#
# Over a four-year observation the phase of a millihertz binary accumulates to
# ~1e6 radians.  Held in float32 that carries ~0.06 rad of error -- more than
# enough to destroy a matched filter, yet far too small to notice in a plot.
# The float64 functions therefore raise RuntimeError unless the application has
# enabled double precision with jax.config.update("jax_enable_x64", True); they
# never fall back to float32 silently.  binary_galactica.single provides float32
# waveforms, with the frequency split into a reference bin and a small offset,
# for processes that keep x64 disabled.
from binary_galactica import single
from binary_galactica.constants import DAY, YEAR
from binary_galactica.detector import tdi_waveform
from binary_galactica.glass import (
    BACKENDS,
    DEFAULT_RESPONSE_SAMPLES,
    FORMATS,
    FSTAR,
    GLASS_TO_CONTINUOUS,
    LARM,
    SNR_MIN,
    UCB_MODEL_NP,
    Alignment,
    ExtrinsicPriorDraw,
    FisherCache,
    FisherProposal,
    PriorDraw,
    ProposalMix,
    UCBPrior,
    WaveformResult,
    ae_noise,
    analytic_orbits,
    analytic_snr,
    arm_vectors,
    detector_tensor,
    draw_from_fisher,
    fm_shift,
    orbit_cache,
    params_from_native,
    polarization_tensor,
    psi_phi_jump,
    single_source_mix,
    slow_time_grid,
    snr_cut,
    synthesize,
    tdi_response,
    template_on_window,
    to_continuous_ft,
    ucb_alignment,
    ucb_bandwidth,
    ucb_bw,
    ucb_fdot,
    ucb_prior,
    ucb_waveform,
    xyz_to_ae,
    xyz_to_aet,
)
from binary_galactica.parameters import (
    NUM_PARAMS,
    PARAMETER_NAMES,
    GBParams,
    from_glass_vector,
    to_glass_vector,
)
from binary_galactica.waveform import (
    amplitudes,
    frequency,
    phase,
    polarizations,
)

__version__ = "0.6.0"

__all__ = [
    "BACKENDS",
    "DEFAULT_RESPONSE_SAMPLES",
    "tdi_response",
    "DAY",
    "NUM_PARAMS",
    "PARAMETER_NAMES",
    "YEAR",
    "GBParams",
    "amplitudes",
    "frequency",
    "phase",
    "polarizations",
    "FORMATS",
    "FSTAR",
    "GLASS_TO_CONTINUOUS",
    "LARM",
    "SNR_MIN",
    "UCB_MODEL_NP",
    "Alignment",
    "ExtrinsicPriorDraw",
    "FisherCache",
    "FisherProposal",
    "PriorDraw",
    "ProposalMix",
    "UCBPrior",
    "WaveformResult",
    "ae_noise",
    "analytic_orbits",
    "analytic_snr",
    "arm_vectors",
    "detector_tensor",
    "draw_from_fisher",
    "fm_shift",
    "orbit_cache",
    "params_from_native",
    "polarization_tensor",
    "psi_phi_jump",
    "single_source_mix",
    "slow_time_grid",
    "snr_cut",
    "synthesize",
    "template_on_window",
    "to_continuous_ft",
    "ucb_alignment",
    "ucb_bandwidth",
    "ucb_bw",
    "ucb_fdot",
    "ucb_prior",
    "ucb_waveform",
    "xyz_to_ae",
    "xyz_to_aet",
    "tdi_waveform",
    "to_glass_vector",
    "from_glass_vector",
    "single",
]
