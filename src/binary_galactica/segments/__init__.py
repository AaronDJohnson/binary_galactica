"""GLASS-style sampling of galactic-binary frequency segments.

One :class:`UCBBlock` holds every segment of the band and implements
enchilada's ``Block`` protocol structurally (enchilada is not imported).
Trans-dimensional sampling uses impulse's ``RJPTSampler``, imported lazily.
"""

from __future__ import annotations

from .catalogue import CATALOGUE_FIELDS, ldc_to_native, native_to_ldc
from .config import BACKENDS, UCBBlockConfig
from .fisher import FisherBank, fisher_decomposition, floor_eigenvalues
from .likelihood import SegmentLikelihood, quantize_bins
from .moves import FisherMove, FmShiftMove, PriorRedrawMove, PsiPhiMove, SlotMove
from .prior import SegmentPrior
from .runner import COMPONENT_PERIODIC, ChainArchive, SegmentRunner, glass_ladder
from .table import (
    Segment,
    SegmentTable,
    build_segment_table,
    doppler_bins,
    padding_bins,
)

__all__ = [
    "BACKENDS",
    "CATALOGUE_FIELDS",
    "COMPONENT_PERIODIC",
    "ChainArchive",
    "FisherBank",
    "FisherMove",
    "FmShiftMove",
    "PriorRedrawMove",
    "PsiPhiMove",
    "Segment",
    "SegmentLikelihood",
    "SegmentPrior",
    "SegmentRunner",
    "SegmentTable",
    "SlotMove",
    "UCBBlockConfig",
    "build_segment_table",
    "doppler_bins",
    "fisher_decomposition",
    "floor_eigenvalues",
    "glass_ladder",
    "ldc_to_native",
    "native_to_ldc",
    "padding_bins",
    "quantize_bins",
]
