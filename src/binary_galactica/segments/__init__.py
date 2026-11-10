"""GLASS-style sampling of galactic-binary frequency segments.

One :class:`UCBBlock` holds every segment of the band and implements
enchilada's ``Block`` protocol structurally (enchilada is not imported).
Trans-dimensional sampling uses impulse's ``RJPTSampler``, imported lazily.
"""

from __future__ import annotations

from .block import CHANNEL_SETS, UCBBlock
from .catalogue import CATALOGUE_FIELDS, ldc_to_native, native_to_ldc
from .config import BACKENDS, UCBBlockConfig
from .executor import (
    AdvanceResult,
    ProcessExecutor,
    RunnerSpec,
    SerialExecutor,
    assign_shards,
)
from .fisher import FisherBank, fisher_decomposition, floor_eigenvalues
from .likelihood import SegmentLikelihood, quantize_bins
from .moves import FisherMove, FmShiftMove, PriorRedrawMove, PsiPhiMove, SlotMove
from .prior import SegmentPrior
from .render import Contribution, render_sources
from .runner import COMPONENT_PERIODIC, ChainArchive, SegmentRunner, glass_ladder
from .table import (
    Segment,
    SegmentTable,
    build_segment_table,
    doppler_bins,
    padding_bins,
)

__all__ = [
    "AdvanceResult",
    "BACKENDS",
    "CATALOGUE_FIELDS",
    "CHANNEL_SETS",
    "COMPONENT_PERIODIC",
    "ChainArchive",
    "Contribution",
    "FisherBank",
    "FisherMove",
    "FmShiftMove",
    "PriorRedrawMove",
    "ProcessExecutor",
    "PsiPhiMove",
    "RunnerSpec",
    "Segment",
    "SegmentLikelihood",
    "SegmentPrior",
    "SegmentRunner",
    "SegmentTable",
    "SerialExecutor",
    "SlotMove",
    "UCBBlock",
    "UCBBlockConfig",
    "assign_shards",
    "build_segment_table",
    "doppler_bins",
    "fisher_decomposition",
    "floor_eigenvalues",
    "glass_ladder",
    "ldc_to_native",
    "native_to_ldc",
    "padding_bins",
    "quantize_bins",
    "render_sources",
]
