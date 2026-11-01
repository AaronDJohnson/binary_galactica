"""GLASS-style sampling of galactic-binary frequency segments.

One :class:`UCBBlock` holds every segment of the band and implements
enchilada's ``Block`` protocol structurally (enchilada is not imported).
Trans-dimensional sampling uses impulse's ``RJPTSampler``, imported lazily.
"""

from __future__ import annotations

from .catalogue import CATALOGUE_FIELDS, ldc_to_native, native_to_ldc
from .config import BACKENDS, UCBBlockConfig
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
    "Segment",
    "SegmentTable",
    "UCBBlockConfig",
    "build_segment_table",
    "doppler_bins",
    "ldc_to_native",
    "native_to_ldc",
    "padding_bins",
]
