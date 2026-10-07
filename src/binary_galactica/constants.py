"""Time units for callers, in SI seconds.

``YEAR`` is the astronomical year published by the LISA Science Ground Segment
through :mod:`lisaconstants`. This package does not define its own year. The
GLASS backend keeps a separate pinned literal in
:mod:`binary_galactica.glass.constants`.
"""

from __future__ import annotations

import lisaconstants

#: Astronomical year in seconds, from :mod:`lisaconstants` (LISA SGS).
YEAR: float = lisaconstants.ASTRONOMICAL_YEAR

#: Mean solar day in seconds. :mod:`lisaconstants` does not define a day.
DAY: float = 86400.0

__all__ = ["DAY", "YEAR"]
