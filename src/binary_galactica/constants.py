"""Physical constants and time units, in SI.

Values follow the LISA Data Challenge conventions so that observation times
quoted here agree bit-for-bit with those used by ``ldasoft`` and ``GBGPU``.
"""

from __future__ import annotations

#: Sidereal year in seconds (LDC ``YRSID_SI``).  LISA observation times are
#: conventionally quoted in sidereal years because the constellation's orbit is
#: tied to Earth's, so this -- not the Julian year -- is the natural unit.
YEAR: float = 31_558_149.763_545_603

#: Solar day in seconds.
DAY: float = 86_400.0

__all__ = ["DAY", "YEAR"]
