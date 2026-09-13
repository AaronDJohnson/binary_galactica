"""Physical constants and time units, in SI.

Values follow the LISA Data Challenge conventions. The GLASS backend keeps
its own pinned literals in :mod:`binary_galactica.glass.constants`; its
sidereal-year literal differs slightly from the physical unit exposed here.
"""

from __future__ import annotations

#: Sidereal year in seconds (LDC ``YRSID_SI``).  LISA observation times are
#: conventionally quoted in sidereal years because the constellation's orbit is
#: tied to Earth's, so this -- not the Julian year -- is the natural unit.
YEAR: float = 31_558_149.763_545_603

#: Solar day in seconds.
DAY: float = 86_400.0

__all__ = ["DAY", "YEAR"]
