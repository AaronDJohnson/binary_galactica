"""Settings of the UCB segment block, written verbatim to each run directory."""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from ..glass import BACKENDS, DEFAULT_RESPONSE_SAMPLES, SNR_MIN

# BACKENDS is re-exported from the glass package. The conventions gate tied
# (0.99696 for both backends) and the user chose "onfly" as the default.

_POSITIVE_INTS = (
    "n_target",
    "min_interior_bins",
    "k_max_floor",
    "steps_per_turn",
    "thin",
    "ntemps",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class UCBBlockConfig:
    """Every setting of :class:`~binary_galactica.segments.UCBBlock`.

    Defaults follow the design spec of 2026-10-31:
    - GLASS's twelve-chain ladder ``1.2**i``, with the hottest chain at ``1e12``;
    - 1000 steps per Gibbs turn;
    - about 25% birth/death moves;
    - the within-model mix of Fisher 0.5, with prior redraw, fm shift and
      psi-phi jump at 1/6 each.

    ``band_bins`` is an explicit half-open interior band ``[lo, hi)`` in rfft
    bins; ``None`` covers the start set. Each segment's maximum source count
    is ``K = max(k_max_floor, 2 * owned start sources)``; its count prior is
    uniform on ``0 .. K``.

    Two tiling options are off by default, which tiles the whole band:
    - ``skip_gaps`` leaves the stretches with no start sources unsampled: the
      start sources are split into clusters and only the clusters are tiled.
      No segment covers a gap, so reversible jump cannot add a source there.
    - ``max_interior_bins`` is a soft cap on interior width, at least
      ``min_interior_bins``: wider interiors are split before the merge pass,
      which may widen a piece past the cap where the padding rule needs it.

    See :mod:`binary_galactica.segments.table` for both rules.
    """

    n_target: int = 10
    min_interior_bins: int = 16
    band_bins: tuple[int, int] | None = None
    skip_gaps: bool = False
    max_interior_bins: int | None = None
    k_max_floor: int = 10
    steps_per_turn: int = 1000
    thin: int = 1
    ntemps: int = 12
    ladder_step: float = 1.2
    hottest_temperature: float = 1.0e12
    birth_death: bool = True
    birth_death_fraction: float = 0.25
    fisher_weight: float = 0.5
    prior_weight: float = 1.0 / 6.0
    fm_shift_weight: float = 1.0 / 6.0
    psi_phi_weight: float = 1.0 / 6.0
    am_weight: float = 0.0
    scam_weight: float = 0.0
    de_weight: float = 0.0
    fisher_match_bins: float = 4.0
    snr_min: float = SNR_MIN
    backend: str = "onfly"
    response_samples: int | None = DEFAULT_RESPONSE_SAMPLES

    def __post_init__(self) -> None:
        """Validate every field; raise ``ValueError`` on the first bad one."""
        for name in _POSITIVE_INTS:
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        if self.thin > self.steps_per_turn:
            raise ValueError(
                f"thin must not exceed steps_per_turn={self.steps_per_turn}, "
                f"got {self.thin}"
            )
        if self.band_bins is not None:
            lo, hi = self.band_bins
            if not (_is_int(lo) and _is_int(hi) and 0 < lo < hi):
                raise ValueError(
                    f"band_bins must be integers 0 < lo < hi, got {self.band_bins!r}"
                )
            object.__setattr__(self, "band_bins", (int(lo), int(hi)))
        if not isinstance(self.skip_gaps, bool):
            raise ValueError("skip_gaps must be a bool")
        if self.max_interior_bins is not None:
            cap = self.max_interior_bins
            if not _is_int(cap) or cap < self.min_interior_bins:
                raise ValueError(
                    "max_interior_bins must be None or an integer >= "
                    f"min_interior_bins={self.min_interior_bins}, got {cap!r}"
                )
            object.__setattr__(self, "max_interior_bins", int(cap))
        if not (math.isfinite(self.ladder_step) and self.ladder_step > 1.0):
            raise ValueError("ladder_step must be finite and greater than 1")
        floor = self.ladder_step ** max(self.ntemps - 2, 0)
        if not (
            math.isfinite(self.hottest_temperature)
            and self.hottest_temperature >= floor
        ):
            raise ValueError(
                "hottest_temperature must be finite and at least "
                f"ladder_step**(ntemps - 2) = {floor:g}"
            )
        if not isinstance(self.birth_death, bool):
            raise ValueError("birth_death must be a bool")
        if not 0.0 <= self.birth_death_fraction < 1.0:
            raise ValueError("birth_death_fraction must lie in [0, 1)")
        within = self.within_model_weights()
        weights = [*within.values(), self.am_weight, self.scam_weight, self.de_weight]
        if any(not (math.isfinite(w) and w >= 0.0) for w in weights):
            raise ValueError("move weights must be finite and non-negative")
        if sum(within.values()) <= 0.0:
            raise ValueError("at least one within-model move needs a positive weight")
        if not (math.isfinite(self.fisher_match_bins) and self.fisher_match_bins > 0):
            raise ValueError("fisher_match_bins must be finite and positive")
        if not (math.isfinite(self.snr_min) and self.snr_min >= 0.0):
            raise ValueError("snr_min must be finite and non-negative")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.response_samples is not None and (
            not _is_int(self.response_samples) or self.response_samples < 4
        ):
            raise ValueError("response_samples must be None or an integer >= 4")
        if self.response_samples is not None:
            object.__setattr__(self, "response_samples", int(self.response_samples))

    def within_model_weights(self) -> dict[str, float]:
        """Return the relative weights of the four within-model moves."""
        return {
            "fisher": self.fisher_weight,
            "prior": self.prior_weight,
            "fm_shift": self.fm_shift_weight,
            "psi_phi": self.psi_phi_weight,
        }

    def birth_death_weight(self) -> float:
        """Return the weight making birth/death ``birth_death_fraction`` of moves."""
        if not self.birth_death:
            return 0.0
        total = sum(self.within_model_weights().values())
        total += self.am_weight + self.scam_weight + self.de_weight
        return total * self.birth_death_fraction / (1.0 - self.birth_death_fraction)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary of every field."""
        data = asdict(self)
        if self.band_bins is not None:
            data["band_bins"] = list(self.band_bins)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UCBBlockConfig:
        """Build a configuration, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown UCBBlockConfig keys: {unknown}")
        values = dict(data)
        if values.get("band_bins") is not None:
            values["band_bins"] = tuple(values["band_bins"])
        return cls(**values)

    def write(self, path: str | Path) -> None:
        """Write the configuration as sorted, indented JSON."""
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        Path(path).write_text(text + "\n")

    @classmethod
    def read(cls, path: str | Path) -> UCBBlockConfig:
        """Read a configuration written by :meth:`write`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


__all__ = ["BACKENDS", "UCBBlockConfig"]
