"""Segment table: GLASS-style frequency segments, fixed for the whole run.

A segment has a half-open interior ``[lo, hi)`` of rfft bins and ``pad`` bins
of padding per side. The frequency prior and the likelihood window cover the
padded band ``[lo - pad, hi + pad)``; the segment owns, and publishes, only the
sources whose ``f0*T`` lies in its interior (GLASS's ``qpad`` layout).

Interiors come from a greedy tiling upward in frequency. Each boundary sits
just above the midpoint between the ``n_target``-th and ``(n_target+1)``-th
remaining start source, with at least ``min_interior_bins`` bins per interior.
Padding per side is twice GLASS's Doppler-plus-drift bandwidth:

    bw = 2 T [(4 + 2 pi f AU / c) / yr + |fdot| T]

evaluated at the interior's upper edge with ``sin(theta) = 1`` and the largest
``|fdot|`` among the owned start sources. A merge pass then enforces
``w_s >= max(pad_s, pad_{s-1} + pad_{s+1})``. Padding then reaches only
adjacent interiors, and windows of the same parity are disjoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..glass import UCBPrior, ucb_bw, ucb_prior
from ..glass.constants import AU, CLIGHT, PI2, YEAR
from .config import UCBBlockConfig


def doppler_bins(f: float, fdot: float, T: float) -> float:
    """Return GLASS's Doppler-plus-drift bandwidth in bins, with ``sin(theta)=1``."""
    return 2.0 * T * ((4.0 + PI2 * f * (AU / CLIGHT)) / YEAR + abs(fdot) * T)


def padding_bins(f: float, fdot: float, T: float) -> int:
    """Return the padding per side: twice :func:`doppler_bins`, rounded up."""
    return int(math.ceil(2.0 * doppler_bins(f, fdot, T)))


def _floor_pow2(n: int) -> int:
    return 1 << (int(n).bit_length() - 1)


@dataclass(frozen=True)
class Segment:
    """One segment: interior ``[lo, hi)``, padding, and static settings.

    ``owned`` holds indices into the start set passed to
    :func:`build_segment_table`.
    """

    index: int
    lo: int
    hi: int
    pad: int
    bw: int
    k_max: int
    owned: tuple[int, ...]

    @property
    def width(self) -> int:
        """Return the interior width in bins."""
        return self.hi - self.lo

    @property
    def bin_lo(self) -> int:
        """Return the first bin of the padded window."""
        return self.lo - self.pad

    @property
    def bin_hi(self) -> int:
        """Return one past the last bin of the padded window."""
        return self.hi + self.pad

    @property
    def n_bins(self) -> int:
        """Return the number of bins in the padded window."""
        return self.bin_hi - self.bin_lo

    def owns(self, q: float) -> bool:
        """Return whether ``f0*T = q`` lies in the interior."""
        return self.lo <= q < self.hi

    def prior(self, T: float) -> UCBPrior:
        """Return GLASS's uniform prior over the padded band."""
        return ucb_prior(T=T, fmin=self.bin_lo / T, fmax=self.bin_hi / T)


@dataclass(frozen=True)
class SegmentTable:
    """Every segment of the band, plus the start sources left outside it."""

    T: float
    n_freq: int
    segments: tuple[Segment, ...]
    outside: tuple[int, ...] = ()

    def owners(self, q: ArrayLike) -> NDArray[np.int64]:
        """Return the owning segment of each ``f0*T``, or ``-1`` outside the band."""
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        los = np.array([s.lo for s in self.segments], dtype=np.float64)
        idx = np.searchsorted(los, q, side="right") - 1
        out = np.full(q.shape, -1, dtype=np.int64)
        ok = (idx >= 0) & (q < self.segments[-1].hi)
        out[ok] = idx[ok]
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "T": self.T,
            "n_freq": self.n_freq,
            "outside": list(self.outside),
            "segments": [
                {**asdict(s), "owned": list(s.owned)} for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentTable:
        """Rebuild a table from :meth:`to_dict` output."""
        segments = tuple(
            Segment(**{**s, "owned": tuple(s["owned"])}) for s in data["segments"]
        )
        return cls(
            T=float(data["T"]),
            n_freq=int(data["n_freq"]),
            segments=segments,
            outside=tuple(data["outside"]),
        )

    def write(self, path: str | Path) -> None:
        """Write the table as indented JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> SegmentTable:
        """Read a table written by :meth:`write`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


def _tile(qs: NDArray[np.float64], lo: int, hi: int, n_target: int, min_w: int) -> list[int]:
    """Return interior edges from the greedy midpoint rule over sorted ``qs``."""
    edges = [lo]
    cursor = lo
    while cursor < hi:
        rest = qs[qs >= cursor]
        if rest.size <= n_target:
            nxt = hi
        else:
            mid = 0.5 * (rest[n_target - 1] + rest[n_target])
            nxt = int(np.floor(mid)) + 1
        nxt = max(nxt, cursor + min_w)
        if hi - nxt < min_w:
            nxt = hi
        edges.append(nxt)
        cursor = nxt
    return edges


def _owned(edges: list[int], q: NDArray[np.float64], members: NDArray[np.int64]):
    """Return, per interior, the start-set indices whose ``q`` it contains."""
    return [
        tuple(int(i) for i in members[(q[members] >= a) & (q[members] < b)])
        for a, b in zip(edges[:-1], edges[1:])
    ]


def _pads(edges, owned, native, T) -> list[int]:
    pads = []
    for (a, b), own in zip(zip(edges[:-1], edges[1:]), owned):
        fdot = float(np.max(np.abs(native[list(own), 7]))) / T**2 if own else 0.0
        pads.append(padding_bins(b / T, fdot, T))
    return pads


def _first_violation(edges: list[int], pads: list[int]) -> int | None:
    n = len(pads)
    for s in range(n):
        left = pads[s - 1] if s > 0 else 0
        right = pads[s + 1] if s < n - 1 else 0
        if edges[s + 1] - edges[s] < max(pads[s], left + right):
            return s
    return None


def build_segment_table(
    native: ArrayLike, *, T: float, n_freq: int, config: UCBBlockConfig
) -> SegmentTable:
    """Tile the band of a GLASS-native start set into segments.

    ``native`` is ``(n, 8)``, as from :func:`~binary_galactica.segments.ldc_to_native`.
    The band is ``config.band_bins`` if given, otherwise
    ``[floor(min f0*T), floor(max f0*T) + 1)``, widened about its centre to at
    least ``min_interior_bins`` and to the padding that width implies (so a lone
    source builds a valid one-segment table). The widened band is shifted to keep the
    padded window inside the grid interior, but always covers every start source; the grid check below is the final guard.
    """
    x = np.asarray(native, dtype=np.float64).reshape(-1, 8)
    q = x[:, 0]
    if config.band_bins is not None:
        band_lo, band_hi = config.band_bins
    else:
        if q.size == 0:
            raise ValueError("an empty start set needs explicit band_bins")
        band_lo = int(np.floor(q.min()))
        band_hi = int(np.floor(q.max())) + 1
        fdot_max = float(np.max(np.abs(x[:, 7]))) / T**2
        width = max(band_hi - band_lo, config.min_interior_bins)
        pad = 0
        for _ in range(4):  # the padding depends weakly on the upper edge
            width = max(width, pad)
            centre = 0.5 * (band_lo + band_hi)
            lo = int(np.floor(centre - 0.5 * width))
            hi = lo + width
            pad = padding_bins(hi / T, fdot_max, T)
            if width >= pad:
                break
        width = max(width, pad)
        lo = int(np.floor(centre - 0.5 * width))
        hi = lo + width
        shift = max(0, pad + 1 - lo) - max(0, hi - (n_freq - 1 - pad))
        lo, hi = lo + shift, hi + shift
        band_lo = min(lo, int(np.floor(q.min())))
        band_hi = max(hi, int(np.floor(q.max())) + 1)
    inside = (q >= band_lo) & (q < band_hi)
    members = np.nonzero(inside)[0]
    outside = tuple(int(i) for i in np.nonzero(~inside)[0])
    edges = _tile(np.sort(q[members]), band_lo, band_hi, config.n_target,
                  config.min_interior_bins)

    while True:
        owned = _owned(edges, q, members)
        pads = _pads(edges, owned, x, T)
        bad = _first_violation(edges, pads)
        if bad is None:
            break
        if len(pads) == 1:
            raise ValueError(
                f"band [{band_lo}, {band_hi}) is narrower than its padding "
                f"({pads[0]}); widen config.band_bins"
            )
        if bad == 0:
            neighbour = 1
        elif bad == len(pads) - 1:
            neighbour = bad - 1
        else:
            w_left = edges[bad] - edges[bad - 1]
            w_right = edges[bad + 2] - edges[bad + 1]
            neighbour = bad - 1 if w_left < w_right else bad + 1
        del edges[max(bad, neighbour)]

    segments = []
    for s, ((a, b), own, pad) in enumerate(zip(zip(edges[:-1], edges[1:]), owned, pads)):
        n_bins = (b - a) + 2 * pad
        if a - pad < 1 or b + pad > n_freq - 1:
            raise ValueError(
                f"segment {s} window [{a - pad}, {b + pad}) leaves the grid interior "
                f"[1, {n_freq - 1})"
            )
        if own:
            fdot = float(np.max(np.abs(x[list(own), 7]))) / T**2
            amp = float(np.max(np.exp(x[list(own), 3])))
        else:
            fdot, amp = 0.0, 0.0
        bw = min(ucb_bw(b / T, fdot, 0.0, amp, T, n_freq), _floor_pow2(n_bins))
        seg = Segment(
            index=s,
            lo=int(a),
            hi=int(b),
            pad=int(pad),
            bw=int(bw),
            k_max=max(config.k_max_floor, 2 * len(own)),
            owned=own,
        )
        prior = seg.prior(T)
        outside_prior = [i for i in own if not prior.contains(x[i])]
        if outside_prior:
            raise ValueError(
                f"start sources {outside_prior} lie outside the prior of segment {s}"
            )
        segments.append(seg)
    return SegmentTable(T=float(T), n_freq=int(n_freq), segments=tuple(segments),
                        outside=outside)


__all__ = [
    "Segment",
    "SegmentTable",
    "build_segment_table",
    "doppler_bins",
    "padding_bins",
]
