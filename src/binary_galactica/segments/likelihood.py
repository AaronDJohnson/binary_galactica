r"""Masked ``K``-slot Gaussian likelihood of one segment.

With residual ``r``, per-slot templates ``h_i`` and coefficient variance
``C = (T/2) S`` (enchilada's ``DataCovariance`` convention) on the segment's
padded bins, the log likelihood is

.. math::
    \ln L = -\sum_{A,E}\sum_{\rm bins} |r - \textstyle\sum_{i<k} h_i|^2 / C,

which is ``-0.5 * DataCovariance.quadratic_form`` without the constant
log-determinant. It is ``-inf`` if any active source has
``SNR^2 = 2 sum |h_i|^2 / C_ref < snr_min^2``. Here ``C_ref`` is a reference
covariance fixed for the run, so the truncated target is the same in every
Gibbs turn.

Templates are placed with a *traced* first bin. Window lengths are padded to
:func:`quantize_bins`, with zero inverse variance in the extra bins. One XLA
compile therefore serves every segment with the same
``(BW, n_pad, K, backend, fmt, response_samples)``. On the on-fly backend an
automatic (``None``) response grid is resolved from the concrete ``T`` at
construction, so only that grid compiles rather than every automatic branch.

Each sampler move changes one source, so evaluating a batch re-renders only the
sources it has not seen. A source's template depends on its eight parameters
alone, given the window and waveform settings, so templates are cached under
the parameters' exact bytes. That key needs no knowledge of which chain, slot
or move produced a row. Births and deaths shift slots, tempering swaps
exchange rows, the sampler may pass a subset of rows, and a rejected
proposal's old value comes back. All of these are cache hits. The cache keeps
the least recently used entries up to ``cache_slots`` and is cleared with
:meth:`SegmentLikelihood.clear_cache`. The cache-free evaluation is kept as
:meth:`SegmentLikelihood.lnlike_uncached`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..glass import GLASS_TO_CONTINUOUS, SNR_MIN, ucb_waveform
from ..glass.tdi import FORMATS
from ..glass.waveform import _AUTO_RESPONSE_SAMPLES, _duration_boundaries
from .config import BACKENDS, UCBBlockConfig
from .table import Segment


def quantize_bins(n: int) -> int:
    """Return the smallest ``2**m`` or ``3 * 2**(m-2)`` that is at least ``n``.

    >>> [quantize_bins(n) for n in (1, 2, 3, 5, 7, 100)]
    [1, 2, 3, 6, 8, 128]
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    p = 1 << (n - 1).bit_length()
    three = 3 * (p >> 2)
    return three if p >= 4 and three >= n else p


# Batch sizes of the cached-template renderer. A call costs about 0.1 ms plus
# about 45 us per source, and a step usually renders one new source per chain,
# so the last chunk is padded only up to the next size.
RENDER_SIZES = (4, 8, 12, 16)


@dataclass(frozen=True)
class _Kernels:
    lnlike: Callable[..., Any]
    templates: Callable[..., Any]
    fisher: Callable[..., Any]
    render: Callable[..., Any]
    assemble: Callable[..., Any]


@functools.cache
def _render_kernel(
    BW: int, backend: str, fmt: str, response_samples: int | None
) -> Callable[..., Any]:
    """Compact templates ``(n, 2, BW)``, window-relative bins and finiteness."""

    def compact(p, T, t0, bin_lo):
        w = ucb_waveform(
            p,
            T=T,
            BW=BW,
            t0=t0,
            fmt=fmt,
            backend=backend,
            response_samples=response_samples,
        )
        c = GLASS_TO_CONTINUOUS(T)
        h = jnp.stack([c * w["A"], c * w["E"]])
        idx = jnp.asarray(w["freq_bin_indices"] - bin_lo, dtype=jnp.int32)
        finite = jnp.all(jnp.isfinite(w["A"])) & jnp.all(jnp.isfinite(w["E"]))
        return h, idx, finite

    return jax.jit(jax.vmap(compact, in_axes=(0, None, None, None)))


@functools.cache
def _assemble_kernel(n_pad: int, k_max: int) -> Callable[..., Any]:
    """Log likelihood of rows built from compact templates (see ``lnlike_row``)."""
    zero = jnp.zeros((), dtype=jnp.complex128)

    def row(h, idx, finite, k, residual, inv_c, inv_c_ref, snr2_min):
        active = jnp.arange(k_max) < k
        valid = (idx >= 0) & (idx < n_pad) & active[:, None]
        safe = jnp.where(valid, idx, n_pad)  # out of range: dropped, never wrapped
        vals = jnp.where(valid[:, None, :], h, zero)
        flat = safe.reshape(-1)
        model = jnp.zeros((2, n_pad), dtype=jnp.complex128)
        model = model.at[0, flat].add(vals[:, 0].reshape(-1), mode="drop")
        model = model.at[1, flat].add(vals[:, 1].reshape(-1), mode="drop")
        d = residual - model
        ll = -jnp.sum(inv_c * (d.real**2 + d.imag**2))
        ref = jnp.concatenate([inv_c_ref, jnp.zeros((2, 1))], axis=1)[:, safe]
        power = vals.real**2 + vals.imag**2
        snr2 = 2.0 * jnp.sum(jnp.moveaxis(ref, 0, 1) * power, axis=(1, 2))
        bad = jnp.any(active & (~finite | (snr2 < snr2_min)))
        ll = jnp.where(bad, -jnp.inf, ll)
        return jnp.where(jnp.isnan(ll), -jnp.inf, ll)

    return jax.jit(jax.vmap(row, in_axes=(0, 0, 0, 0) + (None,) * 4))


class _TemplateStore:
    """Compact templates keyed by a source's exact parameter bytes, LRU-evicted.

    Row 0 is a permanent all-zero template that inactive slots point at.
    """

    def __init__(self, BW: int) -> None:
        self.BW = int(BW)
        self.renders = 0
        self.clear()

    def clear(self) -> None:
        self._row: dict[bytes, int] = {}
        self._key: list[bytes | None] = [None]
        self.h = np.zeros((1, 2, self.BW), dtype=np.complex128)
        self.idx = np.full((1, self.BW), -1, dtype=np.int32)
        self.finite = np.ones(1, dtype=bool)
        self._stamp = np.zeros(1, dtype=np.int64)
        self._free: list[int] = []
        self._clock = 0

    def __len__(self) -> int:
        return len(self._row)

    @property
    def capacity(self) -> int:
        return len(self._key) - 1

    def _grow(self, rows: int) -> None:
        if rows <= 0:
            return
        start = len(self._key)
        self._key.extend([None] * rows)
        self.h = np.concatenate([self.h, np.zeros((rows, 2, self.BW), self.h.dtype)])
        self.idx = np.concatenate([self.idx, np.full((rows, self.BW), -1, np.int32)])
        self.finite = np.concatenate([self.finite, np.ones(rows, dtype=bool)])
        self._stamp = np.concatenate([self._stamp, np.zeros(rows, dtype=np.int64)])
        self._free.extend(range(start + rows - 1, start - 1, -1))

    def _reserve(self, need: int, capacity: int) -> list[int]:
        """Free ``need`` rows, evicting entries this call has not used."""
        self._grow(capacity - self.capacity)
        if len(self._free) < need:
            stale = np.flatnonzero(self._stamp[1:] < self._clock) + 1
            stale = stale[[self._key[i] is not None for i in stale]]
            if len(self._free) + len(stale) < need:
                self._grow(need - len(self._free) - len(stale))
            # Evict the older half of the stale entries (at least what is needed),
            # so eviction runs rarely rather than on every call.
            count = min(len(stale), max(need - len(self._free), len(stale) // 2))
            for i in stale[np.argsort(self._stamp[stale], kind="stable")[:count]]:
                del self._row[self._key[i]]  # type: ignore[arg-type]
                self._key[i] = None
                self._free.append(int(i))
        return [self._free.pop() for _ in range(need)]

    def rows(
        self,
        sources: NDArray[np.float64],
        capacity: int,
        render: Callable[[NDArray[np.float64]], tuple[Any, Any, Any]],
    ) -> NDArray[np.int64]:
        """Return each source's store row, rendering the sources not yet stored."""
        self._clock += 1
        buf = np.ascontiguousarray(sources, dtype=np.float64).tobytes()
        width = 8 * 8
        out = np.zeros(len(sources), dtype=np.int64)
        missing: dict[bytes, list[int]] = {}
        get = self._row.get
        for i in range(len(sources)):
            key = buf[i * width : (i + 1) * width]
            row = get(key)
            if row is None:
                missing.setdefault(key, []).append(i)
            else:
                out[i] = row
        self._stamp[out] = self._clock  # protect this call's hits from eviction
        if missing:
            keys = list(missing)
            rows = self._reserve(len(keys), capacity)
            h, idx, finite = render(sources[[missing[k][0] for k in keys]])
            self.h[rows], self.idx[rows], self.finite[rows] = h, idx, finite
            self._stamp[rows] = self._clock
            for key, row in zip(keys, rows, strict=True):
                self._row[key] = row
                self._key[row] = key
                out[missing[key]] = row
            self.renders += len(keys)
        return out


@functools.cache
def _kernels(
    BW: int,
    n_pad: int,
    k_max: int,
    backend: str,
    fmt: str,
    response_samples: int | None,
) -> _Kernels:
    zero = jnp.zeros((), dtype=jnp.complex128)

    def place(p, T, t0, bin_lo):
        w = ucb_waveform(
            p,
            T=T,
            BW=BW,
            t0=t0,
            fmt=fmt,
            backend=backend,
            response_samples=response_samples,
        )
        c = GLASS_TO_CONTINUOUS(T)
        idx = jnp.asarray(w["freq_bin_indices"] - bin_lo, dtype=jnp.int32)
        # Send every out-of-window index to one positive out-of-range slot, as
        # template_on_window does: negative indices would otherwise wrap.
        valid = (idx >= 0) & (idx < n_pad)
        safe = jnp.where(valid, idx, n_pad)
        out = jnp.zeros((2, n_pad), dtype=jnp.complex128)
        out = out.at[0, safe].add(jnp.where(valid, c * w["A"], zero), mode="drop")
        out = out.at[1, safe].add(jnp.where(valid, c * w["E"], zero), mode="drop")
        # Invalid traced parameters give NaN channels (and NaN bin indices on
        # the Fourier backend, which could otherwise drop every bin): make the
        # whole template NaN so the likelihood becomes -inf.
        finite = jnp.all(jnp.isfinite(w["A"])) & jnp.all(jnp.isfinite(w["E"]))
        return jnp.where(finite, out, jnp.nan)

    def dummy(bin_lo):
        base = jnp.asarray([0.0, 0.0, 1.0, -50.0, 0.0, 0.5, 0.5, 0.0])
        return base.at[0].set(bin_lo + 0.5 * n_pad + 0.25)

    def lnlike_row(x, k, residual, inv_c, inv_c_ref, snr2_min, T, t0, bin_lo):
        comps = x.reshape(k_max, 8)
        active = jnp.arange(k_max) < k
        safe = jnp.where(active[:, None], comps, dummy(bin_lo)[None, :])
        h = jax.vmap(place, in_axes=(0, None, None, None))(safe, T, t0, bin_lo)
        h = jnp.where(active[:, None, None], h, zero)
        d = residual - jnp.sum(h, axis=0)
        ll = -jnp.sum(inv_c * (d.real**2 + d.imag**2))
        snr2 = 2.0 * jnp.sum(inv_c_ref[None] * (h.real**2 + h.imag**2), axis=(1, 2))
        ll = jnp.where(jnp.any(active & (snr2 < snr2_min)), -jnp.inf, ll)
        return jnp.where(jnp.isnan(ll), -jnp.inf, ll)

    def fisher_one(p, inv_c, T, t0, bin_lo):
        J = jax.jacfwd(place)(p, T, t0, bin_lo)
        return 2.0 * jnp.einsum("cni,cnj,cn->ij", jnp.conj(J), J, inv_c).real

    many = jax.vmap(place, in_axes=(0, None, None, None))
    return _Kernels(
        lnlike=jax.jit(jax.vmap(lnlike_row, in_axes=(0, 0) + (None,) * 7)),
        templates=jax.jit(many),
        fisher=jax.jit(jax.vmap(fisher_one, in_axes=(0, None, None, None, None))),
        render=_render_kernel(BW, backend, fmt, response_samples),
        assemble=_assemble_kernel(n_pad, k_max),
    )


def _inverse(c: ArrayLike, shape: tuple[int, int]) -> NDArray[np.float64]:
    """Return ``1 / c``, with weight zero where ``c`` is 0 or ``+inf``.

    NaN or negative entries raise ``ValueError``.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.shape != shape:
        raise ValueError(f"covariance must have shape {shape}, got {c.shape}")
    if np.any(np.isnan(c) | (c < 0)):
        raise ValueError("covariance must not be NaN or negative")
    inv = np.zeros(shape)
    ok = np.isfinite(c) & (c > 0)
    inv[ok] = 1.0 / c[ok]
    return inv


def _response_samples(
    T: float, backend: str, response_samples: int | None
) -> int | None:
    """Resolve the on-fly automatic grid for a concrete ``T`` (glass's own policy)."""
    if backend != "onfly" or response_samples is not None:
        return response_samples
    boundaries = np.asarray(_duration_boundaries(jnp.float64))
    return _AUTO_RESPONSE_SAMPLES[int(np.sum(boundaries < T))]


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


class SegmentLikelihood:
    """Gaussian A/E likelihood on bins ``bin_lo .. bin_lo + n_bins - 1``.

    Call :meth:`set_data` once per turn and :meth:`set_reference` once per
    run before evaluating. ``x`` rows are padded impulse vectors
    ``(n, 8 * k_max)`` with NaN past ``k``.

    Evaluation reuses cached templates (see the module docstring).
    ``cache_slots`` bounds the cache. The default is four times the first
    batch's rows times ``k_max``, and the cache grows past the bound only to
    hold one batch's sources.
    """

    def __init__(
        self,
        *,
        T: float,
        t0: float,
        bin_lo: int,
        n_bins: int,
        BW: int,
        k_max: int,
        backend: str = "onfly",
        fmt: str = "sangria",
        response_samples: int | None = None,
        snr_min: float = SNR_MIN,
        cache_slots: int | None = None,
    ) -> None:
        """Validate the window and settings and prepare the template evaluator."""
        if int(n_bins) < 1 or int(k_max) < 1:
            raise ValueError("n_bins and k_max must be positive")
        if cache_slots is not None and (not _is_int(cache_slots) or cache_slots < 1):
            raise ValueError("cache_slots must be None or a positive integer")
        if not _is_int(BW) or BW <= 0 or BW % 2:
            raise ValueError(f"BW must be a positive even integer, got {BW!r}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        if response_samples is not None and (
            not _is_int(response_samples) or response_samples < 4
        ):
            raise ValueError("response_samples must be None or an integer >= 4")
        self.T = float(T)
        if not (np.isfinite(self.T) and self.T > 0):
            raise ValueError(f"T must be finite and positive, got {T!r}")
        self.t0 = float(t0)
        if not np.isfinite(self.t0):
            raise ValueError(f"t0 must be finite, got {t0!r}")
        self.bin_lo = int(bin_lo)
        self.n_bins = int(n_bins)
        self.k_max = int(k_max)
        self.n_pad = quantize_bins(self.n_bins)
        self.snr_min = float(snr_min)
        resolved = _response_samples(self.T, backend, response_samples)
        self.response_samples = None if resolved is None else int(resolved)
        self.key: tuple[int, int, int, str, str, int | None] = (
            int(BW),
            self.n_pad,
            self.k_max,
            backend,
            fmt,
            self.response_samples,
        )
        self.kernels = _kernels(*self.key)
        self.cache_slots = None if cache_slots is None else int(cache_slots)
        self._store = _TemplateStore(int(BW))
        self._residual: jax.Array | None = None
        self._inv_c: jax.Array | None = None
        self._inv_c_ref: jax.Array | None = None

    @classmethod
    def for_segment(
        cls,
        segment: Segment,
        *,
        T: float,
        t0: float,
        config: UCBBlockConfig,
        k_max: int | None = None,
    ) -> SegmentLikelihood:
        """Build the likelihood of one table segment with the run's settings."""
        return cls(
            T=T,
            t0=t0,
            bin_lo=segment.bin_lo,
            n_bins=segment.n_bins,
            BW=segment.bw,
            k_max=segment.k_max if k_max is None else k_max,
            backend=config.backend,
            response_samples=config.response_samples,
            snr_min=config.snr_min,
        )

    def _padded(self, a: NDArray[Any], dtype: Any) -> jax.Array:
        out = np.zeros((2, self.n_pad), dtype=dtype)
        out[:, : self.n_bins] = a
        return jnp.asarray(out)

    def _scalars(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return jnp.float64(self.T), jnp.float64(self.t0), jnp.int64(self.bin_lo)

    def set_data(self, residual: ArrayLike, c: ArrayLike) -> None:
        """Set this turn's ``(2, n_bins)`` A/E residual and coefficient variance.

        A bin with ``c == 0`` or ``c == +inf`` is excluded (zero weight), as
        enchilada stores its excluded DC bin with zero covariance. NaN or
        negative ``c`` raises ``ValueError``.
        """
        r = np.asarray(residual, dtype=np.complex128)
        if r.shape != (2, self.n_bins) or not np.all(np.isfinite(r)):
            raise ValueError(f"residual must be finite with shape (2, {self.n_bins})")
        inv_c = _inverse(c, (2, self.n_bins))
        self._residual = self._padded(r, np.complex128)
        self._inv_c = self._padded(inv_c, np.float64)

    def set_reference(self, c_ref: ArrayLike) -> None:
        """Set the run-fixed ``(2, n_bins)`` covariance that defines the SNR cut.

        Zero or ``+inf`` entries exclude their bin from the SNR; NaN or
        negative entries raise ``ValueError``, as in :meth:`set_data`.
        """
        self._inv_c_ref = self._padded(_inverse(c_ref, (2, self.n_bins)), np.float64)

    def _batch(self, sources: ArrayLike) -> tuple[jax.Array, int]:
        x = np.asarray(sources, dtype=np.float64).reshape(-1, 8)
        n = x.shape[0]
        if n == 0:
            raise ValueError("need at least one source")
        m = quantize_bins(n)
        padded = np.concatenate([x, np.repeat(x[:1], m - n, axis=0)])
        return jnp.asarray(padded), n

    def templates(self, sources: ArrayLike) -> NDArray[np.complex128]:
        """Return each source's ``(2, n_bins)`` A/E template on the window."""
        x, n = self._batch(sources)
        h = self.kernels.templates(x, *self._scalars())
        return np.asarray(h)[:n, :, : self.n_bins]

    def snr(self, sources: ArrayLike) -> NDArray[np.float64]:
        """Return each source's SNR against the reference covariance."""
        if self._inv_c_ref is None:
            raise RuntimeError("call set_reference first")
        h = self.templates(sources)
        inv = np.asarray(self._inv_c_ref)[:, : self.n_bins]
        return np.sqrt(2.0 * np.sum(inv[None] * np.abs(h) ** 2, axis=(1, 2)))

    def fisher_matrices(self, sources: ArrayLike) -> NDArray[np.float64]:
        """Return ``2 Re sum (dh/dx)^H (dh/dx) / C`` per source, shape ``(n, 8, 8)``."""
        if self._inv_c is None:
            raise RuntimeError("call set_data first")
        x, n = self._batch(sources)
        F = self.kernels.fisher(x, self._inv_c, *self._scalars())
        return np.asarray(F)[:n]

    def _rows(
        self, x: ArrayLike, k: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        if self._residual is None or self._inv_c is None or self._inv_c_ref is None:
            raise RuntimeError("call set_data and set_reference first")
        rows = np.asarray(x, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 8 * self.k_max:
            raise ValueError(f"x must have shape (n, {8 * self.k_max})")
        ks = np.asarray(k, dtype=np.int64).reshape(-1)
        if ks.shape[0] != rows.shape[0]:
            raise ValueError(f"k has {ks.shape[0]} entries for {rows.shape[0]} rows")
        if np.any((ks < 0) | (ks > self.k_max)):
            raise ValueError(f"k must lie in 0 .. {self.k_max}")
        return rows, ks

    def _render(self, sources: NDArray[np.float64]) -> tuple[Any, Any, Any]:
        """Compact templates of ``sources``, in chunks of the :data:`RENDER_SIZES`."""
        hs, idxs, finites = [], [], []
        big = RENDER_SIZES[-1]
        for start in range(0, len(sources), big):
            chunk = sources[start : start + big]
            n = len(chunk)
            size = next(s for s in RENDER_SIZES if s >= n)
            pad = np.concatenate([chunk, np.repeat(chunk[:1], size - n, 0)])
            h, idx, finite = self.kernels.render(jnp.asarray(pad), *self._scalars())
            hs.append(np.asarray(h)[:n])
            idxs.append(np.asarray(idx)[:n])
            finites.append(np.asarray(finite)[:n])
        return np.concatenate(hs), np.concatenate(idxs), np.concatenate(finites)

    def __call__(self, x: ArrayLike, k: ArrayLike) -> NDArray[np.float64]:
        """Return the log likelihood of each row; NaN becomes ``-inf``.

        Equal to :meth:`lnlike_uncached` up to summation order (about 1e-15
        relative), from cached templates of the rows' active sources.
        """
        rows, ks = self._rows(x, k)
        n = rows.shape[0]
        active = np.arange(self.k_max)[None, :] < ks[:, None]
        sources = rows.reshape(n, self.k_max, 8)[active]
        capacity = self.cache_slots or 4 * n * self.k_max
        ids = np.zeros((n, self.k_max), dtype=np.int64)
        ids[active] = self._store.rows(sources, capacity, self._render)
        store = self._store
        out = self.kernels.assemble(
            jnp.asarray(store.h[ids]),
            jnp.asarray(store.idx[ids]),
            jnp.asarray(store.finite[ids]),
            jnp.asarray(ks),
            self._residual,
            self._inv_c,
            self._inv_c_ref,
            jnp.float64(self.snr_min**2),
        )
        return np.asarray(out, dtype=np.float64)

    def lnlike_uncached(self, x: ArrayLike, k: ArrayLike) -> NDArray[np.float64]:
        """Return the log likelihood with every slot's template rendered afresh."""
        rows, ks = self._rows(x, k)
        out = self.kernels.lnlike(
            jnp.asarray(rows),
            jnp.asarray(ks),
            self._residual,
            self._inv_c,
            self._inv_c_ref,
            jnp.float64(self.snr_min**2),
            *self._scalars(),
        )
        return np.asarray(out, dtype=np.float64)

    def clear_cache(self) -> None:
        """Drop every cached template and free the cache's memory."""
        self._store.clear()

    def cache_info(self) -> dict[str, int]:
        """Return the cache's ``entries``, ``capacity`` and cumulative ``renders``."""
        store = self._store
        return {
            "entries": len(store),
            "capacity": store.capacity,
            "renders": store.renders,
        }


__all__ = ["SegmentLikelihood", "quantize_bins"]
