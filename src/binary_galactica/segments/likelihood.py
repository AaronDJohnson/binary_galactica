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
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class _Kernels:
    lnlike: Callable[..., Any]
    templates: Callable[..., Any]
    fisher: Callable[..., Any]


@functools.lru_cache(maxsize=None)
def _kernels(
    BW: int, n_pad: int, k_max: int, backend: str, fmt: str, response_samples: int | None
) -> _Kernels:
    zero = jnp.zeros((), dtype=jnp.complex128)

    def place(p, T, t0, bin_lo):
        w = ucb_waveform(p, T=T, BW=BW, t0=t0, fmt=fmt, backend=backend,
                         response_samples=response_samples)
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


def _response_samples(T: float, backend: str, response_samples: int | None) -> int | None:
    """Resolve the on-fly automatic grid for a concrete ``T`` (glass's own policy)."""
    if backend != "onfly" or response_samples is not None:
        return response_samples
    boundaries = np.asarray(_duration_boundaries(jnp.float64))
    return _AUTO_RESPONSE_SAMPLES[int(np.sum(boundaries < T))]


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


class SegmentLikelihood:
    """Gaussian A/E likelihood on bins ``bin_lo .. bin_lo + n_bins - 1``.

    Call :meth:`set_data` once per turn and :meth:`set_reference` once per
    run before evaluating. ``x`` rows are padded impulse vectors
    ``(n, 8 * k_max)`` with NaN past ``k``.
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
    ) -> None:
        if int(n_bins) < 1 or int(k_max) < 1:
            raise ValueError("n_bins and k_max must be positive")
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
        self.bin_lo = int(bin_lo)
        self.n_bins = int(n_bins)
        self.k_max = int(k_max)
        self.n_pad = quantize_bins(self.n_bins)
        self.snr_min = float(snr_min)
        resolved = _response_samples(self.T, backend, response_samples)
        self.response_samples = None if resolved is None else int(resolved)
        self.key: tuple[int, int, int, str, str, int | None] = (
            int(BW), self.n_pad, self.k_max, backend, fmt, self.response_samples
        )
        self.kernels = _kernels(*self.key)
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

    def __call__(self, x: ArrayLike, k: ArrayLike) -> NDArray[np.float64]:
        """Return the log likelihood of each row; NaN becomes ``-inf``."""
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


__all__ = ["SegmentLikelihood", "quantize_bins"]
