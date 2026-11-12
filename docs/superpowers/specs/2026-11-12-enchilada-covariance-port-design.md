# Port the UCB block to enchilada's `Covariance` (design)

Date: 2026-11-12
Branch: `enchilada-covariance`, from `parallel-sweep` at dae2c43
Status: design approved in discussion on 2026-11-12

## Why

enchilada 0.3.0 (commit 7586a70, now on `main`) replaced `DataCovariance` with
`Covariance` and changed the `Block` protocol's second parameter to
`covariance`. `Wheel` calls `block.sample(conditional_residual=...,
covariance=..., current_block_state=..., rng=...)` by keyword. The UCB block
still names that parameter `noise_covariance` and still reads
`covariance_matrix` and `active_mask`. Against enchilada `main` it therefore
raises `TypeError` on the first turn.

The three tests deselected since 2026-11-11 (`test_check_block`,
`test_check_block_three_segments`, `test_matches_enchilada_quadratic_form`) fail
because of this break, with `module 'enchilada' has no attribute
'DataCovariance'`.

## User decisions

- "Port first (Recommended)": port the block before TDI-2 spec 2, so spec 2 can
  emit CD1-L noise as an enchilada `Covariance`.
- "0.3.0+ only (Recommended)": accept only the new `Covariance` and drop
  `DataCovariance` support.

## Design

**What the block reads.** The block identifies the covariance by duck typing,
because a source checkout of enchilada reports version `0+unknown`. It uses
only:

- `channel_names`;
- `pair(first, second)`;
- `form`, if present.

enchilada's pending covariance-simplification spec keeps `pair` and removes
`form`, so this access survives that change.

**`UCBBlock.sample(conditional_residual, covariance, current_block_state, *,
rng)`.** The parameter is renamed to match the protocol. Positional callers are
unaffected.

**`UCBBlock._diagonals(covariance)`** returns the `(2, n_freq)` A/E variances
`C = (T/2) S`.

- It raises `TypeError` for `None` or an object without `pair`. The message
  says to register with `data_domain="frequency"` and to give `Wheel` an
  `initial_covariance`.
- It raises `ValueError` when:
  - `form` is present and is not `"diagonal"`;
  - `channel_names` differ from the data's;
  - `pair("A","A")` or `pair("E","E")` does not have shape `(n_freq,)`;
  - any variance inside the table's span `[segments[0].bin_lo,
    segments[-1].bin_hi)` is not finite and positive;
  - `|pair("A","E")|` exceeds `1e-12 * sqrt(C_A C_E)` in that span. This is the
    existing "uncorrelated A and E" rule.
- The `active_mask` → `inf` path is removed. A `Covariance` has a positive
  variance at every bin, DC included, and the segment windows never contain DC.
- `reference_covariance=` takes the same type.

**`segments/likelihood.py`.** Only the docstring changes. The log likelihood
`-sum |r - sum h|^2 / C` is `-0.5` times the Gaussian quadratic form
`2 sum |r - sum h|^2 / C` over the stacked real Fourier coordinates, without
the constant log-determinant.

**`tools/segments/runs.py: covariance(data, variances, lo, hi)`** keeps its
signature. It returns `enchilada.Covariance.like(data, {ch: variance array})`.
DC takes bin 1's instrument variance. Removing the dense `(n_freq, 3, 3)` array
saves about 0.9 GB at CD1-L size.

**Tests.**

- The fake covariance in `tests/segments/test_block.py` becomes a minimal
  stand-in with `channel_names`, `form` and `pair`.
- The rejection tests cover:
  - `None`;
  - an object without `pair`;
  - full form;
  - wrong channels;
  - wrong shape;
  - a non-positive variance in the band;
  - correlated A–E.
- `_check_block_run` uses `Covariance.from_psd`, with the PSD model floored at
  bin 1, and `check_block(..., initial_covariance=...)`.
- enchilada 0.3.0 has no `quadratic_form`. The oracle test therefore becomes
  `test_matches_enchilada_covariance_convention`. It checks the block's log
  likelihood against `-sum_ch sum_k |x_k|^2 / C_k`, with `C` read through
  `Covariance.pair`. It also checks that enchilada's own `noise_psd` equals
  `2 C / T` on the band, which pins the `(T/2) S` convention to enchilada.

**Docs.** In the README and in the block and likelihood docstrings, the name
changes from `DataCovariance` to `Covariance`.

## Out of scope

- The TDI-2 code on the local `ucb-segment-block`.
- CD1-L data and noise (TDI-2 spec 2).
- Noise blocks.
- `pyproject` pins: enchilada stays an undeclared, optional test dependency.

The port and TDI-2 meet in TDI-2 spec 3.

## Acceptance

With enchilada `main` (0.3.0) on `PYTHONPATH`:

- the full suite passes with no deselects (516 tests at dae2c43, none added);
- `ruff check .` and `ruff format --check .` are clean;
- `python -m mypy src` stays at 40 errors in 4 files.
