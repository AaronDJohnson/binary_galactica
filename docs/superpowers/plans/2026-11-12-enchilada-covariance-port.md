# Port the UCB block to enchilada's Covariance: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the UCB segment block, its tests and the segment tools work with enchilada 0.3.0's `Covariance` and its `Block` protocol.

**Architecture:**
- `UCBBlock.sample` takes `covariance=`.
- `_diagonals` reads the per-bin A/E variances through `covariance.pair`, using duck typing on `channel_names`, `pair` and an optional `form`.
- `tools/segments/runs.py` builds `Covariance.like`.
- The enchilada-backed tests move to `Covariance.from_psd` and `check_block(initial_covariance=)`.

The spec is `docs/superpowers/specs/2026-11-12-enchilada-covariance-port-design.md`.

**Tech Stack:** Python 3.11, NumPy, JAX 0.7.2, pytest, enchilada `main` (0.3.0) from source.

**User decisions (already made):**
- "Port first (Recommended)": port the block before TDI-2 spec 2.
- "0.3.0+ only (Recommended)": accept only the new `Covariance`, with no `DataCovariance` fallback.

**Environment (every shell):**

```bash
cd /Users/aaron/Documents/lisa/binary_galactica/.worktrees/enchilada-covariance
export PYTHONPATH=src:$HOME/Documents/GitHub/impulse_mcmc:$HOME/Documents/lisa/enchilada/src
```

- Python is `/Users/aaron/miniconda3/bin/python`.
- The enchilada checkout's code equals enchilada `main` (0.3.0). It reports version `0+unknown`.

**Baseline at dae2c43:**
- 516 tests collected.
- Exactly three fail, each with `AttributeError: module 'enchilada' has no attribute 'DataCovariance'`:
  - `test_check_block`
  - `test_check_block_three_segments`
  - `test_matches_enchilada_quadratic_form`
- ruff is clean.
- mypy: `Found 40 errors in 4 files`.

---

### Task 1: The block reads enchilada's Covariance

**Goal:** `UCBBlock.sample` takes `covariance=`, and `_diagonals` reads A/E variances through `covariance.pair`. The fake covariance and the rejection tests use the new shape.

**Files:**
- Modify: `src/binary_galactica/segments/block.py`:
  - module docstring, lines 3-8;
  - `reference_covariance` docstring, around line 74;
  - `_diagonals`, around lines 216-237;
  - `sample`, around lines 413-452.
- Modify: `src/binary_galactica/segments/likelihood.py` (module docstring, lines 1-16).
- Test: `tests/segments/test_block.py`:
  - `_Data.block_state`, around line 32;
  - `_cov`, around lines 48-61;
  - the domain test, around line 182;
  - `test_rejects_bad_covariances`, around lines 188-200.

**Acceptance Criteria:**
- [ ] `UCBBlock.sample`'s second parameter is named `covariance`.
- [ ] `_diagonals` raises:
  - `TypeError` naming "enchilada Covariance" and "initial_covariance", for `None` or an object without `pair`;
  - `ValueError` for full form, wrong channels, wrong shape, a non-finite or non-positive variance inside the table span, and correlated A–E.
- [ ] A non-finite variance outside the table span (e.g. at DC) is accepted.
- [ ] `python -m pytest tests/segments/test_block.py -q -m "not slow"` passes.

**Verify:** `python -m pytest tests/segments/test_block.py -q -m "not slow"` → all passed, 0 failed

**Steps:**

- [ ] **Step 1: Update the fakes and write the failing rejection tests**

In `tests/segments/test_block.py`, replace `_Data.block_state` and `_cov`:

```python
class _Data(SimpleNamespace):
    def block_state(self, template=None, *, covariance=None, parameters=None):
        return SimpleNamespace(template=template, parameters=parameters)
```

```python
class _Cov(SimpleNamespace):
    """Minimal stand-in for a diagonal enchilada ``Covariance``."""

    def pair(self, first, second):
        if (first, second) in self.pairs:
            return self.pairs[(first, second)]
        return np.zeros_like(self.pairs[(first, first)])


def _cov(channels=CHANNELS):
    f = np.arange(N_FREQ) / T_SHORT
    f[0] = f[1]
    variance = 0.5 * T_SHORT * ae_noise(f)
    return _Cov(
        channel_names=channels,
        form="diagonal",
        pairs={(ch, ch): variance.copy() for ch in channels},
    )
```

In the domain test, replace

```python
    with pytest.raises(TypeError, match="DataCovariance"):
        block.sample(_grid(), SimpleNamespace(data_domain="time"), None, rng=rng)
```

with

```python
    with pytest.raises(TypeError, match="enchilada Covariance"):
        block.sample(_grid(), SimpleNamespace(data_domain="time"), None, rng=rng)
    with pytest.raises(TypeError, match="initial_covariance"):
        block.sample(_grid(), None, None, rng=rng)
```

Replace the body of `test_rejects_bad_covariances` with:

```python
def test_rejects_bad_covariances(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    rng = np.random.default_rng(0)
    correlated = _cov()
    a, e = correlated.pairs[("A", "A")], correlated.pairs[("E", "E")]
    correlated.pairs[("A", "E")] = 0.1 * np.sqrt(a * e)
    with pytest.raises(ValueError, match="uncorrelated"):
        block.sample(_grid(), correlated, None, rng=rng)
    with pytest.raises(ValueError, match="covariance channels"):
        block.sample(_grid(), _cov(("A", "E", "X")), None, rng=rng)
    full = _cov()
    full.form = "full"
    with pytest.raises(ValueError, match="diagonal"):
        block.sample(_grid(), full, None, rng=rng)
    short = _cov()
    short.pairs[("A", "A")] = short.pairs[("A", "A")][:-1]
    with pytest.raises(ValueError, match="shape"):
        block.sample(_grid(), short, None, rng=rng)
    lo = block.table.segments[0].bin_lo
    zero = _cov()
    zero.pairs[("E", "E")][lo] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        block.sample(_grid(), zero, None, rng=rng)
    # DC lies outside every segment window, so a bad value there is never read.
    outside = _cov()
    outside.pairs[("A", "A")][0] = np.inf
    assert np.isinf(block._diagonals(outside)[0, 0])
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python -m pytest tests/segments/test_block.py -q -m "not slow"`

Expected: FAIL. The block still reads `covariance_matrix`, so every test that passes `_cov()` raises `TypeError: noise_covariance must be a frequency-domain DataCovariance`.

- [ ] **Step 3: Port the block**

In `src/binary_galactica/segments/block.py`, replace the module docstring's first two numbered items:

```
1. checks that the residual and covariance are frequency-domain and on the
   construction grid (channels, length, sample rate and start time), and that
   the ledger's turn matches the block's;
2. takes the A and E coefficient variances ``C = (T/2) S`` from the
   covariance;
```

with

```
1. checks that the residual is frequency-domain and on the construction grid
   (channels, length, sample rate and start time), that the covariance is an
   enchilada ``Covariance`` with the data's channels, and that the ledger's
   turn matches the block's;
2. takes the A and E coefficient variances ``C = (T/2) S`` from
   ``covariance.pair``;
```

In the class docstring, replace `Frequency ``DataCovariance`` defining the SNR cut for the run;` with `enchilada ``Covariance`` on the data's frequency grid defining the SNR cut for the run;`, and rewrap that paragraph to 88 columns.

Replace `_diagonals` with:

```python
    def _diagonals(self, cov: Any) -> NDArray[np.float64]:
        """Return ``(2, n_freq)`` A/E coefficient variances from a ``Covariance``.

        Only ``channel_names``, ``pair`` and an optional ``form`` are read, so
        any enchilada ``Covariance`` from 0.3.0 on works. Variances are checked
        inside the table's span only; the segment windows never read DC.
        """
        if cov is None or not callable(getattr(cov, "pair", None)):
            raise TypeError(
                "covariance must be an enchilada Covariance (0.3.0 or later); "
                "register the block with data_domain='frequency' and give the "
                "Wheel an initial_covariance"
            )
        if getattr(cov, "form", "diagonal") != "diagonal":
            raise ValueError("the UCB block needs a diagonal-form Covariance")
        if tuple(cov.channel_names) != self.channels:
            raise ValueError("covariance channels differ from the data channels")
        diag = [np.asarray(cov.pair(ch, ch)) for ch in ("A", "E")]
        if any(d.shape != (self.n_freq,) for d in diag):
            raise ValueError(f"covariance arrays must have shape {(self.n_freq,)}")
        c = np.stack([d.real for d in diag]).astype(np.float64)
        lo, hi = self.table.segments[0].bin_lo, self.table.segments[-1].bin_hi
        band = c[:, lo:hi]
        if not np.all(np.isfinite(band) & (band > 0)):
            raise ValueError(
                "covariance variances must be finite and positive in the block's band"
            )
        cross = np.asarray(cov.pair("A", "E"))[lo:hi]
        if np.any(np.abs(cross) > 1e-12 * np.sqrt(band[0] * band[1])):
            raise ValueError("the UCB block needs uncorrelated A and E noise")
        return c
```

In `sample`, rename the parameter `noise_covariance: Any,` to `covariance: Any,`, and change `c = self._diagonals(noise_covariance)` to `c = self._diagonals(covariance)`.

- [ ] **Step 4: Update the likelihood docstring**

In `src/binary_galactica/segments/likelihood.py`:
- Change "``C = (T/2) S`` (enchilada's ``DataCovariance`` convention)" to "``C = (T/2) S`` (enchilada's ``Covariance.from_psd`` convention)".
- Change "which is ``-0.5 * DataCovariance.quadratic_form`` without the constant log-determinant." to "which is ``-0.5`` times the Gaussian quadratic form ``2 sum |r - sum h|^2 / C`` over the stacked real Fourier coordinates, without the constant log-determinant."

Rewrap both to 88 columns.

- [ ] **Step 5: Run the tests to see them pass**

```bash
python -m pytest tests/segments/test_block.py -q -m "not slow"
python -m pytest tests/segments -q -m "not slow" --deselect tests/segments/test_likelihood.py::test_matches_enchilada_quadratic_form
ruff check . && ruff format --check .
```

Expected:
- Both pytest runs pass. Task 2 replaces the deselected test.
- ruff is clean.
- `grep -rn "noise_covariance\|covariance_matrix\|active_mask" src` prints nothing.

- [ ] **Step 6: Commit**

```bash
git add src/binary_galactica/segments/block.py src/binary_galactica/segments/likelihood.py tests/segments/test_block.py
git commit -F - <<'EOF'
Port the UCB block to enchilada's Covariance

enchilada 0.3.0 replaced DataCovariance with Covariance and calls
block.sample(covariance=...) by keyword. The block now takes covariance=
and reads the A/E variances through Covariance.pair, checking them only
inside the table's span.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```

---

### Task 2: Move the enchilada-backed tests and the tools to Covariance

**Goal:** Port the real-enchilada tests, `tools/segments/runs.py` and the README to `Covariance`, so the full suite passes with no deselects.

**Files:**
- Modify: `tests/segments/test_block.py` (`_Noise`, around line 94; `_check_block_run`, around lines 458-510).
- Modify: `tests/segments/test_likelihood.py` (`test_matches_enchilada_quadratic_form`, around lines 257-278).
- Modify: `tools/segments/runs.py` (`covariance`, around lines 216-245).
- Modify: `README.md` (around lines 531-532).

**Acceptance Criteria:**
- [ ] `test_check_block` and `test_check_block_three_segments` pass against enchilada `main`.
- [ ] `test_matches_enchilada_covariance_convention` replaces `test_matches_enchilada_quadratic_form` and passes.
- [ ] `runs.covariance` returns an `enchilada.Covariance` with its signature unchanged.
- [ ] The full suite passes with no deselects. ruff is clean. mypy stays at 40 errors in 4 files.
- [ ] `grep -rn "DataCovariance\|initial_noise_covariance\|quadratic_form(" src tests tools README.md` prints nothing.

**Verify:** `python -m pytest -q` → `516 passed`, nothing deselected

**Steps:**

- [ ] **Step 1: Port `_check_block_run`**

In `tests/segments/test_block.py`, replace `_Noise` with:

```python
class _Noise:
    """A/E PSD floored at the first nonzero bin; ``Covariance.from_psd`` evaluates DC."""

    def psd(self, f):
        return ae_noise(np.maximum(f, 1.0 / T_SHORT))
```

In `_check_block_run`:
- `cov = enchilada.DataCovariance.from_psd(grid, _Noise())` becomes `cov = enchilada.Covariance.from_psd(grid, _Noise())`;
- `var = cov.covariance_matrix[:, i, i].real` becomes `var = cov.pair(ch, ch).real`;
- `initial_noise_covariance=cov,` becomes `initial_covariance=cov,`.

- [ ] **Step 2: Replace the likelihood oracle test**

In `tests/segments/test_likelihood.py`, replace `test_matches_enchilada_quadratic_form` with:

```python
def test_matches_enchilada_covariance_convention():
    """The log likelihood is -0.5 times the quadratic form 2 sum |x|^2 / C.

    ``C`` is read through enchilada's ``Covariance.pair``, and enchilada's own
    ``noise_psd`` confirms ``C = (T/2) S`` on the band.
    """
    enchilada = pytest.importorskip("enchilada")
    lik, r, c = _lik()
    h = _independent(SOURCES[0]) + _independent(SOURCES[1])
    band = slice(BIN_LO, BIN_LO + N_BINS)
    variances = {}
    for i, ch in enumerate("AE"):
        full = np.ones(N_FREQ)
        full[band] = c[i]
        variances[ch] = full
    data = enchilada.L1Data(
        {ch: np.zeros(N_FREQ, dtype=complex) for ch in "AE"},
        sample_rate_hz=1 / DT,
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        channel_names=("A", "E"),
        num_time_samples=N_TIME,
        data_domain="frequency",
    )
    cov = enchilada.Covariance.like(data, variances)
    T = N_TIME * DT
    expected = 0.0
    for i, ch in enumerate("AE"):
        C = cov.pair(ch, ch)[band]
        np.testing.assert_allclose(cov.noise_psd(ch, data)[band], 2 * C / T, rtol=1e-12)
        expected += -0.5 * np.sum(2 * np.abs(r[i] - h[i]) ** 2 / C)
    got = lik(_row(SOURCES)[None], np.array([2]))[0]
    assert got == pytest.approx(expected, rel=1e-10)
```

If `noise_psd` does not equal `2 C / T`, stop and report the observed ratio. Do not loosen the assertion: it pins the block's `(T/2) S` convention to enchilada's.

- [ ] **Step 3: Port `runs.covariance`**

In `tools/segments/runs.py`, replace `covariance` with:

```python
def covariance(data, variances: dict, lo: int, hi: int):
    """Return a diagonal enchilada ``Covariance`` on ``data``'s frequency grid.

    It uses ``variances`` on ``[lo, hi)`` and the instrument-only
    :func:`instrument_variance` elsewhere. DC takes bin 1's instrument variance,
    because ``Covariance`` needs a positive variance at every bin; the segment
    windows never include DC. Callers must pass ``lo`` and ``hi`` covering the
    block's whole padded band (``segments[0].bin_lo`` to ``segments[-1].bin_hi``),
    or the block sees instrument-only noise there.
    """
    import enchilada

    n_freq = data.num_time_samples // 2 + 1
    T = data.Tobs
    f = np.arange(n_freq) / T
    f[0] = f[1]
    base = instrument_variance(f, T)
    arrays = {}
    for ch in data.channel_names:
        diag = base.copy()
        diag[lo:hi] = variances[ch]
        arrays[ch] = diag
    return enchilada.Covariance.like(data, arrays)
```

Callers (`compare_glass.py:111`, `parallel_timing.py:76`, `segment767.py:93`) keep working unchanged. Check them with `grep -n "runs.covariance" -A4 tools/segments/*.py`.

- [ ] **Step 4: Update the README**

In `README.md`, replace

```
1. The block reads the A and E coefficient variances `C = (T/2) S` from the
   frequency `DataCovariance`. S is the one-sided PSD in the data's units. For
```

with

```
1. The block reads the A and E coefficient variances `C = (T/2) S` from the
   enchilada `Covariance` (0.3.0 or later) through `covariance.pair`. S is the
   one-sided PSD in the data's units. For
```

Rewrap the rest of that list item if needed. Then grep the README for any other enchilada API names that changed in 0.3.0 (`DataCovariance`, `initial_noise_covariance`, `noise_covariance`), and fix them the same way.

- [ ] **Step 5: Run everything**

```bash
python -m pytest -q
ruff check . && ruff format --check .
python -m mypy src 2>&1 | tail -1
grep -rn "DataCovariance\|initial_noise_covariance\|quadratic_form(" src tests tools README.md
```

Expected:
- `516 passed`, with nothing deselected.
- ruff is clean.
- `Found 40 errors in 4 files`.
- The grep prints nothing.

- [ ] **Step 6: Commit**

```bash
git add tests/segments/test_block.py tests/segments/test_likelihood.py tools/segments/runs.py README.md
git commit -F - <<'EOF'
Move the enchilada-backed tests and the tools to Covariance

check_block runs pass initial_covariance= and Covariance.from_psd; the
likelihood oracle reads C through Covariance.pair and pins C = (T/2) S
to enchilada's noise_psd, since 0.3.0 has no quadratic_form.
runs.covariance returns Covariance.like, so the full suite passes with
no deselects.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
```
