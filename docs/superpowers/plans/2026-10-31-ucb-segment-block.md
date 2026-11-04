# UCB Segment Block Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample the LDC2a Sangria galactic-binary band in GLASS-style frequency segments, as one enchilada block (`binary_galactica.segments.UCBBlock`) that drives impulse's reversible-jump parallel-tempered sampler from injected start values.

**Architecture:** A segment table, fixed at construction, tiles the band into interiors with padding (GLASS's layout). Each segment has two parts. One is a jitted, masked `K`-slot Gaussian likelihood on its padded window, with the SNR cut. The other is an `impulse.RJPTSampler` with GLASS's moves (Fisher, prior redraw, fm shift, psi-phi, prior birth/death). One `UCBBlock.sample()` call is one Gibbs turn. It subtracts the other segments' owned sources, advances every segment (even indices, then odd), and publishes the owned sources' template in A/E/T.

**Tech Stack:**
- Python 3.11 and JAX 0.7 (float64).
- NumPy.
- impulse-mcmc, `rjmcmc` branch (`RJPTSampler`).
- enchilada, tests and tools only.
- h5py, scipy and pandas, tools only.
- pytest.

**Spec:** `docs/superpowers/specs/2026-10-31-ucb-segment-block-design.md` (commit 15f62b5).

**User decisions (already made):**
- "I don't want to implement the GMM stuff. We're going to be starting at the injected values with this one. So no need for F-stat search either."
- "We do need birth/death moves, which we can get from ~/Documents/GitHub/impulse_mcmc in the current branch that is being modified to include RJMCMC. No need to make that in this repo."
- "This will all be done in a blocked Gibbs sampler which I will set up. We will see the noise model fixed at every iteration." The framework is enchilada (`/Users/aaron/Documents/lisa/enchilada`).
- The start set is LDC catalogue rows, supplied by the user's noise-block package.
- Choices made in the design review:
  - a new `binary_galactica` subpackage;
  - 12 months of data;
  - `obs` minus the true MBHB and VGB streams;
  - one UCB block holding every segment;
  - the full band with GLASS-like variable widths;
  - the SNR < 5 cut kept.
- The spec was approved ("approved"), including three changes from self-review:
  - SNR is measured against a reference PSD fixed for the run;
  - start-set sources below the cut are dropped and recorded;
  - Fisher matrices are looked up by frequency.

**Implementation choices beyond the spec (surfaced to the user at handoff):**
- **Continue bridge.** impulse has `RJPTSampler` but no continue call. Each turn re-calls `sampler.sample(positions, N, initial_k=ks)` on the *same* sampler object.
  - Kept across turns: the ladder and swap counts (`ptstate`), the RNG streams, the proposal bundle and chain stats. `_set_initial_state` recomputes lnlike/lnprior for the changed residual and PSD, which is the spec's "recompute" requirement.
  - Lost: the iteration counter (so `num_adapt` would count per turn; we leave it `None`), checkpoint continuity, and the chain files (impulse truncates them).
  - The runner archives each turn's cold chain itself.
- **Channels.** The data must be `("A","E")` or `("A","E","T")`; anything else raises `ValueError`. T is derived from X/Y/Z. `fmt` is fixed to `"sangria"`, the convention the gate validates.
- **Compile reuse.** Templates are placed with a *traced* `bin_lo`, and window lengths are padded up to a short ladder (`quantize_bins`). XLA then compiles once per `(BW, n_pad, K)` rather than once per segment.
- **Parity rule.** The table enforces `interior_s >= max(pad_s, pad_{s-1} + pad_{s+1})` by merging segments. This makes same-parity windows disjoint, as the spec states.

**Environment for every command** (repository root):

```bash
export PYTHONPATH=src:$HOME/Documents/GitHub/impulse_mcmc:$HOME/Documents/lisa/enchilada/src
git -C ~/Documents/GitHub/impulse_mcmc branch --show-current   # must print: rjmcmc
```

Local baseline (memory `local-toolchain-baseline`): ruff is not installed, so report it as not run. `python -m mypy src` has about 42 pre-existing errors; compare against that list, not against zero. Tests that need impulse, enchilada, h5py or the Sangria file must skip cleanly when it is absent. `pyproject.toml` runs `--doctest-modules` over `src` and `--strict-markers`, so:
- no new module may import impulse, enchilada or h5py at module level;
- do not add pytest markers.

## File structure

```
src/binary_galactica/segments/
  __init__.py      public names (grows task by task)
  config.py        UCBBlockConfig: every setting, validation, JSON
  catalogue.py     LDC catalogue rows <-> GLASS-native 8-vectors (NumPy)
  table.py         Segment, SegmentTable, build_segment_table, padding rule
  likelihood.py    SegmentLikelihood: jitted masked K-slot lnlike, SNR, Fisher
  prior.py         SegmentPrior: uniform count x UCBPrior per active slot
  fisher.py        fisher_decomposition, FisherBank (turn-fixed, by frequency)
  moves.py         SlotMove + FisherMove/PriorRedrawMove/FmShiftMove/PsiPhiMove
  runner.py        SegmentRunner (RJPTSampler bridge), ChainArchive, glass_ladder
  render.py        render_sources/Contribution: owned sources on the rfft grid
  block.py         UCBBlock: the enchilada block (one Gibbs turn per sample())
tools/segments/
  sangria.py           Sangria h5 readers (application side)
  conventions_gate.py  catalogue templates vs noise-free streams (Task 3)
  runs.py              shared helpers for the validation tools
  isolated_source.py   fixed-count check on one synthetic source (Task 9)
  segment767.py        crowded band, fixed count and RJ (Task 10)
  compare_glass.py     real segments vs injections and GLASS chains (Task 11)
  dry_run.py           full-band table, compile keys, time per turn (Task 12)
tests/segments/
  __init__.py, support.py, test_*.py
```

---

### Task 1: Scaffold the subpackage and its configuration

**Goal:** Create `binary_galactica.segments` with a validated, JSON-serializable `UCBBlockConfig` holding the spec's defaults, plus the test support module and packaging metadata.

**Files:**
- Modify: `pyproject.toml` (optional extra `segments`; ruff per-file ignores)
- Create: `src/binary_galactica/segments/__init__.py`
- Create: `src/binary_galactica/segments/config.py`
- Create: `tests/segments/__init__.py`
- Create: `tests/segments/support.py`
- Test: `tests/segments/test_config.py`

**Acceptance Criteria:**
- [ ] `UCBBlockConfig()` defaults: `n_target=10`, `steps_per_turn=1000`, `ntemps=12`, `k_max_floor=10`, `backend="fourier"`, within-model weights sum to 1 with Fisher 0.5.
- [ ] `birth_death_weight()` gives birth/death exactly 25% of the total move weight; 0 when `birth_death=False`.
- [ ] Every invalid setting in the test table raises `ValueError`.
- [ ] `write`/`read` round-trips; unknown keys raise `ValueError`.
- [ ] `pyproject.toml` declares `segments = ["impulse-mcmc>=2.0"]`.

**Verify:** `python -m pytest tests/segments/test_config.py -q` → `14 passed`

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/segments/__init__.py` is an empty file.

`tests/segments/support.py`:

```python
"""Shared helpers for the segment-block tests (not a conftest: import it)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools" / "segments"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

SANGRIA = Path(
    os.environ.get(
        "BG_SANGRIA_H5",
        "/Users/aaron/Documents/lisa/project-catalog/LDC2_sangria_training_v2.h5",
    )
)
requires_sangria = pytest.mark.skipif(
    not SANGRIA.exists(), reason="Sangria training file not found; set BG_SANGRIA_H5"
)

#: A short grid that keeps tests fast: 2**16 samples at 15 s, about 11 days.
#: A source with ln A = -48 at 3 mHz has SNR ~38 here against (T/2)*ae_noise.
DT = 15.0
N_TIME = 2**16
T_SHORT = N_TIME * DT
N_FREQ = N_TIME // 2 + 1


def rj_impulse():
    """Import impulse with ``RJPTSampler`` or skip (needs the ``rjmcmc`` branch)."""
    impulse = pytest.importorskip("impulse")
    if not hasattr(impulse, "RJPTSampler"):
        pytest.skip("impulse lacks RJPTSampler; put the rjmcmc branch on PYTHONPATH")
    return impulse


def ldc_rows(**overrides) -> np.ndarray:
    """Return one LDC catalogue row as a structured array, with field overrides."""
    values = {
        "Amplitude": 2.0e-22,
        "EclipticLatitude": 0.3,
        "EclipticLongitude": 1.7,
        "Frequency": 3.0e-3,
        "FrequencyDerivative": 1.0e-17,
        "Inclination": 0.9,
        "InitialPhase": 2.1,
        "Polarization": 0.6,
    }
    values.update(overrides)
    dtype = [(name, np.float64) for name in values]
    return np.array([tuple(values.values())], dtype=dtype)


def native_source(q: float, lnA: float = -48.0, fdot_T2: float = 0.0) -> np.ndarray:
    """Return a GLASS-native source at ``f0*T = q`` with generic angles."""
    return np.array([q, 0.2, 1.1, lnA, 0.3, 0.7, 2.0, fdot_T2])
```

`tests/segments/test_config.py`:

```python
import json

import pytest

from binary_galactica.segments import UCBBlockConfig


def test_defaults_follow_the_spec():
    cfg = UCBBlockConfig()
    assert (cfg.n_target, cfg.steps_per_turn, cfg.ntemps, cfg.k_max_floor) == (
        10,
        1000,
        12,
        10,
    )
    assert cfg.backend == "fourier"
    weights = cfg.within_model_weights()
    assert weights["fisher"] == pytest.approx(0.5)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_birth_death_weight_gives_the_requested_fraction():
    cfg = UCBBlockConfig()
    total = sum(cfg.within_model_weights().values())
    bd = cfg.birth_death_weight()
    assert bd / (bd + total) == pytest.approx(0.25)
    assert UCBBlockConfig(birth_death=False).birth_death_weight() == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_target": 0},
        {"ntemps": 1.5},
        {"ladder_step": 1.0},
        {"birth_death_fraction": 1.0},
        {"fisher_weight": -0.1},
        {
            "fisher_weight": 0.0,
            "prior_weight": 0.0,
            "fm_shift_weight": 0.0,
            "psi_phi_weight": 0.0,
        },
        {"backend": "wavelet"},
        {"band_bins": (10, 5)},
        {"response_samples": 2},
        {"snr_min": -1.0},
    ],
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        UCBBlockConfig(**kwargs)


def test_json_round_trip(tmp_path):
    cfg = UCBBlockConfig(band_bins=(100, 900), steps_per_turn=50, backend="onfly")
    path = tmp_path / "config.json"
    cfg.write(path)
    assert UCBBlockConfig.read(path) == cfg
    assert json.loads(path.read_text())["band_bins"] == [100, 900]


def test_unknown_keys_raise():
    with pytest.raises(ValueError, match="unknown"):
        UCBBlockConfig.from_dict({"steps": 3})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_config.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'binary_galactica.segments'`

- [ ] **Step 3: Implement the configuration**

`src/binary_galactica/segments/config.py`:

```python
"""Settings of the UCB segment block, written verbatim to each run directory."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from ..glass import DEFAULT_RESPONSE_SAMPLES, SNR_MIN

#: Waveform backends the block can use; the conventions gate picks the default.
BACKENDS = ("fourier", "onfly")

_POSITIVE_INTS = (
    "n_target",
    "min_interior_bins",
    "k_max_floor",
    "steps_per_turn",
    "thin",
    "ntemps",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


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
    """

    n_target: int = 10
    min_interior_bins: int = 16
    band_bins: tuple[int, int] | None = None
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
    backend: str = "fourier"
    response_samples: int | None = DEFAULT_RESPONSE_SAMPLES

    def __post_init__(self) -> None:
        """Validate every field; raise ``ValueError`` on the first bad one."""
        for name in _POSITIVE_INTS:
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.band_bins is not None:
            lo, hi = self.band_bins
            if not (_is_int(lo) and _is_int(hi) and 0 < lo < hi):
                raise ValueError(
                    f"band_bins must be integers 0 < lo < hi, got {self.band_bins!r}"
                )
            object.__setattr__(self, "band_bins", (lo, hi))
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
```

`src/binary_galactica/segments/__init__.py`:

```python
"""GLASS-style sampling of galactic-binary frequency segments.

One :class:`UCBBlock` holds every segment of the band and implements
enchilada's ``Block`` protocol structurally (enchilada is not imported).
Trans-dimensional sampling uses impulse's ``RJPTSampler``, imported lazily.
"""

from __future__ import annotations

from .config import BACKENDS, UCBBlockConfig

__all__ = ["BACKENDS", "UCBBlockConfig"]
```

In `pyproject.toml`, extend `[project.optional-dependencies]` and the ruff per-file ignores:

```toml
[project.optional-dependencies]
dev = ["pytest>=7", "ruff>=0.5", "mypy>=1.10"]
segments = ["impulse-mcmc>=2.0"]
```

```toml
"src/binary_galactica/segments/*" = ["N802", "N803", "N806"]
"tests/segments/*" = ["N802", "N803", "N806"]
"tools/segments/*" = ["N802", "N803", "N806", "D103"]
```

(Add the three lines at the end of `[tool.ruff.lint.per-file-ignores]`.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_config.py -q`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/binary_galactica/segments tests/segments
git commit -m "Add the segments subpackage and its configuration"
```

---

### Task 2: Convert LDC catalogue rows to native vectors

**Goal:** One tested host-NumPy conversion from LDC catalogue rows (`sky/dgb/cat` fields) to GLASS-native 8-vectors and back. The periodic angles are folded into `UCBPrior`'s intervals.

**Files:**
- Create: `src/binary_galactica/segments/catalogue.py`
- Modify: `src/binary_galactica/segments/__init__.py`
- Test: `tests/segments/test_catalogue.py`

**Acceptance Criteria:**
- [ ] `ldc_to_native` equals `to_glass_vector` (with `-phi0` folded into `[0, 2pi)`) to `rtol=1e-14` on 50 random rows.
- [ ] `native_to_ldc(ldc_to_native(rows))` returns every field to `rtol=1e-12`.
- [ ] h5py-shaped `(n, 1)` structured rows, and mappings of scalars, are accepted.
- [ ] Polarization folds into `[0, pi)`, longitude and `-phi0` into `[0, 2pi)`.
- [ ] Missing fields, non-finite values, and non-positive amplitude or frequency raise `ValueError`.

**Verify:** `python -m pytest tests/segments/test_catalogue.py src/binary_galactica/segments/catalogue.py -q` → `9 passed` (8 tests + 1 doctest)

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/segments/test_catalogue.py`:

```python
import numpy as np
import pytest

from binary_galactica import GBParams
from binary_galactica.parameters import to_glass_vector
from binary_galactica.segments import CATALOGUE_FIELDS, ldc_to_native, native_to_ldc

from .support import ldc_rows

T = 31_536_000.0


def _random_rows(n, seed=1):
    rng = np.random.default_rng(seed)
    rows = np.zeros(n, dtype=[(name, np.float64) for name in CATALOGUE_FIELDS])
    rows["Amplitude"] = 10 ** rng.uniform(-23, -21, n)
    rows["EclipticLatitude"] = np.arcsin(rng.uniform(-1, 1, n))
    rows["EclipticLongitude"] = rng.uniform(0, 2 * np.pi, n)
    rows["Frequency"] = rng.uniform(1e-4, 2e-2, n)
    rows["FrequencyDerivative"] = rng.uniform(-1e-17, 1e-15, n)
    rows["Inclination"] = np.arccos(rng.uniform(-1, 1, n))
    rows["InitialPhase"] = rng.uniform(0.01, 2 * np.pi - 0.01, n)
    rows["Polarization"] = rng.uniform(0, np.pi, n)
    return rows


def test_matches_to_glass_vector():
    rows = _random_rows(50)
    theta = GBParams(
        amp=rows["Amplitude"],
        f0=rows["Frequency"],
        fdot=rows["FrequencyDerivative"],
        phi0=rows["InitialPhase"],
        iota=rows["Inclination"],
        psi=rows["Polarization"],
        lam=rows["EclipticLongitude"],
        beta=rows["EclipticLatitude"],
    )
    expected = np.array(to_glass_vector(theta, T))
    expected[:, 6] = np.mod(expected[:, 6], 2 * np.pi)
    np.testing.assert_allclose(ldc_to_native(rows, T), expected, rtol=1e-14, atol=1e-15)


def test_round_trip():
    rows = _random_rows(20, seed=2)
    back = native_to_ldc(ldc_to_native(rows, T), T)
    for name in CATALOGUE_FIELDS:
        np.testing.assert_allclose(back[name], rows[name], rtol=1e-12, err_msg=name)


def test_h5py_shaped_rows_and_folds():
    rows = ldc_rows(Polarization=4.0, EclipticLongitude=7.0, InitialPhase=0.5)
    native = ldc_to_native(rows.reshape(1, 1), T)
    assert native.shape == (1, 8)
    assert native[0, 5] == pytest.approx(4.0 - np.pi)
    assert native[0, 2] == pytest.approx(7.0 - 2 * np.pi)
    assert native[0, 6] == pytest.approx(2 * np.pi - 0.5)


def test_mapping_input():
    row = {name: ldc_rows()[name][0] for name in CATALOGUE_FIELDS}
    np.testing.assert_array_equal(ldc_to_native(row, T), ldc_to_native(ldc_rows(), T))


@pytest.mark.parametrize(
    "bad", [{"Amplitude": 0.0}, {"Frequency": -1e-3}, {"Inclination": np.nan}]
)
def test_invalid_rows_raise(bad):
    with pytest.raises(ValueError):
        ldc_to_native(ldc_rows(**bad), T)


def test_missing_field_raises():
    rows = ldc_rows()
    reduced = rows[[name for name in rows.dtype.names if name != "Polarization"]]
    with pytest.raises(ValueError, match="Polarization"):
        ldc_to_native(reduced, T)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_catalogue.py -q`
Expected: `ImportError: cannot import name 'CATALOGUE_FIELDS'`

- [ ] **Step 3: Implement the conversion**

`src/binary_galactica/segments/catalogue.py`:

```python
"""LDC catalogue rows <-> GLASS-native 8-vectors, in host NumPy float64.

The native vector is :func:`binary_galactica.parameters.to_glass_vector`'s
``[f0*T, sin(beta), lambda, ln A, cos(iota), psi, -phi0, fdot*T**2]``, computed
in NumPy so that it needs no JAX precision flag. Longitude and ``-phi0`` are
folded into ``[0, 2*pi)`` and ``psi`` into ``[0, pi)``: the periodic intervals
of :class:`~binary_galactica.glass.UCBPrior`. The waveform is unchanged by
``psi -> psi + pi``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Fields every start-set row must carry (the ``sky/dgb/cat`` names).
CATALOGUE_FIELDS = (
    "Amplitude",
    "EclipticLatitude",
    "EclipticLongitude",
    "Frequency",
    "FrequencyDerivative",
    "Inclination",
    "InitialPhase",
    "Polarization",
)

TWO_PI = 2.0 * np.pi


def _column(rows: Any, name: str) -> NDArray[np.float64]:
    try:
        values = rows[name]
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"catalogue rows have no {name!r} field") from exc
    return np.asarray(values, dtype=np.float64).reshape(-1)


def ldc_to_native(rows: Any, T: float) -> NDArray[np.float64]:
    """Convert LDC catalogue rows to ``(n, 8)`` GLASS-native vectors.

    ``rows`` is a NumPy structured array of any shape (h5py reads Sangria
    catalogues as ``(n, 1)``) or a mapping from field name to values.

    >>> row = {"Amplitude": 1e-22, "EclipticLatitude": 0.3,
    ...        "EclipticLongitude": 1.0, "Frequency": 3e-3,
    ...        "FrequencyDerivative": 1e-17, "Inclination": 0.5,
    ...        "InitialPhase": 1.0, "Polarization": 4.0}
    >>> ldc_to_native(row, 1.0e7).shape
    (1, 8)
    """
    if not (np.isfinite(T) and T > 0):
        raise ValueError("T must be finite and positive")
    if not isinstance(rows, Mapping) and getattr(rows, "dtype", None) is None:
        raise ValueError("rows must be a structured array or a mapping of fields")
    cols = {name: _column(rows, name) for name in CATALOGUE_FIELDS}
    n = cols["Frequency"].size
    if any(col.size != n for col in cols.values()):
        raise ValueError("catalogue fields differ in length")
    if not all(np.all(np.isfinite(col)) for col in cols.values()):
        raise ValueError("catalogue rows must be finite")
    if np.any(cols["Amplitude"] <= 0) or np.any(cols["Frequency"] <= 0):
        raise ValueError("Amplitude and Frequency must be positive")
    native = np.empty((n, 8), dtype=np.float64)
    native[:, 0] = cols["Frequency"] * T
    native[:, 1] = np.sin(cols["EclipticLatitude"])
    native[:, 2] = np.mod(cols["EclipticLongitude"], TWO_PI)
    native[:, 3] = np.log(cols["Amplitude"])
    native[:, 4] = np.cos(cols["Inclination"])
    native[:, 5] = np.mod(cols["Polarization"], np.pi)
    native[:, 6] = np.mod(-cols["InitialPhase"], TWO_PI)
    native[:, 7] = cols["FrequencyDerivative"] * T * T
    return native


def native_to_ldc(native: ArrayLike, T: float) -> dict[str, NDArray[np.float64]]:
    """Convert ``(n, 8)`` native vectors to LDC catalogue fields."""
    x = np.asarray(native, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(f"native vectors must have shape (n, 8), got {x.shape}")
    return {
        "Amplitude": np.exp(x[:, 3]),
        "EclipticLatitude": np.arcsin(np.clip(x[:, 1], -1.0, 1.0)),
        "EclipticLongitude": x[:, 2].copy(),
        "Frequency": x[:, 0] / T,
        "FrequencyDerivative": x[:, 7] / (T * T),
        "Inclination": np.arccos(np.clip(x[:, 4], -1.0, 1.0)),
        "InitialPhase": np.mod(-x[:, 6], TWO_PI),
        "Polarization": x[:, 5].copy(),
    }


__all__ = ["CATALOGUE_FIELDS", "ldc_to_native", "native_to_ldc"]
```

Replace `src/binary_galactica/segments/__init__.py`'s import block and `__all__` with:

```python
from .catalogue import CATALOGUE_FIELDS, ldc_to_native, native_to_ldc
from .config import BACKENDS, UCBBlockConfig

__all__ = [
    "BACKENDS",
    "CATALOGUE_FIELDS",
    "UCBBlockConfig",
    "ldc_to_native",
    "native_to_ldc",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_catalogue.py src/binary_galactica/segments/catalogue.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/binary_galactica/segments tests/segments/test_catalogue.py
git commit -m "Convert LDC catalogue rows to GLASS-native vectors"
```

---

### Task 3: Conventions gate against noise-free Sangria streams

**Goal:** Measure, for every verification binary and for bright isolated `dgb`/`igb` sources, the noise-weighted match and complex scale of catalogue-driven A/E templates against `dt*rfft` of the population's own noise-free stream, with both backends. Then review the results with the user, who picks the default backend.

> **USER-ORDERED GATE — NON-SKIPPABLE.** This task was requested by the user in the current conversation. It MUST NOT be closed by walking around it, by declaring it "verified inline", or by substituting a cheaper check. Close only after every item in `acceptanceCriteria` has been re-validated independently, with output captured.

**Files:**
- Create: `tools/segments/sangria.py`
- Create: `tools/segments/conventions_gate.py`
- Create (generated): `tools/performance/segments-conventions.json`
- Test: `tests/segments/test_conventions_gate.py`
- Modify (only if the user picks on-fly): `src/binary_galactica/segments/config.py` (`backend` default), `tests/segments/test_config.py`

**Acceptance Criteria:**
- [ ] `python tools/segments/conventions_gate.py` writes `tools/performance/segments-conventions.json`. It has entries for all 36 VGBs and 5 bright isolated `dgb` and 5 `igb` sources, each with `match` and `scale` for `fourier` and `onfly`.
- [ ] HPLib, SDSSJ0651 and V407Vul reach match > 0.999 and `|scale - 1| < 2e-3` with both backends (the test).
- [ ] The full table and the worst isolated match per backend have been shown to the user. The user's backend choice is recorded in the JSON as `"backend_choice"` and is the `UCBBlockConfig.backend` default.
- [ ] Any isolated source with match < 0.99 has been investigated as a convention bug (phase, polarization, amplitude, FFT units, orbit phase, BW) and explained to the user; no threshold was relaxed.

**Verify:** `python -m pytest tests/segments/test_conventions_gate.py -q` → `1 passed`; then `python tools/segments/conventions_gate.py` prints the table and `worst isolated match: {...}`

**Steps:**

- [ ] **Step 1: Write the failing test**

`tests/segments/test_conventions_gate.py`:

```python
import numpy as np

from binary_galactica.segments import CATALOGUE_FIELDS, ldc_to_native

from .support import SANGRIA, requires_sangria


@requires_sangria
def test_bright_verification_binaries_match():
    import conventions_gate
    import sangria

    n, dt = sangria.grid(SANGRIA)
    T, n_freq = n * dt, n // 2 + 1
    spectra = sangria.aet_spectra(*sangria.load_xyz("vgb", SANGRIA))
    rows = sangria.load_catalogue("vgb", ("Name", *CATALOGUE_FIELDS), SANGRIA)
    native = ldc_to_native(rows, T)
    names = [name.decode() for name in rows["Name"]]
    for name in ("HPLib", "SDSSJ0651", "V407Vul"):
        j = names.index(name)
        for backend in ("fourier", "onfly"):
            result = conventions_gate.match_source(
                native[j], spectra, T, n_freq, backend
            )
            assert result["match"] > 0.999, (name, backend, result)
            scale = complex(*result["scale"])
            assert abs(scale - 1.0) < 2e-3, (name, backend, result)
            assert np.isfinite(result["match"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/segments/test_conventions_gate.py -q`
Expected: `ModuleNotFoundError: No module named 'conventions_gate'` (or `1 skipped` if the Sangria file is absent; then set `BG_SANGRIA_H5`)

- [ ] **Step 3: Implement the Sangria readers**

`tools/segments/sangria.py`:

```python
"""Read the LDC2a Sangria training file (application side, not package API).

The block does not build data. These helpers do it for the tools and tests:
- time-domain XYZ streams;
- their ``dt * rfft`` A/E/T spectra, in GLASS's AET convention;
- catalogue rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from binary_galactica.glass import xyz_to_aet  # noqa: E402
from binary_galactica.segments import CATALOGUE_FIELDS  # noqa: E402

DEFAULT_PATH = Path(
    os.environ.get(
        "BG_SANGRIA_H5",
        "/Users/aaron/Documents/lisa/project-catalog/LDC2_sangria_training_v2.h5",
    )
)
POPULATIONS = ("dgb", "igb", "vgb", "mbhb")


def _key(stream: str) -> str:
    if stream == "obs":
        return "obs/tdi"
    if stream in POPULATIONS:
        return f"sky/{stream}/tdi"
    raise ValueError(f"unknown stream {stream!r}")


def grid(path: Path = DEFAULT_PATH) -> tuple[int, float]:
    """Return ``(number of samples, dt)`` of the observation."""
    with h5py.File(path, "r") as f:
        ds = f["obs/tdi"]
        t = ds[:2, 0]["t"]
        return int(ds.shape[0]), float(t[1] - t[0])


def load_xyz(
    stream: str, path: Path = DEFAULT_PATH
) -> tuple[dict[str, np.ndarray], float]:
    """Return ``({"X", "Y", "Z"} -> samples, dt)`` for one stream."""
    with h5py.File(path, "r") as f:
        data = f[_key(stream)][:, 0]
    dt = float(data["t"][1] - data["t"][0])
    return {ch: np.array(data[ch], dtype=np.float64) for ch in "XYZ"}, dt


def combine(
    plus: Sequence[str], minus: Iterable[str] = (), path: Path = DEFAULT_PATH
) -> tuple[dict[str, np.ndarray], float]:
    """Sum the ``plus`` streams and subtract the ``minus`` streams."""
    total, dt = load_xyz(plus[0], path)
    signed = [(name, 1.0) for name in plus[1:]] + [(name, -1.0) for name in minus]
    for name, sign in signed:
        other, other_dt = load_xyz(name, path)
        if other_dt != dt:
            raise ValueError("streams have different cadences")
        for ch in "XYZ":
            total[ch] += sign * other[ch]
    return total, dt


def aet_series(xyz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return GLASS's A, E and T combinations of an XYZ stream."""
    A, E, T = (np.asarray(c) for c in xyz_to_aet(xyz["X"], xyz["Y"], xyz["Z"]))
    return {"A": A, "E": E, "T": T}


def aet_spectra(xyz: dict[str, np.ndarray], dt: float) -> dict[str, np.ndarray]:
    """Return ``dt * rfft`` of the A, E and T combinations of an XYZ stream."""
    return {ch: dt * np.fft.rfft(series) for ch, series in aet_series(xyz).items()}


def load_catalogue(
    population: str,
    fields: Sequence[str] = CATALOGUE_FIELDS,
    path: Path = DEFAULT_PATH,
    index: Sequence[int] | None = None,
) -> np.ndarray:
    """Return catalogue rows of one population as a 1-D structured array.

    ``index`` selects rows by position and returns them in increasing index
    order (h5py fancy indexing); ``None`` reads every row.
    """
    with h5py.File(path, "r") as f:
        view = f[f"sky/{population}/cat"].fields(list(fields))
        if index is None:
            return view[:, 0]
        order = sorted({int(i) for i in index})
        if not order:
            return view[0:0, 0]
        return view[order][:, 0]


__all__ = [
    "DEFAULT_PATH",
    "aet_series",
    "aet_spectra",
    "combine",
    "grid",
    "load_catalogue",
    "load_xyz",
]
```

- [ ] **Step 4: Implement the gate tool**

`tools/segments/conventions_gate.py`:

```python
"""Conventions gate: catalogue-driven templates against noise-free Sangria streams.

For each verification binary (``sky/vgb``) and a few bright isolated ``dgb`` and
``igb`` sources, convert the catalogue row with
:func:`binary_galactica.segments.ldc_to_native`, render A/E with both backends
on the 12-month grid, and compare with ``dt * rfft`` of the population's own
noise-free stream. Report the 1/ae_noise-weighted match and complex scale on a
``2*BW`` window. No taper is applied.

    PYTHONPATH=src python tools/segments/conventions_gate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sangria  # enables x64 before the package is used

from binary_galactica.glass import (
    FSTAR,
    ae_noise,
    analytic_snr,
    template_on_window,
    ucb_bw,
)
from binary_galactica.segments import CATALOGUE_FIELDS, ldc_to_native

BACKENDS = ("fourier", "onfly")
OUT = Path(__file__).resolve().parents[1] / "performance" / "segments-conventions.json"


def window(native: np.ndarray, T: float, n_freq: int) -> tuple[int, int, int]:
    """Return ``(bin_lo, n_bins, BW)`` of the ``2*BW`` window around a source."""
    bw = ucb_bw(
        native[0] / T, native[7] / T**2, native[1], float(np.exp(native[3])), T, n_freq
    )
    q = int(np.floor(native[0]))
    return q - bw, 2 * bw, bw


def match_source(native, spectra, T, n_freq, backend):
    """Return the noise-weighted match and least-squares complex scale."""
    bin_lo, n_bins, bw = window(native, T, n_freq)
    h = np.asarray(
        template_on_window(
            native, T=T, bin_lo=bin_lo, n_bins=n_bins, BW=bw, backend=backend
        )
    )
    sl = slice(bin_lo, bin_lo + n_bins)
    d = np.stack([spectra["A"][sl], spectra["E"][sl]])
    w = 1.0 / ae_noise((bin_lo + np.arange(n_bins)) / T)
    hd = np.sum(w * np.conj(h) * d)
    hh = np.sum(w * np.abs(h) ** 2)
    dd = np.sum(w * np.abs(d) ** 2)
    scale = hd / hh
    return {
        "match": float(hd.real / np.sqrt(hh * dd)),
        "scale": [float(scale.real), float(scale.imag)],
        "bw": int(bw),
        "bin_lo": int(bin_lo),
    }


def bright_isolated(population, T, n_freq, count, path):
    """Return indices of the brightest sources with no rival above 5% SNR."""
    fields = ("Frequency", "FrequencyDerivative", "Amplitude", "EclipticLatitude")
    cat = sangria.load_catalogue(population, fields, path)
    f, a = cat["Frequency"], cat["Amplitude"]
    snr = analytic_snr(a, ae_noise(f), np.sin(f / FSTAR), np.sqrt(T))
    order = np.argsort(f)
    f_sorted, snr_sorted = f[order], snr[order]
    picked: list[int] = []
    for i in np.argsort(snr)[::-1][:2000]:
        bw = ucb_bw(
            f[i],
            cat["FrequencyDerivative"][i],
            np.sin(cat["EclipticLatitude"][i]),
            a[i],
            T,
            n_freq,
        )
        lo, hi = np.searchsorted(f_sorted, [f[i] - bw / T, f[i] + bw / T])
        rival = np.sort(snr_sorted[lo:hi])[-2] if hi - lo > 1 else 0.0
        if rival < 0.05 * snr[i]:
            picked.append(int(i))
        if len(picked) == count:
            break
    return sorted(picked)


def neighbours(freqs: np.ndarray, j: int, T: float, half_bins: int) -> int:
    """Return how many other sources of the list sit within ``half_bins``."""
    return int(np.sum(np.abs(freqs - freqs[j]) * T < half_bins) - 1)


def run(path: Path, count: int) -> dict:
    """Return the gate report for every VGB and ``count`` dgb/igb sources."""
    n, dt = sangria.grid(path)
    T, n_freq = n * dt, n // 2 + 1
    report: dict = {"T": T, "dt": dt, "populations": {}}
    for population in ("vgb", "dgb", "igb"):
        spectra = sangria.aet_spectra(*sangria.load_xyz(population, path))
        index = (
            None
            if population == "vgb"
            else bright_isolated(population, T, n_freq, count, path)
        )
        rows = sangria.load_catalogue(
            population, ("Name", *CATALOGUE_FIELDS), path, index=index
        )
        native = ldc_to_native(rows, T)
        freqs = np.asarray(rows["Frequency"], dtype=np.float64)
        entries = []
        for j, row in enumerate(rows):
            _, _, bw = window(native[j], T, n_freq)
            name = row["Name"]
            entry = {
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "frequency_hz": float(freqs[j]),
                "fdot": float(row["FrequencyDerivative"]),
                "neighbours_in_window": neighbours(freqs, j, T, bw)
                if population == "vgb"
                else 0,
            }
            for backend in BACKENDS:
                entry[backend] = match_source(native[j], spectra, T, n_freq, backend)
            entries.append(entry)
        report["populations"][population] = entries
    isolated = [
        e
        for entries in report["populations"].values()
        for e in entries
        if e["neighbours_in_window"] == 0
    ]
    report["worst_match"] = {b: min(e[b]["match"] for e in isolated) for b in BACKENDS}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report = run(args.path, args.count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for population, entries in report["populations"].items():
        for e in entries:
            cells = "  ".join(
                f"{b}: m={e[b]['match']:.6f} "
                f"s={e[b]['scale'][0]:+.5f}{e[b]['scale'][1]:+.5f}i"
                for b in BACKENDS
            )
            print(
                f"{population:4s} {e['name']:>16s} {e['frequency_hz'] * 1e3:8.4f} mHz"
                f"  nb={e['neighbours_in_window']}  {cells}"
            )
    print("worst isolated match:", report["worst_match"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/segments/test_conventions_gate.py -q`
Expected: `1 passed`

- [ ] **Step 6: Run the gate and review it with the user**

Run: `python tools/segments/conventions_gate.py`
Expected: one line per source and `worst isolated match: {'fourier': ..., 'onfly': ...}`. Prior evidence (spec `2026-10-11-physical-amplitude.md`) puts isolated VGB scales at 0.9994–0.99999.

If any isolated source has a match below 0.99, debug it with superpowers-extended-cc:systematic-debugging. Look first at the catalogue conversion (sign of `InitialPhase`, polarization period) and the FFT units; do not relax the threshold. Then show the user the printed table and the worst match per backend, and ask which backend becomes the default. Record the answer:

```bash
python - <<'EOF'
import json, pathlib
p = pathlib.Path("tools/performance/segments-conventions.json")
r = json.loads(p.read_text())
r["backend_choice"] = "fourier"   # the user's answer: "fourier" or "onfly"
p.write_text(json.dumps(r, indent=2) + "\n")
EOF
```

If the user picks `onfly`, set `backend: str = "onfly"` in `UCBBlockConfig`. In `tests/segments/test_config.py`, change `assert cfg.backend == "fourier"` to `assert cfg.backend == "onfly"` and, in `test_json_round_trip`, change `backend="onfly"` to `backend="fourier"`. Re-run `python -m pytest tests/segments/test_config.py -q` → `14 passed`.

- [ ] **Step 7: Commit**

```bash
git add tools/segments/sangria.py tools/segments/conventions_gate.py \
  tools/performance/segments-conventions.json tests/segments/test_conventions_gate.py \
  src/binary_galactica/segments/config.py tests/segments/test_config.py
git commit -m "Gate catalogue conventions against noise-free Sangria streams"
```

---
### Task 4: Build the segment table

**Goal:** Tile the band into GLASS-style segments, fixed for the run. Each segment gets:
- a half-open interior `[lo, hi)` from a greedy `N_target` midpoint rule;
- per-side padding of `2*bw`, from the Doppler-plus-drift rule;
- a merge pass that enforces `w_s >= max(pad_s, pad_{s-1} + pad_{s+1})`;
- ownership of start sources;
- a power-of-two BW and `K`;
- grid and prior checks;
- JSON output.

**Files:**
- Create: `src/binary_galactica/segments/table.py`
- Modify: `src/binary_galactica/segments/__init__.py`
- Test: `tests/segments/test_table.py`

**Acceptance Criteria:**
- [ ] Interiors tile the band contiguously; every in-band start source is owned by exactly one segment (`lo <= f0*T < hi`); out-of-band sources are listed in `table.outside`.
- [ ] Every segment satisfies `width >= max(pad, pad_left + pad_right)`. Windows of the same parity are pairwise disjoint.
- [ ] With sparse sources (no merges needed), every segment but the last holds exactly `n_target` sources.
- [ ] `pad == ceil(2 * doppler_bins(hi/T, max|fdot|, T))`.
- [ ] `bw` is a power of two `<= n_bins`.
- [ ] `k_max == max(k_max_floor, 2 * owned)`.
- [ ] `prior(T)` spans `[bin_lo, bin_hi]` in `f0*T`.
- [ ] These raise `ValueError`:
  - a window touching DC or Nyquist;
  - a start source outside its segment prior;
  - an empty start set without `band_bins`.
- [ ] JSON round trip reproduces the table.

**Verify:** `python -m pytest tests/segments/test_table.py -q` → `11 passed`

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/segments/test_table.py`:

```python
import math

import numpy as np
import pytest

from binary_galactica.segments import (
    SegmentTable,
    UCBBlockConfig,
    build_segment_table,
    doppler_bins,
    padding_bins,
)

from .support import native_source

T = 31_536_000.0
N_FREQ = 3_153_601


def _sources(qs, lnA=-50.0, fdot_T2=0.0):
    return np.array([native_source(q, lnA=lnA, fdot_T2=fdot_T2) for q in qs])


def _check_rules(table):
    segs = table.segments
    for a, b in zip(segs[:-1], segs[1:]):
        assert a.hi == b.lo
    for s, seg in enumerate(segs):
        left = segs[s - 1].pad if s > 0 else 0
        right = segs[s + 1].pad if s < len(segs) - 1 else 0
        assert seg.width >= max(seg.pad, left + right)
        assert seg.bw & (seg.bw - 1) == 0 and seg.bw <= seg.n_bins
    for a, b in zip(segs[:-2], segs[2:]):
        assert a.bin_hi <= b.bin_lo


def test_doppler_and_padding_rule():
    f, fdot = 4e-3, 2e-16
    expected = (
        2
        * T
        * (
            (4 + 2 * np.pi * f * 1.49597870700e11 / 299792458.0) / 3.15581497632e7
            + fdot * T
        )
    )
    assert doppler_bins(f, fdot, T) == pytest.approx(expected, rel=1e-12)
    assert padding_bins(f, -fdot, T) == math.ceil(2 * expected)


def test_sparse_sources_fill_n_target():
    qs = 120_000 + 80.0 * np.arange(205) + 0.37
    table = build_segment_table(
        _sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
    _check_rules(table)
    counts = [len(seg.owned) for seg in table.segments]
    assert all(c == 10 for c in counts[:-1]) and 0 < counts[-1] <= 10
    assert sum(counts) == len(qs)


def test_ownership_is_half_open_and_unique():
    qs = 120_000 + 80.0 * np.arange(60) + 0.37
    table = build_segment_table(
        _sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
    owners = table.owners(qs)
    assert np.all(owners >= 0)
    for i, q in enumerate(qs):
        seg = table.segments[owners[i]]
        assert seg.lo <= q < seg.hi and i in seg.owned
    assert table.owners(np.array([table.segments[1].lo]))[0] == 1


def test_dense_cluster_forces_merges():
    rng = np.random.default_rng(3)
    qs = np.sort(120_000 + rng.uniform(0, 40, 30))
    cfg = UCBBlockConfig(band_bins=(119_800, 120_300))
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    _check_rules(table)
    assert sum(len(s.owned) for s in table.segments) == 30
    assert max(len(s.owned) for s in table.segments) > 10


def test_padding_uses_owned_fdot():
    qs = 120_000 + 80.0 * np.arange(20) + 0.37
    fast = _sources(qs, fdot_T2=0.5)
    table = build_segment_table(fast, T=T, n_freq=N_FREQ, config=UCBBlockConfig())
    seg = table.segments[0]
    assert seg.pad == padding_bins(seg.hi / T, 0.5 / T**2, T)


def test_k_max_and_prior_band():
    qs = 120_000 + 80.0 * np.arange(12) + 0.37
    cfg = UCBBlockConfig(n_target=12, k_max_floor=10)
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    seg = table.segments[0]
    assert seg.k_max == 24
    prior = seg.prior(T)
    assert prior.lo[0] == pytest.approx(seg.bin_lo, abs=1e-6)
    assert prior.hi[0] == pytest.approx(seg.bin_hi, abs=1e-6)


def test_explicit_band_lists_outside_sources():
    qs = np.array([119_000.5, 120_010.5, 120_030.5, 125_000.5])
    cfg = UCBBlockConfig(band_bins=(120_000, 120_400))
    table = build_segment_table(_sources(qs), T=T, n_freq=N_FREQ, config=cfg)
    assert table.outside == (0, 3)
    assert table.segments[0].lo == 120_000 and table.segments[-1].hi == 120_400


def test_window_touching_dc_raises():
    with pytest.raises(ValueError, match="grid"):
        build_segment_table(
            _sources([3.5]), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
        )


def test_source_outside_prior_raises():
    with pytest.raises(ValueError, match="prior"):
        build_segment_table(
            _sources([120_000.5], lnA=-60.0),
            T=T,
            n_freq=N_FREQ,
            config=UCBBlockConfig(),
        )


def test_empty_start_set_needs_band():
    with pytest.raises(ValueError, match="band_bins"):
        build_segment_table(
            np.empty((0, 8)), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
        )


def test_json_round_trip(tmp_path):
    qs = 120_000 + 80.0 * np.arange(30) + 0.37
    table = build_segment_table(
        _sources(qs), T=T, n_freq=N_FREQ, config=UCBBlockConfig()
    )
    path = tmp_path / "segment_table.json"
    table.write(path)
    assert SegmentTable.read(path) == table
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_table.py -q`
Expected: `ImportError: cannot import name 'SegmentTable'`

- [ ] **Step 3: Implement the table**

`src/binary_galactica/segments/table.py`:

```python
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
            "segments": [{**asdict(s), "owned": list(s.owned)} for s in self.segments],
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


def _tile(
    qs: NDArray[np.float64], lo: int, hi: int, n_target: int, min_w: int
) -> list[int]:
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
    ``[floor(min f0*T), floor(max f0*T) + 1)``, widened to at least
    ``min_interior_bins``.
    """
    x = np.asarray(native, dtype=np.float64).reshape(-1, 8)
    q = x[:, 0]
    if config.band_bins is not None:
        band_lo, band_hi = config.band_bins
    else:
        if q.size == 0:
            raise ValueError("an empty start set needs explicit band_bins")
        band_lo = int(np.floor(q.min()))
        band_hi = max(int(np.floor(q.max())) + 1, band_lo + config.min_interior_bins)
    inside = (q >= band_lo) & (q < band_hi)
    members = np.nonzero(inside)[0]
    outside = tuple(int(i) for i in np.nonzero(~inside)[0])
    edges = _tile(
        np.sort(q[members]), band_lo, band_hi, config.n_target, config.min_interior_bins
    )

    while True:
        owned = _owned(edges, q, members)
        pads = _pads(edges, owned, x, T)
        bad = _first_violation(edges, pads)
        if bad is None:
            break
        if len(pads) == 1:
            raise ValueError(
                f"band [{band_lo}, {band_hi}) is narrower than its padding ({pads[0]})"
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
    for s, ((a, b), own, pad) in enumerate(
        zip(zip(edges[:-1], edges[1:]), owned, pads)
    ):
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
    return SegmentTable(
        T=float(T), n_freq=int(n_freq), segments=tuple(segments), outside=outside
    )


__all__ = [
    "Segment",
    "SegmentTable",
    "build_segment_table",
    "doppler_bins",
    "padding_bins",
]
```

Replace the import block and `__all__` of `src/binary_galactica/segments/__init__.py` with:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_table.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/binary_galactica/segments tests/segments/test_table.py
git commit -m "Build the GLASS-style segment table"
```

---

### Task 5: The segment likelihood

**Goal:** A jitted, temperature-vectorized `K`-slot Gaussian log likelihood on a segment's padded window. Inactive slots are masked, the SNR cut is measured against a reference covariance, and each source has its SNR and a JAX-derivative Fisher matrix. Compiles are shared through traced placement and quantized window lengths.

**Files:**
- Create: `src/binary_galactica/segments/likelihood.py`
- Modify: `src/binary_galactica/segments/__init__.py`
- Test: `tests/segments/test_likelihood.py`

**Acceptance Criteria:**
- [ ] `lnlike` equals `-sum |r - sum_i h_i|^2 / C` with each `h_i` from `template_on_window`, to `rtol=1e-10`.
- [ ] NaN, inf or garbage in inactive slots leaves the value unchanged.
- [ ] An active source below SNR 5 under the reference gives `-inf`; invalid parameters give `-inf`, not NaN.
- [ ] `snr()` equals `sqrt((4/T) sum |h|^2 / ae_noise)` from an independent template, and `fisher[3, 3] == snr**2` when `C == C_ref`.
- [ ] On a full grid with an active mask on the window, `lnlike == -0.5 * DataCovariance.quadratic_form` (enchilada oracle; skips without enchilada).
- [ ] On Sangria's noise-only stream, whitened by a one-sided Welch PSD, the mean power per complex bin is `1 ± 0.03` (skips without the file).
- [ ] Two segments whose window lengths quantize alike share one compiled kernel.

**Verify:** `python -m pytest tests/segments/test_likelihood.py src/binary_galactica/segments/likelihood.py -q` → `12 passed` (11 tests + 1 doctest; the oracle and Welch tests skip if enchilada or the Sangria file is missing)

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/segments/test_likelihood.py`:

```python
import numpy as np
import pytest

from binary_galactica.glass import ae_noise, template_on_window
from binary_galactica.segments import SegmentLikelihood, quantize_bins

from .support import (
    DT,
    N_FREQ,
    N_TIME,
    SANGRIA,
    T_SHORT,
    native_source,
    requires_sangria,
)

BIN_LO, N_BINS, BW = 2900, 120, 64
SOURCES = np.array([native_source(2949.12), native_source(2960.4, lnA=-48.5)])


def _c():
    f = (BIN_LO + np.arange(N_BINS)) / T_SHORT
    return np.stack([0.5 * T_SHORT * ae_noise(f)] * 2)


def _lik(k_max=3, snr_min=5.0):
    lik = SegmentLikelihood(
        T=T_SHORT,
        t0=0.0,
        bin_lo=BIN_LO,
        n_bins=N_BINS,
        BW=BW,
        k_max=k_max,
        backend="fourier",
        snr_min=snr_min,
    )
    rng = np.random.default_rng(0)
    c = _c()
    r = np.sqrt(c / 2) * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    lik.set_data(r, c)
    lik.set_reference(c)
    return lik, r, c


def _row(sources, k_max=3, fill=np.nan):
    x = np.full(8 * k_max, fill)
    x[: sources.size] = np.ravel(sources)
    return x


def _independent(src):
    return np.asarray(
        template_on_window(
            src, T=T_SHORT, bin_lo=BIN_LO, n_bins=N_BINS, BW=BW, backend="fourier"
        )
    )


def test_quantize_bins_ladder():
    for n in range(1, 400):
        m = quantize_bins(n)
        assert m >= n
        assert m & (m - 1) == 0 or (m % 3 == 0 and (m // 3) & (m // 3 - 1) == 0)
        assert quantize_bins(m) == m


def test_masked_sum_equals_explicit_sum():
    lik, r, c = _lik()
    h = _independent(SOURCES[0]) + _independent(SOURCES[1])
    expected = -np.sum(np.abs(r - h) ** 2 / c)
    got = lik(_row(SOURCES)[None], np.array([2]))[0]
    assert got == pytest.approx(expected, rel=1e-10)
    empty = lik(_row(SOURCES[:0])[None], np.array([0]))[0]
    assert empty == pytest.approx(-np.sum(np.abs(r) ** 2 / c), rel=1e-12)


@pytest.mark.parametrize("fill", [np.nan, np.inf, 1e300, -7.0])
def test_inactive_slots_cannot_leak(fill):
    lik, _, _ = _lik()
    base = lik(_row(SOURCES[:1])[None], np.array([1]))[0]
    other = lik(_row(SOURCES[:1], fill=fill)[None], np.array([1]))[0]
    assert other == base


def test_snr_cut_and_invalid_parameters():
    lik, _, _ = _lik()
    weak = native_source(2949.12, lnA=-54.0)
    assert lik(_row(weak)[None], np.array([1]))[0] == -np.inf
    off, _, _ = _lik(snr_min=0.0)
    assert np.isfinite(off(_row(weak)[None], np.array([1]))[0])
    bad = native_source(2949.12)
    bad[4] = 1.5
    assert lik(_row(bad)[None], np.array([1]))[0] == -np.inf


def test_snr_matches_independent_inner_product_and_fisher():
    lik, _, _ = _lik()
    f = (BIN_LO + np.arange(N_BINS)) / T_SHORT
    for src in SOURCES:
        h = _independent(src)
        rho = np.sqrt(4.0 / T_SHORT * np.sum(np.abs(h) ** 2 / ae_noise(f)))
        assert lik.snr(src[None])[0] == pytest.approx(rho, rel=1e-10)
    F = lik.fisher_matrices(SOURCES)
    assert F.shape == (2, 8, 8)
    np.testing.assert_allclose(
        F, np.swapaxes(F, 1, 2), rtol=1e-10, atol=1e-10 * np.max(np.abs(F))
    )
    np.testing.assert_allclose(F[:, 3, 3], lik.snr(SOURCES) ** 2, rtol=1e-8)
    assert np.all(np.linalg.eigvalsh(F) > -1e-8 * F[:, 3, 3][:, None])


def test_temperature_batch_matches_rows():
    lik, _, _ = _lik()
    rows = np.stack([_row(SOURCES), _row(SOURCES[:1]), _row(SOURCES[:0])])
    k = np.array([2, 1, 0])
    batch = lik(rows, k)
    single = [lik(rows[i][None], k[i : i + 1])[0] for i in range(3)]
    np.testing.assert_allclose(batch, single, rtol=1e-12)


def test_compile_key_sharing():
    a = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=1000, n_bins=100, BW=64, k_max=3)
    b = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=5000, n_bins=120, BW=64, k_max=3)
    c = SegmentLikelihood(T=T_SHORT, t0=0.0, bin_lo=5000, n_bins=200, BW=64, k_max=3)
    assert a.key == b.key and a.kernels is b.kernels
    assert c.key != a.key


def test_matches_enchilada_quadratic_form():
    enchilada = pytest.importorskip("enchilada")
    lik, r, c = _lik()
    h = _independent(SOURCES[0]) + _independent(SOURCES[1])
    cov = np.ones((N_FREQ, 2, 2)) * np.eye(2)
    cov[BIN_LO : BIN_LO + N_BINS, 0, 0] = c[0]
    cov[BIN_LO : BIN_LO + N_BINS, 1, 1] = c[1]
    mask = np.zeros(N_FREQ, dtype=bool)
    mask[BIN_LO : BIN_LO + N_BINS] = True
    dc = enchilada.DataCovariance(
        cov,
        channel_names=("A", "E"),
        sample_rate_hz=1 / DT,
        num_time_samples=N_TIME,
        data_domain="frequency",
        active_mask=mask,
    )
    full = {ch: np.zeros(N_FREQ, dtype=complex) for ch in "AE"}
    for i, ch in enumerate("AE"):
        full[ch][BIN_LO : BIN_LO + N_BINS] = r[i] - h[i]
    got = lik(_row(SOURCES)[None], np.array([2]))[0]
    assert got == pytest.approx(-0.5 * dc.quadratic_form(full), rel=1e-10)


@requires_sangria
def test_noise_only_whitened_power_is_unity():
    signal = pytest.importorskip("scipy.signal")
    import sangria

    xyz, dt = sangria.combine(["obs"], ["dgb", "igb", "vgb", "mbhb"], SANGRIA)
    series = sangria.aet_series(xyz)
    n = series["A"].size
    T = n * dt
    bin_lo, n_bins = int(3e-3 * T), 20_000
    freqs = (bin_lo + np.arange(n_bins)) / T
    c = []
    r = []
    for ch in "AE":
        f_w, s_w = signal.welch(series[ch], fs=1.0 / dt, nperseg=2**16)
        c.append(0.5 * T * np.interp(freqs, f_w, s_w))
        r.append(dt * np.fft.rfft(series[ch])[bin_lo : bin_lo + n_bins])
    lik = SegmentLikelihood(T=T, t0=0.0, bin_lo=bin_lo, n_bins=n_bins, BW=16, k_max=1)
    lik.set_data(np.stack(r), np.stack(c))
    lik.set_reference(np.stack(c))
    power = -lik(np.full((1, 8), np.nan), np.array([0]))[0] / (2 * n_bins)
    assert abs(power - 1.0) < 0.03, power
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_likelihood.py -q`
Expected: `ImportError: cannot import name 'SegmentLikelihood'`

- [ ] **Step 3: Implement the likelihood**

`src/binary_galactica/segments/likelihood.py`:

```python
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
``(BW, n_pad, K, backend, fmt, response_samples)``.
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
from .config import UCBBlockConfig
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
    )


def _inverse(c: ArrayLike, shape: tuple[int, int]) -> NDArray[np.float64]:
    c = np.asarray(c, dtype=np.float64)
    if c.shape != shape:
        raise ValueError(f"covariance must have shape {shape}, got {c.shape}")
    inv = np.zeros(shape)
    ok = np.isfinite(c) & (c > 0)
    inv[ok] = 1.0 / c[ok]
    return inv


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
        backend: str = "fourier",
        fmt: str = "sangria",
        response_samples: int | None = None,
        snr_min: float = SNR_MIN,
    ) -> None:
        if int(n_bins) < 1 or int(k_max) < 1 or int(BW) < 2:
            raise ValueError("n_bins and k_max must be positive and BW at least 2")
        self.T = float(T)
        self.t0 = float(t0)
        self.bin_lo = int(bin_lo)
        self.n_bins = int(n_bins)
        self.k_max = int(k_max)
        self.n_pad = quantize_bins(self.n_bins)
        self.snr_min = float(snr_min)
        self.key = (int(BW), self.n_pad, self.k_max, backend, fmt, response_samples)
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
        """Set this turn's ``(2, n_bins)`` A/E residual and coefficient variance."""
        r = np.asarray(residual, dtype=np.complex128)
        if r.shape != (2, self.n_bins) or not np.all(np.isfinite(r)):
            raise ValueError(f"residual must be finite with shape (2, {self.n_bins})")
        self._residual = self._padded(r, np.complex128)
        self._inv_c = self._padded(_inverse(c, (2, self.n_bins)), np.float64)

    def set_reference(self, c_ref: ArrayLike) -> None:
        """Set the run-fixed ``(2, n_bins)`` covariance that defines the SNR cut."""
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
```

In `src/binary_galactica/segments/__init__.py`, add `from .likelihood import SegmentLikelihood, quantize_bins` after the config import, and add `"SegmentLikelihood"` and `"quantize_bins"` to `__all__` in sorted position.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_likelihood.py src/binary_galactica/segments/likelihood.py -q`
Expected: `12 passed`. Without enchilada or the Sangria file the corresponding tests are reported as skipped. That is acceptable only when the dependency really is absent; on this machine both are present.

- [ ] **Step 5: Commit**

```bash
git add src/binary_galactica/segments tests/segments/test_likelihood.py
git commit -m "Add the masked K-slot segment likelihood"
```

---

### Task 6: Count prior, Fisher bank and GLASS moves

**Goal:** Three pieces:
- `SegmentPrior`: a uniform count on `k_min..k_max` times `UCBPrior` per active slot.
- A turn-fixed, frequency-keyed `FisherBank` with floored eigenvalues.
- The four symmetric single-source moves (Fisher, prior redraw, fm shift, psi-phi) as impulse custom jumps.

Each move is checked for exact one-step prior invariance.

**Files:**
- Create: `src/binary_galactica/segments/prior.py`
- Create: `src/binary_galactica/segments/fisher.py`
- Create: `src/binary_galactica/segments/moves.py`
- Modify: `src/binary_galactica/segments/__init__.py`
- Test: `tests/segments/test_moves.py`

**Acceptance Criteria:**
- [ ] `SegmentPrior` equals `-log(k_max - k_min + 1) + sum_{i<k} UCBPrior.log_prior(x_i)`. It ignores inactive slots and is `-inf` outside the box or for `k` out of range.
- [ ] `floor_eigenvalues` guarantees `|v_ji| / sqrt(lambda_i) <= width_j`. `fisher_decomposition` returns `None` for non-finite input.
- [ ] `FisherBank.lookup` returns the nearest stored source within `match_bins`, otherwise `None`.
- [ ] Every move returns a new row with `k` kept and inactive slots NaN, and returns the current row at `k = 0`. A `FisherMove` proposal never maps to a different matrix than its origin.
- [ ] For each of the four moves, one step plus prior-only Metropolis on 4000 prior draws leaves all 16 slot coordinates uniform (KS `p > 1e-4`).
- [ ] All four moves register on an `RJPTSampler` (skips without impulse's `rjmcmc` branch).

**Verify:** `python -m pytest tests/segments/test_moves.py -q` → `13 passed`

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/segments/test_moves.py`:

```python
from types import SimpleNamespace

import numpy as np
import pytest

from binary_galactica.glass import ucb_prior
from binary_galactica.segments import (
    FisherBank,
    FisherMove,
    FmShiftMove,
    PriorRedrawMove,
    PsiPhiMove,
    SegmentPrior,
    fisher_decomposition,
    floor_eigenvalues,
)

from .support import rj_impulse

T = 31_536_000.0
PRIOR = ucb_prior(T=T, fmin=1000 / T, fmax=1100 / T)
WIDTHS = PRIOR.hi - PRIOR.lo
K_MAX = 2


class _Layout:
    """The RJLayout slot convention for n_shared = 0 and 8 parameters per slot."""

    ndim = 8 * K_MAX

    @staticmethod
    def k(row):
        return int(row[-1])

    @staticmethod
    def slot(j):
        return slice(8 * j, 8 * j + 8)


def _stats(row, rng, chain=0):
    view = np.asarray(row, dtype=float).view()
    view.flags.writeable = False
    return SimpleNamespace(
        current_sample=view, layout=_Layout, rng=rng, chain_index=chain
    )


def _covering_bank(rng):
    bank = FisherBank(1, match_bins=4.0)
    qs = np.arange(1000.0, 1101.0, 2.0)
    evalues, evectors = [], []
    for _ in qs:
        Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
        F = Q @ np.diag(1.0 / (0.3 * WIDTHS) ** 2) @ Q.T
        ev, vec = fisher_decomposition(F, WIDTHS)
        evalues.append(ev)
        evectors.append(vec)
    bank.store(0, qs, np.array(evalues), np.array(evectors))
    return bank


def _moves(rng):
    return {
        "fisher": FisherMove(_covering_bank(rng)),
        "prior": PriorRedrawMove(PRIOR),
        "fm_shift": FmShiftMove(T),
        "psi_phi": PsiPhiMove(),
    }


def test_segment_prior_matches_box_prior():
    rng = np.random.default_rng(1)
    sp = SegmentPrior(PRIOR, k_max=3)
    x = np.full((2, 24), np.nan)
    x[0, :16] = PRIOR.sample(rng, size=2).ravel()
    x[1, :8] = PRIOR.sample(rng)
    out = sp(x, np.array([2, 1]))
    expected = -np.log(4) + np.array(
        [
            PRIOR.log_prior(x[0, :8]) + PRIOR.log_prior(x[0, 8:16]),
            PRIOR.log_prior(x[1, :8]),
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    bad = x.copy()
    bad[0, 4] = 1.2
    assert sp(bad, np.array([2, 1]))[0] == -np.inf
    assert sp(x, np.array([4, 1]))[0] == -np.inf
    assert SegmentPrior(PRIOR, k_max=3, k_min=2)(x, np.array([2, 1]))[1] == -np.inf


def test_floor_eigenvalues_bounds_unit_steps():
    rng = np.random.default_rng(2)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    ev = floor_eigenvalues(np.array([1e-30, -1.0, np.nan, 1, 2, 3, 4, 5]), Q, WIDTHS)
    steps = np.abs(Q) / np.sqrt(ev)[None, :]
    assert np.all(steps <= WIDTHS[:, None] * (1 + 1e-12))
    assert fisher_decomposition(np.full((8, 8), np.nan), WIDTHS) is None


def test_bank_lookup_nearest_within_range():
    bank = FisherBank(2, match_bins=4.0)
    bank.store(
        1, np.array([1010.0, 1020.0]), np.ones((2, 8)), np.tile(np.eye(8), (2, 1, 1))
    )
    assert bank.lookup(1, 1013.9) == 0
    assert bank.lookup(1, 1016.0) == 1
    assert bank.lookup(1, 1030.0) is None
    assert bank.lookup(0, 1010.0) is None
    bank.clear()
    assert bank.lookup(1, 1010.0) is None


@pytest.mark.parametrize("name", ["fisher", "prior", "fm_shift", "psi_phi"])
def test_move_contract(name):
    rng = np.random.default_rng(3)
    move = _moves(rng)[name]
    row = np.full(8 * K_MAX + 1, np.nan)
    row[-1] = 0
    out, qxy = move(_stats(row, rng))
    assert qxy == 0.0 and np.array_equal(out, row, equal_nan=True)
    row[:8] = PRIOR.sample(rng)
    row[-1] = 1
    for _ in range(50):
        out, qxy = move(_stats(row, rng))
        assert (
            out[-1] == 1
            and np.all(np.isnan(out[8:-1]))
            and np.all(np.isfinite(out[:8]))
        )
        assert not np.shares_memory(out, row)


def test_fisher_move_keeps_its_matrix():
    rng = np.random.default_rng(4)
    bank = _covering_bank(rng)
    move = FisherMove(bank)
    for _ in range(500):
        src = PRIOR.sample(rng)
        before = bank.lookup(0, src[0])
        new = move.propose(src.copy(), rng, 0)
        if new is not None:
            assert bank.lookup(0, new[0]) == before


@pytest.mark.parametrize("name", ["fisher", "prior", "fm_shift", "psi_phi"])
def test_one_step_leaves_prior_invariant(name):
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(5)
    move = _moves(rng)[name]
    sp = SegmentPrior(PRIOR, k_max=K_MAX, k_min=K_MAX)
    n = 4000
    x = PRIOR.sample(rng, size=n * K_MAX).reshape(n, 8 * K_MAX)
    out = np.empty_like(x)
    for i in range(n):
        new, _ = move(_stats(np.append(x[i], K_MAX), rng))
        y = PRIOR.wrap(new[:-1].reshape(K_MAX, 8)).reshape(-1)
        out[i] = y if np.isfinite(sp(y[None], np.array([K_MAX]))[0]) else x[i]
    for d in range(8 * K_MAX):
        lo, hi = PRIOR.lo[d % 8], PRIOR.hi[d % 8]
        p = stats.kstest((out[:, d] - lo) / (hi - lo), "uniform").pvalue
        assert p > 1e-4, (name, d, p)


def test_moves_register_on_rjptsampler(tmp_path):
    impulse = rj_impulse()
    rng = np.random.default_rng(6)
    sampler = impulse.RJPTSampler(
        n_shared=0,
        component_dim=8,
        max_components=K_MAX,
        min_components=K_MAX,
        lnlike=lambda x, k: np.zeros(len(x)),
        lnprior=SegmentPrior(PRIOR, k_max=K_MAX, k_min=K_MAX),
        component_proposal=impulse.IndependentComponentProposal(
            draw=lambda r: PRIOR.sample(r), logpdf=PRIOR.log_prior
        ),
        component_periodic={2: (0.0, 2 * np.pi), 5: (0.0, np.pi), 6: (0.0, 2 * np.pi)},
        vectorized=True,
        ntemps=1,
        am_weight=0,
        scam_weight=0,
        de_weight=0,
        outdir=str(tmp_path),
        seed=1,
        verbose=False,
    )
    for move in _moves(rng).values():
        sampler.add_custom_jump(move, 1.0)
    sampler.sample(PRIOR.sample(rng, size=K_MAX).ravel(), 200, initial_k=K_MAX)
    rates = sampler.proposal_acceptance_rates()
    assert {"FisherMove", "PriorRedrawMove", "FmShiftMove", "PsiPhiMove"} <= set(
        str(rates)
    )
```

The last assertion turns the acceptance report (a nested dict) into a string. It only checks that each move name appears, without assuming the report's nesting.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_moves.py -q`
Expected: `ImportError: cannot import name 'FisherBank'`

- [ ] **Step 3: Implement the prior, the Fisher bank and the moves**

`src/binary_galactica/segments/prior.py`:

```python
"""Joint prior of one segment: uniform count times GLASS's box per source."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..glass import UCBPrior


class SegmentPrior:
    """``log p(k) + sum_{i<k} UCBPrior.log_prior(x_i)`` for padded impulse rows.

    ``p(k)`` is uniform on ``k_min .. k_max``; slots past ``k`` are ignored.
    Periodic angles are wrapped before the box test, as in
    :meth:`UCBPrior.log_prior`. Rows are ``(n, 8 * k_max)`` with ``k`` of
    shape ``(n,)``, impulse's vectorized convention.
    """

    def __init__(self, prior: UCBPrior, k_max: int, k_min: int = 0) -> None:
        if not 0 <= int(k_min) <= int(k_max):
            raise ValueError("need 0 <= k_min <= k_max")
        self.prior = prior
        self.k_max = int(k_max)
        self.k_min = int(k_min)
        self.log_pk = -math.log(self.k_max - self.k_min + 1)
        self.log_component = -prior.log_volume

    def __call__(self, x: ArrayLike, k: ArrayLike) -> NDArray[np.float64]:
        """Return the log prior of each row."""
        rows = np.atleast_2d(np.asarray(x, dtype=np.float64))
        ks = np.atleast_1d(np.asarray(k)).astype(np.int64)
        comps = rows.reshape(rows.shape[0], self.k_max, 8)
        wrapped = self.prior.wrap(comps)
        with np.errstate(invalid="ignore"):
            inside = np.all(
                (wrapped >= self.prior.lo) & (wrapped <= self.prior.hi), axis=-1
            )
        active = np.arange(self.k_max)[None, :] < ks[:, None]
        ok = np.all(inside | ~active, axis=1) & (ks >= self.k_min) & (ks <= self.k_max)
        return np.where(ok, self.log_pk + ks * self.log_component, -np.inf)


__all__ = ["SegmentPrior"]
```

`src/binary_galactica/segments/fisher.py`:

```python
"""Fisher decompositions held fixed for one Gibbs turn, keyed by frequency."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def floor_eigenvalues(
    evalues: ArrayLike, evectors: ArrayLike, widths: ArrayLike
) -> NDArray[np.float64]:
    """Raise eigenvalues so that no unit step leaves the prior widths.

    Along eigenvector ``v_i`` a unit step moves coordinate ``j`` by
    ``v_ji / sqrt(lambda_i)``, so ``lambda_i >= max_j (v_ji / w_j)**2``.
    Non-finite and non-positive eigenvalues take that floor.
    """
    ev = np.asarray(evalues, dtype=np.float64)
    vec = np.asarray(evectors, dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    floor = np.max((vec / w[:, None]) ** 2, axis=0)
    ev = np.where(np.isfinite(ev), ev, 0.0)
    return np.maximum(ev, floor)


def fisher_decomposition(
    F: ArrayLike, widths: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return floored ``(evalues, evectors)`` with ``evectors[:, i]``, or ``None``."""
    F = np.asarray(F, dtype=np.float64)
    if not np.all(np.isfinite(F)):
        return None
    evalues, evectors = np.linalg.eigh(0.5 * (F + F.T))
    return floor_eigenvalues(evalues, evectors, widths), evectors


class FisherBank:
    """Per-chain Fisher decompositions, each stored with its source's ``f0*T``.

    A source uses the stored decomposition of the same chain whose frequency is
    nearest its own, within ``match_bins``. The assignment depends only on the
    source's frequency and on matrices fixed for the turn, so a Fisher jump
    that keeps the assignment is symmetric, whatever births, deaths and swaps
    do to the slots.
    """

    def __init__(self, n_chains: int, match_bins: float) -> None:
        self.match_bins = float(match_bins)
        self._q = [np.empty(0) for _ in range(n_chains)]
        self._evalues = [np.empty((0, 8)) for _ in range(n_chains)]
        self._evectors = [np.empty((0, 8, 8)) for _ in range(n_chains)]

    def clear(self) -> None:
        """Forget every stored decomposition."""
        n = len(self._q)
        self._q = [np.empty(0) for _ in range(n)]
        self._evalues = [np.empty((0, 8)) for _ in range(n)]
        self._evectors = [np.empty((0, 8, 8)) for _ in range(n)]

    def store(
        self, chain: int, q: ArrayLike, evalues: ArrayLike, evectors: ArrayLike
    ) -> None:
        """Replace one chain's decompositions."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        self._q[chain] = q
        self._evalues[chain] = np.asarray(evalues, dtype=np.float64).reshape(q.size, 8)
        self._evectors[chain] = np.asarray(evectors, dtype=np.float64).reshape(
            q.size, 8, 8
        )

    def count(self, chain: int) -> int:
        """Return the number of decompositions stored for a chain."""
        return int(self._q[chain].size)

    def lookup(self, chain: int, q: float) -> int | None:
        """Return the index of the nearest stored source within range, else ``None``."""
        qs = self._q[chain]
        if qs.size == 0:
            return None
        d = np.abs(qs - q)
        i = int(np.argmin(d))
        return i if d[i] <= self.match_bins else None

    def decomposition(
        self, chain: int, i: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the stored ``(evalues, evectors)``."""
        return self._evalues[chain][i], self._evectors[chain][i]


__all__ = ["FisherBank", "fisher_decomposition", "floor_eigenvalues"]
```

`src/binary_galactica/segments/moves.py`:

```python
"""GLASS's single-source moves as impulse custom jumps (all symmetric).

Each move picks one active source uniformly and proposes a new value for it.
It returns the current row at ``k = 0`` or when the move declines. impulse
wraps the periodic coordinates afterwards, and the prior rejects anything
outside the box.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..glass import UCBPrior, draw_from_fisher, fm_shift, psi_phi_jump
from .fisher import FisherBank


class SlotMove:
    """Base class: uniform choice of an active slot, symmetric proposal."""

    def __call__(self, chain_stats: Any) -> tuple[NDArray[np.float64], float]:
        """Return ``(new row, 0.0)`` for impulse's ``add_custom_jump``."""
        row = np.array(chain_stats.current_sample, dtype=np.float64)
        layout = chain_stats.layout
        k = layout.k(row)
        if k == 0:
            return row, 0.0
        rng = chain_stats.rng
        sl = layout.slot(int(rng.integers(k)))
        new = self.propose(row[sl].copy(), rng, int(chain_stats.chain_index))
        if new is not None:
            row[sl] = new
        return row, 0.0

    def propose(
        self, source: NDArray[np.float64], rng: np.random.Generator, chain: int
    ) -> NDArray[np.float64] | None:
        """Return the proposed source, or ``None`` to stay."""
        raise NotImplementedError


class FisherMove(SlotMove):
    """One eigendirection of the turn-fixed Fisher matrix nearest in frequency."""

    def __init__(self, bank: FisherBank) -> None:
        self.bank = bank

    def propose(self, source, rng, chain):
        """Jump along the matched decomposition; decline if the match would change."""
        i = self.bank.lookup(chain, float(source[0]))
        if i is None:
            return None
        evalues, evectors = self.bank.decomposition(chain, i)
        new, _ = draw_from_fisher(source, rng, evalues, evectors)
        if self.bank.lookup(chain, float(new[0])) != i:
            return None
        return new


class PriorRedrawMove(SlotMove):
    """Redraw all eight parameters of the source from the segment prior."""

    def __init__(self, prior: UCBPrior) -> None:
        self.prior = prior

    def propose(self, source, rng, chain):
        """Return a fresh prior draw."""
        return self.prior.sample(rng)


class FmShiftMove(SlotMove):
    """GLASS's annual-modulation sideband hop, with longitude jitter."""

    def __init__(self, T: float) -> None:
        self.T = float(T)

    def propose(self, source, rng, chain):
        """Shift ``f0*T`` by ``s*T/yr`` and jitter the longitude."""
        return fm_shift(source, rng, self.T)[0]


class PsiPhiMove(SlotMove):
    """GLASS's continuous psi-phi0 correlation jump."""

    def propose(self, source, rng, chain):
        """Shift ``psi`` by ``+-d/2`` and ``phi0`` by ``+-d``."""
        return psi_phi_jump(source, rng)[0]


__all__ = ["FisherMove", "FmShiftMove", "PriorRedrawMove", "PsiPhiMove", "SlotMove"]
```

In `src/binary_galactica/segments/__init__.py`, add these imports after the config import:

```python
from .fisher import FisherBank, fisher_decomposition, floor_eigenvalues
from .moves import FisherMove, FmShiftMove, PriorRedrawMove, PsiPhiMove, SlotMove
from .prior import SegmentPrior
```

Add the new names to `__all__`, keeping it sorted: `"FisherBank"`, `"FisherMove"`, `"FmShiftMove"`, `"PriorRedrawMove"`, `"PsiPhiMove"`, `"SegmentPrior"`, `"SlotMove"`, `"fisher_decomposition"`, `"floor_eigenvalues"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_moves.py -q`
Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add src/binary_galactica/segments tests/segments/test_moves.py
git commit -m "Add the count prior, Fisher bank and GLASS moves"
```

---
### Task 7: Drive one segment with impulse's RJPTSampler

**Goal:** A `SegmentRunner` that owns one segment's `RJPTSampler` (GLASS ladder, the four custom moves, prior birth/death), its likelihood and its Fisher bank. Each Gibbs turn it re-calls `sample()` on the same sampler from the previous final state, refreshing the Fisher matrices first. It copies the cold chain into a `ChainArchive` after each turn.

**Files:**
- Create: `src/binary_galactica/segments/runner.py`
- Modify: `src/binary_galactica/segments/__init__.py`
- Test: `tests/segments/test_runner.py`

**Acceptance Criteria:**
- [ ] `glass_ladder(12)` is `1.2**i` for `i < 11`, followed by `1e12`.
- [ ] `ChainArchive` round-trips per-turn cold samples and reports the turn of each row.
- [ ] A fixed-count run keeps `k`, gives finite cold sources and likelihoods, stores one Fisher decomposition per chain and source, and archives `steps_per_turn / thin` rows.
- [ ] After a residual change, every chain's stored likelihood equals the likelihood recomputed for the new residual.
- [ ] A prior-only reversible-jump run at `K = 3` visits each count with frequency `0.25 ± 0.05`. The archived `f0*T` of slot 0 is uniform over the padded band (KS `p > 1e-3`).
- [ ] A fixed-count runner with no start sources raises `ValueError`.

**Verify:** `python -m pytest tests/segments/test_runner.py -q` → `6 passed`

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/segments/test_runner.py`:

```python
import numpy as np
import pytest

from binary_galactica.glass import ae_noise, template_on_window
from binary_galactica.segments import (
    ChainArchive,
    Segment,
    SegmentRunner,
    UCBBlockConfig,
    glass_ladder,
)

from .support import T_SHORT, native_source, rj_impulse

SEG = Segment(index=0, lo=2920, hi=2984, pad=4, bw=64, k_max=3, owned=(0,))
SOURCE = native_source(2949.12)


def _window(seed=0, offset=0.0):
    n = SEG.n_bins
    f = (SEG.bin_lo + np.arange(n)) / T_SHORT
    c = np.stack([0.5 * T_SHORT * ae_noise(f)] * 2)
    rng = np.random.default_rng(seed)
    noise = np.sqrt(c / 2) * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    h = np.asarray(
        template_on_window(
            SOURCE, T=T_SHORT, bin_lo=SEG.bin_lo, n_bins=n, BW=SEG.bw, backend="fourier"
        )
    )
    return noise + h + offset * np.sqrt(c), c


def _config(**kw):
    base = {"steps_per_turn": 50, "ntemps": 2, "hottest_temperature": 10.0}
    base.update(kw)
    return UCBBlockConfig(**base)


def test_glass_ladder():
    ladder = glass_ladder(12)
    np.testing.assert_allclose(ladder[:11], 1.2 ** np.arange(11), rtol=1e-15)
    assert ladder[-1] == 1e12
    assert glass_ladder(1).tolist() == [1.0]


def test_archive_round_trip(tmp_path):
    archive = ChainArchive(tmp_path)
    archive.append(0, samples=np.ones((3, 16)), k=np.full(3, 2), lnlike=np.zeros(3))
    archive.append(1, samples=np.zeros((2, 16)), k=np.full(2, 1), lnlike=np.ones(2))
    chain = archive.load()
    assert chain["samples"].shape == (5, 16)
    assert chain["turn"].tolist() == [0, 0, 0, 1, 1]
    assert chain["k"].tolist() == [2, 2, 2, 1, 1]


def test_fixed_count_smoke(tmp_path):
    rj_impulse()
    r, c = _window()
    runner = SegmentRunner(
        SEG,
        SOURCE[None],
        T=T_SHORT,
        t0=0.0,
        config=_config(birth_death=False),
        outdir=tmp_path,
        seed=1,
        reference=c,
    )
    runner.advance(r, c)
    assert runner.ks.tolist() == [1, 1]
    assert np.all(
        np.isfinite(runner.cold_sources())
    ) and runner.cold_sources().shape == (1, 8)
    assert runner.bank.count(0) == 1 and runner.bank.count(1) == 1
    chain = runner.archive.load()
    assert chain["samples"].shape == (50, 8)
    assert np.all(chain["k"] == 1) and np.all(np.isfinite(chain["lnlike"]))
    assert not list((tmp_path / "impulse").glob("chain_*"))


def test_residual_change_recomputes_likelihoods(tmp_path):
    rj_impulse()
    r, c = _window()
    runner = SegmentRunner(
        SEG,
        SOURCE[None],
        T=T_SHORT,
        t0=0.0,
        config=_config(steps_per_turn=5),
        outdir=tmp_path,
        seed=2,
        reference=c,
    )
    runner.advance(r, c)
    r2, _ = _window(offset=50.0)
    runner.advance(r2, c)
    expected = runner.likelihood(runner.positions, runner.ks)
    np.testing.assert_allclose(runner.sampler.state.lnlikes, expected, rtol=1e-12)


def test_prior_only_count_is_uniform(tmp_path):
    rj_impulse()
    stats = pytest.importorskip("scipy.stats")
    cfg = _config(ntemps=1, steps_per_turn=20_000, thin=10)
    runner = SegmentRunner(
        SEG,
        np.empty((0, 8)),
        T=T_SHORT,
        t0=0.0,
        config=cfg,
        outdir=tmp_path,
        seed=3,
        prior_only=True,
    )
    runner.advance()
    chain = runner.archive.load()
    freq = np.bincount(chain["k"], minlength=4) / chain["k"].size
    np.testing.assert_allclose(freq, 0.25, atol=0.05)
    f0 = chain["samples"][::10, 0][chain["k"][::10] >= 1]
    p = stats.kstest((f0 - SEG.bin_lo) / SEG.n_bins, "uniform").pvalue
    assert p > 1e-3


def test_fixed_count_needs_sources(tmp_path):
    with pytest.raises(ValueError, match="start source"):
        SegmentRunner(
            SEG,
            np.empty((0, 8)),
            T=T_SHORT,
            t0=0.0,
            config=_config(birth_death=False),
            outdir=tmp_path,
            seed=4,
            prior_only=True,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_runner.py -q`
Expected: `ImportError: cannot import name 'ChainArchive'`

- [ ] **Step 3: Implement the runner**

`src/binary_galactica/segments/runner.py`:

```python
"""One segment's reversible-jump sampler, continued across Gibbs turns.

impulse has no call that continues a run after the likelihood changes. Each
turn therefore calls ``RJPTSampler.sample`` again on the *same* sampler,
starting from the previous turn's final state. These survive across turns:
- the temperature ladder and its adaptation (``ptstate``, so the hottest chain
  stays fixed and the intermediate ones keep adapting);
- the RNG streams;
- the proposal bundle and the per-chain statistics.

``sample`` recomputes every chain's likelihood for the new residual and PSD
and raises if any value is not finite. impulse restarts its iteration count
and truncates its chain files on every call, so the runner copies the cold
chain into a :class:`ChainArchive` after each turn and deletes impulse's
files. ``num_adapt`` stays ``None``, because it would count per turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import UCBBlockConfig
from .fisher import FisherBank, fisher_decomposition
from .likelihood import SegmentLikelihood
from .moves import FisherMove, FmShiftMove, PriorRedrawMove, PsiPhiMove
from .prior import SegmentPrior
from .table import Segment

#: Periodic coordinates of each source and their intervals (UCBPrior's).
COMPONENT_PERIODIC = {2: (0.0, 2.0 * np.pi), 5: (0.0, np.pi), 6: (0.0, 2.0 * np.pi)}


def _impulse() -> Any:
    try:
        import impulse
    except ImportError as exc:  # pragma: no cover - exercised without impulse
        raise ImportError(
            "the segment block needs impulse-mcmc with RJPTSampler "
            "(pip install 'binary-galactica[segments]')"
        ) from exc
    if not hasattr(impulse, "RJPTSampler"):
        raise ImportError("impulse lacks RJPTSampler; install its rjmcmc branch")
    return impulse


def glass_ladder(
    ntemps: int, step: float = 1.2, hottest: float = 1.0e12
) -> NDArray[np.float64]:
    """Return GLASS's ladder ``step**i`` with the hottest chain at ``hottest``."""
    ladder = step ** np.arange(int(ntemps), dtype=np.float64)
    if ntemps > 1:
        ladder[-1] = hottest
    return ladder


class ChainArchive:
    """Cold-chain samples of one segment, one compressed file per turn."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def append(
        self, turn: int, *, samples: ArrayLike, k: ArrayLike, lnlike: ArrayLike
    ) -> None:
        """Write one turn's cold samples ``(n, D)``, counts and likelihoods."""
        np.savez_compressed(
            self.directory / f"turn_{int(turn):05d}.npz",
            samples=np.asarray(samples, dtype=np.float64),
            k=np.asarray(k, dtype=np.int64),
            lnlike=np.asarray(lnlike, dtype=np.float64),
        )

    def files(self) -> list[Path]:
        """Return the per-turn files in turn order."""
        return sorted(self.directory.glob("turn_*.npz"))

    def load(self) -> dict[str, NDArray[Any]]:
        """Concatenate every turn; ``turn`` gives each row's turn."""
        parts = [np.load(p) for p in self.files()]
        if not parts:
            raise FileNotFoundError(f"no archived turns in {self.directory}")
        turns = [int(p.stem.split("_")[1]) for p in self.files()]
        return {
            "samples": np.concatenate([z["samples"] for z in parts]),
            "k": np.concatenate([z["k"] for z in parts]),
            "lnlike": np.concatenate([z["lnlike"] for z in parts]),
            "turn": np.concatenate(
                [np.full(z["k"].shape, t, dtype=np.int64) for z, t in zip(parts, turns)]
            ),
        }


class SegmentRunner:
    """Sampler, likelihood and Fisher bank of one segment.

    ``start`` holds the segment's ``(n, 8)`` owned start sources; every chain
    starts there. With ``config.birth_death`` off, the count is fixed at
    ``n``. Otherwise it ranges over ``0 .. segment.k_max``. ``reference`` is
    the run-fixed ``(2, n_bins)`` covariance of the SNR cut. ``prior_only``
    replaces the likelihood with zero (no data, no cut, no Fisher matrices).
    """

    def __init__(
        self,
        segment: Segment,
        start: ArrayLike,
        *,
        T: float,
        t0: float,
        config: UCBBlockConfig,
        outdir: str | Path,
        seed: int,
        reference: ArrayLike | None = None,
        prior_only: bool = False,
    ) -> None:
        start = np.asarray(start, dtype=np.float64).reshape(-1, 8)
        self.segment = segment
        self.config = config
        self.T = float(T)
        self.fixed_count = not config.birth_death
        if self.fixed_count and start.shape[0] == 0:
            raise ValueError("a fixed-count segment needs at least one start source")
        impulse = _impulse()
        self.k_max = start.shape[0] if self.fixed_count else segment.k_max
        if start.shape[0] > self.k_max:
            raise ValueError(
                f"{start.shape[0]} start sources exceed k_max={self.k_max}"
            )
        k_min = self.k_max if self.fixed_count else 0
        self.prior = segment.prior(T)
        self.log_prior = SegmentPrior(self.prior, self.k_max, k_min)
        self.likelihood: SegmentLikelihood | None = None
        if not prior_only:
            if reference is None:
                raise ValueError(
                    "a runner with a likelihood needs the reference covariance"
                )
            self.likelihood = SegmentLikelihood.for_segment(
                segment, T=T, t0=t0, config=config, k_max=self.k_max
            )
            self.likelihood.set_reference(reference)
        self.bank = FisherBank(config.ntemps, config.fisher_match_bins)
        self.outdir = Path(outdir)
        self.archive = ChainArchive(self.outdir / "cold")
        self.turn = 0
        adaptive = (config.am_weight + config.scam_weight + config.de_weight) > 0
        self.sampler = impulse.RJPTSampler(
            n_shared=0,
            component_dim=8,
            max_components=self.k_max,
            min_components=k_min,
            lnlike=self._lnlike,
            lnprior=self.log_prior,
            component_proposal=impulse.IndependentComponentProposal(
                draw=self._draw, logpdf=self.prior.log_prior
            ),
            birth_death_weight=config.birth_death_weight(),
            component_redraw_weight=0.0,
            component_periodic=COMPONENT_PERIODIC,
            vectorized=True,
            jax=True,
            ntemps=config.ntemps,
            ladder=glass_ladder(
                config.ntemps, config.ladder_step, config.hottest_temperature
            ),
            am_weight=config.am_weight,
            scam_weight=config.scam_weight,
            de_weight=config.de_weight,
            buffer_size=2_000 if adaptive else 16,
            cov_update=100 if adaptive else config.steps_per_turn,
            save_freq=config.steps_per_turn,
            seed=int(seed),
            outdir=str(self.outdir / "impulse"),
            verbose=False,
        )
        moves = {
            "fisher": FisherMove(self.bank),
            "prior": PriorRedrawMove(self.prior),
            "fm_shift": FmShiftMove(self.T),
            "psi_phi": PsiPhiMove(),
        }
        for name, weight in config.within_model_weights().items():
            if weight > 0:
                self.sampler.add_custom_jump(moves[name], weight)
        x0 = np.full(8 * self.k_max, np.nan)
        x0[: start.size] = start.reshape(-1)
        self._x = np.tile(x0, (config.ntemps, 1))
        self._k = np.full(config.ntemps, start.shape[0], dtype=np.int64)

    def _draw(self, rng: np.random.Generator) -> NDArray[np.float64]:
        return self.prior.sample(rng)

    def _lnlike(
        self, x: NDArray[np.float64], k: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        if self.likelihood is None:
            return np.zeros(len(x))
        return self.likelihood(x, k)

    @property
    def positions(self) -> NDArray[np.float64]:
        """Return every chain's padded parameters ``(ntemps, 8 * k_max)``."""
        return self._x.copy()

    @property
    def ks(self) -> NDArray[np.int64]:
        """Return every chain's source count."""
        return self._k.copy()

    def cold_sources(self) -> NDArray[np.float64]:
        """Return the cold chain's active sources ``(k, 8)``."""
        k = int(self._k[0])
        return self._x[0, : 8 * k].reshape(k, 8).copy()

    def refresh_fisher(self) -> None:
        """Recompute every chain's Fisher decompositions at its current sources."""
        self.bank.clear()
        if self.likelihood is None:
            return
        rows = [self._x[c, : 8 * int(k)].reshape(-1, 8) for c, k in enumerate(self._k)]
        owners = np.concatenate([np.full(len(r), c) for c, r in enumerate(rows)])
        if owners.size == 0:
            return
        stack = np.concatenate(rows)
        matrices = self.likelihood.fisher_matrices(stack)
        widths = self.prior.hi - self.prior.lo
        for c in range(len(rows)):
            qs, evalues, evectors = [], [], []
            for src, F in zip(stack[owners == c], matrices[owners == c]):
                dec = fisher_decomposition(F, widths)
                if dec is not None:
                    qs.append(src[0])
                    evalues.append(dec[0])
                    evectors.append(dec[1])
            self.bank.store(
                c,
                np.array(qs),
                np.array(evalues).reshape(-1, 8),
                np.array(evectors).reshape(-1, 8, 8),
            )

    def advance(
        self, residual: ArrayLike | None = None, c: ArrayLike | None = None
    ) -> None:
        """Run one Gibbs turn on this turn's ``(2, n_bins)`` residual and variance."""
        if self.likelihood is not None:
            if residual is None or c is None:
                raise ValueError("advance needs the residual and its covariance")
            self.likelihood.set_data(residual, c)
        self.refresh_fisher()
        self.sampler.sample(
            self._x,
            self.config.steps_per_turn,
            thin=self.config.thin,
            initial_k=self._k,
        )
        width = 8 * self.k_max
        state = np.asarray(self.sampler.state.positions)
        self._x = np.array(state[:, :width])
        self._k = state[:, width].astype(np.int64)
        chain = self.sampler.load_chain()
        self.archive.append(
            self.turn,
            samples=chain["samples"][0],
            k=chain["k"][0],
            lnlike=chain["lnlike"][0],
        )
        for path in (self.outdir / "impulse").glob("chain_*"):
            path.unlink()
        self.turn += 1


__all__ = ["COMPONENT_PERIODIC", "ChainArchive", "SegmentRunner", "glass_ladder"]
```

In `src/binary_galactica/segments/__init__.py`, add `from .runner import COMPONENT_PERIODIC, ChainArchive, SegmentRunner, glass_ladder` and add the four names to `__all__` in sorted position. `runner.py` imports impulse only inside `_impulse()`, so importing the package still needs neither impulse nor its doctests.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_runner.py -q`
Expected: `6 passed`

If `test_residual_change_recomputes_likelihoods` fails, impulse did not recompute the likelihoods on `sample()`, and the continue bridge is invalid. Stop and report to the user; do not work around it.

- [ ] **Step 5: Commit**

```bash
git add src/binary_galactica/segments tests/segments/test_runner.py
git commit -m "Drive each segment with impulse's RJPTSampler"
```

---

### Task 8: The UCB block

**Goal:** `UCBBlock` implements enchilada's `Block` protocol structurally:
- it validates the inputs;
- it takes the A/E coefficient variances from the covariance;
- it drops start sources below the SNR cut under the reference and records them;
- it sweeps even then odd segments over the conditional residual minus the other segments' owned sources;
- it publishes the owned sources over their full band in every data channel.

`enchilada.testing.check_block` must pass.

**Files:**
- Create: `src/binary_galactica/segments/render.py`
- Create: `src/binary_galactica/segments/block.py`
- Modify: `src/binary_galactica/segments/__init__.py`
- Test: `tests/segments/test_block.py`

**Acceptance Criteria:**
- [ ] Channels other than `("A","E")` / `("A","E","T")`, time-domain data and a non-frequency covariance raise.
- [ ] Construction writes `config.json`, `segment_table.json` and `dropped_start_sources.json`. A start source below SNR 5 under the reference is dropped and recorded with its SNR.
- [ ] Segments advance in the order even then odd. Each one receives `r - (model - own)` on its window, where `model` holds the other segments' current owned sources.
- [ ] Sources in the padding are not published; `parameters["segments"][s]["count"]` counts owned sources only.
- [ ] `enchilada.testing.check_block` passes for two cycles on a toy frequency-domain L1Data, and two turns are archived.

**Verify:** `python -m pytest tests/segments/test_block.py -q` → `7 passed`

**Steps:**

- [ ] **Step 1: Write the failing tests**

`tests/segments/test_block.py`:

```python
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from binary_galactica.glass import ae_noise
from binary_galactica.segments import UCBBlock, UCBBlockConfig, render_sources

from .support import DT, N_FREQ, N_TIME, T_SHORT, ldc_rows, rj_impulse

block_module = importlib.import_module("binary_galactica.segments.block")
CHANNELS = ("A", "E", "T")
QS = (2940.3, 2952.3, 2964.3)


class _Data(SimpleNamespace):
    def block_state(self, template=None, *, covariance_matrix=None, parameters=None):
        return SimpleNamespace(template=template, parameters=parameters)


def _grid(channels=CHANNELS, domain="frequency", data=None):
    data = data or {ch: np.zeros(N_FREQ, dtype=complex) for ch in channels}
    return _Data(
        channel_names=channels,
        num_time_samples=N_TIME,
        sample_rate_hz=1 / DT,
        start_time_gps=0.0,
        data_domain=domain,
        channel_data=data,
    )


def _cov(channels=CHANNELS):
    f = np.arange(N_FREQ) / T_SHORT
    f[0] = f[1]
    m = np.zeros((N_FREQ, len(channels), len(channels)))
    for i in range(len(channels)):
        m[:, i, i] = 0.5 * T_SHORT * ae_noise(f)
    mask = np.ones(N_FREQ, dtype=bool)
    mask[0] = False
    return SimpleNamespace(
        data_domain="frequency",
        covariance_matrix=m,
        channel_names=channels,
        active_mask=mask,
    )


def _rows(qs=QS, amplitudes=None):
    amplitudes = amplitudes or [np.exp(-48.0)] * len(qs)
    rows = np.concatenate(
        [ldc_rows(Frequency=q / T_SHORT, Amplitude=a) for q, a in zip(qs, amplitudes)]
    )
    return rows


def _config(**kw):
    base = {
        "band_bins": (2930, 2980),
        "n_target": 1,
        "min_interior_bins": 8,
        "steps_per_turn": 10,
        "ntemps": 2,
        "hottest_temperature": 10.0,
    }
    base.update(kw)
    return UCBBlockConfig(**base)


class _FakeRunner:
    calls: list = []
    extra = None

    def __init__(self, segment, start, **kwargs):
        self.segment = segment
        self.start = np.array(start).reshape(-1, 8)

    def advance(self, residual, c):
        _FakeRunner.calls.append((self.segment.index, np.array(residual)))

    def cold_sources(self):
        if _FakeRunner.extra is None:
            return self.start.copy()
        return np.concatenate([self.start, _FakeRunner.extra(self.segment)])


@pytest.fixture
def fake_runner(monkeypatch):
    _FakeRunner.calls = []
    _FakeRunner.extra = None
    monkeypatch.setattr(block_module, "SegmentRunner", _FakeRunner)
    return _FakeRunner


def test_rejects_unsupported_inputs(tmp_path, fake_runner):
    with pytest.raises(ValueError, match="channels"):
        UCBBlock(
            grid=_grid(("X", "Y", "Z")),
            start_sources=_rows(),
            outdir=tmp_path,
            config=_config(),
        )
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="frequency"):
        block.sample(_grid(domain="time"), _cov(), None, rng=rng)
    with pytest.raises(TypeError, match="DataCovariance"):
        block.sample(_grid(), SimpleNamespace(data_domain="time"), None, rng=rng)


def test_writes_run_files(tmp_path, fake_runner):
    UCBBlock(grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config())
    for name in ("config.json", "segment_table.json", "dropped_start_sources.json"):
        assert (tmp_path / name).exists()
    assert UCBBlockConfig.read(tmp_path / "config.json") == _config()


def test_drops_weak_start_sources(tmp_path, fake_runner):
    rows = _rows(amplitudes=[np.exp(-48.0), 1e-23, np.exp(-48.0)])
    block = UCBBlock(
        grid=_grid(),
        start_sources=rows,
        outdir=tmp_path,
        config=_config(),
        reference_covariance=_cov(),
    )
    dropped = json.loads((tmp_path / "dropped_start_sources.json").read_text())
    assert [d["index"] for d in dropped] == [1]
    assert dropped[0]["reason"] == "snr below cut" and dropped[0]["snr"] < 5
    assert block.owned_start[1] == []


def test_sweep_order_and_conditional_residual(tmp_path, fake_runner):
    # Zero data: segment 1 must then see exactly minus its neighbours' sources.
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    block.sample(_grid(), _cov(), None, rng=np.random.default_rng(1))
    assert [s for s, _ in fake_runner.calls] == [0, 2, 1]
    seg = block.table.segments[1]
    others = sum(
        render_sources(
            block.start_native[[i]],
            T=T_SHORT,
            t0=0.0,
            BW=block.table.segments[i].bw,
            n_channels=3,
            n_freq=N_FREQ,
        ).window(seg.bin_lo, seg.bin_hi)[:2]
        for i in (0, 2)
    )
    assert np.max(np.abs(others)) > 0
    scale = np.max(np.abs(others))
    np.testing.assert_allclose(
        fake_runner.calls[2][1], -others, rtol=1e-12, atol=1e-12 * scale
    )


def test_publishes_owned_sources_only(tmp_path, fake_runner):
    def padding_source(segment):
        src = block.start_native[segment.owned[0]].copy()
        src[0] = segment.bin_lo + 0.5
        return src[None]

    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    fake_runner.extra = padding_source
    state = block.sample(_grid(), _cov(), None, rng=np.random.default_rng(2))
    counts = [s["count"] for s in state.parameters["segments"]]
    assert counts == [1, 1, 1]
    assert set(state.template) == set(CHANNELS)
    first = state.parameters["segments"][0]
    np.testing.assert_allclose(first["ldc"]["Frequency"], QS[0] / T_SHORT, rtol=1e-12)


def test_initial_block_state_renders_the_start_set(tmp_path, fake_runner):
    block = UCBBlock(
        grid=_grid(), start_sources=_rows(), outdir=tmp_path, config=_config()
    )
    state = block.initial_block_state(_grid())
    expected = sum(
        render_sources(
            block.start_native[[i]],
            T=T_SHORT,
            t0=0.0,
            BW=block.table.segments[i].bw,
            n_channels=3,
            n_freq=N_FREQ,
        ).window(0, N_FREQ)
        for i in range(3)
    )
    for j, ch in enumerate(CHANNELS):
        np.testing.assert_allclose(
            state.template[ch], expected[j], rtol=1e-12, atol=1e-30
        )


def test_check_block(tmp_path):
    enchilada = pytest.importorskip("enchilada")
    from enchilada.testing import check_block

    rj_impulse()
    kwargs = {
        "sample_rate_hz": 1 / DT,
        "tdi_generation": "1.5",
        "physical_observable": "fractional_frequency",
        "channel_names": CHANNELS,
        "num_time_samples": N_TIME,
        "data_domain": "frequency",
    }
    grid = enchilada.L1Data(
        {ch: np.zeros(N_FREQ, dtype=complex) for ch in CHANNELS}, **kwargs
    )

    class _Noise:
        def psd(self, f):
            return ae_noise(f)

    cov = enchilada.DataCovariance.from_psd(grid, _Noise())
    config = UCBBlockConfig(
        band_bins=(2920, 2984), steps_per_turn=10, ntemps=2, hottest_temperature=10.0
    )
    rows = _rows(qs=(2952.3,))
    block = UCBBlock(grid=grid, start_sources=rows, outdir=tmp_path, config=config)
    signal = block.initial_block_state(grid).template
    rng = np.random.default_rng(3)
    data = {}
    for i, ch in enumerate(CHANNELS):
        var = cov.covariance_matrix[:, i, i].real
        noise = np.sqrt(var / 2) * (
            rng.standard_normal(N_FREQ) + 1j * rng.standard_normal(N_FREQ)
        )
        noise[0] = 0.0
        noise[-1] = np.sqrt(var[-1]) * rng.standard_normal()
        data[ch] = signal[ch] + noise
    observed = enchilada.L1Data(data, **kwargs)
    check_block(
        block,
        observed,
        num_cycles=2,
        initial_block_state=block.initial_block_state(observed),
        initial_noise_covariance=cov,
    )
    assert block.turn == 2
    assert len(list((tmp_path / "segment_00000" / "cold").glob("turn_*.npz"))) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/segments/test_block.py -q`
Expected: `ImportError: cannot import name 'UCBBlock'`

- [ ] **Step 3: Implement rendering**

`src/binary_galactica/segments/render.py`:

```python
"""Owned sources rendered over their full band on the rfft grid.

Templates are in ``dt * rfft`` units: GLASS's raw output times ``sqrt(T)``.
They are given in A and E, and in GLASS's ``T = (X + Y + Z) / sqrt(3)`` when
the data has a third channel.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..glass import GLASS_TO_CONTINUOUS, ucb_waveform
from ..glass.constants import INV_SQ3
from .likelihood import quantize_bins


@dataclass(frozen=True)
class Contribution:
    """A sparse block of the published template on bins ``bin_lo .. bin_hi - 1``."""

    bin_lo: int
    values: NDArray[np.complex128]

    @property
    def bin_hi(self) -> int:
        """Return one past the last bin."""
        return self.bin_lo + self.values.shape[1]

    @classmethod
    def empty(cls, n_channels: int) -> Contribution:
        """Return a contribution with no bins."""
        return cls(0, np.zeros((n_channels, 0), dtype=np.complex128))

    def add_to(self, model: NDArray[np.complex128], sign: float = 1.0) -> None:
        """Add ``sign`` times this contribution into a full ``(channels, n_freq)`` model."""
        if self.values.shape[1]:
            model[:, self.bin_lo : self.bin_hi] += sign * self.values

    def window(self, bin_lo: int, bin_hi: int) -> NDArray[np.complex128]:
        """Return the contribution on ``[bin_lo, bin_hi)``, zero elsewhere."""
        out = np.zeros((self.values.shape[0], bin_hi - bin_lo), dtype=np.complex128)
        lo, hi = max(bin_lo, self.bin_lo), min(bin_hi, self.bin_hi)
        if hi > lo:
            out[:, lo - bin_lo : hi - bin_lo] = self.values[
                :, lo - self.bin_lo : hi - self.bin_lo
            ]
        return out


@functools.lru_cache(maxsize=None)
def _renderer(
    BW: int, backend: str, fmt: str, response_samples: int | None
) -> Callable[..., Any]:
    def one(p, T, t0):
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
        t = (w["X"] + w["Y"] + w["Z"]) * INV_SQ3
        return w["freq_bin_indices"], c * jnp.stack([w["A"], w["E"], t])

    return jax.jit(jax.vmap(one, in_axes=(0, None, None)))


def render_sources(
    sources: ArrayLike,
    *,
    T: float,
    t0: float,
    BW: int,
    n_channels: int,
    n_freq: int,
    backend: str = "fourier",
    fmt: str = "sangria",
    response_samples: int | None = None,
) -> Contribution:
    """Render ``(n, 8)`` sources in A, E (and T if ``n_channels == 3``).

    Bins outside ``[1, n_freq - 1)`` are dropped, so DC and Nyquist stay zero.
    """
    if n_channels not in (2, 3):
        raise ValueError("n_channels must be 2 (A, E) or 3 (A, E, T)")
    x = np.asarray(sources, dtype=np.float64).reshape(-1, 8)
    n = x.shape[0]
    if n == 0:
        return Contribution.empty(n_channels)
    batch = np.concatenate([x, np.repeat(x[:1], quantize_bins(n) - n, axis=0)])
    idx, vals = _renderer(int(BW), backend, fmt, response_samples)(
        jnp.asarray(batch), jnp.float64(T), jnp.float64(t0)
    )
    idx = np.asarray(idx)[:n]
    vals = np.asarray(vals)[:n, :n_channels]
    if not (np.all(np.isfinite(idx)) and np.all(np.isfinite(vals))):
        raise ValueError("cannot render sources with invalid parameters")
    idx = idx.astype(np.int64)
    lo = max(int(idx.min()), 1)
    hi = min(int(idx.max()) + 1, n_freq - 1)
    out = np.zeros((n_channels, max(hi - lo, 0)), dtype=np.complex128)
    for i in range(n):
        ok = (idx[i] >= lo) & (idx[i] < hi)
        np.add.at(out, (slice(None), idx[i][ok] - lo), vals[i][:, ok])
    return Contribution(lo, out)


__all__ = ["Contribution", "render_sources"]
```

- [ ] **Step 4: Implement the block**

`src/binary_galactica/segments/block.py`:

```python
"""Every segment of the band in one enchilada block; one Gibbs turn per call.

One :meth:`UCBBlock.sample` call:
1. checks that the residual and covariance are frequency-domain and on the
   construction grid;
2. takes the A and E coefficient variances ``C = (T/2) S`` from the
   covariance;
3. sweeps even segments, then odd. Each segment's sampler sees the conditional
   residual minus the other segments' current owned sources, on its padded
   window;
4. publishes the owned sources over their full band in every data channel.

The block implements enchilada's ``Block`` protocol structurally and never
imports enchilada. States are built with ``conditional_residual.block_state``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .catalogue import ldc_to_native, native_to_ldc
from .config import UCBBlockConfig
from .likelihood import SegmentLikelihood
from .render import Contribution, render_sources
from .runner import SegmentRunner
from .table import Segment, build_segment_table

#: Channel layouts the block accepts; A and E are always rows 0 and 1.
CHANNEL_SETS = (("A", "E"), ("A", "E", "T"))


def _names(rows: Any, n: int) -> list[str]:
    try:
        raw = np.asarray(rows["Name"]).reshape(-1)
    except (KeyError, ValueError, IndexError, TypeError):
        return [f"source_{i}" for i in range(n)]
    return [x.decode() if isinstance(x, bytes) else str(x) for x in raw]


class UCBBlock:
    """GLASS-style UCB segments as one enchilada block (frequency domain).

    Parameters
    ----------
    grid
        An L1Data in any domain. It fixes ``num_time_samples``,
        ``sample_rate_hz``, ``start_time_gps`` and ``channel_names``, which must
        be ``("A", "E")`` or ``("A", "E", "T")``.
    start_sources
        LDC catalogue rows with the ``sky/dgb/cat`` fields; ``Name`` is optional.
    outdir
        Run directory for the configuration, segment table, dropped sources and
        chains.
    config
        :class:`UCBBlockConfig`; the default holds the spec's settings.
    reference_covariance
        Optional frequency ``DataCovariance`` defining the SNR cut for the run.
        ``None`` takes the first turn's covariance.
    name
        Registration name.

    Register with ``Wheel.add(..., data_domain="frequency")``.
    """

    def __init__(
        self,
        *,
        grid: Any,
        start_sources: Any,
        outdir: str | Path,
        config: UCBBlockConfig | None = None,
        reference_covariance: Any = None,
        name: str = "ucb",
    ) -> None:
        self.name = name
        self.config = config if config is not None else UCBBlockConfig()
        channels = tuple(grid.channel_names)
        if channels not in CHANNEL_SETS:
            raise ValueError(f"channels must be one of {CHANNEL_SETS}, got {channels}")
        self.channels = channels
        self.num_time_samples = int(grid.num_time_samples)
        self.sample_rate_hz = float(grid.sample_rate_hz)
        self.T = self.num_time_samples / self.sample_rate_hz
        self.t0 = float(grid.start_time_gps)
        self.n_freq = self.num_time_samples // 2 + 1
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.start_native = ldc_to_native(start_sources, self.T)
        self.start_names = _names(start_sources, self.start_native.shape[0])
        self.table = build_segment_table(
            self.start_native, T=self.T, n_freq=self.n_freq, config=self.config
        )
        self.config.write(self.outdir / "config.json")
        self.table.write(self.outdir / "segment_table.json")
        self.dropped: list[dict[str, Any]] = [
            self._record(i, None, "outside band") for i in self.table.outside
        ]
        self.owned_start = {seg.index: list(seg.owned) for seg in self.table.segments}
        self.c_ref: NDArray[np.float64] | None = None
        self.turn = 0
        self._runners: dict[int, SegmentRunner | None] | None = None
        self._sources: dict[int, NDArray[np.float64]] = {}
        self._contrib: dict[int, Contribution] = {}
        self._model: NDArray[np.complex128] | None = None
        if reference_covariance is not None:
            self._set_reference(self._diagonals(reference_covariance))
        self._write_dropped()

    # ---- inputs -----------------------------------------------------------

    def _record(self, i: int, snr: float | None, reason: str) -> dict[str, Any]:
        ldc = native_to_ldc(self.start_native[i], self.T)
        return {
            "index": int(i),
            "name": self.start_names[i],
            "reason": reason,
            "snr": snr,
            "native": self.start_native[i].tolist(),
            "ldc": {key: float(value[0]) for key, value in ldc.items()},
        }

    def _write_dropped(self) -> None:
        text = json.dumps(self.dropped, indent=1)
        (self.outdir / "dropped_start_sources.json").write_text(text + "\n")

    def _check_data(self, data: Any) -> None:
        if getattr(data, "data_domain", None) != "frequency":
            raise ValueError(
                "the UCB block needs frequency-domain data; register it with "
                "data_domain='frequency'"
            )
        if (
            tuple(data.channel_names) != self.channels
            or int(data.num_time_samples) != self.num_time_samples
            or float(data.sample_rate_hz) != self.sample_rate_hz
        ):
            raise ValueError("data grid or channels differ from the construction grid")

    def _diagonals(self, cov: Any) -> NDArray[np.float64]:
        """Return ``(2, n_freq)`` A/E coefficient variances; inactive bins are inf."""
        if getattr(cov, "data_domain", None) != "frequency" or not hasattr(
            cov, "covariance_matrix"
        ):
            raise TypeError(
                "noise_covariance must be a frequency-domain DataCovariance"
            )
        if tuple(cov.channel_names) != self.channels:
            raise ValueError("covariance channels differ from the data channels")
        m = np.asarray(cov.covariance_matrix)
        n = len(self.channels)
        if m.shape != (self.n_freq, n, n):
            raise ValueError(f"covariance must have shape {(self.n_freq, n, n)}")
        c = np.stack([m[:, 0, 0].real, m[:, 1, 1].real]).astype(np.float64)
        mask = getattr(cov, "active_mask", None)
        if mask is not None:
            c[:, ~np.asarray(mask, dtype=bool)] = np.inf
        lo, hi = self.table.segments[0].bin_lo, self.table.segments[-1].bin_hi
        scale = np.sqrt(np.abs(c[0, lo:hi] * c[1, lo:hi]))
        if np.any(np.abs(m[lo:hi, 0, 1]) > 1e-12 * scale):
            raise ValueError("the UCB block needs uncorrelated A and E noise")
        return c

    def _set_reference(self, c_ref: NDArray[np.float64]) -> None:
        """Fix the SNR-cut covariance and drop start sources below the cut."""
        self.c_ref = c_ref
        for seg in self.table.segments:
            own = self.owned_start[seg.index]
            if not own:
                continue
            lik = SegmentLikelihood.for_segment(
                seg, T=self.T, t0=self.t0, config=self.config
            )
            lik.set_reference(c_ref[:, seg.bin_lo : seg.bin_hi])
            keep = []
            for i, rho in zip(own, lik.snr(self.start_native[own])):
                if rho >= self.config.snr_min:
                    keep.append(i)
                else:
                    self.dropped.append(self._record(i, float(rho), "snr below cut"))
            self.owned_start[seg.index] = keep
        self._write_dropped()

    # ---- model ------------------------------------------------------------

    def _render(self, seg: Segment, sources: NDArray[np.float64]) -> Contribution:
        return render_sources(
            sources,
            T=self.T,
            t0=self.t0,
            BW=seg.bw,
            n_channels=len(self.channels),
            n_freq=self.n_freq,
            backend=self.config.backend,
            response_samples=self.config.response_samples,
        )

    def _start_sources(self) -> dict[int, NDArray[np.float64]]:
        return {
            seg.index: self.start_native[self.owned_start[seg.index]].reshape(-1, 8)
            for seg in self.table.segments
        }

    def _model_of(self, contrib: dict[int, Contribution]) -> NDArray[np.complex128]:
        model = np.zeros((len(self.channels), self.n_freq), dtype=np.complex128)
        for part in contrib.values():
            part.add_to(model)
        return model

    def _template(
        self, model: NDArray[np.complex128]
    ) -> dict[str, NDArray[np.complex128]]:
        return {ch: model[i].copy() for i, ch in enumerate(self.channels)}

    def _parameters(self, sources: dict[int, NDArray[np.float64]]) -> dict[str, Any]:
        segments = []
        for seg in self.table.segments:
            x = sources[seg.index].reshape(-1, 8)
            segments.append(
                {
                    "index": seg.index,
                    "lo": seg.lo,
                    "hi": seg.hi,
                    "count": int(x.shape[0]),
                    "native": x.copy(),
                    "ldc": native_to_ldc(x, self.T),
                }
            )
        return {"turn": self.turn, "segments": segments}

    def initial_block_state(self, data: Any) -> Any:
        """Return the start set's published state on ``data``'s grid."""
        sources = self._start_sources()
        contrib = {
            seg.index: self._render(seg, sources[seg.index])
            for seg in self.table.segments
        }
        return data.block_state(
            self._template(self._model_of(contrib)),
            parameters=self._parameters(sources),
        )

    def _start(self, rng: np.random.Generator) -> None:
        assert self.c_ref is not None
        self._sources = self._start_sources()
        self._runners = {}
        for seg in self.table.segments:
            start = self._sources[seg.index]
            if not self.config.birth_death and start.shape[0] == 0:
                self._runners[seg.index] = None
                continue
            self._runners[seg.index] = SegmentRunner(
                seg,
                start,
                T=self.T,
                t0=self.t0,
                config=self.config,
                outdir=self.outdir / f"segment_{seg.index:05d}",
                seed=int(rng.integers(2**32)),
                reference=self.c_ref[:, seg.bin_lo : seg.bin_hi],
            )
        self._contrib = {
            seg.index: self._render(seg, self._sources[seg.index])
            for seg in self.table.segments
        }
        self._model = self._model_of(self._contrib)

    # ---- Block protocol ---------------------------------------------------

    def sample(
        self,
        conditional_residual: Any,
        noise_covariance: Any,
        current_block_state: Any,
        *,
        rng: np.random.Generator,
    ) -> Any:
        """Run one Gibbs turn and return the published block state."""
        data = conditional_residual
        self._check_data(data)
        c = self._diagonals(noise_covariance)
        if self.c_ref is None:
            self._set_reference(c)
        if self._runners is None:
            self._start(rng)
        assert self._runners is not None and self._model is not None
        r = np.stack(
            [
                np.asarray(data.channel_data[ch], dtype=np.complex128)
                for ch in ("A", "E")
            ]
        )
        for parity in (0, 1):
            for seg in self.table.segments[parity::2]:
                runner = self._runners[seg.index]
                if runner is None:
                    continue
                lo, hi = seg.bin_lo, seg.bin_hi
                own = self._contrib[seg.index].window(lo, hi)[:2]
                runner.advance(
                    r[:, lo:hi] - (self._model[:2, lo:hi] - own), c[:, lo:hi]
                )
                cold = runner.cold_sources()
                keep = cold[(cold[:, 0] >= seg.lo) & (cold[:, 0] < seg.hi)]
                new = self._render(seg, keep)
                self._contrib[seg.index].add_to(self._model, -1.0)
                new.add_to(self._model, 1.0)
                self._contrib[seg.index] = new
                self._sources[seg.index] = keep
        self._model = self._model_of(self._contrib)
        self.turn += 1
        return data.block_state(
            self._template(self._model), parameters=self._parameters(self._sources)
        )


__all__ = ["CHANNEL_SETS", "UCBBlock"]
```

In `src/binary_galactica/segments/__init__.py`, add:

```python
from .block import CHANNEL_SETS, UCBBlock
from .render import Contribution, render_sources
```

Add `"CHANNEL_SETS"`, `"Contribution"`, `"UCBBlock"` and `"render_sources"` to `__all__` in sorted position.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/segments/test_block.py -q`
Expected: `7 passed`

- [ ] **Step 6: Run the whole segment suite and the package tests**

Run: `python -m pytest tests/segments src/binary_galactica/segments -q`
Expected: every test passes; the only skips are for absent optional data.

Run: `python -m pytest -q`
Expected: the existing suite still passes (1159 tests before this branch, plus the new ones).

- [ ] **Step 7: Commit**

```bash
git add src/binary_galactica/segments tests/segments/test_block.py
git commit -m "Add the UCB segment block for enchilada"
```

---
### Task 9: Isolated synthetic source at fixed count

**Goal:** A tool that injects one bright source (SNR ≈ 38) into Gaussian noise with `C = (T/2) ae_noise` on one 12-month segment window. It runs `SegmentRunner` at fixed count and checks that the cold-chain posterior covers the injection, with widths consistent with the Fisher matrix. It also adds the shared helpers the later tools use.

**Files:**
- Create: `tools/segments/runs.py`
- Create: `tools/segments/isolated_source.py`
- Create (generated): `tools/performance/segments-isolated.json`
- Test: `tests/segments/test_isolated_source.py`

**Acceptance Criteria:**
- [ ] `python tools/segments/isolated_source.py` prints `PASS` and writes `tools/performance/segments-isolated.json`. The checks are:
  - `|posterior mean offset| / posterior std < 4` for all 8 parameters;
  - posterior std / Fisher sigma in `[0.6, 1.6]` for `f0*T`, `sin(beta)` and longitude.
- [ ] The short test run keeps every mean offset within 6 Fisher sigma and reports a finite time per step.

**Verify:** `python -m pytest tests/segments/test_isolated_source.py -q` → `1 passed`; then `python tools/segments/isolated_source.py` → last line `PASS`

**Steps:**

- [ ] **Step 1: Write the failing test**

`tests/segments/test_isolated_source.py`:

```python
import numpy as np

from .support import rj_impulse


def test_short_isolated_run(tmp_path):
    rj_impulse()
    import isolated_source

    report = isolated_source.run(turns=2, steps=200, ntemps=4, outdir=tmp_path)
    offset = np.asarray(report["mean_offset"]) / np.asarray(report["fisher_sigma"])
    assert np.all(np.abs(offset) < 6), offset
    assert np.isfinite(report["seconds_per_step"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/segments/test_isolated_source.py -q`
Expected: `ModuleNotFoundError: No module named 'isolated_source'`

- [ ] **Step 3: Implement the shared helpers**

`tools/segments/runs.py`:

```python
"""Shared helpers for the segment validation tools (application side)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sangria  # enables x64 before the package is used

from binary_galactica.glass import FSTAR, ae_noise, analytic_snr
from binary_galactica.segments import CATALOGUE_FIELDS

PERFORMANCE = Path(__file__).resolve().parents[1] / "performance"
CHANNELS = ("A", "E", "T")
PERIODS = {2: 2.0 * np.pi, 5: np.pi, 6: 2.0 * np.pi}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_report(name: str, report: dict) -> Path:
    """Write a JSON report under ``tools/performance``."""
    PERFORMANCE.mkdir(parents=True, exist_ok=True)
    path = PERFORMANCE / name
    path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    return path


def analytic_snrs(freq: np.ndarray, amp: np.ndarray, T: float) -> np.ndarray:
    """GLASS's analytic SNR estimate without confusion noise."""
    return analytic_snr(amp, ae_noise(freq), np.sin(freq / FSTAR), np.sqrt(T))


def catalogue_rows(
    path: Path, *, T: float, band=None, snr_min=None, populations=("dgb", "igb")
) -> np.ndarray:
    """Return dgb/igb rows in ``band`` (bins) above ``snr_min``, sorted by frequency."""
    dtype = [("Name", "U32")] + [(name, "f8") for name in CATALOGUE_FIELDS]
    parts = []
    for population in populations:
        cat = sangria.load_catalogue(population, ("Frequency", "Amplitude"), path)
        keep = np.ones(cat.shape[0], dtype=bool)
        q = cat["Frequency"] * T
        if band is not None:
            keep &= (q >= band[0]) & (q < band[1])
        if snr_min is not None:
            keep &= analytic_snrs(cat["Frequency"], cat["Amplitude"], T) >= snr_min
        rows = sangria.load_catalogue(
            population, ("Name", *CATALOGUE_FIELDS), path, index=np.nonzero(keep)[0]
        )
        out = np.zeros(rows.shape[0], dtype=dtype)
        out["Name"] = [
            x.decode() if isinstance(x, bytes) else str(x) for x in rows["Name"]
        ]
        for name in CATALOGUE_FIELDS:
            out[name] = rows[name]
        parts.append(out)
    rows = np.concatenate(parts)
    return rows[np.argsort(rows["Frequency"], kind="stable")]


def running_median_variance(
    spectra: dict, lo: int, hi: int, half_width: int = 500
) -> dict:
    """Return each channel's coefficient variance on ``[lo, hi)``.

    The estimate is the running median of ``|r|^2`` divided by ``ln 2``, the
    median of an exponential variable with unit mean.
    """
    from scipy.ndimage import median_filter

    a, b = max(lo - half_width, 1), hi + half_width
    out = {}
    for ch, s in spectra.items():
        power = np.abs(s[a:b]) ** 2
        smooth = median_filter(power, size=2 * half_width + 1, mode="nearest")
        out[ch] = smooth[lo - a : hi - a] / np.log(2.0)
    return out


def frequency_data(spectra: dict, n: int, dt: float, channels=CHANNELS):
    """Return a frequency-domain enchilada L1Data of the given spectra."""
    import enchilada

    return enchilada.L1Data(
        {ch: spectra[ch] for ch in channels},
        sample_rate_hz=1.0 / dt,
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        channel_names=channels,
        num_time_samples=n,
        data_domain="frequency",
    )


def covariance(data, variances: dict, lo: int, hi: int):
    """Return a diagonal DataCovariance.

    It uses ``variances`` on ``[lo, hi)`` and ``(T/2) ae_noise`` elsewhere;
    DC is inactive.
    """
    import enchilada

    n_freq = data.num_time_samples // 2 + 1
    T = data.Tobs
    f = np.arange(n_freq) / T
    f[0] = f[1]
    base = 0.5 * T * ae_noise(f)
    m = np.zeros((n_freq, len(data.channel_names), len(data.channel_names)))
    for i, ch in enumerate(data.channel_names):
        diag = base.copy()
        diag[lo:hi] = variances[ch]
        m[:, i, i] = diag
    mask = np.ones(n_freq, dtype=bool)
    mask[0] = False
    return enchilada.DataCovariance(
        m,
        channel_names=data.channel_names,
        sample_rate_hz=data.fs,
        num_time_samples=data.N,
        data_domain="frequency",
        active_mask=mask,
    )


def run_block(block, data, cov, turns: int, seed: int):
    """Run ``turns`` Gibbs turns of one block alone; return the state and seconds per turn."""
    rng = np.random.default_rng(seed)
    state = block.initial_block_state(data)
    seconds = []
    for _ in range(turns):
        start = time.perf_counter()
        state = block.sample(data, cov, state, rng=rng)
        seconds.append(time.perf_counter() - start)
    return state, seconds


def recovery(
    chain: dict, injections: np.ndarray, *, burn: float = 0.25, tol: float = 1.0
):
    """Return, for each injection, how often a cold-chain source sits within ``tol`` bins.

    Also returns the histogram of the count after burn-in.
    """
    first = int(burn * chain["k"].size)
    samples, ks = chain["samples"][first:], chain["k"][first:]
    comps = samples.reshape(samples.shape[0], -1, 8)
    rows = np.arange(samples.shape[0])
    out = []
    for inj in np.atleast_2d(injections):
        d = np.abs(comps[..., 0] - inj[0])
        d = np.where(np.isnan(d), np.inf, d)
        j = np.argmin(d, axis=1)
        near = d[rows, j]
        hit = near < tol
        out.append(
            {
                "q": float(inj[0]),
                "lnA": float(inj[3]),
                "recovered": float(hit.mean()) if hit.size else 0.0,
                "median_dq": float(np.median(near[hit])) if hit.any() else None,
                "median_dlnA": float(np.median(comps[rows, j, 3][hit] - inj[3]))
                if hit.any()
                else None,
            }
        )
    return out, np.bincount(ks).tolist()
```

- [ ] **Step 4: Implement the isolated-source tool**

`tools/segments/isolated_source.py`:

```python
"""Fixed-count check on one isolated synthetic source over 12 months.

Inject one source (SNR ~38) into Gaussian noise with C = (T/2) ae_noise on a
single segment window, run SegmentRunner at fixed count, and compare the
cold-chain posterior with the injection and with the Fisher matrix.

    PYTHONPATH=src:... python tools/segments/isolated_source.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import runs  # enables x64 before the package is used

from binary_galactica.glass import ae_noise, template_on_window, ucb_fdot
from binary_galactica.segments import SegmentRunner, UCBBlockConfig, build_segment_table

T = 31_536_000.0
N_FREQ = 3_153_601


def injection(f0: float = 3e-3, lnA: float = -50.0) -> np.ndarray:
    """Return the injected native source."""
    return np.array(
        [f0 * T + 0.3, 0.2, 1.1, lnA, 0.3, 0.7, 2.0, ucb_fdot(0.3, f0) * T**2]
    )


def run(
    *, turns: int = 20, steps: int = 1000, ntemps: int = 12, seed: int = 11, outdir=None
):
    """Run the check and return the report."""
    src = injection()
    q = int(np.floor(src[0]))
    config = UCBBlockConfig(
        band_bins=(q - 40, q + 40),
        birth_death=False,
        steps_per_turn=steps,
        ntemps=ntemps,
    )
    seg = build_segment_table(src[None], T=T, n_freq=N_FREQ, config=config).segments[0]
    f = (seg.bin_lo + np.arange(seg.n_bins)) / T
    c = np.stack([0.5 * T * ae_noise(f)] * 2)
    rng = np.random.default_rng(seed)
    noise = np.sqrt(c / 2) * (
        rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape)
    )
    h = np.asarray(
        template_on_window(
            src,
            T=T,
            bin_lo=seg.bin_lo,
            n_bins=seg.n_bins,
            BW=seg.bw,
            backend=config.backend,
        )
    )
    outdir = (
        Path(outdir)
        if outdir is not None
        else Path(tempfile.mkdtemp(prefix="isolated-"))
    )
    runner = SegmentRunner(
        seg,
        src[None],
        T=T,
        t0=0.0,
        config=config,
        outdir=outdir,
        seed=seed,
        reference=c,
    )
    start = time.perf_counter()
    for _ in range(turns):
        runner.advance(h + noise, c)
    seconds = time.perf_counter() - start
    chain = runner.archive.load()
    x = chain["samples"][int(0.25 * chain["k"].size) :]
    d = x - src
    for i, period in runs.PERIODS.items():
        d[:, i] = (d[:, i] + period / 2) % period - period / 2
    mean, std = d.mean(axis=0), d.std(axis=0)
    sigma = np.sqrt(
        np.diag(np.linalg.inv(runner.likelihood.fisher_matrices(src[None])[0]))
    )
    ratio = std / sigma
    checks = {
        "mean_within_4_std": bool(np.all(np.abs(mean / std) < 4)),
        "widths_match_fisher": bool(np.all((ratio[:3] > 0.6) & (ratio[:3] < 1.6))),
    }
    return {
        "injection": src,
        "snr": float(runner.likelihood.snr(src[None])[0]),
        "segment": {"bin_lo": seg.bin_lo, "n_bins": seg.n_bins, "bw": seg.bw},
        "mean_offset": mean,
        "std": std,
        "fisher_sigma": sigma,
        "width_ratio": ratio,
        "checks": checks,
        "turns": turns,
        "steps": steps,
        "ntemps": ntemps,
        "seconds": seconds,
        "seconds_per_step": seconds / (turns * steps),
        "acceptance": runner.sampler.proposal_acceptance_rates(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--ntemps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    report = run(turns=args.turns, steps=args.steps, ntemps=args.ntemps, seed=args.seed)
    path = runs.write_report("segments-isolated.json", report)
    names = ["f0T", "sinb", "lam", "lnA", "cosi", "psi", "phi0", "fdotT2"]
    for i, name in enumerate(names):
        print(
            f"{name:7s} offset={report['mean_offset'][i]:+.3e} std={report['std'][i]:.3e} "
            f"fisher={report['fisher_sigma'][i]:.3e} ratio={report['width_ratio'][i]:.2f}"
        )
    print(
        f"snr={report['snr']:.1f}  {report['seconds_per_step'] * 1e3:.2f} ms/step  -> {path}"
    )
    ok = all(report["checks"].values())
    print("PASS" if ok else f"FAIL {report['checks']}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test, then the tool**

Run: `python -m pytest tests/segments/test_isolated_source.py -q`
Expected: `1 passed`

Run: `python tools/segments/isolated_source.py`
Expected: eight parameter lines, the SNR and timing line, then `PASS`. On `FAIL`, use superpowers-extended-cc:systematic-debugging. Do not widen the bounds.

- [ ] **Step 6: Commit**

```bash
git add tools/segments/runs.py tools/segments/isolated_source.py \
  tools/performance/segments-isolated.json tests/segments/test_isolated_source.py
git commit -m "Check the block on one isolated synthetic source"
```

---

### Task 10: Segment767, fixed count and reversible jump

**Goal:** Run the block on gbnoisefit's crowded Segment767 band, using Sangria `obs` minus the true MBHB and VGB streams. The covariance is a running-median estimate, which is also the SNR reference. The start set is the 18 catalogue rows matching the npz frequencies. Run once at fixed count and once with birth/death, and report recovery and the count posterior.

**Files:**
- Create: `tools/segments/segment767.py`
- Create (generated): `tools/performance/segments-767.json`
- Test: `tests/segments/test_segment767.py`

**Acceptance Criteria:**
- [ ] `start_rows` matches all 18 npz frequencies to catalogue rows at `rtol=1e-12`; the test asserts 18 distinct hits.
- [ ] `--mode fixed`: every injection with reference SNR ≥ 7 is recovered (a cold source within 1 bin) in ≥ 90% of post-burn-in samples.
- [ ] `--mode rj`: every injection with reference SNR ≥ 10 is recovered in ≥ 80% of samples. The count posterior is recorded.
- [ ] Both modes are written to `tools/performance/segments-767.json` and the results are shown to the user.

**Verify:** `python -m pytest tests/segments/test_segment767.py -q` → `1 passed`; then `python tools/segments/segment767.py --mode fixed` and `--mode rj` each end with `PASS`

**Steps:**

- [ ] **Step 1: Write the failing test**

`tests/segments/test_segment767.py`:

```python
import numpy as np
import pytest

from .support import SANGRIA, requires_sangria


@requires_sangria
def test_start_rows_match_the_npz():
    import segment767
    import sangria

    if not segment767.NPZ.exists():
        pytest.skip("Segment767 npz not found; set BG_SEGMENT767")
    n, dt = sangria.grid(SANGRIA)
    rows = segment767.start_rows(SANGRIA, n * dt)
    freqs = np.load(segment767.NPZ)["initial_sources_native"][:, 0]
    assert rows.shape[0] == 18 and len(set(rows["Name"])) == 18
    np.testing.assert_allclose(np.sort(rows["Frequency"]), np.sort(freqs), rtol=1e-12)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/segments/test_segment767.py -q`
Expected: `ModuleNotFoundError: No module named 'segment767'`

- [ ] **Step 3: Implement the tool**

`tools/segments/segment767.py`:

```python
"""Crowded band: gbnoisefit's Segment767 (18 sources) at fixed count and with RJ.

Data: Sangria obs minus the true MBHB and VGB streams. Noise: a running median
of the data's own power. It absorbs the unresolved galaxy, and serves as both
the turn's covariance and the SNR-cut reference. Start set: the 18 catalogue
rows matching the npz's ``initial_sources_native`` frequencies (column 0, Hz).

    PYTHONPATH=src:... python tools/segments/segment767.py --mode fixed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import runs
import sangria

from binary_galactica.segments import (
    ChainArchive,
    SegmentLikelihood,
    UCBBlock,
    UCBBlockConfig,
    ldc_to_native,
)

NPZ = Path(
    os.environ.get(
        "BG_SEGMENT767",
        str(
            Path.home()
            / "Documents/lisa/gbnoisefit/vendor/sangria_segment_767/segment_767.npz"
        ),
    )
)
MARGIN = 300


def start_rows(path: Path, T: float) -> np.ndarray:
    """Return the catalogue rows of the npz's 18 sources."""
    freqs = np.asarray(np.load(NPZ)["initial_sources_native"][:, 0])
    band = (int(np.floor(freqs.min() * T)) - 2, int(np.floor(freqs.max() * T)) + 3)
    rows = runs.catalogue_rows(path, T=T, band=band)
    picked = []
    for f in freqs:
        j = int(np.argmin(np.abs(rows["Frequency"] - f)))
        if abs(rows["Frequency"][j] - f) > 1e-12 * f:
            raise ValueError(f"no catalogue row at {f!r} Hz")
        picked.append(j)
    if len(set(picked)) != len(freqs):
        raise ValueError("two npz sources matched the same catalogue row")
    return rows[sorted(picked)]


def run(
    mode: str,
    *,
    turns: int,
    steps: int,
    seed: int,
    outdir: Path,
    path: Path = sangria.DEFAULT_PATH,
) -> dict:
    """Run one mode and return its report."""
    n, dt = sangria.grid(path)
    T = n * dt
    z = np.load(NPZ)
    band = (int(np.floor(float(z["f_lo"]) * T)), int(np.ceil(float(z["f_hi"]) * T)))
    rows = start_rows(path, T)
    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], path))
    data = runs.frequency_data(spectra, n, dt)
    lo, hi = band[0] - MARGIN, band[1] + MARGIN
    cov = runs.covariance(data, runs.running_median_variance(spectra, lo, hi), lo, hi)
    config = UCBBlockConfig(
        band_bins=band,
        n_target=len(rows),
        birth_death=(mode == "rj"),
        steps_per_turn=steps,
    )
    block = UCBBlock(
        grid=data,
        start_sources=rows,
        outdir=outdir / mode,
        config=config,
        reference_covariance=cov,
    )
    _, seconds = runs.run_block(block, data, cov, turns, seed)
    native = ldc_to_native(rows, T)
    report = {
        "mode": mode,
        "band": band,
        "segments": len(block.table.segments),
        "dropped": block.dropped,
        "seconds_per_turn": seconds,
        "sources": [],
    }
    for seg in block.table.segments:
        lik = SegmentLikelihood.for_segment(seg, T=T, t0=0.0, config=config)
        lik.set_reference(block.c_ref[:, seg.bin_lo : seg.bin_hi])
        inj = native[(native[:, 0] >= seg.lo) & (native[:, 0] < seg.hi)]
        chain = ChainArchive(block.outdir / f"segment_{seg.index:05d}" / "cold").load()
        found, counts = runs.recovery(chain, inj)
        for entry, rho in zip(found, lik.snr(inj) if len(inj) else []):
            entry["snr"] = float(rho)
        report["sources"].extend(found)
        report.setdefault("count_histogram", []).append(counts)
    floor, need = (7.0, 0.9) if mode == "fixed" else (10.0, 0.8)
    report["criterion"] = {"snr_at_least": floor, "recovered_at_least": need}
    report["pass"] = all(
        s["recovered"] >= need for s in report["sources"] if s["snr"] >= floor
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("fixed", "rj"), required=True)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=767)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    outdir = args.outdir or Path(tempfile.mkdtemp(prefix="segment767-"))
    report = run(
        args.mode, turns=args.turns, steps=args.steps, seed=args.seed, outdir=outdir
    )
    out = runs.PERFORMANCE / "segments-767.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged[args.mode] = report
    runs.write_report("segments-767.json", merged)
    for s in report["sources"]:
        print(
            f"q={s['q']:.2f} snr={s['snr']:6.1f} recovered={s['recovered']:.2f} "
            f"dq={s['median_dq']} dlnA={s['median_dlnA']}"
        )
    print("counts:", report["count_histogram"], " dropped:", len(report["dropped"]))
    print(f"run directory: {outdir}")
    print("PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then both modes**

Run: `python -m pytest tests/segments/test_segment767.py -q`
Expected: `1 passed`

Run: `python tools/segments/segment767.py --mode fixed`, then `python tools/segments/segment767.py --mode rj`
Expected: one line per injection, the count histogram, then `PASS`. Show the user both reports, including any dropped start sources. On `FAIL`, use superpowers-extended-cc:systematic-debugging before changing anything.

- [ ] **Step 5: Commit**

```bash
git add tools/segments/segment767.py tools/performance/segments-767.json \
  tests/segments/test_segment767.py
git commit -m "Run the block on Segment767 at fixed count and with RJ"
```

---

### Task 11: Real Sangria segments against injections and GLASS chains

**Goal:** Run the block (reversible jump) on a few real Sangria bands, each covering one GLASS 12-month segment. Compare each GLASS detection's chain median and spread in frequency and ln amplitude with the block's cold-chain counterpart and with the nearest injection.

**Files:**
- Create: `tools/segments/compare_glass.py`
- Create (generated): `tools/performance/segments-glass-compare.json`
- Test: `tests/segments/test_compare_glass.py`

**Acceptance Criteria:**
- [ ] `glass_segments(3)` returns three GLASS segment indices with 8–15 detections in 3–5 mHz. `glass_chain` reads each detection's chain, which has `Frequency` and `Amplitude` columns.
- [ ] `python tools/segments/compare_glass.py` writes `tools/performance/segments-glass-compare.json`. It has:
  - per detection: GLASS and block medians and standard deviations;
  - the separation in combined sigma;
  - the nearest injection.
- [ ] ≥ 80% of GLASS detections with GLASS SNR ≥ 10 have a block counterpart in ≥ 50% of samples, within 3 combined sigma in both frequency and ln amplitude. The table has been shown to the user.

**Verify:** `python -m pytest tests/segments/test_compare_glass.py -q` → `1 passed`; then `python tools/segments/compare_glass.py` → last line `PASS`

**Steps:**

- [ ] **Step 1: Write the failing test**

`tests/segments/test_compare_glass.py`:

```python
import pytest


def test_glass_segments_and_chains():
    pytest.importorskip("pandas")
    pytest.importorskip("tables")
    import compare_glass

    if not (compare_glass.GLASS_DIR / "UCB.h5").exists():
        pytest.skip("GLASS outputs not found; set BG_GLASS_DIR")
    det, chosen = compare_glass.glass_segments(3)
    assert len(chosen) == 3
    for seg in chosen:
        rows = det[det["segment"] == seg]
        assert 8 <= len(rows) <= 15
        chain = compare_glass.glass_chain(det, rows.index[0])
        assert {"Frequency", "Amplitude"} <= set(chain.columns)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/segments/test_compare_glass.py -q`
Expected: `ModuleNotFoundError: No module named 'compare_glass'`

- [ ] **Step 3: Implement the tool**

`tools/segments/compare_glass.py`:

```python
"""Compare the block with injections and with GLASS's chains on a few segments.

GLASS's 12-month run used T = 31,457,280 s and Sangria spans 31,536,000 s, so
the comparison is made in Hz and ln amplitude, not in bins.

    PYTHONPATH=src:... python tools/segments/compare_glass.py --segments 3
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import runs
import sangria

from binary_galactica.segments import (
    ChainArchive,
    UCBBlock,
    UCBBlockConfig,
    padding_bins,
)

GLASS_DIR = Path(
    os.environ.get(
        "BG_GLASS_DIR",
        str(Path.home() / "Documents/lisa/project-catalog/data/old/glass_h5_files"),
    )
)
MARGIN = 300


def glass_segments(count: int, fmin=3e-3, fmax=5e-3, nmin=8, nmax=15):
    """Return GLASS's detections and ``count`` segment indices with ``nmin..nmax`` of them."""
    import pandas as pd

    det = pd.read_hdf(GLASS_DIR / "UCB.h5", "detections")
    det = det[(det["Frequency"] >= fmin) & (det["Frequency"] < fmax)]
    sizes = det.groupby("segment").size()
    chosen = sorted(int(s) for s in sizes[(sizes >= nmin) & (sizes <= nmax)].index)[
        :count
    ]
    return det, chosen


def glass_chain(det, name):
    """Return GLASS's posterior chain of one detection."""
    import pandas as pd

    return pd.read_hdf(GLASS_DIR / det.loc[name, "chain file"], f"/{name}_chain")


def _counterpart(chain: dict, q: float, burn: float = 0.25, tol: float = 2.0):
    first = int(burn * chain["k"].size)
    comps = chain["samples"][first:].reshape(chain["samples"][first:].shape[0], -1, 8)
    d = np.abs(comps[..., 0] - q)
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, axis=1)
    rows = np.arange(comps.shape[0])
    hit = d[rows, j] < tol
    return comps[rows, j][hit], float(hit.mean()) if hit.size else 0.0


def compare_segment(
    seg_index, det, spectra, data, *, T, turns, steps, seed, outdir, path
):
    """Run the block on one GLASS segment's band and compare every detection."""
    rows_g = det[det["segment"] == seg_index]
    q = rows_g["Frequency"].to_numpy() * T
    lo, hi = int(np.floor(q.min())) - 5, int(np.floor(q.max())) + 6
    pad = padding_bins(hi / T, 0.0, T)
    if hi - lo < pad + 1:
        grow = pad + 1 - (hi - lo)
        lo, hi = lo - grow // 2, hi + grow - grow // 2
    start = runs.catalogue_rows(path, T=T, band=(lo, hi), snr_min=7.0)
    config = UCBBlockConfig(
        band_bins=(lo, hi), n_target=max(len(start), 1), steps_per_turn=steps
    )
    a, b = lo - MARGIN, hi + MARGIN
    cov = runs.covariance(data, runs.running_median_variance(spectra, a, b), a, b)
    block = UCBBlock(
        grid=data,
        start_sources=start,
        outdir=outdir / f"glass_{seg_index}",
        config=config,
        reference_covariance=cov,
    )
    runs.run_block(block, data, cov, turns, seed)
    chains = [
        ChainArchive(block.outdir / f"segment_{s.index:05d}" / "cold").load()
        for s in block.table.segments
    ]
    owners = block.table.owners(q)
    inj_q = start["Frequency"] * T
    out = []
    for (name, row), qd, owner in zip(rows_g.iterrows(), q, owners):
        g = glass_chain(det, name)
        gf, glnA = g["Frequency"].to_numpy(), np.log(g["Amplitude"].to_numpy())
        if owner >= 0:
            ours, frac = _counterpart(chains[owner], float(np.median(gf)) * T)
        else:
            ours, frac = np.empty((0, 8)), 0.0
        entry = {
            "name": name,
            "glass_snr": float(row["SNR"]),
            "found_fraction": frac,
            "glass_f": [float(np.median(gf)), float(np.std(gf))],
            "glass_lnA": [float(np.median(glnA)), float(np.std(glnA))],
        }
        if len(ours) > 1:
            of, olnA = ours[:, 0] / T, ours[:, 3]
            entry["block_f"] = [float(np.median(of)), float(np.std(of))]
            entry["block_lnA"] = [float(np.median(olnA)), float(np.std(olnA))]
            entry["sep_f"] = abs(entry["block_f"][0] - entry["glass_f"][0]) / np.hypot(
                entry["block_f"][1], entry["glass_f"][1]
            )
            entry["sep_lnA"] = abs(
                entry["block_lnA"][0] - entry["glass_lnA"][0]
            ) / np.hypot(entry["block_lnA"][1], entry["glass_lnA"][1])
        if len(inj_q):
            j = int(np.argmin(np.abs(inj_q - qd)))
            entry["injection"] = {
                "name": str(start["Name"][j]),
                "f": float(start["Frequency"][j]),
                "lnA": float(np.log(start["Amplitude"][j])),
            }
        entry["overlap"] = bool(
            frac >= 0.5
            and entry.get("sep_f", np.inf) < 3
            and entry.get("sep_lnA", np.inf) < 3
        )
        out.append(entry)
    return {
        "segment": seg_index,
        "band": [lo, hi],
        "start_sources": len(start),
        "dropped": len(block.dropped),
        "detections": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segments", type=int, default=3)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    args = parser.parse_args()
    n, dt = sangria.grid(args.path)
    T = n * dt
    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], args.path))
    data = runs.frequency_data(spectra, n, dt)
    det, chosen = glass_segments(args.segments)
    outdir = Path(tempfile.mkdtemp(prefix="glass-compare-"))
    results = [
        compare_segment(
            s,
            det,
            spectra,
            data,
            T=T,
            turns=args.turns,
            steps=args.steps,
            seed=args.seed + s,
            outdir=outdir,
            path=args.path,
        )
        for s in chosen
    ]
    bright = [d for r in results for d in r["detections"] if d["glass_snr"] >= 10]
    fraction = float(np.mean([d["overlap"] for d in bright])) if bright else 0.0
    report = {
        "T_block": T,
        "T_glass": 31_457_280.0,
        "segments": results,
        "bright_overlap_fraction": fraction,
        "pass": fraction >= 0.8,
    }
    runs.write_report("segments-glass-compare.json", report)
    for r in results:
        print(
            f"GLASS segment {r['segment']} band {r['band']} start={r['start_sources']}"
        )
        for d in r["detections"]:
            print(
                f"  {d['name']} snr={d['glass_snr']:6.1f} found={d['found_fraction']:.2f} "
                f"sep_f={d.get('sep_f')} sep_lnA={d.get('sep_lnA')} overlap={d['overlap']}"
            )
    print(f"bright overlap fraction {fraction:.2f}; run directory: {outdir}")
    print("PASS" if report["pass"] else "FAIL")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the tool**

Run: `python -m pytest tests/segments/test_compare_glass.py -q`
Expected: `1 passed`

Run: `python tools/segments/compare_glass.py`
Expected: one line per GLASS detection, then `PASS`. Show the user the table. On `FAIL`, inspect the failing detections (which are missing, and which are offset in frequency or amplitude) before changing anything.

- [ ] **Step 5: Commit**

```bash
git add tools/segments/compare_glass.py tools/performance/segments-glass-compare.json \
  tests/segments/test_compare_glass.py
git commit -m "Compare the block with GLASS chains on real Sangria segments"
```

---

### Task 12: Full-band dry run

**Goal:** Build the full-band segment table from a stand-in start set (every `dgb`/`igb` row with analytic SNR > 7) and report:
- the segment, width, padding, BW and `K` distributions;
- the distinct compile keys;
- an estimate of runner memory and archive size per turn;
- the measured time per turn on sampled segments, extrapolated to the band at 1000 steps per turn.

**Files:**
- Create: `tools/segments/dry_run.py`
- Create (generated): `tools/performance/segments-dry-run.json`

**Acceptance Criteria:**
- [ ] `python tools/segments/dry_run.py` writes `tools/performance/segments-dry-run.json` with:
  - `n_segments`;
  - `compile_keys`;
  - distributions;
  - `memory_bytes`;
  - `archive_bytes_per_turn`;
  - per-sample `first_turn_s` and `steady_turn_s`;
  - `predicted_turn_s`.
- [ ] Every table segment satisfies the padding rule (checked in the tool, which raises otherwise).
- [ ] The report has been shown to the user, including the predicted seconds per full-band turn and the compile-key count.

**Verify:** `python tools/segments/dry_run.py --sample 6 --steps 100` → prints the summary, then a last line `wrote <repo>/tools/performance/segments-dry-run.json`

**Steps:**

- [ ] **Step 1: Implement the tool**

`tools/segments/dry_run.py`:

```python
"""Full-band dry run: segment table, compile keys, memory and time per turn.

The stand-in start set is every dgb/igb catalogue row with analytic SNR > 7
(no confusion noise); the user's noise block supplies the real one. Timing runs
one cold turn (which includes compilation) and one warm turn on a few segments
spread across the band. Data: obs minus MBHB and VGB, with a running-median
covariance. The warm per-step time is extrapolated to every segment with a
least-squares fit in ntemps * K * BW.

    PYTHONPATH=src:... python tools/segments/dry_run.py --sample 6 --steps 100
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import runs
import sangria

from binary_galactica.segments import (
    SegmentLikelihood,
    SegmentRunner,
    UCBBlockConfig,
    build_segment_table,
    ldc_to_native,
    quantize_bins,
)


def _stats(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "min": float(v.min()),
        "median": float(np.median(v)),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


def run(sample: int, steps: int, seed: int, path: Path) -> dict:
    """Build the table, time sampled segments, and return the report."""
    n, dt = sangria.grid(path)
    T, n_freq = n * dt, n // 2 + 1
    rows = runs.catalogue_rows(path, T=T, snr_min=7.0)
    native = ldc_to_native(rows, T)
    config = UCBBlockConfig(steps_per_turn=steps)
    start = time.perf_counter()
    table = build_segment_table(native, T=T, n_freq=n_freq, config=config)
    table_s = time.perf_counter() - start
    segs = table.segments
    for s, seg in enumerate(segs):
        left = segs[s - 1].pad if s else 0
        right = segs[s + 1].pad if s < len(segs) - 1 else 0
        if seg.width < max(seg.pad, left + right):
            raise AssertionError(f"segment {s} violates the padding rule")
    keys = Counter((seg.bw, quantize_bins(seg.n_bins), seg.k_max) for seg in segs)
    ntemps = config.ntemps
    dims = np.array([8 * seg.k_max for seg in segs])
    memory = int(np.sum(ntemps * (16 * dims + dims**2) * 8))
    full_steps = UCBBlockConfig().steps_per_turn
    archive = int(np.sum((full_steps // config.thin) * (dims + 2) * 8))

    spectra = sangria.aet_spectra(*sangria.combine(["obs"], ["mbhb", "vgb"], path))
    pick = np.unique(np.linspace(0, len(segs) - 1, sample).round().astype(int))
    work = Path(tempfile.mkdtemp(prefix="dry-run-"))
    timings = []
    for s in pick:
        seg = segs[s]
        lo, hi = seg.bin_lo, seg.bin_hi
        var = runs.running_median_variance({ch: spectra[ch] for ch in "AE"}, lo, hi)
        c = np.stack([var["A"], var["E"]])
        r = np.stack([spectra["A"][lo:hi], spectra["E"][lo:hi]])
        own = native[list(seg.owned)]
        if len(own):
            lik = SegmentLikelihood.for_segment(seg, T=T, t0=0.0, config=config)
            lik.set_reference(c)
            own = own[lik.snr(own) >= config.snr_min]
        runner = SegmentRunner(
            seg,
            own,
            T=T,
            t0=0.0,
            config=config,
            outdir=work / f"seg{s}",
            seed=seed + int(s),
            reference=c,
        )
        t1 = time.perf_counter()
        runner.advance(r, c)
        first = time.perf_counter() - t1
        t2 = time.perf_counter()
        runner.advance(r, c)
        steady = time.perf_counter() - t2
        timings.append(
            {
                "segment": int(s),
                "bw": seg.bw,
                "n_pad": quantize_bins(seg.n_bins),
                "k_max": seg.k_max,
                "owned": len(own),
                "first_turn_s": first,
                "steady_turn_s": steady,
                "per_step_s": steady / steps,
            }
        )
    x = np.array([[1.0, ntemps * t["k_max"] * t["bw"]] for t in timings])
    y = np.array([t["per_step_s"] for t in timings])
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    work_units = np.array(
        [ntemps * seg.k_max * seg.bw for seg in segs], dtype=np.float64
    )
    predicted = float(np.sum(coef[0] + coef[1] * work_units) * full_steps)
    return {
        "T": T,
        "start_sources": int(native.shape[0]),
        "n_segments": len(segs),
        "outside_band": len(table.outside),
        "table_seconds": table_s,
        "compile_keys": len(keys),
        "compile_key_counts": {str(k): v for k, v in sorted(keys.items())},
        "width": _stats([seg.width for seg in segs]),
        "pad": _stats([seg.pad for seg in segs]),
        "n_bins": _stats([seg.n_bins for seg in segs]),
        "bw": dict(Counter(seg.bw for seg in segs)),
        "k_max": _stats([seg.k_max for seg in segs]),
        "owned": _stats([len(seg.owned) for seg in segs]),
        "memory_bytes": memory,
        "archive_bytes_per_turn": archive,
        "timings": timings,
        "fit_seconds_per_step": {
            "intercept": float(coef[0]),
            "per_template_bin": float(coef[1]),
        },
        "steps_per_turn_predicted": full_steps,
        "predicted_turn_s": predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=6)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--path", type=Path, default=sangria.DEFAULT_PATH)
    args = parser.parse_args()
    report = run(args.sample, args.steps, args.seed, args.path)
    path = runs.write_report("segments-dry-run.json", report)
    print(
        f"{report['start_sources']} start sources -> {report['n_segments']} segments, "
        f"{report['compile_keys']} compile keys"
    )
    print(f"width {report['width']}  pad {report['pad']}  K {report['k_max']}")
    print(
        f"memory ~{report['memory_bytes'] / 2**30:.2f} GiB, archive "
        f"~{report['archive_bytes_per_turn'] / 2**30:.2f} GiB per turn (uncompressed)"
    )
    for t in report["timings"]:
        print(
            f"  seg {t['segment']:5d} BW={t['bw']:4d} K={t['k_max']:3d} "
            f"cold {t['first_turn_s']:.1f}s warm {t['steady_turn_s']:.2f}s"
        )
    print(
        f"predicted full-band turn at {report['steps_per_turn_predicted']} steps: "
        f"{report['predicted_turn_s'] / 3600:.1f} h"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the dry run**

Run: `python tools/segments/dry_run.py --sample 6 --steps 100`
Expected: the summary lines, then `wrote <repo>/tools/performance/segments-dry-run.json`. Show the user:
- the segment count and compile-key count;
- the memory and archive estimates;
- the predicted hours per full-band turn.

The spec flagged about 10⁸ waveform evaluations per turn, so a large number here is expected information, not a failure.

- [ ] **Step 3: Commit**

```bash
git add tools/segments/dry_run.py tools/performance/segments-dry-run.json
git commit -m "Measure the full-band segment table and time per turn"
```

---

### Task 13: Document the block and release 0.6.0

**Goal:** Document the segment block:
- amend the README statement that likelihoods and trans-dimensional searches stay outside the package;
- add a short usage section;
- bump the version to 0.6.0.

Then confirm the full suite and the mypy baseline.

**Files:**
- Modify: `README.md` (the "Supporting modules" paragraph; a new "UCB segment block" section before "## Validation")
- Modify: `pyproject.toml:7`
- Modify: `src/binary_galactica/__init__.py:81`

**Acceptance Criteria:**
- [ ] The README no longer says trans-dimensional searches stay outside the package. It has a "UCB segment block" section covering:
  - the block, its inputs, channels and `C = (T/2) S`;
  - the reference-PSD SNR cut and its effect on the count posterior;
  - the impulse extra;
  - the continue bridge and its losses.
- [ ] `pyproject.toml` and `binary_galactica.__version__` read `0.6.0`.
- [ ] `python -m pytest -q` passes; skips are only for absent optional dependencies.
- [ ] `python -m mypy src` reports no error in `src/binary_galactica/segments/`, and no more errors overall than the pre-existing baseline.

**Verify:** `python -m pytest -q` → all pass; `python -m mypy src 2>&1 | grep -c "segments/"` → `0`

**Steps:**

- [ ] **Step 1: Amend the supporting-modules paragraph**

In `README.md`, replace

```markdown
Priors/proposals use NumPy on the host and native coordinates. They are building
blocks rather than an inference engine:
likelihoods, fitted noise, trans-dimensional searches, and WDM integration
remain outside this package.
```

with

```markdown
Priors/proposals use NumPy on the host and native coordinates. They are building
blocks rather than an inference engine. The one sampler-facing exception is
`binary_galactica.segments` (below), a GLASS-style segment likelihood and
enchilada block that drives impulse's reversible-jump sampler. Fitted noise and
WDM integration remain outside this package.
```

- [ ] **Step 2: Add the usage section**

In `README.md`, insert before `## Validation`:

````markdown
## UCB segment block

`binary_galactica.segments` samples the galactic-binary band the way GLASS's
global fit did, as one block of an enchilada blocked-Gibbs sampler. The band
is tiled into segments, each with an interior plus padding. Each segment has
its own `impulse.RJPTSampler`, with:
- GLASS's twelve-chain ladder;
- the Fisher, prior-redraw, fm-shift and psi-phi moves;
- prior birth/death on about 25% of moves.

Sampling starts at injected values, so there are no F-statistic or GMM
proposals. Install the sampler with `python -m pip install -e ".[segments]"`;
it needs impulse with `RJPTSampler` (its `rjmcmc` branch until that is
released).

```python
from binary_galactica.segments import UCBBlock, UCBBlockConfig

block = UCBBlock(
    grid=l1_data,                       # channels ("A", "E") or ("A", "E", "T")
    start_sources=catalogue_rows,       # LDC rows with the sky/dgb/cat fields
    outdir="runs/ucb",
    config=UCBBlockConfig(),            # written to runs/ucb/config.json
    reference_covariance=starting_cov,  # fixes the SNR < 5 cut for the run
)
wheel.add(block_to_register=block,
          initial_block_state=block.initial_block_state(frequency_l1_data),
          data_domain="frequency")
```

Each `sample()` call is one Gibbs turn:
1. The block reads the A and E coefficient variances `C = (T/2) S` from the
   frequency `DataCovariance`.
2. It sweeps even segments, then odd. Each segment sees the conditional
   residual minus the other segments' owned sources, on its padded window.
3. It publishes the owned sources over their full band in every data channel
   (T is derived from X/Y/Z).

The SNR cut is measured against a reference covariance fixed for the run.
Start sources below it are dropped and listed in
`dropped_start_sources.json`. As in GLASS, the cut truncates the source prior
without renormalizing it, so the posterior on the count is not exactly that of
the uniform count prior. Prior-only runs have no likelihood and no cut, and
recover the uniform count.

impulse has no continue call, so each turn re-calls `sample()` on the same
sampler. The ladder, its adaptation, the RNG streams and the proposal state
carry over. The iteration count and impulse's own chain files restart each
turn, and the block copies every turn's cold chain to
`segment_*/cold/turn_*.npz`. Validation reports are in
`tools/performance/segments-*.json`.
````

- [ ] **Step 3: Bump the version**

In `pyproject.toml` set `version = "0.6.0"`. In `src/binary_galactica/__init__.py:81` set `__version__ = "0.6.0"`.

- [ ] **Step 4: Run the full suite and mypy**

Run: `python -m pytest -q`
Expected: all tests pass. Skips are only reported for optional dependencies that are really absent.

Run: `python -m mypy src 2>&1 | tail -1` and `python -m mypy src 2>&1 | grep -c "segments/"`
Expected: the total is no higher than the pre-existing baseline (about 42, see memory `local-toolchain-baseline`), and the `segments/` count is `0`. If not, fix the new modules' annotations; do not touch the pre-existing errors. ruff is not installed: report it as not run.

- [ ] **Step 5: Commit**

```bash
git add README.md pyproject.toml src/binary_galactica/__init__.py
git commit -m "Document the UCB segment block and release 0.6.0"
```

---

## Spec coverage

| Spec section | Tasks |
|---|---|
| Decisions: home, README amendment, impulse as optional extra | 1, 13 |
| Data (`obs` − MBHB − VGB), 12 months, built by the application | 3, 10, 11, 12 (tools); the block takes `T` from L1Data (8) |
| Segment table: band, greedy widths, padding rule, ownership, prior support, BW, `K`, written to the run directory | 4, 8 |
| LDC conversion and conventions gate | 2, 3 |
| `lnlike`, SNR cut against a run-fixed reference, drops recorded | 5, 8 |
| `lnprior`, birth/death from the prior, within-model moves and weights | 6, 7 |
| Turn-fixed Fisher matrices matched by frequency | 6, 7 |
| One turn: checks, conditional residual per segment, even/odd sweep, recompute, publish | 7, 8 |
| Requirements on impulse (continue call): bridged by re-calling `sample()` | 7, 13 |
| Verification 1–7 | 3; 5; 6 and 7; 9 and 10; 7 and 8; 7, 10 and 11; 12 |

## Execution amendments (2026-11-01)

Decisions and findings made during execution that change the tasks above:

- **Backend (user):** "we should use tdionthefly for ALL domains". `UCBBlockConfig.backend` defaults to `"onfly"`. A WDM-domain likelihood follows as a separate spec addendum.
- **Phase convention (Task 3 finding):** LDC `InitialPhase` is GLASS's native φ0 directly; the spec's −φ0 was wrong.
- **Template bandwidth (user):** GLASS `ucb_bw` with safety 1.
- **Tasks 10 and 11 band design (user, "Band around target"):**
  - A single band misses loud catalogue sources sitting in its padding. Both tools therefore run a multi-segment band around the target, built by `runs.target_band` (Segment767 ±600 bins; GLASS segments ±300).
  - The start set is the catalogue at analytic SNR ≥ 7.
  - Results are judged over the inner segments only (`runs.inner_segments`), and a pass must not be vacuous.
  - Chains are read through `UCBBlock.cold_chain`.
- **Noise reference (user, "Residual noise ref"):** the validation tools use the running median of obs − MBHB − VGB minus the start-rule catalogue sources (`runs.residual_spectra`) as both covariance and SNR reference. The raw-data median dropped 34 of 83 start sources in GLASS segment 276; the residual median dropped 1.
- **Task 11 segment filter (controller):** `glass_segments` uses `nmax` 20. The plan's 8–15 matched only one GLASS segment.
- **Units (finding):** `ae_noise` is GLASS's phase-format PSD. Sangria `dt * rfft` data has instrument variance `0.5 T ae_noise · 4 (f/f*)^2` (`runs.instrument_variance`), pinned by a test against Sangria's noise-only stream.
- **Sweep order (finding, final review):** a segment's template bandwidth can exceed `2 * pad`, so template tails reach same-parity windows. The block therefore defines the sweep as sequential and in place: even segments, then odd, each seeing its neighbours' updated sources (`block.py`, `test_sweep_subtracts_updated_neighbours`). The spec's "can later run in a process pool" holds only if `bw <= 2 * pad` is enforced, or tails are truncated to the padded window.
- **Branch base:** this branch was cut from `fix/physical-amplitude` (0.5.0, the native amplitude convention). Merging it also lands that change unless that branch merges first.
