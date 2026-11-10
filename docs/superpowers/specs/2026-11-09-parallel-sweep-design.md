# Parallel even/odd sweep for the UCB segment block

Run each parity of the segment sweep in parallel and keep the blocked Gibbs
sampler exact: a parallel sweep must give the same chains as a serial one.
This is the first of the sub-projects the user called required after PR #11:
the parallel sweep, then TDI 2.0, then the WDM likelihood, then (optionally) a
search. Work builds on branch `ucb-segment-block` (0.7.0, `6dc3c4d`).

## Decisions

| Topic | Decision |
|---|---|
| Order of sub-projects | "Sweep, TDI2, WDM, search". |
| Exactness | "A. Template-aware merge": change the merge rule so that same-parity segments share no bins. The table changes; the dry runs and validation reports are re-run. |
| Backend | "Local processes, pluggable": persistent spawn processes on one machine, behind a small executor interface that an MPI backend can implement later. |
| Steps per turn | Fixed `steps_per_turn`, as today. Per-segment step counts from the autocorrelation time are a later feature. |

## Why the sweep is sequential today

`UCBBlock._turn` updates the shared model after each segment, and the next
segment of the same parity reads it. That order matters because an owned
template can reach a same-parity neighbour's padded window:

- A source owned by segment `s` has carrier bin `c = trunc(f0*T)` in
  `[lo_s, hi_s)` and a template on bins `c - bw_s//2 ... c + bw_s//2 - 1`
  (`glass/waveform.py`, `qmin = q - BW // 2`).
- Segment `s+2`'s window starts at `lo_{s+2} - pad_{s+2} = hi_{s+1} - pad_{s+2}`.
- The current merge rule `w_s >= max(pad_s, pad_{s-1} + pad_{s+1})` keeps
  same-parity *windows* apart but not their templates, because `bw` is capped
  at `floor_pow2(width + 2*pad)` rather than at `2*pad`.

On the Sangria start set (10369 sources, 12 months) `bw > 2*pad` in 336 of
879 segments, and in 106 an owned template actually reaches a same-parity
window, by up to 157 bins. GLASS's globalfit (`exchange_ucb_source_params`,
even then odd MPI ranks) shares sources only with adjacent segments and
accepts that approximation. This design removes it.

## Segment table

### Merge rule

Define the reach of segment `s` as `r_s = bw_s // 2 + 1`: owned templates lie
inside `[lo_s - r_s, hi_s + r_s)`. The extra bin is a margin on the carrier
truncation. The merge pass in `table._tiling` uses

```
w_s >= max(pad_s, pad_{s-1} + r_{s+1}, r_{s-1} + pad_{s+1})
```

with an absent neighbour (the end of a tiling) contributing nothing to the
two sums. Which neighbour a failing interior merges into is unchanged (the
narrower one). That gives `hi_{s-1} + r_{s-1} <= lo_{s+1} - pad_{s+1}`, and the
mirror inequality, so no owned template of `s-1` reaches the window of `s+1`
and vice versa.

`bw` must be known inside the loop, so the loop computes it with the same
formula as today, `min(ucb_bw(hi/T, fdot_max, 0, amp_max, T, n_freq),
floor_pow2(width + 2*pad))`, recomputing it for each candidate tiling.
`build_segment_table` stores that value on each `Segment` instead of
computing it again. The per-segment pad, GLASS's bandwidth, the window
`[lo - pad, hi + pad)`, `k_max` and the ownership rule are unchanged.

### Table-wide check

The merge loop runs per tiling: the whole band, or one cluster under
`skip_gaps`. Segments `s` and `s+2` that straddle a gap belong to different
tilings, so the loop never compares them. After all tilings are built,
`build_segment_table` checks every same-parity pair `(s, s+2)` in the global
index order:

```
hi_s + r_s <= lo_{s+2} - pad_{s+2}
```

A violation raises `ValueError` naming both segments and the overlap in bins.
Sangria has none, with or without `skip_gaps`. The check is a public
function, `same_parity_disjoint(table) -> None`, so the block can call it on
a table read from disk.

### Effect on Sangria

From a prototype of the rule (scratch script, not committed):

| | Segments | Same-parity reach | Owned median / p99 / max | Window median |
|---|---|---|---|---|
| Today | 879 | 106 segments | 10 / 21 / 31 | 339 bins |
| New rule | 779 | none | 10 / 30 / 42 | 387 bins |
| Today, `skip_gaps` | 971 | 104 segments | 10 / 21 / 31 | 337 bins |
| New rule, `skip_gaps` | 871 | none | 10 / 30 / 42 | 382 bins |

`k_max = max(10, 2*owned)` rises to 84 in the most crowded segment.

## One turn

`UCBBlock._turn` becomes one algorithm for every executor:

```
for parity in (0, 1):
    jobs = {s: (r[:, lo:hi] - (model[:2, lo:hi] - own_s), c[:, lo:hi])
            for s in segments of this parity with a runner}
    results = executor.advance(jobs)        # {s: (owned, contribution)}
    for s in sorted(results):               # table index order
        contrib[s].add_to(model, -1.0)
        results[s].contribution.add_to(model, 1.0)
        contrib[s], sources[s] = results[s].contribution, results[s].owned
```

Every job of a parity is built from the model as it stood when the parity
started. The table-wide check guarantees that no same-parity update touches
another same-parity window, so this produces exactly the residual windows the
sequential sweep would produce on the same table. The updates are applied in
table index order, not completion order, so the floating-point model does
not depend on which worker finishes first.

The SNR cut, `initial_block_state` and the seeds stay in the main process.
Seeds are drawn from the block's `rng` in table order as today, so the chains
do not depend on the worker count.

## Executors

New module `segments/executor.py`.

```python
@dataclass(frozen=True)
class RunnerSpec:            # everything SegmentRunner(...) takes; picklable
    segment: Segment
    start: NDArray[np.float64]
    T: float
    t0: float
    config: UCBBlockConfig
    outdir: Path
    seed: int
    reference: NDArray[np.float64]
    n_channels: int
    n_freq: int

@dataclass(frozen=True)
class AdvanceResult:
    owned: NDArray[np.float64]       # cold sources the segment owns
    contribution: Contribution       # rendered over the full band

class SegmentExecutor(Protocol):
    def start(self, specs: Sequence[RunnerSpec]) -> None: ...
    def advance(self, jobs: Mapping[int, tuple[NDArray, NDArray]]) -> dict[int, AdvanceResult]: ...
    def close(self) -> None: ...
```

`advance` runs `SegmentRunner.advance`, takes the cold sources the segment
owns, and renders them with `render_sources` (A, E and, with three channels,
T). That is the code in today's loop, moved into one function both executors
call.

- **`SerialExecutor`**: holds the runners in the main process and runs the
  jobs in index order. This is the default and is today's behaviour.
- **`ProcessExecutor(workers)`**: `multiprocessing` with the `spawn` start
  method, because forking a process that has initialised JAX is unsafe. Each
  worker builds the runners of its shard from their `RunnerSpec`s and keeps
  them for the whole run, so samplers never cross a process boundary. Per
  parity the main process sends each worker the jobs of its shard and gathers
  the results. Workers render the published contributions themselves, so
  rendering is not serialized in the main process; each worker therefore
  compiles the full-band render kernel as well as the likelihood kernels.
  Workers inherit the parent's environment unless the measurement (below)
  shows that pinned XLA threading flags are needed; the executor then takes
  them as an argument.
- **Shards** are fixed when the executor starts. Segments of each parity are
  assigned separately, longest first, to the least-loaded worker, with the
  cost estimate `k_max * bw + n_bins`. Each worker so gets a balanced share
  of both parities. The function is `assign_shards(segments, workers) ->
  list[list[int]]`.

### Block API

- `UCBBlock(..., workers: int = 1)`. `workers=1` uses `SerialExecutor`; more
  uses `ProcessExecutor`. It is a constructor argument and not part of
  `UCBBlockConfig`, because it does not change the posterior and does not
  belong in the run's `config.json`.
- `UCBBlock.close()` closes the executor. The block is a context manager, and
  a `weakref.finalize` stops worker processes if the block is collected or
  the interpreter exits.
- `cold_chain(i)` reads the archive from disk through a new classmethod,
  `ChainArchive.read(directory)`, because the constructor refuses a
  directory that already holds turns and the runner may live in another
  process.
- The `sample()` docstring's paragraph on why the sweep is sequential is
  replaced by the snapshot description above.

### Failures

- An exception in a worker returns to the main process, which raises
  `RuntimeError` with the worker's formatted traceback and the segment index.
- A worker that dies (end-of-file on its pipe) raises `RuntimeError` naming
  the shard and its segments.
- Either case marks the block failed, as a mid-sweep failure does today, so
  every later `sample()` raises. `close()` still stops the remaining workers.

### Known cost

Startup compiles twice: the SNR cut builds likelihoods in the main process,
and the runners build them again in the workers. Not addressed here.

## Verification

1. **Determinism first.** On a synthetic 5-segment band, two turns:
   - `ProcessExecutor(1)` and `ProcessExecutor(3)` must give bitwise equal
     published states and archives;
   - `SerialExecutor` and `ProcessExecutor(1)` likewise. If a spawned
     worker's XLA threading makes these differ, workers get pinned flags; if
     pinned flags still differ, the README states the tolerance instead of
     bitwise equality, before anything else is built on the claim.
2. **Table.** The merge rule and `same_parity_disjoint` on crowded synthetic
   sets, on gap straddles under `skip_gaps` (including a constructed
   violation that must raise), and with `max_interior_bins`. Sources rendered
   at both interior edges of each segment must have `freq_bin_indices`
   outside every same-parity window.
3. **Sweep equivalence.** On a table that passes the check, the snapshot
   sweep and the old sequential loop give identical windows and models.
4. **Executor.** `assign_shards` balances each parity; a worker exception and
   a killed worker both raise as above and mark the block failed; `close()`
   and the finalizer leave no live processes.
5. **Measurement.** A tool times turns on a Sangria sub-band for 1, 2, 4, 8
   and 16 workers, each with inherited and with single-threaded XLA flags,
   and records wall time per turn, aggregate steps per second and peak
   resident memory per worker in `tools/performance/segments-parallel.json`.
   Workers use about 3.5 GB each, so the default stays `workers=1`; the
   README recommends a count from this measurement.
6. **Re-validation on the new table**, run with the parallel executor: both
   dry runs (`segments-dry-run.json`, `segments-dry-run-skip-gaps.json`),
   the segment-767 band (`segments-767.json`) and the GLASS comparison on
   segments 113, 125 and 276 (`segments-glass-compare.json`). The README's
   numbers (segment counts, widest segment, owned maximum, turn time) are
   updated from them, with the parallel turn time added.

## Out of scope

- Per-segment step counts from the measured autocorrelation time ("sample
  each segment until it has one independent sample and then rotate").
- An MPI backend; the executor interface leaves room for one.
- Removing the double compile at startup.
- Moving runners between workers to rebalance during a run.
