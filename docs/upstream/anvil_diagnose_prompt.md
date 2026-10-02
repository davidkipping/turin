# Task: make `anvil.diagnose` bounded in memory and fast past 2^21 draws

**Who is asking.** `turin` (github.com/davidkipping/turin) fits Kepler/TESS
transit light curves with anvil's ChEES-HMC, 512 chains, extending the same
chains in doubling rounds up to 16,384 draws/chain and calling
`anvil.diagnose` on the pooled draws after every round. Everything below
was measured on an Apple M2 Pro (32 GB) against anvil 0.2.0. turin already
works around the memory half (see "What turin does now"), so nothing is
blocking; the speed half is what will hurt next, on short-period targets.

Two asks, independent of each other. **Item 1 is small and safe; item 2 is
the one that matters for turin's next regime.**

**Deliverable discipline.** As before: no new dependencies (mlx + numpy
only), tests beside the existing ones, `docs/` updated, version bumped,
pushed to `https://github.com/davidkipping/anvil`. Say what landed and how
turin should detect it.

## Context to read first

1. `anvil/anvil/diagnostics.py`: `diagnose`, `_rank_normalize_all` (and
   the `_MX_SORT_LIMIT` fallback to `_rank_normalize`), `_autocov`,
   `_ess_batched`.
2. turin's workaround, already pushed: `turin/sampling.py`, function
   `assess` (the per-parameter loop) and the comment above it.

## Measurements

Synthetic float32 draws, 512 chains. "joint" is one `anvil.diagnose(chain)`
call; "per-param" is turin's loop of `diagnose(chain[..., i:i+1])`, which
gives identical R-hat and ESS.

| draws/chain | total rows | dim | call | time | MLX peak | RSS peak |
|---|---|---|---|---|---|---|
| 4,000 | 2.0 M (below 2^21) | 8 | joint | 0.3 s | 1.1 GB | 0.9 GB |
| 4,000 | 2.0 M | 8 | per-param | 0.2 s | 0.15 GB | 0.45 GB |
| 16,400 | 8.4 M (above 2^21) | 8 | joint | 9.9 s | **10.5 GB** | 4.0 GB |
| 16,400 | 8.4 M | 8 | per-param | 9.6 s (rank 9.1 s) | 1.3 GB | 1.8 GB |
| 16,400 | 8.4 M | 16 | joint | 23.1 s | **21.0 GB** | 5.6 GB |
| 16,400 | 8.4 M | **105** | per-param | **129.5 s (rank 122.6 s)** | 1.3 GB | — |

Two separate problems show up here.

1. **Memory: the joint call's MLX peak is ~1.3 GB per parameter at 8.4 M
   rows** (10.5 GB at dim 8, 21.0 GB at dim 16), so dim 105 would need
   ~140 GB. It comes from `_autocov`: the FFT of every chain of every
   parameter at once. At dim 8 the padded input is 16,384 x 1,024 split
   chains x 8; the complex spectrum, `f * conj(f)` and the inverse
   transform are each ~540 MB, plus MLX's FFT workspace, and the result
   comes back as float64. On a 32 GB machine this made turin swap.
2. **Speed: above 2^21 rows, ranking is ~95% of the cost and ~45x slower
   per draw than below it**, because `_rank_normalize_all` falls back from
   the MLX `argsort` (which stops returning a permutation past 2^21 rows,
   as your comment records) to a float64 numpy stable argsort, one
   parameter at a time: ~1.15 s per parameter at 8.4 M rows. turin's
   short-period targets (100+ transits, one timing parameter each) reach
   dim ~105, where that is **~2 minutes of diagnostics per sampling
   round**, on every round of a fit that can take six.

## Item 1: bound `_autocov`'s memory (results unchanged)

Bulk ESS only uses each chain's lag-0 variance and the autocovariance
**averaged over chains** (`acov.mean(axis=1)` in `_ess_single`), never the
per-chain autocovariances themselves. So `_autocov`/`_ess_batched` can
accumulate that average over chunks of chains (and of parameters), with
peak memory set by the chunk rather than by `N x M x dim`.

Requirements:
- Identical R-hat and ESS to the current code, to float32 noise (turin's
  check of its own per-parameter loop against the joint call: max |dR-hat|
  0.0, max |dESS| 0.006 on ESS ~ 10^3).
- Peak memory bounded by a budget independent of `dim` and of `M`. A
  keyword such as `diagnose(chain, names=None, memory_budget=...)` with a
  sensible default would let turin feature-detect it by signature
  (turin's `capabilities._has_param` pattern), and drop its own loop.
- Optional and only if it is clean: compute only the lags the Geyer
  initial monotone sequence actually reaches. For ChEES that is a handful;
  the FFT computes all `N`.

## Item 2: fast exact ranks above 2^21 rows

`_rank_normalize_all` needs, per parameter, the rank of every draw among
all `N x M` draws. Above `_MX_SORT_LIMIT` it is currently a float64 numpy
`argsort(kind="stable")` per parameter. Any approach that keeps the ranks
**exact** (they are integers, so this is achievable) and avoids the MLX
argsort limit is welcome. Possibilities to evaluate, not prescriptions:

- sort chunks of at most 2^21 rows on the GPU, then merge or rank across
  chunks (e.g. global ranks from per-chunk sorted runs via `searchsorted`);
- a float32 numpy sort (default kind) where ties are impossible in
  practice, with the stable float64 path kept for ties;
- batching parameters through whatever path wins, rather than looping.

Acceptance target, from turin's short-period case: **8.4 M rows x 105
parameters in well under 15 s** (currently ~123 s for the ranking alone),
with ranks identical to the current path and memory within item 1's
budget. Please report the timing you reach at that size and at dim 8.

If MLX's `argsort` bug past 2^21 rows has a minimal reproducer, filing it
with MLX would help everyone; your comment already has the measurement.

## What turin does now, and will do

- `turin.sampling.assess` calls `anvil.diagnose` one parameter at a time.
  That bounds the MLX peak at ~1.3 GB regardless of `dim`, but does nothing
  for the ranking time.
- Once item 1 lands with a detectable signature, turin will go back to a
  single call; once item 2 lands, turin's per-round diagnostic cost at
  short period should drop from ~2 minutes to seconds.
- turin keeps working against anvil versions without either change.
