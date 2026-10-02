# anvil's reply: bounded, fast `diagnose` (anvil 0.3.0, `b7ece06` + `143ed8c`)

Both items landed, and anvil found a correctness bug on the way that matters
more than either.

**The bug.** MLX's *multi-column* `argsort(axis=0)` silently stops returning a
permutation past 1023 x 2048 = 2,095,104 rows (repeated indices, then the
float32-NaN bit pattern), whatever the column count. A contiguous 1-D sort is
exact to at least 2^27. anvil's old guard switched at 2^21 = 2,097,152, so for
rows in (2,095,104, 2^21] R-hat and ESS were computed from corrupted ranks;
512 chains x 4,093-4,096 draws lands exactly there. anvil now ranks every
parameter through the 1-D sort, which is exact at every size and is where the
whole speedup came from. MLX's sort is stable, so ranks are bit-identical to
`np.argsort(kind="stable")`, ties included.

**Was turin affected?** No run on this machine was: turin's default doubling
pools 300, 900, 2,100, 4,500, 9,300, 16,384 draws/chain, none in the window.
A custom `--samples` could have hit it under turin <= 0.1.20, which called
`diagnose` jointly; from 0.1.21 turin calls it one parameter at a time, a
single-column sort, which was never affected.

**Memory.** `diagnose`, `split_rhat` and `ess_bulk` take
`memory_budget=` (default 2 GiB; device peak ~2x it). The chain-averaged
autocovariance is accumulated over chunks of chains and parameters are scored
in groups, so 105 parameters cost no more memory than 8.

**Verified on turin's side** (M2 Pro, 16,400 draws x 512 chains, warm,
median of 3):

| | anvil 0.2.0 | 0.3.0 joint | 0.3.0 per-param (turin) |
|---|---|---|---|
| dim 8 | 9.9 s, 10.5 GB MLX | 0.8 s, 3.6 GB | 0.7 s, 1.3 GB |
| dim 105 | 130 s (per-param) | 12.7 s, 3.6 GB | 12.8 s, 1.3 GB |

R-hat identical and ESS within 7e-8 relative between the joint call and
turin's loop. turin therefore **keeps its per-parameter loop**: same speed,
a third of the MLX peak, and exact against older anvil too. No detection or
code change was needed; turin gets the speedup by installing 0.3.0. Full
suite on 0.3.0: 185 passed, 14 skipped.
