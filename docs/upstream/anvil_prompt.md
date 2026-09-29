# Task: three additions to `anvil` needed by `turin`

**Who is asking.** `turin`
(`/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/turin`) is the
GPU successor to `hurin`: it fits Kepler/TESS transit light curves with
MetalPlanet as the forward model and anvil as the sampler, replacing
hurin's jaxoplanet + NumPyro NUTS. Its likelihood is a per-epoch
profile-likelihood Legendre detrending — the nuisance polynomial
coefficients are solved analytically inside every log-density evaluation —
so it is a custom `LogDensity`, not `ChunkedGaussianLogLike`. Everything
below is a gap turin hits in normal operation; turin ships working
fallbacks for all of it, so nothing here is a blocker, but each item
either costs turin real science capability or real wall-clock.

Items are in priority order. **Item 1 is the one that matters.** Items 2
and 3 are small and self-contained. Item 4 is optional.

**Deliverable discipline.** Please keep anvil's existing conventions: no
new dependencies (mlx + numpy only), tests beside the existing ones,
`docs/` updated where behaviour changes, and push to
`https://github.com/davidkipping/anvil` when done. When you are finished,
say which items landed and whether any API differs from what is proposed
here — turin feature-detects rather than assuming, but it needs to know
what to detect.

## Context to read first

Paths relative to
`/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/`:

1. `anvil/anvil/engine.py` — `run()` and `Results`. The whole of item 1
   lives here.
2. `anvil/anvil/kernels/base.py` — the `init` / `step` / `init_adapt` /
   `adapt` / `make_params` contract, in particular that
   `make_params(adapt_state, warmup=False)` already returns exactly the
   frozen values a continuation needs.
3. `anvil/anvil/kernels/chees.py` — `ChEESAdaptState`, `make_params`, and
   `_finish` (item 2b is one line in `_finish`).
4. `anvil/anvil/rng.py` — `KeyStream` derives keys from
   `(seed, iteration, role)`. This is why item 1 needs an iteration
   offset and not much else.
5. `anvil/anvil/transforms.py` lines 123-135 — `log_det_jac` (item 2a).
6. `anvil-gp/docs/anvil-chees-boundary.md` — the measured write-up of
   item 2, including the suggested fixes. Item 2 is "do what this
   document already says".
7. `anvil-gp/docs/anvil-stretch-width.md` — the measured write-up behind
   item 3.

---

## Item 1 — Resumable runs (continue a run without re-adapting)

### Why turin needs it

turin inherits hurin's convergence strategy: run, check R-hat and ESS,
and if they are short, **extend the same chains** rather than restart.
hurin does this with NumPyro by reusing the adapted step size and inverse
mass matrix with `num_warmup=0`, and it is what makes long runs
affordable — a target that needs 16k draws per chain pays warmup once,
not once per doubling round. hurin also persists that state to a
`_resume.pkl`, so `pyhurin --KOI-448.02` re-run tomorrow picks up where it
stopped. turin's CLI keeps both behaviours (`--extend1`, `--extend2`,
auto-resume).

anvil's `run()` cannot do this. It always calls `kernel.init` on a fresh
`u0` and always runs the warmup loop; there is no way in. The fallback
turin ships is to re-warm-up from the previous final positions, which
throws away the adapted step size, trajectory length and preconditioner
and spends the warmup budget again every round. On a 400-warmup,
512-chain ChEES run that is a ~50% tax on every extension, and worse, the
re-adapted run is not a continuation of the same Markov chain, so draws
cannot honestly be concatenated.

### What to build

Everything needed is already in `Results`: `final_state` (the full
`ChainState`, positions + cached log-prob + grad) and `final_params` (the
frozen `step_size`, `traj_length`, and either `inv_mass` or
`sigma`/`corr`/`lrinv`). The missing piece is a way to hand them back.

Proposed API — adapt the spelling to taste, but please keep the
capability:

```python
res1 = anvil.run(kernel, target, u0, n_warmup=400, n_samples=200, seed=1)
res2 = anvil.run(kernel, target, resume=res1, n_samples=400)
# res2 continues res1's chains with res1's frozen adaptation,
# no warmup, no re-initialization, and a non-overlapping key stream.
```

- `resume=` accepts a `Results` (or the state object from `load_state`
  below). When given, `u0` becomes optional/ignored, `n_warmup` must be 0
  (raise if not — silently discarding a requested warmup would be worse),
  `kernel.init` is not called, and `params` comes from the resumed state
  rather than `kernel.make_params`.
- **The RNG must not repeat.** The sampling loop draws
  `keys.key(n_warmup + t)`. A naive resume with `n_warmup=0` reuses the
  original run's *warmup* keys, which correlates the continuation with
  the adaptation phase. Carry the total iteration count consumed so far
  (warmup + sampling × thin) in the resumable state and offset from it,
  so a resumed run's keys continue the same stream. Please assert this in
  a test: a resume must produce different draws than the same call with
  the offset omitted.
- `accept_fraction` and `extras["n_divergent"]` should describe the
  resumed segment only; turin aggregates across rounds itself.
- Resuming an `EnsembleKernel` run should work too (its `make_params` is
  empty, so this is nearly free) — turin offers `--sampler=ensemble` as a
  gradient-free cross-check and wants extension there as well.

### Persistence

turin needs this across process boundaries, not just in memory:

```python
res.save_state(path)            # .npz, no pickle
state = anvil.load_state(path)  # -> object accepted by resume=
anvil.run(kernel, target, resume=state, n_samples=N)
```

Please store the arrays plainly (`u`, `log_prob`, `grad`, the params, the
iteration counter, `seed`, `n_chains`, `dim`, and a small
`kernel`/version tag) so a mismatch can be detected and reported rather
than crashing. turin will refuse a resume whose shape or kernel
disagrees with the current CLI flags, and it would rather read a tag than
guess. Note `dense=True` carries `corr` and `lrinv` as `(dim, dim)`
matrices — those must round-trip, since recomputing them is exactly the
warmup work being avoided.

### Acceptance

- A run of `n_warmup=W, n_samples=2N` and a run of `n_warmup=W,
  n_samples=N` followed by a resume of `n_samples=N` recover the same
  posterior moments on an analytic target (within MC error), and the
  resumed segment's draws are not identical to the first segment's.
- No re-adaptation on resume: the frozen params in the continuation are
  bit-identical to `res1.final_params`.
- Round-trip through `save_state`/`load_state` is bit-identical, for both
  `dense=True` and `dense=False`.
- A resume with mismatched `dim` or `n_chains` raises a clear error.

---

## Item 2 — ChEES cannot leave a bounded parameter's boundary

This is `anvil-gp/docs/anvil-chees-boundary.md`, measured 2026-09-20 and
still unfixed on `main`. Please implement its first two suggested fixes.

### Why turin needs it

turin samples 7 parameters (linear-ephemeris mode) or 5 + one per transit
epoch (TTV mode, routinely 30-80 parameters), **all of them bounded**:
`k ∈ (0, 1)`, a b-fraction in `(0, 1)`, `T14 ∈ (0, 3 T14_NEA)`,
`q1, q2 ∈ (0, 1)`, timing offsets in `± max TTV`. A transit posterior
regularly pushes against those edges — grazing geometries sit at the b
boundary by construction, and a low-signal epoch's timing offset wanders
to its prior edge. With hundreds of chains, a handful arriving at an edge
is routine rather than exceptional, and under the current code each one
is lost for the rest of the run: absorbing, silent, and it drags the
shared step size down for every other chain.

### What to build

Both fixes are in the write-up; restating them so this brief stands alone:

**2a. A non-saturating bounded log-Jacobian** (`transforms.py:128`).
`log(sig) + log1p(-sig)` is exactly `-|u| - 2·log1p(exp(-|u|))`, which is
finite and accurate for every float32 `u`. The current form is `-inf`
past `u ≈ 18` because `mx.sigmoid(18.0) == 1.0` in float32. Note
`to_model` already uses the well-conditioned `mid + half·tanh(u/2)` form,
so this is bringing `log_det_jac` into line with it rather than changing
the transform.

**2b. Do not let a divergence veto an escape** (`chees.py:135`).
`accept = (log_u < dH_safe) & ~diverged` rejects a proposal that would
move a chain from a non-finite current log-probability to a finite one —
the one move that rescues it. A proposal should not be vetoed as
divergent when the *current* state is non-finite.

The write-up's third suggestion (per-chain step size) is explicitly **not**
being asked for here. 2a and 2b make the boundary survivable, which is
what turin needs; sampling a chain whose gradient is 10^6 times its
neighbours' is a different project.

### Acceptance

A regression test, in the spirit of the one already in anvil-gp's
`tests/test_calibration.py`: on a bounded target, chains started at
`u = 15` (and at `u = 25`, past where the old code returns `-inf`) move
within a few dozen iterations and join the bulk, while the rest of the
ensemble is unaffected. Also please confirm the fix does not shift results
on a run with no boundary chains — 2a changes the log-Jacobian's *value*
nowhere, only its conditioning, so a well-behaved run should agree to
float32 noise.

---

## Item 3 — Per-chain divergence counts in `extras`

`engine.py` already accumulates `divergent_sum` as a `(n_chains,)` array
and then throws the resolution away:
`extras={"n_divergent": int(np.array(divergent_sum).sum())}`. Please also
expose the per-chain vector, e.g.
`extras["divergent_per_chain"]` — keeping the existing scalar key for
compatibility.

**Why.** turin implements the trapped-chain check that
`anvil-gp/docs/anvil-stretch-width.md` asks for ("Report it") on its own
side: per-chain median log-probability against the ensemble, plus
per-chain acceptance. Both are already available (`get_log_prob()` is
`(n_kept, n_chains)` and `accept_fraction` is per-chain), so **no upstream
change is required for that check** — but divergences are the third leg of
the same diagnosis and are the one piece turin cannot recover. A chain
that both sits low in log-probability and diverges on most proposals is
stuck at a boundary (item 2); one that sits low with healthy acceptance is
in a secondary mode (the stretch-width finding). turin wants to tell the
user which, because the remedies differ.

---

## Item 4 (optional) — A progress callback

`run(..., progress=True)` prints to stdout on a fixed cadence. turin
drives runs from a CLI that owns its own progress format and wants to
fold sampler progress into it (and later into a resumable batch log).
A `callback(phase, iteration, info)` hook — called on the same cadence
`progress` already uses, with `phase` in `{"warmup", "sample"}` and
`info` carrying the numbers the print statements already compute
(iterations/s, mean acceptance, divergence count, step size) — would let
turin turn anvil's printing off and render its own line. Strictly a
convenience; turin's fallback is to let anvil print.

---

## What turin does in the meantime

turin has a `capabilities.py` that feature-detects each of the above
(`"resume" in inspect.signature(anvil.run).parameters`,
`hasattr(Results, "save_state")`, `"divergent_per_chain" in res.extras`)
and falls back:

| gap | fallback |
|---|---|
| no resume | re-warm-up from the previous final positions, with a warning that the extension is not a continuation |
| boundary trap | MAP-centred 1e-3 init ball to keep chains off the edges; any chain that still freezes is reported, not rescued |
| per-chain divergences | trapped-chain verdict from per-chain log-prob and acceptance only, without distinguishing boundary from secondary mode |
| no callback | anvil prints its own progress |

So please do not hold anything for turin's sake; land what is worth
landing on its own merits and report what changed.
