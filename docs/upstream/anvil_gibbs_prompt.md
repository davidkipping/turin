# Task: a supported way to move chains between resumed segments

**Who is asking.** `turin` (the sibling clone `turin/`) fits Kepler/TESS
transit light curves with MetalPlanet as the forward model and anvil's
ChEES-HMC as the sampler. This brief asks for **one small addition**:
a method that returns a `ResumeState` with the chains moved to new
positions and every kernel-specific cached quantity recomputed. turin
already ships a working fallback, so nothing here is a blocker; the ask is
to replace a dependence on anvil's internal state layout with a contract.

**Deliverable discipline.** As before: no new dependencies (mlx + numpy
only), tests beside the existing ones, `docs/` updated, and push to
`https://github.com/davidkipping/anvil`. When done, say whether the API
landed as proposed or differs. turin feature-detects, so it needs to know
the exact name to detect.

## Context to read first

Paths relative to the directory holding the sibling clones:

1. `anvil/anvil/engine.py` — `ResumeState` (its `state` dict and
   `check()`), `Results.resume_state()`, and `run(..., resume=...)`.
2. `anvil/anvil/kernels/chees.py` — `ChEESHMC.init`, which builds the
   state `{"u", "log_prob", "grad"}`; and `EnsembleKernel.init` in
   `kernels/ensemble.py` for comparison.
3. `turin/turin/gibbs.py` and `turin/turin/capabilities.py`
   (`set_positions`) — the move turin applies, and the fallback this
   replaces.

## Why turin needs it

turin's TTV fit samples one mid-transit time `dtau_i` per transit
alongside five shape parameters. A weakly constrained transit has a timing
posterior with several separated bumps, hours apart, and **ChEES does not
move chains between them**. The pooled draws then weight each bump by how
many chains happened to settle there, and that epoch's R-hat never passes
however long the run. On Kepler long-period targets this sent fits to the
16,384 draws/chain cap and cost hours.

turin's fix exploits a structural fact: given the shape parameters, each
transit's likelihood term depends on that transit's time alone. So each
chain can redraw every `dtau_i` from its exact one-dimensional conditional
on a grid spanning the whole prior, with a Metropolis-Hastings correction,
and land in any bump with the right probability ("grid-Gibbs"). ChEES
alternated with an exact move is still one Markov chain, so turin pools the
segments exactly as it already pools resumed rounds.

Measured offline on KOI-4848.01 (4 transits, 512 chains, 2,100 draws/chain),
against an exact marginal computed by brute-force grid over posterior shape
draws:

| | ChEES alone | ChEES + grid-Gibbs every 100 draws |
|---|---|---|
| weak epoch's distance from the exact marginal (TV) | 0.162 | 0.019 |
| that epoch's R-hat | 1.634 | 1.047 |
| that epoch's bulk ESS | 827 | 7,570 |
| wall clock | 653 s | 790 s |

turin applies the move between `anvil.run` segments: take
`results.resume_state()`, move the chains, resume with `n_warmup=0`.

## The problem with doing that today

Moving the chains means writing `state["u"]` **and** the cached
`state["log_prob"]` and `state["grad"]` that ChEES carries from one
iteration to the next. turin's fallback does exactly that, which is correct
for the ChEES state of anvil 0.1, but it depends on anvil's internal state
layout:

- If ChEES (or a future kernel) caches anything else — a Hessian estimate,
  a momentum, a per-chain step-size multiplier — turin would resume from a
  stale cache. That is a silent wrong answer, not a crash. turin's fallback
  therefore refuses any state whose keys are not exactly
  `{"u", "log_prob", "grad"}`, which is safe but brittle.
- The ensemble kernel's state is different, so turin cannot offer the move
  there at all.

anvil knows its own kernels' state; turin should not have to.

## The ask

```python
class ResumeState:
    def with_positions(self, u: mx.array, target: LogDensity) -> "ResumeState":
        """A copy of this state with the chains moved to ``u``
        ((n_chains, dim), float32, unconstrained space), every cached
        per-chain quantity the kernel keeps recomputed at ``u`` against
        ``target``, and everything else -- frozen params, iteration, seed,
        kernel name -- unchanged."""
```

Requirements:

1. **Recompute through the kernel.** The cleanest route is probably a
   kernel-level `refresh(state, u, target) -> ChainState` (ChEES: call
   `target.log_prob_and_grad(u)`; ensemble: `target.log_prob(u)`), which
   `with_positions` calls via the kernel class named in `self.kernel`.
   Whatever the mechanism, a future kernel that adds cached state must be
   forced to say how to refresh it, not silently copy stale values.
2. **Validate.** Shape `(n_chains, dim)` and finite `log_prob` at the new
   positions; raise with a legible message otherwise, matching the style
   of `ResumeState.check()`.
3. **Leave the key stream alone.** `iteration` and `seed` are unchanged,
   so the next resumed segment draws the same keys it would have drawn
   without the move. turin's move uses its own RNG.
4. **Not a copy-in-place.** Return a new `ResumeState`; turin may keep the
   old one for diagnostics.
5. **Test** that `run(resume=state.with_positions(u, target))` gives
   identical draws to a fresh `run(u0=u, ...)` with the same frozen
   params, for ChEES and the ensemble kernel; and that a deliberately
   stale `log_prob` (setting `u` by hand) is what the method exists to
   prevent.

Out of scope: a per-iteration user-move hook inside `run()`. turin
measured "move before warmup as well" against "move between segments
only" and found no difference, so segment boundaries are enough.

## What turin does with it

`turin/capabilities.py` detects `hasattr(anvil.ResumeState,
"with_positions")` and calls it in place of its own fallback; the
capability shows in `turin --capabilities`. Nothing else in turin changes.
