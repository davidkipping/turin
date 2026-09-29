# Reply: all four items landed

Answering `anvil_prompt.md`. Everything asked for is on `main` at
`https://github.com/davidkipping/anvil` (commit `f97f583`, 2026-09-29).
115 tests green, docs build clean.

Two API details differ from the brief — both marked **DEVIATION** below,
both in the direction of making the proposed spelling work rather than
against it. Nothing else changed: every existing call site is unaffected.

## Feature detection

The three probes named in the brief all answer True:

```python
"resume" in inspect.signature(anvil.run).parameters   # item 1
hasattr(anvil.Results, "save_state")                  # item 1, persistence
"divergent_per_chain" in res.extras                   # item 3
"callback" in inspect.signature(anvil.run).parameters # item 4
```

Item 2 has no API surface to detect: it is a behaviour fix. If turin wants
to gate its MAP-centred 1e-3 init-ball workaround on it, probe the
transform directly — the old code returns `-inf` here and the new one
does not:

```python
import mlx.core as mx
from anvil import ParamSpec, Transform
_tr = Transform([ParamSpec("p", lo=0.0, hi=1.0)])
boundary_is_survivable = bool(
    mx.isfinite(_tr.log_det_jac(mx.array([[25.0]]))).item())
```

---

## Item 1 — resumable runs

```python
res1 = anvil.run(kernel, target, u0, n_warmup=400, n_samples=200, seed=1)
res2 = anvil.run(kernel, target, resume=res1, n_samples=400)
```

`resume=` takes a `Results` or a `ResumeState` (from `anvil.load_state`).
The chains continue from `final_state` — positions, cached log-prob and
gradient — with `final_params` frozen exactly as warmup left them:
`kernel.init` is not called, `kernel.make_params` is not called, nothing
re-adapts. Works for `EnsembleKernel` and `RandomWalkMetropolis` as well as
`ChEESHMC`.

**DEVIATION 1 — `n_warmup` defaults to 0 on a resume.** The brief's
spelling passes no `n_warmup`, which would have hit the fresh-run default
of 500 and raised. `n_warmup` is now `int | None = None`, resolving to 500
for a fresh run and 0 for a resume. An *explicit* nonzero with `resume=`
still raises `ValueError`, as asked — the silent-discard failure mode the
brief warned about is what is prevented, not the ergonomic default.

**DEVIATION 2 — `seed` is `int | None = None`** (was `int = 0`). On a
resume it defaults to the seed recorded in the state, so the stream really
is a continuation; pass an explicit seed to fork a segment deliberately.
Fresh runs behave exactly as before (`None` → 0).

### The RNG offset

Handled as the brief specified. `Results` carries

```python
res.iters_consumed   # n_warmup + n_samples*thin, ACCUMULATED across resumes
res.seed             # the effective seed
```

and a continuation draws `key(iters_consumed + t)`. Asserted two ways in
`tests/test_resume.py`: a resume with the offset artificially zeroed
produces different draws (`test_resume_does_not_replay_the_first_segment_keys`),
and three chained segments are pairwise distinct.

### What else rides along

The brief's list of state was complete for the params, but two host-side
*sequences* also have to continue or a resumed segment silently re-treads
ground the first one covered:

- ChEES's Halton jitter index (trajectory-length jitter is a sequence, not
  a draw),
- the ensemble's move-mixing draw count.

Both live in `res.kernel_ckpt` (`{"halton_iter": 50.0}`,
`{"move_draws": 20.0}`) and round-trip through the file. The mechanism is
three new optional methods on `Kernel` — `attach` / `checkpoint` /
`restore` — where `attach` is "the setup `init` does, minus building the
state". A custom `LogDensity` needs nothing; a custom *kernel* only needs
these if it holds host-side state.

`attach` also fixes a trap turin would eventually have hit: `ChEESHMC`
takes its dense-or-diagonal configuration from the **saved params**, not
from its constructor. A run that asked for `dense=True` and was silently
downgraded at init (fewer than `4*dim` chains) resumes the way it actually
ran, and a `ChEESHMC(dense=False)` handed a dense state follows the state.

### Accounting

`accept_fraction`, `extras["n_divergent"]` and
`extras["divergent_per_chain"]` describe the **resumed segment only**, as
requested. `res2.n_warmup == 0`.

### Persistence

```python
res.save_state("run_state.npz")
state = anvil.load_state("run_state.npz")
res2 = anvil.run(kernel, target, resume=state, n_samples=N)
```

Plain `np.savez`, `allow_pickle=False` on read, every entry under a
transparent name. A dense ChEES run writes exactly this:

```
format             int64 ()        # 1; bump on an incompatible layout change
anvil_version      <U10 ()
kernel             <U8  ()         # "ChEESHMC"
iteration          int64 ()        # what to offset the key stream by
seed               int64 ()
n_chains, dim      int64 ()
state__u           float32 (n_chains, dim)
state__log_prob    float32 (n_chains,)
state__grad        float32 (n_chains, dim)
params__step_size   float32 ()
params__traj_length float32 ()
params__sigma       float32 (dim,)
params__corr        float32 (dim, dim)      # dense=True only
params__lrinv       float32 (dim, dim)      # dense=True only
ckpt__halton_iter   float64 ()
```

Non-dense swaps `corr`/`lrinv`/`sigma` for `params__inv_mass (dim,)`. The
`(dim, dim)` factors round-trip bit-identically — tested for both
`dense=True` and `dense=False`, on the arrays *and* their dtypes, and a
run resumed from disk is bit-identical to the same resume in memory.

turin can read `kernel`, `n_chains`, `dim`, `seed` and `iteration` with
plain numpy and refuse a mismatch before ever calling `anvil.run`. anvil
checks what it can itself and raises naming both sides:

| situation | raises |
|---|---|
| `dim` disagrees with `target.dim` | `ValueError: resume state has dim=4, but the target has dim=6` |
| a `u0` passed alongside has the wrong shape | `ValueError: ... this is a configuration mismatch, not an initialization` |
| state written by a different kernel | `ValueError: resume state was written by ChEESHMC, but this run uses RandomWalkMetropolis ...` |
| internally inconsistent file | `ValueError: resume state is inconsistent: declares 999 chains ...` |
| unreadable format version | `ValueError: ...: state format 99, this anvil reads 1` |
| explicit `n_warmup` with `resume=` | `ValueError: ... n_warmup must be 0 ...` |
| `resume=` given something else | `TypeError: resume= takes a Results or a ResumeState ...` |
| neither `u0` nor `resume=` | `TypeError: run() needs initial positions u0, unless resume= is given` |

Since turin refuses on its own terms anyway, note it can pass `u0` **and**
`resume=` together: `u0` is not used, but its shape is checked, which is a
free `n_chains` assertion against the CLI flags.

### Acceptance, as specified

All four criteria are tests in `tests/test_resume.py`:

- `W + 2N` against `W + N` then a resume of `N` agree on the posterior
  mean (within 0.15 sd) and sd (within 15%) for all three kernels, and the
  segments are not identical draws;
- the continuation's frozen params are bit-identical to `res1.final_params`;
- `save_state`/`load_state` round-trips bit-identically, dense and not;
- mismatched `dim` / `n_chains` raise.

---

## Item 2 — the boundary is survivable

Both suggested fixes are in; the third (per-chain step size) was left
alone as the brief asked.

**2a.** `log_det_jac`'s bounded branch is now
`log(w) - |u| - 2*log1p(exp(-|u|))`. Finite for every float32 `u`, and
*more* accurate well before saturation: −15.0000 against −15.0261 at
`u = 15`. Asserted against the float64 replica over the ordinary sampling
range to 3e-5 absolute, which is the "does not shift a well-behaved run"
check — the value is unchanged, only the conditioning.

**2b.** A divergence now only vetoes a proposal made from a **finite**
current state. Applied to both `_finish` and `_finish_dense`.

`tests/test_boundary.py` starts chains at `u = 15` and `u = 25` on a flat
bounded target: they rejoin the bulk within the run, have moved (not
merely gone finite), keep acceptance above 0.1, and the run records zero
divergences. A companion test checks that eight boundary chains out of 128
do not corrupt the answer — the u-space marginal of a flat prior over a
box is the standard logistic, recovered to 8% in sd.

**One extra fix, found while testing this path.** With a subset of chains
on `-inf`, dual averaging drives the step size to float32 zero and
`ChEESHMC.step` computed `ceil(h*T/eps)` — a `ZeroDivisionError` from
inside the sampler. It now takes a single inert step, so such a run ends
with a stalled-chain signature that `warmup_report` and the divergence
counts can describe. turin is more likely than most callers to hit this,
since a hard prior edge on a low-signal epoch is exactly the setup.

---

## Item 3 — per-chain divergences

```python
res.extras["n_divergent"]           # unchanged scalar int
res.extras["divergent_per_chain"]   # np.ndarray, shape (n_chains,)
```

The per-chain vector sums to the scalar (asserted on Neal's funnel, which
reliably produces both divergences and a skewed distribution of them
across chains).

---

## Item 4 — progress callback

```python
anvil.run(..., progress=False, callback=lambda phase, iteration, info: ...)
```

`phase` is `"warmup"` or `"sample"`. `info` carries what the progress line
computes:

| key | phase | note |
|---|---|---|
| `total` | both | iterations in this phase (`n_samples * thin` while sampling) |
| `rate` | both | iterations/s |
| `accept` | both | mean acceptance (running mean over the phase while sampling) |
| `elapsed`, `eta` | both | seconds |
| `step_size` | both | `None` for kernels without one (ensemble, RWM) |
| `n_divergent` | sample | running total |

Cadence is the one `progress` already uses: every 10% of a phase by
default, or every N iterations with `progress=N` — an explicit `progress`
sets the cadence for both channels. **A callback does not turn printing
on**: `progress=False, callback=...` is silent, so turin can render its
own line without redirecting stdout. The sampling pipeline is drained
before each sampling-phase callback, so `n_divergent` and `accept` are
current rather than `pipeline` iterations stale.

---

## Not done

The write-up's per-chain step size (boundary suggestion 3) — explicitly
out of scope per the brief, and still is. If turin later finds chains
whose gradients are orders of magnitude apart *after* 2a and 2b, that is
worth reopening as its own piece of work rather than as a patch.
