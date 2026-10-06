# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project overview

**turin** fits transiting-exoplanet light curves from Kepler and TESS on
the Apple GPU. It is the successor to **hurin** (`../hurin`), keeping that
package's science pipeline — MAST download, NASA Exoplanet Archive
ephemerides, per-epoch windowing, cross-validated Legendre order
selection, and **profile-likelihood polynomial detrending** — while
replacing the two compute-bound halves:

| | hurin (CPU) | turin (GPU) |
|---|---|---|
| forward model | jaxoplanet | **MetalPlanet** (`../MetalPlanet`, MLX/Metal) |
| sampler | NumPyro NUTS, 2 chains | **anvil** (`../anvil`): ChEES-HMC, hundreds of chains |

Correlated-noise (GP) likelihoods via `anvil-gp` (`../anvil-gp`) are a
planned future direction, not in v1; `likelihood.py` keeps a noise-model
seam for them.

**Scope of v1:** LinEph and TTV fit modes (pyhurin's batch pipeline),
circular orbits, white noise. Deferred: TTV-PL (hurin's Newton-profiled
per-epoch times), GPs, eccentricity, the Flask web frontend.

## The external packages

MetalPlanet, anvil and anvil-gp are **strictly external**, and so is
**hurin** — turin is its successor, not its owner. Never edit any of them
from this repo. Where turin needs something they lack, or spots something
wrong in them, the workflow is to write a brief in `docs/upstream/` for
that package's own Claude session; the user runs it there and reports back
in a matching `*_reply.md`. Existing briefs:

- `docs/upstream/anvil_prompt.md` — resumable runs (needed for
  `--extend`/auto-resume), the ChEES bounded-parameter boundary trap,
  per-chain divergence counts.
- `docs/upstream/anvil_gibbs_prompt.md` — `ResumeState.with_positions`,
  so grid-Gibbs can move chains without turin rewriting ChEES's internal
  `u`/`log_prob`/`grad` cache itself. **Landed in anvil `849159f`**; see
  `anvil_gibbs_reply.md`. turin's own state-rewrite fallback was removed
  in 0.1.24, when turin started requiring anvil >= 0.3.0.
- `docs/upstream/anvil_diagnose_prompt.md` — bound `anvil.diagnose`'s
  memory and make ranking fast at large draw counts. **Landed in anvil
  0.3.0**; see `anvil_diagnose_reply.md`. anvil also found that MLX's
  multi-column argsort silently corrupts ranks past 2,095,104 rows (no turin
  run was in the affected window). Dim 105 at the cap: ~130 s -> 12.8 s per
  round. turin now makes one `diagnose` call with a 256 MiB budget (0.55 GB
  peak, as fast as its old per-parameter loop, which 0.1.24 removed).
- `docs/upstream/metalplanet_prompt.md` — optional: a `tau`-input fused
  kernel with in-kernel exposure integration. **Landed**; turin uses it.
  MetalPlanet 0.7.0 later added `ld_basis=True` for another downstream
  package; turin uses it *only* for `--ld=collapsed` (see "Collapsed limb
  darkening" below) -- for the default path it is no optimisation, since
  turin makes one kernel call and the basis triples the intermediate.
- `docs/upstream/hurin_lightcurve_prompt.md` — hurin downloads every row of
  a MAST name search, which can include a neighbouring star (KOI-7592.01:
  two KIC targets, both at 0"), and stitches them. turin 0.1.26 restricts to
  the archive KIC/TIC and refuses repeated timestamps. **Landed in hurin
  0.1.70** (`88d15ac`): finding confirmed, all four asks taken.

**Inbound, implemented: collapsed limb darkening (`--ld=collapsed`).**
SquishierPlanet proposed it (`docs/upstream/turin_collapsed_ld_prompt.md`
in that repo, with turin's questions and their answers beside it) and
supplied a validated reference implementation; turin 0.1.39-0.1.41 built it
as `turin/ldmarg.py` (see "Collapsed limb darkening" below), and 0.1.43
passed real-target acceptance against the default on KOI-518.02 and
KOI-448.02 (`docs/bench/collapsed_ld_acceptance.md`).

**Every brief turin has sent has landed, so nothing is outstanding
upstream.** Do not open a new one without being asked to: write the finding
down here or in `docs/`, and let the user decide whether it goes upstream. A
brief commits someone else's session to work.
- `docs/upstream/hurin_doc_prompt.md` — documentation only: hurin's
  "1:1 grazing odds" is true marginally over `k ~ U(0,1)` but reads as a
  claim about the user's own target, where the odds are `k/(1-k)`.
  **Landed in hurin 0.1.69** (`50c566a`); see `hurin_reply.md`. hurin
  re-verified the claim against its own `_sample_b_k` before changing
  anything, adopted the wording, and recorded the `k_max = 1` dependency
  beside its `k` prior. Nothing diverges, so
  `docs/hurin-differences.md` needs no entry.

turin has now changed hurin three times by this route: the Kipping (2013)
limb-darkening fix (hurin 0.1.68), the `_MODEL_REV` guard turin then
adopted itself, and this wording. Porting hurin's science is how turin
finds these, so expect more — and **verify numerically before filing**. I
once talked myself into a "documentation error" in hurin's `(b, k)` prior
on a single careless measurement (I had drawn `k ~ U(0, 0.3)`, where
`E[k]/E[1-k] = 0.18`, and read that as a broken 1:1 claim); hurin was
correct, and the brief said so explicitly. A brief that overstates its
case wastes an upstream session's time and spends credibility that the
next real bug needs.

**Push before handing over a brief.** A brief that points at turin files
must point at pushed ones: the upstream session reads them from GitHub or its
own clone, not from this machine. The anvil grid-Gibbs brief went out with
its turin commit unpushed, and anvil could not find the files it cited.

**turin requires a minimum anvil instead of carrying fallbacks.** It used
to feature-detect each anvil capability and fall back without it; every one
of those landed upstream, and anvil < 0.3.0 also carries a silent rank
corruption in `diagnose`, so since 0.1.24 `capabilities.require_anvil()`
refuses anything older than `capabilities.MIN_ANVIL` (0.3.0) with the
upgrade command, before touching the disk. **anvil 0.4.2 is current and is
what turin runs against, but `MIN_ANVIL` stays 0.3.0** (0.4.2's API was
audited call-by-call unchanged, and the full suite passes at 0.4.2 with
MetalPlanet 0.9.7: 212 passed, 14 skipped where no hurin clone sits beside
turin for the parity tests, 2026-10-05): 0.4.0's fix is
for a target mutated between `run` calls, and turin's target holds only
static tensors — grid-Gibbs moves positions, not the target. 0.4.1 and
0.4.2 are review follow-ups to that fix (`Kernel.retrace(target)` is now
the first thing `run` does, before `init` and `attach`); they change no
signature turin calls, and the ordering is inert for a static target.
The first turin change that *does* mutate the target between segments must
raise `MIN_ANVIL` to 0.4.0 in the same commit, because that failure is
silent and passes every convergence check (see `capabilities.MIN_ANVIL`). **turin must always run against
the packages as currently published on GitHub.** When turin starts relying
on a new upstream feature, raise `MIN_ANVIL` in the same change; do not add
detection plus a fallback. `MIN_METALPLANET` (0.7.0, for `ld_basis`) is
checked **only** when `--ld=collapsed` is asked for
(`capabilities.require_metalplanet`); the default path uses nothing newer
than 0.6.1 and stays ungated. That is the general rule for a feature-scoped
minimum: gate the feature, not the package.

## Architecture

```
turin/
  cli.py          kwargs CLI (argv regex grammar + did-you-mean)
  data/           lightcurve.py, ephemeris.py, preprocessing.py (ported from hurin)
  prep.py         windows, epoch segmentation, centering, Legendre CV, exposure nodes
  model.py        MLX transit model: params -> per-epoch tau -> z -> flux
  profile.py      batched per-epoch Legendre profile solve (exact | ratio)
  likelihood.py   epoch-block-chunked Gaussian logL, priors, anvil Transform/target
  params.py       LinEph/TTV parameter specs, b-k prior modes, derived quantities
  seeding.py      template-sweep tau seeds, batched multi-start MAP
  sampling.py     anvil driver, convergence loop, diagnostics, trapped-chain checks
  gibbs.py        grid-Gibbs: exact per-epoch timing move between ChEES segments
  ldmarg.py       --ld=collapsed: LD integrated out (MarginalLDLogProb), q1,q2 drawn after (OmegaSampler)
  outputs.py      CSV/PDF products, resume state
  capabilities.py minimum-anvil gate, installed versions, MLX cache release
```

## Load-bearing invariants

### Epoch-centred time coordinates (float32 safety)

The GPU is float32, whose resolution at absolute BKJD/BTJD magnitudes
(~1500 d) is 5-10 s — a hard floor on timing precision, and MetalPlanet
measures 1.7e-4 in flux after 1,000 orbits without this. So, exactly as in
hurin and as `metalplanet.orbit.epoch_center_times` prescribes:

- All large-minus-large subtraction happens **once, in float64 NumPy**, on
  the host. Times reaching MLX are per-epoch residuals; epoch numbers are
  exact small integers.
- **Sampled parameters are offsets, never absolutes**: `dP` and `dtau0`
  from the recentred input ephemeris, `dtau_i` from each epoch's predicted
  time. Absolutes are reconstructed in float64 immediately after sampling,
  so every downstream consumer sees physical units.
- `anvil.ParamSpec(report_offset=...)` carries the absolute value back out
  in float64.

### The float32 discipline (anvil's five rules)

Read `../anvil/docs/precision.md` before touching `likelihood.py`. The
ones that bite:

- **`mx.array(np_float64_array)` silently yields float32.** Always pass
  `dtype=`. A `np.float64` scalar entering a compiled graph force-evaluates
  it mid-trace.
- **Recentre the likelihood**: accumulate `0.5*(1-r)*(1+r)` and keep the
  exact `-N/2` as a float64 host constant, rather than `-0.5*r**2`. This
  sharpens HMC's energy difference, which `validate_precision` cannot see.
- **Stay in deviation space end to end.** The model returns `f - 1`
  (`model.transit_flux_dev`), the design stores `y - 1` (`y_dev`), and the
  residual is formed as `(y_dev - f_dev) - (1 + f_dev)*L c`. Never subtract
  two quantities near 1 to recover one near 1e-4: doing so cancels four
  digits, and it measurably cost 1.6% on `dlogL/dk` and 8x on the
  log-density error before it was fixed.
- Run `anvil.validate_precision(target, ball)` before every production
  fit, and `certify` after. The measured baseline for a 772-point,
  9-epoch fit is a median float32 error of 3e-4 against float64.
- **float32 gradients are less accurate than the float32 log-density**, and
  that is fine. `dlogL/dk` carries ~8e-3 relative error because `k` acts
  through three partly cancelling channels. It does not bias the posterior:
  leapfrog with any smooth force field is still volume-preserving and
  reversible, and the Metropolis step uses the log-density. Validate
  gradients against float64 autodiff (and closed forms), not against the
  float32 path.

### The profile likelihood

Each epoch's local baseline is `g(t) = 1 + L c` with `L` the Legendre
design matrix on times mapped to [-1, 1] within the epoch window; the
total model is `f_transit * g`. The coefficients `c` are **not sampled**:
they are solved analytically inside every log-density evaluation (a
profile likelihood — a plug-in MLE of the nuisance parameters, with no
`-0.5 log det` Occam factor).

Three solves, `--PL=` (`auto`, the default, measures them per target and
picks the fastest that is no less precise than `exact`; see `plselect.py`):

- **`exact`**: the true flux-space profile, design matrix `F L` (with
  `F = diag(f_transit)`) against target `y - f_transit`.
- **`ratio`**: hurin's form, a WLS fit in ratio space (`y / f_transit`,
  weights `1/sigma^2`), which drops the `f_transit` factors from the
  weights. It agrees with `exact` to O(depth) and exists to reproduce
  hurin's likelihood surface for validation.
- **`hybrid`**: `ratio`'s static factorization as a preconditioner, refined
  back to the exact normal equations.

`--ld=collapsed` forces `exact`: its omega gradient comes from the envelope
theorem, which holds only when `c` is the flux-space minimiser.

Implementation notes: design matrices are built once in NumPy and uploaded
(never rebuilt inside the traced likelihood); columns above each epoch's
CV-selected order are masked, with masked diagonal entries set to 1 so the
batched shape stays fixed; `+1e-10 I` Tikhonov regularization as in hurin.
**MLX `linalg` runs on the CPU only**, so the per-epoch solve is an
unrolled Cholesky in elementwise MLX ops (at most 6x6, fixed size,
differentiable, GPU-resident).

### Exposure integration

Kepler long cadence (29.4 min) is comparable to a short transit's ingress, so
finite-exposure integration is mandatory. turin uses MetalPlanet's
`flux_dev_from_tau` with the **contact rule** (`model.N_GL` Gauss-Legendre
nodes per contact sub-interval), which integrates inside the kernel so the
sub-exposure axis never reaches MLX.

It replaced supersampling for **accuracy** (the "~2x slower" measured then
predates the point-ordering fix below and is no longer a guide to cost).
What supersampling was costing, at the `n_sub` that Kipping (2010) Eq. 40
picks:

- flux error 1.5e-4, about **2.3% of a transit depth**, against 6.4e-8;
- `dF/d(period)` wrong by **~100x and with the wrong sign**. Differentiating
  a supersampled kinked integrand amplifies its error, because the error
  oscillates as nodes cross the contacts.

`N_GL` is 5, judged on the log-likelihood (the table is in `model.py`): its
error there is sd 5e-4, 40x below the float32 noise the `--PL` probe
accepts, and the worst gradient component is off by <0.6%. It was 9 until
0.1.9, chosen on the kernel's own `dF/d(period)` while divergence (below)
made cost look flat in `N_GL`; 5 is 1.55x faster.

**Point order is a 3x speed lever.** The kernel runs one point per GPU
thread in SIMD groups of 32, and a group pays for any branch one member
takes; an in-transit point costs ~8x an out-of-transit one. Storage order
(epoch by epoch, ~10% of each window in transit) puts a transit point in
most groups, so nearly every group paid the in-transit price.
`model.phase_order` sorts the flattened points by time from the reference
mid-transit, and `build_grid(mask=...)` moves padded slots, which
`segment_epochs` parks at the epoch *centre*, to quadrature. KOI-448.02, 512
chains, compiled value+grad: 45.8 ms before, 14.1 ms after (bit-identical
log-likelihoods), 9.1 ms with `N_GL=5`. The full KOI-448.02 LinEph run (256
chains, 200 warmup + 200 draws) went from 549 s to 118 s, medians within
0.02 sigma. Any new route into the kernel should keep both measures. Before
this, cost looked flat in `N_GL` because divergence swamped the arithmetic;
it is not flat any more (9.1 / 14.1 / 18.0 ms at 5 / 9 / 12).

Two things to keep in mind. `geometry="chord"` has no kernel path and always
supersamples. And **finite differences are not a valid gradient reference**
under the contact rule: it freezes its split points (exact, since moving an
interior split of a continuous integrand cancels), so an FD that recomputes
them measures the quadrature's parameter sensitivity instead — and an FD
straddling a contact is wrong at any step size.

### Collapsed limb darkening (`--ld=collapsed`, `ldmarg.py`)

Opt-in, LinEph only. It is **collapsed Gibbs sampling**: the sampler targets
the marginal `p(theta | D)` with the quadratic limb darkening *integrated
out*, and `q1, q2` are then *drawn from their conditional* for every kept
theta (`OmegaSampler`) to fill hurin's product columns. Same posterior as
`--ld=sampled`, two fewer sampled dimensions. Keep the three mechanisms
apart by name, because two of them were measured and rejected: (1) the
marginal in the log-density -- the feature; (2) the post-hoc conditional
draws -- the Gibbs half, products only; and **not** (a) *profiling* the LD,
sampling `L(x*) + prior` at the best-fit LD (0.38 sigma bias on k, b, T14,
46% wider k on KOI-518.02), nor (b) *non-collapsed* Gibbs, alternating
theta | omega and omega | theta (5x slower, and it would mutate the target
between segments, which needs `MIN_ANVIL` 0.4.0).

The mechanism: MetalPlanet's `ld_basis=True` (>= 0.7.0) gives the three
vertex laws of the Kipping triangle from one kernel launch
(`model.vertex_flux_devs`, phase order kept), and every quadratic law's
light curve is the convex mix `sum omega_j F_j`, linear in
`x = (omega1, omega2)`. For each `x` the baseline is profiled exactly as
always (the only profiling anywhere here), giving `L(x)`; its gradient and
Gauss-Newton Hessian come from the envelope theorem via a Schur complement
(hence `--PL=exact`). The sampled value is `L(x*) + 1/2 g H^-1 g + log Z`:
the quadratic model's peak about an expansion point `x*` (one Newton step +
exact QP projection, **detached** -- the value depends on it only at second
order), integrated over the triangle against the prior induced by uniform
`q` by 20x20 Gauss-Legendre in whitened coordinates.

Measured (`tests/test_ldmarg.py`, against references that do not share the
scheme's approximations): conditional density = sampled density at the
same `(q1, q2)` to 5e-14 fp64; collapsed value = brute-force lattice
integral with the true `L` to 4e-5; draws reproduce the exact lattice
conditional (means within 0.8 se, sds 2%); zero draws on the box edge with
the posterior piled against the `q = (1, 0)` corner; detached gradient vs
finite differences 6e-6 (at `exp_time = 0` only -- FD is invalid under the
contact rule); blocking exact. Sampled-mode products stayed byte-identical
across all three commits.

Load-bearing details:

- `OmegaSampler` **always starts from a proposal draw** and has no way to
  pass a start. Starting at `x*` -- often on a triangle edge -- froze 5% of
  draws exactly on the box edge in the reference's first version, because
  rounding put `x*` a hair outside the proposal's support.
- `epoch_log_lik` raises: a shared omega couples every epoch, so grid-Gibbs
  cannot run, and TTV + collapsed is refused (CLI and pipeline).
- Blocks are sized at `COLLAPSED_BYTES_PER_POINT` (2.5x the default; 271
  vs 121 B per chain-point measured). Per-evaluation MLX peak ~2.2x.
- Products: the summary's `q1`/`q2` rows leave R-hat/ESS blank (conditional
  draws, not sampler output); the chains `loglike` column is the sampler's
  stored target exactly as in sampled mode (so the ML row is its argmax);
  the ML-row light curve uses the conditional mode at the best theta.
- Real-target acceptance (0.1.43; LinEph, 512 chains, same seed, to
  convergence): every parameter within 0.011 sigma and 2.5% width of the
  default, KS at the self-resample level, zero `q` draws on the box edge
  (KOI-448.02: `x*` on a triangle edge for 54% of draws, 460,800 draws).
  **Cost: break-even on KOI-518.02 (164.0 vs 163.7 ESS/s), 28% slower on
  KOI-448.02 (39.5 vs 54.5).** That is why it stays opt-in.
- **A claim that did not survive measurement, recorded so it is not
  repeated:** turin's docs argued collapsed mode would help on wall-hugging
  limb darkening, since `q1, q2` are bounded and ChEES can freeze a chain at
  an edge. On KOI-448.02 (`q1, q2` = 0.96, 0.93) the sampled run had no
  trapped chains, and collapsed was slower. Its value is the dimension
  reduction itself, which matters for laws with more coefficients.

Note `n_gl` defaults to 5 in MetalPlanet from 0.7.0, matching
`model.N_GL`; turin passes it explicitly, so that default is inert here.
`u1`/`u2` became optional keywords in the same release but stayed
positional, so turin's existing call is unaffected, and 0.7.0's default
outputs and gradients are bitwise-equal to 0.6.1's.

**Verified against MetalPlanet 0.9.7** (`957eb3a`), turin 0.1.33, 216 tests
green. 0.8.0 put eccentric orbits on `flux_dev_from_tau` as the
*keyword-only* `secosw=`/`sesinw=`, so turin's positional
`(tau, period, a, b, r, u1, u2)` call is untouched; omitted, they change
nothing, and 0.8.1 states circular output stays bit-identical (40 arrays,
both precisions, both LD laws). That ordering is the thing to re-check on
any future bump: inserted positionally before `u1`, turin would have passed
limb darkening as an eccentricity term, silently.

0.8.1 fixed **NaN gradients on grazing transits, circular included**
(`0 * inf` where the contact clip collapses the inner pair). No turin
result was affected: the guard is a `stop_gradient` in
`exposure.contact_offsets`, shared by the frontend and `flux_dev_from_tau`,
and only the frontend's differentiable graph had reached the bad branch.
**0.1.35 pins that guard** --
`test_grazing_gradient_matches_a_finite_differenced_reference` asserts
finite, non-zero `d/dk` and `d/db` through the contact rule on both sides
of `b = 1 - k`; remove the guard and its three collapsed-pair cases go NaN
(measured in review). The older
`test_gradients_are_finite_and_nonzero_in_fp32_at_awkward_geometry` runs
at `exp_time=0` and never builds a contact, so it cannot see this.

The same test checks gradient *accuracy* against MetalPlanet's frontend at
`n_gl=32`, central-differenced in float64. Be precise about what that
reference is: it is the **same contact rule**, differenced with its split
points free to move -- not an independent geometry -- so a bug in the
shared contact location would be invisible to it. It is nonetheless the
true derivative: it is converged (`n_gl` 16 to 128 agree to <=1.3e-7), and
the exact integral does not depend on where it is split. Measured against
it, turin's frozen-split gradient error falls ~8x per doubling of turin's
own `n_gl` (near the edge, `d/db`: 6.5e-5 at 5, 1.2e-5 at 9, 2.3e-6 at 16,
3.0e-7 at 32), which is what the tolerances are set from. **Grazing is the
easy regime** (~1e-9); the hard one is just *inside* the edge, where the
inner pair survives as two narrow sub-intervals. KOI-448.02's turin-defaults
fit (0.1.12: `k = 0.060 +/- 0.007`, `b = 0.932 +/- 0.090`, so `1 - k =
0.940`) has its mean just inside and its posterior straddling the edge by
about a sigma; the hurin-compat fit of the same target is the one at
`b = 0.949` against `0.952`.

Two corrections recorded here because the 0.1.35 commit got them wrong.
First, that test does **not** pin `flux_dev_from_tau`'s detachment of the
split points (`metal.py`), and no accuracy test can: splits that move
track the kink, so a moving-split gradient agrees with the truth *better*
at finite `n_gl` than the frozen one -- review measured that removing the
detachment passes every case. That detachment is MetalPlanet's internal
choice and turin should not pin it. Second, the 1.9e-6 "hard regime" figure
was turin's own finite-`n_gl` error, not the reference's. The FD warning
above is about *turin's* path: finite-differencing it recomputes the splits
and so measures a different quantity from its autodiff (1.5e-5 apart at
`N_GL = 5` wherever the inner pair exists) -- which is why the reference
is the frontend and not turin.

### Why turin does not use MetalPlanet's sampler-facing API

`metalplanet.anvil.make_quad_transit_flux` and `make_transit_target` bake
in a linear ephemeris, offer no exposure integration, carry only a scalar
`df0` baseline, and parametrize by a/R*. turin needs per-epoch mid-times,
long-cadence integration, per-epoch polynomials, and hurin's
`T14` parametrization. So turin builds `tau` and `z` itself and calls
**`metalplanet.metal.flux_dev_metal(z, r, u1, u2)`** — a fused fp32 kernel
with a VJP in `z`, which falls back to the analytic fp64 path on the CPU
stream (that fallback is turin's `log_prob_hi`). Similarly turin writes its
own likelihood rather than using `ChunkedGaussianLogLike`, which chunks the
data axis across epoch boundaries and so cannot hold a per-epoch solve.

### Grid-Gibbs on the epoch times

ChEES does not cross between separated timing modes of a weak transit, and
the pooled draws then weight each mode by chain count, not probability. The
TTV fit therefore interleaves `gibbs.GridGibbs` every `gibbs.SEGMENT` = 25 draws
(`--gibbsgrid=on`, the default). It rests on one structural fact: **given the
shape parameters, each epoch's likelihood term and timing prior depend on
that epoch's time alone**. Anything that breaks that factorization (noise
correlated across epochs, a dynamical TTV model coupling the times) breaks
the move, and must turn it off rather than leave it running.

- The interval is 25 draws (`gibbs.SEGMENT`, read by the round loop from
  the move). Measured against 100 on three Kepler targets: the two with a
  second timing mode (KOI-7776.01, KOI-5162.01) went from unconverged at
  the cap (162 / 88 min) to converged at 9,300 draws (105 / 58 min); the
  control with none (KOI-5228.01) cost +10% and its worst timing distance
  from the exact marginal fell 0.032 -> 0.011. Sweeps are ~15% of a round
  at 25. One seed per target.
- It is exact, not approximate: an independence proposal from the grid
  density plus a Metropolis-Hastings correction. Grid resolution affects
  acceptance (93-98% at 512 cells), never correctness.
- One sweep sets every epoch to grid point g at once and reads all epochs'
  terms from `ProfiledTransitLogProb.epoch_log_lik`: O(N) in the number of
  transits, not O(N^2). Keep it that way for short-period targets.
- Validated against exact marginals: by the same factorization,
  `p(dtau_i | D) = E_shape[p(dtau_i | shape, D_i)]`, a 1-D grid per epoch
  averaged over posterior shape draws, with no sampler involved. On
  KOI-4848.01 and KOI-5897.01 the weak epochs' total-variation distance from
  it fell from 0.11-0.20 to 0.02. `tests/test_gibbs.py` checks the same
  property on synthetic two-bump data; use the reference again before
  changing the move. It does **not** fix
  the separate grazing/non-grazing bimodality in (k, b), which still holds
  shape R-hat up on low-SNR targets.

### Memory: inference on every draw, export on a bounded subsample

At the 16,384-draw cap a 270 MB float32 chain used to inflate to 8.8 GB of
process memory and a 10.5 GB MLX peak, with a 288 s export every round, and
on a 32 GB machine that swapped. The fixes, and what must not regress:

- `sampling.assess` makes one `anvil.diagnose` call with
  `memory_budget=DIAGNOSE_MEMORY_BUDGET` (256 MiB): anvil >= 0.3 chunks the
  autocovariance and ranks per parameter, so the MLX peak is ~0.55 GB at
  any dimension. Before 0.3 the joint call held ~1.3 GB per parameter and
  turin looped over parameters instead. Never thin for R-hat/ESS: ChEES
  draws are nearly independent, so a thinned chain under-reports ESS.
- Everything downstream of the verdict (summary, products, plots,
  `certify`, `width_breakdown`, `whitened_shape`) uses
  `sampling.export_thin`, which caps both rows and rows x parameters, so a
  short-period fit with 100+ epoch times stays bounded too.
- `Continuation` caches its concatenation; `_ml_row` indexes the one winning
  draw; `export_chains` streams the CSV in chunks (byte-identical to the
  old writer, checked including NaN/-0/inf).
- `mx.clear_cache()` after every round (`capabilities.clear_mlx_cache`).
- Heavy products (chains tarball, PDFs) after round 0, then at most every
  `pipeline.HEAVY_EVERY_S`, and always at the end.

**Epoch blocking is not one of these fixes, and should not be counted as
one.** Two measured facts, from SquishierPlanet's probe of a compiled
value+grad at 512 chains (`docs/upstream/` in that repo):

- `likelihood.BYTES_PER_POINT = 48` calls itself pessimistic and is not:
  the default path measured **121 B per (chain, point)** at peak, so the
  constant is ~2.5x optimistic. It has never bitten because the 2 GiB
  budget leaves real targets on one block regardless -- at 223 points per
  epoch it first splits past ~390 epochs.
- Splitting barely moves the peak anyway. Inside one compiled value+grad the
  whole graph is live, so every block's backward intermediates coexist:
  forcing 3 blocks measured 0.174 -> 0.153 GB. Blocking bounds the size of a
  single kernel launch, not the peak.

So a per-evaluation memory problem is not solved by lowering the block
budget. The levers that did work are the ones listed above. `--ld=collapsed`
sizes its blocks with `ldmarg.COLLAPSED_BYTES_PER_POINT` (2.5x) through the
`bytes_per_point` argument, so its blocks are proportionally smaller; the
same caveat applies -- that bounds the launch, not the ~2.2x peak.

To check a change here, measure rather than reason: fill a `Continuation`
with synthetic cap-sized draws (16,400 x 512 chains) and time
`assess` + `_export_all` + `certify` in a subprocess, reading peak RSS and
`mx.get_peak_memory()`. On KOI-5616.01's layout that measured peak RSS
8.76 -> 2.84 GB, MLX 10.5 -> 1.3 GB, and 306 -> 41 s per round.

### Known upstream hazards

- **ChEES can permanently freeze a chain at a bounded parameter's edge**
  (`../anvil-gp/docs/anvil-chees-boundary.md`). All of turin's parameters
  are bounded and transit posteriors sit at edges (grazing geometries,
  low-signal epochs). Mitigated by the MAP-centred 1e-3 init ball; a
  frozen chain is reported, not silently pooled.
- **The stretch move traps chains in secondary modes and the reported
  width inherits it** — 4x too wide, R-hat 1.037 the only tell
  (`../anvil-gp/docs/anvil-stretch-width.md`). turin therefore always runs
  a trapped-chain check: per-chain median log-prob and acceptance against
  the ensemble, with widths broken down by chain before they are quoted.
- **`dense=True` silently downgrades** to diagonal below `4*dim` chains.

## Validation against hurin

`docs/hurin-differences.md` is the reference, and its **KOI-518.02 section is
the primary comparison** — cite that one. hurin 0.1.68 vs turin 0.1.12, both
converged, identical cached input, turin at its defaults: every parameter
within 0.03 sigma, the 27 transit times within 0.02 sigma, and 11.1x less wall
clock (6,037 s to 544 s) with 750-1,460x the ESS per second.

That target was chosen because `T14/P = 0.005` and a 513 ppm depth make both
remaining model differences (chord vs circular, ratio vs exact profile)
negligible, so the comparison isolates the samplers. The KOI-448.02 and
KOI-5162.01 sections are secondary and kept for what they alone show: the
limb-darkening difference on a grazing system (against pre-fix hurin 0.1.62),
and hurin's documented trapped-mode failure.

When re-running a hurin comparison: pass `--chains=8`, because hurin's default
of 2 does not converge on one round and makes the comparison a straw man; run
the two **sequentially**, since hurin saturates ~5 cores and would otherwise
contend; and confirm both packages hold the same cached light curve (compare
byte sizes of `../hurin/cache/<T>.pkl` and `~/.cache/turin/<T>.pkl`).

## Conventions inherited from hurin

- **One patch bump per commit.** 0.1.N is commit N; update
  `pyproject.toml`, `turin/__init__.py` and a `VERSIONS.md` row together.
- **Bump `MODEL_REV` (`turin/__init__.py`) in any commit that changes the
  value of the log-density**, and say so in the VERSIONS.md row. Resume
  state records it and `ResumeState.check_model_rev` refuses to continue
  chains sampled under an older revision, so a correction cannot silently
  contaminate a long run. hurin added the same mechanism in 0.1.68 after a
  limb-darkening fix changed its likelihood. The `GUARDS` tuple is the
  other half: it catches the *user* asking for a different model,
  `MODEL_REV` catches the model changing underneath them.
- **Provenance stamping** on every product: `# turin <version>
  rev<MODEL_REV> | <launch command>` (plus the run status when a fit is in
  progress or unconverged) on **line 2** of CSVs (after the header, because
  `np.genfromtxt(names=True)` treats a leading comment as the header), a
  `turin_version.txt` member appended after the CSV in tarballs (readers
  use `getmembers()[0]`), PDF Creator/Subject metadata, and version +
  command keys in resume state.
- **Product filenames and columns match hurin exactly**, so existing
  analysis scripts keep working: 7 LinEph products, 9 TTV.
- `--tag=<name>` namespaces a run into an independent auto-resume lineage,
  inserted into every product filename before the extension.
- Resume state records the guards `b_prior`, `profile_mode`, `geometry`,
  `sampler`, `ttv_max`, `gibbsgrid` and `ld` (`ResumeState.GUARDS`), plus
  `tag` and chain count; a mismatched resume exits naming the stored value.
  A new guard gets a class default that old pickles read correctly.

## Environment

Dev environment is `turin/.venv` (Python 3.11), with MetalPlanet and anvil
installed editable from the sibling clones:

```bash
.venv/bin/python -m pip install -e ../MetalPlanet -e ../anvil
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest
```

No conda environment on this machine has MLX; the `hurin` conda env
(`~/miniconda3/envs/hurin`) has jaxoplanet and NumPyro and is used only to
generate hurin parity fixtures.
