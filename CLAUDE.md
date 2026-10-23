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
| forward model | jaxoplanet | **Elómë** (`elome`, `../elome`, MLX/Metal; formerly MetalPlanet) |
| sampler | NumPyro NUTS, 2 chains | **anvil** (`../anvil`): ChEES-HMC, hundreds of chains |

Correlated-noise (GP) likelihoods via `anvil-gp` (`../anvil-gp`) are a
planned future direction, not in v1; `likelihood.py` keeps a noise-model
seam for them.

**Scope of v1:** LinEph and TTV fit modes (pyhurin's batch pipeline),
circular orbits, white noise. Deferred: TTV-PL (hurin's Newton-profiled
per-epoch times), GPs, eccentricity, the Flask web frontend.

## The external packages

elome (formerly MetalPlanet), anvil and anvil-gp are **strictly external**, and so is
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
- `docs/upstream/metalplanet_prompt.md` (MetalPlanet, now elome) — optional: a `tau`-input fused
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
as `turin/ldmarg.py` (see "Collapsed limb darkening" below). Real-target
acceptance against the default on KOI-518.02 and KOI-448.02 was recorded in
0.1.43 and its analysis corrected in 0.1.44 after code review
(`docs/bench/collapsed_ld_acceptance.md`).

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
of those landed upstream, so `capabilities.require_anvil()` refuses anything
older than `capabilities.MIN_ANVIL` with the upgrade command, before
touching the disk. **`MIN_ANVIL` is 0.7.1 since turin 0.1.60**, because
`sampling.chain_health` now reads `results.extras["lost_chains"]`: anvil's
end-of-run judgement of lost chains, made on an eager re-evaluation of the
target as well as the kernel's record (the two can disagree at a float32
feasibility edge; anvil measured -17,000 recorded vs -6.8e30 eager on a GP
target). The same range brings two protections turin gets for free: since
0.7.0 one chain at a non-finite or failure-sentinel log-density no longer
drags ChEES's shared step size to zero for all chains, and since 0.5.0 an
MLX compile refusal makes `run` continue uncompiled (turin logs a note)
instead of aborting. Measured before raising it: a KOI-518.02 ChEES run (64
chains, 150 warmup, 100 + 100 draws resumed, 50 after a `with_positions`
move) is **bitwise identical** under anvil 0.4.2 and 0.7.1, chains and
log-probabilities. Earlier floors: 0.3.0 for `diagnose`'s rank-corruption
fix; 0.4.0's fix for a target mutated between `run` calls never applied
(turin's target is static -- grid-Gibbs moves positions); any turin change
that does mutate the target between segments relies on it and is covered
by 0.7.1 already. **`MIN_ANVIL` is 0.9.0 since turin 0.1.71**, for
`anvil.optimize` (see "The MAP search" below). **turin must always run
against the packages as currently published on GitHub.** When turin starts relying on a new
upstream feature, raise `MIN_ANVIL` in the same change; do not add
detection plus a fallback.

**elome is gated per feature, and only for hybrid4.** MetalPlanet was
renamed Elómë (package `elome`, `../elome`, on PyPI) in its 0.14.0, and the
old `metalplanet` import alias was removed in 0.16.0. Every elome release
has `ld_basis`, so the `MIN_METALPLANET` (0.7.0) check that `--ld=collapsed`
used to make became vacuous and was removed in 0.1.60. Since 0.1.67
`--law=hybrid4` needs `capabilities.MIN_ELOME_HYBRID4` = 0.17.0
(`require_elome_for_law`, called in `pipeline.run`): from 0.17.0 elome reads
hybrid4's weights on LDhybrid's curvature basis, and an older one reads the
same four numbers on other shapes. `pyproject.toml` pins `elome>=0.17.1`,
the version verified. Should turin need a newer elome feature, gate the
feature that needs it, not the package.

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
  seeding.py      template-sweep tau seeds; MAP and chain starts via anvil.optimize
  mle.py          --mle: maximum likelihood (LinEph round/oblate, then TTV), LD profiled, best fit only
  sampling.py     anvil driver, convergence loop, diagnostics, trapped-chain checks
  gibbs.py        grid-Gibbs: exact per-epoch timing move between ChEES segments
  ldmarg.py       --ld=collapsed: LD integrated out (MarginalLDLogProb), q1,q2 drawn after (OmegaSampler)
  ldhybrid.py     --law=hybrid4 --ld=collapsed: the same over a 4-simplex (MarginalHybridLogProb, HybridOmegaSampler)
  simplexmass.py  GHK over a simplex, two-pass MGS/TSQR, triangular solves, exact simplex QP on R
  normtail.py     tail-stable log Phi, its inverse, truncated-normal draws (MLX has no erfc)
  outputs.py      CSV/PDF products, resume state
  capabilities.py minimum-anvil gate, installed versions, MLX cache release
```

## Load-bearing invariants

### Which epochs are fitted (`prep.segment_epochs`, 0.1.55)

Every predicted transit gets a window of `N_DURATIONS` durations either
side, whether or not its centre lies inside the light curve (each point's
*own* nearest transit number, `prep.nearest_transit`; windowing, other-planet
masking and segmentation all use it). A point belongs to its nearest transit
and only inside its window. A window spanning a Kepler quarter or TESS
semi-sector (labels recorded per point at download,
`data.lightcurve.segment_labels`) keeps only the stretch holding the
predicted centre. An epoch is fitted with `MIN_PTS` = 2 points **whether or
not any is in transit**: with TTVs possible that is the fit's call, not the
segmentation's. Do not reintroduce an in-transit test or an in-span transit
list -- the latter is the bug class behind 0.1.50 and 0.1.55. The user's
convention, adopted as is: each TESS semi-sector is treated like a Kepler
quarter.

### Epoch-centred time coordinates (float32 safety)

The GPU is float32, whose resolution at absolute BKJD/BTJD magnitudes
(~1500 d) is 5-10 s — a hard floor on timing precision, and elome
measures 1.7e-4 in flux after 1,000 orbits without this. So, exactly as in
hurin and as `elome.orbit.epoch_center_times` prescribes:

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
design matrix on times mapped to [-1, 1] across *that epoch's own data*
(`prep.basis_x`, since 0.1.55; it was the window). The polynomial space and
so the profiled log-density are the same either way (float64 identical to
1e-12 on KOI-518.02 and KOI-448.02), but a window holding data on one side
only -- a quarter split, a transit in a gap -- left the normal matrix's
condition number at 6e5 (K=3, data on x in [0.4, 1]) up to 1e16, beyond
float32 and float64 alike; over the data's own span it stays under 10. The
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
finite-exposure integration is mandatory. turin uses elome's
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

**Mixed cadences (0.1.86).** The exposure is not one number per light curve.
A Kepler `--sc` fit takes short cadence only in the quarters that have it, and
a TESS star can have 2-minute sectors beside FFI ones, so each epoch carries
its own: the median step of its quarter or semi-sector
(`prep.segment_cadences`, merged within 5% by `snap_cadences`; an epoch lies
within one segment, so one cadence). `epoch_data["exposure"]` exists only when
epochs differ, and a single-cadence light curve takes the old path exactly (no
key, one kernel call, same point order). With it, `build_grid(exp_epoch=)`
sorts the points by exposure and then by phase, and `model._tau_kernel` makes
one `flux_dev_from_tau` call per exposure on its contiguous slice (elome takes
`exp_time` as a scalar), which keeps the point-order speed measure inside each
call; `select` re-derives the groups for a block, and a block whose epochs
share one exposure uses that one, not the grid's representative `exp_time`
(the most common, for plots and the fallback). Anything new that calls the tau
kernel goes through `_tau_kernel`, not `elome.flux_dev_from_tau` directly.
No `MODEL_REV` bump: a one-cadence light curve is bitwise unchanged. A mixed
one is a different likelihood, guarded by `ResumeState.exposures` /
`check_exposure` (the data decide it, so it is not in `GUARDS`). The sigma
clip runs per cadence (its window is a number of cadences). Refused:
`--geometry=chord`, whose supersampling takes one value. Found by asking
whether turin knew Kepler's and TESS's integration times: it never did, it
measured one from the stitched series. The 17 TOIs run before the fix were all
2-minute in every sector.

Two things to keep in mind. `geometry="chord"` has no kernel path and always
supersamples. And **finite differences are not a valid gradient reference**
under the contact rule: it freezes its split points (exact, since moving an
interior split of a continuous integrand cancels), so an FD that recomputes
them measures the quadrature's parameter sensitivity instead — and an FD
straddling a contact is wrong at any step size.

### The limb-darkening law (`--law`, 0.1.61)

`--law=quadratic` (default, Kipping `q1, q2`) or `--law=hybrid4`: the
four-weight rational law of LDhybrid v1.0 as elome and SquishierPlanet
define it, with its weights on the **curvature basis** since 0.1.67 (LDhybrid's
`docs/upstream/hybrid4_curvature_simplex.md`; elome >= 0.17): there the
simplex `w_j >= 0, sum w_j <= 1` is exactly the profiles that are positive,
monotonic, concave in mu^2 and of convex slope in mu^2. Sampled as LDhybrid
does, through `q1..q4` uniform on [0, 1], depth `s = sum w = q1^(1/4)` first
and then stick-breaking over the four shapes (`model.ld_weights` ->
`elome.ld.hybrid4_from_q`; `ld_weights_np` is the float64 replica, clamped as
the MLX map is). Uniform `q` gives uniform `w` (a flat 5-component
Dirichlet; checked by moments).

**The 0.1.67 basis change (MODEL_REV 10).** Before it the weights were on
`T = (1 - mu^4, (1 - mu^2)^2, Pi_e0, Pi_e1)`, stick-broken from `q` by
`simplex_from_q` -- a different simplex of profiles. Only the q -> w map
changed in turin: the light curves for a given curvature-basis `w`, the
generators `{1, 1 - S_j}` (`ld_basis` columns) and `F*` come from elome
(`hybrid_norms`), so collapsed hybrid4's value is **bitwise** what 0.1.66
gave on elome 0.17.1, and sampled hybrid4's equals 0.1.66's at the
re-expressed `q` to 8.5e-14 (KOI-518.02, fp64). Quadratic stays bitwise.
The two hybrid4 measurements made before 0.1.67 on the old simplex --
KOI-448.02's grazing split below and the five-target collapsed acceptance
-- were re-measured on the curvature basis in 0.1.76. Note that 0.1.61-0.1.66 run on elome >= 0.17
fitted the new simplex (stick-broken `q`, curvature-basis `w`); on elome
< 0.17, the old one. The law lives on the layout (`ParamLayout.law`,
`.ld_names`, `.n_shape` = 3 + number of LD coordinates), and everything
downstream reads it from there: `likelihood.unpack` takes columns **by
name**, and every former hard-coded `5:` (TTV times, grid-Gibbs, seeding,
ML mid-times) is `layout.n_shape`. `transit_flux_dev(q=, law=)` keeps its
`q1=`/`q2=` form for the quadratic law, whose path is untouched: the
refactor left KOI-518.02's LinEph (sampled and collapsed) and TTV
log-density, gradient and per-epoch terms **bitwise identical**, hence no
`MODEL_REV` bump; hybrid4 is a new model behind the `law` resume guard.

Verified against references that do not share turin's route: elome's
`TransitModel` frontend with the contact rule (fp64 1e-13, fp32 2e-6,
grazing included); the map and LDhybrid's check values written out in the
test, elome's numpy maps and SquishierPlanet's `hybrid4_from_q` (1e-15); the
four profile conditions evaluated from the brief's basis, holding on random
`q` and failing just outside each of the five faces (the region is exact,
not sufficient); float32 log-density and
gradient against float64; the per-epoch factorisation and a grid-Gibbs
sweep that leaves all 7 shape columns untouched; and a slow 256-chain
injection-recovery of a hybrid4 truth (every parameter within 3 sigma).
elome's and SquishierPlanet's hybrid4 light curves agree to 1e-16 (same
poles, eps = 0.0286, 0.3858).

Products: `q1..q4` are sampled rows (R-hat, ESS); `w1..w4` are derived rows
and chain columns, placed before `loglike`. Refused, CLI and pipeline:
`--geometry=chord` (hurin parity; hurin has only quadratic). This and the
collapsed form below are **Steps 1 and 2 of SquishierPlanet's
`docs/upstream/turin_oblate_prompt.md`**, for hybrid4 only, built on the
user's request; Step 3 (oblate planets) is not done, and hybrid2/5 are not
exposed. Low-SNR grazing, KOI-448.02 (default prior, `--PL=exact`): on the
**old** simplex four free weights faked a V shape, and sampled hybrid4
split between the quadratic law's grazing solution (k 0.16) and a small
central planet with extreme limb darkening (k 0.027, b 0.27, q1 on its wall
for 68% of draws), unconverged at R-hat 1.26 after three rounds. **On the
curvature basis that second solution is gone** (0.1.76, one seed, elome
0.18.5): converged in two rounds (R-hat 1.0055 on k, 687 s), no draw with
k < 0.04 at b < 0.6 (min k 0.039), 99.8% grazing, k's 95% range
0.055-0.88, q1 within 0.01 of a wall for 2% of draws. The curvature simplex
admits only profiles that are monotonic and concave in mu^2, which is what
excluded the V-faking ones. The MAP search still logs a top-decile spread of
12.2 (a rival basin in the density), which the chains did not keep. On
KOI-518.02 the laws cost the same. That target is atypical (the user's
point): do not generalise from it.

### Oblate planets (`--oblate`, 0.1.65-0.1.66)

Step 3 of SquishierPlanet's `docs/upstream/turin_oblate_prompt.md`, built on
the user's request with two decisions taken with them: a ρ⋆ prior and a
degeneracy warning instead of the brief's ridge sampler (deferred until
Kepler injections show the false mode matters), and `--ld=collapsed` as the
`--oblate` default (`cli.resolve_ld`; `Args.ld` is `None` until resolved).
hybrid4, LinEph and circular geometry only when sampling
(`cli.oblate_problem`, refused in CLI and pipeline; an oblate TTV layout
exists for `--mle` since 0.1.84 and `build_target` refuses to sample it);
`oblate` is a resume guard (old states read round).

- **Model.** elome's `flux_dev_from_tau(..., f=, theta=)` (radians), `k` the
  area-equivalent radius √(AB), `a_over_rstar` keeps the circle relation, so
  T14 is that circle's duration (a coordinate; the log says so). `f`/`theta`
  reach elome **only when oblate** (`model.oblate_kw`): elome's `f = 0` is
  not bitwise its spherical forms (9.7e-8 of the depth measured), and every
  spherical path is **bitwise** 0.1.64 (16 quantities on KOI-518.02,
  quadratic and hybrid4, sampled, collapsed and TTV, value and gradient).
- **Coordinates** `(fc, fs) = √f (cos 2θ, sin 2θ)` after T14, on the box
  `±√F_MAX` cut to the disc `f <= F_MAX = 0.6` (`params.oblate_from_fcfs`).
  Uniform there is uniform in `f` and `θ`; β-triangle `b >= 0` with θ over a
  half-turn covers every geometry once.
- **Domain.** elome returns NaN for `k > (1-f)^1.5` (reachable: `k`'s box
  goes to 1). `likelihood.oblate_fields` masks from the *parameters* and
  hands the kernel a harmless point (`k_flux`, `f_flux`) where the point is
  not allowed; `oblate_log_prior` gives it -inf. Masking the output instead
  can leave NaN in the gradient. Pinned by a test (a mutation that drops the
  swap fails it in both `--ld` modes).
- **Collapsed** needs nothing new beyond passing `f`, `θ` through
  `_gen_blocks`: the conditional equals the sampled oblate likelihood to
  1e-12, and value and draws match the exact marginal and conditional.
  Memory: no change (collapsed peak is the GHK workspace, 1.54 vs 1.56 GB
  round vs oblate); cost 224 ms collapsed vs 164 ms sampled oblate vs 81 ms
  collapsed round (512 chains, 860 points, value+grad).
- **Start.** The MAP's 128 starts spread `(fc, fs)` by ~±0.1 (the brief's
  three-way nudge needs no code). `seeding.floor_oblateness` moves the
  chains' centre to `f >= 1e-4` (f = fc² + fs² is stationary at the
  origin, and turin's `atan2` for theta has a NaN gradient at exactly
  fc = fs = 0; before elome 0.18.6 its fp32 d/df was also 0 below 1e-5),
  and `seeding.chain_starts`
  spreads chains over anvil's Laplace approximation (sigmas capped at 0.5
  unconstrained units, far starts pulled back). Oblate only, so round
  starts are a 1e-3 ball. Since 0.1.71 the MAP is `anvil.optimize` (see
  "The MAP search"); 0.1.65-0.1.70's Adam needed 1,500 iterations
  (`N_ITER_OBLATE`), 512 starts (`N_STARTS_OBLATE`) and a polishing stage
  (`polish_map`) and still ended 1-14 nats below the PHOENIX truth
  (collapsed; sampled 531-535), all three removed. anvil from 128 starts
  reaches the truth's value on all four seeds (collapsed, 36 s; sampled
  540.0 every seed, 7 s).
- **`--PL=auto`** resolves to exact without probing under `--oblate`.
- **Products.** `fc, fs` sampled (R-hat, ESS); `f`, `theta` (degrees),
  semi-axes `A, B` derived, after `w1..w4` and before `loglike`; a log line
  with `f`'s median, 16-84% range and 95% upper limit, and the prior.
- **Known limitation, kept on purpose:** the b-k grazing bound and taper use
  `k`, not the ellipse's half-width across the chord `h(theta)` (between `k`
  and `A = k/√(1-f)`), so `1 + k < b < 1 + h` has zero prior and the grazing
  odds are slightly off. The light curve is exact. The user decided
  (2026-10-16) not to fix it: oblate fits keep the spherical prior so the two
  stay homogeneous. Documented in the README's oblate limits (0.1.70). The
  oblate light-curve tests include grazing geometries since 0.1.78
  (`test_oblate.GRAZING`: b = 0.95 without second and third contacts, 1.0
  centred on the limb, 1.05 a sliver, at k 0.095, f 0.1), against elome's
  frontend at the usual 1e-13 / 2e-6 and SquishierPlanet (b = 1.0) at 1e-12.
  The `1 + k < b < 1 + h` sliver itself is untestable through turin's
  parametrisation: the circle relation for a/R* ends at `b = 1 + k`.
- References in `tests/test_oblate.py`: elome's `TransitModel` (fp64 1e-13,
  fp32 2e-6) and SquishierPlanet's float64 `model.light_curve` (1e-12).
  Injection (f = 0.1, θ = 30°, b = 0.7, 30 ppm, 1-min cadence, collapsed):
  f = 0.099 ± 0.004, θ = 30.9 ± 2.2°, converged in one round.
- **ρ⋆ prior (0.1.66, any mode and law):** `--rhostar=MEAN,SIGMA`, a
  Gaussian in log10(ρ⋆/kg m⁻³) -- the `logrho` product's unit -- through the
  fitted a/R* (`model.log10_rho_mx`, `params.rhostar_log_prior`). A resume
  guard whose "none" is the value `()`, not a skipped guard, so a run with a
  prior can never continue one without. Checked against the host reporting
  function on all four (oblate x ld) layouts to 1e-9. The brief's table:
  a/R* known to 3% (ρ⋆ to ~9%, sigma ~0.04 dex) cut SquishierPlanet's
  false-oblate share from 0.20 to 0.09.
- **Degeneracy warning (0.1.66):** under `--oblate` the log always gives the
  posterior share of the aligned-oblate-on-a-central-chord family
  (`b < 0.1`, `|theta| < 15°`, `f > 0.05`; `pipeline.aligned_central_share`)
  and warns above 5%, naming `--rhostar`. The ridge sampler is not built.
- **Acceptance: deliberately not done.** The brief's tests 2-6 (PHOENIX
  bias, the b = 0.3 false-oblate share and the rho* table, a real-target
  f -> 0 fit) are SquishierPlanet's science questions, not checks of turin's
  code; the user stopped them (2026-10-15) after two runs. Those two, on
  the PHOENIX truth above through `pipeline.run`: **collapsed** converged in
  one round, 517 s end to end, f = 0.1006 +/- 0.0029, theta = 29.7 +/-
  1.5 deg, every parameter within 0.6 sd, certify 0.09 MCSE; **sampled**
  agreed on f and theta but was unconverged after 4,500 draws/chain (~100
  min): one chain trapped at acceptance 0.007 (2,279 of the run's 2,603
  divergences), R-hat 1.18 on `q2`. That is the case for collapsed as the
  `--oblate` default (one seed each, not 0.1.44's two). README section
  "Oblate planets" since 0.1.69.

### Collapsed hybrid4 (`--law=hybrid4 --ld=collapsed`, `ldhybrid.py`, 0.1.62; fixed 0.1.64)

`ldmarg`'s collapsed Gibbs over a 4-simplex: five generator curves
`{1, 1-S_j}` (the curvature simplex's vertices) from one `ld_basis` launch (`model.generator_flux_devs`; any
hybrid4 law is `sum omega_g F_g`, 9e-18), induced prior
`-5 log sum omega_g/F*_g`, the baseline profiled exactly, `L(x)`'s quadratic
model integrated over the simplex. `ldmarg.py` is untouched (bitwise). What
had to change from 2-D, each decided by measurement (the probe and its
numbers: `docs/bench/collapsed_hybrid_probe.md`):

- **Integral by GHK**, 1,024 frozen scrambled-Sobol points, layout
  `(dep, order)` frozen at the fit's MAP (`ldhybrid.freeze_layout`). Error
  ~1/N; re-choosing the layout per draw did not help. The layout and the
  **points themselves** are stored in `ResumeState` (not a seed: scipy
  could change), so resume and `--replot` rebuild the identical target.
- **Algebra on R, never H.** Two-pass MGS of the projected columns, TSQR
  across blocks, ridge as appended rows, unrolled triangular solves, the
  GHK factor from a second QR of `R^-T T^T`. KOI-448.02's H reached cond
  7e9 (float32 solves 12x wrong); R's is <= 8.5e4.
- **x\* is the exact QP optimum of the centroid model** (face enumeration,
  31 supports, each by QR on R: `simplexmass.qp_simplex_r`), then detached.
  A Euclidean projection was tried and is pinned against by a test.
- **Normal tails** (`normtail.py`): NR erfcc in log form, Acklam inverse,
  and one lower-tail truncated normal after reflecting to a non-positive
  centre (a third of the work of three where-branches; GHK dominates cost).
- **Absolute ridge** (`RIDGE_H_ABS` = 1e-3, 0.1.64, MODEL_REV 9). With no
  transit in the data (k -> 0) the curvature vanishes, and the relative ridge
  vanishes with it. The model's Gaussian then centres ~1e10 away, and GHK
  cannot cancel the magnitudes: on KOI-2650.01 the value came out +3,456
  nats too high at k = 0. That false peak trapped 70 of 512 chains, and
  ChEES's step size collapsed to 1.2e-10. The ridge moves the integrand by
  at most 1e-3 nats, because the simplex is <= sqrt(2) across. Pinned by
  `test_no_transit_gives_the_flat_answer_in_both_precisions` (a mutation to
  0 fails it).
- **The prior is normalised** (`LOG_PRIOR_NORM` = log 4! - sum log F*,
  closed form; -0.768476 on the old basis, -1.258593 on the curvature basis
  since 0.1.67, Monte Carlo checked). Collapsed `loglike` was 0.7685 above sampled
  mode's; it is now on the same scale. Harmless before (a constant), and
  wrong to describe as model error, which is how 0.1.63's review first
  read the offset.
- **Conditional draws** (`HybridOmegaSampler`, rewritten 0.1.64). A
  Metropolis chain against the exact conditional. Each move leaves the
  quadratic model's truncated Gaussian invariant, so only `(L - L_model) +
  log p` enters the test (acceptance 0.92-0.95 on the acceptance targets).
  Each move is, with equal chance:
  - **exact HMC** in whitened coordinates (Pakman & Paninski 2014). Its
    travel time scales as 1/(distance from the mean to the simplex) and is
    capped at the simplex's whitened size, so a thin far-out layer is not
    crossed hundreds of times. `HMC_T` = pi/8.
  - an **exact 1-D draw** along one of the simplex's 10 edge directions.
    These mix a corner, which is a needle in whitened units, where HMC alone
    hit the bounce cap 79% of the time.

  128 steps (`N_STEPS`; fixed, because stopping on the chain's history is
  biased). The chain starts 1e-6 inside the simplex next to `x*`, since the
  mass sits against it. That start cannot become an edge atom: the chain
  moves, and a proposal that rounds onto a face is refused (`_inside`).
  This replaced an independence sampler through 32 GHK draws. That sampler
  was exact in the limit, but GHK's sequential weights degenerate wherever
  the conditional is narrow and off the model's axis. It left KOI-4016.01's
  `q1` 0.41 sd off (85 MCSE); on synthetic thetas it was off by up to 23 sd
  at 32 proposals and 5 sd at 2,048; and it put 3 draws on the q box's
  wall. Hit-and-run, tried next, took hundreds of steps in a corner.
  Validated against references that share no turin machinery: box
  importance sampling of the true L, plus an exponential proposal for a
  vertex-piled case. Lesson: a reference must fit the target's geometry.
  Uniform-q SNIS had an effective sample size of 1-60 at narrow thetas.
  On the curvature basis (0.1.67) the refined q box failed too: q1 =
  depth^4 is thin and long-tailed, and the box cut its tail (q1's sd 0.0076
  against 0.017 at k 0.095, b 0.7, which made the sampler look 2x too
  wide). The box reference now importance-samples in x with the prior weight
  explicit (ESS 700-4,700; the sampler within 0.06 sd and 8% width at every
  test theta).
- **The value's error is GHK's rule error, not the quadratic model's.**
  The model is accurate wherever measured: the draws' acceptance is 0.9+,
  and true minus model was 0.00 at a vertex case. With a consistent layout
  GHK converges to the true marginal as N grows. At 1,024 points it errs by
  0.005-0.064 nats (sd over posterior thetas) on the acceptance targets, and
  by up to ~2 nats at narrow off-axis thetas far from any posterior (0.004
  at 4,096). Accepted and documented, not fixed: a larger rule costs ~2/3 of
  the evaluation per doubling. A probe that integrates one layout's GHK mean
  and covariance under another layout produces a fake "layout bug" of
  ~1 nat. That happened in 0.1.64's review; every layout agrees to 0.001
  when done consistently.
- **Export rows are capped at 160,000** for this mode
  (`pipeline._EXPORT_ROWS_COLLAPSED_HYBRID`): every exported row needs a
  conditional draw (~1.3 ms), and the general 1,000,000-row budget would
  make each export take 20+ minutes. R-hat and ESS still see every draw.
- **Real-target acceptance** (`docs/bench/collapsed_hybrid_acceptance.md`):
  the user's five-target suite (KOI-4016.01, 4087.01, 1430.03, 1361.01,
  2650.01) against sampled hybrid4. The first pass (0.1.63) found the three
  defects above. The second (0.1.64) agrees on every parameter: within
  0.016 sd, 0 shifts beyond 3 MCSE, widths within 1.4%, 0 wall atoms.
  The chain-split null failed 9 of 70 comparisons, all marginal. A second
  seed of both modes showed those were run-level noise: the cross-mode KS
  is no larger than two seeds of one mode (KOI-4016.01: medians 0.0038 vs
  0.0046). The chain-split null is optimistic here, as warned below.
  KOI-2650.01's collapsed runs reproducibly hold 0.3% of draws in a far
  grazing tail (k > 0.06, up to 0.998) that sampled runs essentially never
  visit (0-2 chains). The collapsed density there matches the true marginal
  (+/-0.05 nats at 30 tail thetas), so the likely reading is that sampled
  mode under-explores it; this is not proven.
  **Re-measured on the curvature basis (0.1.76; turin 0.1.74, elome
  0.18.4): passes the same way.** Within 0.015 sd (median 0.004); two
  shifts beyond 3 MCSE (KOI-4016.01 q4 4.1, KOI-4087.01 q1 3.1), both gone
  at seed 1 (0.001, 0.006 sd), and 4016.01's collapsed q4 moved the same
  0.011 sd between its own seeds; widths 0.992-1.015; KS 68/70 against the
  chain-split null, cross-mode no larger than same-mode across seeds on
  4016.01 and 4087.01 (on 2650.01 0.0069 vs 0.0056 at the max, on k, which
  carries the tail; medians equal); 0 wall atoms. KOI-2650.01's tail is
  reproduced (collapsed 0.17-0.44% of draws at k > 0.06, max k 0.86-0.996;
  sampled 0-0.10%, max 0.043-0.128).
  **Cost: 5.7-8.2x the wall time (median 6.7x), 0.09-0.27x the ESS/s
  (median 0.17x)**; on the old basis 3.7-8.7x (median 4.4x) and 0.23-0.53x.
  Collapsed mode did not slow down (8-27% faster); sampled mode got 1.1-2.1x
  faster (elome 0.18.2's contact rule, the anvil MAP search) while the
  per-chain GHK cost, plus ~200 s of post-hoc draws per export, stayed. It
  stays opt-in.
- How the accuracy was mis-measured once, recorded so it is not repeated:
  uniform-simplex QMC is not a reference for a peaked integrand (it was off
  by up to thousands of nats at high SNR and was misread as a bug in x*);
  importance-sample the true L from the model's own GHK draws instead.

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

The mechanism: elome's `ld_basis=True` (MetalPlanet >= 0.7.0; every elome) gives the three
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
- Real-target acceptance (`docs/bench/collapsed_ld_acceptance.md`;
  LinEph, 512 chains, to convergence): every parameter within 0.011 sigma
  and 2.5% width of the default; KS judged against a null of each run's own
  chains split in half (not random rows -- draws are autocorrelated); zero
  `q` draws on the box edge (KOI-448.02: `x*` on a triangle edge for 54% of
  draws). One pairing failed that null (KOI-518.02 `q1`, seed 0); a second
  seed of both modes showed two opposite run-level fluctuations, not a mode
  difference. KOI-448.02 ran `--nongrazing`; real grazing posteriors are
  covered by the 20-target battery (`docs/bench/collapsed_ld_battery.md`):
  14 targets with grazing mass up to 78%, reproduced to within 0.4
  percentage points, every parameter within 0.013 sigma, 0/160 shifts
  beyond 3 MCSE.
  **Cost, minimum ESS over the shared five parameters per end-to-end wall
  second: break-even on KOI-518.02 (156.5 vs 157.4), 1.41x the time per
  effective sample on KOI-448.02 (37.6 vs 53.1); on 20 Kepler targets the
  median is 1.26x the LinEph wall time and 0.77x the ESS/s (faster on 6 of
  20, clearly only at SNR 272).** That is why it stays opt-in.
- **How not to judge a two-sampler comparison, learned here.** Random-row
  halves of one run are not a null for two runs (within-chain
  autocorrelation); split by chain. Even that can be optimistic, since
  ChEES shares adaptation across a run's chains, so a surprising single
  result gets a second seed of *both* modes before it is called a bias or
  a pass. And cost must use end-to-end wall time and the same parameter set
  in both modes: collapsed mode's post-hoc `q` draws happen outside the
  sampling rounds, and its summary has no ESS for `q1, q2`.
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

**Checked against elome 0.20.0** (`3b44004`) **and anvil 0.11.0**
(`231d41f`), turin 0.1.87. elome 0.18.8-0.20.0: eccentric and ringed
fixes and speed (unused), exposures outside the contacts skipped (stated
bitwise), and **the quadratic law's float32 kernels run 8 elliptic-integral
iterations instead of 10**. Measured as the A/B below (same turin, elome
0.18.7 from `git archive` on `PYTHONPATH`, KOI-518.02, 64 vectors, 30
quantities, now with the profiled TTV `--mle` targets round and oblate):
every float64 value and every hybrid4 and oblate float32 one **bitwise**;
the quadratic float32 values and gradients move at rounding -- median 1e-4
nats, about a tenth of their own float32 error, which is unchanged (median
|fp32 - fp64| 1.98e-3 -> 1.67e-3 sampled, 9.2e-4 -> 8.9e-4 collapsed, TTV
and per-epoch terms likewise), gradients by <= 5e-5 of each column's
scale. No MODEL_REV bump: the model is unchanged, its float32 rounding is
not. anvil 0.10.2 (review fixes: `optimize`'s bound settling, `optima`
merging, L-BFGS rechecks; `Transform.to_model` branches; dense ChEES, which
turin does not use) and 0.11.0 (**the diagonal preconditioner's variance
accumulated centred, without the 1e-4-in-u floor**: ChEES draws change for
any warmup past 50 iterations). Measured: KOI-518.02 LinEph + TTV to
convergence under 0.10.1 and 0.11.0 (512 chains, one round each, 394 vs
400 s): every parameter within 0.0072 sd (43 compared), none beyond 3 MCSE
(max 1.77), widths 0.994-1.005, bulk ESS 0.91-1.04x (median 0.99); no
parameter there is narrow enough in u for the floor to have mattered.
`--mle` (round TTV quadratic, LinEph and oblate hybrid4) on the suite plus
KOI-518.02: all 24 ln L identical to 4 decimals, TTV certified every time.
The pipeline's TTV MAP came out 0.47 below in the 0.11.0 run, but that is
its start (from the LinEph chains, which differ), not `optimize`: from the
same starts the TTV MAP is identical under both at 8 seeds, whose own
spread is 1.2 nats. The 0.11.0 TTV run's warmup report said TOO SHORT
(spread still drifting) and converged anyway (R-hat 1.0093). Followed up
(0.1.88) at seeds 1 and 2 under both: TTV TOO SHORT at 1 of 3 seeds under
0.10.1 and 2 of 3 under 0.11.0, every run converged in one round (R-hat
<= 1.0093, ESS 45,000-53,000), LinEph settled every time. So it is
KOI-518.02's TTV warmup sitting at the report's edge (it settles at
iteration ~780 of 800 when it does), not anvil 0.11.0. `MIN_ANVIL` stays 0.9.0. Full suite: 474 passed.

**Checked against elome 0.18.7** (`962aedb`) **and anvil 0.10.1**
(`f3f289f`), turin 0.1.79. elome 0.18.6 fixed oblate and ring gradients at
f = 0 (d/df below the fp32 switch f = 1e-5 is now the ellipse's at the
switch, not 0; values unchanged) and routes the fp64 graph's far roots
through their partner inside the disc; 0.18.7 is documentation. Measured as
the 0.18.5 A/B below (KOI-518.02, 64 vectors): **24 of 26 quantities
bitwise**; oblate sampled and collapsed float64 values differ by 4e-16 and
3e-15 relative (the far-root route). Below the switch turin's d/dfc is now
non-zero, proportional to fc (-1.40e-4 at f = 1e-6, sampled); at f = 1e-4,
turin's `F_FLOOR`, it is unchanged, so no turin behaviour moves. anvil
0.10.0 added `anvil.evidence` (unused) and 0.10.1 keeps dead chains out of
ChEES's mass-matrix moments and trajectory-length gradient (a protection
turin gets for free; healthy runs stated bitwise), plus an `optimize`
rescoring guard. Measured: ChEES on KOI-518.02 (64 chains, 100 warmup + 60
draws, quadratic LinEph and collapsed hybrid4 oblate) gives **bitwise**
draws and log-probabilities under anvil 0.9.1 and 0.10.1. `MIN_ANVIL` stays
0.9.0.

**Checked against elome 0.18.5** (`9a93899`), turin 0.1.77. 0.18.3-0.18.5
touch only ringed planets (`rings.py`, `rings_tau.py`: float64 speed and
memory bounds) and a comment in `api.py`; turin does not use rings.
Measured as an A/B on identical vectors, elome 0.18.2 (`git archive
c5db3eb`, on `PYTHONPATH`) against 0.18.5, KOI-518.02, 64 vectors: **26
quantities bitwise identical** and finite -- the 16 below plus oblate
hybrid4 sampled and collapsed (float32 value and gradient, float64 value)
and the `--mle` profiled targets, quadratic and hybrid4 (float32 and
float64 value). The 0.1.76 acceptance ran on 0.18.4. Full suite: 438 passed.

**Checked against elome 0.18.2** (`c5db3eb`), turin 0.1.71. 0.18.0 added
ringed planets (`flux_dev_ring`, loaded lazily; turin does not use them);
0.18.1 and 0.18.2 skip zero-weight quadrature nodes in the oblate and
spherical contact-rule kernels (~3x faster, stated bitwise). Measured: the
16 spherical quantities on KOI-518.02 (quadratic LinEph sampled, collapsed
and TTV with per-epoch terms; hybrid4 sampled and collapsed; value,
gradient, float64) are **bitwise** turin 0.1.67 on elome 0.17.1. Oblate was
not dumped; its references in `tests/test_oblate.py` pass. Full suite: 408
passed (0.1.71).

**Checked against elome 0.17.1** (`59fff57`), turin 0.1.67. 0.16.3 seeds the
oblate kernels' quartics by Ferrari's closed form (oblate float32 1.2-1.6x
faster, not bitwise; spherical and float64 bitwise); 0.16.4 fixes NaN
float32 gradients for small, nearly circular oblate planets (`r f` below
~1.5e-7; turin's `F_FLOOR` start, 1e-4, was above it for `k > 1.5e-3`);
0.17.0 put hybrid4's weights on the curvature basis (see "The
limb-darkening law"), and 0.17.1 is fixes from its review (`hybrid4_to_q_np`
raises off the simplex; the numpy maps no longer clamp). Measured: the 10
quadratic quantities on KOI-518.02 (LinEph sampled and collapsed, TTV,
value, gradient, per-epoch) are **bitwise** turin 0.1.66 on elome 0.16.2.
The oblate figures in "Oblate planets" were measured on 0.16.2.

**Checked against elome 0.16.2** (`476cb45`), turin 0.1.60, after the
rename. 0.13.x added oblate planets (keyword-only `f=`, `theta=`, hybrid
laws only; spherical paths bitwise unchanged); 0.14.0 renamed the package
with outputs and kernel sources bitwise 0.13.2; 0.15.0 moved hybrid4's
poles; 0.16.x is a public release and review fixes to the frontend and
priors, none on a path turin calls. Measured: turin at 0.1.59 on
MetalPlanet 0.12.2 against turin 0.1.60 on elome 0.16.2, KOI-518.02, 64
vectors: float32 and float64 log-density and float32 gradient **bitwise
identical**, for `--ld=sampled` and `--ld=collapsed` (`ld_basis`) alike.
Signatures unchanged. Full suite: 318 passed, 0 skipped.

Before that, **checked against MetalPlanet 0.12.2** (`e507941`), turin 0.1.58. 0.11.0
added occultors larger than the star (`rp > 1`, a total-occultation regime)
and rewrote kernel sources for it, but states `rp < 1` outputs and VJPs
bitwise unchanged; turin bounds `k <= 1`, and its float32 log-density and
gradient and float64 log-density on KOI-518.02 (64 vectors) are bitwise
identical under 0.10.7 and 0.12.2. 0.11.0 also moved
`orbit.separation_circular`'s far-side push (points behind the star), which
turin's supersampled route calls; those points are out of transit either
way. 0.12.x touches only the hybrid laws, which turin does not use. No
signature turin calls changed. Full suite: 322 passed, 0 skipped.

Before that, **checked against MetalPlanet 0.10.7** (`f69b482`). 0.10.0
added the hybrid2/4/5 limb-darkening laws as keyword-only `limb_dark=`
(default `"quadratic"`) and `u=` on `flux_dev_from_tau` and
`flux_dev_metal`; turin stays quadratic and passes neither, its positional
calls and the quadratic `ld_basis` (`(C, m, 3)`) are unchanged, and
MetalPlanet reports the 18 quadratic kernel sources byte-identical and its
296-array corpus bitwise. 0.10.5-0.10.7 rewrote the data-dtype contract
(`metalplanet.dtypes`: the data's dtype is the computation's, fp64 re-routed
to the CPU stream, off-dtype parameters cast); that changes only calls that
used to raise or that pass numpy/fp64 data on the GPU, and turin passes
explicitly typed mx arrays and does its fp64 on the CPU stream already.
Full suite at 0.1.52: 302 passed, 2 failed -- the hurin parity test of
`segment_epochs`, stale since 0.1.50 and fixed in 0.1.53, not MetalPlanet.
At 0.1.54, with a hurin clone present: 309 passed, 0 skipped, 1 xfailed
(the then-open `extract_near_transit_data` edge case, fixed in 0.1.55).
At 0.1.55: 320 passed, 0 skipped.
`MIN_METALPLANET` stayed 0.7.0 (removed in 0.1.60 with the rename).

**The hurin parity tests skip without a hurin clone** beside turin, so a
session without one can break them unseen: 0.1.50 diverged from hurin's
`segment_epochs` on purpose, reported "14 skipped", and the stale test was
found only when a clone was present. Since 0.1.54 `tests/conftest.py` lists
every hurin skip in a "hurin parity UNVERIFIED" section at the end of the
run, and `TURIN_REQUIRE_HURIN=1` makes them fail; a clone that is present
but will not import now fails rather than skips.

Earlier, **verified against MetalPlanet 0.9.7** (`957eb3a`), turin 0.1.33.
0.8.0 put eccentric orbits on `flux_dev_from_tau` as the
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

The same test checks gradient *accuracy* against elome's frontend at
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
detachment passes every case. That detachment is elome's internal
choice and turin should not pin it. Second, the 1.9e-6 "hard regime" figure
was turin's own finite-`n_gl` error, not the reference's. The FD warning
above is about *turin's* path: finite-differencing it recomputes the splits
and so measures a different quantity from its autodiff (1.5e-5 apart at
`N_GL = 5` wherever the inner pair exists) -- which is why the reference
is the frontend and not turin.

### Why turin does not use elome's sampler-facing API

`elome.anvil.make_quad_transit_flux` and `make_transit_target` bake
in a linear ephemeris, offer no exposure integration, carry only a scalar
`df0` baseline, and parametrize by a/R*. turin needs per-epoch mid-times,
long-cadence integration, per-epoch polynomials, and hurin's
`T14` parametrization. So turin builds `tau` and `z` itself and calls
**`elome.metal.flux_dev_metal(z, r, u1, u2)`** — a fused fp32 kernel
with a VJP in `z`, which falls back to the analytic fp64 path on the CPU
stream (that fallback is turin's `log_prob_hi`). Similarly turin writes its
own likelihood rather than using `ChunkedGaussianLogLike`, which chunks the
data axis across epoch boundaries and so cannot hold a per-epoch solve.

### The transit-count floor and the per-transit SNR gate (`--TTVsnr`, default 3)

**A TTV fit needs `pipeline.MIN_TTV_TRANSITS` = 3 epochs with data in
transit** (0.1.85, the user's call; the count is the epochs whose expected
SNR at the LinEph best fit exceeds `_SNR_COVERED`). With two, a linear
ephemeris fits the times exactly and the sampler only returns the timing
priors; with one, even the period is free. Fewer skips the sampled TTV fit
(`<target>_ttv_skipped.csv` with the count, exit 0 on LinEph's verdict) and
refuses `--mle`'s; `--TTVsnr=0` and `--SNRgate=0` do **not** lift it, and it
is not an option. Found on TESS: TOI-6007.01 and TOI-6075.01 (period 950 and
833 d, two transits each) both ran full TTV fits, whose epoch-0 offset is
zero by construction. Not evaluated without a LinEph
best fit for the lineage (`--modes=ttv` alone), as the SNR test below.

Before a TTV fit, `pipeline.transit_snrs` computes each epoch's expected
single-transit SNR, `sqrt(sum (d/sigma)^2)` with `d` the LinEph best fit's
transit depth at each real point (the matched-filter SNR on noiseless data,
so gaps and window edges count). If the median over epochs with data in
transit is below the threshold, the TTV fit is skipped and
`<target>_ttv_skipped.csv` says why. Epochs whose transit fell in a gap
score exactly 0 and are excluded from the median: 0.1.55's epoch rule fits
them, and counting them would bias the gate toward skipping.

Why: KOI-4926.01 (19 transits, median 2.6) spent 10.4 h on a TTV fit that
ended unconverged at the cap, with timing errors of 1.5-20 h, and freeing
19 times let the shape fit noise (T14 12.3 h against LinEph's 6.8 h; the
best log-likelihood improved 12 nats for 17 extra parameters). Of the
first 59 Kepler targets, 4 fall below 3 (KOI-5870.01 2.96, 5879.01 2.72,
3283.03 2.27, 4926.01 2.61), about 17 h of batch time, two unconverged.
The gate runs before the resume checks (an unfinished TTV lineage from an
older MODEL_REV would refuse first); a converged TTV lineage still reports
as converged.

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

### The float64 re-score releases its graph (`model.release_graph`, 0.1.89)

`anvil.optimize` re-scores its top 16 starts in float64 on the CPU stream in
one unchunked call (`_Fp64Wrapper`). MLX builds the whole lazy graph before
evaluating, and elome's oblate float64 kernel is thousands of small ops per
point, so a target of 11k-60k points exceeded Metal's 499,000-buffer limit
(`[metal::malloc] Resource limit`). `_tau_kernel` therefore evaluates each
call as it is made while `release_graph()` is active, which `_Fp64Wrapper`
enters. **Only for value-only paths**: `mx.eval` inside `mx.grad` or
`mx.compile` is an error. Values are bitwise unchanged.

**Oblate targets are capped at 3 epoch blocks (`likelihood.MAX_OBLATE_BLOCKS`,
0.1.90).** MLX's compiler raises `IndexError: unordered_map::at: key not
found` on the oblate value+gradient over 4 or more blocks, at any batch size
(KOI-254.01 forced from 3 to 6 blocks; KOI-1225.01 passes at 3, fails at 4).
It is not the compile cache (held references still fail), the batch, or one
epoch. anvil catches only MLX's `RuntimeError`, so it escaped `optimize`.
Large targets hit it via the memory budget (KOI-971.01: 26 blocks). The cap
costs no memory (blocking bounds a launch, not the peak). Round layouts are
untouched. Why 4 is the limit is MLX's; not investigated further.

### The MAP search (`seeding.find_map`, anvil.optimize, 0.1.71)

Every fresh fit starts its chains from `anvil.optimize` on 128 starts (a
0.25 sd ball in u around the input ephemeris, the template-sweep times for
TTV): L-BFGS then damped Newton (`method="mixed"`), batches capped at
`seeding.MAX_ROWS` = 512 rows (its Newton step sends `2 dim` gradient rows
per start; a collapsed hybrid4 row is ~3 MB). It replaced turin's own
fixed-step Adam (400 iterations; 1,500 and 512 starts plus a polish for
oblate), which anvil's own benchmark shows ending every start short.

- **`objective="density"`**: the sampler's log-density in u, Jacobian
  included, as Adam maximised. The Jacobian keeps the point off the box
  edges where ChEES can freeze a chain. turin's priors (b-k, TTV, oblate,
  rho*) live inside `model_log_prob`, not on ParamSpecs, so anvil's `"mle"`
  and `"map"` would both include them; a true maximum-likelihood fit needs
  them separated first.
- Measured against Adam on the user's suite plus KOI-518.02 (128 starts,
  pipeline starts, gap in fp64 log-density to the best of the three
  methods): **LinEph, mixed matches Adam on all six (0.000) in 0.3-0.8 s
  against 0.8-1.3 s**; L-BFGS alone 0.000-0.08 short. **TTV: mixed lands
  0.35-3.2 nats below Adam on three of six** (KOI-518.02 dim 36, 0.40;
  1430.03 dim 22, 0.35; 2650.01 dim 47, 3.15, where L-BFGS found the best
  and Adam was 0.94 below): a different local timing optimum, not a
  gradient fault (fp32 and fp64 agree at both points to 5e-4; on 1430.03
  epoch 4's time differs by 0.7 in u, and Adam's point is not even
  stationary, |g| 0.21). Harmless for starting chains -- the typical set
  lies ~dim/2 nats below the peak, and grid-Gibbs resamples each epoch's
  time -- but it means a user-facing ML fit of TTV would need more than
  this. Cost 0.9-3.3 s per TTV search.
- **anvil's `laplace_starts` refuses a Hessian that is not negative
  definite**, and an oblate fit of a round planet always has one (its
  optimum sits in the f ~ 0 dead zone). turin's recipe, which anvil's
  ported, floored such directions at the 0.5 cap instead;
  `seeding.chain_starts` applies that floor to anvil's `hessian_u` before
  anvil draws. A test pins it (`test_chain_starts_are_spread_and_finite`).
- **anvil's `optima` is not a multimodality signal here**: it calls starts
  0.05 apart in u distinct, so one optimum on a wide posterior is listed
  several times (KOI-518.02 LinEph: 7 "optima" within 0.007 nats, spread
  along beta, q1, q2, whose sd in u is 0.8-1.2). The log's warning keeps
  turin's rule, the top decile of the starts' values disagreeing by more
  than 10 (`seeding.top_spread`).
- Both anvil points above are findings, not briefs; whether either goes
  upstream is the user's call.
- **The start ball stays** (measured 0.1.83, scratchpad `map_starts.py`, 4
  seeds, default and LinEph-fitted sweep shape). Holding the timing columns
  near the template seeds (0.02 sd in u, shape still 0.25) reached the best
  on every KOI-518.02 seed where the ball fell 0.12-1.08 short, tied on
  4016.01, 4087.01 and 1361.01, but fell 7.0 short on one KOI-1430.03 seed
  and 5-8 short on 2650.01 (below `--TTVsnr` anyway); half held, half ball
  dominated neither. Under a nat at the MAP does not matter for starting
  chains, so no change.

### Maximum likelihood (`--mle`, `mle.py`, 0.1.72-0.1.74; TTV 0.1.82)

A fit without sampling, built on the user's request with these decisions
taken with them: **the best fit only**, no uncertainties (0.1.74); the
**likelihood alone** (priors off, `--rhostar` refused), LinEph first
and **TTV from it since 0.1.82** (below; under `--oblate` too since
0.1.84), and
**`--oblate` also fits the round model** and reports
ΔlnL/ΔBIC/ΔAIC (no p-value: at f = 0 theta is undefined and f is on its
boundary, so Wilks does not apply). Products are `{T}_lineph_mle_*`
(`fit`, `lcdata`, `fold`, and `round`/`compare` under `--oblate`); no
resume state, no lineage, so an MCMC lineage of the same target is never
touched (pinned by a test). `--fresh` clears one family only
(`outputs.clear_products(mle=)`, 0.1.80): both share the `{T}_{mode}_`
prefix, and until then `--fresh --mle` deleted the MCMC lineage and an MCMC
`--fresh` the MLE products.

- **`build_target(likelihood_only=True)`**: `prior=False` on the
  likelihood classes keeps only `oblate_log_prior` (a 0/-inf domain mask
  that must stay: outside the domain `oblate_fields` feeds the kernel a
  stand-in point, so the likelihood there is finite but meaningless);
  `model_log_norm = log_const` so anvil's `log_value` is the absolute ln L.
  Sampling targets are untouched (bitwise, 16 quantities).
- **Limb darkening profiled, at the user's suggestion**: every law is a
  convex mix of `ld_basis` generator curves, so `ProfileLDLogProb` /
  `ProfileHybridLogProb` maximise `L(theta, x)` over the simplex by
  `N_PROFILE` = 4 passes of the collapsed modes' own Newton step + exact QP
  (a sequential QP; 4 passes reach rounding on three real targets x two
  laws), detached, with the envelope-theorem gradient. Exact for a maximum;
  *not* for sampling (it drops the LD volume -- the 0.1.39 bias), which is
  why the collapsed modes integrate. Verified against a brute-force simplex
  maximum (lattice/Dirichlet + float64 SLSQP, every quadratic case and
  hybrid4 round/oblate), float64 finite differences, and the joint
  (theta, q) likelihood, which cannot beat it (on hybrid4 the joint search
  from wide starts ended 0.010 nats short, weights on faces).
- **The best fit only, no uncertainties** (the user's decision, 0.1.74:
  "mle is just the best fit"). 0.1.73 reported Laplace sds from the joint
  curvature and held flat directions; they were removed with
  `MIN_CURVATURE` and the joint polish. What that taught, kept here: at
  b near 0 the likelihood is flat in beta to float32 noise (KOI-518.02,
  -0.02/+0.05 in u), and sds conditional on such a held parameter were
  far narrower than the posterior (k 4.2e-4 vs MCMC 1.2e-3).
- **The oblate search starts from the round fit only** (the user's
  scheme, 0.1.74): round first, then all 128 starts at its geometry (0.02
  sd in u) with f = `F_FLOOR` at random angles, in (fc, fs) as the sampler.
  Measured (`mle.ROUND_SEED_SD`'s comment) against a ball around the input
  ephemeris, alone and with 8 round seeds (0.1.73), 3 seeds each: six
  PHOENIX truths all agree within 0.01 nats, round seeding 20-40% faster;
  on the suite best on KOI-1430.03 (the ball alone once ended 0.076
  *below* the round fit there), level on 4087.01 and 1361.01, 0.04-0.27
  short of the ball's best seed on 2650.01 and 4016.01, whose optima lie
  on a flat ridge at the F_MAX edge (no scheme was consistent there).
- **TTV** (0.1.82, `mle.fit_ttv`; plan and decisions taken with the user
  2026-10-20). Three agents looked for why `anvil.optimize` missed TTV
  optima (0.1.71: 0.35-3.2 nats short on 3 of 6). The template seeds were
  not the cause; the MAP's 0.25 sd start ball is, since it moves every
  epoch's time ~0.56 T14 in u-space, and the sweep's tie-break and prior
  are a posterior's choices. The search:
  - **Gate:** the median expected single-transit SNR at the LinEph best
    fit (`transit_snrs`) against `--SNRgate` (default
    `pipeline.MLE_SNR_GATE` = 5, the user's; refused without `--mle`);
    below it the run stops after LinEph. It never touches the MCMC
    lineage's resume state or `ttv_skipped` note.
  - **Block coordinate ascent** from the LinEph fit's times
    (`lineph_to_ttv`): `timing_step` puts every epoch at its float64-
    checked best over its whole box (512-cell float32 grid from one
    batched `epoch_log_lik` call on the sampled-LD likelihood with q at
    the profile's maximiser, a 64-point zoom, moved only where its term
    rose), then `anvil.optimize` polishes everything on the profiled-LD
    target (32 starts: shape 0.25 sd in u, half spread over beta; times
    0.002 sd). Until a timing step moves no epoch (the certificate; cap 5
    passes). The LD is fitted: held only within the timing step.
  - **Products** `{T}_ttv_mle_*`: fit, times (errors NaN, O-C against the
    LinEph fit's P and tau0), lcdata, fold, oc (no error bars), compare
    (dlnL, n_extra, dBIC, dAIC; no p-value).
  - **Measured.** Synthetic: decoy data (`test_mle_ttv.py`: each epoch at
    least its dense float64 grid maximum; a local-only mutation fails it)
    and an injected 12-minute TTV end to end (times within 2 min, two
    passes, 2.4 s). Real (0.1.83, scratchpad `ttv_mle_bench.py`): against
    0.1.71's Adam, mixed and L-BFGS from the old starts, float64 ln L on
    the likelihood alone and LD-profiled at their points too, `fit_ttv`
    matches or beats all three on all six -- KOI-518.02 +0.17 over the
    best, 1430.03 +0.65, 1361.01 +0.017, 4016.01 and 4087.01 equal to
    mixed -- in 0.7-6.2 s, certified every time. The gate at 5 refuses only
    KOI-2650.01 (median SNR 2.26); run anyway, the search ends 50 nats above
    the others there, which is noise being fitted -- the gate's reason.
    KOI-315.01 (TTVs to 4 h from the catalogue ephemeris): SNR 20, dlnL
    +3,579 for 36 times, certified, 40 s end to end. **Global check** there
    (scratchpad `k315/mle_vs_mcmc_ttv.py`): a TTV MCMC (512 chains,
    grid-Gibbs, converged in 26 min) scored draw by draw on the float64
    likelihood alone; the MLE beats all 230,400 draws by 6.67 nats, against
    6.71 expected for the best of that many draws from a 43-dimensional
    Gaussian peak (median draw 21.2 below, theory 21.17), and its times
    sit within 0.80 posterior sd of the medians. Passes 3-5 there moved
    epochs for +0.001 nats or less; on the oblate fit (below) such moves
    kept the certificate from ever firing, so since 0.1.84 an epoch counts
    as moved only above `mle.MOVE_NATS` = 1e-3 (a smaller rise is still
    taken).
  - **Oblate** (0.1.84, the user's request). `--mle --oblate` with TTV:
    round LinEph, oblate LinEph (as before), the gate at the round LinEph
    fit, the round TTV fit from it, then the oblate TTV fit
    (`fit_ttv(oblate=True, round_fit=)`) from the better of the two
    starts it nests -- the oblate LinEph fit at `dtau_i = dtau0 + n_i dP`,
    and the round TTV fit at f = `F_FLOOR` -- with the other and
    `N_TTV_ANGLES` = 8 round-fit copies at random angles (0.02 sd in u,
    the `ROUND_SEED_SD` scheme) among the first polish's starts. Products
    are the oblate fit's plus `ttv_mle_round.csv`; `compare.csv` keeps
    its TTV-vs-LinEph rows and adds `*_oblate` oblate-vs-round rows.
    `collapsed_ld_problem` stands aside under `--mle` (`--oblate` sets
    `ld` to collapsed, but `--mle` profiles whatever it says), and
    `cli.resolve_mle` is gone. Tests: the oblate TTV profile nests LinEph's
    at dP = 0 (value, shape gradient) and is the simplex maximum; the
    oblate sampled-LD terms sum to the likelihood and move one epoch at a
    time; sampling refused; pipeline end to end on the injected TTV
    (round and oblate fits, the nesting, times within 2 min).
    KOI-315.01: round TTV 21349.973 (certified, one pass, 25 s); oblate
    TTV 21350.327 (one pass, 60 s polish, 2.6 s timing step), dlnL 0.35
    over round, dBIC +15.1; f = 0.600 at the F_MAX edge (the ridge, as
    for LinEph). 163 s of compute for all four fits (the run took 697 s
    because the Exoplanet Archive returned 526 errors for 530 s; the
    first run, 817 s, had five uncertified passes and probably the same
    outage). The oblate LinEph fit there is a poor grazing one (k 0.16,
    b 1.06, 1 of 128 starts converged): a linear ephemeris smears a
    transit with 4-hour TTVs, and the TTV fit does not depend on it,
    since the round TTV start won.
  - **No Laplace step** (`mle.HESSIAN = False`, 0.1.83): `--mle` reports
    the best fit only, and anvil's step can raise. A flat epoch's Hessian
    row is exactly zero; rounding gave `eigh` +1.8e-12, so anvil's
    negative-definite check passed and its `inv` raised `LinAlgError`
    (KOI-4087.01, anvil 0.10.1). An anvil finding, not a brief (the user's
    call). `seeding.find_map` keeps the step (chain starts need it); its
    density objective has the box Jacobian's curvature in every column, so
    no row is exactly zero.
- **Profiled LD on TTV layouts** (0.1.81, the base of a TTV `--mle`):
  `params.ttv_layout(..., ld="collapsed")` drops the LD coordinates;
  `ProfileLDLogProb` / `ProfileHybridLogProb` accept a TTV layout
  (`TTV_OK`), while the `Marginal*` sampling classes still refuse it and
  `epoch_log_lik` stays refused (a shared LD couples the epochs). Both
  classes now use the parent's by-name `unpack` (26 quantities bitwise).
  **Nested only up to the period:** at `dtau_i = dtau0 + n_i dP` the TTV
  profile equals LinEph's exactly when `dP = 0`; otherwise TTV mode holds
  the period at the reference (as hurin), and the period also sets the
  shape through a/R* (8e-9 and 2e-7 nats at dP = 2e-5 d on the test
  cases). Tested as the LinEph profile: brute-force simplex maximum, FD
  gradient, the joint (theta, q) MLE, the TTV prior off.
- **The round search spreads its starts over b** (0.1.75,
  `mle.round_starts`): of 128 starts, `N_SPREAD_BETA` = 64 have beta
  stratified over (0, 1), the rest the 0.25 ball around the input
  ephemeris. 0.1.73-0.1.74's ball alone missed the best fit on two of the
  suite's five targets: found by evaluating the exact float64 likelihood
  alone at **every draw** of a LinEph MCMC run (153,600 per target; screened
  in float32 on the GPU, the top 5,000 in float64; scratchpad
  `mle_vs_mcmc2.py`). KOI-4016.01 and KOI-1430.03 hold a second peak at
  b ~ 0.85-0.91 with weak limb darkening, 0.33 and 0.10 nats above the
  central chord with limb darkening on a corner (q1 = 1, q2 = 0 on 4016.01)
  where every start landed; the top-spread warning cannot see it, since all
  starts agreed. With the spread the MLE beats every draw on all five
  (+0.02 to +0.13), at seeds 0-3 on both targets. **Sampling is not
  affected**: the posterior in b there is broad and nearly flat over
  0-0.9 (two bumps on a ridge, not separated modes), and chains from the
  single MAP-centred ball cross it (KOI-4016.01: R-hat 1.007, bulk ESS
  56,000 on beta in one round), so `seeding.find_map` keeps its ball.
- The user's suite, `--mle --law=hybrid4 --oblate` (Kepler long cadence,
  0.1.75): dlnL 0.001-1.42 and dBIC +10.5 to +16.1 (round preferred) on all
  five; f's optimum drifts to the F_MAX edge (0.596-0.600) on four, where
  the likelihood is nearly flat in f, and the fit notes it. The comparison,
  not f's value, is the result. The spread raised two round hybrid4 fits
  (KOI-4016.01 2374.632 -> 2374.924, which took its oblate fit to 2375.545,
  dlnL 0.621). Wall time: ~5 s round (quadratic), 20-60 s oblate (both
  fits), data loading included.

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
contend; and confirm both packages hold the same cached light curve: compare
the `time`, `flux` and `flux_err` arrays of `../hurin/cache/<T>.pkl` and
`~/.cache/turin/<T>.pkl` (not byte sizes: since 0.1.55 turin's cache also
holds per-point `segment` labels). Expect turin to fit more epochs than
hurin where a transit fell in a gap (`docs/hurin-differences.md` section 4).

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

Dev environment is `turin/.venv` (Python 3.11), with elome and anvil
installed editable from the sibling clones:

```bash
.venv/bin/python -m pip install -e ../elome -e ../anvil
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest
```

No conda environment on this machine has MLX; the `hurin` conda env
(`~/miniconda3/envs/hurin`) has jaxoplanet and NumPyro and is used only to
generate hurin parity fixtures.
