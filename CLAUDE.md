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
  `anvil_gibbs_reply.md`. turin's fallback (which refuses any state it does
  not recognise) stays for older installs, and a test requires the two
  paths to give bit-identical chains.
- `docs/upstream/metalplanet_prompt.md` — optional: a `tau`-input fused
  kernel with in-kernel exposure integration.
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

`turin/capabilities.py` feature-detects every upstream capability and
falls back when it is absent. All four anvil asks and the MetalPlanet one
have since landed; the fallbacks remain because turin must keep working
against older installs. **turin must always run against the
packages as currently published on GitHub.** When adding a dependency on
an upstream feature, add the detection and the fallback in the same
change.

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
  outputs.py      CSV/PDF products, resume state
  capabilities.py upstream feature detection + fallbacks
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

Two modes, `--profile=`:

- **`exact`** (default): the true flux-space profile, design matrix
  `F L` (with `F = diag(f_transit)`) against target `y - f_transit`.
- **`ratio`**: hurin's form, a WLS fit in ratio space (`y / f_transit`,
  weights `1/sigma^2`), which drops the `f_transit` factors from the
  weights. It agrees with `exact` to O(depth) and exists to reproduce
  hurin's likelihood surface for validation.

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
TTV fit therefore interleaves `gibbs.GridGibbs` every 100 draws
(`--gibbsgrid=on`, the default). It rests on one structural fact: **given the
shape parameters, each epoch's likelihood term and timing prior depend on
that epoch's time alone**. Anything that breaks that factorization (noise
correlated across epochs, a dynamical TTV model coupling the times) breaks
the move, and must turn it off rather than leave it running.

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
- **Provenance stamping** on every product: `# turin <version> | <launch
  command>` on **line 2** of CSVs (after the header, because
  `np.genfromtxt(names=True)` treats a leading comment as the header), a
  `turin_version.txt` member appended after the CSV in tarballs (readers
  use `getmembers()[0]`), PDF Creator/Subject metadata, and version +
  command keys in resume state.
- **Product filenames and columns match hurin exactly**, so existing
  analysis scripts keep working: 7 LinEph products, 9 TTV.
- `--tag=<name>` namespaces a run into an independent auto-resume lineage,
  inserted into every product filename before the extension.
- Resume state records `bprior`, `ttv_max`, `tag`, `profile`, `sampler`
  and chain count; a mismatched resume exits naming the stored value.

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
