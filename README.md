# turin

GPU transit fitting for Kepler and TESS, on Apple Silicon.

turin downloads PDC light curves from MAST, queries ephemerides from the NASA
Exoplanet Archive, detrends each transit's local baseline with a
profile-likelihood Legendre polynomial, and samples the transit parameters
with hundreds of parallel chains on the Metal GPU.

It is the successor to [hurin](https://github.com/davidkipping/hurin), and
keeps that package's science pipeline and command line while replacing both
compute-bound halves:

| | hurin (CPU) | turin (GPU) |
|---|---|---|
| forward model | jaxoplanet | [MetalPlanet](https://github.com/davidkipping/MetalPlanet) |
| sampler | NumPyro NUTS, 2 chains | [anvil](https://github.com/davidkipping/anvil) ChEES-HMC, 512 chains |
| precision | float32 (JAX x64 off) | float32 GPU with an exact float64 reference path |

---

## Install

Requires Python >= 3.11 and Apple Silicon (MLX has no CUDA or x86 backend).

```bash
git clone https://github.com/davidkipping/turin.git
cd turin
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
```

For development against local clones of the sibling packages:

```bash
.venv/bin/python -m pip install -e ../MetalPlanet -e ../anvil
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest              # add -m slow for injection-recovery
```

## Quick start

```bash
turin --KOI-448.02
turin --TOI-406.01
```

Each run writes its products into a `<TARGET>/` directory and can be
interrupted: products are re-exported after every sampling round, and
re-running the same command resumes where it stopped.

```bash
turin --capabilities        # what the installed anvil/MetalPlanet support
turin --help
```

## What a fit does

1. **Download and condition.** Quality mask, sigma clip with a
   degenerate-window guard, masking of other known planets in the system, then
   recentring the reference epoch on the median occupied transit, which
   decorrelates the period from the epoch.
2. **Choose a polynomial order per transit** by 10-fold cross-validation on
   the out-of-transit points, K = 0..5.
3. **Fit the linear ephemeris** (7 parameters), then **fit per-transit
   times** (5 shape parameters plus one time per transit), seeded from the
   linear-ephemeris maximum likelihood.

Finite exposures are integrated by the contact rule inside MetalPlanet's
kernel, which matters more than it sounds: the supersampling it replaced was
wrong by ~2% of a transit depth at a typical sub-exposure count, and got
`dF/d(period)` wrong by ~100x with the wrong sign.

The baseline polynomial coefficients are never sampled. They are solved
analytically inside every log-density evaluation — a profile likelihood — so a
30-transit fit carries 7 parameters rather than 7 + 90.

### Fit modes

| mode | parameters | notes |
|---|---|---|
| `lineph` | `dP, dtau0, k, beta, T14, q1, q2` | period and epoch as offsets from the recentred ephemeris |
| `ttv` | `k, beta, T14, q1, q2` + one `dtau` per transit | period fixed; timings seeded by a template sweep |

`beta` is the impact parameter as a fraction of its k-dependent bound, which
is how a fixed box represents hurin's conditional-uniform (b, k) prior; `b`
and `log10_rho` are derived in float64 after sampling and appear in every
product.

### The (b, k) prior

Whether a transit exists at all depends on `b` and `k` together — the planet
crosses the star only for `b < 1 + k` — so the two cannot be given
independent priors without distorting both. All three modes sample
`beta ~ U(0, 1)` and map it to `b`, differing in the bound and in the
correction that keeps the *joint* density uniform rather than merely
conditional:

| `--bprior=` | `b =` | region | extra log-prior |
|---|---|---|---|
| `transiting` (default) | `beta (1 + k)` | all transiting geometries | `+log1p(k)` and a linear grazing taper |
| `nongrazing` | `beta (1 - k)` | non-grazing only | `+log1p(-k)` |
| `box` | `2 beta` | `b < 2`, independent of `k` | none (legacy) |

- **`transiting`** is the default and the right choice for almost everything.
  It covers grazing geometries, and its two terms do two things. `+log1p(k)`
  makes the density uniform over the transiting region rather than merely
  conditionally uniform in `b`. The taper then makes the marginal `p(k)`
  **exactly** uniform: at each `k` the grazing band `1-k < b < 1+k` has width
  `2k` and the taper integrates to 1/2 across it, so it carries prior mass
  exactly `k` against `1-k` non-grazing, and the two sum to 1 independent of
  `k`. Without the taper `p(k)` would go as `1+k`, favouring large planets.

  Two readings of "grazing odds" follow, and it is worth keeping them apart.
  *Marginally*, over the default `k ~ U(0, 1)`, grazing and non-grazing have
  equal prior mass — 1:1, since `k` and `1-k` each integrate to 1/2. *At a
  fixed `k`*, the odds are `k : (1-k)`, so for a shallow target like
  KOI-518.02 (`k = 0.023`) only about 2% of the prior mass is grazing. That is
  the correct geometric weighting, not a bias — a small planet genuinely has
  little room to graze — but do not read the 1:1 figure as "grazing is a
  coin-flip for my planet".
- **`nongrazing`** (alias `--nongrazing`) *asserts* that the transit is not
  grazing, restricting to `b < 1 - k`. Use it when that is a claim you want to
  make: it matters most for systems sitting at the boundary, where it visibly
  tightens `k`, `b` and `T14`. KOI-448.02 is the example in
  `docs/hurin-differences.md` (`b = 0.949` against `1 - k = 0.952`).
- **`box`** is hurin's historical prior, two independent uniforms, and is kept
  only to reproduce old results. Avoid it for new work. With `k ~ 0.02` about
  half its prior volume sits at `b > 1 + k`, where no transit occurs — harmless
  to correctness, since the likelihood is negligible there, but wasted
  sampling — and because `b`'s range no longer tracks its physical `k`-dependent
  limit, the implied marginal `p(k)` is not uniform.

The prior is a resume guard, so changing it on an existing lineage exits
rather than pooling chains drawn under different priors. `model.impact_parameter`
holds the coordinate map and `params.bk_log_prior` the weights.

### Choosing the profile solve

The baseline coefficients can be solved three ways (`--PL`), trading exactness
against speed. Which is right depends on the target, not on the code, so by
default turin measures rather than guesses: it compares all three against a
float64 reference over a ball scaled to the posterior's *own* width, then
times them, and takes the fastest that is no less precise than `exact`. This
costs a few seconds, is printed in full, and the winner is recorded in the
products and reused on resume. Naming a mode skips the probe.

The calibration to the posterior width is the part that matters. Judged on a
ball much wider than the posterior, `ratio` looks catastrophic; judged on one
much narrower, everything passes. Only at the posterior's own scale does the
verdict mean "this would distort the answer".

### Weak transits and grid-Gibbs

A transit observed at low signal-to-noise, or with a data gap across it, has a
timing posterior with several separated bumps, often hours apart. ChEES-HMC
does not move chains between such bumps. Left alone, each bump is weighted by
however many chains happen to settle there, and that transit's R-hat never
passes however long the fit runs.

So by default the TTV fit interleaves an exact Gibbs step. Every 100 draws,
each chain redraws every transit time from its conditional posterior, given
that chain's current shape parameters. The conditional is evaluated on a grid
spanning the whole timing prior, so a chain can land in any bump, and a
Metropolis-Hastings correction keeps the step exact. It is possible because,
for fixed shape, each transit's likelihood depends on its own time alone.
That also makes one sweep cost about one extra log-density evaluation per
grid point however many transits there are.

On KOI-4848.01 and KOI-5897.01, against exact timing marginals computed by
brute force, this cut a weak transit's distance from the true posterior by
5-8x (e.g. 0.162 to 0.019 in total variation). `--gibbsgrid=off` gives
independent ChEES chains only. The setting is a resume guard, so a lineage
cannot pool chains drawn both ways.

## Options

```
--fresh                      ignore existing resume state for this lineage
--extend1 / --extend2        extend the LinEph / TTV fit by one round
--modes=lineph,ttv           which fits to run
--tag=NAME                   namespace a run into its own resume lineage

--chains=N                   sampling chains (default 512, raised to 4*dim)
--sampler=chees|ensemble     gradient-based (default) or gradient-free
--warmup=N --samples=N --max-samples=N --leapfrog=N --seed=N
--gibbsgrid=on|off           TTV fits: move chains between timing modes with
                             an exact grid-Gibbs step (default on), see below

--bprior=transiting|nongrazing|box      (b, k) prior, see above; --nongrazing
                             is an alias for --bprior=nongrazing
--TTVmax=MINUTES             declared TTV amplitude; sets the timing priors
--PL=auto|exact|hybrid|ratio how the baseline coefficients are solved
                             (default auto: measured per target)
--geometry=circular|chord    true circular orbit (default) or hurin's chord

--sc                         prefer short cadence
--cache-dir=PATH --outdir=PATH --clear-cache
--capabilities --version --help
```

## Output products

Filenames and columns match hurin exactly, so existing analysis scripts keep
working. Seven products per LinEph fit, nine per TTV fit:

```
<T>_<mode>_summary.csv        parameters, percentiles, R-hat, bulk/tail ESS
<T>_<mode>_chains.csv.tar.gz  the full joint posterior, plus b, log10_rho, loglike
<T>_<mode>_logrho.csv         the stellar-density posterior
<T>_<mode>_lcdata.csv         time, flux, flux_err, model_flux
<T>_<mode>_fold.pdf           phase-folded transit, baseline divided out
<T>_<mode>_corner.pdf         corner plot with 1/1.5/2-sigma contours
<T>_<mode>_resume.pkl         resume state
<T>_ttv_times.csv             epoch, tmid, tmid_err, O-C, SNR, npts, chi2
<T>_ttv_oc.pdf                O-C against a refitted linear ephemeris
```

Every product records turin's version and the exact command that made it.

## Differences from hurin

Against **hurin >= 0.1.68** two differences remain, and
`--geometry=chord --PL=ratio` reproduces hurin for comparison:

- **Orbit.** jaxoplanet's `TransitOrbit` is a straight-chord constant-speed
  approximation, not a Keplerian orbit; turin uses the true circular
  projection, which is also what makes the reported `log10_rho` consistent
  with the fitted geometry. Worth 4.6e-6 in flux at T14/P = 0.019.
- **Profile form.** hurin fits the baseline in ratio space, an O(depth)
  approximation; turin defaults to the exact flux-space profile.

A third difference is now **resolved upstream**: hurin's limb-darkening map
dropped both factors of two from Kipping (2013), so its prior could not reach
the negative `u2` that real stars often prefer. turin's port surfaced it and
hurin 0.1.68 adopted the correct map, so both packages now agree. It mattered:
on a grazing system the truncated prior was quietly tightening `k` and `b`.

Limb darkening is Kipping (2013) throughout. That is compatible with the
polynomial (Agol et al.) formulation MetalPlanet implements, because the
quadratic law is exactly that law's `N=2` case
(`I(mu)/I0 = 1 - sum u_n (1-mu)^n`); the Green's basis MetalPlanet uses
internally is an affine change of basis, not a different law. Both are pinned
by tests.

### Validated against hurin on KOI-518.02

The headline check, and the primary comparison in
`docs/hurin-differences.md`: hurin 0.1.68 and turin 0.1.12 fitting the same
cached Kepler light curve (27 transits, 513 ppm deep), both converged, turin
at its **defaults** rather than in compatibility mode.

| | agreement |
|---|---|
| LinEph, all 7 parameters + `b`, `log10_rho` | ≤ 0.02 sigma |
| TTV, 5 shape parameters | ≤ 0.03 sigma |
| 27 transit times | median 0.01 sigma, worst 0.02 sigma |

Posterior widths agree to 0.98-1.11x. Nothing is shared between the two
implementations — MLX against JAX, ChEES-HMC against NUTS, an unrolled
Cholesky against `jnp.linalg.solve` — so this is evidence for both. The target
is chosen precisely because the two remaining model differences are negligible
on it (`T14/P = 0.005`, 513 ppm), which is what leaves the samplers as the
only thing being compared; on a grazing or deep system they are *not*
negligible, and `docs/hurin-differences.md` measures that case too.

`docs/hurin-differences.md` has all the measurements and the reasoning.

## Accuracy

Pinned by the test suite, against independent references:

| check | tolerance | measured |
|---|---|---|
| float64 model vs MetalPlanet's verified frontend | 1e-14 | 3.3e-16 |
| float32 model vs the same | 5e-7 | 1.6e-7 |
| exposure integration vs MetalPlanet's contact rule | 1e-14 | <1e-14 |
| exposure integration vs a high-order float64 reference | 1e-7 | 6.4e-8 |
| profile solve vs `np.linalg.solve`, float64 | 1e-9 | ~1e-12 |
| `ratio` mode vs hurin's own solve | hurin's float32 floor | 1e-8 |
| `hybrid` vs `exact` (3 refinements, 1% depth) | 1.2e-7 | 5.4e-9 |
| float64 gradients vs finite differences | 1e-5 | 2e-9 to 2e-7 |
| float32 log-density vs float64 (`validate_precision`) | 0.1 | 3e-4 |

Every run reports `anvil.validate_precision` before sampling and refuses to
start if the float32 error approaches the Metropolis scale.

## Performance

A fit's cost is one log-density-and-gradient evaluation per leapfrog step,
times ~30 steps per iteration, times the iterations. Against turin 0.1.7,
same machine, same anvil, same command, same seed:

| | 0.1.7 | now |
|---|---|---|
| one value+gradient (KOI-448.02, 512 chains) | 45.8 ms | **9.1 ms** |
| KOI-448.02, 30 transits, LinEph, 256 chains | 549 s | **118 s** |
| KOI-5162.01, 3 transits, LinEph + TTV | 220 s | **119 s** |

Posterior medians agree to 0.02 sigma across the two, and the KOI-5162.01
transit times to 0.05 sigma against the 0.1.0 run in `docs/`.

Against **hurin** on KOI-518.02 (27 transits, LinEph + TTV, both packages
converged in one round, same machine, sequential):

| | hurin 0.1.68, 8 NUTS chains | turin 0.1.12, 512 ChEES chains |
|---|---|---|
| wall clock | 6,037 s | **544 s** (11.1x) |
| min bulk ESS, LinEph / TTV | 453 / 1,519 | **79,145 / 70,942** |
| ESS per second | 0.22 / 0.38 | **324 / 285** (1,460x / 750x) |
| peak memory | 4.8 GB | 17.3 GB |

The ESS gain is much larger than the wall-clock gain, which is the point:
hundreds of chains is what the GPU buys, and it yields two to three orders of
magnitude more independent draws per second. The cost is memory — 512 chains
peaked at 17.3 GB here, so a 16 GB machine needs `--chains` lowered.

Almost all of what is left is MetalPlanet's kernel; turin's own profile solve,
residual and priors are 0.7 ms of the 9.1. Two things got it there, and both
are turin's responsibility rather than MetalPlanet's:

- **Points go to the kernel in phase order** (`model.phase_order`). The
  kernel runs one point per GPU thread in SIMD groups of 32, a group pays for
  any branch one member takes, and an in-transit point costs ~8x an
  out-of-transit one. Ordered by transit, epoch after epoch with ~10% of each
  window in transit, nearly every group held a transit point and paid the
  full price. Sorting by time from mid-transit — a static permutation, since
  the reference ephemeris is fixed — was worth **2.7x** on its own, with
  bit-identical log-likelihoods.
- **Padded slots sit at quadrature.** Epochs are padded to a common length,
  and those slots were parked at the epoch *centre*, where each cost a full
  in-transit evaluation despite being weighted zero.

If you add another route into the kernel, keep both: they are invisible in
the answer and together worth 1.6x (KOI-5162.01) to 3.3x (KOI-448.02) on the
log-density, the more the larger a fraction of each window is out of transit.

`N_GL`, the Gauss-Legendre nodes per contact sub-interval, is 5. It is chosen
on the log-likelihood rather than on the kernel's own `dF/d(period)`: at five
nodes the error in `logL` has sd 5e-4, 40x below the float32 noise the `--PL`
probe already accepts, and the worst gradient component is off by 0.5% of a
posterior sd against float32's own 1.2%.

## Reliability

hurin's headline failure was chains settling into a secondary transit-timing
mode and reporting a confident wrong answer — R-hat 1.01 while sitting 35.6
log-units below the global optimum. turin carries three defences:

- **Template-sweep seeding.** Each transit time is seeded at its own posterior
  peak, found by sliding the transit template across the allowed range, with
  an edge margin, interior-local-maxima-only candidates, and a tie-break
  toward the predicted time.
- **A rival-gap warning** per transit, when a competing timing mode is within
  10 log-units of the seed.
- **A trapped-chain check** after every round, from per-chain log-probability
  and acceptance, which reports posterior widths both with and without the
  flagged chains. No chain is ever silently dropped.

## Not in this version

The Flask web frontend, profile-likelihood transit times (hurin's TTV-PL),
eccentric orbits, and Gaussian-process noise. The likelihood keeps a noise-model
seam so [anvil-gp](https://github.com/davidkipping/anvil-gp) can be dropped in:
its `GPLogLike` takes the transit-times-baseline product as a mean function
with no adapter.

## Citation

If turin contributes to a publication, please cite it along with MetalPlanet
and anvil.
