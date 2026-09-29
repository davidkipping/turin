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

## Options

```
--fresh                      ignore existing resume state for this lineage
--extend1 / --extend2        extend the LinEph / TTV fit by one round
--modes=lineph,ttv           which fits to run
--tag=NAME                   namespace a run into its own resume lineage

--chains=N                   sampling chains (default 512, raised to 4*dim)
--sampler=chees|ensemble     gradient-based (default) or gradient-free
--warmup=N --samples=N --max-samples=N --leapfrog=N --seed=N

--bprior=transiting|nongrazing|box      (b, k) prior; --nongrazing is an alias
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

`docs/hurin-differences.md` has the measurements and the reasoning.

## Accuracy

Pinned by the test suite, against independent references:

| check | tolerance | measured |
|---|---|---|
| float64 model vs MetalPlanet's verified frontend | 1e-14 | 3.3e-16 |
| float32 model vs the same | 5e-7 | 1.6e-7 |
| exposure integration vs MetalPlanet supersampling | 1e-14 | 3.3e-16 |
| profile solve vs `np.linalg.solve`, float64 | 1e-9 | ~1e-12 |
| `ratio` mode vs hurin's own solve | hurin's float32 floor | 1e-8 |
| `hybrid` vs `exact` (3 refinements, 1% depth) | 1.2e-7 | 5.4e-9 |
| float64 gradients vs finite differences | 1e-5 | 2e-9 to 2e-7 |
| float32 log-density vs float64 (`validate_precision`) | 0.1 | 3e-4 |

Every run reports `anvil.validate_precision` before sampling and refuses to
start if the float32 error approaches the Metropolis scale.

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
