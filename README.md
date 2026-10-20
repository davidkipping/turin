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
| forward model | jaxoplanet | [Elómë](https://github.com/davidkipping/elome) (`elome`, formerly MetalPlanet) |
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
.venv/bin/python -m pip install -e ../elome -e ../anvil
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
turin --capabilities        # what the installed anvil/elome support
turin --help
```

## What a fit does

1. **Download and condition.** Quality mask, sigma clip with a
   degenerate-window guard, masking of other known planets in the system, then
   recentring the reference epoch on the median occupied transit, which
   decorrelates the period from the epoch.
2. **Cut one window per transit**, 5 durations either side of every
   predicted transit time. A window that spans a Kepler quarter or TESS
   semi-sector boundary, where the flux often jumps, keeps only the side
   holding the predicted time. Any window with at least 2 points is fitted,
   whether or not the transit landed in the data: with TTVs possible, that
   is for the fit to decide.
3. **Choose a polynomial order per transit** by 10-fold cross-validation on
   the out-of-transit points, K = 0..5.
4. **Fit the linear ephemeris** (7 parameters), then **fit per-transit
   times** (5 shape parameters plus one time per transit), seeded from the
   linear-ephemeris maximum likelihood.

Finite exposures are integrated by the contact rule inside elome's
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
| `lineph --ld=collapsed` | `dP, dtau0, k, beta, T14` | limb darkening integrated out; `q1, q2` drawn from their conditional (see below) |
| either, `--law=hybrid4` | `q1, q2, q3, q4` in place of `q1, q2` | the hybrid4 law; weights `w1..w4` derived (see below) |
| `lineph --law=hybrid4 --ld=collapsed` | `dP, dtau0, k, beta, T14` | hybrid4's weights integrated out; `q1..q4` drawn from their conditional |
| `lineph --law=hybrid4 --oblate` | adds `fc, fs` after `T14` | an oblate planet; `f`, `theta`, `A`, `B` derived (see "Oblate planets") |
| `--mle` (LinEph) | `dP, dtau0, k, beta, T14` [+ `fc, fs`], limb darkening profiled | a maximum-likelihood fit instead of sampling (see "Maximum likelihood") |

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

### The limb-darkening law (`--law`)

`--law=quadratic`, the default, is the quadratic law sampled through
Kipping's (2013) `q1, q2`. `--law=hybrid4` is the four-weight rational law
of LDhybrid v1.0, as defined in [Elómë](https://github.com/davidkipping/elome)
and SquishierPlanet: even powers of μ plus two double poles,
`I/I(1) = 1 - sum_j w_j S_j(mu)`, with the weights on its **curvature
basis** (LDhybrid, Elómë >= 0.17; turin >= 0.1.67). On PHOENIX J-band stars SquishierPlanet
measures its worst-case light-curve error at 1.0 ppm, against 65 ppm for
the quadratic law. Elómë evaluates it on the same tau kernel, with the same
in-kernel exposure integration.

In that basis the simplex `w_j >= 0, sum w_j <= 1` is **exactly** the set of
hybrid4 profiles that are positive, monotonic, concave in μ² and of convex
slope in μ², not just a sufficient region. It is sampled the way LDhybrid
samples it, and the way turin samples the quadratic law: four coordinates
`q1..q4`, each uniform on [0, 1], with `q1 = (1 - I(0))^4` the depth at the
limb and the rest stick-breaking it over the four shapes. A uniform cube
gives a uniform simplex, as Kipping's map gives a uniform quadratic
triangle. Its `q` and `w` are those of LDhybrid's tables, so priors from
them apply directly. The summary and chains carry `q1..q4` (sampled, with
R-hat and ESS) and the physical weights `w1..w4` (derived, like `b`).

- **Works in both fit modes**, `lineph` and `ttv`, with grid-Gibbs, and with
  `--ld=collapsed` for LinEph (below).
- **Needs circular geometry** (`chord` exists for hurin parity, and hurin
  has only the quadratic law).
- A lineage never changes law: resuming with another `--law` is refused.
- Weights fitted with older hybrid4 definitions are not interchangeable
  with these: neither Elómë < 0.15's pole ladder nor turin < 0.1.67's basis
  (weights on `1 - mu^4`, `(1 - mu^2)^2` and the two poles, stick-broken from
  `q`). Old weights convert with `w = w_old @ inv(HYBRID4_BASIS)`; old `q` do
  not convert one to one, so recompute them from `w`. That old simplex was a
  different set of profiles (it included some convex in μ², and missed some
  concave ones), so a posterior that sat against its edges can move. turin
  refuses to resume a hybrid4 run from before 0.1.67, and needs Elómë >= 0.17.
- **Weak grazing transits.** Before 0.1.67, four free weights could make a
  transit V-shaped: on KOI-448.02 (default prior), whose quadratic fit is
  99.7% grazing at `k = 0.16`, sampled hybrid4 split its chains between
  that and a small planet on a central chord with extreme limb darkening
  (`k = 0.027, b = 0.27`, `q1` against its wall) and had not converged
  after three rounds (R-hat 1.26). The curvature basis admits only profiles
  that are monotonic and concave in mu^2, and that second solution is gone:
  the same fit now converges in two rounds (R-hat 1.006), 99.8% grazing,
  with no small-planet draws (one seed, 0.1.76). On a typical target
  (KOI-518.02) the two laws cost the same and both converge in one round.

### Collapsed limb darkening (`--ld=collapsed`)

An opt-in alternative for the LinEph fit, for either law
(`--law=hybrid4 --ld=collapsed`: see "Collapsed hybrid4" below). Instead of
sampling `q1, q2` with everything else, turin integrates the quadratic limb
darkening out of the log-density, samples the other five parameters from that **marginal**
posterior, and then draws `q1, q2` for every kept sample from their exact
**conditional** given the rest. Marginal plus conditional draws of the
integrated block is collapsed Gibbs sampling: the joint posterior is the same
as the default's, `q1` and `q2` included, and every product carries the same
columns.

It works because any quadratic law's light curve is exactly a convex mixture
of the three vertex laws of the Kipping (2013) triangle, so the transit model
is linear in the mixing weights. For each weighting the baseline polynomials
are profiled exactly as always; the integral over the weights is done to
quadrature precision (checked against brute-force integration to 4e-5). The
scheme was proposed and validated by the SquishierPlanet project.

**It is not profiling.** Setting the limb darkening to its best fit at each
sample instead of integrating over it is the obvious shortcut and it is wrong:
measured, it biased `k`, `b` and `T14` by up to 0.38σ and inflated `k`'s width
by 46% on KOI-518.02.

Why it is opt-in rather than the default: it gives the same answer at no
gain in speed for the two-coefficient quadratic law, and costs about 2.2x the
memory per log-density evaluation. Measured against the default on two real
targets (LinEph, 512 chains, same seed, run to convergence; ESS/s is the
minimum bulk ESS over the five parameters both modes sample, over end-to-end
wall time, so collapsed mode's post-hoc draws are counted):

| target | ESS per second, sampled | collapsed | collapsed time per ESS |
|---|---|---|---|
| KOI-518.02 | 157.4 | 156.5 | 1.01x |
| KOI-448.02 (`--nongrazing`) | 53.1 | **37.6** | **1.41x** |

On KOI-518.02 it is break-even; on KOI-448.02 it takes 1.41x the wall time
per effective sample. Across 20 Kepler targets the median is 1.26x the LinEph
wall time (0.77x the ESS/s), with identical posteriors including grazing ones
(`docs/bench/collapsed_ld_battery.md`). We had
expected the opposite there: `q1, q2` sit against the edge of the box
(0.96, 0.93), and taking them out of ChEES looked like it should help. It did
not -- the sampled run had no trapped chains, so there was nothing to fix,
and collapsed mode's dearer evaluations set the cost. Its value is the
dimension reduction itself, which grows with the number of limb-darkening
coefficients (the case that motivated it); for the quadratic law, use it when
you want the limb darkening out of the sampled vector, not for speed.

Restrictions, each refused up front with the reason:

- **LinEph only** (`--modes=lineph`). A shared limb darkening couples every
  epoch, which breaks the per-epoch factorisation grid-Gibbs needs.
- **`--PL=exact`** (`auto` resolves to it without probing). The integral's
  gradient comes from the envelope theorem, which holds only for the exact
  flux-space baseline; under `ratio` the formulas would be wrong.
- **Circular geometry** (elome's `ld_basis` exists only on its tau kernel).

In the summary, the `q1`/`q2` rows leave R-hat and ESS blank: they are exact
conditional draws, not sampler output. The ML-row light curve (lcdata, fold
plot) uses the conditional mode of `q1, q2` at the best draw. A lineage is
either sampled or collapsed, never both: `ld` is a resume guard.

**Validated against the default on real data**, KOI-518.02 and KOI-448.02,
every parameter including `q1, q2`: medians within 0.011σ, widths within
2.5%, and two-sample KS distances judged against a null built from each
run's own chains split in half. One pairing failed that null (KOI-518.02
`q1`, seed 0), and a second seed of both modes showed it was two run-level
fluctuations in opposite directions, not a difference between the modes.
KOI-448.02 is the hard case for the limb darkening -- its expansion point
lies on the edge of the triangle for 54% of draws -- and none of its 460,800
`q1, q2` draws lands on the edge of the box. That run used `--nongrazing`,
so grazing geometries are covered by synthetic tests only. The default
mode's chains are byte-identical to those from before the feature existed
(KOI-518.02, 153,600 draws). Details: `docs/bench/collapsed_ld_acceptance.md`.

### Collapsed hybrid4 (`--law=hybrid4 --ld=collapsed`)

The same idea for hybrid4's four weights, which take five generator light
curves and live on a 4-simplex instead of a triangle. Every hybrid4 law is a
convex mix of the generators `{1, 1 - S_j}`, the vertices of its curvature
simplex (exact, to 9e-18), and uniform
`q1..q4` induce the prior `-5 log sum_g omega_g/F*_g` on the mixing weights.
The baseline is profiled exactly as always, and the profiled likelihood is
integrated over the simplex under its quadratic model about its exact
constrained optimum. Two things differ from the quadratic law's version:

- **The integral is GHK** (Geweke-Hajivassiliou-Keane: sequential truncated
  normals on a fixed 1,024-point scrambled-Sobol rule, so the value is
  deterministic and differentiable). Against a 2^18-point reference at
  posterior draws its error is at most 0.015 nats on KOI-518.02, and 0.03 at
  the 95th percentile on KOI-448.02's two-solution posterior. The rule's
  layout is frozen at the fit's MAP, and the layout and points are stored
  with the fit, so a resume or `--replot` rebuilds the identical target.
- **The algebra runs on the triangular factor of the curvature**, never the
  curvature itself, which reaches a condition number of 7e9 on KOI-448.02
  (beyond float32); its factor's is at most 8.5e4. A small fixed ridge keeps
  the curvature from vanishing where the data cannot see the limb
  darkening at all (no transit, k -> 0), which otherwise made a false peak.
- **The post-hoc draws of `q1..q4`** use a Metropolis chain whose moves are
  exact for the quadratic model truncated to the simplex: exact HMC with
  analytic reflections (Pakman & Paninski 2014), alternating with exact
  draws along the simplex's edges, 128 steps per draw.

Validated on the user's five-target Kepler suite (KOI-4016.01, 4087.01,
1430.03, 1361.01, 2650.01; SNR 11.5-52) against sampled hybrid4, with a
second seed of both modes on three targets
(`docs/bench/collapsed_hybrid_acceptance.md`):

- every parameter, `q1..q4` and `w1..w4` included, within 0.016 sd, with no
  shift beyond 3 Monte Carlo standard errors that a second seed did not
  remove;
- 16-84% widths within 1.5%;
- the two modes differ no more than two seeds of one mode do;
- no `q` draw on the box edge.

That was first shown on hybrid4's pre-0.1.67 basis and re-measured on the
curvature basis in 0.1.76, with the same result. It took a second pass. The
first found three defects, all now fixed:

- the conditional draws were biased by up to 0.41 sd where the limb
  darkening is weakly constrained;
- chains fell into a false peak at k = 0;
- a few draws landed exactly on the box edge.

The collapsed value matches the true marginal to about 0.01 nats near
these posteriors (sd over posterior draws 0.005-0.06, the 1,024-point GHK
rule's own error).

On KOI-2650.01, collapsed mode reproducibly carries 0.2-0.4% of its draws
in a far grazing tail (k > 0.06) that sampled mode's chains barely visit. Collapsed mode's density there matches the true marginal, so the
likely reading is that sampled mode under-explores it.

**It is slower: 5.7-8.2x the wall time of sampled hybrid4 (median 6.7x),
0.09-0.27x the effective samples per second (median 0.17x).** The GHK
integral is a fixed per-chain cost, so the gap is widest on small datasets.
It has widened since 0.1.64 (median 4.4x) because sampled mode got about
twice as fast, not because collapsed mode slowed. Sampled hybrid4
stays the default. Collapsed mode is for taking the limb darkening out of
the sampled vector, or for a target whose sampled fit mixes poorly in `k`.
Products, restrictions and the resume guard are as for the quadratic law,
with `q1..q4` and derived `w1..w4`.

### Oblate planets (`--oblate`)

A rotating planet is flattened, and its silhouette on the star is an ellipse
rather than a circle. `--oblate` fits that ellipse: its flattening
`f = 1 - B/A` (A and B the semi-axes) and the sky angle `theta` of its long
axis, alongside the usual geometry. The light curves come from Elómë's
oblate kernel, on the GPU with the same in-kernel exposure integration.
The idea and its design checks come from SquishierPlanet.

```bash
turin --KOI-1.01 --law=hybrid4 --oblate --modes=lineph
```

**What the parameters mean:**

- `k` is the **area-equivalent radius**, `sqrt(A B)`: the round planet that
  blocks the same light, so it stays comparable with a round fit's `k`.
- `T14` keeps the round relation to `a/R*`, so it is the duration of that
  area-equivalent circle, not the ellipse's. The derived `b` and
  `log10_rho` follow from the same relation.
- `f` and `theta` are sampled as `fc, fs = sqrt(f) (cos 2theta,
  sin 2theta)`, which is uniform in `f` on [0, 0.6] and in `theta`. `theta`
  covers a half-turn, so it is reported in degrees on [-90, 90).

**Requirements:**

- `--law=hybrid4`. A limb-darkening law with too little freedom pushes its
  misfit into the shape. In SquishierPlanet's tests a two-parameter law
  made a round planet at b = 0.3 look oblate (f = 0.082 +/- 0.016) and
  moved a true `theta` of 30 degrees to 63, while hybrid4 stayed within 0.8
  sigma. That is SquishierPlanet's measurement, not one turin repeats.
- LinEph only (`--modes=lineph`), circular geometry.
- `--ld=collapsed` is the default under `--oblate`; `--ld=sampled` is still
  available. On a synthetic oblate planet, collapsed converged in one round
  (517 s), while sampled mode had one of 512 chains stuck in the
  limb-darkening coordinates and had not converged after 100 minutes. That
  is one seed per mode.
- A lineage never switches: resuming with or without `--oblate` against a
  run made the other way is refused.

**The signal is in ingress and egress.** An ellipse and a circle of the same
area block the same light once the planet is fully on the disc. What
differs is how the planet crosses the limb. So oblateness needs data that
resolve ingress: short cadence, many transits, or a large planet. Kepler's
29.4-minute long cadence smears most of it. For scale: on a synthetic
K-dwarf transit with SquishierPlanet's PHOENIX limb darkening (k = 0.15,
a/R* = 20, b = 0.7, 60 s cadence, three transits at 17 ppm, the
information of one at 10 ppm), turin measured f = 0.1006 +/- 0.0029 and
theta = 29.7 +/- 1.5 degrees against the injected 0.1 and 30, every other
parameter within 0.6 sigma.

**Upper limits.** A round planet's `f` piles up at 0, so the run log
reports f's median, its 16-84% range and a 95% upper limit. The limit
depends on the prior: uniform in `f` up to 0.6.

**A false solution, and `--rhostar`.** SquishierPlanet found that a round
planet at b = 0.3 is nearly degenerate with an oblate planet aligned with
its orbit (theta near 0) on a central chord (b near 0), which fits only
2-3 log-likelihood units worse. Being broader, that family held about 20%
of the posterior in their tests. turin always logs the posterior share of
that family (b < 0.1, |theta| < 15 degrees, f > 0.05), and warns above 5%.
The false solution needs a different a/R*, so a prior on the stellar
density suppresses it. `--rhostar=MEAN,SIGMA` is a Gaussian prior on
log10(rho*/kg m^-3), the unit of the `logrho` product (the Sun is 3.149).
SquishierPlanet's measurement of what it buys, which turin has not
repeated:

| a/R* known to | rho* known to | `--rhostar` sigma | false-solution share |
|---|---|---|---|
| no prior | | | 0.20-0.23 |
| 5% | ~15% | ~0.065 | 0.18 |
| 3% | ~9% | ~0.039 | 0.09 |
| 1% | ~3% | ~0.013 | ~0 |

`--rhostar` works for any fit, oblate or not, and a lineage keeps whatever
prior it started with.

**How the fit is started.** As for every fit, the best point comes from
anvil's optimiser (`anvil.optimize`) on 128 starts. Chains start spread over
a Laplace approximation at that point, never exactly at f = 0, where `f`
has no gradient.

**Limits:**

- Points where the planet is too large for its flattening
  (k > (1 - f)^1.5) are outside the model and get zero probability.
- For grazing oblate planets the light curve is exact but the (b, k)
  prior is the round planet's. It caps `b` at `1 + k` and puts its grazing
  taper on the band `1 - k < b < 1 + k`, while an ellipse grazes out to
  `1 + h`, with `h` its half-width across the chord (between `k` and the
  semi-major axis `A = k/sqrt(1 - f)`, depending on `theta`). So the barely
  grazing sliver `1 + k < b < 1 + h`, at most `k (1/sqrt(1 - f) - 1)` wide
  (5% of `k` at f = 0.1, 20% at 0.3), has zero prior, and the grazing odds
  are slightly off. This is deliberate: oblate fits use the same prior as
  spherical ones, so the two stay directly comparable. It matters only when
  the data put `b` at the grazing edge. turin's oblate light-curve tests
  cover grazing geometries (b = 0.95, 1.0, 1.05) as well as b = 0.3 and 0.7.
- turin's oblate code has been checked against Elómë's own frontend (to
  1e-13) and SquishierPlanet's independent light curves (to 1e-12). It
  has been tested on synthetic transits, not yet on real targets.
- An oblate evaluation costs more than a round one: at 512 chains and 860
  points, 224 ms collapsed and 164 ms sampled, against 81 ms for collapsed
  hybrid4 on a round planet. That was measured before Elómë 0.16.3 made its
  oblate kernels 1.2-1.6x faster.

### Maximum likelihood (`--mle`)

`--mle` replaces sampling with a maximum-likelihood fit: the LinEph model,
round or `--oblate`, and then (not under `--oblate`) the TTV model, when
the transits are strong enough to time. It takes seconds rather than
minutes:

```bash
turin --KOI-518.02 --mle                 # LinEph, then TTV
turin --KOI-518.02 --mle --modes=lineph  # LinEph only
turin --KOI-1361.01 --mle --law=hybrid4 --oblate
```

**What is maximised.** The likelihood alone. turin's priors are off: the
(b, k) prior's area and grazing terms, and `--rhostar` (refused under
`--mle`). The hard limits stay: each parameter's box, the impact
parameter's bound under `--bprior` (`b <= 1 + k`, or `1 - k` with
`--nongrazing`), and for an oblate planet the domain where it fits inside
the star.

**The limb darkening is profiled.** Every limb-darkening law turin fits is
a weighted mix of a few fixed light curves, the same structure the
collapsed modes use. So at each geometry the best limb darkening is a small
exact optimisation over the allowed weights, and the optimiser only has to
search the geometry. For a maximum this gives exactly the joint best fit.
It also reaches a limb darkening on the edge of the allowed region
directly, which is common: hybrid4's best weights sat on a face for 94-100%
of the geometries tested. (Profiling would be biased for *sampling*, which
is why `--ld=collapsed` integrates instead.)

**Where the search starts.** 128 starts, half in a small spread around the
catalogue ephemeris and half with the impact parameter spread evenly over
its whole range. At low SNR b trades off against the limb darkening along
a long, nearly flat ridge that can hold more than one peak. On KOI-4016.01
and KOI-1430.03 the best fit is at b ~ 0.88-0.91 with weak limb darkening;
the ephemeris spread alone found only a central chord with limb darkening
on a corner, 0.33 and 0.10 below it in ln L (0.1.74). Checked by evaluating
the likelihood at every draw of a LinEph MCMC run on each of the five
suite targets: the MLE is above all of them.

**The best fit, nothing else.** There are no error bars: for those,
sample. Where the likelihood is flat the best-fit value of that parameter
is arbitrary. That is common for b near 0, a poorly measured limb
darkening, and f at Kepler long cadence. KOI-1361.01's oblate fit put f at
0.600, the edge of its range, for 0.06 in ln L, and the log and products
note an f at its edge.

**Oblate planets: the comparison is the result.** `--mle --oblate` first
fits the round planet, then starts every oblate search from that best fit
at f ~ 0 (the geometry slightly spread, the angle random), in (fc, fs)
as the sampler does, and reports ΔlnL = lnL(oblate) -
lnL(round), ΔBIC and ΔAIC (negative favours oblate). The models are nested,
so ΔlnL >= 0. No p-value is given: at f = 0 the angle is undefined and f
sits on its boundary, so the usual chi-squared calibration does not apply.
For the evidence, calibrate ΔlnL with injections, or sample. If the oblate
optimum is an aligned planet on a central chord, the family that mimics a
round planet at larger b, the log warns.

Starting from the round fit was measured against starting around the
catalogue ephemeris. On six synthetic oblate and round planets (PHOENIX
limb darkening, 1-minute cadence) both reach the same best fit, and the
round start is 20-40% faster. On the Kepler targets below it is level or
better on three, and up to 0.27 short of the other's luckiest start on
two, where the best fits lie on a ridge of near-equal likelihood. It can
never end below the round fit, which the ephemeris start once did.

On the five Kepler long-cadence targets of turin's test suite, every
oblate fit gained 0.001-1.4 in ln L for its two extra parameters, BIC
preferred round by 10.5-16, and f's best value drifted to the edge of its
range on four. At that cadence the likelihood barely tells f apart. A fit
takes about 5 s round and 20-60 s with `--oblate` (both fits), data loading
included.

**TTV.** After LinEph, the expected SNR of each single transit is
computed at the LinEph best fit. If the median, over epochs with data in
transit, is below `--SNRgate` (default 5) the run stops there: a weak
transit's best time is little more than noise, and it has no error bar to
say so. (`--SNRgate` is stricter than sampling's `--TTVsnr`, 3, for that
reason; 0 always fits.)

Otherwise the TTV fit starts from the LinEph best fit, with each epoch at
its linear-ephemeris time, and alternates two steps:

- **A timing step.** Given the shape and limb darkening, each epoch's
  likelihood depends on its own time alone, so every epoch's time is put at
  its best over its whole timing range on a grid, refined, and kept only if
  it improves. This crosses between separated timing peaks, which a local
  optimiser cannot: started inside one peak, it stays there.
- **A joint polish** of everything, the limb darkening profiled.

It stops when a timing step moves no epoch, which certifies that no
transit time could improve on its own. TTV mode holds the period at the
catalogue value (as hurin and the sampler do), so it nests LinEph up to the
period's tiny effect on the transit shape. The log and products compare
the two fits: ΔlnL, ΔBIC and ΔAIC for the extra epoch times (no p-value:
timing likelihoods have several peaks). An epoch with no data in transit is
noted: its time is arbitrary. Tested so far on synthetic data, where it
recovers injected TTVs; the check against TTV posteriors of real targets is
pending.

**Products** (`<T>_lineph_mle_*` and `<T>_ttv_mle_*`, so they never mix with a sampled
lineage's files; `--fresh --mle` clears only these, and a sampled run's
`--fresh` leaves them):

```
<T>_lineph_mle_fit.csv       Parameter, MLE; then b, log10_rho
                             (w1..w4, f, theta, A, B as they apply), lnL,
                             n_points, n_params (baseline coefficients
                             included), BIC, AIC
<T>_lineph_mle_lcdata.csv    time, flux, flux_err, model_flux
<T>_lineph_mle_fold.pdf      phase-folded transit at the best fit
<T>_lineph_mle_round.csv     --oblate: the round fit, same columns
<T>_lineph_mle_compare.csv   --oblate: lnL of each, dlnL, dBIC, dAIC
<T>_ttv_mle_fit.csv          as lineph_mle_fit, the times as dtau_<n>
                             (absolute mid-transit times)
<T>_ttv_mle_times.csv        hurin's times columns; the errors NaN, the O-C
                             against the LinEph best fit
<T>_ttv_mle_lcdata.csv       time, flux, flux_err, model_flux
<T>_ttv_mle_fold.pdf         folded on the fitted times
<T>_ttv_mle_oc.pdf           O-C against the LinEph best fit, no error bars
<T>_ttv_mle_compare.csv      lnL of each, dlnL, n_extra, dBIC, dAIC
```

### Weak transits and grid-Gibbs

A transit observed at low signal-to-noise, or with a data gap across it, has a
timing posterior with several separated bumps, often hours apart. ChEES-HMC
does not move chains between such bumps. Left alone, each bump is weighted by
however many chains happen to settle there, and that transit's R-hat never
passes however long the fit runs.

So by default the TTV fit interleaves an exact Gibbs step. Every 25 draws,
each chain redraws every transit time from its conditional posterior, given
that chain's current shape parameters. The conditional is evaluated on a grid
spanning the whole timing prior, so a chain can land in any bump, and a
Metropolis-Hastings correction keeps the step exact. It is possible because,
for fixed shape, each transit's likelihood depends on its own time alone.
That also makes one sweep cost about one extra log-density evaluation per
grid point however many transits there are.

On KOI-4848.01 and KOI-5897.01, against exact timing marginals computed by
brute force, this cut a weak transit's distance from the true posterior by
5-8x (e.g. 0.162 to 0.019 in total variation).

The sweep runs every 25 draws rather than every 100. On KOI-7776.01 and
KOI-5162.01, whose weakest transit has two timing modes, sweeping every 100
left both unconverged at the draw cap, while every 25 converged both about
35% sooner. On a target with no second mode it costs ~10% and still sharpens
the timing posteriors. `--gibbsgrid=off` gives independent ChEES chains only. The setting is a resume guard, so a lineage
cannot pool chains drawn both ways.

## Options

```
--fresh                      ignore existing resume state for this lineage
--replot                     redraw a finished fit's figures and ttv_times.csv
                             from its saved products, without sampling (for
                             figures made by an older turin; summary, chains
                             and resume state are left untouched; refused
                             unless this turin reproduces the fit's best
                             log-density)
--extend1 / --extend2        extend the LinEph / TTV fit by one round
--modes=lineph,ttv           which fits to run
--tag=NAME                   namespace a run into its own resume lineage

--chains=N                   sampling chains (default 512, raised to 4*dim)
--sampler=chees|ensemble     gradient-based (default) or gradient-free
--warmup=N                   warmup iterations (default 800)
--samples=N --max-samples=N --leapfrog=N --seed=N
--gibbsgrid=on|off           TTV fits: move chains between timing modes with
                             an exact grid-Gibbs step (default on), see below

--bprior=transiting|nongrazing|box      (b, k) prior, see above; --nongrazing
                             is an alias for --bprior=nongrazing
--TTVmax=MINUTES             declared TTV amplitude; sets the timing priors
--TTVsnr=X                   skip the TTV fit when the median expected SNR of a
                             single transit (LinEph best fit, epochs with data
                             in transit) is below X; default 3, 0 always fits.
                             A skip writes <target>_ttv_skipped.csv (reason and
                             each epoch's SNR) and exits 0 on LinEph's verdict
--PL=auto|exact|hybrid|ratio how the baseline coefficients are solved
                             (default auto: measured per target)
--geometry=circular|chord    true circular orbit (default) or hurin's chord
--ld=sampled|collapsed       sample q1, q2 (default), or integrate the limb
                             darkening out (LinEph only), see above
--law=quadratic|hybrid4      limb-darkening law (default quadratic), see above;
                             with --ld=collapsed for LinEph too
--oblate                     an oblate planet (f, theta); needs --law=hybrid4
                             and --modes=lineph; implies --ld=collapsed
--rhostar=MEAN,SIGMA         Gaussian prior on log10(rho*/kg m^-3), any fit
--mle                        maximum likelihood instead of sampling (LinEph,
                             round or --oblate, then TTV), see above
--SNRgate=X                  --mle: fit TTVs only above this median
                             single-transit SNR (default 5)

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
                              (and w1..w4 before loglike under --law=hybrid4;
                              then f, theta, A, B under --oblate)
<T>_<mode>_logrho.csv         the stellar-density posterior
<T>_<mode>_lcdata.csv         time, flux, flux_err, model_flux
<T>_<mode>_fold.pdf           phase-folded transit, baseline divided out
<T>_<mode>_corner.pdf         corner plot with 1/1.5/2-sigma contours: every
                              parameter and every transit time; past 14
                              transits, the first and last 7 with a "..."
                              row and column between them
<T>_<mode>_resume.pkl         resume state
<T>_ttv_times.csv             epoch, tmid, tmid_err, O-C, SNR, npts, chi2
                              (SNR: sqrt(2 dlnL) per transit against a
                              no-transit fit with its own baseline)
<T>_ttv_oc.pdf                O-C against a linear ephemeris fitted to the
                              measured times, weighted by their errors; its
                              chi2 measures how non-linear the timings are
```

Values quoted on the figures follow one rounding rule: both errors are rounded
to two significant figures, the more precise of their decimal places is
taken, and the median and both errors are quoted to it (628.208 +0.017
-0.120; 12350 +1230 -990).

Every product records turin's version and the exact command that made it, on
line 2 of each CSV. While a run is going that line also says `in progress:
round N`, and a fit that stops at the draw cap unconverged is stamped
`UNCONVERGED: worst R-hat ... at the 16384 draws/chain cap`. The exit code says
the same thing for scripts: 0 when every fit converged, 3 when the run
finished but a fit did not, anything else for an error.

turin downloads only the target's own star: a MAST name search can also return
a neighbouring star's light curves (KOI-7592.01 returns two stars), so the
results are matched to the host's KIC or TIC from the Exoplanet Archive, and a
stitched light curve that still repeats timestamps is refused.

**Large runs are subsampled for export, not for inference.** R-hat and ESS are
computed on every draw. Everything else in the products is computed from a
systematic subsample of at most 10^6 rows (fewer for high-dimensional fits,
so that rows x parameters stays under 1.6 x 10^7). That covers the summary
percentiles, the chains and logrho files, the plots and the float32
certificate, and every chain stays equally represented. When thinning
applies, line 2 of the chains CSV says so, e.g.
`... | thinned 1/9: 932,864 of 8,396,800 draws`. The chains tarball and PDFs
are written after the first round, then at most every 20 minutes, and always
at the end; the small products and the resume state are written every round.

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
polynomial (Agol et al.) formulation elome implements, because the
quadratic law is exactly that law's `N=2` case
(`I(mu)/I0 = 1 - sum u_n (1-mu)^n`); the Green's basis elome uses
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
| float64 model vs elome's verified frontend | 1e-14 | 3.3e-16 |
| float32 model vs the same | 5e-7 | 1.6e-7 |
| exposure integration vs elome's contact rule | 1e-14 | <1e-14 |
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

Almost all of what is left is elome's kernel; turin's own profile solve,
residual and priors are 0.7 ms of the 9.1. Two things got it there, and both
are turin's responsibility rather than elome's:

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
  flagged chains. Chains anvil judges lost at the end of a run (checked on
  an eager re-evaluation of the target too, which can disagree with the
  compiled one at a float32 edge) are flagged as well. No chain is ever
  silently dropped.

## Not in this version

The Flask web frontend, profile-likelihood transit times (hurin's TTV-PL),
eccentric orbits, and Gaussian-process noise. The likelihood keeps a noise-model
seam so [anvil-gp](https://github.com/davidkipping/anvil-gp) can be dropped in:
its `GPLogLike` takes the transit-times-baseline product as a mean function
with no adapter.

## Citation

If turin contributes to a publication, please cite it along with elome
and anvil.

## License

MIT; see [LICENSE](LICENSE).
