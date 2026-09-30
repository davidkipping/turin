# Where turin's model differs from hurin's, and why

Measured 2026-09-28/29 on an M-series Mac (12 cores) with MLX 0.32.3,
MetalPlanet 0.6.1, and hurin in its own conda env. The per-difference
measurements are reproduced by `tests/test_model.py`; the real-data sections
record whole fits and are reproduced by re-running the commands they quote.

turin is not bit-compatible with hurin, and all the differences are
switchable, so any hurin result can be reproduced for comparison.

**Start with the KOI-518.02 section**, immediately below. It is the primary
comparison: current versions of both packages, both converged, identical
input, and a target on which all three model differences are negligible — so
it measures the samplers, and it is where the headline agreement (≤0.03σ on
every parameter) and the headline speedup (11.1x end to end) come from.

**One of the three differences has since been fixed upstream.** turin's port
surfaced a limb-darkening bug in hurin, and hurin 0.1.68 corrected it, so
against current hurin only the orbit model and the profile form differ. The
measurements for the limb-darkening case are kept below because they explain
hurin results published before 0.1.68 — including the KOI-448.02 comparison in
this document, which was run against hurin 0.1.62.

## The primary comparison: KOI-518.02

**Read this section first.** It is the only fully matched head-to-head — both
packages at current versions, both fully converged, both from the same cached
light curve — and so it is the one to quote. The two older comparisons below
are kept because each says something the primary one cannot: KOI-448.02
exercises the limb-darkening difference on a grazing system (against a hurin
that still carried the bug), and KOI-5162.01 is hurin's documented failure
case.

### Why this target is the right primary example

Counter-intuitively, the comparison is most informative where the *model*
differences vanish, because then it isolates the sampler. KOI-518.02 does
that:

- `T14/P = 0.0049`, so the straight-chord approximation (difference 2) is
  accurate far below the float32 floor;
- the transit is **513 ppm** deep, so the ratio-space profile's `O(depth)`
  error (the profile-form difference) is invisible;
- hurin is 0.1.68, so the limb-darkening map (difference 1) is identical.

Every remaining difference is therefore the sampler and the arithmetic, and
turin ran at its **defaults** — circular orbit, exact profile, `--PL=auto`
(which measured the three solves and chose `exact`) — not in compatibility
mode. Agreement here is the strong statement; compatibility mode is not
needed to obtain it.

### Setup

    hurin 0.1.68   pyhurin.py --KOI-518.02 --chains=8 --fresh
    turin 0.1.12   turin.cli --KOI-518.02 --fresh          (512 chains)

One machine (12 cores, 8 performance), runs **sequential** so neither
contended for CPU, and both reading a byte-identical cached light curve
(1,821,274 bytes in each package's own cache). 27 occupied epochs, Kepler
long cadence, `P = 44.0004 d`, duration 5.11 h.

`--chains=8` rather than hurin's default of 2. At 2 chains hurin needed four
rounds and was still at bulk ESS 124 when its third extension finished, so 8
chains is what makes hurin converge in one round and the comparison fair
rather than a straw man. Both samplers vectorize their chains
(`chain_method="vectorized"`; MLX on the GPU), so this compares each one used
as designed. The counts cannot be matched meaningfully in both directions:
512 NUTS chains on CPU is not something anyone would run, and 8 ChEES chains
defeats its purpose.

An independent check that the *conditioning* matches before any sampling: both
packages' cross-validation chose **identical** Legendre orders across all 27
epochs — K=0: 10, K=1: 11, K=2: 4, K=3: 1, K=4: 1.

### The posteriors agree

Both packages converged on their first round, with no warnings from either.
`shift` is the difference in medians over the two uncertainties in quadrature.

#### LinEph

| parameter | hurin | turin | shift | width turin/hurin |
|---|---|---|---|---|
| `P` | 44.000302 ± 1.37e-04 | 44.000300 ± 1.42e-04 | -0.01σ | 1.04 |
| `tau0` | 870.768302 ± 1.29e-03 | 870.768281 ± 1.28e-03 | -0.01σ | 0.99 |
| `k` | 0.022643 ± 1.10e-03 | 0.022667 ± 1.21e-03 | +0.02σ | 1.10 |
| `b` | 0.4275 ± 0.244 | 0.4339 ± 0.249 | +0.02σ | 1.02 |
| `T14` | 0.213874 ± 4.94e-03 | 0.213904 ± 5.49e-03 | +0.00σ | 1.11 |
| `q1` | 0.4070 ± 0.233 | 0.4009 ± 0.229 | -0.02σ | 0.98 |
| `q2` | 0.3667 ± 0.265 | 0.3653 ± 0.266 | -0.00σ | 1.00 |
| `log10_rho` | 3.3445 ± 0.222 | 3.3418 ± 0.247 | -0.01σ | 1.11 |

#### TTV (32 parameters: 5 shape + 27 transit times)

| parameter | hurin | turin | shift | width turin/hurin |
|---|---|---|---|---|
| `k` | 0.022851 ± 1.56e-03 | 0.022844 ± 1.58e-03 | -0.00σ | 1.01 |
| `b` | 0.5010 ± 0.264 | 0.4898 ± 0.265 | -0.03σ | 1.01 |
| `T14` | 0.212884 ± 7.71e-03 | 0.212765 ± 7.87e-03 | -0.01σ | 1.02 |
| `q1` | 0.4255 ± 0.238 | 0.4167 ± 0.241 | -0.03σ | 1.01 |
| `q2` | 0.3812 ± 0.264 | 0.3791 ± 0.267 | -0.01σ | 1.01 |
| `log10_rho` | 3.2985 ± 0.309 | 3.3103 ± 0.310 | +0.03σ | 1.00 |

And the 27 transit times, the product this pipeline exists to make: median
shift **0.01σ**, worst **0.02σ**, with uncertainties agreeing to 0.96-1.04x.
Nothing is shared between the two implementations — MLX against JAX, ChEES-HMC
against NUTS, an unrolled Cholesky against `jnp.linalg.solve`, 512 chains
against 8 — so agreement at this level is evidence for both.

### The benchmark

| | hurin | turin | ratio |
|---|---|---|---|
| LinEph sampling | 2,042.4 s | 244.2 s | 8.4x |
| TTV sampling | 3,981.7 s | 249.1 s | 16.0x |
| **total wall clock** | **6,037 s** (100.6 min) | **544 s** (9.1 min) | **11.1x** |
| rounds to converge | 1 + 1 | 1 + 1 | — |
| worst R-hat | 1.013 (`k`) | 1.006 (`beta`) | — |
| peak resident memory | 4.8 GB | 17.3 GB | 0.28x |

Because both converged in one round, this is *time to a converged answer*,
not time for a fixed amount of work.

Sampling efficiency, as minimum bulk ESS over the parameters both packages
report, divided by that mode's own sampling time:

| mode | hurin min ESS | turin min ESS | hurin ESS/s | turin ESS/s | ratio |
|---|---|---|---|---|---|
| LinEph | 453 (`k`) | 79,145 | 0.22 | 324 | **1,460x** |
| TTV | 1,519 (`k`) | 70,942 | 0.38 | 285 | **750x** |

The ESS ratio is far larger than the wall-clock ratio, which is the real
result: turin is not merely finishing sooner, it is buying two to three orders
of magnitude more independent information per second. Most of that is the
jump from 8 chains to 512, which is what the GPU makes affordable and is the
reason anvil exists. turin also logged **zero divergences** in both modes, and
its float32 certificate put the reweighting bias at 0.03x the Monte Carlo
standard error at ESS = 10,000, with 99.9996% of the ESS retained.

The summary CSVs behind every number above are in `docs/bench/`, with the
comparison script and the commands to reproduce both fits.

### Caveats

- **Memory is turin's cost**: 17.3 GB peak against hurin's 4.8 GB, from 512
  chains plus the fused kernel's backward grids. On a 16 GB machine this
  target needs `--chains` lowered. `likelihood.epoch_block_size` bounds the
  per-block transient but not the chain axis.
- **Single run each.** ChEES wall time carries roughly 10% seed-to-seed
  noise (measured on KOI-5162.01), so 11.1x is not pinned tighter than that.
- **The chain counts are not matched**, deliberately, as argued above. A
  chains-matched comparison would be a different experiment: it would measure
  the two kernels, not the two packages.

## 1. The limb-darkening map — FIXED UPSTREAM in hurin 0.1.68

**Status: resolved.** turin's port surfaced this, and hurin adopted the
correct map in commit `bac2362`, "Fix Kipping (2013) limb-darkening map: both
factors of 2 were missing". Against hurin >= 0.1.68 this difference no longer
exists; `--ld=hurin` now reproduces *pre-0.1.68* hurin only. The measurements
below are kept because they explain the older published results, including
the KOI-448.02 comparison further down, which was made against hurin 0.1.62.

The original finding: `hurin/transit_fit.py:262` said
"Converts q1,q2 to u1,u2 via Kipping (2013) reparameterization" and then
computes

    u1 = sqrt(q1) * q2            u2 = sqrt(q1) * (1 - q2)

where Kipping (2013) Eqs. 15-16 — and `metalplanet.ld.q_to_u` — are

    u1 = 2 sqrt(q1) q2            u2 = sqrt(q1) (1 - 2 q2)

Both factors of two are missing. Consequences:

- hurin's `q2` is **exactly twice** Kipping's, so a reported `q2` is not the
  quantity the citation defines. `q1` is unaffected: both maps give
  `u1 + u2 = sqrt(q1)`.
- More seriously, hurin's unit square maps onto **only the `u1, u2 >= 0`
  corner** of the physically-allowed triangle. Kipping's `q2 > 0.5` gives
  `u2 < 0`, which hurin cannot represent at all. Negative `u2` is common in
  quadratic-law fits to real stars, so this is a prior that silently
  excludes part of the answer rather than a cosmetic relabelling.
- The whole point of Kipping (2013) is that the unit square maps *onto* the
  physical triangle, uniformly. hurin's variant is uniform over a
  sub-triangle, so its limb-darkening prior is not the intended one.

At `(q1, q2) = (0.30, 0.45)`, `k = 0.08`, `b = 0.35` the two maps differ by
**2.4e-4 in flux** — about 3% of a 7.2e-3 transit depth, and three orders of
magnitude above the fp32 noise floor. This is the largest of the three
differences by far.

turin defaults to `ld_map="kipping"`; `ld_map="hurin"` reproduces pre-0.1.68
hurin. Any hurin result produced before 0.1.68 carries the truncated prior,
which matters most for grazing systems (see the KOI-448.02 section below).

## 2. The orbit: jaxoplanet's `TransitOrbit` is a straight chord, not a circle

`jaxoplanet/orbits/transit.py` builds a constant-speed straight-line
crossing, not a Keplerian orbit:

    speed = 2 sqrt((1+k)^2 - b^2) / T14
    x = speed * dt,   y = b (constant),   z = sqrt(x^2 + b^2)

MetalPlanet's `separation_circular` is the true circular projection,

    z^2 = a^2 sin^2(phi) + b^2 cos^2(phi),   phi = 2 pi tau / P

The two agree to O((T14/P)^2). At `T14/P = 0.019` they differ by **4.6e-6 in
flux** — small, but still ~30x the fp32 floor, so not negligible.

turin defaults to `geometry="circular"` for two reasons. It is the physically
correct model, and it makes the fit self-consistent with its own reporting:
turin derives `log10_rho` from `(T14, P, k, b)` through the Seager &
Mallen-Ornelas Eq. 8 inversion, and under `circular` that is exactly the
relation the likelihood used. Under hurin's chord model the density is
derived through a relation the fit did not use.

`geometry="chord"` reproduces hurin.

## 3. Precision: hurin runs jaxoplanet in float32

hurin never enables JAX x64, so its forward model is float32 throughout —
about 2e-7 in flux, and a 5-10 s floor on absolute timing at BKJD
magnitudes, which is why hurin (and turin) work in epoch-centred
coordinates.

turin is float32 on the GPU for the same reason, at MetalPlanet's measured
1.6e-7, with an exact float64 CPU path available for
`validate_precision`, `certify` and all reporting. So turin is no worse in
its hot loop and strictly better where it matters for adjudication.

## Secondary comparison: KOI-448.02, where the model differences bite

Kept for what KOI-518.02 cannot show. This target is **grazing**, so the
limb-darkening correction changes the answer, and the comparison is against
hurin **0.1.62** — before the fix — which is why it is not the primary
example. Read it as a study of the limb-darkening difference, not as a
head-to-head of current versions.

Both packages fitted the same cached Kepler light curve (30 transits, 2,552
points, `--nongrazing`, per-epoch cross-validated Legendre orders). turin ran
256 ChEES chains; hurin's published run used 8 NUTS chains.

| | hurin (jaxoplanet + NUTS) | turin, hurin-compat | turin, defaults |
|---|---|---|---|
| P (d) | 43.576133 ± 2.89e-4 | 43.576139 ± 2.92e-4 | 43.576136 ± 2.91e-4 |
| tau0 | 848.410461 ± 2.39e-3 | 848.410448 ± 2.37e-3 | 848.410060 ± 2.35e-3 |
| k | 0.048176 ± 2.14e-3 | 0.048074 ± 2.13e-3 | **0.060349 ± 7.02e-3** |
| b | 0.949226 ± 3.24e-3 | 0.949229 ± 3.33e-3 | **0.932235 ± 9.04e-2** |
| T14 (d) | 0.330667 ± 9.89e-3 | 0.330605 ± 9.75e-3 | **0.346631 ± 1.68e-2** |
| q1 | 0.930606 ± 9.86e-2 | 0.929672 ± 9.81e-2 | 0.962508 ± 5.67e-2 |
| q2 | 0.631963 ± 2.86e-1 | 0.617320 ± 2.87e-1 | 0.925980 ± 1.87e-1 |
| log10 rho | 1.812024 ± 4.23e-2 | 1.812622 ± 4.28e-2 | 1.911142 ± 1.66e-1 |

**Two conclusions, and the second is the interesting one.**

**1. The stack is validated.** In compatibility mode for *that* hurin version
(`--geometry=chord --ld=hurin --profile=ratio`) turin reproduces hurin on
every parameter to **0.04 sigma or better**, with widths agreeing to 1-3%.
Nothing is shared between the two implementations: MLX against JAX, ChEES-HMC
against NUTS, an unrolled Cholesky against `jnp.linalg.solve`, 256 chains
against 8. Agreement at that level is strong evidence for both.

**2. Fixing the limb-darkening map widens grazing-transit posteriors, because
the truncated prior was quietly regularizing them.** This now applies to hurin
too, from 0.1.68 on: the effect below is a property of the correction, not of
turin. KOI-448.02 is
grazing (b = 0.949 against 1 - k = 0.952), and the corrected map lets the fit
reach limb-darkening coefficients hurin could not represent at all:

| fit | u1 | u2 | inside the physical triangle |
|---|---|---|---|
| hurin published | +0.610 | +0.355 | yes |
| turin, hurin-compat | +0.595 | +0.369 | yes |
| turin, defaults | **+1.817** | **-0.836** | yes (`u1 + 2 u2 = +0.145`) |

The default fit prefers a strongly limb-darkened profile that brightens again
toward the limb. That is inside Kipping's triangle -- barely -- so the prior
permits it, and once it is reachable it trades against the grazing geometry
and inflates k, b and T14 together. hurin's map, restricted to `u2 >= 0`,
excluded that family and so reported tighter shape parameters than the data
alone support.

This is worth knowing before quoting either number. The corrected map is the
published one and its prior is uniform over the physically-allowed region, so
it is the honest default. But a fit that lands near the corner of that region
is telling you the light curve does not constrain limb darkening, and for a
grazing system that uncertainty propagates straight into the radius ratio. A
physically-motivated limb-darkening prior (from model atmospheres for the
star's parameters) would be the principled fix, and turin does not implement
one yet.

The ephemeris is unaffected either way: P and tau0 agree across all three
columns, which is the part of the fit that does not care about the limb.

## Secondary comparison: KOI-5162.01, the trapped-mode target

The target hurin documents as its headline failure: a 628-day period with
only **three** observed transits, where hurin's chains settled 35.6 log-units
below the global optimum with R-hat <= 1.01. turin ran 256 ChEES chains,
LinEph then TTV, with template-sweep timing seeds.

| epoch | hurin t_mid | turin t_mid | agreement | turin's precision |
|---|---|---|---|---|
| -1 | 152.70142 ± 0.1055 | 152.74169 ± 0.0432 | 0.35 sigma | 2.4x tighter |
| 0 | 781.18863 ± 0.0885 | 781.15475 ± 0.0175 | 0.38 sigma | 5.1x tighter |
| +1 | 1408.47498 ± 0.3368 | 1408.22771 ± 0.0380 | 0.73 sigma | 8.9x tighter |

The times agree, so neither fit is in a different mode. What changed is the
width: turin's timing uncertainties are 2.4x to 8.9x smaller, with bulk ESS
of 19,000-27,000 per transit time against hurin's few thousand. On a target
with three transits and a weak per-epoch constraint, that is the difference
between 256 chains seeded at each transit's own likelihood peak and a handful
started at the linear ephemeris.

turin's own diagnostic flagged the right epoch unprompted:

    WARNING: 1 epoch(s) have a rival timing mode within 10 log-units
      epoch -1: seed -127.0 min, rival gap 3.7

Two notes on reading that table. The O-C values for epochs -1 and +1 are
identical by construction, not by coincidence: a least-squares line through
three equally spaced points forces `r(-1) = r(+1) = -r(0)/2`, so a
three-transit O-C diagram carries exactly one degree of freedom. And the
whole fit took 29 seconds per sampling round, on turin 0.1.0.

That last number is not comparable to a current one, and the reason is worth
recording. 0.1.0 integrated the exposure by supersampling, which is cheap and
wrong (see section 3); adopting MetalPlanet's contact-rule kernel in 0.1.6
bought the accuracy at roughly twice the cost per evaluation, and the same
fit took 220 s. Feeding that kernel its points in phase order (0.1.8) and
setting `N_GL` from the log-likelihood (0.1.9) brought it back to 119 s —
still above 82 s, but now for a model that integrates exposures correctly.
The transit times themselves moved by at most 0.05 sigma across all of it,
and their uncertainties by under 0.6%, so the table above stands as measured.

## Verification

`tests/test_model.py` pins all of it:

| check | tolerance | measured |
|---|---|---|
| fp64 `circular` vs MetalPlanet's batman-style frontend | 1e-14 | 3.3e-16 |
| fp32 `circular` vs the same | 5e-7 | 1.6e-7 |
| fp64 exposure integration vs MetalPlanet supersampling | 1e-14 | 3.3e-16 |
| `chord` + `hurin` LD vs hurin/jaxoplanet | 1e-6 | 2.2e-7 |
| `circular` vs `chord` | 1e-6 < d < 1e-4 | 4.6e-6 |
| Kipping vs hurin LD map | > 1e-5 | 2.4e-4 |

## A note on gradient testing

Finite differences are a **weak** reference for this model and should not be
trusted below ~1e-6 relative. MLX's float64 `sin`/`cos` are only
float32-accurate (measured 2.6e-8 relative error), so anything routed
through them is a staircase at that scale and a small-step FD differentiates
the staircase, not the function. An early FD check of `a_over_rstar`
disagreed with the truth by 11% for exactly this reason while the autodiff
value was correct to 6.6e-8.

Two stronger references are used instead:

- **Closed forms.** `d(a/R*)/dT14` has an elementary expression; autodiff
  matches it to 6.6e-8.
- **Exact structural invariants.** Under a linear ephemeris one epoch's
  mid-time is `dtau0 + n dP`, so `dF/d(dP)` must equal `n dF/d(dtau0)`
  identically. In `chord` mode, where the period enters nowhere else, turin
  satisfies this to 1.5e-16.

`a_over_rstar` uses `metalplanet.trig.sincos` (Cody-Waite reduced) rather
than `mx.sin` so the float64 reference path does not inherit that 2.6e-8
staircase.
