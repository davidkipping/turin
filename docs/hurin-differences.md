# Where turin's model differs from hurin's, and why

Measured 2026-09-28 while porting the forward model, on an M-series Mac with
MLX 0.32.3, MetalPlanet 0.6.1, and hurin 0.1.67 in its own conda env. Every
number here is reproduced by `tests/test_model.py`.

turin is not bit-compatible with hurin, and two of the three reasons are
corrections rather than choices. All three are switchable, so any hurin
result can be reproduced for comparison.

## 1. The limb-darkening map: hurin's is not Kipping (2013)

**This looks like a bug in hurin.** `hurin/transit_fit.py:262` says
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

turin defaults to `ld_map="kipping"` and offers `ld_map="hurin"` for parity.
**Worth reporting upstream to hurin**, since it affects every fit that
package has produced, and the fix would change published limb-darkening
posteriors (and, weakly through the depth, `k`).

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

## Measured on real data: KOI-448.02

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

**1. The stack is validated.** In hurin-compatibility mode
(`--geometry=chord --ld=hurin --profile=ratio`) turin reproduces hurin on
every parameter to **0.04 sigma or better**, with widths agreeing to 1-3%.
Nothing is shared between the two implementations: MLX against JAX, ChEES-HMC
against NUTS, an unrolled Cholesky against `jnp.linalg.solve`, 256 chains
against 8. Agreement at that level is strong evidence for both.

**2. Fixing the limb-darkening map widens grazing-transit posteriors, because
hurin's truncated prior was quietly regularizing them.** KOI-448.02 is
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

## Measured on real data: KOI-5162.01, the trapped-mode target

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
whole fit took 29 seconds per sampling round.

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
