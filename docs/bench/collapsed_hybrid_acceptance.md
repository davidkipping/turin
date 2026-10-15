# Collapsed hybrid4: real-target acceptance

> **Measured on hybrid4's old basis** (turin 0.1.62-0.1.64, elome 0.16.2):
> weights on `1 - mu^4`, `(1 - mu^2)^2` and the two poles, stick-broken from
> `q`. Since turin 0.1.67 (elome >= 0.17) the weights are on LDhybrid's
> curvature basis, a different simplex of profiles; nothing here has been
> re-measured on it.

`--law=hybrid4 --ld=collapsed` against `--law=hybrid4` (sampled), LinEph, on
the user's five-target suite: KOI-4016.01, KOI-4087.01, KOI-1430.03,
KOI-1361.01, KOI-2650.01.

These are typical Kepler targets: confirmed planets around cool K and M
hosts, P = 35–125 d, 8–42 transits, archive SNR 11.5–52, b = 0.00–0.82, none
grazing.

    # sampled: turin 0.1.63 (that path is bitwise unchanged in 0.1.64)
    turin --KOI-<n> --modes=lineph --PL=exact --law=hybrid4 --tag=h4s
    # collapsed: turin 0.1.64
    turin --KOI-<n> --modes=lineph --PL=exact --law=hybrid4 --ld=collapsed --tag=h4c2
    # second seed of both modes on three targets: --seed=1, tags h4s_s1 / h4c2_s1

Every run used:
- 512 chains, warmup 800, the default doubling rounds, seed 0;
- one run at a time on an Apple M2 Max (64 GB), with the light curves cached
  first so no wall time is a download;
- `--PL=exact` in both modes, so only the limb-darkening treatment differs.

The comparison applies 0.1.44's rules:
- shift in sd and in Monte Carlo standard errors;
- 16–84% width ratio;
- two-sample KS against a chain-split null;
- ESS/s over end-to-end wall time, on the five parameters both modes sample;
- `q` draws exactly on 0 or 1.

Then a second seed of both modes on the targets that needed one.
`collapsed_hybrid_compare.py` and `collapsed_hybrid_seeds.py` produced every
number below.

## First pass (0.1.63): failed

The first pass ran 0.1.63's collapsed hybrid4. The geometry agreed everywhere
the sampler ran, but three defects showed up.

| defect | where | cause |
|---|---|---|
| `q`/`w` biased up to 0.41 sd (85 MCSE) | KOI-4016.01 (0.04–0.06 sd on 4087.01, 1430.03) | the conditional draws: an independence sampler through 32 GHK draws, whose weights degenerate where the conditional is narrow and off the model's axis |
| 70/512 chains trapped at k = 0; step size 1.2e-10 | KOI-2650.01 (run killed) | no transit means no curvature; the relative ridge vanished and GHK returned +3,456 nats, a false peak |
| 3 `q` draws exactly on the box wall | KOI-4016.01, KOI-1361.01 | a float32 truncated-normal draw rounded onto its interval's end |

Cost was 3.2–7.6× sampled mode's wall time. The fixes are in VERSIONS.md
0.1.64; the design and its measurements are in CLAUDE.md, "Collapsed
hybrid4".

## Second pass (0.1.64): passes

### Accuracy

Fourteen quantities per target: P, tau0, k, b, T14, log10_rho, q1..q4 and
w1..w4.

| target | max \|shift\| | max \|shift\| / MCSE | width 16–84% | KS (chain-split null) | walls |
|---|---|---|---|---|---|
| KOI-4016.01 | 0.012 sd | 2.4 | 0.996–1.009 | 10/14 | 0 |
| KOI-4087.01 | 0.016 sd | 2.6 | 0.986–1.005 | 9/14 | 0 |
| KOI-1430.03 | 0.009 sd | 1.9 | 0.992–1.001 | 14/14 | 0 |
| KOI-1361.01 | 0.006 sd | 1.2 | 0.993–1.004 | 14/14 | 0 |
| KOI-2650.01 | 0.007 sd | 1.3 | 0.991–1.004 | 14/14 | 0 |

The worst `q`/`w` shift went down on every target:

| target | 0.1.63 | 0.1.64 |
|---|---|---|
| KOI-4016.01 | 0.411 sd | 0.012 sd |
| KOI-4087.01 | 0.038 sd | 0.012 sd |
| KOI-1430.03 | 0.064 sd | 0.009 sd |
| KOI-1361.01 | 0.013 sd | 0.005 sd |
| KOI-2650.01 | — | 0.007 sd |

Conditional-draw acceptance is 0.89–0.95, and no `q` draw lands on a wall.

**The KS failures are run-level noise, not a difference between the
modes.** A second seed of both modes (KOI-4016.01, KOI-4087.01, KOI-2650.01)
gives a true two-run null: two runs of the same mode across seeds. The
cross-mode KS distances are no larger than that null:

| target | same mode, across seeds: median / max | across modes: median / max |
|---|---|---|
| KOI-4016.01 | 0.0046 / 0.0068 | 0.0038 / 0.0088 |
| KOI-4087.01 | 0.0031 / 0.0050 | 0.0029 / 0.0063 |
| KOI-2650.01 | 0.0029 / 0.0044 | 0.0030 / 0.0048 |

Two runs of one mode reach KS distances of 0.0068, above the chain-split
null's thresholds of about 0.005. That null is optimistic, as CLAUDE.md
warns, since ChEES shares adaptation across a run's chains.

KOI-4087.01's seed-0 KS failures on `k` and `b` (0.0063, 0.0062) were one
run's fluctuation. At seed 1 the same pairs give 0.0036 and 0.0034.

### KOI-2650.01's grazing tail

KOI-2650.01's collapsed runs carry a far grazing tail that its sampled runs
do not. The difference is reproducible across seeds:

| run | draws with k > 0.06 | chains visiting | grazing | max k |
|---|---|---|---|---|
| sampled, seed 0 | 0.014% | 2 | 0.32% | 0.065 |
| sampled, seed 1 (3 rounds, 537,600 draws) | 0.000% | 0 | 0.34% | 0.049 |
| collapsed, seed 0 | 0.28% | 72 | 0.60% | 0.86 |
| collapsed, seed 1 | 0.33% | 43 | 0.63% | 0.998 |

This tail sets `k`'s sd, a 4.6× ratio, while its 16–84% width agrees to
0.996.

Collapsed mode's density in the tail is correct. At 30 tail thetas (k from
0.07 to 0.81), its value matches the true marginal to −0.05..+0.03 nats,
mean 0.000. That marginal is importance sampling of the true L over q.

The likely reading is that **sampled mode under-explores the tail**. In 9-D
the tail needs large k jointly with suitable limb darkening, a narrow path
for ChEES. Sampled mode's seed-1 run needed 3 rounds, with R-hat on β/k up
to 1.014 and k's ESS stalling. This is the same signature the quadratic
battery recorded (`collapsed_ld_battery.md`, KOI-5387.01). It is not proven:
the true mass at k > 0.06 has no reference independent of both samplers.

### Cost

Seed-0 runs. ESS/s is the minimum bulk ESS over the shared five parameters
per end-to-end second.

| target | wall, sampled | wall, collapsed | × | ESS/s, sampled | ESS/s, collapsed | × | peak RSS × |
|---|---|---|---|---|---|---|---|
| KOI-4016.01 | 126 s | 1,089 s | 8.65 | 365.3 | 82.2 | 0.23 | 1.30 |
| KOI-4087.01 | 279 s | 1,221 s | 4.37 | 224.0 | 69.7 | 0.31 | 1.18 |
| KOI-1430.03 | 266 s | 1,224 s | 4.60 | 195.2 | 67.8 | 0.35 | 1.19 |
| KOI-1361.01 | 419 s | 1,549 s | 3.70 | 142.2 | 64.4 | 0.45 | 1.17 |
| KOI-2650.01 | 398 s | 1,577 s | 3.96 | 124.4 | 65.9 | 0.53 | 1.16 |

The median is 4.4× the wall time and 0.35× the ESS/s.

Collapsed mode draws 1.4–2.1× the ESS per draw, because it samples fewer
dimensions. But each evaluation carries a fixed per-chain GHK cost, and the
post-hoc conditional draws add about 200 s per export. So the cost is worst
on small datasets: KOI-4016.01 has 354 points.

**Sampled hybrid4 stays the default.** Collapsed is opt-in for wanting the
limb darkening out of the sampled vector. Its one observed advantage here is
the grazing tail above, which is the reason to try it on a target whose
sampled fit mixes poorly in `k`.
