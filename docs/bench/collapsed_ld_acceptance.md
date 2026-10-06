# `--ld=collapsed` acceptance against the default (turin 0.1.42)

Agreed with SquishierPlanet, who proposed the scheme. Both modes, same
machine, run sequentially, nothing else on the GPU:

    turin --KOI-518.02 --modes=lineph --PL=exact --tag=s
    turin --KOI-518.02 --modes=lineph --PL=exact --tag=c --ld=collapsed
    turin --KOI-448.02 --modes=lineph --PL=exact --nongrazing --tag=s
    turin --KOI-448.02 --modes=lineph --PL=exact --nongrazing --tag=c --ld=collapsed

512 chains, seed 0 (the default), default warmup and doubling, to
convergence. `--PL=exact` for both modes, because collapsed requires it.
`--nongrazing` on KOI-448.02 as in `hurin-differences.md`.
`collapsed_ld_compare.py` reads the chains tarballs and run logs.

## Criteria

Per parameter: `|median_c - median_s| / sd_s <= 0.03`; `sd_c / sd_s` in
`[0.97, 1.03]`; two-sample KS no more than about the KS between two random
halves of the sampled chains (the self-resample level); zero `q1` or `q2`
draws exactly on the box edge.

## KOI-518.02 (153,600 draws each)

| param | shift / sd | width ratio | KS c vs s | KS self |
|---|---|---|---|---|
| P | +0.003 | 0.997 | 0.0026 | 0.0037 |
| tau0 | +0.000 | 1.001 | 0.0025 | 0.0055 |
| k | +0.004 | 0.995 | 0.0030 | 0.0051 |
| b | +0.001 | 1.003 | 0.0027 | 0.0034 |
| T14 | -0.001 | 0.998 | 0.0028 | 0.0027 |
| q1 | -0.011 | 0.997 | 0.0063 | 0.0056 |
| q2 | +0.004 | 0.997 | 0.0025 | 0.0052 |
| log10_rho | -0.001 | 1.001 | 0.0021 | 0.0056 |

The expansion point `x*` lay on a triangle edge for 23% of draws, and MH
acceptance was 0.93. No draw landed on the box edge.

## KOI-448.02 (460,800 draws each) -- the sharp case

| param | shift / sd | width ratio | KS c vs s | KS self |
|---|---|---|---|---|
| P | +0.003 | 0.997 | 0.0022 | 0.0033 |
| tau0 | -0.002 | 1.001 | 0.0019 | 0.0017 |
| k | +0.001 | 1.009 | 0.0014 | 0.0022 |
| b | -0.001 | 1.025 | 0.0038 | 0.0044 |
| T14 | +0.000 | 1.005 | 0.0011 | 0.0022 |
| q1 | -0.001 | 0.994 | 0.0013 | 0.0035 |
| q2 | +0.002 | 0.995 | 0.0029 | 0.0026 |
| log10_rho | +0.001 | 1.020 | 0.0026 | 0.0019 |

The limb darkening piles against the edge here: `x*` was on a triangle edge
for 54% of draws, and 17% of `q1` draws lie within 0.01 of 1. This is the
regime where the reference's first sampler froze 5% of draws exactly on the
edge.

Here, no draw lands on the edge, and the closest is 2.2e-8 from `q1 = 1`.
The draws near the edge form a continuous tail, not an atom: within 1e-2,
1e-3, 1e-4 and 1e-6 of `q1 = 1` there are 78,060, 8,383, 886 and 9 draws,
falling by a factor of 10 per decade as a density that is finite at the
boundary should. MH acceptance was 0.98.

## Cost

| | sampled | collapsed |
|---|---|---|
| KOI-518.02: rounds, sampling s, min bulk ESS, ESS/s | 1, 411, 67,295, **163.7** | 1, 572, 93,819, **164.0** |
| KOI-448.02: rounds, sampling s, min bulk ESS, ESS/s | 2, 1,194, 65,007, **54.5** | 2, 1,524, 60,224, **39.5** |
| peak process RSS, KOI-518.02 / KOI-448.02 | 12.2 / 12.4 GB | 11.9 / 12.1 GB |

KOI-518.02 is break-even, matching SquishierPlanet's 157 against 162.
KOI-448.02 is 28% slower in collapsed mode. The sampled run there had no
trapped chains, so removing the edge-hugging `q1, q2` from ChEES had nothing
to fix, and collapsed mode's more expensive evaluations set the cost.

Whole-process RSS is set by the export and diagnostics, not by the
log-density, so it does not show the ~2.2x per-evaluation MLX peak measured
separately.

## The default path is unchanged

The sampled KOI-518.02 chains (153,600 draws) were also run on 0.1.38, the
last commit before the feature. The two chain bodies are byte-identical; only
the version stamp on line 2 differs.
