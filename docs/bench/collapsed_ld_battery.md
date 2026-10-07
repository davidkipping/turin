# `--ld=collapsed` on 20 Kepler targets

The first 20 rows of a Kepler target list (KOI-5162.01 to KOI-2686.01, P =
211-628 d, 2-7 transits, catalogue SNR 7.6-272), LinEph only, each fitted in
both limb-darkening modes on the same machine (Apple M2 Pro, 32 GB):

    # collapsed: turin 0.1.45, anvil 0.4.2, MetalPlanet 0.9.7, 2026-10-06
    turin --KOI-<n> --ld=collapsed --modes=lineph --tag=ldc

    # default (--ld=sampled), from earlier batteries run with defaults:
    #   rows 1-10:  turin 0.1.34, anvil 0.4.0, MetalPlanet 0.9.7 (--tag=v034)
    #   rows 11-20: turin 0.1.38, anvil 0.4.2, MetalPlanet 0.9.7

Both modes: 512 chains, warmup 800, the default doubling rounds to
convergence or the 16,384-draw cap, seed 0. The default path's LinEph
sampling is unchanged across 0.1.34-0.1.45 (0.1.39 recorded sampled-mode
chains byte-identical before and after collapsed LD was added), and anvil
0.4.0 and 0.4.2 sample identically for turin's static target. The default
runs used `--PL=auto`, which **chose `exact` on all 20 targets** — the solve
collapsed mode forces — so the comparison isolates the limb-darkening
treatment. Targets ran one at a time with nothing else on the GPU; the
default runs also fitted TTV, which is excluded by timing the LinEph fit
alone from its log.

`collapsed_ld_battery_compare.py` produced every number below. It uses the
method of `collapsed_ld_compare.py` (shift in sd and in Monte Carlo standard
errors, width ratio, two-sample KS against a chain-split null, ESS/s on the
five parameters both modes sample over wall time that includes the post-hoc
`q` draws), pointed at these runs.

## Accuracy: the same posterior

20 targets × 8 parameters (P, tau0, k, b, T14, q1, q2, log10_rho):

| | |
|---|---|
| median \|shift\| | 0.0017 sd |
| largest \|shift\| | 0.013 sd (KOI-518.03 `b`) |
| shifts beyond 3 Monte Carlo standard errors | 0 / 160 |
| KS within the chain-split null's 95th percentile | 155 / 160 (about 8 failures expected by chance) |
| width of the bulk (16-84%), collapsed / default | 0.97-1.04 |
| `q1`, `q2` draws exactly on the box edge (collapsed) | 4, out of millions |

The sd ratio reaches 4.85 (KOI-5387.01 `k`) and 1.71 (KOI-701.04 `k`), but
those are the **grazing tail**, not the bulk: their 16-84% widths agree to
0.998 and 1.021. The tail holds 0.03% vs 0.17% (5387) and 2.3% vs 2.7% (701)
of draws, and its far end sets the sd (KOI-701.04's 99.9th percentile of `k`
is 0.38 vs 0.67). Neither mode moves chains in and out of that tail
efficiently, so its sampled mass is noisy in both, and it is what failed the
KS null on KOI-701.04 (`k`, `b`, `T14`). The other two KS failures are
KOI-7591.01 `dP` and `dtau0`, at 0.0037 against a 0.0035 threshold and 0.0034
against 0.0034.

### Real grazing posteriors

The acceptance runs (`collapsed_ld_acceptance.md`) used `--nongrazing` on
KOI-448.02, leaving grazing covered only by synthetic tests. These used the
default transiting prior, and **14 of the 20 targets carry real grazing mass**
(`b > 1 - k`), up to 78%. Collapsed mode reproduces it on every one, to
within 0.4 percentage points:

| target | grazing fraction, default | collapsed |
|---|---|---|
| KOI-5228.01 | 77.59% | 77.72% |
| KOI-7776.01 | 63.96% | 63.98% |
| KOI-7592.01 | 38.58% | 38.61% |
| KOI-5897.01 | 22.79% | 22.64% |
| KOI-5749.01 | 18.18% | 17.98% |
| KOI-5574.01 | 16.69% | 16.91% |
| KOI-5616.01 | 12.79% | 13.04% |
| KOI-4848.01 | 9.63% | 9.47% |
| KOI-7591.01 | 7.39% | 7.47% |
| KOI-5764.01 | 6.49% | 6.60% |
| KOI-3975.01 | 2.89% | 3.07% |
| KOI-701.04 | 2.32% | 2.71% |
| KOI-5162.01 | 0.99% | 1.05% |
| KOI-7567.01 | 0.85% | 0.89% |

## Cost: usually slower

| | default | collapsed |
|---|---|---|
| LinEph wall time, all 20 | 279 min | **343 min (+23%)** |
| per target, collapsed / default | | median **1.26x** (0.31-3.0x) |
| ESS per second, collapsed / default | | median **0.77x** (0.39-2.83x) |
| targets where collapsed is faster per ESS | | 6 / 20 |

Collapsed mode draws ~20% fewer samples per second (KOI-5162.01: 1,596 vs
2,016 draws/s) and pays for the post-hoc `q` draws at every export, while
mixing slightly better per draw (10-25% more ESS at equal draws on
KOI-5162.01). It wins clearly on the highest-SNR target, KOI-868.01 (SNR 272,
a Jupiter): 2.83x the ESS/s, 3.2 against 10.2 min. Across the 20 the ESS/s
ratio rises only weakly with catalogue SNR (Spearman rho 0.37, p = 0.11), so
this does not establish an SNR threshold above which it pays.

Per-target ratios are noisy: the stopping rule acts round by round, so one
round more or fewer swings a target's wall time by up to ~2x (KOI-5897.01
converged in 1 round sampled, 3 collapsed; KOI-5574.01 the reverse). The
medians over 20 are the figures to quote.

## Memory

Both small. Collapsed: peak process RSS 2.2 GB (default 2.7); GPU allocation
typically 3.5-4.7 GB per target against 3.0-3.8 GB, so ~0.5-1 GB more, though
the single highest peak is the default's (5.1 vs 4.9 GB, both on
KOI-5387.01). macOS memory pressure normal throughout in both.

## Verdict

Equivalent posteriors, including real grazing ones, at ~1.25x the LinEph
cost on typical Kepler targets. Consistent with the acceptance runs
(break-even to 1.41x). It supports keeping `--ld=sampled` the default; and
since collapsed mode is LinEph-only, it cannot shorten the TTV fits, which
dominate turin's wall time.

## Reproducing

The default-mode runs come from turin's battery scripts and the collapsed
ones from

    for each target in rows 1-20:
      /usr/bin/time -l turin --KOI-<n> --tag=ldc --ld=collapsed --modes=lineph

with a 2-second resource monitor beside each. The comparison script expects
each run's timestamped log, `monitor.csv`, `wall.txt` and `time_and_stderr.txt`
in `benchmarks/<date>_<target>[_<tag>]/`, and the products (summary and chains
tarball) in `<target>/`. Chains tarballs are not kept here (tens of MB each).
