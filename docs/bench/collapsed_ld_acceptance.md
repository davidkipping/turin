# `--ld=collapsed` acceptance against the default

Runs made on turin 0.1.42. The collapsed-LD code was unchanged from 0.1.41,
and 0.1.42 only touched TTV display. The results are recorded in 0.1.43,
and corrected in 0.1.44 after a code review of the analysis (below). Both
modes ran on the same machine, sequentially, with nothing else on the GPU:

    turin --KOI-518.02 --modes=lineph --PL=exact --tag=s
    turin --KOI-518.02 --modes=lineph --PL=exact --tag=c --ld=collapsed
    turin --KOI-448.02 --modes=lineph --PL=exact --nongrazing --tag=s
    turin --KOI-448.02 --modes=lineph --PL=exact --nongrazing --tag=c --ld=collapsed
    # replication, KOI-518.02 only: the same two with --seed=1

Settings: 512 chains, default warmup and doubling, run to convergence, seed
0 unless stated. `--PL=exact` for both modes, because collapsed mode
requires it. `collapsed_ld_compare.py` reads the chains tarballs and the
run logs.

**KOI-448.02 ran under `--nongrazing`** (b < 1 − k), as in
`hurin-differences.md`. That excludes grazing geometries, so no real grazing
posterior has been run in collapsed mode. Grazing is covered by synthetic
tests: `vertex_flux_devs` at b > 1 − k in `test_model.py`, and a grazing
case in every brute-force test in `test_ldmarg.py`.

## Method, and what the first version got wrong

Per parameter, the script reports:

- **shift** — `(median_c − median_s) / sd_s`, and the same in Monte Carlo
  standard errors;
- **width** — `sd_c / sd_s`;
- **KS** — the two-sample KS distance;
- **edge draws** — `q` draws exactly on the box edge.

The KS distance needs a null: how large it gets for two runs drawn from the
same distribution. The 0.1.43 write-up used random *rows* of the sampled
run. Draws within a chain are autocorrelated (+0.55 at one draw's lag for
KOI-448.02's `k`), so that reference was not a null for two independent
runs. On top of that, its pass rule allowed 1.5× that reference without
saying so.

This version splits each run's **chains** into random halves, 40 times, and
scales each half-versus-half KS by 1/√2 to full-run size. It pools those
values from both runs, which brackets unequal ESS between the modes. A
parameter passes if its KS is at or below that null's 95th percentile.

Even this null can be optimistic. ChEES shares one adapted step size and
trajectory length across all 512 chains, so chains within a run are not
fully independent. The KOI-518.02 replication with a second seed is the
empirical check on that.

## KOI-518.02 (153,600 draws per run)

| param | shift / sd | shift / MC se | width | KS | null 95% | percentile |
|---|---|---|---|---|---|---|
| P | +0.003 | +0.60 | 0.997 | 0.0026 | 0.0054 | 29% |
| tau0 | +0.000 | +0.10 | 1.001 | 0.0025 | 0.0048 | 22% |
| k | +0.004 | +0.67 | 0.995 | 0.0030 | 0.0059 | 38% |
| b | +0.001 | | 1.003 | 0.0027 | 0.0057 | 22% |
| T14 | −0.001 | −0.19 | 0.998 | 0.0028 | 0.0059 | 20% |
| **q1** | **−0.011** | **−2.35** | 0.997 | **0.0063** | 0.0051 | **100%** |
| q2 | +0.004 | +0.94 | 0.997 | 0.0025 | 0.0050 | 34% |
| log10_rho | −0.001 | | 1.001 | 0.0021 | 0.0059 | 4% |

Against this null, **`q1` fails on seed 0**: its KS exceeds all 80 null
values, and its median is 2.35 standard errors low. The 0.1.43 write-up
called this a pass.

**Replication.** A second run in each mode with `--seed=1` gives KS for
`q1`:

| | same mode: s0–s1, c0–c1 | cross mode: c0–s0, c1–s1, c0–s1, c1–s0 |
|---|---|---|
| KS | 0.0035, 0.0038 | **0.0063**, 0.0028, 0.0038, 0.0045 |

and median shifts in sampled-sd units:

| | s0–s1 | c0–c1 | c0–s0 | c1–s1 | c0–s1 | c1–s0 |
|---|---|---|---|---|---|---|
| shift | +0.0071 | −0.0045 | −0.0113 | +0.0004 | −0.0042 | −0.0067 |

A real difference between the modes would make every collapsed run differ
from every sampled run. Instead, only the seed-0 pairing is elevated: the
seed-1 pairing has the *smallest* KS of all, and the other two cross pairs
sit at same-mode levels.

The seed-0 discrepancy is two run-level fluctuations in opposite
directions. s0 lies 0.0071σ high relative to s1, and c0 lies 0.0045σ low
relative to c1, so the two compound in that one pairing. Pooling both seeds
of each mode, collapsed vs sampled differ by −0.0056σ, inside the
within-mode spread between seeds (0.0045–0.0071σ).

Across all eight parameters, the 32 cross-mode KS values (0.0018–0.0063)
span the same range as the 16 same-mode ones (0.0020–0.0048), apart from
that one compounded pair. Conclusion: **no evidence of a difference between
the modes.**

Redrawing seed 0's `q1, q2` on its own θ draws with 8, 32 and 128 MH steps
and three seeds leaves the shift where it is (−0.006 to −0.0125σ). The
post-hoc conditional sampler is therefore not the source.

`q`: no draw lands exactly on the box edge. `x*` lies on a triangle edge for
23% of draws, and MH acceptance is 0.93.

## KOI-448.02 (460,800 draws per run)

| param | shift / sd | shift / MC se | width | KS | null 95% | percentile |
|---|---|---|---|---|---|---|
| P | +0.003 | +1.49 | 0.997 | 0.0022 | 0.0024 | 89% |
| tau0 | −0.002 | −1.02 | 1.001 | 0.0019 | 0.0028 | 54% |
| k | +0.001 | +0.19 | 1.009 | 0.0014 | 0.0053 | 9% |
| b | −0.001 | | 1.025 | 0.0038 | 0.0052 | 71% |
| T14 | +0.000 | +0.02 | 1.005 | 0.0011 | 0.0048 | 0% |
| q1 | −0.001 | −0.37 | 0.994 | 0.0013 | 0.0030 | 18% |
| q2 | +0.002 | +0.77 | 0.995 | 0.0029 | 0.0036 | 82% |
| log10_rho | +0.001 | | 1.020 | 0.0026 | 0.0047 | 61% |

Everything passes against the calibrated null.

This is the edge case for the limb darkening. `x*` lies on a triangle edge
for 54% of draws, and 17% of `q1` draws are within 0.01 of 1 — the regime
where the reference implementation's first sampler froze 5% of draws
exactly on the edge. Here no draw is on the edge; the closest is 2.2e-8 from
`q1 = 1`.

The near-edge counts form a continuous tail, not an atom. There are 78,060,
8,383, 886 and 9 draws within 1e-2, 1e-3, 1e-4 and 1e-6 of `q1 = 1`,
falling tenfold per decade as a density that is finite at the boundary
should. MH acceptance is 0.98.

## Cost (seed-0 runs)

ESS/s is the minimum bulk ESS over the five parameters both modes sample
(dP, dtau0, k, beta, T14), divided by **end-to-end wall time**. Wall time
includes collapsed mode's post-hoc conditional draws, which a
sampling-rounds-only time omits.

| | sampled | collapsed | collapsed time per ESS |
|---|---|---|---|
| KOI-518.02 | 67,295 ESS / 428 s = **157.4** | 93,819 / 600 s = **156.5** | **1.01×** |
| KOI-448.02 | 65,007 / 1,224 s = **53.1** | 60,224 / 1,601 s = **37.6** | **1.41×** |

Collapsed mode breaks even on KOI-518.02 and takes 1.41× the wall time per
effective sample on KOI-448.02. Both modes needed two rounds there. The
sampled run had no trapped chains, so removing the edge-hugging `q1, q2`
from ChEES had nothing to fix.

The seed-1 replication runs are not used for cost: tests briefly shared the
GPU during the seed-1 sampled run.

Peak process RSS was about 12 GB in both modes (12.2 / 11.9 GB on
KOI-518.02, 12.4 / 12.1 GB on KOI-448.02). It is a whole-run figure. What
sets it was not isolated here: it covers the MAP search, sampling, export
and diagnostics, not the 2.84 GB "assess + export + certify" measurement in
CLAUDE.md's memory section. For scale, the whole-run figure for the same
target at 0.1.12 was 17.3 GB. Collapsed mode's per-evaluation MLX peak
(about 2.2× the default) was measured separately.

## The default path is unchanged

The sampled KOI-518.02 **chains** (153,600 draws) were also run on 0.1.38,
the last commit before the feature. The chain bodies are byte-identical;
only the version stamp on line 2 differs. That covers the chains on this one
target.

Separately, the synthetic test in 0.1.39 and 0.1.41 compared the chains,
summary, lcdata and logrho bodies. No other product or target was compared
on real data. TTV products did change in 0.1.42, by design: the O-C
reference line.
