# Collapsed hybrid4: the probe behind the design (turin 0.1.62)

> **Measured on hybrid4's old basis** (turin 0.1.62-0.1.64, elome 0.16.2):
> weights on `1 - mu^4`, `(1 - mu^2)^2` and the two poles, stick-broken from
> `q`. Since turin 0.1.67 (elome >= 0.17) the weights are on LDhybrid's
> curvature basis, a different simplex of profiles; nothing here has been
> re-measured on it.

Before `turin/ldhybrid.py` was written, an independent NumPy float64 probe
(the generator curves from elome's `ld_basis` through turin's grid, all else
NumPy) measured whether turin's collapsed construction carries from the
quadratic law's 2-D triangle to hybrid4's 4-simplex. It ran at 100 posterior
draws of sampled hybrid4 fits (turin 0.1.61, `--modes=lineph --PL=exact`,
512 chains): KOI-518.02 (converged, one round) and KOI-448.02 (two-solution
posterior, unconverged after three rounds, draws through round 2).

**Gate A, the quadratic model of the profiled `L(x)`.** Collapsed value minus
the true `log int exp(L) p` (importance sampling of the true `L` from the
model, cross-checked by plain uniform-simplex QMC on four draws):

| | 1 Newton pass | 2 | 3 | IS vs uniform QMC |
|---|---|---|---|---|
| KOI-518.02 | < 1e-4 (all draws) | < 1e-4 | < 1e-4 | 3e-4 to 1.5e-3 |
| KOI-448.02 | < 1e-4 (all draws) | < 1e-4 | < 1e-4 | up to 0.05 (QMC's noise) |

**Gate B, float32 conditioning.** Condition number of the projected,
rescaled omega columns: median 58 / max 200 (KOI-518.02), median 80 / max
8.5e4 (KOI-448.02); of `H`: up to 4e4 and 7e9. float32 `H` entries were
accurate to 2.5e-7 by either a Gram sum or MGS, but solving with the worst
`H` was wrong by up to 12x either way: hence all algebra on `R`.

**GHK error, frozen layout, against a 2^18 reference** (60 draws each):

| N | KOI-518.02 median / 95% / max | KOI-448.02 median / 95% / max |
|---|---|---|
| 256 | 0.007 / 0.032 / 0.053 | 0.031 / 0.098 / 0.146 |
| 1024 | 0.003 / 0.009 / 0.015 | 0.010 / 0.032 / 0.085 |
| 4096 | 0.001 / 0.005 / 0.007 | 0.002 / 0.017 / 0.024 |

A layout chosen afresh at each draw was no better (KOI-448.02, N=256, max
0.46). turin uses N = 1024.

**Sampled hybrid4 vs quadratic** (the same runs, `--PL=exact`, default prior):

| | quadratic | hybrid4 |
|---|---|---|
| KOI-518.02 | 449 s, 1 round, R-hat 1.0056, ESS 62,293 | 455 s, 1 round, R-hat 1.0045, ESS 69,149 |
| KOI-448.02 | 2,214 s, 3 rounds, converged | 3 rounds, R-hat 1.26 (k), ESS 1,402, stopped |

KOI-448.02 is a low-SNR grazing system, not a typical target: hybrid4 opens
a small-planet, extreme-limb-darkening solution there (k 0.027, b 0.27, q1
within 0.01 of its wall for 68% of draws) beside the quadratic law's grazing
one (k 0.16, b 1.10), and chains do not cross.

The implementation then reproduced the probe: production float64 against the
probe's NumPy GHK to 4e-7 on the same rule; collapsed against the true
integral to < 5e-4 nats near the posterior on synthetic data at the Kepler
noise level (< 5e-3 at 4e-5 per point).
