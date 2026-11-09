# Gaia acceptance on 13 known hot Jupiters

Plan step 5. The runs used the 0.1.105 code: the 5-unit box, the ephemeris
Gibbs move with differential-evolution steps, and the edge warning. Their
products are stamped 0.1.104, because the batch ran before the version bump,
with the 0.1.105 changes in the working tree. Each is one DR3 source, read
from MetalLS's October DataLink downloads (the Gaia archive returned 503s at
the time), and ran one seed, sequentially, alone on the GPU:

    turin --GaiaDR3-<id> --period=P --t0=T0 --T14=H --depth=D

The ephemeris is the NASA Exoplanet Archive's (`gaia_cands.txt`; t0
converted to Gaia time with no TCB-TDB shift, ~18 s, which is negligible
here). Four more runs were seeded at a rival period peak found in step 0
(`gaia_cands_rival.txt`). Settings: 512 chains, default warmup, run to
convergence or to the 16,384-draw cap. Wall time 4.2 h for all 17.

The literature values come from the archive's composite table, queried
2026-11-09 (`gaia_literature.csv`). `gaia_acceptance.py` reads the chains
tarballs and logs. For each quantity it gives:

- the difference between the posterior median and the literature value;
- that difference in posterior sd (half the 16-84% range);
- the two-sided tail probability `p` of the literature value under the
  draws, convolved with the literature error.

For the epoch, the literature value is propagated to the fitted epoch
(`n` orbits), and its error grows by `n * sigma_P`. `T0` is the fit's
`T0_TDB`.

## Convergence

    run                conv   wall rnds   R-hat  P-seed  P-true   T0-tr             k          b  edgeP  edgeT
    KELT-16_b           yes    57s    1  1.0074   -0.92    -0.9   -1.8m 0.112+-0.003 0.33+-0.18  0.000  0.000  a
    WASP-18_b           yes   420s    3  1.0056   -2.93    -2.9   +3.1m 0.096+-0.077 0.38+-0.33  0.000  0.000  a
    HD_189733_b         yes  1928s    6  1.0068   +1.32    +1.3  +18.4m 0.189+-0.163 0.67+-0.38  0.003  0.000  a
    WASP-12_b           yes   482s    4  1.0047   +0.09    +0.1   +0.7m 0.142+-0.163 0.70+-0.40  0.000  0.000  a
    HAT-P-1_b            NO  1951s    6  1.0149   +0.44    +0.4  +49.1m 0.117+-0.186 0.68+-0.37  0.012  0.000  a
    Kepler-12_b          NO  1258s    6  1.0221   +0.03    +0.0  +45.3m 0.227+-0.250 0.65+-0.40  0.115  0.020  a
    HATS-61_b           yes   254s    2  1.0064   -1.31    -1.3  +76.9m 0.085+-0.153 0.66+-0.40  0.000  0.000  a
    Kepler-490_b        yes   139s    2  1.0055   +0.54    +0.5  +60.5m 0.220+-0.236 1.01+-0.37  0.000  0.000  a
    NGTS-13_b           yes   326s    3  1.0079   +0.21    +0.2 +169.3m 0.075+-0.065 0.40+-0.30  0.000  0.000  a
    Kepler-43_b         yes   855s    5  1.0060   +0.11    +0.1   +8.6m 0.108+-0.151 0.56+-0.37  0.000  0.000  a
    CoRoT-12_b          yes   277s    3  1.0061   -1.24    -1.2  +23.6m 0.151+-0.165 0.61+-0.39  0.000  0.000  a
    WASP-163_b          yes   805s    5  1.0057   -0.87    -0.9  -35.8m 0.141+-0.174 0.76+-0.39  0.045  0.000  a
    NGTS-5_b             NO  1263s    6  1.0164   -0.63    -0.6  -53.7m 0.182+-0.228 0.57+-0.37  0.088  0.000  a
    Kepler-12_b_rival    NO  1185s    6  1.0406   +1.33   +18.8 -591.3m 0.267+-0.255 0.63+-0.40  0.233  0.227  a
    Kepler-43_b_rival   yes   836s    5  1.0059   -0.43    +0.1   +8.7m 0.108+-0.149 0.56+-0.37  0.000  0.000  a
    WASP-163_b_rival     NO  1734s    6  1.0354   -1.09   -77.2  -28.4m 0.238+-0.227 0.73+-0.39  0.149  0.000  a
    NGTS-5_b_rival       NO  1459s    6  1.0117   +1.76    -6.6 -117.8m 0.193+-0.221 0.63+-0.38  0.237  0.000  a

`P-seed` and `P-true` are the posterior median's distance from the seed
and from the archive period, in posterior sd. `edgeP` and `edgeT` are the
shares of draws in the outer 10% of the box, both sides together.

- **10 of 13 archive-seeded fits converged.** The three that didn't stopped
  at the cap:
  - HAT-P-1 b: R-hat 1.015, 1.2% at the edge;
  - Kepler-12 b: R-hat 1.022, 11.5% at the edge, warned;
  - NGTS-5 b: R-hat 1.016, 8.8% at the edge toward its rival peak, warned.
- **Mass at the edge doesn't block convergence.** WASP-163 b converged with
  4.5% at the edge, and was warned.
- **Rival-seeded fits stay at their seed.** Kepler-43 b's "rival" is a lobe
  of the true peak, so its fit converged onto the true period. The other
  three:
  - stayed at their seeds (18.8, 77.2 and 6.6 sd from the true period);
  - did not converge (R-hat 1.012-1.041);
  - were warned, with 15-24% of their draws at the edge facing the comb.
- **For comparison, a 15-unit box** (same batch) converged 8 of 13. It
  pulled NGTS-5 b's rival-seeded fit onto the true period, against the rule
  that a fit stays at its seeded peak, so the box stayed at 5 units.

## Against the literature

`k` has no literature value for NGTS-13 b, nor `b` for NGTS-5 b. The `s`
column gives each band's fitted error scale (median), with the log's robust
chi out of transit in brackets.

             run conv                  P [s]               T0 [min]                T14 [h]                        k                        b                                      s (robust chi)    n
       KELT-16_b  yes -0.569 (-1.0sd, p0.43)  -2.11 (-0.4sd, p0.64) +0.165 (+0.7sd, p0.43) +0.00513 (+2.0sd, p0.09) +0.00927 (+0.0sd, p0.99) G 3.28, BP 3.02, RP 2.14 (G 2.57, BP 2.56, RP 2.26)  139
       WASP-18_b  yes  -2.91 (-3.0sd, p0.02)  +2.76 (+0.6sd, p0.50) +0.943 (+2.7sd, p0.00) -0.00574 (-0.6sd, p0.57)  +0.0179 (+0.1sd, p0.94) G 3.38, BP 1.94, RP 2.48 (G 3.40, BP 1.88, RP 1.93)  679
     HD_189733_b  yes  +15.2 (+1.9sd, p0.08)  +18.1 (+0.8sd, p0.37)  +2.18 (+1.8sd, p0.19)  +0.0338 (+0.4sd, p0.04) +0.00617 (+0.0sd, p0.99) G 9.65, BP 6.72, RP 7.95 (G 8.08, BP 5.73, RP 8.23) 1562
       WASP-12_b  yes +0.482 (+0.1sd, p0.91) +0.381 (+0.0sd, p0.98) -0.496 (-0.6sd, p0.46)  +0.0253 (+0.3sd, p0.07)    +0.28 (+0.7sd, p0.56) G 3.63, BP 3.97, RP 7.38 (G 3.38, BP 2.24, RP 5.74) -138
       HAT-P-1_b   NO  +24.2 (+0.5sd, p0.42)  +48.8 (+1.2sd, p0.28) -0.129 (-0.1sd, p0.93) -0.00113 (-0.0sd, p0.97)  -0.0712 (-0.2sd, p0.85) G 2.37, BP 3.07, RP 1.71 (G 1.68, BP 2.03, RP 1.89)  765
     Kepler-12_b   NO  +6.53 (+0.0sd, p0.97)  +44.9 (+0.1sd, p0.84) +0.526 (+0.2sd, p0.87)   +0.108 (+0.4sd, p0.28)   +0.481 (+1.1sd, p0.25) G 2.44, BP 2.17, RP 1.49 (G 1.88, BP 2.01, RP 1.22)  538
       HATS-61_b  yes  -76.4 (-1.2sd, p0.13)  +76.6 (+1.4sd, p0.02)  +1.42 (+0.6sd, p0.59)  +0.0113 (+0.2sd, p0.35)   +0.154 (+0.4sd, p0.76) G 3.26, BP 1.87, RP 1.82 (G 2.40, BP 1.47, RP 1.40)  -33
    Kepler-490_b  yes  +9.98 (+0.6sd, p0.57)  +60.2 (+1.5sd, p0.04)  +4.65 (+2.8sd, p0.01)   +0.125 (+0.5sd, p0.07)   +0.957 (+2.9sd, p0.02) G 1.89, BP 1.55, RP 1.12 (G 2.12, BP 1.18, RP 0.81)  756
       NGTS-13_b  yes  +4.45 (+0.2sd, p0.81)   +169 (+1.2sd, p0.18)  +6.46 (+1.4sd, p0.25)                        -   +0.244 (+0.8sd, p0.42) G 4.12, BP 1.85, RP 1.74 (G 4.47, BP 1.79, RP 1.98) -281
     Kepler-43_b  yes  +2.08 (+0.1sd, p0.89)  +8.24 (+0.4sd, p0.61) -0.714 (-0.7sd, p0.54)  +0.0217 (+0.4sd, p0.07)  -0.0877 (-0.2sd, p0.83) G 3.11, BP 1.52, RP 1.25 (G 2.46, BP 1.47, RP 1.44)  797
      CoRoT-12_b  yes  -7.45 (-1.3sd, p0.28)  +23.3 (+1.8sd, p0.36) -0.434 (-0.6sd, p0.56)  +0.0198 (+0.2sd, p0.13)  +0.0409 (+0.1sd, p0.92) G 1.53, BP 2.81, RP 1.37 (G 2.06, BP 2.45, RP 1.53) 1084
      WASP-163_b  yes  -13.6 (-1.0sd, p0.18)  -36.1 (-1.1sd, p0.28) +0.983 (+0.9sd, p0.36)  +0.0219 (+0.2sd, p0.24)   +0.312 (+0.8sd, p0.50) G 6.95, BP 3.10, RP 2.95 (G 7.79, BP 3.20, RP 2.86) -280
        NGTS-5_b   NO  -41.3 (-0.6sd, p0.63)    -54 (-0.8sd, p0.44)  +1.63 (+1.0sd, p0.36)  +0.0237 (+0.1sd, p0.78)                        - G 2.72, BP 2.38, RP 1.09 (G 4.15, BP 2.45, RP 0.76)  -96

- **Period and epoch.** All 26 are within 1.9 sd, except WASP-18 b's period
  (-2.9 s, 3.0 sd, p 0.02). Epochs 20 min to 3 h from the literature
  (NGTS-13 b +169 min, HATS-61 b +77, Kepler-490 b +61) are each within
  1.8 sd. With one or two points in transit per band, that is the width of
  the epoch posterior; it is not a bias.
- **Shape.** `k`, `b` and `T14` are mostly weak: k's sd is 0.065-0.25 on
  every target but KELT-16 b (0.003). Two disagree:
  - **WASP-18 b's `T14`** is 3.15 h against 2.21 (+0.94 h, 2.7 sd,
    p < 0.005);
  - **Kepler-490 b's posterior is grazing**: b 1.01 against 0.05 (2.9 sd,
    p 0.02), and T14 4.7 h long (2.8 sd, p 0.01).

  Both are single seeds. A stellar-density prior (`--rhostar`; Gaia's FLAME
  gives one) is the natural way to tie T14 and b down on sparse data, and
  was not used here.
- **In total**, 7 of 63 comparisons have p < 0.05 (3.2 expected) and 2 have
  p <= 0.01 (0.6 expected). The excess is in WASP-18 b and Kepler-490 b, as
  above.
- **Error scales.** The fitted `s` tracks the robust chi: the ratio's median
  is 1.08, 80% lie within 0.89-1.39, and the full range is 0.66-1.77
  (39 bands). Catalogue errors are 1.1-9.7x too small.

Not done here: the injection-coverage run, about 50 injections on DR3
cadences, which waits for the user's go-ahead.
