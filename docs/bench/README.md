# Benchmark and acceptance artefacts

Two records live here.

- `collapsed_ld_acceptance.md` and `collapsed_ld_compare.py`: `--ld=collapsed`
  against the default on KOI-518.02 and KOI-448.02 (see that file; the
  chains tarballs are not kept, being hundreds of MB, so the script reruns
  only against freshly made runs).
- Everything below: the hurin/turin comparison on KOI-518.02.

## KOI-518.02 hurin/turin benchmark artefacts

The summaries behind the primary comparison in `../hurin-differences.md`,
kept so the numbers quoted there can be checked without re-running two fits.
Only the summary and transit-time CSVs are here; the chain tarballs are not
(tens of MB, and the summaries carry everything the comparison uses).

Each file records its own provenance on line 2 — package version and the
exact command.

    KOI-518.02_hurin_lineph_summary.csv   hurin 0.1.68, 8 NUTS chains
    KOI-518.02_hurin_ttv_summary.csv
    KOI-518.02_hurin_ttv_times.csv
    KOI-518.02_turin_lineph_summary.csv   turin 0.1.12, 512 ChEES chains
    KOI-518.02_turin_ttv_summary.csv
    KOI-518.02_turin_ttv_times.csv

## Re-running the comparison

    .venv/bin/python docs/bench/compare.py docs/bench docs/bench 6037.17 543.71

The two directory arguments are where to find hurin's and turin's products;
the two numbers are the wall-clock seconds each took, used only for the ESS/s
rows. `compare.py` reads either this flattened layout or a live run's
`<dir>/KOI-518.02/` layout, so the same script works on fresh output.

## Reproducing the fits

    # hurin: 8 chains, not its default of 2 -- see hurin-differences.md
    cd <scratch>/hurin && /usr/bin/time -l \
      <hurin-env>/bin/python -u <hurin>/pyhurin.py --KOI-518.02 --chains=8 --fresh

    # turin: defaults
    cd <scratch>/turin && /usr/bin/time -l \
      .venv/bin/python -u -m turin.cli --KOI-518.02 --fresh

Run them **sequentially** — hurin saturates about five cores, and overlapping
the two would flatter turin. Both should read a cached light curve; confirm
`../hurin/cache/KOI-518.02.pkl` and `~/.cache/turin/KOI-518.02.pkl` have the
same byte size first, so the fits see identical input.

Note that `Bulk_ESS` is blank for `b` and `log10_rho`: both are derived in
float64 after sampling rather than sampled, so the ESS columns compared are
the sampled parameters only.
