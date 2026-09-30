"""Side-by-side hurin/turin comparison for one target.

Reads each package's own summary CSVs (same filenames and columns by design)
and reports medians in units of the combined uncertainty, plus the width
ratio, ESS and ESS per second.
"""
import os
import sys

import numpy as np

BENCH = os.path.dirname(os.path.abspath(__file__))
TARGET = "KOI-518.02"

# hurin's parameter names vs turin's, where they differ. turin samples
# offsets from the recentred ephemeris and reports absolutes under the same
# names, so P/tau0 line up; the rest are identical.
ALIAS = {"P": "dP", "tau0": "dtau0"}


def load(path):
    """Parse a summary CSV by hand.

    Both packages write ``Parameter,Median,Std,16th,84th,R-hat,Bulk_ESS,
    Tail_ESS``, but hurin's per-epoch Legendre rows carry an extra epoch
    field, so those lines have nine columns. Keep only the eight-column
    physical-parameter rows; the coefficients are nuisance parameters that
    turin never samples and so cannot be compared anyway.
    """
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            f = line.split(",")
            if len(f) != len(header):
                continue
            row = {}
            for k, v in zip(header[1:], f[1:]):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = float("nan")
            out[f[0]] = row
    return out


def col(d, key, *alts):
    for k in (key, *alts):
        if k in d:
            return d[k]
    return float("nan")


def find(d, who, kind):
    """Locate a product under either layout.

    A live run writes ``<dir>/<TARGET>/<TARGET>_<kind>.csv``; this directory
    holds them flattened and tagged, ``<TARGET>_<who>_<kind>.csv``.
    """
    for cand in (os.path.join(d, TARGET, f"{TARGET}_{kind}.csv"),
                 os.path.join(d, f"{TARGET}_{who}_{kind}.csv")):
        if os.path.exists(cand):
            return cand
    return os.path.join(d, TARGET, f"{TARGET}_{kind}.csv")


def compare(mode, h_dir, t_dir, h_secs=None, t_secs=None):
    h = load(find(h_dir, "hurin", f"{mode}_summary"))
    t = load(find(t_dir, "turin", f"{mode}_summary"))
    if h is None or t is None:
        print(f"[{mode}] missing summary (hurin={h is not None}, "
              f"turin={t is not None})")
        return
    print(f"\n### {mode}")
    print(f"{'param':10s} {'hurin':>26s} {'turin':>26s} "
          f"{'shift':>8s} {'sd_t/sd_h':>10s} {'ESS h':>8s} {'ESS t':>9s}")
    for hname, row in h.items():
        tname = ALIAS.get(hname, hname)
        if tname not in t:
            continue
        trow = t[tname]
        mh, sh = row["Median"], row["Std"]
        mt, st = trow["Median"], trow["Std"]
        denom = np.hypot(sh, st) or float("nan")
        eh = col(row, "Bulk_ESS")
        et = col(trow, "Bulk_ESS")
        print(f"{hname:10s} {mh:14.7g} +- {sh:8.2g} {mt:14.7g} +- {st:8.2g} "
              f"{(mt - mh) / denom:7.2f}s {st / sh:10.2f} {eh:8.0f} {et:9.0f}")
    if h_secs and t_secs:
        eh = np.nanmin([col(r, "Bulk_ESS") for r in h.values()])
        et = np.nanmin([col(r, "Bulk_ESS") for r in t.values()])
        print(f"  wall  hurin {h_secs:8.1f}s   turin {t_secs:8.1f}s   "
              f"speedup {h_secs / t_secs:5.1f}x")
        print(f"  min bulk ESS/s  hurin {eh / h_secs:8.2f}   "
              f"turin {et / t_secs:8.2f}   ratio {(et / t_secs) / (eh / h_secs):5.1f}x")


def ttv_times(h_dir, t_dir):
    ph = find(h_dir, "hurin", "ttv_times")
    pt = find(t_dir, "turin", "ttv_times")
    if not (os.path.exists(ph) and os.path.exists(pt)):
        return
    a = np.atleast_1d(np.genfromtxt(ph, delimiter=",", names=True, dtype=None,
                                    encoding="utf-8"))
    b = np.atleast_1d(np.genfromtxt(pt, delimiter=",", names=True, dtype=None,
                                    encoding="utf-8"))
    common = np.intersect1d(a["epoch"], b["epoch"])
    ia = np.searchsorted(a["epoch"], common)
    ib = np.searchsorted(b["epoch"], common)
    dt = (b["tmid"][ib] - a["tmid"][ia])
    den = np.hypot(a["tmid_err"][ia], b["tmid_err"][ib])
    ratio = a["tmid_err"][ia] / b["tmid_err"][ib]
    print(f"\n### transit times ({common.size} common epochs of "
          f"{a.size} hurin / {b.size} turin)")
    print(f"  |shift| median {np.median(np.abs(dt / den)):.2f} sigma, "
          f"max {np.abs(dt / den).max():.2f} sigma")
    print(f"  turin tighter by: median {np.median(ratio):.2f}x, "
          f"range {ratio.min():.2f}-{ratio.max():.2f}x")
    worst = np.argmax(np.abs(dt / den))
    print(f"  largest shift at epoch {int(common[worst])}: "
          f"{a['tmid'][ia][worst]:.5f} +- {a['tmid_err'][ia][worst]:.5f} vs "
          f"{b['tmid'][ib][worst]:.5f} +- {b['tmid_err'][ib][worst]:.5f}")


if __name__ == "__main__":
    h_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BENCH, "hurin")
    t_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BENCH, "turin")
    hs = float(sys.argv[3]) if len(sys.argv) > 3 else None
    ts = float(sys.argv[4]) if len(sys.argv) > 4 else None
    for mode in ("lineph", "ttv"):
        compare(mode, h_dir, t_dir, hs, ts)
    ttv_times(h_dir, t_dir)
