"""Acceptance: --ld=collapsed against the default (--ld=sampled).

For each parameter (P, tau0, k, b, T14, q1, q2, log10_rho), from the chains
tarballs:

- **shift**: ``(median_c - median_s) / sd_s``, and the same shift in units of
  its Monte Carlo standard error, ``sqrt(pi/2) sd sqrt(1/ESS_s + 1/ESS_c)``;
- **width**: ``sd_c / sd_s``;
- **shape**: two-sample KS distance, judged against a **calibrated null**
  (below), not against an arbitrary multiple of anything;
- **edge atoms**: count of ``q1`` or ``q2`` draws exactly at 0 or 1.

The KS null. Draws within a chain are autocorrelated (KOI-448.02 ``k``: +0.55
at one draw's lag), so splitting a run into random *rows* treats them as
independent and gives a reference that is not the null for two independent
runs. Instead, each run's chains are split at random into two halves, the KS
between the halves taken, and scaled by ``1/sqrt(2)``: two independent
half-runs carry half the ESS each, so their KS scales as ``2/sqrt(E)``
against ``sqrt(2/E)`` for two full runs. Doing this for *both* runs and
pooling the 2R values brackets the case where the two modes' ESS differ. A
parameter passes if its observed KS is at or below the pooled null's 95th
percentile; the observed value's percentile is printed.

Cost: end-to-end wall time (``done in``), which includes collapsed mode's
post-hoc conditional draws; sampling-round time alone is printed too.
ESS/s uses the minimum bulk ESS over the five parameters **both** modes
sample (dP, dtau0, k, beta, T14), never q1, q2, which collapsed mode does
not sample.

Expects the run directories beside it, as the acceptance runs laid them out:
``518/`` and ``448/`` (each with ``--tag=s`` and ``--tag=c`` products and the
``/usr/bin/time -l`` run logs ``<n>_s.log``, ``<n>_c.log``), and ``518old/``
(the sampled KOI-518.02 run on 0.1.38). See ``collapsed_ld_acceptance.md``.
"""
import glob
import hashlib
import math
import os
import re
import tarfile

import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = ["dP", "dtau0", "k", "b", "T14", "q1", "q2", "log10_rho"]
SHARED_SAMPLED = ["dP", "dtau0", "k", "beta", "T14"]
LABEL = {"dP": "P", "dtau0": "tau0"}
N_CHAINS = 512
N_SPLITS = 40


def read_chains(path):
    """``(columns, draws, body_sha)``; rows are draw-major, chain = row % C."""
    with tarfile.open(path) as t:
        raw = t.extractfile(t.getmembers()[0]).read()
    lines = raw.decode().splitlines()
    draws = np.loadtxt(lines[2:], delimiter=",")
    sha = hashlib.sha256("\n".join(lines[2:]).encode()).hexdigest()
    return lines[0].split(","), draws, sha


def read_summary(path):
    out = {}
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            if not line.startswith("#"):
                row = dict(zip(header, line.rstrip("\n").split(",")))
                out[row["Parameter"]] = row
    return out


def read_log(path):
    txt = open(path).read()
    rounds = [float(x) for x in re.findall(r"round \d+ took ([\d.]+)s", txt)]
    done = re.search(r"done in ([\d.]+)s", txt)
    rss = re.search(r"(\d+)\s+maximum resident set size", txt)
    acc = re.findall(r"MH acceptance ([\d.]+); expansion point on a triangle "
                     r"edge for (\d+)%", txt)
    return dict(sampling=sum(rounds), rounds=len(rounds),
                wall=float(done.group(1)), rss_gb=int(rss.group(1)) / 1e9,
                acc=acc[-1] if acc else None)


def ks_null(x, rng):
    """KS between random chain-halves of one run, scaled to full-run size."""
    chain = np.arange(len(x)) % N_CHAINS
    out = []
    for _ in range(N_SPLITS):
        half = rng.permutation(N_CHAINS) < N_CHAINS // 2
        sel = half[chain]
        out.append(ks_2samp(x[sel], x[~sel]).statistic / math.sqrt(2.0))
    return np.array(out)


def ess(summary, name):
    v = summary.get(name, {}).get("Bulk_ESS", "")
    return float(v) if v else None


def compare(tgt, d, stem):
    cs, s, _ = read_chains(f"{d}/{tgt}_lineph_chains.s.csv.tar.gz")
    cc, c, _ = read_chains(f"{d}/{tgt}_lineph_chains.c.csv.tar.gz")
    sum_s = read_summary(f"{d}/{tgt}_lineph_summary.s.csv")
    sum_c = read_summary(f"{d}/{tgt}_lineph_summary.c.csv")
    rng = np.random.default_rng(0)
    print(f"\n### {tgt}: {len(s)} sampled draws, {len(c)} collapsed")
    print(f"{'param':10s} {'shift/sd':>9s} {'shift/se':>9s} {'width':>7s} "
          f"{'KS':>7s} {'null95':>7s} {'pctile':>7s}  verdict")
    for p in PARAMS:
        a, b = s[:, cs.index(p)], c[:, cc.index(p)]
        shift = (np.median(b) - np.median(a)) / a.std()
        e_s = ess(sum_s, p)
        e_c = ess(sum_c, p) or e_s            # collapsed q1, q2: assume equal
        se = (math.sqrt(math.pi / 2) * math.sqrt(1 / e_s + 1 / e_c)
              if e_s else float("nan"))
        width = b.std() / a.std()
        ks = ks_2samp(a, b).statistic
        null = np.concatenate([ks_null(a, rng), ks_null(b, rng)])
        q95 = np.quantile(null, 0.95)
        pct = 100 * np.mean(null <= ks)
        ok = abs(shift) <= 0.03 and 0.97 <= width <= 1.03 and ks <= q95
        print(f"{LABEL.get(p, p):10s} {shift:+9.3f} {shift / se:+9.2f} "
              f"{width:7.3f} {ks:7.4f} {q95:7.4f} {pct:6.0f}%  "
              f"{'ok' if ok else 'FAIL'}")
    edge = int(sum(np.sum((c[:, cc.index(q)] <= 0) | (c[:, cc.index(q)] >= 1))
                   for q in ("q1", "q2")))
    print(f"collapsed q draws exactly at 0 or 1: {edge}")
    q1 = c[:, cc.index("q1")]
    print("collapsed q1 draws within d of 1, d = 1e-2, 1e-3, 1e-4, 1e-6: "
          + ", ".join(str(int(np.sum(q1 > 1 - d))) for d in (1e-2, 1e-3, 1e-4, 1e-6))
          + f"; closest {1 - q1.max():.1e}")
    rows = {}
    for tag, sm in (("s", sum_s), ("c", sum_c)):
        lg = read_log(f"{d}/{stem}_{tag}.log")
        m = min(ess(sm, n) for n in SHARED_SAMPLED)
        rows[tag] = m / lg["wall"]
        print(f"  {'sampled  ' if tag == 's' else 'collapsed'}: {lg['rounds']} "
              f"round(s); wall {lg['wall']:.0f} s (sampling {lg['sampling']:.0f}"
              f" s); min bulk ESS over the shared 5 = {m:.0f} -> "
              f"{m / lg['wall']:.1f} ESS/s; peak process RSS {lg['rss_gb']:.1f} GB"
              + (f"; MH acceptance {lg['acc'][0]}, x* on a triangle edge "
                 f"{lg['acc'][1]}%" if lg["acc"] else ""))
    print(f"  collapsed takes {rows['s'] / rows['c']:.2f}x the wall time per "
          "effective sample")


if __name__ == "__main__":
    for tgt, d, stem in (("KOI-518.02", f"{HERE}/518", "518"),
                         ("KOI-448.02", f"{HERE}/448", "448")):
        if glob.glob(f"{d}/{tgt}_lineph_chains.c.csv.tar.gz"):
            compare(tgt, d, stem)
    old = f"{HERE}/518old/KOI-518.02_lineph_chains.s.csv.tar.gz"
    new = f"{HERE}/518/KOI-518.02_lineph_chains.s.csv.tar.gz"
    if os.path.exists(old) and os.path.exists(new):
        same = read_chains(old)[2] == read_chains(new)[2]
        print(f"\nsampled KOI-518.02 chains body (lines 3 on), 0.1.38 vs 0.1.42: "
              f"{'BYTE-IDENTICAL' if same else 'DIFFER'}")
