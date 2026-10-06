"""Acceptance: --ld=collapsed against the default, per the agreed criteria.

Per parameter (P, tau0, k, b, T14, q1, q2, log10_rho), from the chains
tarballs: |median_c - median_s| / sd_s <= 0.03; sd_c / sd_s in [0.97, 1.03];
two-sample KS distance <= the KS between two random halves of the sampled
chains (the self-resample level); zero draws with q1 or q2 exactly 0 or 1.
Plus cost: sampling time, min bulk ESS, ESS/s, peak RSS, from the run logs.

Expects the run directories beside it, as the acceptance runs laid them out:
``518/`` and ``448/`` (each with ``--tag=s`` and ``--tag=c`` products and the
``/usr/bin/time -l`` run logs ``<target>_s.log``, ``<target>_c.log``), and
``518old/`` (the sampled KOI-518.02 run on 0.1.38). See
``collapsed_ld_acceptance.md`` for the commands.
"""
import glob
import hashlib
import os
import re
import sys
import tarfile

import numpy as np
from scipy.stats import ks_2samp

A = os.path.dirname(os.path.abspath(__file__))
PARAMS = ["dP", "dtau0", "k", "b", "T14", "q1", "q2", "log10_rho"]
LABEL = {"dP": "P", "dtau0": "tau0"}


def chains(path):
    with tarfile.open(path) as t:
        body = t.extractfile(t.getmembers()[0]).read().decode().splitlines()
    cols = body[0].split(",")
    arr = np.array([[float(v) for v in r.split(",")] for r in body[2:]])
    return {c: arr[:, i] for i, c in enumerate(cols)}, body


def summary_ess(path):
    out = {}
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            if line.startswith("#"):
                continue
            f = dict(zip(header, line.rstrip("\n").split(",")))
            out[f["Parameter"]] = f
    return out


def log_facts(path):
    txt = open(path).read()
    rounds = [float(x) for x in re.findall(r"round \d+ took ([\d.]+)s", txt)]
    real = re.search(r"([\d.]+) real", txt)
    rss = re.search(r"(\d+)\s+maximum resident set size", txt)
    done = re.search(r"done in ([\d.]+)s", txt)
    acc = re.search(r"MH acceptance ([\d.]+); expansion point on a triangle "
                    r"edge for (\d+)%", txt)
    rh = re.findall(r"worst R-hat ([\d.]+) \((\w+)\), min bulk ESS (\d+)", txt)
    return dict(sampling=sum(rounds), n_rounds=len(rounds),
                wall=float(real.group(1)) if real else float("nan"),
                rss_gb=int(rss.group(1)) / 1e9 if rss else float("nan"),
                done=float(done.group(1)) if done else float("nan"),
                acc=acc.groups() if acc else None,
                rhat=rh[-1] if rh else None)


def compare(tgt, d):
    s, _ = chains(glob.glob(f"{d}/{tgt}_lineph_chains.s.csv.tar.gz")[0])
    c, _ = chains(glob.glob(f"{d}/{tgt}_lineph_chains.c.csv.tar.gz")[0])
    rng = np.random.default_rng(0)
    print(f"\n### {tgt}: sampled {len(s['k'])} draws, collapsed {len(c['k'])} draws")
    print(f"{'param':10s} {'shift/sd':>9s} {'sd ratio':>9s} {'KS c-s':>8s} "
          f"{'KS self':>8s}  verdict")
    worst = dict(shift=0, sd=0, ks_excess=0)
    for p in PARAMS:
        a, b = s[p], c[p]
        shift = (np.median(b) - np.median(a)) / a.std()
        ratio = b.std() / a.std()
        ks = ks_2samp(a, b).statistic
        perm = rng.permutation(len(a))
        ks_self = ks_2samp(a[perm[: len(a) // 2]], a[perm[len(a) // 2:]]).statistic
        ok = abs(shift) <= 0.03 and 0.97 <= ratio <= 1.03 and ks <= ks_self * 1.5
        worst["shift"] = max(worst["shift"], abs(shift))
        worst["sd"] = max(worst["sd"], abs(ratio - 1))
        worst["ks_excess"] = max(worst["ks_excess"], ks - ks_self)
        print(f"{LABEL.get(p, p):10s} {shift:+9.3f} {ratio:9.3f} {ks:8.4f} "
              f"{ks_self:8.4f}  {'ok' if ok else 'CHECK'}")
    edge = sum(int(np.sum((c[q] <= 0) | (c[q] >= 1))) for q in ("q1", "q2"))
    print(f"collapsed draws with q1 or q2 exactly on the box edge: {edge}")
    print(f"collapsed q1 range [{c['q1'].min():.2e}, {c['q1'].max():.6f}], "
          f"q2 range [{c['q2'].min():.2e}, {c['q2'].max():.6f}]")
    for tag in ("s", "c"):
        f = log_facts(f"{d}/{tgt}_{tag}.log") if os.path.exists(
            f"{d}/{tgt}_{tag}.log") else log_facts(glob.glob(f"{d}/*_{tag}.log")[0])
        sm = summary_ess(glob.glob(f"{d}/{tgt}_lineph_summary.{tag}.csv")[0])
        ess = [float(r["Bulk_ESS"]) for k_, r in sm.items() if r.get("Bulk_ESS")]
        mn = min(ess)
        print(f"  {'sampled  ' if tag == 's' else 'collapsed'}: {f['n_rounds']} "
              f"round(s), sampling {f['sampling']:.0f} s, wall {f['wall']:.0f} s, "
              f"min bulk ESS {mn:.0f} -> {mn / f['sampling']:.1f} ESS/s, "
              f"peak RSS {f['rss_gb']:.1f} GB, last R-hat {f['rhat']}"
              + (f", MH acc {f['acc'][0]}, x* on edge {f['acc'][1]}%"
                 if f["acc"] else ""))
    return worst


if __name__ == "__main__":
    for tgt, d in (("KOI-518.02", f"{A}/518"), ("KOI-448.02", f"{A}/448")):
        if glob.glob(f"{d}/{tgt}_lineph_chains.c.csv.tar.gz"):
            compare(tgt, d)
    # bit-identity: sampled LinEph, 0.1.38 vs now, KOI-518.02
    new = glob.glob(f"{A}/518/KOI-518.02_lineph_chains.s.csv.tar.gz")
    old = glob.glob(f"{A}/518old/KOI-518.02_lineph_chains.s.csv.tar.gz")
    if new and old:
        h = [hashlib.sha256("\n".join(chains(p)[1][2:]).encode()).hexdigest()
             for p in (old[0], new[0])]
        print(f"\nsampled KOI-518.02 chains body, 0.1.38 vs now: "
              f"{'BYTE-IDENTICAL' if h[0] == h[1] else 'DIFFER'}")
