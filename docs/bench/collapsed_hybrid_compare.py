"""Collapsed hybrid4 (--law=hybrid4 --ld=collapsed, tag h4c) against sampled
hybrid4 (tag h4s), LinEph, on the user's five-target suite.

Method as docs/bench/collapsed_ld_battery_compare.py (0.1.44's rules):

- cost: end-to-end wall time ("done in", light curves pre-cached so neither
  mode pays a download; includes collapsed mode's post-hoc q draws), rounds,
  peak RSS, ESS/s on the minimum bulk ESS over the five parameters both
  modes sample (dP, dtau0, k, beta, T14);
- accuracy, from the chains: per parameter the median shift in sampled-mode
  sd and in Monte Carlo standard errors, the width ratio (sd and 16-84%),
  and the two-sample KS distance judged against a chain-split null (each
  run's chains split into random halves 40 times, scaled by 1/sqrt 2,
  pooled; pass at or below the null's 95th percentile); q draws exactly on
  0 or 1.

Run layout: <dir>/out/ holds both modes' products (--outdir), tagged h4s
(sampled) and h4c2 (collapsed, 0.1.64); <dir>/<TARGET>_<tag>.log are the
/usr/bin/time -l logs. See collapsed_hybrid_acceptance.md.

usage: collapsed_hybrid_compare.py <dir> [collapsed tag, default h4c2]
"""
import io
import os
import re
import sys
import tarfile

import numpy as np
import pandas as pd

TARGETS = ["KOI-4016.01", "KOI-4087.01", "KOI-1430.03", "KOI-1361.01",
           "KOI-2650.01"]
N_CHAINS = 512
SHARED = ["dP", "dtau0", "k", "beta", "T14"]
PARAMS = ["dP", "dtau0", "k", "b", "T14", "log10_rho",
          "q1", "q2", "q3", "q4", "w1", "w2", "w3", "w4"]


def summary(path):
    out = {}
    for row in open(path).read().splitlines()[2:]:
        f = row.split(",")
        out[f[0]] = dict(median=float(f[1]), sd=float(f[2]),
                         rhat=float(f[5]) if f[5] else np.nan,
                         ess=float(f[6]) if f[6] else np.nan)
    return out


def chains(path):
    tf = tarfile.open(path)
    body = tf.extractfile(tf.getmembers()[0]).read().decode()
    return pd.read_csv(io.StringIO(body), comment="#")


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / a.size
                               - np.searchsorted(b, grid, "right") / b.size)))


def split_null(x, rng, reps=40):
    chain = np.arange(x.size) % N_CHAINS
    out = []
    for _ in range(reps):
        half = rng.permutation(N_CHAINS)[:N_CHAINS // 2]
        m = np.isin(chain, half)
        out.append(ks(x[m], x[~m]) / np.sqrt(2.0))
    return out


def log_facts(path):
    txt = open(path).read()
    done = re.findall(r"done in ([0-9.]+)s", txt)
    rounds = re.findall(r"round (\d+) took ([0-9.]+)s", txt)
    rss = re.search(r"(\d+)\s+maximum resident set size", txt)
    conv = re.search(r"lineph: (converged at \d+|.*?) draws/chain", txt)
    acc = re.search(r"MH acceptance ([0-9.]+); expansion point on (.*?) for "
                    r"(\d+)% of draws", txt)
    trapped = re.findall(r"trapped|frozen|lost", txt, re.I)
    return dict(wall=float(done[-1]) if done else np.nan,
                rounds=len(rounds),
                sampling_s=sum(float(t) for _, t in rounds),
                rss_GB=int(rss.group(1)) / 2 ** 30 if rss else np.nan,
                converged=bool(re.search(r"lineph: converged", txt)),
                mh_acc=float(acc.group(1)) if acc else np.nan,
                face_pct=int(acc.group(3)) if acc else np.nan,
                health_words=len(trapped))


def main(D, sfx="", ctag="h4c2"):
    out = f"{D}/out"
    rng = np.random.default_rng(0)
    perf, accu = [], []
    for T in TARGETS:
        tags = {"sampled": "h4s" + sfx, "collapsed": ctag + sfx}
        have = {m: os.path.exists(f"{out}/{T}/{T}_lineph_chains.{t}.csv.tar.gz")
                or os.path.exists(f"{out}/{T}_lineph_chains.{t}.csv.tar.gz")
                for m, t in tags.items()}
        if not all(have.values()):
            continue
        base = f"{out}/{T}" if os.path.isdir(f"{out}/{T}") else out
        S, C = {}, {}
        for m, t in tags.items():
            sm = summary(f"{base}/{T}_lineph_summary.{t}.csv")
            lf = log_facts(f"{D}/{T}_{t}.log")
            ess = min(sm[p]["ess"] for p in SHARED)
            rhat = max(sm[p]["rhat"] for p in SHARED)
            perf.append(dict(target=T, mode=m, rounds=lf["rounds"],
                             converged=lf["converged"],
                             wall_s=round(lf["wall"], 1),
                             sampling_s=round(lf["sampling_s"], 1),
                             min_ess=int(ess),
                             ess_per_s=round(ess / lf["wall"], 2),
                             worst_rhat=round(rhat, 4),
                             rss_GB=round(lf["rss_GB"], 2),
                             mh_acc=lf["mh_acc"], face_pct=lf["face_pct"]))
            S[m] = sm
            C[m] = chains(f"{base}/{T}_lineph_chains.{t}.csv.tar.gz")
        cs, cc = C["sampled"], C["collapsed"]
        gs = np.mean(cs.b > 1 - cs.k)
        gc = np.mean(cc.b > 1 - cc.k)
        for p in PARAMS:
            xs, xc = cs[p].to_numpy(), cc[p].to_numpy()
            sd = np.std(xs)
            e_s = S["sampled"].get(p, {}).get("ess", np.nan)
            e_c = S["collapsed"].get(p, {}).get("ess", np.nan)
            if not np.isfinite(e_c):       # q, w, b, rho: no ESS in collapsed
                e_c = e_s
            mcse = (np.sqrt(np.pi / 2) * sd * np.sqrt(1 / e_s + 1 / e_c)
                    if np.isfinite(e_s) else np.nan)
            d = np.median(xc) - np.median(xs)
            null = split_null(xs, rng) + split_null(xc, rng)
            k = ks(xs, xc)
            iq = lambda x: np.subtract(*np.percentile(x, [84, 16]))
            accu.append(dict(
                target=T, param=p, shift_sd=round(d / sd, 4) if sd else np.nan,
                shift_mcse=round(d / mcse, 2) if mcse else np.nan,
                width_sd=round(np.std(xc) / sd, 3) if sd else np.nan,
                width_1684=round(iq(xc) / iq(xs), 3) if iq(xs) else np.nan,
                ks=round(k, 4), ks_null95=round(np.percentile(null, 95), 4),
                ks_pass=bool(k <= np.percentile(null, 95)),
                edge=int(np.sum((xc <= 0) | (xc >= 1)))
                if p.startswith("q") else 0,
                edge_sampled=int(np.sum((xs <= 0) | (xs >= 1)))
                if p.startswith("q") else 0))
        accu.append(dict(target=T, param="grazing_frac",
                         shift_sd=round(gc - gs, 4), ks=np.nan,
                         width_sd=np.nan, ks_pass=True,
                         ks_null95=np.nan, edge=0, edge_sampled=0,
                         shift_mcse=np.nan, width_1684=round(gs, 4)))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    P = pd.DataFrame(perf)
    A = pd.DataFrame(accu)
    P.to_csv(f"{D}/perf_{ctag}{sfx}.csv", index=False)
    A.to_csv(f"{D}/accuracy_{ctag}{sfx}.csv", index=False)
    print(P.to_string(index=False))
    if len(P):
        pv = P.pivot(index="target", columns="mode",
                     values=["wall_s", "ess_per_s"])
        r = pv["ess_per_s"]["collapsed"] / pv["ess_per_s"]["sampled"]
        t = pv["wall_s"]["collapsed"] / pv["wall_s"]["sampled"]
        print("\ncollapsed / sampled, per target:")
        print(pd.DataFrame(dict(wall=t.round(2), ess_per_s=r.round(2))).to_string())
    print()
    print(A.to_string(index=False))
    a = A[A.param != "grazing_frac"]
    if len(a):
        print(f"\n{a.target.nunique()} targets, {len(a)} comparisons: "
              f"|shift| median {a.shift_sd.abs().median():.4f} sd, max "
              f"{a.shift_sd.abs().max():.4f} sd; |shift| > 3 MCSE: "
              f"{int((a.shift_mcse.abs() > 3).sum())}; width (16-84) "
              f"{a.width_1684.min():.3f}-{a.width_1684.max():.3f}; KS pass "
              f"{int(a.ks_pass.sum())}/{len(a)}; q on a wall: collapsed "
              f"{int(a.edge.sum())}, sampled {int(a.edge_sampled.sum())}")


if __name__ == "__main__":
    main(sys.argv[1], "", sys.argv[2] if len(sys.argv) > 2 else "h4c2")
