"""--ld=collapsed (0.1.45, tag ldc) against the default --ld=sampled LinEph fit,
rows 1-20 of Kepler_targets.csv.

Baselines: rows 1-10 from the 0.1.34 battery (tag v034), rows 11-20 from the
0.1.38 run (untagged). The default LinEph sampling path is unchanged since
(0.1.39 recorded it byte-identical before/after collapsed LD was added).

Per target, LinEph only (the baselines also ran TTV, which is excluded by
using the log's timestamps from the "lineph:" header to the next mode):

- cost: LinEph wall time (sampling + exports + certificate, and for
  collapsed the post-hoc q draws), rounds, ESS/s on the minimum bulk ESS over
  the five parameters both modes sample (dP, dtau0, k, beta, T14);
- memory: peak process RSS and GPU allocation inside the LinEph window;
- accuracy, from the chains: per parameter (P, tau0, k, b, T14, q1, q2,
  log10_rho) the median shift in baseline sd and in Monte Carlo standard
  errors, the width ratio, and the two-sample KS distance judged against a
  chain-split null (both runs' chains split into random halves 40 times,
  scaled by 1/sqrt 2, pooled; pass at or below the null's 95th percentile),
  as in docs/bench/collapsed_ld_compare.py; plus q draws exactly at 0 or 1.

Writes ldc_compare_perf.csv and ldc_compare_accuracy.csv into benchmarks/.
Results and the run layout it expects: collapsed_ld_battery.md.

usage: collapsed_ld_battery_compare.py [perf|accuracy|all]
       (run from the repository root)
"""
import glob
import io
import os
import re
import sys
import tarfile

import numpy as np
import pandas as pd

B = "benchmarks"
N_CHAINS = 512
SHARED = ["dP", "dtau0", "k", "beta", "T14"]
PARAMS = ["dP", "dtau0", "k", "b", "T14", "q1", "q2", "log10_rho"]


def targets():
    csv = pd.read_csv("Kepler_targets.csv").iloc[:20]
    return [f"KOI-{int(p[1:6])}.{p.split('.')[1]}" for p in csv.planet_id]


def base_tag(i):
    return "v034" if i < 10 else ""


def run_dir(T, tag):
    pat = f"{B}/*_{T}_{tag}" if tag else f"{B}/2026-10-0[56]_{T}"
    hits = sorted(d for d in glob.glob(pat) if os.path.exists(f"{d}/wall.txt"))
    return hits[-1] if hits else None


def secs(hms):
    h, m, s = map(int, hms.split(":"))
    return 3600 * h + 60 * m + s


def lineph_window(log):
    """(start, end) seconds-of-day of the LinEph fit, and its rounds."""
    lines = open(log).read().splitlines()
    start = end = None
    rounds = 0
    for l in lines:
        if re.search(r"\] lineph: \d+ epochs", l):
            start = secs(l[:8])
        elif start is not None and end is None:
            if re.search(r"\] ttv: \d+ epochs", l) or "done in" in l:
                end = secs(l[:8])
            m = re.search(r"round (\d+) took", l)
            if m:
                rounds = int(m.group(1)) + 1
    if end is None:
        end = secs(lines[-1][:8])
    return start, end, rounds


def peaks(d, start, end):
    """Peak RSS and GPU allocation inside the LinEph window."""
    mon = pd.read_csv(f"{d}/monitor.csv")
    first = secs(open(f"{d}/turin.log").readline()[:8])
    lo, hi = (start - first) % 86400, (end - first) % 86400
    w = mon[(mon.t_s >= lo) & (mon.t_s <= hi)]
    if w.empty:
        w = mon
    return w.proc_rss_MB.max() / 1024, w.gpu_alloc_MB.max() / 1024, int(w.pressure_level.max())


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
    """KS between random chain halves, scaled to two full runs."""
    chain = np.arange(x.size) % N_CHAINS
    out = []
    for _ in range(reps):
        half = rng.permutation(N_CHAINS)[:N_CHAINS // 2]
        m = np.isin(chain, half)
        out.append(ks(x[m], x[~m]) / np.sqrt(2.0))
    return out


def perf():
    rows = []
    for i, T in enumerate(targets()):
        db, dc = run_dir(T, base_tag(i)), run_dir(T, "ldc")
        for label, d, tag in (("sampled", db, base_tag(i)), ("collapsed", dc, "ldc")):
            if d is None:
                rows.append(dict(target=T, mode=label, note="not run"))
                continue
            s0, s1, rounds = lineph_window(f"{d}/turin.log")
            wall = (s1 - s0) % 86400
            rss, gpu, press = peaks(d, s0, s1)
            sfx = f".{tag}" if tag else ""
            sm = summary(f"{T}/{T}_lineph_summary{sfx}.csv")
            ess = min(sm[p]["ess"] for p in SHARED)
            rhat = max(sm[p]["rhat"] for p in SHARED)
            rows.append(dict(target=T, mode=label, rounds=rounds,
                             lineph_min=round(wall / 60, 1),
                             min_ess=int(ess), ess_per_s=round(ess / wall, 1),
                             worst_rhat=round(rhat, 4), rss_GB=round(rss, 1),
                             gpu_GB=round(gpu, 1), pressure=press))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    df.to_csv(f"{B}/ldc_compare_perf.csv", index=False)
    done = df.dropna(subset=["lineph_min"])
    both = done.groupby("target").filter(lambda g: len(g) == 2)
    if len(both):
        p = both.pivot(index="target", columns="mode",
                       values=["lineph_min", "ess_per_s", "rss_GB", "gpu_GB"])
        print(f"\n{len(p)} targets with both modes:")
        for m in ("sampled", "collapsed"):
            print(f"  {m:9s}: LinEph total {p['lineph_min'][m].sum():.0f} min, "
                  f"peak RSS {p['rss_GB'][m].max():.1f} GB, peak GPU "
                  f"{p['gpu_GB'][m].max():.1f} GB")
        r = p["ess_per_s"]["collapsed"] / p["ess_per_s"]["sampled"]
        t = p["lineph_min"]["collapsed"] / p["lineph_min"]["sampled"]
        print(f"  collapsed / sampled: wall time median {t.median():.2f}x "
              f"(range {t.min():.2f}-{t.max():.2f}); ESS/s median "
              f"{r.median():.2f}x (range {r.min():.2f}-{r.max():.2f})")


def accuracy():
    rng = np.random.default_rng(0)
    rows = []
    for i, T in enumerate(targets()):
        tag = base_tag(i)
        pb = f"{T}/{T}_lineph_chains{('.' + tag) if tag else ''}.csv.tar.gz"
        pc = f"{T}/{T}_lineph_chains.ldc.csv.tar.gz"
        if not (os.path.exists(pb) and os.path.exists(pc)):
            continue
        cb, cc = chains(pb), chains(pc)
        sb = summary(f"{T}/{T}_lineph_summary{('.' + tag) if tag else ''}.csv")
        sc = summary(f"{T}/{T}_lineph_summary.ldc.csv")
        for p in PARAMS:
            if p not in cb or p not in cc:
                continue
            xb, xc = cb[p].to_numpy(), cc[p].to_numpy()
            sd = np.std(xb)
            ess_b = sb.get(p, {}).get("ess", np.nan)
            ess_c = sc.get(p, {}).get("ess", np.nan)
            shift = (np.median(xc) - np.median(xb)) / sd if sd > 0 else np.nan
            mcse = (np.sqrt(np.pi / 2) * sd * np.sqrt(1 / ess_b + 1 / ess_c)
                    if np.isfinite(ess_b) and np.isfinite(ess_c) else np.nan)
            null = split_null(xb, rng) + split_null(xc, rng)
            k = ks(xb, xc)
            rows.append(dict(target=T, param=p, shift_sd=round(shift, 4),
                             shift_mcse=round((np.median(xc) - np.median(xb)) / mcse, 2)
                             if np.isfinite(mcse) and mcse > 0 else np.nan,
                             width=round(np.std(xc) / sd, 3) if sd > 0 else np.nan,
                             ks=round(k, 4), ks_null95=round(np.percentile(null, 95), 4),
                             ks_pass=bool(k <= np.percentile(null, 95)),
                             edge_q=int(np.sum((xc == 0) | (xc == 1)))
                             if p in ("q1", "q2") else 0))
        print(f"  {T} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{B}/ldc_compare_accuracy.csv", index=False)
    pd.set_option("display.width", 220)
    worst = df.loc[df.groupby("target").shift_sd.apply(lambda s: s.abs().idxmax())]
    print(worst[["target", "param", "shift_sd", "shift_mcse", "width"]].to_string(index=False))
    print(f"\n{df.target.nunique()} targets, {len(df)} parameter comparisons:")
    print(f"  |shift| median {df.shift_sd.abs().median():.4f} sd, max "
          f"{df.shift_sd.abs().max():.4f} sd; |shift| > 3 MCSE: "
          f"{int((df.shift_mcse.abs() > 3).sum())}")
    print(f"  width ratio range {df.width.min():.3f}-{df.width.max():.3f}")
    print(f"  KS within the chain-split null (95%): {int(df.ks_pass.sum())}/{len(df)}")
    fails = df[~df.ks_pass]
    if len(fails):
        print(fails[["target", "param", "ks", "ks_null95", "shift_sd", "width"]].to_string(index=False))
    print(f"  q draws on the box edge (collapsed): {int(df.edge_q.sum())}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("perf", "all"):
        perf()
    if what in ("accuracy", "all"):
        accuracy()
