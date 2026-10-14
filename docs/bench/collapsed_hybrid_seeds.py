"""Two-seed check (0.1.44's rule): for a target, the KS distance and median
shift between any two runs, for every parameter. Same-mode pairs across
seeds (sampled s0 vs s1, collapsed s0 vs s1) are a true two-run null; the
cross-mode pairs are judged against them.

Run layout as collapsed_hybrid_compare.py, plus seed-1 runs tagged h4s_s1
and h4c2_s1 (--seed=1).

usage: collapsed_hybrid_seeds.py DIR TARGET
"""
import io
import sys
import tarfile

import numpy as np
import pandas as pd

D, T = sys.argv[1], sys.argv[2]
PARAMS = ["dP", "dtau0", "k", "b", "T14", "log10_rho",
          "q1", "q2", "q3", "q4", "w1", "w2", "w3", "w4"]
RUNS = {"S0": "h4s", "S1": "h4s_s1", "C0": "h4c2", "C1": "h4c2_s1"}
PAIRS = [("S0", "S1"), ("C0", "C1"), ("S0", "C0"), ("S1", "C1"),
         ("S0", "C1"), ("S1", "C0")]


def chains(tag):
    tf = tarfile.open(f"{D}/out/{T}_lineph_chains.{tag}.csv.tar.gz")
    return pd.read_csv(io.StringIO(tf.extractfile(tf.getmembers()[0])
                                   .read().decode()), comment="#")


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    g = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, g, "right") / a.size
                               - np.searchsorted(b, g, "right") / b.size)))


C = {k: chains(v) for k, v in RUNS.items()}
rows = []
for p in PARAMS:
    sd = np.std(C["S0"][p])
    r = {"param": p}
    for a, b in PAIRS:
        x, y = C[a][p].to_numpy(), C[b][p].to_numpy()
        r[f"KS {a}-{b}"] = round(ks(x, y), 4)
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(f"{T}: KS distance between runs (same-mode pairs first = the null)")
print(df.to_string(index=False))
same = df[["KS S0-S1", "KS C0-C1"]].to_numpy()
cross = df[["KS S0-C0", "KS S1-C1", "KS S0-C1", "KS S1-C0"]].to_numpy()
print(f"same-mode KS: median {np.median(same):.4f}, max {same.max():.4f}; "
      f"cross-mode KS: median {np.median(cross):.4f}, max {cross.max():.4f}")
g = {k: np.mean(v.b > 1 - v.k) * 100 for k, v in C.items()}
print("grazing %: " + ", ".join(f"{k} {g[k]:.2f}" for k in RUNS))
print("k max: " + ", ".join(f"{k} {C[k].k.max():.3f}" for k in RUNS))
