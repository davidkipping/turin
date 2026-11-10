"""Coverage of k, b, T14 if each fit had carried a --rhostar prior: the
posterior draws importance-weighted by a Gaussian in log10(rho*/kg m^-3)
centred on truth + N(0, sigma) (a noisy catalogue value), sigma per argv.
python inj_rhostar_rw.py INJDIR SIGMA_DEX (after inj_coverage.py INJDIR, all runs)"""
import sys, glob, io, tarfile, numpy as np, pandas as pd
from scipy import stats
D, sig = sys.argv[1], float(sys.argv[2])
rng = np.random.default_rng(7)
T = pd.read_csv(f"{D}/truth.csv")
R = pd.read_csv(f"{D}/coverage_runs.csv").set_index("name")
U = {p: [] for p in ["k", "b", "T14"]}; ess = []; conv = []
for _, t in T.iterrows():
    tar = tarfile.open(glob.glob(f"{D}/runs/{t['name']}/*_chains.csv.tar.gz")[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0]).read()
                                 .decode()), comment="#",
                     usecols=["k", "b", "T14", "log10_rho"])
    mu = np.log10(t.rho) + rng.normal(0, sig)
    lw = -0.5 * ((ch.log10_rho - mu) / sig) ** 2
    w = np.exp(lw - lw.max()); w /= w.sum()
    ess.append(1 / np.sum(w ** 2)); conv.append(R.loc[t["name"], "conv"])
    for p, ref in [("k", t.k), ("b", t.b), ("T14", t.T14_h / 24)]:
        U[p].append(float(np.sum(w[ch[p].to_numpy() < ref])))
conv = np.array(conv)
print(f"sigma {sig} dex; weight ESS min {min(ess):.0f} median {np.median(ess):.0f}")
for sel, lab in [(np.ones_like(conv, bool), "all"), (conv, "converged")]:
    for p in U:
        u = np.array(U[p])[sel]
        print(f"  {lab:9s} {p:4s} 68% {np.mean(abs(u-.5)<.34):.2f} 95% "
              f"{np.mean(abs(u-.5)<.475):.2f} mean u {u.mean():.2f} KS p "
              f"{stats.kstest(u, 'uniform').pvalue:.3f}")
