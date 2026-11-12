"""Coverage of the prior-drawn injections (sbc_build.py): u, the posterior
quantile of each truth. With truths from turin's prior and a data-only
selection, u is uniform for every parameter if the fit is right.
python sbc_coverage.py DIR [1 = converged only]"""
import glob, io, os, re, sys, tarfile
import numpy as np, pandas as pd
from scipy import stats

D = sys.argv[1]
only_conv = len(sys.argv) > 2 and sys.argv[2] == "1"
truth = pd.read_csv(f"{D}/truth.csv")
PARAMS = ["P", "T0", "k", "b", "T14", "s_G", "s_BP", "s_RP",
          "q1_G", "q2_G", "q1_BP", "q2_BP", "q1_RP", "q2_RP"]
runs = []
for _, T in truth.iterrows():
    n = T["name"]
    tars = glob.glob(f"{D}/runs/{n}/*_lineph_chains.csv.tar.gz")
    if not tars or not os.path.exists(f"{D}/runs/{n}/done"):
        continue
    log = open(f"{D}/runs/{n}.log").read()
    rh = re.findall(r"worst R-hat ([\d.]+)", log)
    tar = tarfile.open(tars[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0])
                                 .read().decode()), comment="#")
    draws = {"P": ch["dP"], "T0": ch["dtau0"], "k": ch["k"], "b": ch["b"],
             "T14": ch["T14"] * 24}
    ref = {"P": T["P"], "T0": T["T0"], "k": T["k"], "b": T["b"],
           "T14": T["T14_h"]}
    for p in PARAMS[5:]:
        draws[p], ref[p] = ch[p], T[p]
    row = dict(name=n, conv="lineph: converged" in log,
               rhat=float(rh[-1]) if rh else np.nan, snr=T["snr"],
               n_in=T["n_in"], T14_ratio=T["T14_h"] / T["T14_in_h"],
               warned=bool(re.search(r"WARNING: the period posterior", log)))
    for p in PARAMS:
        row["u_" + p] = float(np.mean(np.asarray(draws[p]) < ref[p]))
    runs.append(row)
R = pd.DataFrame(runs)
R.to_csv(f"{D}/coverage_runs{'_conv' if only_conv else ''}.csv", index=False)
print(f"{len(R)} runs finished; converged {int(R.conv.sum())}; "
      f"warned at the box edge {int(R.warned.sum())}")
if only_conv:
    R = R[R.conv]
    print(f"converged only: {len(R)}")
print(f"{'param':7s} {'68%':>6s} {'95%':>6s} {'mean u':>7s} {'KS p':>6s}  outside 99%")
for p in PARAMS:
    u = R["u_" + p].to_numpy()
    far = ", ".join(R.name[np.abs(u - 0.5) >= 0.495])
    print(f"{p:7s} {np.mean(np.abs(u-.5)<.34):6.2f} "
          f"{np.mean(np.abs(u-.5)<.475):6.2f} {u.mean():7.3f} "
          f"{stats.kstest(u, 'uniform').pvalue:6.3f}  {far}")
n = len(R)
print(f"binomial 1-sigma at n={n}: 68% +-{np.sqrt(.68*.32/n):.2f}, "
      f"95% +-{np.sqrt(.95*.05/n):.2f}; mean u +-{np.sqrt(1/12/n):.3f}")
