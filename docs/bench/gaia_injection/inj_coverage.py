"""Coverage of the Gaia injections: for each run and parameter, the
posterior quantile u of the true value (the share of draws below it).
Reports 68% / 95% central-interval coverage, a KS test of u against
uniform, convergence, and the box-edge warnings.
python inj_coverage.py INJDIR [converged-only: 1]"""
import glob, io, os, re, sys, tarfile
import numpy as np, pandas as pd
from scipy import stats

D = sys.argv[1]
only_conv = len(sys.argv) > 2 and sys.argv[2] == "1"
truth = pd.read_csv(f"{D}/truth.csv")
PARAMS = ["P", "T0", "k", "b", "T14", "s_G", "s_BP", "s_RP",
          "q1_G", "q2_G", "q1_BP", "q2_BP", "q1_RP", "q2_RP"]
U, runs = {p: [] for p in PARAMS}, []
for _, T in truth.iterrows():
    n = T["name"]
    log_f = f"{D}/runs/{n}.log"
    tars = glob.glob(f"{D}/runs/{n}/*_lineph_chains.csv.tar.gz")
    if not tars or not os.path.exists(f"{D}/runs/{n}/done"):
        continue
    log = open(log_f).read()
    conv = "lineph: converged" in log
    warned = "WARNING: the period posterior runs on" in log.split(
        "round")[-1] if "round" in log else False
    edge = re.findall(r"draws in the outer 10% of the box \(below, above\): "
                      r"period ([\d.]+)%, ([\d.]+)%; epoch ([\d.]+)%, "
                      r"([\d.]+)%", log)
    edge = [float(x) for x in edge[-1]] if edge else [np.nan] * 4
    rh = re.findall(r"worst R-hat ([\d.]+)", log)
    tar = tarfile.open(tars[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0])
                                 .read().decode()), comment="#")
    nn = np.round((np.median(ch["dtau0"]) - T["t0"]) / T["P"])
    draws = {"P": ch["dP"], "T0": ch["dtau0"], "k": ch["k"], "b": ch["b"],
             "T14": ch["T14"] * 24}
    ref = {"P": T["P"], "T0": T["t0"] + nn * T["P"], "k": T["k"],
           "b": T["b"], "T14": T["T14_h"]}
    for p in PARAMS[5:]:
        draws[p], ref[p] = ch[p], T[p]
    row = dict(name=n, conv=conv, rhat=float(rh[-1]) if rh else np.nan,
               snr=T["snr"], n_in=T["n_in"], edge_max=np.nanmax(edge),
               warned=bool(re.search(r"WARNING: the period posterior", log)))
    for p in PARAMS:
        u = float(np.mean(np.asarray(draws[p]) < ref[p]))
        row["u_" + p] = u
    runs.append(row)
R = pd.DataFrame(runs)
print(f"{len(R)} runs finished; converged {int(R.conv.sum())}; "
      f"warned at the box edge {int(R.warned.sum())}")
if only_conv:
    R = R[R.conv]
    print(f"converged only: {len(R)}")
print(f"{'param':7s} {'68%':>9s} {'95%':>9s} {'KS p':>6s}  outside 99%")
for p in PARAMS:
    u = R["u_" + p].to_numpy()
    c68 = np.mean(np.abs(u - 0.5) < 0.34)
    c95 = np.mean(np.abs(u - 0.5) < 0.475)
    ks = stats.kstest(u, "uniform").pvalue
    far = ", ".join(R.name[np.abs(u - 0.5) >= 0.495])
    print(f"{p:7s} {c68:9.2f} {c95:9.2f} {ks:6.3f}  {far}")
n = len(R)
print(f"binomial 1-sigma on coverage at n={n}: 68% +-{np.sqrt(.68*.32/n):.2f},"
      f" 95% +-{np.sqrt(.95*.05/n):.2f}")
R.to_csv(f"{D}/coverage_runs.csv", index=False)
