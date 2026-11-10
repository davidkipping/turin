"""Shape offsets of the Gaia injections: convergence by points in transit,
each run's posterior median minus truth for k, b, T14 (in half 16-84%
widths and relative), the grazing share, and the best draw's offsets.
python inj_shape.py INJDIR   (after inj_coverage.py wrote coverage_runs.csv)"""
import sys, glob, io, tarfile, numpy as np, pandas as pd
from scipy import stats
D = sys.argv[1]
R = pd.read_csv(f"{D}/coverage_runs.csv")
T = pd.read_csv(f"{D}/truth.csv").set_index("name")
R["g"] = pd.cut(R.n_in, [0, 3, 6, 100], labels=["3", "4-6", "7+"])
print(R.groupby("g", observed=True).agg(
    n=("conv", "size"), conv=("conv", "sum"), warned=("warned", "sum"),
    rhat_med=("rhat", "median")))
out = []
for n in R.name:
    t = T.loc[n]
    tar = tarfile.open(glob.glob(f"{D}/runs/{n}/*_chains.csv.tar.gz")[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0]).read()
                                 .decode()), comment="#",
                     usecols=["k", "b", "T14", "loglike"])
    def z(x, ref):
        lo, hi = np.percentile(x, [16, 84])
        return (np.median(x) - ref) / ((hi - lo) / 2)
    i = int(ch.loglike.idxmax())
    out.append(dict(name=n, graz=np.mean(ch.b > 1 - ch.k),
                    zk=z(ch.k, t.k), zb=z(ch.b, t.b),
                    zT=z(ch.T14 * 24, t.T14_h),
                    T_med=np.median(ch.T14) * 24 / t.T14_h - 1,
                    k_med=np.median(ch.k) / t.k - 1,
                    T_best=ch.T14[i] * 24 / t.T14_h - 1,
                    k_best=ch.k[i] / t.k - 1, b_best=ch.b[i] - t.b))
O = pd.DataFrame(out)
print(O.drop(columns="name").describe().round(2))
print("zT vs grazing share", stats.spearmanr(O.zT, O.graz))
O.to_csv(f"{D}/shape_shifts.csv", index=False)
