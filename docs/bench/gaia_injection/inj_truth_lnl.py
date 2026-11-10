"""ln L (likelihood alone, float64) at the injected truth and at each run's
best draw, from the same prepared data turin fitted. Under a correct model
2 (lnL_best - lnL_truth) is about chi^2 with the free dimension (14)."""
import sys, glob, io, tarfile, numpy as np, pandas as pd, mlx.core as mx
from turin import prep, params
from turin.gaialik import GaiaLogProb
D = sys.argv[1]
T = pd.read_csv(f"{D}/truth.csv")
out = []
for _, t in T.iterrows():
    n = t["name"]
    tars = glob.glob(f"{D}/runs/{n}/*_chains.csv.tar.gz")
    if not tars:
        continue
    eph = dict(period=t.P, epoch=t.t0, duration=t.T14_h, depth=t.depth_ppm)
    pr = prep.prepare_gaia(f"GaiaDR3-{t.sid}", eph,
                           lcfile=f"{D}/{n}.csv", log=None)
    ed = pr.epoch_data
    cv = prep.cv_band_orders(ed, pr.eph["duration"], log=None)
    cen = prep.centering_constants(ed, pr.eph)
    lay = params.gaia_layout(pr.eph, pr.bands)
    tar = tarfile.open(tars[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0])
                                 .read().decode()), comment="#")
    names = list(lay.names)
    best = ch.iloc[int(ch.loglike.idxmax())]
    vb = [best["dP"] - lay.P_ref, best["dtau0"] - lay.tau0_ref] + \
         [best[c] for c in names[2:]]
    T0 = t.t0 + np.round((lay.tau0_ref - t.t0) / t.P) * t.P
    vt = [t.P - lay.P_ref, T0 - lay.tau0_ref, t.k, t.b / (1 + t.k),
          t.T14_h / 24]
    for c in names[5:]:
        if c.startswith("lns_"):
            vt.append(np.log(t["s_" + c[4:]]))
        else:
            vt.append(t[c])
    with mx.stream(mx.cpu):
        lp = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64,
                         prior=False)
        v = np.array(lp(mx.array(np.array([vb, vt]), dtype=mx.float64))) \
            + lp.log_const
    out.append(dict(name=n, lnL_best=v[0], lnL_truth=v[1],
                    d=v[0] - v[1], dim=len(names), stored=best.loglike))
    print(n, f"{v[0] - v[1]:.2f}", flush=True)
O = pd.DataFrame(out)
O.to_csv(f"{D}/truth_lnl.csv", index=False)
from scipy import stats
x = 2 * O.d
k = int(O.dim.iloc[0])
print(f"2 dlnL: median {x.median():.1f} mean {x.mean():.1f} (chi2_{k}: "
      f"median {stats.chi2.median(k):.1f} mean {k}); KS vs chi2_{k} p "
      f"{stats.kstest(x, 'chi2', args=(k,)).pvalue:.3f}; min {x.min():.2f}; "
      f"max {x.max():.1f}")
