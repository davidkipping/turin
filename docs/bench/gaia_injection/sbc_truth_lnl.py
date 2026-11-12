"""SBC: ln L and ln posterior (model space, prior included) at the truth and
at each run's best draw, from the same prepared data turin fitted."""
import sys, glob, io, tarfile, numpy as np, pandas as pd, mlx.core as mx
from turin import prep, params
from turin.gaialik import GaiaLogProb
D = sys.argv[1]
only = sys.argv[2:]
T = pd.read_csv(f"{D}/truth.csv")
out = []
for _, t in T.iterrows():
    n = t["name"]
    if only and n not in only:
        continue
    tars = glob.glob(f"{D}/runs/{n}/*_chains.csv.tar.gz")
    if not tars:
        continue
    eph = dict(period=t.P_in, epoch=t.t0_in, duration=t.T14_in_h,
               depth=t.depth_in_ppm)
    pr = prep.prepare_gaia(f"GaiaDR3-{t.sid}", eph,
                           lcfile=f"{D}/{n}.csv", log=None)
    ed = pr.epoch_data
    cv = prep.cv_band_orders(ed, pr.eph["duration"], log=None)
    cen = prep.centering_constants(ed, pr.eph)
    box = params.gaia_peak_box(pr.eph, pr.time)
    lay = params.gaia_layout(pr.eph, pr.bands, box=box[:2])
    tar = tarfile.open(tars[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0])
                                 .read().decode()), comment="#")
    names = list(lay.names)
    best = ch.iloc[int(ch.loglike.idxmax())]
    vb = [best["dP"] - lay.P_ref, best["dtau0"] - lay.tau0_ref] + \
         [best[c] for c in names[2:]]
    vt = [t.P - lay.P_ref, t.T0 - lay.tau0_ref, t.k, t.b / (1 + t.k),
          t.T14_h / 24]
    for c in names[5:]:
        vt.append(np.log(t["s_" + c[4:]]) if c.startswith("lns_") else t[c])
    X = mx.array(np.array([vb, vt]), dtype=mx.float64)
    with mx.stream(mx.cpu):
        res = {}
        for pri in (False, True):
            lp = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64,
                             prior=pri)
            res[pri] = np.array(lp(X)) + lp.log_const
    out.append(dict(name=n, dlnL=res[False][0] - res[False][1],
                    dlnpost=res[True][0] - res[True][1],
                    lnL_truth=res[False][1], lnpost_truth=res[True][1],
                    T14_best_h=best["T14"] * 24, T14_truth_h=t.T14_h,
                    k_best=best["k"], b_best=best["b"], k=t.k, b=t.b))
    print(n, f"dlnL {out[-1]['dlnL']:.2f} dlnpost {out[-1]['dlnpost']:.2f}",
          flush=True)
O = pd.DataFrame(out)
O.to_csv(f"{D}/truth_lnl{'_' + '_'.join(only) if only else ''}.csv",
         index=False)
pd.set_option("display.width", 200)
print(O.round(3).to_string())
