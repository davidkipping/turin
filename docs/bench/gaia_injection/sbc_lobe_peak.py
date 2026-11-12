"""Maximise the float64 model-space log posterior from the truth and from the
best draw (L-BFGS-B inside the layout bounds); compare the two lobes' peaks."""
import sys, glob, io, tarfile, numpy as np, pandas as pd, mlx.core as mx
from scipy.optimize import minimize
from turin import prep, params
from turin.gaialik import GaiaLogProb
D = sys.argv[1]
T = pd.read_csv(f"{D}/truth.csv")
for n in sys.argv[2:]:
    t = T[T.name == n].iloc[0]
    eph = dict(period=t.P_in, epoch=t.t0_in, duration=t.T14_in_h,
               depth=t.depth_in_ppm)
    pr = prep.prepare_gaia(f"GaiaDR3-{t.sid}", eph, lcfile=f"{D}/{n}.csv",
                           log=None)
    ed = pr.epoch_data
    cv = prep.cv_band_orders(ed, pr.eph["duration"], log=None)
    cen = prep.centering_constants(ed, pr.eph)
    box = params.gaia_peak_box(pr.eph, pr.time)
    lay = params.gaia_layout(pr.eph, pr.bands, box=box[:2])
    names = list(lay.names)
    tar = tarfile.open(glob.glob(f"{D}/runs/{n}/*_chains.csv.tar.gz")[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0])
                                 .read().decode()), comment="#")
    best = ch.iloc[int(ch.loglike.idxmax())]
    vb = np.array([best["dP"] - lay.P_ref, best["dtau0"] - lay.tau0_ref] +
                  [best[c] for c in names[2:]])
    vt = [t.P - lay.P_ref, t.T0 - lay.tau0_ref, t.k, t.b / (1 + t.k),
          t.T14_h / 24]
    for c in names[5:]:
        vt.append(np.log(t["s_" + c[4:]]) if c.startswith("lns_") else t[c])
    vt = np.array(vt)
    lo = np.asarray(lay.lo, float); hi = np.asarray(lay.hi, float)
    sc = (hi - lo)
    lp = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64, prior=True)
    def fg(z):
        with mx.stream(mx.cpu):
            g = mx.value_and_grad(lambda x: lp(x).sum())
            v, gr = g(mx.array((lo + z * sc)[None], dtype=mx.float64))
            return -(float(np.array(v)) + lp.log_const), -np.array(gr)[0] * sc
    with mx.stream(mx.cpu):
        for lab, v0 in (("truth", vt), ("best draw", vb)):
            z0 = np.clip((v0 - lo) / sc, 1e-9, 1 - 1e-9)
            r = minimize(fg, z0, jac=True, method="L-BFGS-B",
                         bounds=[(1e-9, 1 - 1e-9)] * len(z0),
                         options=dict(maxiter=5000))
            x = lo + r.x * sc
            print(n, f"{lab:9s} start {-fg(z0)[0]:.2f} peak {-r.fun:.2f}  "
                  f"dP {x[0]/ (sc[0]/2):+.3f} hw  dtau0 {x[1]/(sc[1]/2):+.3f} hw  "
                  f"k {x[2]:.3f} beta {x[3]:.3f} T14 {x[4]*24:.2f} h  ({r.message})")
