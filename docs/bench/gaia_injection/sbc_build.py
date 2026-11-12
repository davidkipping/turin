"""Simulation-based calibration of turin's Gaia fit: truths drawn from
turin's OWN prior given a candidate, on real DR3 cadences.

Per injection: one DR3 host's times, bands and catalogue errors (MetalLS's
DataLink zips, read only); a candidate (P_in, t0_in, T14_in, depth_in)
drawn independently of the truth, as a BLS would supply; then the truth
from the prior turin puts on that candidate's fit:
  dP, dtau0 ~ U(box) (params.gaia_peak_box about the recentred epoch),
  k ~ U(0, 1), b | k ~ U(0, 1+k) with the linear grazing taper
  (params.bk_log_prior, "transiting"), T14 ~ U(0, 3 T14_in),
  q1, q2 ~ U(0, 1) per band, ln s ~ U(ln 0.3, ln 30) per band.
Flux = 1 + elome float64 transit + s * err * N(0, 1).
Selection is on the DATA alone (so calibration still holds given it):
nothing clipped by prep.clip_gaia; >= 3 points below each band's median by
> 4 robust sigma; those points' median deficit <= 0.05 (the Gaia regime).
python sbc_build.py OUTDIR"""
import csv, glob, math, sys, zipfile
import numpy as np, mlx.core as mx, elome
from turin import model, params, prep
from turin.data import gaia

S = sys.argv[1]
N, N_DEF, DEF_SIG, MAX_DEPTH = 50, 3, 4.0, 0.05
rng = np.random.default_rng(2027)
hosts = []
for z in sorted(glob.glob("/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/MetalLS/data/gaia_pt/dl/*.zip")):
    zf = zipfile.ZipFile(z)
    for name in zf.namelist():
        sid = name.rsplit(" ", 1)[-1].replace(".csv", "")
        lc = gaia.normalise_bands(gaia.parse_datalink_csv(zf.read(name).decode()))
        if all(np.sum(lc["band"] == c) >= 10 for c in range(3)):
            hosts.append((sid, lc))
print(len(hosts), "hosts with >= 10 points in every band")
order = rng.permutation(len(hosts))


def aRs_from_T14(T14, P, k, b):          # turin's circle relation
    s = math.sin(math.pi * T14 / P)
    return math.sqrt(((1 + k) ** 2 - b ** 2) / s ** 2 + b ** 2)


def dev64(tau, P, k, b, T14, q, exp):
    u1, u2 = model.limb_dark_coeffs_np(*q)
    with mx.stream(mx.cpu):
        d = elome.flux_dev_from_tau(
            mx.array(np.asarray(tau), dtype=mx.float64), P,
            aRs_from_T14(T14, P, k, b), b, k, float(u1), float(u2),
            exp_time=exp, integration="contact", n_gl=model.N_GL)
        return np.array(d)


def candidate(t):
    """A BLS-like candidate, independent of the truth."""
    P = math.exp(rng.uniform(math.log(1.0), math.log(10.0)))
    rho = 1408.0 * math.exp(rng.uniform(math.log(0.3), math.log(3.0)))
    aR = (6.674e-11 * rho * (P * 86400) ** 2 / (3 * math.pi)) ** (1 / 3)
    k = rng.uniform(0.05, 0.2)
    b = rng.uniform(0, 1)
    T14 = P / math.pi * math.asin(math.sqrt((1 + k) ** 2 - b ** 2)
                                  / (aR * math.sqrt(1 - (b / aR) ** 2)))
    return P, t.min() + rng.uniform(0, P), T14, k * k * 1e6


def prior_truth(P_in, tau0_ref, T14_in, box):
    k = rng.uniform(0, 1)
    while True:
        b = rng.uniform(0, 1 + k)
        u = (b - (1 - k)) / (2 * k)
        if u <= 0 or rng.uniform() < 1 - u:
            break
    return dict(P=P_in + rng.uniform(-box[0], box[0]),
                T0=tau0_ref + rng.uniform(-box[1], box[1]),
                k=k, b=b, T14=rng.uniform(0, 3 * T14_in),
                q=rng.uniform(0, 1, (3, 2)),
                lns=rng.uniform(params.LNS_LO, params.LNS_HI, 3))


def selected(t, f, sig, band):
    if not prep.clip_gaia(t, f, sig, band).all():
        return False, 0
    deficits = []
    for c in range(3):
        m = band == c
        med = np.median(f[m])
        chi = prep.robust_sigma((f[m] - med) / sig[m])
        low = (med - f[m]) / (sig[m] * chi) > DEF_SIG
        deficits += list((med - f[m])[low])
    ok = len(deficits) >= N_DEF and np.median(deficits) <= MAX_DEPTH
    return ok, len(deficits)


rows = []
for j in order:
    if len(rows) == N:
        break
    sid, lc = hosts[j]
    t, band, sig = lc["time"], lc["band"], lc["flux_err"]
    for attempt in range(400):
        P_in, t0_in, T14_in, depth_in = candidate(t)
        n_pt, _ = prep.nearest_transit(t, P_in, t0_in)
        tau0_ref = t0_in + int(np.round(np.median(np.unique(n_pt)))) * P_in
        eph = dict(period=P_in, epoch=tau0_ref, duration=T14_in * 24)
        box = params.gaia_peak_box(eph, t)[:2]
        tr = prior_truth(P_in, tau0_ref, T14_in, box)
        n = np.round((t - tr["T0"]) / tr["P"])
        tau = t - (tr["T0"] + n * tr["P"])
        dev = np.zeros_like(t)
        for c in range(3):
            m = band == c
            dev[m] = dev64(tau[m], tr["P"], tr["k"], tr["b"], tr["T14"],
                           tr["q"][c], gaia.EXPOSURE_DAYS[gaia.BANDS[c]])
        s = np.exp(tr["lns"])
        flux = 1 + dev + s[band] * sig * rng.standard_normal(t.size)
        ok, n_def = selected(t, flux, sig, band)
        if ok:
            break
    else:
        continue
    name = f"sbc{len(rows):02d}"
    with open(f"{S}/{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "flux", "flux_err", "band"])
        for row in zip(t, flux, sig, band):
            w.writerow([f"{row[0]:.10f}", f"{row[1]:.10f}", f"{row[2]:.6e}",
                        gaia.BANDS[row[3]]])
    snr = math.sqrt(np.sum((dev / (s[band] * sig)) ** 2))
    r = dict(name=name, sid=sid, P_in=P_in, t0_in=t0_in, T14_in_h=T14_in * 24,
             depth_in_ppm=depth_in, tau0_ref=tau0_ref, dP_half=box[0],
             dtau0_half=box[1], P=tr["P"], T0=tr["T0"], k=tr["k"], b=tr["b"],
             T14_h=tr["T14"] * 24)
    for c, bn in enumerate(gaia.BANDS):
        r[f"q1_{bn}"], r[f"q2_{bn}"] = tr["q"][c]
    for c, bn in enumerate(gaia.BANDS):
        r[f"s_{bn}"] = s[c]
    r.update(snr=snr, n_in=int(np.sum(dev < 0)), n_def=n_def,
             n_pts=t.size, attempts=attempt + 1)
    rows.append(r)
    print(name, sid, f"k {tr['k']:.3f} b {tr['b']:.2f} T14 "
          f"{tr['T14']*24:.2f}h (in {T14_in*24:.2f}) s {np.round(s, 1)} "
          f"snr {snr:.0f} n_in {r['n_in']} n_def {n_def} try {attempt+1}",
          flush=True)
with open(f"{S}/truth.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "injections")
