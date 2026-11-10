"""Build 50 Gaia injections on real DR3 cadences (MetalLS's DataLink zips,
read only): each host's times, bands and catalogue errors, with synthetic
flux = 1 + transit (elome float64, contact rule, each band's exposure)
+ s_band * error * N(0, 1). Writes inj/injNN.csv (an --lcfile) and
inj/truth.csv.  python inj_build.py S"""
import csv, glob, math, sys, zipfile
import numpy as np, mlx.core as mx, elome
from turin import model
from turin.data import gaia

S = sys.argv[1]
N, SNR_MIN, NIN_MIN = 50, 7.0, 3
rng = np.random.default_rng(2026)
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


rows, hi = [], 0
for j in order:
    if len(rows) == N:
        break
    sid, lc = hosts[j]
    t, band, sig = lc["time"], lc["band"], lc["flux_err"]
    for attempt in range(200):
        P = math.exp(rng.uniform(math.log(1.0), math.log(10.0)))
        rho = 1408.0 * math.exp(rng.uniform(math.log(0.3), math.log(3.0)))
        aR = (6.674e-11 * rho * (P * 86400) ** 2 / (3 * math.pi)) ** (1 / 3)
        k = rng.uniform(0.05, 0.2)
        b = rng.uniform(0, 1) * (1 + k)
        if b >= aR:
            continue
        sin_i = math.sqrt(1 - (b / aR) ** 2)
        T14 = P / math.pi * math.asin(math.sqrt((1 + k) ** 2 - b ** 2)
                                      / (aR * sin_i))
        t0 = t.min() + rng.uniform(0, P)
        q = rng.uniform(0, 1, (3, 2))
        s = rng.uniform(1, 4, 3)
        n = np.round((t - t0) / P)
        tau = t - (t0 + n * P)
        dev = np.zeros_like(t)
        for c in range(3):
            m = band == c
            dev[m] = dev64(tau[m], P, k, b, T14, q[c],
                           gaia.EXPOSURE_DAYS[gaia.BANDS[c]])
        snr = math.sqrt(np.sum((dev / (s[band] * sig)) ** 2))
        n_in = int(np.sum(dev < 0))
        if snr >= SNR_MIN and n_in >= NIN_MIN:
            break
    else:
        continue
    flux = 1 + dev + s[band] * sig * rng.standard_normal(t.size)
    name = f"inj{len(rows):02d}"
    with open(f"{S}/inj/{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "flux", "flux_err", "band"])
        for row in zip(t, flux, sig, band):
            w.writerow([f"{row[0]:.10f}", f"{row[1]:.10f}", f"{row[2]:.6e}",
                        gaia.BANDS[row[3]]])
    rows.append(dict(name=name, sid=sid, P=P, t0=t0, T14_h=T14 * 24,
                     depth_ppm=k * k * 1e6, k=k, b=b, rho=rho,
                     q1_G=q[0, 0], q2_G=q[0, 1], q1_BP=q[1, 0], q2_BP=q[1, 1],
                     q1_RP=q[2, 0], q2_RP=q[2, 1],
                     s_G=s[0], s_BP=s[1], s_RP=s[2], snr=snr, n_in=n_in,
                     n_pts=t.size, attempts=attempt + 1))
    print(name, sid, f"P {P:.3f} k {k:.3f} b {b:.2f} T14 {T14*24:.2f}h "
          f"SNR {snr:.1f} n_in {n_in} tries {attempt+1}")
with open(f"{S}/inj/truth.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "injections")
