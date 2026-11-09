"""Step-5 acceptance: each archive-seeded Gaia fit against the literature
(NASA Exoplanet Archive composite). For each quantity, the two-sided tail
probability of the literature value under the posterior draws convolved
with the literature's own error (propagated to the fitted epoch for T0).
python gaia_acceptance.py S RUNDIR (S holds cands.txt and lit.csv: here gaia_cands.txt and gaia_literature.csv)"""
import glob, io, re, sys, tarfile
import numpy as np, pandas as pd

S, R = sys.argv[1], sys.argv[2]
lit = pd.read_csv(f"{S}/lit.csv").set_index("pl_name")
rng = np.random.default_rng(0)
rows = []
for line in open(f"{S}/cands.txt"):
    n = line.split()[0]
    L = lit.loc[n.replace("_", " ")]
    log = open(f"{R}/{n}.log").read()
    conv = "lineph: converged" in log
    tar = tarfile.open(glob.glob(f"{R}/{n}/*_lineph_chains.csv.tar.gz")[0])
    ch = pd.read_csv(io.StringIO(tar.extractfile(tar.getmembers()[0])
                                 .read().decode()), comment="#")
    m = len(ch)
    out = dict(run=n, conv="yes" if conv else "NO")

    def tail(x, ref, sd):
        y = x + (rng.normal(0, sd, len(x)) if sd and np.isfinite(sd) else 0)
        p = np.mean(y < ref)
        return 2 * min(p, 1 - p)

    def add(key, x, ref, sd, scale=1.0):
        if not np.isfinite(ref):
            out[key] = "      -"
            return
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        out[key] = (f"{(q50 - ref) * scale:+.3g} "
                    f"({(q50 - ref) / (0.5 * (q84 - q16)):+.1f}sd,"
                    f" p{tail(x, ref, sd):.2f})")

    P_l, sP = L.pl_orbper, L.pl_orbpererr1
    t_l = L.pl_tranmid - 2455197.5
    T0 = ch["T0_TDB"].to_numpy()
    nn = np.round((np.median(T0) - t_l) / P_l)
    pred, spred = t_l + nn * P_l, np.hypot(L.pl_tranmiderr1, nn * sP)
    add("P [s]", ch["dP"].to_numpy(), P_l, sP, 86400)
    add("T0 [min]", T0, pred, spred, 1440)
    add("T14 [h]", ch["T14"].to_numpy() * 24, L.pl_trandur, L.pl_trandurerr1)
    add("k", ch["k"].to_numpy(), L.pl_ratror, L.pl_ratrorerr1)
    add("b", ch["b"].to_numpy(), L.pl_imppar, L.pl_impparerr1)
    chi = re.search(r"Robust chi out of transit: (.*)", log).group(1)
    s = ", ".join(f"{c}" for c in [f"{b} {np.median(ch['s_' + b]):.2f}"
                                    for b in ("G", "BP", "RP")
                                    if "s_" + b in ch])
    out["s (robust chi)"] = f"{s} ({chi})"
    out["n"] = int(nn)
    rows.append(out)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
print(pd.DataFrame(rows).to_string(index=False))
