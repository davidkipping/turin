"""Gaia epoch photometry: reading, the command line, time systems."""

import numpy as np
import pytest

from turin import cli
from turin.data import gaia

_HEADER = ("solution_id,source_id,transit_id,g_transit_time,g_transit_flux,"
           "g_transit_flux_error,g_transit_flux_over_error,g_transit_mag,"
           "bp_obs_time,bp_flux,bp_flux_error,bp_flux_over_error,bp_mag,"
           "rp_obs_time,rp_flux,rp_flux_error,rp_flux_over_error,rp_mag,"
           "variability_flag_g_reject,variability_flag_bp_reject,"
           "variability_flag_rp_reject,g_other_flags,bp_other_flags,"
           "rp_other_flags,rejected_by_photometry")


def _row(tid, t, g, bp, rp, flags=("false", "false", "false"), rej="false"):
    def part(v, dt):
        if v is None:
            return [""] * 5
        f, e = v
        return [f"{t + dt:.10f}", f"{f}", f"{e}", f"{f / e:.4f}", "12.0"]
    return ",".join(["1", "7", str(tid), *part(g, 0.0), *part(bp, 3e-4),
                     *part(rp, 3.1e-4), *flags, "0", "0", "0", rej])


def _csv():
    rows = [
        _row(10, 1700.25, (1000.0, 2.0), (500.0, 1.5), (800.0, 1.8)),
        # BP missing (empty cells), RP rejected by the variability pipeline
        _row(11, 1700.32, (1010.0, 2.0), None, (790.0, 1.8),
             flags=("false", "false", "true")),
        # the whole transit rejected by the photometric pipeline
        _row(12, 1700.40, (5.0, 2.0), (500.0, 1.5), (800.0, 1.8), rej="true"),
        # unusable values: zero G flux, negative BP error
        _row(13, 1650.10, (0.0, 2.0), (505.0, -1.0), (805.0, 1.8)),
    ]
    return "\n".join([_HEADER, *rows]) + "\n"


def test_datalink_rows_keep_what_gaia_keeps():
    lc = gaia.parse_datalink_csv(_csv())
    # transit 13 keeps only RP, 10 all three, 11 only G; 12 nothing
    assert lc["transit_id"].tolist() == [13, 10, 10, 10, 11]
    assert lc["band"].tolist() == [2, 0, 1, 2, 0]
    assert np.all(np.diff(lc["time"]) >= 0)
    np.testing.assert_allclose(lc["time"][1:4], [1700.25, 1700.2503,
                                                 1700.2503100])
    assert lc["flux"][1] == 1000.0 and lc["flux_err"][2] == 1.5


def test_each_band_is_normalised_by_its_own_median():
    lc = gaia.normalise_bands(gaia.parse_datalink_csv(_csv()))
    for code in np.unique(lc["band"]):
        m = lc["band"] == code
        assert np.median(lc["flux"][m]) == pytest.approx(1.0)
    g = lc["band"] == 0          # G: 1000 and 1010, median 1005
    np.testing.assert_allclose(lc["flux_err"][g], 2.0 / 1005.0)


def test_lcfile_reads_bands_by_name_or_code_and_defaults_to_g(tmp_path):
    p = tmp_path / "lc.csv"
    p.write_text("# a comment\ntime,flux,flux_err,band\n"
                 "2.0,10,1,bp\n1.0,20,2,G\n3.0,30,3,2\n4.0,nan,1,G\n")
    lc = gaia.read_lcfile(str(p))
    assert lc["time"].tolist() == [1.0, 2.0, 3.0]
    assert lc["band"].tolist() == [0, 1, 2]
    q = tmp_path / "g.csv"
    q.write_text("time,flux,flux_err\n1.0,5,1\n")
    assert gaia.read_lcfile(str(q))["band"].tolist() == [0]
    bad = tmp_path / "bad.csv"
    bad.write_text("time,flux\n1,2\n")
    with pytest.raises(ValueError, match="flux_err"):
        gaia.read_lcfile(str(bad))
    with pytest.raises(ValueError, match="band"):
        gaia.band_code("V")


def test_lcfile_light_curve_is_normalised_and_not_cached(tmp_path,
                                                         monkeypatch):
    from turin.data import lightcurve
    monkeypatch.setattr(lightcurve, "CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "lc.csv"
    p.write_text("time,flux,flux_err,band\n1,10,1,G\n2,30,1,G\n3,7,1,RP\n")
    lc = gaia.get_lightcurve("GaiaDR4-99", lcfile=str(p))
    assert lc["mission"] == "gaia_dr4" and lc["source"] == str(p)
    np.testing.assert_allclose(lc["flux"], [0.5, 1.5, 1.0])
    assert not (tmp_path / "cache").exists()


def test_dr3_is_downloaded_once_then_cached(tmp_path, monkeypatch):
    from turin.data import lightcurve
    monkeypatch.setattr(lightcurve, "CACHE_DIR", str(tmp_path))
    calls = []

    def fake(sid, progress=None):
        calls.append(sid)
        return gaia.parse_datalink_csv(_csv())
    monkeypatch.setattr(gaia, "download_dr3", fake)
    a = gaia.get_lightcurve("GaiaDR3-7")
    b = gaia.get_lightcurve("GaiaDR3-7")
    assert calls == [7] and (tmp_path / "GaiaDR3-7.pkl").exists()
    np.testing.assert_array_equal(a["flux"], b["flux"])
    assert a["mission"] == "gaia_dr3" and a["source"] == "datalink"


def test_a_release_without_a_reader_needs_lcfile():
    with pytest.raises(ValueError, match="--lcfile"):
        gaia.get_lightcurve("GaiaDR4-1")


def test_target_names_carry_the_release():
    assert gaia.parse_target("gaiadr3-12")["name"] == "GaiaDR3-12"
    assert gaia.parse_target("GaiaDR3-12")["source_id"] == 12
    with pytest.raises(ValueError, match="release"):
        gaia.parse_target("Gaia-12")
    assert gaia.is_gaia("GaiaDR3-1") and not gaia.is_gaia("KOI-1.01")


def test_gaia_time_to_tdb_matches_astropy():
    from astropy.time import Time

    t = np.array([1666.4, 1802.5, 2100.0, 2704.3, 3700.0])
    want = (Time(t + gaia.GAIA_TIME_OFFSET, format="jd", scale="tcb").tdb.jd
            - gaia.GAIA_TIME_OFFSET)
    np.testing.assert_allclose(gaia.gaia_time_to_tdb(t), want, rtol=0,
                               atol=2e-9)          # ~0.2 ms
    shift = (gaia.gaia_time_to_tdb(t) - t) * 86400.0
    assert np.all((shift < -18.0) & (shift > -23.0))


_EPH = ["--period=3.17", "--t0=2171.8", "--T14=2.3", "--depth=15000"]


def test_gaia_target_on_the_command_line():
    a = cli.parse_args(["--gaiadr3-4906145613282734208", *_EPH])
    assert a.target == "GaiaDR3-4906145613282734208"
    assert a.modes == ("lineph",)                   # LinEph by default
    assert (a.period, a.t0, a.T14_hours, a.depth_ppm) == (3.17, 2171.8, 2.3,
                                                          15000.0)
    a = cli.parse_args(["--GaiaDR4-5", *_EPH, "--lcfile=x.csv"])
    assert a.target == "GaiaDR4-5" and a.lcfile == "x.csv"
    assert cli.parse_args(["--GaiaDR3-5", *_EPH, "--t0=-3.5"]).t0 == -3.5


@pytest.mark.parametrize("argv,pattern", [
    (["--Gaia-123", *_EPH], "release"),
    (["--GaiaDR3-1"], "candidate ephemeris"),
    (["--GaiaDR3-1", "--period=3"], "--t0= --T14= --depth="),
    (["--GaiaDR3-1", *_EPH, "--period=-1"], "positive"),
    (["--GaiaDR3-1", *_EPH, "--t0=x"], "Gaia time"),
    (["--GaiaDR4-1", *_EPH], "--lcfile"),
    (["--GaiaDR3-1", *_EPH, "--modes=lineph,ttv"], "LinEph only"),
    (["--GaiaDR3-1", *_EPH, "--modes=ttv"], "LinEph only"),
    (["--GaiaDR3-1", *_EPH, "--mle"], "--mle"),
    (["--GaiaDR3-1", *_EPH, "--oblate", "--law=hybrid4"], "--oblate"),
    (["--GaiaDR3-1", *_EPH, "--sc"], "--sc"),
    (["--GaiaDR3-1", *_EPH, "--ld=collapsed"], "collapsed"),
    (["--GaiaDR3-1", *_EPH, "--geometry=chord"], "circular"),
    (["--GaiaDR3-1", *_EPH, "--PL=ratio"], "PL=exact"),
    (["--GaiaDR3-1", *_EPH, "--TTVmax=30"], "TTVmax"),
    (["--KOI-1.01", "--period=3"], "Gaia targets only"),
    (["--TOI-1.01", "--lcfile=x.csv"], "Gaia targets only"),
])
def test_gaia_refusals_name_their_reason(argv, pattern):
    with pytest.raises(SystemExit, match=pattern):
        cli.parse_args(argv)




# -- the Gaia likelihood: rows per (transit, band), one baseline per band ----

import math

import mlx.core as mx

from turin import likelihood, model, outputs, params, prep, profile
from turin.gaialik import GaiaLogProb

_EXP = {i: gaia.EXPOSURE_DAYS[b] for i, b in enumerate(gaia.BANDS)}


def gaia_times(rng, n_clusters=20, span=950.0, t_start=1700.0):
    """Gaia-like G times: clusters of scans ~6 h apart, each a pair of
    field-of-view transits 106.5 min apart."""
    t = []
    for s in np.sort(rng.uniform(0, span, n_clusters)):
        for j in range(int(rng.integers(2, 6))):
            ts = s + 0.25 * j + rng.uniform(-0.01, 0.01)
            t += [ts, ts + 106.5 / 1440]
    return t_start + np.array(t)


def _aRs(T14, P, k, b):
    s = math.sin(math.pi * T14 / P)
    return math.sqrt(((1 + k) ** 2 - b ** 2) / s ** 2 + b ** 2)


def _dev64(tau, P, k, b, T14, q, exp):
    """elome's float64 kernel at each point's own time from mid-transit."""
    import elome
    u1, u2 = model.limb_dark_coeffs_np(*q)
    with mx.stream(mx.cpu):
        d = elome.flux_dev_from_tau(
            mx.array(np.asarray(tau), dtype=mx.float64), P, _aRs(T14, P, k, b),
            b, k, float(u1), float(u2), exp_time=exp, integration="contact",
            n_gl=model.N_GL)
        return np.array(d)


TRUTH = dict(P=1.37, t0=1701.123, k=0.12, b=0.3, T14=0.09,
             q={0: (0.4, 0.3), 1: (0.5, 0.28), 2: (0.3, 0.33)})


def synthetic_gaia(seed=3, bands=(0, 1, 2), noise=2.0, trend=True):
    """A three-band Gaia-like light curve with a transit, slow per-band
    trends, and noise ``noise`` times its quoted errors."""
    rng = np.random.default_rng(seed)
    tg = gaia_times(rng)
    t = np.concatenate([tg + 27.0 * j / 86400 for j in range(len(bands))])
    band = np.repeat(np.asarray(bands), tg.size)
    o = np.argsort(t, kind="stable")
    t, band = t[o], band[o]
    T = TRUTH
    dev = np.zeros_like(t)
    for c in bands:
        m = band == c
        n = np.round((t[m] - T["t0"]) / T["P"])
        dev[m] = _dev64(t[m] - (T["t0"] + n * T["P"]), T["P"], T["k"],
                        T["b"], T["T14"], T["q"][c], _EXP[c])
    sig = np.where(band == 0, 1e-3, 2e-3)
    base = 1 + (1e-3 * (t - 2200) / 500 * (band + 1) if trend else 0.0)
    flux = (1 + dev) * base + noise * sig * rng.standard_normal(t.size)
    return t, flux, sig, band, dev


def _gaia_fit_inputs(seed=3, **kw):
    t, flux, sig, band, dev = synthetic_gaia(seed, **kw)
    T = TRUTH
    eph = dict(period=T["P"], epoch=T["t0"], duration=T["T14"] * 24,
               depth=T["k"] ** 2 * 1e6)
    ed = prep.segment_gaia(t, flux, sig, band, T["P"], T["t0"],
                           T["T14"] * 24, _EXP)
    cv = prep.cv_band_orders(ed, T["T14"] * 24)
    cen = prep.centering_constants(ed, eph)
    labels = tuple(gaia.BANDS[c] for c in np.unique(band))
    lay = params.gaia_layout(eph, labels)
    return t, flux, sig, band, dev, eph, ed, cv, cen, lay


def _vectors(lay, n=4, seed=0):
    rng = np.random.default_rng(seed)
    T = TRUTH
    beta = T["b"] / (1 + T["k"])
    V = []
    for i in range(n):
        v = [rng.normal(0, 1e-4) * (i > 0), rng.normal(0, 3e-3) * (i > 0),
             T["k"] * (1 + 0.05 * rng.normal() * (i > 0)),
             beta + 0.05 * rng.normal() * (i > 0),
             T["T14"] * (1 + 0.03 * rng.normal() * (i > 0))]
        for band in lay.bands:
            v += list(np.clip(np.array(T["q"][gaia.BANDS.index(band)])
                              + 0.05 * rng.normal(size=2) * (i > 0),
                              0.01, 0.99))
        v += list(rng.uniform(-0.2, 1.2, len(lay.bands)))
        V.append(v)
    return np.array(V)


def _reference_lnL(t, flux, sig, band, lay, cv, ed, v):
    """The likelihood written out from scratch in float64: each band's
    points at their own transit, one Legendre baseline per band over the
    light curve's span, solved by np.linalg.solve, errors times exp(lns)."""
    dP, dt, k, beta, T14 = v[:5]
    b = beta * (1 + k)
    P = lay.P_ref + dP
    mid, half = ed["x_global"]
    tot = 0.0
    for j, label in enumerate(lay.bands):
        c = gaia.BANDS.index(label)
        m = band == c
        n = np.round((t[m] - lay.tau0_ref) / lay.P_ref)
        q = v[5 + 2 * j:7 + 2 * j]
        f = 1 + _dev64(t[m] - (lay.tau0_ref + dt + n * P), P, k, b, T14, q,
                       _EXP[c])
        L = prep.legendre_matrix((t[m] - mid) / half, cv["band_orders"][c])
        A = f[:, None] * L
        w = 1 / sig[m] ** 2
        coef = np.linalg.solve((A.T * w) @ A, (A.T * w) @ (flux[m] - f))
        s = math.exp(v[5 + 2 * len(lay.bands) + j])
        r = (flux[m] - f * (1 + L @ coef)) / (sig[m] * s)
        tot += (-0.5 * np.sum(r ** 2) - m.sum() * math.log(s)
                - np.sum(np.log(sig[m])) - 0.5 * m.sum() * math.log(2 * math.pi))
    return tot


def test_every_transit_and_band_with_a_point_is_a_row():
    t, flux, sig, band, dev, eph, ed, *_ = _gaia_fit_inputs()
    n_pt, _ = prep.nearest_transit(t, eph["period"], eph["epoch"])
    want = {(int(c), int(n)) for c, n in zip(band, n_pt)}
    assert ed["n_epochs"] == len(want)
    assert int(ed["mask"].sum()) == t.size          # nothing windowed away
    n_rows = np.round((ed["epoch_centers"] - eph["epoch"]) / eph["period"])
    assert {(int(c), int(n)) for c, n in zip(ed["band"], n_rows)} == want
    assert (ed["mask"].sum(axis=1) == 1).any()       # one-point rows kept
    assert np.all(np.diff(ed["band"]) >= 0)          # band, then transit
    np.testing.assert_array_equal(ed["exposure"],
                                  [_EXP[c] for c in ed["band"]])


def test_the_gaia_likelihood_is_the_one_written_out_by_hand():
    t, flux, sig, band, dev, eph, ed, cv, cen, lay = _gaia_fit_inputs()
    V = _vectors(lay)
    with mx.stream(mx.cpu):
        lp = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64,
                         prior=False)
        got = np.array(lp(mx.array(V, dtype=mx.float64))) + lp.log_const
    want = [_reference_lnL(t, flux, sig, band, lay, cv, ed, v) for v in V]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-8)
    # the prior is the default layout's (b, k) prior and nothing else
    with mx.stream(mx.cpu):
        lpp = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64)
        withp = np.array(lpp(mx.array(V, dtype=mx.float64))) + lpp.log_const
        bk = np.array(params.bk_log_prior(
            mx.array(V[:, 3:4], dtype=mx.float64),
            mx.array(V[:, 2:3], dtype=mx.float64)))
    np.testing.assert_allclose(withp - got, bk, rtol=0, atol=1e-10)


def test_the_shared_baseline_solve_matches_numpy():
    t, flux, sig, band, dev, eph, ed, cv, cen, lay = _gaia_fit_inputs()
    rng = np.random.default_rng(1)
    f_dev = np.where(ed["mask"] > 0,
                     -0.01 * rng.uniform(size=(3,) + ed["mask"].shape), 0.0)
    orders = np.array([cv["band_orders"][c] for c in range(3)])
    x = prep.gaia_basis_x(ed)
    with mx.stream(mx.cpu):
        d = profile.build_shared_design(ed, orders, ed["band"], x,
                                        dtype=mx.float64)
        c = np.array(profile.solve_shared_coefficients(
            d, mx.array(f_dev, dtype=mx.float64)))
    ref = profile.solve_shared_coefficients_np(ed, orders, ed["band"], x,
                                               1.0 + f_dev)
    for b in range(3):
        rows = np.where(ed["band"] == b)[0]
        # every row of a band carries the band's coefficients
        np.testing.assert_allclose(c[:, rows], np.repeat(
            ref[:, b:b + 1], rows.size, axis=1), rtol=0, atol=1e-12)


def test_gaia_float32_value_and_gradient_against_float64():
    t, flux, sig, band, dev, eph, ed, cv, cen, lay = _gaia_fit_inputs()
    V = _vectors(lay, n=6)
    lo = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float32)
    with mx.stream(mx.cpu):
        hi = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64)
        v64 = np.array(hi(mx.array(V, dtype=mx.float64)))
        g64 = np.array(mx.grad(lambda x: mx.sum(hi(x)))(
            mx.array(V, dtype=mx.float64)))
    v32 = np.array(lo(mx.array(V, dtype=mx.float32)))
    g32 = np.array(mx.grad(lambda x: mx.sum(lo(x)))(
        mx.array(V, dtype=mx.float32)))
    assert np.max(np.abs(v32 - v64)) < 5e-3        # ~400 points
    scale = np.max(np.abs(g64), axis=0) + 1e-6
    assert np.max(np.abs(g32 - g64) / scale) < 1e-3


def test_error_scales_enter_only_through_r_and_their_normalisation():
    """lns_b scales band b's residuals and adds -N_b lns_b; the baseline
    and every other band are untouched."""
    t, flux, sig, band, dev, eph, ed, cv, cen, lay = _gaia_fit_inputs()
    V = _vectors(lay, n=1)
    with mx.stream(mx.cpu):
        lp = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64,
                         prior=False)
        p = lp.unpack(mx.array(V, dtype=mx.float64))
        rows, c = lp._terms(p)
        rows = np.array(rows)[0]
        base = float(np.array(lp(mx.array(V, dtype=mx.float64)))[0])
        i = lay.index("lns_BP")
        W = V.copy()
        W[0, i] += 0.4
        moved = float(np.array(lp(mx.array(W, dtype=mx.float64)))[0])
        c2 = np.array(lp._terms(lp.unpack(mx.array(W, dtype=mx.float64)))[1])
    np.testing.assert_array_equal(np.array(c), c2)
    bp = ed["band"] == 1
    n_bp = int(ed["mask"][bp].sum())
    chi2 = -2.0 * (rows[bp].sum() - 0.5 * n_bp)       # sum r^2 at the old s
    want = (-0.5 * chi2 * (math.exp(-0.8) - 1.0) - n_bp * 0.4)
    assert moved - base == pytest.approx(want, abs=1e-9)


def test_cv_picks_a_trend_where_there_is_one():
    *_, cv, _, _ = _gaia_fit_inputs(trend=True, noise=1.0)
    assert all(k >= 1 for k in cv["band_orders"].values())
    # on white noise the CV can still prefer a higher order, but only by
    # noise-sized margins (the same rule as turin's per-epoch CV)
    for seed in range(4):
        *_, cv, _, _ = _gaia_fit_inputs(seed=seed, trend=False, noise=1.0)
        assert max(np.max(s) for s in cv["scores"].values()) < 2.0


def test_the_band_grid_matches_one_kernel_call_per_band():
    """transit_flux_dev_bands: each band's points in its own call, at its
    own exposure and limb darkening, put back where they came from."""
    t, flux, sig, band, dev, eph, ed, cv, cen, lay = _gaia_fit_inputs()
    V = _vectors(lay, n=2)
    with mx.stream(mx.cpu):
        lp = GaiaLogProb(lay, cen, ed, cv["orders"], dtype=mx.float64)
        fd = np.array(lp.flux_dev(lp.unpack(mx.array(V, dtype=mx.float64))))
    m = ed["mask"] > 0
    rows_t = ed["times_padded"]
    for i, v in enumerate(V):
        dP, dt, k, beta, T14 = v[:5]
        P = lay.P_ref + dP
        for j in range(3):
            sel = m & (ed["band"] == j)[:, None]
            n = np.broadcast_to(cen["n_arr"][:, None], m.shape)[sel]
            tau = rows_t[sel] - (lay.tau0_ref + dt + n * P)
            want = _dev64(tau, P, k, beta * (1 + k), T14,
                          v[5 + 2 * j:7 + 2 * j], _EXP[j])
            # 3e-13: the centred time representation's rounding
            np.testing.assert_allclose(fd[i][sel], want, rtol=0, atol=1e-12)


def test_the_upward_clip_keeps_dips():
    rng = np.random.default_rng(0)
    f = 1 + 1e-3 * rng.standard_normal(200)
    f[10] += 0.02                         # outlier up
    f[20] -= 0.02                         # a transit-like dip
    band = np.zeros(200, dtype=int)
    keep = prep.clip_gaia(np.arange(200.0), f, np.full(200, 1e-3), band)
    assert not keep[10] and keep[20] and keep.sum() == 199


def test_prepare_gaia_recentres_cleans_and_reports(monkeypatch):
    t, flux, sig, band, dev = synthetic_gaia()
    # a band too thin to fit
    keep = band != 2
    keep[np.where(band == 2)[0][::20]] = True       # 7 RP points
    lc = {"time": t[keep], "flux": flux[keep], "flux_err": sig[keep],
          "band": band[keep].astype(np.int8), "mission": "gaia_dr3"}
    monkeypatch.setattr(prep, "get_gaia_lightcurve",
                        lambda *a, **k: dict(lc))
    T = TRUTH
    eph = dict(period=T["P"], epoch=T["t0"] - 300 * T["P"],
               duration=T["T14"] * 24, depth=T["k"] ** 2 * 1e6)
    logs = []
    pd = prep.prepare_gaia("GaiaDR3-7", eph, log=logs.append)
    assert pd.bands == ("G", "BP")
    assert any("RP dropped" in n for n in pd.notes)
    # recentred by whole periods onto the data
    n = (pd.eph["epoch"] - eph["epoch"]) / T["P"]
    assert abs(n - round(n)) < 1e-9 and pd.time.min() < pd.eph["epoch"] \
        < pd.time.max()
    assert set(pd.noise_chi) == {"G", "BP"}
    assert pd.noise_chi["G"] == pytest.approx(2.0, rel=0.25)
    assert pd.mission == "gaia_dr3" and pd.exposure_time == _EXP[0]


def test_gaia_layout_names_and_boxes():
    T = TRUTH
    eph = dict(period=T["P"], epoch=T["t0"], duration=2.0, depth=1e4)
    lay = params.gaia_layout(eph, ("G", "RP"), law="hybrid4")
    assert lay.names == ("dP", "dtau0", "k", "beta", "T14",
                         "q1_G", "q2_G", "q3_G", "q4_G",
                         "q1_RP", "q2_RP", "q3_RP", "q4_RP", "lns_G", "lns_RP")
    assert lay.bands == ("G", "RP") and lay.ld_names == ()
    i = lay.index("lns_G")
    assert (lay.lo[i], lay.hi[i]) == (params.LNS_LO, params.LNS_HI)
    with pytest.raises(ValueError, match="sampled"):
        likelihood.build_target(lay, {}, {}, [], likelihood_only=True)


def test_hybrid4_bands_run_and_differ_by_band():
    t, flux, sig, band, dev, eph, ed, cv, cen, _ = _gaia_fit_inputs()
    lay = params.gaia_layout(eph, ("G", "BP", "RP"), law="hybrid4")
    v = [0, 0, 0.12, 0.27, 0.09] + [0.5, 0.5, 0.5, 0.5] * 3 + [0.7] * 3
    V = np.array([v, v])
    V[1, 5] = 0.9                         # G's depth weight only
    lp = GaiaLogProb(lay, cen, ed, cv["orders"])
    fd = np.array(lp.flux_dev(lp.unpack(mx.array(V, dtype=mx.float32))))
    g = (ed["band"] == 0)[:, None] & (ed["mask"] > 0)
    assert np.abs(fd[0][g] - fd[1][g]).max() > 1e-4
    np.testing.assert_array_equal(fd[0][~g], fd[1][~g])
    assert np.all(np.isfinite(np.array(lp(mx.array(V, dtype=mx.float32)))))


def test_the_float64_exp_is_exact_where_mlx_s_is_not():
    from turin.gaialik import _exp
    x = np.linspace(-4.0, 4.0, 4001)
    with mx.stream(mx.cpu):
        got = np.array(_exp(mx.array(x, dtype=mx.float64)))
    assert np.max(np.abs(got / np.exp(x) - 1)) < 1e-14     # MLX's: 4e-7



def _serve_gaia(monkeypatch, **kw):
    t, flux, sig, band, dev = synthetic_gaia(**kw)
    lc = {"time": t, "flux": flux, "flux_err": sig,
          "band": band.astype(np.int8), "mission": "gaia_dr3",
          "source": "test"}
    monkeypatch.setattr(prep, "get_gaia_lightcurve",
                        lambda *a, **k: {kk: (vv.copy() if hasattr(vv, "copy")
                                              else vv)
                                         for kk, vv in lc.items()})
    T = TRUTH
    # the candidate's ephemeris: right period, epoch off by 12 minutes
    return ["--GaiaDR3-7", f"--period={T['P']}",
            f"--t0={T['t0'] + 12 / 1440}", f"--T14={T['T14'] * 24}",
            f"--depth={T['k'] ** 2 * 1e6}"]


def _gaia_args(tmp_path, argv, **over):
    a = cli.parse_args(argv)
    for key, value in {**dict(chains=64, warmup=150, samples=60,
                              max_samples=60, outdir=str(tmp_path),
                              seed=0), **over}.items():
        setattr(a, key, value)
    return a


def test_gaia_pipeline_end_to_end(tmp_path, monkeypatch):
    import os
    import tarfile

    from turin import pipeline

    argv = _serve_gaia(monkeypatch)
    logs = []
    code = pipeline.run(_gaia_args(tmp_path, argv), log=logs.append)
    assert code in (0, pipeline.EXIT_UNCONVERGED)
    text = "\n".join(logs)
    assert "Gaia bands: G" in text and "not explored" in text
    assert "Robust chi" in text and "MAP log-density" in text
    names = sorted(os.listdir(tmp_path))
    for suffix in ("_lineph_summary.csv", "_lineph_chains.csv.tar.gz",
                   "_lineph_lcdata.csv", "_lineph_fold.pdf",
                   "_lineph_corner.pdf", "_lineph_resume.pkl"):
        assert any(n.endswith(suffix) for n in names), (suffix, names)

    rows = {}
    for line in open(tmp_path / "GaiaDR3-7_lineph_summary.csv"):
        f = line.rstrip("\n").split(",")
        if f[0] and not f[0].startswith(("#", "Parameter")):
            rows[f[0]] = (float(f[1]), float(f[2]))
    T = TRUTH
    for want in ("q1_G", "q2_RP", "lns_BP", "s_G", "s_RP", "T0_TDB",
                 "b", "log10_rho"):
        assert want in rows, want
    # the injected noise is twice the quoted errors in every band
    for b in ("G", "BP", "RP"):
        med, sd = rows[f"s_{b}"]
        assert abs(med - 2.0) < 4 * sd + 0.2, (b, med, sd)
    assert abs(rows["k"][0] - T["k"]) < 4 * rows["k"][1] + 0.01
    assert abs(rows["dP"][0] - T["P"]) < 1e-3
    # the 12-minute epoch error is found, in Gaia time
    n = round((rows["dtau0"][0] - T["t0"]) / T["P"])
    assert abs(rows["dtau0"][0] - (T["t0"] + n * T["P"])) * 1440 < 5
    # and in TDB too (BJD_TDB - 2455197.5; ~19 s earlier than TCB in 2016)
    shift = (rows["T0_TDB"][0] - rows["dtau0"][0]) * 86400
    assert -20 < shift < -18.5

    lc = (tmp_path / "GaiaDR3-7_lineph_lcdata.csv").read_text().splitlines()
    assert lc[0] == "time,flux,flux_err,model_flux,band"
    assert {r.split(",")[-1] for r in lc[2:]} == {"G", "BP", "RP"}
    with tarfile.open(tmp_path / "GaiaDR3-7_lineph_chains.csv.tar.gz") as tar:
        head = tar.extractfile(tar.getmembers()[0]).readline().decode()
    assert "lns_G" in head and "s_G" in head and head.strip().endswith(
        "loglike")
    st = outputs.load_resume(str(tmp_path), "GaiaDR3-7", "lineph")
    assert st.bands == ("G", "BP", "RP") and st.profile_mode == "exact"


def test_gaia_map_starts_hold_the_period_at_the_candidate():
    from turin import seeding
    T = TRUTH
    eph = dict(period=T["P"], epoch=T["t0"], duration=2.0, depth=1e4)
    lay = params.gaia_layout(eph, ("G",))
    sd = seeding.map_start_sd(lay)
    assert sd[lay.index("dP")] == seeding.GAIA_START_SD_DP < 1e-3
    assert sd[lay.index("k")] == seeding.MAP_START_SD
    kep = params.lineph_layout(eph)
    assert np.all(seeding.map_start_sd(kep) == seeding.MAP_START_SD)


def test_the_box_is_the_seeded_peak():
    T = TRUTH
    eph = dict(period=T["P"], epoch=2200.0, duration=T["T14"] * 24,
               depth=1e4)
    t = np.array([1700.0, 2200.0, 2650.0])
    dP_half, dt_half, dP_unit, dt_unit = params.gaia_peak_box(eph, t)
    # the furthest point is 500 d (365 orbits) from the epoch
    n_max = 500.0 / T["P"]
    assert dP_unit == pytest.approx(T["T14"] / (2 * n_max), rel=1e-12)
    assert dt_unit == pytest.approx(T["T14"] / 2, rel=1e-12)
    assert dP_half == pytest.approx(params.GAIA_BOX_UNITS * dP_unit)
    assert dt_half == pytest.approx(params.GAIA_BOX_UNITS * dt_unit)
    lay = params.gaia_layout(eph, ("G",), box=(dP_half, dt_half))
    i, j = lay.index("dP"), lay.index("dtau0")
    assert (lay.lo[i], lay.hi[i], lay.lo[j], lay.hi[j]) == (
        -dP_half, dP_half, -dt_half, dt_half)
    # a long duration on a short period: never a quarter period or more
    wide = dict(eph, duration=0.4 * T["P"] * 24)
    assert params.gaia_peak_box(wide, t)[1] == pytest.approx(T["P"] / 4)


def _exact_ephemeris_draws(C=20000, seed=1):
    """Draws of (dP, dtau0) from their exact conditional at a fixed shape,
    from a dense grid zoomed onto the mass. Returns the layout, ``lp``, the
    transform, the box, the draws (model space) and the two marginal
    references ``(column, grid, cell masses)``."""
    t, flux, sig, band, dev, eph, ed, cv, cen, _ = _gaia_fit_inputs()
    box = params.gaia_peak_box(eph, t)
    lay = params.gaia_layout(eph, ("G", "BP", "RP"), box=box[:2])
    lp = GaiaLogProb(lay, cen, ed, cv["orders"])
    tr = lay.transform()
    v0 = _vectors(lay, n=1)[0]
    iP, iT = lay.index("dP"), lay.index("dtau0")

    def dense(loP, hiP, loT, hiT, n=300):
        gP, gT = np.linspace(loP, hiP, n + 1), np.linspace(loT, hiT, n + 1)
        V = np.repeat(v0[None], n * n, axis=0)
        V[:, iP] = np.repeat(0.5 * (gP[1:] + gP[:-1]), n)
        V[:, iT] = np.tile(0.5 * (gT[1:] + gT[:-1]), n)
        L = np.concatenate([np.array(lp(mx.array(V[s:s + 16384],
                                                  dtype=mx.float32)),
                                     dtype=np.float64)
                            for s in range(0, n * n, 16384)]).reshape(n, n)
        p = np.exp(L - L.max())
        return gP, gT, p / p.sum()

    gP, gT, p = dense(lay.lo[iP], lay.hi[iP], lay.lo[iT], lay.hi[iT])
    # zoom onto the mass, so the reference resolves it
    cP, cT = np.cumsum(p.sum(1)), np.cumsum(p.sum(0))
    a, b = (max(np.searchsorted(cP, 1e-6) - 2, 0),
            min(np.searchsorted(cP, 1 - 1e-6) + 3, 300))
    c, d = (max(np.searchsorted(cT, 1e-6) - 2, 0),
            min(np.searchsorted(cT, 1 - 1e-6) + 3, 300))
    gP, gT, p = dense(gP[a], gP[b], gT[c], gT[d])

    rng = np.random.default_rng(seed)
    i, j = np.divmod(rng.choice(p.size, size=C, p=p.ravel()), 300)
    V = np.repeat(v0[None], C, axis=0)
    V[:, iP] = gP[i] + (gP[1] - gP[0]) * rng.uniform(size=C)
    V[:, iT] = gT[j] + (gT[1] - gT[0]) * rng.uniform(size=C)
    return lay, lp, tr, box, V, ((iP, gP, p.sum(1)), (iT, gT, p.sum(0)))


def _assert_marginals(v1, lay, refs):
    from scipy import stats

    for col, grid, m in refs:
        cdf = np.concatenate([[0.0], np.cumsum(m)])
        res = stats.kstest(v1[:, col], lambda x: np.interp(x, grid, cdf))
        assert res.pvalue > 1e-3, (lay.names[col], res)


def test_the_ephemeris_move_leaves_the_exact_conditional_invariant():
    """Draw (dP, dtau0) from their exact conditional at a fixed shape on a
    dense grid, sweep once with a coarse grid, and the draws must still
    follow it. Without the MH correction the same sweep fails by KS
    statistics of 0.08 (p ~ 1e-100); measured in review."""
    from turin import gibbs

    lay, lp, tr, box, V, refs = _exact_ephemeris_draws()
    move = gibbs.EphemerisGibbs(lp, tr, lay, units=box[2:], n_grid=64,
                                seed=2)
    u1, st = move.sweep(tr.from_model_np(V))
    v1 = tr.model_np(u1)
    assert 0.05 < st.accept.min() and st.accept.max() < 0.95
    # the shape is untouched
    others = [k for k in range(lay.dim) if k not in (refs[0][0], refs[1][0])]
    np.testing.assert_allclose(v1[:, others], V[:, others], rtol=1e-12,
                               atol=1e-12)
    _assert_marginals(v1, lay, refs)


def test_the_de_move_leaves_the_exact_conditional_invariant():
    """The same exact conditional draws, then DE steps along (dP, dtau0)
    against the sampler's density in u: the draws must still follow it, and
    the move must have moved them."""
    import anvil

    from turin import gibbs

    lay, lp, tr, box, V, refs = _exact_ephemeris_draws(seed=3)
    target = anvil.TransformedLogDensity(lp, tr)
    de = gibbs.EnsembleDE(target, lay, n_steps=5, cols=("dP", "dtau0"),
                          seed=4)
    u0 = tr.from_model_np(V)
    u1, acc, move = de.run(u0)
    v1 = tr.model_np(u1)
    assert 0.05 < acc < 0.95 and move.min() > 0.3, (acc, move)
    others = [k for k in range(lay.dim) if k not in (refs[0][0], refs[1][0])]
    np.testing.assert_allclose(v1[:, others], V[:, others], rtol=1e-9,
                               atol=1e-12)
    _assert_marginals(v1, lay, refs)


class _Banana:
    """A 5-D banana in u with exact draws: x0 ~ N(0, 1), x1 = 0.8 (x0^2 - 1)
    + 0.3 z, x2 = 0.9 x0 + 0.2 z', x3, x4 ~ N(0, 1)."""

    names = ("dP", "dtau0", "k", "beta", "T14")

    def index(self, name):
        return self.names.index(name)

    def log_prob(self, u):
        x0, x1, x2, x3, x4 = (u[:, i] for i in range(5))
        return -(0.5 * x0 ** 2 + 0.5 * ((x1 - 0.8 * (x0 ** 2 - 1)) / 0.3) ** 2
                 + 0.5 * ((x2 - 0.9 * x0) / 0.2) ** 2 + 0.5 * x3 ** 2
                 + 0.5 * x4 ** 2)

    def draws(self, C, rng):
        z = rng.standard_normal((C, 5))
        x0 = z[:, 0]
        return np.column_stack([x0, 0.8 * (x0 ** 2 - 1) + 0.3 * z[:, 1],
                                0.9 * x0 + 0.2 * z[:, 2], z[:, 3], z[:, 4]])


@pytest.mark.parametrize("mh", [True, False])
def test_the_de_move_leaves_a_curved_posterior_invariant(mh, monkeypatch):
    """Exact draws of a curved, coupled 5-D posterior through 20 DE steps of
    every coordinate: each marginal and the banana's two conditional
    residuals keep their exact laws. With the Metropolis test removed the
    same check fails (the mutation is the ``mh=False`` case)."""
    from scipy import stats

    from turin import gibbs

    ban = _Banana()
    rng = np.random.default_rng(5)
    u0 = ban.draws(20000, rng)
    de = gibbs.EnsembleDE(ban, ban, n_steps=20, seed=6)
    if not mh:
        monkeypatch.setattr(de, "_log_prob",
                            lambda U: np.zeros(U.shape[0]))
    u1, acc, move = de.run(u0)
    x0, x1, x2 = u1[:, 0], u1[:, 1], u1[:, 2]
    checks = (x0, (x1 - 0.8 * (x0 ** 2 - 1)) / 0.3, (x2 - 0.9 * x0) / 0.2,
              u1[:, 3], u1[:, 4])
    p = min(stats.kstest(c, "norm").pvalue for c in checks)
    if mh:
        assert 0.05 < acc < 0.95 and move.min() > 0.5, (acc, move)
        assert p > 1e-3, p
    else:
        assert p < 1e-10, p


def test_only_gaia_and_ttv_fits_get_a_gibbs_move(tmp_path, monkeypatch):
    from turin import pipeline

    gaia_args = _gaia_args(tmp_path, _serve_gaia(monkeypatch))
    assert pipeline._gibbsgrid("lineph", gaia_args) == "on"
    gaia_args.gibbsgrid = "off"
    assert pipeline._gibbsgrid("lineph", gaia_args) == "off"
    kep = cli.parse_args(["--KOI-518.02"])
    assert pipeline._gibbsgrid("lineph", kep) == "off"
    assert pipeline._gibbsgrid("ttv", kep) == "on"


def test_a_gaia_lineage_refuses_a_different_box(tmp_path, monkeypatch):
    from turin import pipeline

    argv = _serve_gaia(monkeypatch)
    logs = []
    pipeline.run(_gaia_args(tmp_path, argv), log=logs.append)
    text = "\n".join(logs)
    assert "ephemeris Gibbs:" in text and "the seeded period peak" in text
    st = outputs.load_resume(str(tmp_path), "GaiaDR3-7", "lineph")
    assert st.gibbsgrid == "on" and len(st.box) == 6
    lo, hi = st.box[2], st.box[3]
    assert lo == -hi and 0 < hi < 1e-3
    # a different --T14 is a different box
    argv2 = [a if not a.startswith("--T14=") else
             f"--T14={TRUTH['T14'] * 24 * 1.5}" for a in argv]
    with pytest.raises(SystemExit, match="ephemeris box"):
        pipeline.run(_gaia_args(tmp_path, argv2, max_samples=120),
                     log=lambda m: None)
    # a state from before the box (0.1.102) reads (), and is refused too
    st.box = ()
    with pytest.raises(SystemExit, match="0.1.102"):
        st.check_box((1.0,) * 6)
