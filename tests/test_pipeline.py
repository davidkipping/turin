"""The CLI orchestration, end to end, on a synthetic target.

The MAST download and the archive query are replaced, so this exercises
everything turin actually owns -- conditioning, cross-validation, the fit,
every product, the resume state and its guards -- without a network call.
Kept cheap on purpose (few chains, few draws): correctness of the *posterior*
is the injection-recovery tests' job, this is correctness of the plumbing.
"""

from __future__ import annotations

import os
import tarfile

import mlx.core as mx
import numpy as np
import pytest

from turin import cli, mle, model as M, pipeline, prep, outputs
from turin import MODEL_REV as _MODEL_REV

P_TRUE, EPOCH_TRUE, DUR_H = 6.2431, 131.7, 3.1
K_TRUE, B_TRUE, Q1_TRUE, Q2_TRUE = 0.10, 0.28, 0.32, 0.24
T14_TRUE = DUR_H / 24.0
YERR = 3e-4
EPH = {"period": P_TRUE, "epoch": EPOCH_TRUE, "duration": DUR_H,
       "depth": 1e6 * K_TRUE**2, "ingress": 0.35}


@pytest.fixture
def fake_target(monkeypatch):
    """Patch data acquisition to serve a synthetic Kepler-like light curve."""
    return _serve(monkeypatch)


#: Injected TTVs of :func:`fake_ttv_target`, days per epoch: a 12-minute
#: sinusoid, period 5 epochs.
def _ttv_truth(n_epochs):
    return 12.0 / 1440.0 * np.sin(2 * np.pi * np.arange(n_epochs) / 5.0)


@pytest.fixture
def fake_ttv_target(monkeypatch):
    """As :func:`fake_target`, with transit-timing variations injected."""
    return _serve(monkeypatch, ttv=True)


def _serve(monkeypatch, ttv=False):
    rng = np.random.default_rng(99)
    t = np.arange(100.0, 100.0 + 12.0 * P_TRUE, 29.4 / 1440.0)

    # build the truth with turin's own model, one epoch per transit
    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, YERR), P_TRUE, EPOCH_TRUE, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_TRUE, EPOCH_TRUE, DUR_H)
    cen = prep.centering_constants(ed, EPH)
    grid = M.build_grid(cen, prep.supersample_offsets(29.4 / 1440, 5),
                        dtype=mx.float64, exp_time=29.4 / 1440)
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        mid = (_ttv_truth(ed["n_epochs"]) if ttv
               else np.zeros(ed["n_epochs"]))
        f = np.array(M.transit_flux(
            grid, mid=mx.array(mid[None, :], dtype=mx.float64),
            k=col(K_TRUE), b=col(B_TRUE), T14=col(T14_TRUE), q1=col(Q1_TRUE),
            q2=col(Q2_TRUE), period=col(P_TRUE)), dtype=np.float64)[0]

    # scatter the per-epoch model back onto the full time series
    flux = np.ones_like(t)
    mask = ed["mask"] > 0
    for i in range(ed["n_epochs"]):
        idx = np.searchsorted(t, ed["times_padded"][i][mask[i]])
        flux[idx] = f[i][mask[i]]
    flux = flux * (1.0 + 3e-4 * np.sin(2 * np.pi * t / 4.3))
    flux += rng.normal(0.0, YERR, t.size)

    lc = {"time": t, "flux": flux, "flux_err": np.full(t.size, YERR),
          "quality": np.zeros(t.size, dtype=int), "mission": "kepler",
          "cadence_days": 29.4 / 1440.0}

    monkeypatch.setattr(prep, "get_lightcurve",
                        lambda *a, **k: {kk: vv.copy() if hasattr(vv, "copy")
                                         else vv for kk, vv in lc.items()})
    monkeypatch.setattr(prep, "get_ephemeris", lambda *a, **k: dict(EPH))
    monkeypatch.setattr(prep, "get_other_planet_ephemerides", lambda *a, **k: [])
    return lc


#: A run that finished: converged (0) or stopped unconverged at the draw cap
#: (EXIT_UNCONVERGED). The tiny fits here are not meant to converge.
FINISHED = (0, pipeline.EXIT_UNCONVERGED)


def _args(tmp_path, **over):
    base = dict(target="KOI-1.01", modes=("lineph",), chains=32, warmup=60,
                samples=40, max_samples=40, outdir=str(tmp_path), seed=0)
    a = cli.parse_args(["--KOI-1.01"])
    for key, value in {**base, **over}.items():
        setattr(a, key, value)
    return a


def test_lineph_pipeline_writes_every_product(tmp_path, fake_target):
    logs = []
    assert pipeline.run(_args(tmp_path), log=logs.append) in FINISHED
    out = os.path.join(str(tmp_path))
    names = sorted(os.listdir(out))

    for suffix in ("_lineph_summary.csv", "_lineph_chains.csv.tar.gz",
                   "_lineph_logrho.csv", "_lineph_lcdata.csv",
                   "_lineph_fold.pdf", "_lineph_corner.pdf",
                   "_lineph_resume.pkl"):
        assert any(n.endswith(suffix) for n in names), (suffix, names)

    text = "\n".join(logs)
    assert "epoch block" in text and "MAP log-density" in text
    assert "validate_precision" in text or "fp32 error" in text

    # the summary must carry the derived quantities and be machine-readable
    summary = os.path.join(out, "KOI-1.01_lineph_summary.csv")
    arr = np.genfromtxt(summary, delimiter=",", names=True, dtype=None,
                        encoding="utf-8")
    params = set(np.atleast_1d(arr["Parameter"]))
    assert {"dP", "dtau0", "k", "beta", "T14", "q1", "q2", "b",
            "log10_rho"} <= params

    # the chains tarball must have one row per draw and the loglike column
    with tarfile.open(os.path.join(out, "KOI-1.01_lineph_chains.csv.tar.gz")) as tar:
        body = tar.extractfile(tar.getmembers()[0]).read().decode().splitlines()
    assert body[0].split(",")[-1] == "loglike"
    assert len(body) == 2 + 32 * 40


def _summary_rows(path):
    rows = {}
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            if not line.startswith("#"):
                f = line.rstrip("\n").split(",")
                rows[f[0]] = dict(zip(header, f))
    return rows


def test_hybrid4_writes_its_weights_in_both_modes(tmp_path, fake_target):
    """--law=hybrid4 samples q1..q4 and reports the physical weights
    w1..w4 as derived rows and columns, in LinEph and TTV; the TTV epoch
    times start after the four LD columns."""
    logs = []
    assert pipeline.run(_args(tmp_path, modes=("lineph", "ttv"),
                              law="hybrid4"), log=logs.append) in FINISHED
    out = str(tmp_path)
    for mode, first in (("lineph", ["dP", "dtau0", "k", "beta", "T14"]),
                        ("ttv", ["k", "beta", "T14"])):
        rows = _summary_rows(os.path.join(out, f"KOI-1.01_{mode}_summary.csv"))
        names = list(rows)
        assert names[:len(first) + 4] == first + ["q1", "q2", "q3", "q4"]
        for q in ("q1", "q2", "q3", "q4"):
            assert rows[q]["R-hat"] != "" and 0 < float(rows[q]["Median"]) < 1
        w = [float(rows[f"w{j}"]["Median"]) for j in range(1, 5)]
        assert all(rows[f"w{j}"]["R-hat"] == "" for j in range(1, 5))
        assert all(x >= 0 for x in w)

        with tarfile.open(os.path.join(out, f"KOI-1.01_{mode}_chains.csv.tar.gz")) as tar:
            body = tar.extractfile(tar.getmembers()[0]).read().decode().splitlines()
        cols = body[0].split(",")
        assert cols[-7:] == ["b", "log10_rho", "w1", "w2", "w3", "w4",
                             "loglike"]
        d = np.array([[float(x) for x in r.split(",")] for r in body[2:]])
        q = d[:, [cols.index(f"q{j}") for j in range(1, 5)]]
        wd = d[:, [cols.index(f"w{j}") for j in range(1, 5)]]
        np.testing.assert_allclose(wd, M.ld_weights_np(q), rtol=1e-9)
        assert np.all(wd >= 0) and np.all(wd.sum(axis=1) <= 1 + 1e-12)
        if mode == "ttv":
            assert cols[7].startswith("dtau_")

        state = outputs.load_resume(out, "KOI-1.01", mode)
        assert state.law == "hybrid4"
        assert {"q1", "q2", "q3", "q4"} <= set(state.ml_params)
    # the transit-time product has every epoch
    assert os.path.exists(os.path.join(out, "KOI-1.01_ttv_times.csv"))

    # --replot rebuilds the hybrid4 layout and replays its best log-density
    logs = []
    args = _args(tmp_path, modes=("lineph", "ttv"), law="hybrid4")
    args.replot = True
    assert pipeline.run(args, log=logs.append) == 0
    assert sum("replotting" in l for l in logs) == 2, logs


def test_collapsed_hybrid4_lineph_end_to_end(tmp_path, fake_target):
    """--law=hybrid4 --ld=collapsed samples the geometry alone, freezes its
    GHK layout at the MAP and stores the rule with the fit; products carry
    q1..q4 from conditional draws (blank diagnostics) and w1..w4 derived;
    a resume continues on the stored rule and --replot replays it."""
    logs = []
    args = _args(tmp_path, ld="collapsed", law="hybrid4")
    assert pipeline.run(args, log=logs.append) in FINISHED
    text = "\n".join(logs)
    assert "dim 5" in text and "GHK layout frozen at the MAP" in text
    assert "conditional draws of q1, q2, q3, q4" in text
    assert "--PL=exact (required by --ld=collapsed" in text
    assert "refused to compile" not in text        # anvil ran it compiled
    out = str(tmp_path)
    rows = _summary_rows(os.path.join(out, "KOI-1.01_lineph_summary.csv"))
    assert list(rows)[:9] == ["dP", "dtau0", "k", "beta", "T14",
                              "q1", "q2", "q3", "q4"]
    for q in ("q1", "q2", "q3", "q4"):
        assert rows[q]["R-hat"] == "" and 0 < float(rows[q]["Median"]) < 1
    assert rows["k"]["R-hat"] != "" and {"w1", "w4", "b"} <= set(rows)
    with tarfile.open(os.path.join(out, "KOI-1.01_lineph_chains.csv.tar.gz")) as t:
        body = t.extractfile(t.getmembers()[0]).read().decode().splitlines()
    cols = body[0].split(",")
    assert cols == ["dP", "dtau0", "k", "beta", "T14", "q1", "q2", "q3", "q4",
                    "b", "log10_rho", "w1", "w2", "w3", "w4", "loglike"]
    d = np.array([[float(x) for x in r.split(",")] for r in body[2:]])
    assert np.all((d[:, 5:9] > 0) & (d[:, 5:9] < 1))
    st = outputs.load_resume(out, "KOI-1.01", "lineph")
    assert (st.ld, st.law, st.profile_mode) == ("collapsed", "hybrid4", "exact")
    assert len(st.ghk_layout) == 2 and st.ghk_points.shape == (1024, 4)
    assert {"q1", "q2", "q3", "q4"} <= set(st.ml_params)

    logs = []
    args = _args(tmp_path, ld="collapsed", law="hybrid4", extend1=True)
    assert pipeline.run(args, log=logs.append) in FINISHED
    assert not any("GHK layout frozen" in l for l in logs)   # stored rule
    st2 = outputs.load_resume(out, "KOI-1.01", "lineph")
    assert st2.ghk_layout == st.ghk_layout
    assert np.array_equal(st2.ghk_points, st.ghk_points)

    logs = []
    args = _args(tmp_path, ld="collapsed", law="hybrid4")
    args.replot = True
    assert pipeline.run(args, log=logs.append) == 0
    assert any("replotting" in l for l in logs)


@pytest.mark.parametrize("ld", ["collapsed", "sampled"])
def test_oblate_lineph_end_to_end(tmp_path, fake_target, monkeypatch, ld):
    """--oblate samples fc, fs after T14 and reports f, theta (degrees) and
    the semi-axes A, B as derived rows and columns; --PL=auto resolves to
    exact; the chains start on the Laplace spread; a resume and --replot
    rebuild the oblate model. The injection is round, which is all plumbing
    needs. The float64 certificate (82 s of oblate CPU probes here) runs on
    the collapsed default's first fit only; the other runs test plumbing."""
    if ld == "sampled":
        monkeypatch.setattr(pipeline._sampling, "certify",
                            lambda *a, **k: None)
    logs = []
    args = _args(tmp_path, law="hybrid4", oblate=True,
                 ld=None if ld == "collapsed" else "sampled")
    assert pipeline.run(args, log=logs.append) in FINISHED
    text = "\n".join(logs)
    assert "--PL=exact (" in text
    assert "Laplace approximation" in text
    assert "oblateness: f median" in text and "95% upper limit" in text
    assert "aligned oblate on a central chord" in text
    out = str(tmp_path)
    rows = _summary_rows(os.path.join(out, "KOI-1.01_lineph_summary.csv"))
    names = list(rows)
    assert names[:7] == ["dP", "dtau0", "k", "beta", "T14", "fc", "fs"]
    assert rows["fc"]["R-hat"] != "" and rows["fs"]["R-hat"] != ""
    for n in ("f", "theta", "A", "B"):
        assert rows[n]["R-hat"] == ""
    assert 0 <= float(rows["f"]["Median"]) <= 0.6
    with tarfile.open(os.path.join(out, "KOI-1.01_lineph_chains.csv.tar.gz")) as tar:
        body = tar.extractfile(tar.getmembers()[0]).read().decode().splitlines()
    cols = body[0].split(",")
    assert cols[-11:] == ["b", "log10_rho", "w1", "w2", "w3", "w4", "f",
                          "theta", "A", "B", "loglike"]
    d = np.array([[float(x) for x in r.split(",")] for r in body[2:]])
    fc, fs = d[:, cols.index("fc")], d[:, cols.index("fs")]
    f = d[:, cols.index("f")]
    np.testing.assert_allclose(f, fc ** 2 + fs ** 2, rtol=1e-9)
    np.testing.assert_allclose(d[:, cols.index("theta")],
                               np.degrees(0.5 * np.arctan2(fs, fc)), atol=1e-9)
    A = d[:, cols.index("A")]
    np.testing.assert_allclose(A, d[:, cols.index("k")] / np.sqrt(1 - f),
                               rtol=1e-9)
    np.testing.assert_allclose(d[:, cols.index("B")], A * (1 - f), rtol=1e-9)
    assert np.all(f <= 0.6)
    state = outputs.load_resume(out, "KOI-1.01", "lineph")
    assert state.oblate and state.ld == ld and state.profile_mode == "exact"
    assert {"fc", "fs"} <= set(state.ml_params)
    if ld == "collapsed":
        assert "Certificate from" in text
    else:
        return                        # the resume and replot are shared code

    monkeypatch.setattr(pipeline._sampling, "certify", lambda *a, **k: None)
    logs = []
    args = _args(tmp_path, law="hybrid4", oblate=True, extend1=True,
                 ld=None if ld == "collapsed" else "sampled")
    assert pipeline.run(args, log=logs.append) in FINISHED
    logs = []
    args = _args(tmp_path, law="hybrid4", oblate=True,
                 ld=None if ld == "collapsed" else "sampled")
    args.replot = True
    assert pipeline.run(args, log=logs.append) == 0
    assert any("replotting" in l for l in logs)


def test_a_lineage_never_switches_oblateness(tmp_path, fake_target):
    assert pipeline.run(_args(tmp_path, law="hybrid4", ld="sampled"),
                        log=lambda m: None) in FINISHED
    with pytest.raises(SystemExit, match="oblate=False"):
        pipeline.run(_args(tmp_path, law="hybrid4", ld="sampled",
                           oblate=True, extend1=True), log=lambda m: None)


def test_rhostar_prior_runs_logs_and_guards_its_lineage(tmp_path,
                                                       fake_target):
    logs = []
    assert pipeline.run(_args(tmp_path, rhostar=(3.2, 0.05)),
                        log=logs.append) in FINISHED
    assert any("prior: log10(rho*/kg m^-3) ~ N(3.2, 0.05)" in l for l in logs)
    state = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    assert state.rhostar == (3.2, 0.05)
    with pytest.raises(SystemExit, match="rhostar"):
        pipeline.run(_args(tmp_path, extend1=True), log=lambda m: None)


@pytest.mark.parametrize("over,pattern", [
    (dict(oblate=True), "hybrid4"),
    (dict(oblate=True, law="hybrid4", modes=("lineph", "ttv")),
     "LinEph-only"),
    (dict(oblate=True, law="hybrid4", geometry="chord"), "circular"),
])
def test_oblate_refusals_reach_the_pipeline(tmp_path, fake_target, over,
                                           pattern):
    with pytest.raises(SystemExit, match=pattern):
        pipeline.run(_args(tmp_path, **over), log=lambda m: None)
    assert not any(tmp_path.iterdir())


def test_a_lineage_never_switches_law(tmp_path, fake_target):
    assert pipeline.run(_args(tmp_path), log=lambda m: None) in FINISHED
    with pytest.raises(SystemExit, match="law='quadratic'"):
        pipeline.run(_args(tmp_path, law="hybrid4", extend1=True),
                     log=lambda m: None)


@pytest.mark.parametrize("over,pattern", [
    (dict(law="hybrid4", ld="collapsed", modes=("lineph", "ttv")),
     "modes=lineph"),
    (dict(law="hybrid4", geometry="chord"), "circular"),
])
def test_hybrid4_refusals_reach_the_pipeline(tmp_path, fake_target, over,
                                            pattern):
    with pytest.raises(SystemExit, match=pattern):
        pipeline.run(_args(tmp_path, **over), log=lambda m: None)
    assert not any(tmp_path.iterdir())


def test_collapsed_interim_exports_draw_fewer_rows(tmp_path, fake_target,
                                                  monkeypatch):
    """A collapsed mode's interim export (a round that is not the last) caps
    its rows -- each costs a conditional limb-darkening draw that the next
    export replaces -- and the final export draws the full budget. Round 0
    is kept unconverged by an unreachable R-hat target."""
    from turin import ldmarg, sampling

    monkeypatch.setattr(sampling, "RHAT_MAX", 0.0)
    monkeypatch.setattr(pipeline, "_EXPORT_ROWS_COLLAPSED_INTERIM", 64)
    seen = []
    real = ldmarg.draw_limb_darkening

    def spy(lp, v_model, **kw):
        seen.append(len(v_model))
        return real(lp, v_model, **kw)

    monkeypatch.setattr(ldmarg, "draw_limb_darkening", spy)
    logs = []
    args = _args(tmp_path, ld="collapsed", max_samples=80)
    assert pipeline.run(args, log=logs.append) in FINISHED
    assert len(seen) == 2, seen
    assert seen[0] <= 64 < seen[1] == 32 * 80
    assert any("interim export, at most 64 rows" in l for l in logs)


def test_collapsed_lineph_writes_hurin_products(tmp_path, fake_target):
    """--ld=collapsed samples 5 parameters but writes hurin's products
    unchanged: q1, q2 rows and columns come from exact conditional draws."""
    logs = []
    assert pipeline.run(_args(tmp_path, ld="collapsed"),
                        log=logs.append) in FINISHED
    out = str(tmp_path)
    names = sorted(os.listdir(out))
    for suffix in ("_lineph_summary.csv", "_lineph_chains.csv.tar.gz",
                   "_lineph_logrho.csv", "_lineph_lcdata.csv",
                   "_lineph_fold.pdf", "_lineph_corner.pdf",
                   "_lineph_resume.pkl"):
        assert any(n.endswith(suffix) for n in names), (suffix, names)

    text = "\n".join(logs)
    assert "dim 5" in text and "limb darkening (quadratic): collapsed" in text
    assert "--PL=exact (required by --ld=collapsed" in text
    assert "PL probe" not in text                 # never probed
    assert "conditional draws of q1, q2" in text

    # summary: every hurin row; q1, q2 carry no sampler diagnostics
    rows = {}
    with open(os.path.join(out, "KOI-1.01_lineph_summary.csv")) as fh:
        header = fh.readline().strip().split(",")
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split(",")
            rows[f[0]] = dict(zip(header, f))
    assert list(rows)[:7] == ["dP", "dtau0", "k", "beta", "T14", "q1", "q2"]
    assert {"b", "log10_rho"} <= set(rows)
    for name in ("q1", "q2"):
        assert rows[name]["R-hat"] == "" and rows[name]["Bulk_ESS"] == ""
        assert 0 < float(rows[name]["Median"]) < 1
    assert rows["k"]["R-hat"] != ""

    # chains: hurin's columns, q's strictly inside the box
    with tarfile.open(os.path.join(out, "KOI-1.01_lineph_chains.csv.tar.gz")) as tar:
        body = tar.extractfile(tar.getmembers()[0]).read().decode().splitlines()
    assert body[0].split(",") == ["dP", "dtau0", "k", "beta", "T14", "q1",
                                  "q2", "b", "log10_rho", "loglike"]
    draws = np.array([[float(v) for v in r.split(",")] for r in body[2:]])
    assert len(draws) == 32 * 40
    assert np.all((draws[:, 5] > 0) & (draws[:, 5] < 1)
                  & (draws[:, 6] > 0) & (draws[:, 6] < 1))

    state = outputs.load_resume(out, "KOI-1.01", "lineph")
    assert state.ld == "collapsed" and state.profile_mode == "exact"
    assert {"q1", "q2"} <= set(state.ml_params)


def test_collapsed_and_sampled_lineages_never_mix(tmp_path, fake_target):
    pipeline.run(_args(tmp_path, ld="collapsed"), log=lambda m: None)
    with pytest.raises(SystemExit, match="ld"):
        pipeline.run(_args(tmp_path), log=lambda m: None)          # sampled
    other = tmp_path / "other"
    pipeline.run(_args(other), log=lambda m: None)
    with pytest.raises(SystemExit, match="ld"):
        pipeline.run(_args(other, ld="collapsed"), log=lambda m: None)


@pytest.mark.parametrize("over,pattern", [
    (dict(modes=("lineph", "ttv")), "modes=lineph"),
    (dict(profile_mode="ratio"), "envelope theorem"),
    (dict(geometry="chord"), "circular"),
])
def test_pipeline_refuses_collapsed_combinations_the_cli_would(
        tmp_path, fake_target, over, pattern):
    """The tests' _args and any direct caller bypass parse_args, so the
    pipeline refuses on its own -- before touching the disk."""
    with pytest.raises(SystemExit, match=pattern):
        pipeline.run(_args(tmp_path, ld="collapsed", **over), log=lambda m: None)
    assert not any(tmp_path.iterdir())


def test_recovered_parameters_are_in_the_right_region(tmp_path, fake_target):
    """A cheap fit, but it must still land near the injected truth."""
    assert pipeline.run(_args(tmp_path, chains=64, warmup=150, samples=100,
                              max_samples=100), log=lambda m: None) in FINISHED
    arr = np.genfromtxt(os.path.join(str(tmp_path),
                                     "KOI-1.01_lineph_summary.csv"),
                        delimiter=",", names=True, dtype=None, encoding="utf-8")
    med = {str(r["Parameter"]): float(r["Median"]) for r in np.atleast_1d(arr)}
    assert abs(med["dP"] - P_TRUE) < 5e-3
    assert abs(med["k"] - K_TRUE) < 0.4 * K_TRUE
    assert abs(med["T14"] - T14_TRUE) < 0.4 * T14_TRUE
    # the density should be stellar, not absurd
    assert 1.5 < med["log10_rho"] < 5.0


def test_ttv_pipeline_writes_the_two_extra_products(tmp_path, fake_target):
    assert pipeline.run(_args(tmp_path, modes=("ttv",)),
                        log=lambda m: None) in FINISHED
    names = sorted(os.listdir(str(tmp_path)))
    assert any(n.endswith("_ttv_times.csv") for n in names), names
    assert any(n.endswith("_ttv_oc.pdf") for n in names), names

    times = np.genfromtxt(os.path.join(str(tmp_path), "KOI-1.01_ttv_times.csv"),
                          delimiter=",", names=True, dtype=None,
                          encoding="utf-8")
    times = np.atleast_1d(times)
    assert times.size > 3
    for field in ("epoch", "tmid", "tmid_err", "ttv_min", "snr", "npts", "chi2"):
        assert field in times.dtype.names
    # a real transit should be detected in most epochs
    assert np.median(times["snr"]) > 3.0
    assert np.all(times["npts"] > 0)


def test_rerunning_skips_a_converged_mode(tmp_path, fake_target):
    """Auto-resume: a finished mode is not refitted."""
    args = _args(tmp_path)
    pipeline.run(args, log=lambda m: None)
    state = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    assert state is not None
    # force the "done" path regardless of whether the cheap fit converged
    state.done = True
    outputs.save_resume(str(tmp_path), "KOI-1.01", "lineph", state)

    logs = []
    pipeline.run(args, log=logs.append)
    assert "already converged" in "\n".join(logs)


def test_resume_guard_refuses_a_changed_model(tmp_path, fake_target):
    # explicit mode: under --PL=auto the probe's pick is timing-dependent and
    # can itself be ratio, which would make the second run a match
    pipeline.run(_args(tmp_path, profile_mode="exact"), log=lambda m: None)
    with pytest.raises(SystemExit, match="ratio"):
        pipeline.run(_args(tmp_path, profile_mode="ratio"),
                     log=lambda m: None)
    with pytest.raises(SystemExit, match="chord"):
        pipeline.run(_args(tmp_path, geometry="chord"), log=lambda m: None)


def test_fresh_clears_and_refits(tmp_path, fake_target):
    pipeline.run(_args(tmp_path), log=lambda m: None)
    logs = []
    pipeline.run(_args(tmp_path, fresh=True, profile_mode="ratio"),
                 log=logs.append)
    text = "\n".join(logs)
    assert "removed" in text
    state = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    assert state.profile_mode == "ratio"


def test_tagged_runs_are_independent_lineages(tmp_path, fake_target):
    pipeline.run(_args(tmp_path), log=lambda m: None)
    try:
        pipeline.run(_args(tmp_path, tag="alt", profile_mode="ratio"),
                     log=lambda m: None)
    finally:
        outputs.set_run_tag(None)
    names = sorted(os.listdir(str(tmp_path)))
    assert "KOI-1.01_lineph_summary.csv" in names
    assert "KOI-1.01_lineph_summary.alt.csv" in names


def test_ttvmax_wider_than_half_the_period_is_refused(tmp_path, fake_target):
    with pytest.raises(SystemExit, match="half the period"):
        pipeline.run(_args(tmp_path, modes=("ttv",),
                           ttv_max_min=P_TRUE * 24 * 60 * 0.6),
                     log=lambda m: None)


def test_pl_probe_runs_by_default_and_records_its_choice(tmp_path, fake_target):
    """Omitting --PL measures the modes and bakes the winner into the run."""
    from turin import profile

    logs = []
    args = _args(tmp_path)
    assert args.profile_mode == "auto"
    assert pipeline.run(args, log=logs.append) in FINISHED
    text = "\n".join(logs)
    assert "PL probe" in text and "chose" in text
    for mode in profile.PROFILE_MODES:
        assert mode in text

    # the RESOLVED mode is stored, never the literal "auto"
    state = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    assert state.profile_mode in profile.PROFILE_MODES


def test_a_resumed_run_reuses_the_stored_mode_without_reprobing(
        tmp_path, fake_target):
    """A continuation must not silently change its own likelihood."""
    from turin import profile

    pipeline.run(_args(tmp_path), log=lambda m: None)
    state = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    # pin a mode the probe would be unlikely to pick on its own
    state.profile_mode = "ratio"
    state.done = False
    outputs.save_resume(str(tmp_path), "KOI-1.01", "lineph", state)

    logs = []
    pipeline.run(_args(tmp_path), log=logs.append)
    text = "\n".join(logs)
    assert "carried over from the resumed run" in text
    assert "PL probe" not in text
    after = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    assert after.profile_mode == "ratio"


def test_explicit_pl_skips_the_probe(tmp_path, fake_target):
    logs = []
    pipeline.run(_args(tmp_path, profile_mode="exact"), log=logs.append)
    text = "\n".join(logs)
    assert "PL probe" not in text
    assert "--PL=exact" in text
    assert outputs.load_resume(
        str(tmp_path), "KOI-1.01", "lineph").profile_mode == "exact"


def test_products_record_the_model_revision(tmp_path, fake_target):
    from turin import MODEL_REV

    pipeline.run(_args(tmp_path), log=lambda m: None)
    body = open(os.path.join(str(tmp_path),
                             "KOI-1.01_lineph_summary.csv")).read().splitlines()
    assert f"rev{MODEL_REV}" in body[1]
    assert outputs.load_resume(
        str(tmp_path), "KOI-1.01", "lineph").model_rev == MODEL_REV


def test_hurin_parity_mode_runs(tmp_path, fake_target):
    """The parity configuration must be a working configuration, not just flags."""
    assert pipeline.run(
        _args(tmp_path, geometry="chord", profile_mode="ratio"),
        log=lambda m: None) in FINISHED
    state = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    assert (state.geometry, state.profile_mode) == ("chord", "ratio")


def test_an_unconverged_run_exits_3_and_stamps_its_products(tmp_path,
                                                            fake_target):
    """40 draws/chain cannot meet the gates: the run must still finish,
    write every product, exit EXIT_UNCONVERGED, and say so on line 2."""
    logs = []
    code = pipeline.run(_args(tmp_path), log=logs.append)
    assert code == pipeline.EXIT_UNCONVERGED
    assert any("NOT CONVERGED" in l for l in logs)
    for name in ("summary", "logrho", "lcdata"):
        with open(os.path.join(str(tmp_path),
                               f"KOI-1.01_lineph_{name}.csv")) as fh:
            line2 = fh.read().splitlines()[1]
        assert line2.startswith("# turin") and "UNCONVERGED" in line2
    # still readable by hurin-style analysis scripts
    arr = np.genfromtxt(os.path.join(str(tmp_path),
                                     "KOI-1.01_lineph_summary.csv"),
                        delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert "dP" in [str(r["Parameter"]) for r in np.atleast_1d(arr)]


def test_report_outcome_maps_verdicts_to_exit_codes():
    class V:
        def __init__(self, ok):
            self.converged, self.n_draws = ok, 300
            self.worst_rhat = ("k", 1.02)

    log = []
    assert pipeline._report_outcome("T", {"lineph": V(True)}, log.append) == 0
    assert pipeline._report_outcome(
        "T", {"lineph": V(True), "ttv": V(False)}, log.append) == 3
    assert pipeline._report_outcome(
        "T", {"lineph": "converged earlier"}, log.append) == 0


def test_replot_rebuilds_figures_without_sampling(tmp_path, fake_target):
    """--replot redraws the figures and ttv_times from the saved products:
    the times file comes back identical (same layout, likelihood and ML row
    as the fit), the summary and chains are not touched, and nothing
    samples."""
    assert pipeline.run(_args(tmp_path, modes=("lineph", "ttv")),
                        log=lambda m: None) in FINISHED
    d = str(tmp_path)
    times = os.path.join(d, "KOI-1.01_ttv_times.csv")
    summ = os.path.join(d, "KOI-1.01_ttv_summary.csv")
    corner_pdf = os.path.join(d, "KOI-1.01_ttv_corner.pdf")
    before_times = open(times).read().splitlines()
    before_summary = open(summ).read()
    os.utime(corner_pdf, (0, 0))

    logs = []
    args = _args(tmp_path, modes=("lineph", "ttv"))
    args.replot = True
    assert pipeline.run(args, log=logs.append) == 0
    assert not any("round 0" in l for l in logs)          # no sampling
    after_times = open(times).read().splitlines()
    assert after_times[0] == before_times[0]
    assert after_times[1].startswith("# turin")           # line 2 = provenance
    # the same rows: replot reads the medians and sds back from the summary
    # CSV (10 significant figures), so values derived from them match to
    # that precision -- the O-C column (minutes, sub-ms), the errors, and
    # the ML-model statistics snr and chi2, which move by up to ~1e-3 and are
    # written to 3 and 2 decimals, so a value on a rounding boundary flips
    # its last digit (chi2 39.985339 vs 39.984620; snr 73.496 vs 73.497)
    old = np.array([[float(x) for x in r.split(",")] for r in before_times[2:]])
    new = np.array([[float(x) for x in r.split(",")] for r in after_times[2:]])
    oc, snr, chi2 = 3, 5, 7
    np.testing.assert_allclose(np.delete(new, [oc, snr, chi2], 1),
                               np.delete(old, [oc, snr, chi2], 1), rtol=1e-9)
    np.testing.assert_allclose(new[:, oc], old[:, oc], rtol=0, atol=1e-3)
    np.testing.assert_allclose(new[:, snr], old[:, snr], rtol=0, atol=0.0011)
    np.testing.assert_allclose(new[:, chi2], old[:, chi2], rtol=0, atol=0.011)
    assert open(summ).read() == before_summary
    assert os.path.getmtime(corner_pdf) > 0               # rewritten


def test_replot_refuses_a_fit_whose_epochs_this_turin_would_not_select(
        tmp_path, fake_target):
    """A fit made before an epoch-selection change (0.1.55 fits epochs whose
    transit fell in a gap) has a different epoch count from this turin's
    segmentation. --replot must refuse it with the reason, not fail
    building a likelihood whose shapes no longer line up."""
    assert pipeline.run(_args(tmp_path, modes=("lineph",)),
                        log=lambda m: None) in FINISHED
    d = str(tmp_path)
    st = outputs.load_resume(d, "KOI-1.01", "lineph")
    n = len(st.legendre_orders)
    st.legendre_orders = list(st.legendre_orders) + [0]  # one epoch more
    outputs.save_resume(d, "KOI-1.01", "lineph", st)
    logs = []
    args = _args(tmp_path, modes=("lineph",))
    args.replot = True
    assert pipeline.run(args, log=logs.append) == 1
    assert any(f"fitted {n + 1} epochs, but this turin selects {n}" in l
               for l in logs)


def test_replot_refuses_a_fit_its_model_does_not_reproduce(tmp_path,
                                                           fake_target):
    """--replot's "same model" test is the replayed best log-density, not
    MODEL_REV: an older revision replots when it reproduces, and a fit whose
    stored log-density the current model misses (here by 1 nat) is refused
    and left untouched."""
    import io
    import tarfile

    import pandas as pd

    assert pipeline.run(_args(tmp_path, modes=("lineph",)),
                        log=lambda m: None) in FINISHED
    d = str(tmp_path)
    fold = os.path.join(d, "KOI-1.01_lineph_fold.pdf")

    # an older MODEL_REV that reproduces: replotted
    st = outputs.load_resume(d, "KOI-1.01", "lineph")
    st.model_rev = _MODEL_REV - 1
    outputs.save_resume(d, "KOI-1.01", "lineph", st)
    logs = []
    args = _args(tmp_path, modes=("lineph",))
    args.replot = True
    assert pipeline.run(args, log=logs.append) == 0
    assert any("replotting" in l and f"MODEL_REV {_MODEL_REV - 1}" in l
               for l in logs)

    # a stored log-density 1 nat off: refused, figures untouched
    path = os.path.join(d, "KOI-1.01_lineph_chains.csv.tar.gz")
    with tarfile.open(path) as tf:
        members = [(m, tf.extractfile(m).read()) for m in tf.getmembers()]
    text = members[0][1].decode()
    header, stamp = text.splitlines()[:2]
    df = pd.read_csv(io.StringIO(text), comment="#")
    df["loglike"] += 1.0
    body = "\n".join([header, stamp]) + "\n" + df.to_csv(
        index=False, header=False)
    with tarfile.open(path, "w:gz") as tf:
        for (m, data), new in zip(members, [body.encode(), None]):
            data = new if new is not None else data
            m.size = len(data)
            tf.addfile(m, io.BytesIO(data))
    os.utime(fold, (0, 0))
    logs = []
    assert pipeline.run(args, log=logs.append) == 1
    assert any("does not reproduce" in l for l in logs)
    assert os.path.getmtime(fold) == 0


def test_ttv_is_skipped_below_the_per_transit_snr_gate(tmp_path, fake_target):
    """--TTVsnr: with the median single-transit SNR below the threshold the
    TTV fit is skipped -- no sampling, no TTV products, a skipped CSV with
    the reason and each epoch's SNR, exit 0 -- and a later run that passes
    the gate fits TTVs and removes the stale note."""
    logs = []
    args = _args(tmp_path, modes=("lineph", "ttv"), ttv_snr_min=1e6)
    assert pipeline.run(args, log=logs.append) in FINISHED   # LinEph's verdict
    d = str(tmp_path)
    names = os.listdir(d)
    assert "KOI-1.01_ttv_skipped.csv" in names
    assert not any(n.startswith("KOI-1.01_ttv_") and n != "KOI-1.01_ttv_skipped.csv"
                   for n in names), names
    assert any("TTV fit SKIPPED" in l for l in logs)
    assert any("ttv: skipped: median expected SNR" in l for l in logs)
    lines = open(os.path.join(d, "KOI-1.01_ttv_skipped.csv")).read().splitlines()
    assert lines[0] == "epoch,expected_snr" and lines[1].startswith("# turin")
    assert lines[2].startswith("# skipped:")
    snr = np.array([float(r.split(",")[1]) for r in lines[3:]])
    assert np.sum(snr > 0) >= 10        # (an epoch whose transit fell in a gap
                                        # is listed at exactly 0)

    logs = []
    args = _args(tmp_path, modes=("lineph", "ttv"), ttv_snr_min=0.5)
    assert pipeline.run(args, log=logs.append) in FINISHED
    assert any("fitting" in l and "per-transit SNR" in l for l in logs)
    assert "KOI-1.01_ttv_skipped.csv" not in os.listdir(d)
    assert "KOI-1.01_ttv_summary.csv" in os.listdir(d)


def test_ttv_is_skipped_with_too_few_transits_whatever_the_snr_gate(
        tmp_path, fake_target, monkeypatch):
    """Fewer than MIN_TTV_TRANSITS epochs with data in transit: no TTV fit,
    a skipped CSV naming the count, and --TTVsnr=0 does not lift it. At the
    real minimum the same fixture fits (the test above)."""
    monkeypatch.setattr(pipeline, "MIN_TTV_TRANSITS", 10_000)
    logs = []
    args = _args(tmp_path, modes=("lineph", "ttv"), ttv_snr_min=0.0)
    assert pipeline.run(args, log=logs.append) in FINISHED
    d = str(tmp_path)
    names = os.listdir(d)
    assert "KOI-1.01_ttv_skipped.csv" in names
    assert "KOI-1.01_ttv_summary.csv" not in names
    assert any("TTV fit SKIPPED" in l and "with data in transit, fewer than "
               "the 10000" in l for l in logs)
    note = open(os.path.join(d, "KOI-1.01_ttv_skipped.csv")).read()
    assert "# skipped:" in note and "fewer than the 10000" in note


def test_mle_ttv_refuses_with_too_few_transits(tmp_path, fake_target,
                                               monkeypatch):
    """--mle's TTV fit has the same floor, whatever --SNRgate says."""
    monkeypatch.setattr(pipeline, "MIN_TTV_TRANSITS", 10_000)
    logs = []
    assert pipeline.run(_args(tmp_path, modes=("lineph", "ttv"), mle=True,
                              snr_gate=0.0), log=logs.append) == 0
    assert any("TTV fit REFUSED" in l and "fewer than the 10000" in l
               for l in logs)
    assert not any(n.startswith("KOI-1.01_ttv_mle_")
                   for n in os.listdir(str(tmp_path)))


def test_transit_snrs_follow_the_data_present(fake_target):
    """Expected SNR scales as 1/sigma, and a transit with its in-transit
    points removed scores lower; one with none scores 0."""
    pr = prep.prepare_data("KOI-1.01", log=lambda *a: None)
    cen = prep.centering_constants(pr.epoch_data, pr.eph)
    shape = dict(k=K_TRUE, beta=0.3, T14=T14_TRUE, q1=Q1_TRUE, q2=Q2_TRUE)
    s1 = pipeline.transit_snrs(pr, cen, shape)
    covered = s1 > 1e-3                 # the fixture has one transit in a gap
    assert covered.sum() >= 10 and np.all(s1[~covered] == 0)
    i = int(np.argmax(covered))

    ed = pr.epoch_data
    ed["ferr_padded"] = ed["ferr_padded"] * 2.0
    s2 = pipeline.transit_snrs(pr, cen, shape)
    np.testing.assert_allclose(s2, s1 / 2.0, rtol=1e-5)

    # drop a covered epoch's points within T14/2 of its centre: no data in
    # transit, and the other epochs are unchanged
    t = ed["times_padded"][i] - ed["epoch_centers"][i]
    ed["mask"][i, np.abs(t) < 0.6 * T14_TRUE] = 0
    s3 = pipeline.transit_snrs(pr, cen, shape)
    others = np.arange(s3.size) != i
    assert s3[i] < 1e-3
    np.testing.assert_allclose(s3[others], s2[others], rtol=1e-5)


# -- maximum likelihood (--mle) -------------------------------------------------

def _mle_rows(path):
    rows = {}
    with open(path) as fh:
        fh.readline()
        line2 = fh.readline()
        for line in fh:
            if line.startswith("#"):
                continue
            name, *rest = line.strip().split(",")
            rows[name] = rest
    return line2, rows


def test_mle_fits_without_sampling_and_leaves_a_lineage_alone(tmp_path,
                                                              fake_target):
    """--mle writes its own fit, lcdata and fold products, stamped as an
    MLE; recovers the injection; and never touches an MCMC lineage of the
    same target (its resume state is left byte for byte)."""
    out = str(tmp_path)
    lineage = os.path.join(out, "KOI-1.01_lineph_resume.pkl")
    with open(lineage, "wb") as fh:
        fh.write(b"not a real state")
    logs = []
    assert pipeline.run(_args(tmp_path, mle=True), log=logs.append) == 0
    names = sorted(os.listdir(out))
    for suffix in ("_lineph_mle_fit.csv", "_lineph_mle_lcdata.csv",
                   "_lineph_mle_fold.pdf"):
        assert any(n.endswith(suffix) for n in names), (suffix, names)
    assert not any(n.endswith("_lineph_summary.csv") for n in names)
    with open(lineage, "rb") as fh:
        assert fh.read() == b"not a real state"
    text = "\n".join(logs)
    assert "maximum likelihood" in text and "MAP log-density" not in text

    line2, rows = _mle_rows(os.path.join(out, "KOI-1.01_lineph_mle_fit.csv"))
    assert line2.startswith("# turin ") and "maximum likelihood" in line2
    assert list(rows)[:7] == ["dP", "dtau0", "k", "beta", "T14", "q1", "q2"]
    for n in ("b", "log10_rho", "lnL", "n_points", "n_params", "BIC", "AIC"):
        assert n in rows
    assert all(len(v) == 1 for v in rows.values())           # no error bars
    assert abs(float(rows["k"][0]) - K_TRUE) < 0.1 * K_TRUE
    lnL, n_par = float(rows["lnL"][0]), int(rows["n_params"][0])
    n_pts = int(rows["n_points"][0])
    assert float(rows["BIC"][0]) == pytest.approx(
        n_par * np.log(n_pts) - 2 * lnL, rel=1e-9)

    # --fresh --mle clears the MLE fit only, never the lineage (0.1.80) ...
    logs = []
    assert pipeline.run(_args(tmp_path, mle=True, fresh=True),
                        log=logs.append) == 0
    assert "removed 3 existing" in "\n".join(logs)
    with open(lineage, "rb") as fh:
        assert fh.read() == b"not a real state"
    # ... and an MCMC --fresh clears the lineage, never the MLE fit
    removed = outputs.clear_products(out, "KOI-1.01")
    assert removed == ["KOI-1.01_lineph_resume.pkl"]
    assert os.path.exists(os.path.join(out, "KOI-1.01_lineph_mle_fit.csv"))


def test_mle_fits_ttv_from_lineph_and_recovers_the_times(tmp_path,
                                                         fake_ttv_target,
                                                         monkeypatch):
    """--mle with TTV (0.1.82): LinEph first, then the TTV fit from it; the
    ttv_mle products; the injected times recovered; TTV never below LinEph;
    an MCMC TTV lineage and its skipped note untouched. No anvil.optimize
    call asks for the Laplace step (0.1.83): this target has an epoch in a
    gap, whose zero Hessian row can make anvil's inverse raise."""
    import anvil
    real, calls = anvil.optimize, []

    def recording(*a, **k):
        calls.append(k.get("hessian", True))
        return real(*a, **k)

    monkeypatch.setattr(anvil, "optimize", recording)
    out = str(tmp_path)
    keep = {"KOI-1.01_ttv_resume.pkl": b"not a real state",
            "KOI-1.01_ttv_skipped.csv": b"an old note"}
    for name, body in keep.items():
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(body)
    logs = []
    assert pipeline.run(_args(tmp_path, mle=True, modes=("lineph", "ttv")),
                        log=logs.append) == 0
    names = sorted(os.listdir(out))
    for suffix in ("fit.csv", "times.csv", "lcdata.csv", "fold.pdf",
                   "oc.pdf", "compare.csv"):
        assert f"KOI-1.01_ttv_mle_{suffix}" in names, (suffix, names)
    for name, body in keep.items():
        with open(os.path.join(out, name), "rb") as fh:
            assert fh.read() == body
    text = "\n".join(logs)
    assert "ttv maximum likelihood" in text and "ttv vs lineph" in text

    _, rows = _mle_rows(os.path.join(out, "KOI-1.01_ttv_mle_compare.csv"))
    assert float(rows["dlnL"][0]) > 50          # a 12-minute TTV, SNR ~ 100
    times = np.genfromtxt(os.path.join(out, "KOI-1.01_ttv_mle_times.csv"),
                          delimiter=",", names=True, comments="#")
    pr = prep.prepare_data("KOI-1.01", log=lambda *a: None)
    cen = prep.centering_constants(pr.epoch_data, pr.eph)
    truth = (np.asarray(cen["centers_abs"]) + np.asarray(cen["d_arr"])
             + _ttv_truth(pr.epoch_data["n_epochs"]))
    has = times["snr"] > 5
    assert has.sum() >= 8
    np.testing.assert_array_less(np.abs(times["tmid"] - truth)[has] * 1440,
                                 2.0)                        # minutes
    assert np.all(np.isnan(times["tmid_err"]))
    assert len(calls) >= 2 and not any(calls), calls


def test_mle_oblate_fits_ttv_round_and_oblate(tmp_path, fake_ttv_target):
    """--mle --oblate with TTV (0.1.84): the round TTV fit, then the oblate
    one from it; the oblate fit's products plus round.csv; compare.csv
    holds TTV vs LinEph and oblate vs round; the oblate fit nests both
    (never below either); the injected times recovered."""
    out = str(tmp_path)
    logs = []
    assert pipeline.run(_args(tmp_path, mle=True, law="hybrid4", oblate=True,
                              modes=("lineph", "ttv")),
                        log=logs.append) == 0
    names = sorted(os.listdir(out))
    for suffix in ("fit.csv", "round.csv", "times.csv", "lcdata.csv",
                   "fold.pdf", "oc.pdf", "compare.csv"):
        assert f"KOI-1.01_ttv_mle_{suffix}" in names, (suffix, names)
    text = "\n".join(logs)
    assert "[oblate ttv hybrid4] certified" in text
    assert "ttv: oblate vs round: dlnL" in text

    _, fit = _mle_rows(os.path.join(out, "KOI-1.01_ttv_mle_fit.csv"))
    _, rnd = _mle_rows(os.path.join(out, "KOI-1.01_ttv_mle_round.csv"))
    _, lin = _mle_rows(os.path.join(out, "KOI-1.01_lineph_mle_fit.csv"))
    assert {"fc", "fs", "f", "theta", "w1"} <= set(fit) and "fc" not in rnd
    _, cmp = _mle_rows(os.path.join(out, "KOI-1.01_ttv_mle_compare.csv"))
    lnl = float(fit["lnL"][0])
    assert float(cmp["dlnL"][0]) == pytest.approx(lnl - float(lin["lnL"][0]))
    assert float(cmp["dlnL_oblate"][0]) == pytest.approx(
        lnl - float(rnd["lnL"][0]))
    assert float(cmp["dlnL"][0]) > 50                # the injected TTV
    assert -1e-3 < float(cmp["dlnL_oblate"][0]) < 5.0  # a round truth
    times = np.genfromtxt(os.path.join(out, "KOI-1.01_ttv_mle_times.csv"),
                          delimiter=",", names=True, comments="#")
    pr = prep.prepare_data("KOI-1.01", log=lambda *a: None)
    cen = prep.centering_constants(pr.epoch_data, pr.eph)
    truth = (np.asarray(cen["centers_abs"]) + np.asarray(cen["d_arr"])
             + _ttv_truth(pr.epoch_data["n_epochs"]))
    has = times["snr"] > 5
    assert has.sum() >= 8
    np.testing.assert_array_less(np.abs(times["tmid"] - truth)[has] * 1440,
                                 2.0)


def test_a_gpu_kill_in_the_oblate_ttv_fit_keeps_the_round_ttv_fit(
        tmp_path, fake_ttv_target, monkeypatch):
    """0.1.92: macOS killed a long GPU launch in the oblate TTV polish of two
    big scan targets (KOI-7259.01 after 4.5 h, KOI-1631.01 after 13.8 h).
    The round TTV fit is on disk before the oblate one starts, the failure is
    explained in the log, and it still propagates (the target did not
    finish)."""
    real = mle.fit_ttv

    def killed(*a, **kw):
        if kw.get("oblate"):
            raise RuntimeError("[METAL] Command buffer execution failed: "
                               "Impacting Interactivity")
        return real(*a, **kw)

    monkeypatch.setattr(mle, "fit_ttv", killed)
    logs = []
    with pytest.raises(RuntimeError, match="Impacting Interactivity"):
        pipeline.run(_args(tmp_path, mle=True, law="hybrid4", oblate=True,
                           modes=("lineph", "ttv")), log=logs.append)
    names = os.listdir(str(tmp_path))
    assert "KOI-1.01_ttv_mle_round.csv" in names
    assert "KOI-1.01_lineph_mle_fit.csv" in names
    assert "KOI-1.01_ttv_mle_fit.csv" not in names
    assert "oblate ttv FAILED" in "\n".join(logs)


def test_mle_ttv_gate_refuses_weak_transits(tmp_path, fake_target):
    """--SNRgate: below it the run stops after LinEph, with the LinEph
    products and no TTV ones."""
    logs = []
    assert pipeline.run(_args(tmp_path, mle=True, modes=("lineph", "ttv"),
                              snr_gate=1e6), log=logs.append) == 0
    names = os.listdir(str(tmp_path))
    assert "KOI-1.01_lineph_mle_fit.csv" in names
    assert not any("_ttv_" in n for n in names), names
    assert any("TTV fit REFUSED" in l and "--SNRgate=1e+06" in l
               for l in logs)


def test_mle_oblate_fits_the_round_model_too(tmp_path, fake_target):
    """--mle --oblate (hybrid4): the oblate fit, the round fit and their
    comparison. The injection is round, so the nested oblate fit gains
    almost nothing, and BIC prefers round."""
    out = str(tmp_path)
    logs = []
    assert pipeline.run(_args(tmp_path, mle=True, law="hybrid4", oblate=True),
                        log=logs.append) == 0
    _, fit = _mle_rows(os.path.join(out, "KOI-1.01_lineph_mle_fit.csv"))
    _, rnd = _mle_rows(os.path.join(out, "KOI-1.01_lineph_mle_round.csv"))
    assert {"fc", "fs", "f", "theta", "A", "B", "w1", "w4"} <= set(fit)
    assert "fc" not in rnd and "w1" in rnd
    _, cmp = _mle_rows(os.path.join(out, "KOI-1.01_lineph_mle_compare.csv"))
    dlnL = float(cmp["dlnL"][0])
    assert dlnL == pytest.approx(float(fit["lnL"][0]) - float(rnd["lnL"][0]))
    assert -1e-2 < dlnL < 5.0
    assert float(cmp["dBIC"][0]) > 0
    assert "oblate vs round: dlnL" in "\n".join(logs)
