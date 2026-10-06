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

from turin import cli, model as M, pipeline, prep, outputs

P_TRUE, EPOCH_TRUE, DUR_H = 6.2431, 131.7, 3.1
K_TRUE, B_TRUE, Q1_TRUE, Q2_TRUE = 0.10, 0.28, 0.32, 0.24
T14_TRUE = DUR_H / 24.0
YERR = 3e-4
EPH = {"period": P_TRUE, "epoch": EPOCH_TRUE, "duration": DUR_H,
       "depth": 1e6 * K_TRUE**2, "ingress": 0.35}


@pytest.fixture
def fake_target(monkeypatch):
    """Patch data acquisition to serve a synthetic Kepler-like light curve."""
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
        f = np.array(M.transit_flux(
            grid, mid=mx.zeros((1, ed["n_epochs"]), dtype=mx.float64),
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
    assert "dim 5" in text and "limb darkening: collapsed" in text
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


def test_collapsed_refuses_an_old_metalplanet(tmp_path, fake_target,
                                              monkeypatch):
    import metalplanet

    monkeypatch.setattr(metalplanet, "__version__", "0.6.1")
    with pytest.raises(SystemExit, match="MetalPlanet >= 0.7.0"):
        pipeline.run(_args(tmp_path, ld="collapsed"), log=lambda m: None)
    # the default path is not gated on it
    assert pipeline.run(_args(tmp_path), log=lambda m: None) in FINISHED


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
