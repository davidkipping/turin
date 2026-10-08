"""The CLI grammar and the output products.

The product formats are checked against hurin's real files where those exist,
because "matches hurin" is the actual requirement: existing analysis scripts
read these by column name, and `np.genfromtxt(names=True)` is picky about
where the provenance comment goes.
"""

from __future__ import annotations

import os
import tarfile

import numpy as np
import pytest

from turin import cli, outputs


# -- CLI ---------------------------------------------------------------

def test_target_forms():
    for tok, want in (("--KOI-448.02", "KOI-448.02"),
                      ("--TOI-406.01", "TOI-406.01"),
                      ("--koi-999.02", "KOI-999.02")):
        assert cli.parse_args([tok]).target == want


def test_defaults_match_the_documented_ones():
    a = cli.parse_args(["--KOI-1.01"])
    assert a.b_prior == "transiting"
    # omitting --PL asks turin to measure and choose
    assert a.profile_mode == "auto"
    assert a.geometry == "circular"
    assert a.sampler == "chees"
    assert a.modes == ("lineph", "ttv")
    assert a.ttv_max_days is None
    assert a.chains == 512 and a.max_leapfrog == 128
    assert a.ld == "sampled"


def test_nongrazing_is_an_alias():
    assert cli.parse_args(["--KOI-1.01", "--nongrazing"]).b_prior == "nongrazing"


def test_ttvmax_is_minutes_on_the_command_line():
    a = cli.parse_args(["--KOI-1.01", "--TTVmax=720"])
    assert a.ttv_max_min == 720.0
    np.testing.assert_allclose(a.ttv_max_days, 0.5)


def test_hurin_parity_combination_parses():
    """Against hurin >= 0.1.68 the orbit is the only model difference."""
    a = cli.parse_args(["--KOI-1.01", "--geometry=chord", "--PL=ratio"])
    assert (a.geometry, a.profile_mode) == ("chord", "ratio")


@pytest.mark.parametrize("mode", ["auto", "exact", "hybrid", "ratio"])
def test_every_pl_mode_is_accepted(mode):
    assert cli.parse_args(["--KOI-1.01", f"--PL={mode}"]).profile_mode == mode


def test_modes_subset():
    assert cli.parse_args(["--KOI-1.01", "--modes=lineph"]).modes == ("lineph",)
    assert cli.parse_args(["--KOI-1.01", "--modes=ttv"]).modes == ("ttv",)


@pytest.mark.parametrize("argv,pattern", [
    (["--KOI-1.01", "--chanis=8"], "unrecognized"),
    (["--KOI-1.01", "--bprior=nonsense"], "not one of"),
    (["--KOI-1.01", "--sampler=nuts"], "not one of"),
    (["--KOI-1.01", "--PL=approx"], "not one of"),
    (["--KOI-1.01", "--geometry=kepler"], "not one of"),
    (["--KOI-1.01", "--modes=lineph,nope"], "not one of"),
    (["--KOI-1.01", "--chains=0"], "positive"),
    (["--KOI-1.01", "--chains=abc"], "positive"),
    (["--KOI-1.01", "--TTVmax=-5"], "positive"),
    (["--KOI-1.01", "--seed=-1"], "non-negative integer"),
    (["--KOI-1.01", "--seed=abc"], "non-negative integer"),
    (["--KOI-1.01", "--tag=bad tag"], "tag"),
    (["--KOI-1.01", "--KOI-2.01"], "more than one target"),
    (["KOI-1.01"], "unrecognized"),
    (["--KOI-1.01", "--ld=profile"], "not one of"),
    # collapsed limb darkening: each refusal names its reason
    (["--KOI-1.01", "--ld=collapsed", "--modes=lineph", "--PL=ratio"],
     "envelope theorem"),
    (["--KOI-1.01", "--ld=collapsed", "--modes=lineph", "--PL=hybrid"],
     "PL=exact"),
    (["--KOI-1.01", "--ld=collapsed"], "modes=lineph"),     # default modes
    (["--KOI-1.01", "--ld=collapsed", "--modes=lineph", "--geometry=chord"],
     "circular"),
])
def test_bad_arguments_are_rejected(argv, pattern):
    with pytest.raises(SystemExit, match=pattern):
        cli.parse_args(argv)


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_seed_accepts_zero_its_own_default(seed):
    # --seed=0 was refused as "needs a positive integer", so the default seed
    # could not be named explicitly (0.1.45)
    assert cli.parse_args(["--KOI-1.01", f"--seed={seed}"]).seed == seed
    assert cli.parse_args(["--KOI-1.01"]).seed == 0


@pytest.mark.parametrize("pl", ["auto", "exact"])
def test_collapsed_limb_darkening_parses_with_lineph(pl):
    a = cli.parse_args(["--KOI-1.01", "--ld=collapsed", "--modes=lineph",
                        f"--PL={pl}"])
    assert (a.ld, a.modes, a.profile_mode) == ("collapsed", ("lineph",), pl)
    assert cli.collapsed_ld_problem(a) is None


def test_typo_gets_a_suggestion():
    with pytest.raises(SystemExit, match="Did you mean"):
        cli.parse_args(["--KOI-1.01", "--freshh"])


def test_version_and_help_and_capabilities(capsys):
    from turin import __version__

    assert cli.main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
    assert cli.main(["--help"]) == 0
    assert "usage: turin" in capsys.readouterr().out
    assert cli.main(["--capabilities"]) == 0
    assert "anvil" in capsys.readouterr().out


def test_missing_target_is_an_error():
    with pytest.raises(SystemExit, match="no target"):
        cli.main([])


# -- product formats ---------------------------------------------------

@pytest.fixture(autouse=True)
def _no_tag():
    outputs.set_run_tag(None)
    yield
    outputs.set_run_tag(None)


def test_provenance_is_on_line_two_not_line_one(tmp_path):
    """np.genfromtxt(names=True) treats a LEADING comment as the header row."""
    draws = np.random.default_rng(0).normal(size=(50, 2))
    summary = outputs.summarize(draws, ["k", "T14"], rhat=[1.001, 1.002],
                                ess_bulk=[900, 800], ess_tail=[850, 700])
    path = outputs.export_summary(str(tmp_path), "KOI-1.01", "lineph", summary)

    lines = open(path).read().splitlines()
    assert lines[0].startswith("Parameter,Median")
    assert lines[1].startswith("# turin ")
    # and the file is actually readable the way hurin's consumers read it
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None,
                        encoding="utf-8")
    assert "Parameter" in arr.dtype.names and "Bulk_ESS" in arr.dtype.names
    assert set(np.atleast_1d(arr["Parameter"])) == {"k", "T14"}


def test_summary_columns_match_hurin():
    hurin_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "hurin", "KOI-448.02",
        "KOI-448.02_lineph_summary.csv")
    if not os.path.exists(hurin_file):
        pytest.skip("no hurin products to compare against")
    assert (open(hurin_file).readline().strip()
            == "Parameter,Median,Std,16th,84th,R-hat,Bulk_ESS,Tail_ESS")


def test_chains_tarball_layout(tmp_path):
    """The CSV must be member 0; readers use getmembers()[0]."""
    cols = ["k", "T14", "loglike"]
    arrays = [np.arange(5.0), np.arange(5.0) * 2, np.arange(5.0) - 10]
    path = outputs.export_chains(str(tmp_path), "KOI-1.01", "ttv", cols, arrays)
    with tarfile.open(path) as tar:
        members = tar.getmembers()
        assert members[0].name.endswith("_ttv_chains.csv")
        assert members[1].name == "turin_version.txt"
        text = tar.extractfile(members[0]).read().decode()
    lines = text.splitlines()
    assert lines[0] == "k,T14,loglike"
    assert lines[1].startswith("# turin ")
    assert len(lines) == 2 + 5


def test_lcdata_drops_padding_and_sorts_by_time(tmp_path):
    ed = {
        "times_padded": np.array([[5.0, 4.0, 99.0], [1.0, 2.0, 99.0]]),
        "flux_padded": np.array([[1.1, 1.2, 1.0], [1.3, 1.4, 1.0]]),
        "ferr_padded": np.array([[0.1, 0.1, 1e10], [0.1, 0.1, 1e10]]),
        "mask": np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
    }
    model = np.array([[1.05, 1.15, 1.0], [1.25, 1.35, 1.0]])
    path = outputs.export_lcdata(str(tmp_path), "KOI-1.01", "lineph", ed, model)
    arr = np.genfromtxt(path, delimiter=",", names=True, skip_header=0)
    arr = np.atleast_1d(arr)
    # padded rows gone, times ascending
    assert arr.size == 4
    t = arr["time"]
    assert np.all(np.diff(t) > 0)
    assert 99.0 not in t


def test_tag_namespaces_every_product(tmp_path):
    outputs.set_run_tag("r2")
    try:
        assert outputs.tag_part() == ".r2"
        p = outputs.product_path(str(tmp_path), "KOI-1.01", "ttv", "fold", "pdf")
        assert p.endswith("KOI-1.01_ttv_fold.r2.pdf")
    finally:
        outputs.set_run_tag(None)
    assert outputs.product_path(str(tmp_path), "KOI-1.01", "ttv", "fold",
                                "pdf").endswith("KOI-1.01_ttv_fold.pdf")
    with pytest.raises(ValueError, match="tag"):
        outputs.set_run_tag("bad tag")


def test_ttv_times_columns(tmp_path):
    rows = [dict(epoch=-3, tmid=150.5, tmid_err=0.002, ttv_min=-4.2,
                 ttv_err_min=2.9, snr=16.8, npts=89, chi2=95.48)]
    path = outputs.export_ttv_times(str(tmp_path), "KOI-1.01", "ttv", rows)
    lines = open(path).read().splitlines()
    assert lines[0] == "epoch,tmid,tmid_err,ttv_min,ttv_err_min,snr,npts,chi2"
    assert lines[2].startswith("-3,")


def test_resume_state_guards_refuse_a_mismatch():
    state = outputs.ResumeState(
        target="KOI-1.01", mode="lineph", turin_version="0.1.0",
        launch_command="turin --KOI-1.01", tag=None, b_prior="transiting",
        profile_mode="exact", geometry="circular",
        sampler="chees", n_chains=512, ttv_max=None, n_durations=5.0,
        legendre_orders=np.zeros(3), exposure_time=0.02, num_resample=7,
        n_samples_done=300, done=False)

    state.check(b_prior="transiting", profile_mode="exact")   # agrees
    for bad in (dict(b_prior="box"), dict(profile_mode="ratio"),
                dict(geometry="chord"), dict(sampler="ensemble")):
        with pytest.raises(SystemExit, match="use --|Use --"):
            state.check(**bad)


def test_resume_state_round_trips(tmp_path):
    state = outputs.ResumeState(
        target="KOI-1.01", mode="ttv", turin_version="0.1.0",
        launch_command="turin --KOI-1.01", tag=None, b_prior="nongrazing",
        profile_mode="ratio", geometry="chord",
        sampler="ensemble", n_chains=128, ttv_max=0.5, n_durations=7.0,
        legendre_orders=np.array([1, 2, 3]), exposure_time=0.02,
        num_resample=5, n_samples_done=1200, done=True,
        ml_params={"k": 0.1}, last_u=np.zeros((4, 8), dtype=np.float32))
    outputs.save_resume(str(tmp_path), "KOI-1.01", "ttv", state)
    back = outputs.load_resume(str(tmp_path), "KOI-1.01", "ttv")
    assert back.done and back.n_samples_done == 1200
    assert back.ttv_max == 0.5 and back.ml_params == {"k": 0.1}
    assert back.last_u.shape == (4, 8)
    assert outputs.load_resume(str(tmp_path), "KOI-9.99", "ttv") is None


def test_clear_products_respects_the_tag(tmp_path):
    for name in ("KOI-1.01_lineph_summary.csv",
                 "KOI-1.01_lineph_summary.r2.csv",
                 "KOI-1.01_ttv_fold.pdf"):
        open(os.path.join(tmp_path, name), "w").close()

    removed = outputs.clear_products(str(tmp_path), "KOI-1.01")
    assert "KOI-1.01_lineph_summary.csv" in removed
    assert "KOI-1.01_ttv_fold.pdf" in removed
    # the tagged lineage is a different run and must survive
    assert "KOI-1.01_lineph_summary.r2.csv" not in removed
    assert os.path.exists(os.path.join(tmp_path,
                                       "KOI-1.01_lineph_summary.r2.csv"))


def test_fit_linear_ephemeris_weights_by_timing_error():
    """A time from a chain stuck in the wrong mode carries a large error and
    so barely moves the line; a well-measured TTV is not discarded but
    shows up in chi2 (the old MAD clip threw out real TTVs)."""
    epochs = np.arange(12.0)
    P, tau0 = 9.3456, 120.5
    times = tau0 + P * epochs
    err = np.full(12, 1e-3)

    wrong = times.copy()
    wrong[7] += 0.5                      # badly wrong, but known to be poor
    poor = err.copy()
    poor[7] = 1.0
    P_fit, tau0_fit, chi2 = outputs.fit_linear_ephemeris(epochs, wrong, poor)
    np.testing.assert_allclose(P_fit, P, atol=1e-5)
    np.testing.assert_allclose(tau0_fit, tau0, atol=1e-4)
    assert chi2 < 1.0

    ttv = times.copy()
    ttv[7] += 0.03                       # a real, well-measured TTV: 30 sigma
    _, _, chi2 = outputs.fit_linear_ephemeris(epochs, ttv, err)
    assert chi2 > 500.0

    # no errors: equal weights, an ordinary straight-line fit
    P_u, tau0_u, _ = outputs.fit_linear_ephemeris(epochs, times)
    np.testing.assert_allclose([P_u, tau0_u], [P, tau0], atol=1e-8)


def test_fit_linear_ephemeris_on_koi_2686():
    """The measured KOI-2686.01 times: all seven contribute (no clipping),
    and the timings are decisively non-linear."""
    n = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    oc = np.array([17.6, -49.5, 10.0, 13.9, 20.5, 19.2, -31.7]) / 1440.0
    err = np.array([2.5, 2.3, 2.6, 3.0, 3.2, 2.5, 2.7]) / 1440.0
    times = 1000.0 + 211.03 * n + oc
    P, tau0, chi2 = outputs.fit_linear_ephemeris(n, times, err)
    assert 700 < chi2 < 820            # 761 for 5 dof
    resid = (times - (tau0 + P * n)) * 1440.0
    assert resid[1] < -40              # epoch -2 is a TTV, not a reject


def test_summarize_percentiles_and_diagnostics():
    rng = np.random.default_rng(1)
    draws = rng.normal(3.0, 0.5, (20000, 1))
    s = outputs.summarize(draws, ["x"], rhat=[1.004], ess_bulk=[1234],
                          ess_tail=[999])["x"]
    assert abs(s["median"] - 3.0) < 0.02
    assert abs(s["std"] - 0.5) < 0.02
    assert s["lo"] < s["median"] < s["hi"]
    assert s["rhat"] == 1.004 and s["ess_bulk"] == 1234 and s["ess_tail"] == 999


# -- likelihood revision flagging --------------------------------------

def _state(**over):
    base = dict(
        target="KOI-1.01", mode="lineph", turin_version="0.1.0",
        launch_command="turin --KOI-1.01", tag=None, b_prior="transiting",
        profile_mode="exact", geometry="circular", sampler="chees",
        n_chains=512, ttv_max=None, n_durations=5.0,
        legendre_orders=np.zeros(3), exposure_time=0.02, num_resample=7,
        n_samples_done=300, done=False)
    base.update(over)
    return outputs.ResumeState(**base)


def test_ld_is_a_resume_guard_and_old_states_read_sampled():
    """A lineage sampled with q1, q2 and one that integrated them out are
    different targets in different dimensions; never pool them. States
    written before the field existed sampled q1, q2."""
    assert "ld" in outputs.ResumeState.GUARDS
    assert _state().ld == "sampled"
    _state(ld="collapsed").check(ld="collapsed")          # must not raise
    with pytest.raises(SystemExit, match="ld"):
        _state(ld="collapsed").check(ld="sampled")
    with pytest.raises(SystemExit, match="ld"):
        _state().check(ld="collapsed")


def test_model_rev_defaults_to_one_for_states_written_before_the_field():
    assert _state().model_rev == 1


def test_current_model_rev_passes():
    from turin import MODEL_REV

    _state(model_rev=MODEL_REV).check_model_rev()      # must not raise


def test_an_older_model_rev_refuses_to_continue():
    from turin import MODEL_REV

    with pytest.raises(SystemExit, match="cannot be continued"):
        _state(model_rev=MODEL_REV - 1).check_model_rev()
    # the message must name both revisions and the way out
    try:
        _state(model_rev=MODEL_REV - 1).check_model_rev()
    except SystemExit as exc:
        text = str(exc)
        assert str(MODEL_REV) in text and str(MODEL_REV - 1) in text
        assert "--fresh" in text


def test_a_done_lineph_state_survives_an_older_revision():
    """It is never continued -- it only seeds the TTV initialization."""
    from turin import MODEL_REV

    logs = []
    _state(model_rev=MODEL_REV - 1, mode="lineph", done=True
           ).check_model_rev(log=logs.append)
    assert any("only seeds" in m for m in logs)

    # but an unfinished one, or a ttv one, still refuses
    with pytest.raises(SystemExit):
        _state(model_rev=MODEL_REV - 1, mode="lineph", done=False
               ).check_model_rev()
    with pytest.raises(SystemExit):
        _state(model_rev=MODEL_REV - 1, mode="ttv", done=True
               ).check_model_rev()


def test_provenance_records_the_model_revision():
    from turin import MODEL_REV

    assert f"rev{MODEL_REV}" in outputs.provenance()


class _FakeSearch:
    """The slice of lightkurve's SearchResult that _restrict_to_star uses."""

    def __init__(self, names):
        self.table = {"target_name": list(names)}

    def __getitem__(self, idx):
        return _FakeSearch([self.table["target_name"][i] for i in idx])

    def __len__(self):
        return len(self.table["target_name"])


def test_restrict_to_star_drops_a_neighbouring_star():
    """KOI-7592.01's real search: its host plus a neighbour, both at 0".
    Downloading every row stitched the two stars together."""
    from turin.data import lightcurve as L

    names = ["kplr008423352"] * 18 + ["kplr008423344"] * 14
    info = {"name": "KOI-7592.01", "type": "koi"}
    kept = L._restrict_to_star(_FakeSearch(names), 8423344, info)
    assert set(kept.table["target_name"]) == {"kplr008423344"}
    assert len(kept) == 14
    with pytest.raises(ValueError, match="several stars"):
        L._restrict_to_star(_FakeSearch(names), None, info)
    with pytest.raises(ValueError, match="not for"):
        L._restrict_to_star(_FakeSearch(names), 1234567, info)
    # one star and no number: nothing to choose between
    assert len(L._restrict_to_star(_FakeSearch(names[:18]), None, info)) == 18
    # TESS target names are bare TIC numbers
    assert L._target_number("261136679") == 261136679
    assert L._target_number("kplr008423344") == 8423344


def test_mixed_target_check_flags_repeated_timestamps():
    from turin.data import lightcurve as L

    one = np.arange(0.0, 30.0, 0.0204)
    assert L._mixed_target_problem(one) is None
    two = np.concatenate([one, one[::3]])           # a second star, same epochs
    assert "repeated timestamps" in L._mixed_target_problem(two)


def test_one_rule_decides_occupied_and_fitted_epochs():
    """An epoch with a single point in its transit zone is both counted as
    occupied and fitted. The two used to disagree (>1 vs >=1), so a target
    could log 2 occupied epochs and fit 3 (KOI-5897.01)."""
    from turin import prep

    P, T0, dur_h = 10.0, 5.0, 4.8
    cad = 29.4 / 1440
    t = np.arange(0.0, 40.0, cad)
    # epoch 2 (tc = 25): remove every in-transit cadence but one
    tc2, half = 25.0, 0.5 * dur_h / 24
    in2 = np.where(np.abs(t - tc2) <= half)[0]
    t = np.delete(t, in2[1:])
    f = np.ones_like(t)
    e = np.full_like(t, 1e-4)

    tw, fw, ew = prep.extract_near_transit_data(t, f, e, P, T0, dur_h)
    ed = prep.segment_epochs(tw, fw, ew, P, T0, dur_h)
    assert np.any(np.isclose(ed["epoch_centers"], tc2))


def test_search_star_queries_by_catalogue_number(monkeypatch):
    """The host's KIC/TIC is searched directly when the archive has it (a
    name can miss, or pull in a neighbour); the name only when it does not.
    A neighbour in the result is still filtered out."""
    from turin.data import lightcurve as L

    queries = []

    def fake_search(query, **kw):
        queries.append((query, kw))
        return _FakeSearch(["kplr008423344"] * 14 + ["kplr008423352"] * 2)

    monkeypatch.setattr(L.lk, "search_lightcurve", fake_search)
    koi = {"name": "KOI-7592.01", "type": "koi"}
    got = L._search_star(koi, 8423344, "Kepler", author="Kepler")
    assert queries[-1] == ("KIC 8423344", {"author": "Kepler"})
    assert set(got.table["target_name"]) == {"kplr008423344"}
    L._search_star({"name": "TOI-406.01", "type": "toi"}, 8423344, "TESS")
    assert queries[-1][0] == "TIC 8423344"
    with pytest.raises(ValueError, match="several stars"):
        L._search_star(koi, None, "Kepler")       # name search, two stars
    assert queries[-1][0] == "KOI-7592.01"


def test_occupied_epochs_come_from_the_fitted_segmentation(monkeypatch):
    """prepare_data's occupied epochs (which set the recentred epoch) must be
    exactly the epochs segment_epochs fits, including its minimum-points and
    nearest-centre rules. Epoch at tc=25 keeps one in-transit point and
    nothing else in its window, below MIN_PTS, so segment_epochs drops it;
    the occupied count must drop it too."""
    from turin import prep

    P, T0, dur_h = 10.0, 5.0, 4.8
    cad = 29.4 / 1440
    t = np.arange(0.0, 45.0, cad)
    tc = 25.0
    near = np.where(np.abs(t - tc) <= 5.0 * dur_h / 24)[0]
    keep = np.ones(t.size, bool)
    keep[near] = False
    keep[near[len(near) // 2]] = True        # 1 point, below MIN_PTS
    t = t[keep]
    lc = {"time": t, "flux": np.ones_like(t), "flux_err": np.full(t.size, 1e-4),
          "quality": np.zeros(t.size, int), "mission": "kepler",
          "cadence_days": cad}
    eph = {"period": P, "epoch": T0, "duration": dur_h, "depth": 1000.0}
    monkeypatch.setattr(prep, "get_lightcurve", lambda *a, **k: dict(lc))
    monkeypatch.setattr(prep, "get_ephemeris", lambda *a, **k: dict(eph))
    monkeypatch.setattr(prep, "get_other_planet_ephemerides",
                        lambda *a, **k: [])
    logs = []
    prepared = prep.prepare_data("KOI-1.01", log=logs.append)
    tw, fw, ew = prep.extract_near_transit_data(
        prepared.time, prepared.flux, prepared.flux_err, P, T0, dur_h)
    fitted = prep.segment_epochs(tw, fw, ew, P, T0, dur_h)
    assert not np.any(np.isclose(fitted["epoch_centers"], tc))
    assert prepared.n_occupied == fitted["n_epochs"]


@pytest.mark.parametrize("median,plus,minus,want", [
    # errors to 2 s.f., each one's decimal places, the larger wins
    (628.20791596, 0.01718, 0.1202, ("628.208", "0.017", "0.120")),
    (3.8033, 0.0613, 0.1611, ("3.803", "0.061", "0.161")),
    # 0.00996 rounds to 0.010 (3 places), 0.0021 has 4: four wins
    (0.02464, 0.00996, 0.0021, ("0.0246", "0.0100", "0.0021")),
    (2.5, 0.5, 0.5, ("2.50", "0.50", "0.50")),
    # negative places: 1234 -> 1200 (hundreds), 987 -> 990 (tens)
    (12345, 1234, 987, ("12350", "1230", "990")),
    (12345, 1234, 1234, ("12300", "1200", "1200")),
    (-12345, 1234, 987, ("-12350", "1230", "990")),
])
def test_quote_follows_the_rounding_rule(median, plus, minus, want):
    from turin.plots import quote

    assert quote(median, plus, minus) == want


def test_quote_falls_back_when_errors_are_unusable():
    from turin.plots import quote

    # no usable error at all: four significant figures for the median
    assert quote(1.234567, 0.0, 0.0) == ("1.235", "0", "0")
    # one usable error sets the places; the other is shown as it is
    assert quote(5.0, float("nan"), 0.12) == ("5.00", "NaN", "0.12")


@pytest.mark.parametrize("n_epochs,panels,gap", [
    (8, 13, None), (14, 19, None), (15, 20, 11), (22, 20, 11)])
def test_corner_shows_all_epochs_or_first_and_last_seven(n_epochs, panels, gap):
    from turin import pipeline

    names = ["k", "beta", "T14", "q1", "q2"] + [
        f"dtau_{n}" for n in range(-(n_epochs // 2), n_epochs - n_epochs // 2)]
    cols, gap_after = pipeline._corner_columns(names)
    # the "..." adds one row/column when present
    assert len(cols) + (gap_after is not None) == panels
    assert gap_after == gap
    if gap is not None:
        shown = [names[i] for i in cols[5:]]
        assert shown == names[5:12] + names[-7:]
    # LinEph: all seven, never a gap
    lin = ["dP", "dtau0", "k", "beta", "T14", "q1", "q2"]
    assert pipeline._corner_columns(lin) == (list(range(7)), None)


def test_points_after_an_uncovered_transit_stay_out_of_other_epochs():
    """KOI-5749.01's case: a transit whose centre falls in a data gap, with
    points just after it inside its window. Those points used to join the
    next epoch a whole period away (the transit list was built from centres
    inside the data's span), so a baseline polynomial was evaluated at
    x ~ -490. Every kept point must lie within its own epoch's window. Since
    0.1.55 that transit is fitted as an epoch of its own: its window holds
    enough points, and whether the transit itself fell in the data is for
    the fit to decide (prep.MIN_PTS)."""
    from turin import prep

    P, T0, dur_h = 10.0, 5.0, 2.0
    cad = 29.4 / 1440
    t = np.arange(0.0, 40.0, cad)
    t = t[(t < 4.0) | (t > 5.1)]          # the gap swallows the transit at 5.0
    f, e = np.ones_like(t), np.full(t.size, 1e-4)
    tw, fw, ew = prep.extract_near_transit_data(t, f, e, P, T0, dur_h)
    assert np.any((tw > 5.05) & (tw < 5.5))   # the edge points survive windowing
    ed = prep.segment_epochs(tw, fw, ew, P, T0, dur_h)
    np.testing.assert_allclose(ed["epoch_centers"], [5.0, 15.0, 25.0, 35.0])
    assert ed["n_in_transit"][0] == 0             # the transit fell in the gap
    m = ed["mask"] > 0
    for i, c in enumerate(ed["epoch_centers"]):
        assert np.all(np.abs(ed["times_padded"][i][m[i]] - c)
                      <= ed["half_window"] + 1e-12)


@pytest.mark.parametrize("P,T0,dur_h,span,gap", [
    (10.0, 5.0, 2.0, (0.0, 40.0), (4.0, 5.1)),     # the case above
    (9.3456, 120.5, 4.2, (100.0, 168.0), (131.0, 134.5)),
    (0.9, 100.3, 4.0, (100.0, 108.0), (103.1, 103.6)),  # windows overlap
    (0.9, 100.3, 4.0, (100.31, 108.0), None),      # starts inside a transit
])
def test_segmentation_does_not_depend_on_prewindowing(P, T0, dur_h, span, gap):
    """Since 0.1.50 a point is kept only inside its own epoch's window, so
    segmenting the raw light curve gives exactly what segmenting the
    windowed one does. hurin's segmentation does not have this property (an
    epoch takes in every point nearest it, about a period of data on raw
    input), which is why this lives here rather than in the hurin parity
    tests, where it would also go unrun without a hurin clone. Fails if
    0.1.50's window check is removed."""
    from turin import prep

    t = np.arange(*span, 29.4 / 1440)
    if gap:
        t = t[(t < gap[0]) | (t > gap[1])]
    rng = np.random.default_rng(3)
    f, e = 1 + rng.normal(0, 1e-4, t.size), np.full(t.size, 1e-4)
    raw = prep.segment_epochs(t, f, e, P, T0, dur_h)
    win = prep.segment_epochs(
        *prep.extract_near_transit_data(t, f, e, P, T0, dur_h), P, T0, dur_h)
    assert raw["n_epochs"] == win["n_epochs"] > 1
    for key in ("times_padded", "flux_padded", "ferr_padded", "mask",
                "epoch_centers"):
        np.testing.assert_array_equal(raw[key], win[key], err_msg=key)


def _continuous(t0, t1):
    t = np.arange(t0, t1, 29.4 / 1440)
    return t, np.ones_like(t), np.full(t.size, 1e-4)


@pytest.mark.parametrize("cut,gap,want", [
    (24.8, None, "after"),      # break before the centre: keep the later side
    (25.1, None, "before"),     # break after the centre: keep the earlier side
    (25.0, (24.9, 25.2), "before"),    # centre in the gap: nearer side 0.1 d
    (24.9, (24.75, 25.05), "after"),   # centre in the gap: nearer side 0.05 d
])
def test_a_window_across_a_quarter_break_keeps_one_side(cut, gap, want):
    """A window spanning a quarter or semi-sector boundary keeps only the
    stretch whose range contains the predicted centre, or, with the centre
    in the gap between them, the stretch whose nearest point is closer."""
    from turin import prep

    P, T0, dur_h, tc = 10.0, 5.0, 2.4, 25.0
    t, f, e = _continuous(0.0, 40.0)
    if gap:
        keep = (t < gap[0]) | (t > gap[1])
        t, f, e = t[keep], f[keep], e[keep]
    seg = np.where(t < cut, 7, 8)
    ed = prep.segment_epochs(t, f, e, P, T0, dur_h, segment=seg)
    i = int(np.argmin(np.abs(ed["epoch_centers"] - tc)))
    assert np.isclose(ed["epoch_centers"][i], tc)
    kept = ed["times_padded"][i][ed["mask"][i] > 0]
    assert np.all(kept >= cut) if want == "after" else np.all(kept < cut)
    # epochs whose windows hold one stretch are untouched by the labels
    plain = prep.segment_epochs(t, f, e, P, T0, dur_h)
    for j, c in enumerate(ed["epoch_centers"]):
        if not np.isclose(c, tc):
            np.testing.assert_array_equal(ed["times_padded"][j],
                                          plain["times_padded"][j])


def test_an_epoch_needs_two_points_and_no_transit():
    """MIN_PTS = 2, in transit or not: an epoch whose window holds two
    out-of-transit points is fitted (its transit may have moved, under TTVs);
    one point is not enough to say anything beyond its own baseline."""
    from turin import prep

    P, T0, dur_h = 10.0, 5.0, 2.4
    t, f, e = _continuous(0.0, 40.0)
    hw = prep.N_DURATIONS * dur_h / 24
    for n_keep, kept in ((2, True), (1, False)):
        near = np.where(np.abs(t - 25.0) <= hw)[0]
        drop = near[:-n_keep]                 # the last n_keep: out of transit
        keep = np.ones(t.size, bool)
        keep[drop] = False
        ed = prep.segment_epochs(t[keep], f[keep], e[keep], P, T0, dur_h)
        assert np.any(np.isclose(ed["epoch_centers"], 25.0)) == kept
    assert prep.MIN_PTS == 2


def _fake_target(monkeypatch, lc, eph, others=()):
    from turin import prep
    monkeypatch.setattr(prep, "get_lightcurve", lambda *a, **k: dict(lc))
    monkeypatch.setattr(prep, "get_ephemeris", lambda *a, **k: dict(eph))
    monkeypatch.setattr(prep, "get_other_planet_ephemerides",
                        lambda *a, **k: [dict(o) for o in others])


def test_other_planet_transit_cut_by_the_start_is_masked(monkeypatch):
    """A sibling planet's transit centred just before the light curve begins
    still has its in-transit points masked. The old transit list (centres
    inside the data's span) never listed it, so they stayed in."""
    from turin import prep

    t, f, e = _continuous(0.0, 45.0)
    lc = {"time": t, "flux": f, "flux_err": e, "quality": np.zeros(t.size, int),
          "mission": "kepler", "cadence_days": 29.4 / 1440}
    eph = {"period": 10.0, "epoch": 5.0, "duration": 2.4, "depth": 1000.0}
    other = {"period": 7.0, "epoch": -0.02, "duration": 2.0}
    _fake_target(monkeypatch, lc, eph, [other])
    prepared = prep.prepare_data("KOI-1.01", log=None)
    half = 0.5 * 2.0 / 24
    for c in (-0.02, 6.98, 13.98):       # the cut transit and two inside
        assert not np.any(np.abs(prepared.time - c) <= half)
    assert np.any(prepared.time < half)   # just outside it survives


def test_prepare_data_carries_segment_labels_through_its_filters(monkeypatch):
    """Quality flags and the sigma clip remove points; the quarter labels
    must stay with their points, so the quarter split acts on the epoch
    whose window spans the boundary and on no other."""
    from turin import prep

    t, f, e = _continuous(0.0, 45.0)
    rng = np.random.default_rng(5)
    f = f + rng.normal(0, 1e-4, t.size)
    f[::97] += 0.05                                   # outliers to clip
    quality = np.zeros(t.size, int)
    quality[::53] = 1                                  # flagged cadences
    seg = np.where(t < 24.8, 3, 4)                     # Q3 / Q4 at 24.8
    lc = {"time": t, "flux": f, "flux_err": e, "quality": quality,
          "segment": seg, "mission": "kepler", "cadence_days": 29.4 / 1440}
    eph = {"period": 10.0, "epoch": 5.0, "duration": 2.4, "depth": 1000.0}
    _fake_target(monkeypatch, lc, eph)
    ed = prep.prepare_data("KOI-1.01", log=None).epoch_data
    i = int(np.argmin(np.abs(ed["epoch_centers"] - 25.0)))
    kept = ed["times_padded"][i][ed["mask"][i] > 0]
    assert kept.size > 10 and np.all(kept >= 24.8)
    j = int(np.argmin(np.abs(ed["epoch_centers"] - 15.0)))
    hw = ed["half_window"]
    k15 = ed["times_padded"][j][ed["mask"][j] > 0]
    assert k15.min() < 15.0 - 0.9 * hw and k15.max() > 15.0 + 0.9 * hw


def _lk_curve(time, meta):
    import lightkurve as lk
    lc = lk.LightCurve(time=time, flux=np.ones_like(time),
                       flux_err=np.full(time.size, 1e-4))
    lc.meta.update(meta)
    return lc


def test_segment_labels_kepler_quarters_and_tess_semi_sectors():
    from turin.data import lightcurve as L

    q5 = _lk_curve(np.arange(0.0, 3.0, 0.02), {"QUARTER": 5})
    q6 = _lk_curve(np.arange(3.5, 6.0, 0.02), {"QUARTER": 6})
    np.testing.assert_array_equal(
        L.segment_labels([q5, q6], "kepler"),
        np.r_[np.full(len(q5.time), 5), np.full(len(q6.time), 6)])

    # a TESS sector splits at its largest gap (the orbit gap, 2 d here),
    # not at the shorter mid-orbit downlinks (0.22 d)
    t = np.r_[np.arange(0.0, 5.0, 0.01), np.arange(5.22, 11.0, 0.01),
              np.arange(13.0, 18.0, 0.01), np.arange(18.22, 24.0, 0.01)]
    s61 = _lk_curve(t, {"SECTOR": 61})
    lab = L.segment_labels([s61], "tess")
    np.testing.assert_array_equal(lab, np.where(t < 12.0, 122, 123))
    # a sector with no gap long enough stays whole
    short = _lk_curve(np.arange(0.0, 10.0, 0.01), {"SECTOR": 9})
    assert set(L.segment_labels([short], "tess")) == {18}
    # no header keyword: each file is its own stretch
    a = _lk_curve(np.arange(0.0, 1.0, 0.1), {})
    b = _lk_curve(np.arange(2.0, 3.0, 0.1), {})
    lab = L.segment_labels([a, b], "kepler")
    assert len(set(lab[:len(a.time)])) == 1 and len(set(lab)) == 2


def test_a_cache_without_segment_labels_is_downloaded_again(tmp_path,
                                                            monkeypatch):
    import pickle
    from turin.data import lightcurve as L

    monkeypatch.setattr(L, "CACHE_DIR", str(tmp_path))
    t = np.arange(0.0, 10.0, 0.02)
    old = {"time": t, "flux": np.ones_like(t), "flux_err": np.ones_like(t),
           "quality": np.zeros(t.size, int), "mission": "kepler",
           "cadence_days": 0.02}
    with open(L._cache_path(L.parse_target("KOI-1.01")), "wb") as fh:
        pickle.dump(old, fh)
    fresh = dict(old, segment=np.zeros(t.size, dtype=np.int64))
    calls = []
    monkeypatch.setattr(L, "_catalog_id", lambda *a, **k: "KIC 1")
    monkeypatch.setattr(L, "_download",
                        lambda *a, **k: calls.append(1) or dict(fresh))
    msgs = []
    got = L.get_lightcurve("KOI-1.01", progress=msgs.append)
    assert calls == [1] and "segment" in got
    assert any("quarters and semi-sectors" in m for m in msgs)
    # and the refreshed cache is used as is next time
    assert "segment" in L.get_lightcurve("KOI-1.01") and calls == [1]


def test_binned_errors_come_from_the_formal_errors():
    """The fold plot's bin error is median(formal error)/sqrt(n), not the
    bin's own scatter (which put ~40% noise on 4-point bins and read the
    transit's slope as scatter): equal errors give equal bars, whatever
    the points do."""
    from turin.plots import bin_points
    t = np.arange(10.0)
    y = np.array([0, 5, 0, 5, 1, 1, 1, 1, 9, 9.0])     # last 2 points dropped
    e = np.array([2, 2, 2, 2, 1, 3, 3, 3, 7, 7.0])
    bt, by, be = bin_points(t, y, e, 4)
    np.testing.assert_allclose(bt, [1.5, 5.5])
    np.testing.assert_allclose(by, [2.5, 1.0])
    np.testing.assert_allclose(be, [2 / 2, 3 / 2])
