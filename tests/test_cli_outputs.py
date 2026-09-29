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
    assert a.profile_mode == "exact"
    assert a.geometry == "circular"
    assert a.ld_map == "kipping"
    assert a.sampler == "chees"
    assert a.modes == ("lineph", "ttv")
    assert a.ttv_max_days is None
    assert a.chains == 512 and a.max_leapfrog == 128


def test_nongrazing_is_an_alias():
    assert cli.parse_args(["--KOI-1.01", "--nongrazing"]).b_prior == "nongrazing"


def test_ttvmax_is_minutes_on_the_command_line():
    a = cli.parse_args(["--KOI-1.01", "--TTVmax=720"])
    assert a.ttv_max_min == 720.0
    np.testing.assert_allclose(a.ttv_max_days, 0.5)


def test_hurin_compat_combination_parses():
    a = cli.parse_args(["--KOI-1.01", "--geometry=chord", "--ld=hurin",
                        "--profile=ratio"])
    assert (a.geometry, a.ld_map, a.profile_mode) == ("chord", "hurin", "ratio")


def test_modes_subset():
    assert cli.parse_args(["--KOI-1.01", "--modes=lineph"]).modes == ("lineph",)
    assert cli.parse_args(["--KOI-1.01", "--modes=ttv"]).modes == ("ttv",)


@pytest.mark.parametrize("argv,pattern", [
    (["--KOI-1.01", "--chanis=8"], "unrecognized"),
    (["--KOI-1.01", "--bprior=nonsense"], "not one of"),
    (["--KOI-1.01", "--sampler=nuts"], "not one of"),
    (["--KOI-1.01", "--profile=approx"], "not one of"),
    (["--KOI-1.01", "--geometry=kepler"], "not one of"),
    (["--KOI-1.01", "--ld=quadratic"], "not one of"),
    (["--KOI-1.01", "--modes=lineph,nope"], "not one of"),
    (["--KOI-1.01", "--chains=0"], "positive"),
    (["--KOI-1.01", "--chains=abc"], "positive"),
    (["--KOI-1.01", "--TTVmax=-5"], "positive"),
    (["--KOI-1.01", "--tag=bad tag"], "tag"),
    (["--KOI-1.01", "--KOI-2.01"], "more than one target"),
    (["KOI-1.01"], "unrecognized"),
])
def test_bad_arguments_are_rejected(argv, pattern):
    with pytest.raises(SystemExit, match=pattern):
        cli.parse_args(argv)


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
        profile_mode="exact", geometry="circular", ld_map="kipping",
        sampler="chees", n_chains=512, ttv_max=None, n_durations=5.0,
        legendre_orders=np.zeros(3), exposure_time=0.02, num_resample=7,
        n_samples_done=300, done=False)

    state.check(b_prior="transiting", profile_mode="exact")   # agrees
    for bad in (dict(b_prior="box"), dict(profile_mode="ratio"),
                dict(geometry="chord"), dict(ld_map="hurin"),
                dict(sampler="ensemble")):
        with pytest.raises(SystemExit, match="use --|Use --"):
            state.check(**bad)


def test_resume_state_round_trips(tmp_path):
    state = outputs.ResumeState(
        target="KOI-1.01", mode="ttv", turin_version="0.1.0",
        launch_command="turin --KOI-1.01", tag=None, b_prior="nongrazing",
        profile_mode="ratio", geometry="chord", ld_map="hurin",
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


def test_fit_linear_ephemeris_is_robust_to_an_outlier():
    epochs = np.arange(12.0)
    P, tau0 = 9.3456, 120.5
    times = tau0 + P * epochs
    times[7] += 0.5                       # one badly wrong measurement
    P_fit, tau0_fit, keep = outputs.fit_linear_ephemeris(epochs, times)
    np.testing.assert_allclose(P_fit, P, atol=1e-6)
    np.testing.assert_allclose(tau0_fit, tau0, atol=1e-5)
    assert not keep[7]


def test_summarize_percentiles_and_diagnostics():
    rng = np.random.default_rng(1)
    draws = rng.normal(3.0, 0.5, (20000, 1))
    s = outputs.summarize(draws, ["x"], rhat=[1.004], ess_bulk=[1234],
                          ess_tail=[999])["x"]
    assert abs(s["median"] - 3.0) < 0.02
    assert abs(s["std"] - 0.5) < 0.02
    assert s["lo"] < s["median"] < s["hi"]
    assert s["rhat"] == 1.004 and s["ess_bulk"] == 1234 and s["ess_tail"] == 999
