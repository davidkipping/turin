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


def test_the_pipeline_refuses_a_gaia_target_until_it_can_fit_one(tmp_path):
    from turin import pipeline
    a = cli.parse_args(["--GaiaDR3-1", *_EPH, f"--outdir={tmp_path}"])
    with pytest.raises(SystemExit, match="not yet fitted"):
        pipeline.run(a, log=lambda m: None)
