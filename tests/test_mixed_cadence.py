"""A light curve that mixes cadences: every epoch is integrated over the
exposure it was observed with.

A Kepler ``--sc`` fit takes short cadence where a quarter has it and long
where not; a TESS star can have 2-minute sectors beside FFI ones. Until 0.1.86
the median step of the whole series was every point's exposure.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from turin import model as M
from turin import outputs, prep

P_REF, T14, K, B = 9.3456, 0.175, 0.08, 0.35
Q = (0.30, 0.225)
SHORT, LONG = 2.0 / 1440, 29.4 / 1440


@pytest.fixture(autouse=True)
def _cpu(request):
    """float64 runs on the CPU stream; both sides of every comparison here go
    through the same ops, which is what these tests need. The end-to-end run
    keeps its default (GPU) stream."""
    if "pipeline" in request.node.name:
        yield
        return
    with mx.stream(mx.cpu):
        yield


def _centering(n_epochs=5, n_pts=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(-0.3, 0.3, (n_epochs, n_pts)), axis=1)
    return dict(times_centered=t, n_arr=np.arange(n_epochs, dtype=float),
                d_arr=np.zeros(n_epochs), P_ref=P_REF, tau0_ref=0.0)


def _sub(cen, rows):
    return {k: (np.asarray(v)[rows] if np.ndim(v) else v)
            for k, v in cen.items()}


def _grid(cen, exp_epoch=None, exp_time=0.0, dtype=mx.float64):
    return M.build_grid(cen, prep.supersample_offsets(exp_time, 1),
                        dtype=dtype, exp_time=exp_time, exp_epoch=exp_epoch)


def _col(v, dtype=mx.float64, n=1):
    return mx.full((n, 1), float(v), dtype=dtype)


def _flux(grid, **kw):
    n = grid.n_epochs
    return np.array(M.transit_flux_dev(
        grid, mid=mx.zeros((1, n), dtype=grid.dtype), k=_col(K), b=_col(B),
        T14=_col(T14), q=tuple(_col(x) for x in Q), period=_col(P_REF),
        **kw), dtype=np.float64)[0]


EXPS = np.array([SHORT, LONG, SHORT, LONG, LONG])


def test_each_epoch_gets_its_own_exposure():
    """The mixed grid's epoch i is what a one-cadence grid gives it at that
    epoch's own exposure -- and differs from the median-step answer."""
    cen = _centering()
    mixed = _grid(cen, EXPS, exp_time=LONG)
    assert len(mixed.groups) == 2
    got = _flux(mixed)
    for i, e in enumerate(EXPS):
        want = _flux(_grid(_sub(cen, [i]), exp_time=e))[0]
        np.testing.assert_allclose(got[i], want, rtol=0, atol=1e-13)
    # the old behaviour, one exposure for all: wrong for the short epochs
    one = _flux(_grid(cen, exp_time=LONG))
    assert np.abs(one[0] - got[0]).max() > 1e-5
    np.testing.assert_allclose(one[1], got[1], rtol=0, atol=1e-13)


def test_blocks_keep_their_own_exposure():
    """The likelihood's epoch blocks (``select``) carry the right exposure,
    including a block whose epochs all share one that is not the grid's."""
    cen = _centering()
    mixed = _grid(cen, EXPS, exp_time=LONG)
    whole = _flux(mixed)
    for lo, hi in [(0, 1), (1, 2), (0, 3), (2, 5), (3, 5)]:
        blk = mixed.select(lo, hi)
        np.testing.assert_allclose(_flux(blk), whole[lo:hi], rtol=0, atol=1e-13)
    assert mixed.select(0, 1).exp_time == SHORT and not mixed.select(0, 1).groups
    assert mixed.select(3, 5).exp_time == LONG


def test_one_cadence_is_the_grid_it_always_was():
    cen = _centering()
    plain = _grid(cen, exp_time=LONG)
    same = _grid(cen, np.full(5, LONG), exp_time=LONG)
    assert not same.groups and same.exp_epoch is None
    np.testing.assert_array_equal(_flux(plain), _flux(same))
    np.testing.assert_array_equal(np.array(plain.order), np.array(same.order))


def test_a_limb_darkening_basis_goes_through_the_groups():
    """``ld_basis`` outputs carry a trailing axis; the per-exposure calls must
    join it correctly (vertex laws and hybrid4's generators)."""
    cen = _centering()
    mixed = _grid(cen, EXPS, exp_time=LONG)
    kw = dict(mid=mx.zeros((1, 5), dtype=mx.float64), k=_col(K), b=_col(B),
              T14=_col(T14), period=_col(P_REF))
    got = [np.array(x)[0] for x in M.vertex_flux_devs(mixed, **kw)]
    gen = [np.array(x)[0] for x in M.generator_flux_devs(mixed, **kw)]
    for i, e in enumerate(EXPS):
        one = _grid(_sub(cen, [i]), exp_time=e)
        k1 = dict(kw, mid=mx.zeros((1, 1), dtype=mx.float64))
        for a, w in zip(got, M.vertex_flux_devs(one, **k1)):
            np.testing.assert_allclose(a[i], np.array(w)[0, 0], atol=1e-13)
        for a, w in zip(gen, M.generator_flux_devs(one, **k1)):
            np.testing.assert_allclose(a[i], np.array(w)[0, 0], atol=1e-13)


def test_hybrid4_and_the_float32_path_use_each_epochs_exposure():
    cen = _centering()
    for dtype, tol in ((mx.float64, 1e-13), (mx.float32, 2e-6)):
        mixed = _grid(cen, EXPS, exp_time=LONG, dtype=dtype)
        q = tuple(_col(x, dtype) for x in (0.5, 0.4, 0.5, 0.5))
        kw = dict(k=_col(K, dtype), b=_col(B, dtype), T14=_col(T14, dtype),
                  period=_col(P_REF, dtype), law="hybrid4", q=q)
        got = np.array(M.transit_flux_dev(
            mixed, mid=mx.zeros((1, 5), dtype=dtype), **kw), dtype=np.float64)[0]
        for i, e in enumerate(EXPS):
            one = _grid(_sub(cen, [i]), exp_time=e, dtype=dtype)
            want = np.array(M.transit_flux_dev(
                one, mid=mx.zeros((1, 1), dtype=dtype), **kw),
                dtype=np.float64)[0, 0]
            np.testing.assert_allclose(got[i], want, rtol=0, atol=tol)


def test_the_supersampling_fallback_refuses_mixed_cadences():
    mixed = _grid(_centering(), EXPS, exp_time=LONG)
    with pytest.raises(ValueError, match="mixes cadences"):
        M.transit_flux_dev(
            mixed, mid=mx.zeros((1, 5), dtype=mx.float64), k=_col(K), b=_col(B),
            T14=_col(T14), q=tuple(_col(x) for x in Q), period=_col(P_REF),
            geometry="chord")


# -- the data side ---------------------------------------------------------

def test_cadence_is_the_segments_own_median_step():
    t = np.concatenate([np.arange(0, 10, 0.02), np.arange(10, 20, 0.5)])
    seg = np.where(t < 10, 1, 2)
    c = prep.segment_cadences(t, seg)
    assert np.allclose(c[seg == 1], 0.02) and np.allclose(c[seg == 2], 0.5)


def test_snap_merges_equal_cadences_and_keeps_distinct_ones():
    c = np.array([0.0204, 0.0204 * (1 + 1e-7), 0.5, 0.5 * (1 - 1e-6), 0.0204])
    s = prep.snap_cadences(c)
    assert len(np.unique(s)) == 2
    assert s[0] == s[1] == s[4] and s[2] == s[3]
    # TESS 120 s and 200 s are not the same cadence
    assert len(np.unique(prep.snap_cadences(np.array([120., 200.]) / 86400))) == 2


def _mixed_lc(second_cadence, span2=(24.8, 45.0)):
    """Quarter 3 at 29.4 min, quarter 4 from 24.8 d at ``second_cadence``."""
    t1 = np.arange(0.0, span2[0], 29.4 / 1440)
    t2 = np.arange(span2[0], span2[1], second_cadence)
    t = np.concatenate([t1, t2])
    seg = np.where(t < span2[0], 3, 4)
    return {"time": t, "flux": np.ones_like(t),
            "flux_err": np.full(t.size, 1e-4),
            "quality": np.zeros(t.size, int), "segment": seg,
            "mission": "kepler", "cadence_days": float(np.median(np.diff(t)))}


def _fake(monkeypatch, lc, eph):
    monkeypatch.setattr(prep, "get_lightcurve", lambda *a, **k: dict(lc))
    monkeypatch.setattr(prep, "get_ephemeris", lambda *a, **k: dict(eph))
    monkeypatch.setattr(prep, "get_other_planet_ephemerides",
                        lambda *a, **k: [])


EPH = {"period": 10.0, "epoch": 5.0, "duration": 2.4, "depth": 1000.0}


def test_prepare_data_assigns_each_epoch_its_cadence(monkeypatch):
    """Transits at 5, 15 (quarter 3) and 25, 35, 45 (quarter 4); the second
    quarter is one-minute data. Epochs 25 and later carry 1 minute, the
    others 29.4, and the representative exposure is the one with most points."""
    short = 1.0 / 1440
    lc = _mixed_lc(short)
    _fake(monkeypatch, lc, EPH)
    logs = []
    p = prep.prepare_data("KOI-1.01", log=logs.append)
    ed = p.epoch_data
    exp = ed["exposure"]
    for c, e in zip(ed["epoch_centers"], exp):
        want = 29.4 / 1440 if c < 24.8 else short
        assert e == pytest.approx(want, rel=1e-3), (c, e)
    assert np.unique(exp).size == 2
    n = ed["mask"].sum(axis=1)
    assert p.exposure_time == pytest.approx(
        np.unique(exp)[np.argmax([n[exp == u].sum() for u in np.unique(exp)])])
    assert any("Mixed cadences" in m for m in logs)


def test_a_cadence_with_no_fitted_epoch_does_not_set_the_exposure(monkeypatch):
    """Mixed data, but every fitted epoch is in the long-cadence quarter:
    the exposure is that cadence, not the series' median step, and the grid
    is the one-cadence one."""
    short = 1.0 / 1440
    lc = _mixed_lc(short, span2=(27.0, 28.0))   # 1-min data, no transit
    eph = dict(EPH, epoch=5.0, period=10.0)
    _fake(monkeypatch, lc, eph)
    p = prep.prepare_data("KOI-1.01", log=None)
    assert "exposure" not in p.epoch_data
    assert p.exposure_time == pytest.approx(29.4 / 1440, rel=1e-3)


def test_a_one_cadence_light_curve_is_untouched(monkeypatch):
    t = np.arange(0.0, 45.0, 29.4 / 1440)
    lc = {"time": t, "flux": np.ones_like(t), "flux_err": np.full(t.size, 1e-4),
          "quality": np.zeros(t.size, int), "segment": np.where(t < 24.8, 3, 4),
          "mission": "kepler", "cadence_days": 29.4 / 1440}
    _fake(monkeypatch, lc, EPH)
    p = prep.prepare_data("KOI-1.01", log=None)
    assert "exposure" not in p.epoch_data
    assert p.exposure_time == lc["cadence_days"]


# -- resume ----------------------------------------------------------------

def _state(exposure_time, exposures=()):
    s = object.__new__(outputs.ResumeState)
    s.target, s.mode = "TOI-1.01", "lineph"
    s.exposure_time, s.exposures = exposure_time, exposures
    return s


def test_resume_refuses_a_different_exposure_model():
    _state(LONG).check_exposure(LONG)                       # unchanged
    _state(LONG, (SHORT, LONG)).check_exposure(LONG, (SHORT, LONG))
    for stored, now in [(_state(LONG), (LONG, (SHORT, LONG))),
                        (_state(LONG, (SHORT, LONG)), (LONG, ())),
                        (_state(LONG), (SHORT, ()))]:
        with pytest.raises(SystemExit, match="--fresh"):
            stored.check_exposure(*now)


def test_a_state_from_before_the_field_reads_as_one_cadence():
    s = _state(LONG)
    del s.exposures                      # as unpickled from an older turin
    s.check_exposure(LONG)
    with pytest.raises(SystemExit):
        s.check_exposure(LONG, (SHORT, LONG))


# -- end to end ------------------------------------------------------------

def test_pipeline_fits_a_mixed_cadence_light_curve_and_guards_the_resume(
        monkeypatch, tmp_path):
    """The synthetic target with its second half re-sampled at 5x the step.
    The run integrates the two halves over their own exposures, records them
    in the resume state, and refuses to continue them as one cadence."""
    import test_pipeline as TP
    from turin import cli, pipeline

    TP._serve(monkeypatch)
    serve = prep.get_lightcurve

    def mixed(*a, **k):
        lc = serve(*a, **k)
        t = lc["time"]
        late = t >= np.median(t)
        keep = ~late | (np.arange(t.size) % 5 == 0)
        out = {key: (v[keep] if hasattr(v, "shape") and v.shape == t.shape
                     else v) for key, v in lc.items()}
        out["segment"] = np.where(out["time"] < np.median(t), 1, 2)
        return out

    monkeypatch.setattr(prep, "get_lightcurve", mixed)
    logs = []
    args = TP._args(tmp_path)
    assert pipeline.run(args, log=logs.append) in TP.FINISHED
    assert any("one kernel call each" in m for m in logs)
    state = outputs.load_resume(str(tmp_path), "KOI-1.01", "lineph")
    assert len(state.exposures) == 2
    assert state.exposures[1] == pytest.approx(5 * state.exposures[0], rel=1e-3)

    # the same target read as one cadence (an older turin's) cannot resume
    state.exposures = ()
    outputs.save_resume(str(tmp_path), "KOI-1.01", "lineph", state)
    args2 = TP._args(tmp_path, extend1=True)
    with pytest.raises(SystemExit, match="--fresh"):
        pipeline.run(args2, log=logs.append)
