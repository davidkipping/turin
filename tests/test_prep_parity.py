"""The NumPy pipeline ports must agree with hurin exactly.

hurin's ``transit_fit.py`` imports only numpy at module level (JAX is
imported lazily inside the functions that need it), so the original
implementations can be imported directly and compared against, rather than
against recorded fixtures. This is the strongest available check that the
port did not drift.

Skipped if the hurin clone is not beside this one.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from turin import prep

_HURIN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hurin"))


@pytest.fixture(scope="module")
def hurin_tf():
    if not os.path.isdir(_HURIN):
        pytest.skip("hurin clone not found beside turin")
    if _HURIN not in sys.path:
        sys.path.insert(0, _HURIN)
    try:
        import hurin.transit_fit as tf
    except Exception as exc:  # pragma: no cover - environment dependent
        # the clone is there, so this is breakage, not an absent comparison
        pytest.fail(f"hurin clone found but hurin.transit_fit failed to "
                    f"import: {exc}")
    return tf


@pytest.fixture(scope="module")
def lc():
    """A synthetic multi-epoch light curve with gaps and a trend."""
    rng = np.random.default_rng(20260928)
    period, epoch, dur_h = 9.3456, 120.5, 4.2
    # 30-minute cadence over ~7 periods, with a chunk missing
    t = np.arange(100.0, 168.0, 29.4 / 1440.0)
    t = t[(t < 131.0) | (t > 134.5)]
    phase = (t - epoch + 0.5 * period) % period - 0.5 * period
    in_tr = np.abs(phase) < 0.5 * dur_h / 24.0
    flux = 1.0 + 3e-4 * np.sin(2 * np.pi * t / 3.7) + rng.normal(0, 2e-4, t.size)
    flux[in_tr] -= 1.2e-3
    ferr = np.full(t.size, 2e-4)
    return dict(time=t, flux=flux, flux_err=ferr, period=period,
                epoch=epoch, duration_hours=dur_h)


@pytest.fixture(scope="module")
def lc_short():
    """A short-period light curve whose transit windows overlap
    (``half_window > P/2``), so nearest-transit assignment decides every
    point. It starts and ends on a transit centre, so every point's nearest
    transit is itself fitted: at the light curve's edges turin and hurin
    segment differently on purpose (docs/hurin-differences.md section 4)."""
    rng = np.random.default_rng(20261008)
    period, epoch, dur_h = 0.7, 10.0, 3.0
    t = np.arange(10.0, 15.0, 29.4 / 1440.0)
    phase = (t - epoch + 0.5 * period) % period - 0.5 * period
    flux = 1.0 + rng.normal(0, 2e-4, t.size)
    flux[np.abs(phase) < 0.5 * dur_h / 24.0] -= 1e-3
    return dict(time=t, flux=flux, flux_err=np.full(t.size, 2e-4),
                period=period, epoch=epoch, duration_hours=dur_h)


def test_extract_near_transit_data(hurin_tf, lc):
    args = (lc["time"], lc["flux"], lc["flux_err"], lc["period"],
            lc["epoch"], lc["duration_hours"])
    mine = prep.extract_near_transit_data(*args, n_durations=5.0)
    theirs = hurin_tf.extract_near_transit_data(*args, n_durations=5.0)
    for a, b in zip(mine, theirs):
        np.testing.assert_array_equal(a, b)
    assert mine[0].size < lc["time"].size, "windowing should discard points"


def _windowed(lc):
    """The light curve as the pipeline hands it to ``segment_epochs``:
    already cut to the transit windows, at the same width."""
    t, f, e = prep.extract_near_transit_data(
        lc["time"], lc["flux"], lc["flux_err"], lc["period"], lc["epoch"],
        lc["duration_hours"], n_durations=prep.N_DURATIONS)
    return (t, f, e, lc["period"], lc["epoch"], lc["duration_hours"])


@pytest.mark.parametrize("curve", ["lc", "lc_short"])
@pytest.mark.parametrize("tau_shift_max", [0.0, 0.05])
def test_segment_epochs(hurin_tf, request, curve, tau_shift_max):
    # Windowed input, as in the pipeline. The two segmentations agree when
    # every point's nearest transit has its centre inside the data's span,
    # the window holds one quarter, and every epoch has 4 or more points with
    # one in transit; turin's rule differs from hurin's only outside that
    # (docs/hurin-differences.md section 4), and neither curve leaves it.
    lc = request.getfixturevalue(curve)
    args = _windowed(lc)
    mine = prep.segment_epochs(*args, tau_shift_max=tau_shift_max)
    theirs = hurin_tf.segment_epochs(*args, tau_shift_max=tau_shift_max)
    if curve == "lc_short":
        assert mine["half_window"] > lc["period"] / 2   # windows overlap

    assert mine["n_epochs"] == theirs["n_epochs"] > 1
    assert mine["max_pts"] == theirs["max_pts"]
    for key in ("times_padded", "flux_padded", "ferr_padded", "mask",
                "epoch_centers"):
        np.testing.assert_array_equal(mine[key], theirs[key], err_msg=key)
    assert mine["half_window"] == theirs["half_window"]

    # padded slots carry the neutral values the likelihood relies on
    pad = mine["mask"] == 0
    if pad.any():
        assert np.all(mine["flux_padded"][pad] == 1.0)
        assert np.all(mine["ferr_padded"][pad] == 1e10)


@pytest.mark.parametrize("order", range(6))
def test_legendre_matrix(hurin_tf, order):
    x = np.linspace(-1.0, 1.0, 37)
    np.testing.assert_allclose(
        prep.legendre_matrix(x, order),
        hurin_tf._numpy_legendre_matrix(x, order),
        rtol=0, atol=0)
    # spot-check the recurrence against known Legendre polynomials
    L = prep.legendre_matrix(x, order)
    if order >= 2:
        np.testing.assert_allclose(L[:, 2], 0.5 * (3 * x**2 - 1), atol=1e-12)
    if order >= 3:
        np.testing.assert_allclose(L[:, 3], 0.5 * (5 * x**3 - 3 * x), atol=1e-12)


def test_centering_constants(hurin_tf, lc):
    # window first, as the pipeline does: segment_epochs on unwindowed data
    # would assign points up to half a period from their epoch centre
    tw, fw, ew = prep.extract_near_transit_data(
        lc["time"], lc["flux"], lc["flux_err"], lc["period"], lc["epoch"],
        lc["duration_hours"])
    ed = prep.segment_epochs(tw, fw, ew, lc["period"], lc["epoch"],
                             lc["duration_hours"])
    eph = {"period": lc["period"], "epoch": lc["epoch"]}
    mine = prep.centering_constants(ed, eph)
    theirs = hurin_tf._centering_constants(ed, eph)
    for key in ("times_centered", "centers_abs", "n_arr", "d_arr"):
        np.testing.assert_array_equal(mine[key], theirs[key], err_msg=key)
    assert mine["P_ref"] == theirs["P_ref"]
    assert mine["tau0_ref"] == theirs["tau0_ref"]

    # the point of the exercise: centred times and d_arr are O(1) or smaller
    assert np.abs(mine["times_centered"]).max() < 5.1 * lc["duration_hours"] / 24
    assert np.abs(mine["d_arr"]).max() < 1e-6
    # epoch numbers are exact integers
    np.testing.assert_array_equal(mine["n_arr"], np.round(mine["n_arr"]))


def test_optimize_legendre_orders(hurin_tf, lc):
    args = (lc["time"], lc["flux"], lc["flux_err"], lc["period"],
            lc["epoch"], lc["duration_hours"])
    mine = prep.optimize_legendre_orders(*args)
    theirs = hurin_tf.optimize_legendre_orders(*args)
    np.testing.assert_array_equal(mine["orders"], theirs["orders"])
    np.testing.assert_allclose(mine["scores"], theirs["scores"], rtol=1e-12)
    np.testing.assert_allclose(mine["total_scores"], theirs["total_scores"],
                               rtol=1e-12)
    np.testing.assert_array_equal(mine["epoch_counts"], theirs["epoch_counts"])
    assert mine["orders"].min() >= 0 and mine["orders"].max() <= prep.MAX_ORDER


def test_supersample_offsets():
    assert prep.supersample_offsets(0.02, 1).tolist() == [0.0]
    assert prep.supersample_offsets(0.0, 7).tolist() == [0.0]
    off = prep.supersample_offsets(0.02, 5)
    # batman's endpoint-inclusive convention, symmetric about the centre
    assert off.size == 5
    np.testing.assert_allclose(off[0], -0.01)
    np.testing.assert_allclose(off[-1], 0.01)
    np.testing.assert_allclose(off.mean(), 0.0, atol=1e-18)


def test_recommended_resample_matches_hurin_formula():
    eph = {"depth": 1200.0, "duration": 4.2, "ingress": 0.3}
    ferr = np.full(500, 2e-4)
    cadence = 29.4 / 1440.0

    n = prep.recommended_resample(eph, cadence, ferr)
    expected = np.sqrt((1200.0 / 1e6) * cadence * 24.0
                       / (0.8 * 0.3 * 2e-4))
    assert n == max(1, int(np.ceil(expected)))

    # b=0 fallback when the archive has no ingress duration
    n2 = prep.recommended_resample({**eph, "ingress": None}, cadence, ferr)
    assert n2 >= 1
    # degenerate inputs decline rather than raise
    assert prep.recommended_resample({"depth": 0.0, "duration": 4.2},
                                     cadence, ferr) is None
