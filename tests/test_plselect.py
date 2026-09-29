"""The ``--PL`` probe: does it measure the right thing, and decide stably?

Two properties matter and they pull against each other. The probe must be
**discriminating** — it has to notice when a mode is genuinely wrong or
genuinely faster — and it must be **stable**, because a likelihood chosen by
measurement noise would make the same target give different answers on
different days. The tests below check both.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from turin import likelihood, params, plselect, prep, profile, seeding

P_REF, EPOCH, DUR_H = 9.3456, 120.5, 4.2
YERR = 2.5e-4


def make_target(k_true=0.032, order=2, n_periods=30, seed=1):
    """A synthetic target with a real injected transit, plus its MAP."""
    eph = {"period": P_REF, "epoch": EPOCH, "duration": DUR_H,
           "depth": 1e6 * k_true**2}
    rng = np.random.default_rng(seed)
    t = np.arange(100.0, 100.0 + n_periods * P_REF, 29.4 / 1440.0)
    v_true = np.array([0.0, 0.0, k_true, 0.3, DUR_H / 24, 0.35, 0.25])

    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, YERR), P_REF, EPOCH, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_REF, EPOCH, DUR_H)
    cen = prep.centering_constants(ed, eph)
    layout = params.lineph_layout(eph)
    orders = np.full(ed["n_epochs"], order)
    kw = dict(num_resample=5, exposure_time=29.4 / 1440, n_chains_hint=128)

    seed_lp = likelihood.ProfiledTransitLogProb(
        layout, cen, ed, orders, profile_mode="exact", **kw)
    f = np.array(seed_lp.full_model(
        mx.array(v_true[None, :].astype(np.float32)))[0], dtype=np.float64)[0]
    ed = dict(ed)
    ed["flux_padded"] = np.where(ed["mask"] > 0,
                                 f + rng.normal(0.0, YERR, f.shape), 1.0)

    target, transform, lp, _ = likelihood.build_target(
        layout, cen, ed, orders, **kw)
    u0 = transform.from_model_np(v_true)
    m = seeding.find_map(
        target,
        (u0[None, :] + 0.1 * np.random.default_rng(0).standard_normal(
            (64, layout.dim))).astype(np.float32), n_iter=200)
    return dict(layout=layout, centering=cen, epoch_data=ed, orders=orders,
                build_kwargs=kw, transform=transform, u_map=m.u_best)


def probe(fixture, **over):
    kw = dict(build_kwargs=fixture["build_kwargs"], n_chains=128, log=None)
    kw.update(over)
    return plselect.select_pl_mode(
        fixture["layout"], fixture["centering"], fixture["epoch_data"],
        fixture["orders"], fixture["u_map"], fixture["transform"], **kw)


@pytest.fixture(scope="module")
def shallow():
    return make_target(k_true=0.032)


@pytest.fixture(scope="module")
def deep():
    return make_target(k_true=0.10)


# -- what it measures --------------------------------------------------

def test_probe_reports_every_mode_and_a_calibrated_width(shallow):
    c = probe(shallow)
    assert {r.mode for r in c.rows} == set(profile.PROFILE_MODES)
    assert c.probed and c.mode in profile.PROFILE_MODES
    # the width is calibrated from the density, not hard-coded
    assert 1e-4 < c.width < 1.0
    assert c.widths == (c.width, 2 * c.width)
    assert c.calib_drop > 0, "u_map should be a mode, so the ball drops"
    text = c.describe()
    for mode in profile.PROFILE_MODES:
        assert mode in text
    assert "chose" in text


def test_exact_is_always_the_reference_and_always_qualifies(shallow, deep):
    for fx in (shallow, deep):
        c = probe(fx)
        row = c.row("exact")
        assert row.qualifies and row.note == "reference"
        assert c.threshold >= plselect.SD_FLOOR


def test_at_posterior_width_all_modes_are_precision_equivalent(shallow, deep):
    """The finding the whole design rests on.

    Over a ball scaled to the posterior, ``ratio``'s O(depth) error is very
    nearly a constant offset, and constants cancel out of a posterior. Judged
    on a ball far wider than the posterior it would look catastrophic.
    """
    for fx in (shallow, deep):
        c = probe(fx)
        sds = {r.mode: r.sd_worst for r in c.rows}
        assert max(sds.values()) < 10 * min(sds.values()), sds
        assert all(r.qualifies for r in c.rows), sds


def test_the_width_calibration_is_load_bearing(deep):
    """Judged over too wide a ball, ratio is rejected; at the posterior, it is not.

    This is the single most important property of the design. ``ratio``'s
    deviation from ``exact`` grows steeply with the region it is judged over,
    so a fixed ball width does not merely add noise to the decision -- it
    changes the answer. Calibrating to the posterior is what makes the verdict
    mean "this would distort the posterior" rather than "this differs
    somewhere in the prior".
    """
    at_posterior = probe(deep)
    assert at_posterior.row("ratio").qualifies, at_posterior.describe()

    # the same target judged 50x wider than its own posterior
    too_wide = probe(deep, width_multiples=(1.0, 50.0))
    wide_ratio = too_wide.row("ratio")
    assert wide_ratio.sd_worst > 10 * at_posterior.row("ratio").sd_worst
    assert not wide_ratio.qualifies, too_wide.describe()
    assert too_wide.mode == "exact"

    # exact, by contrast, is float32 noise either way and stays in the same
    # ballpark however wide the ball
    assert (too_wide.row("exact").sd_worst
            < 100 * at_posterior.row("exact").sd_worst)


# -- how it decides ----------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_choice_is_stable_across_probe_seeds(shallow, seed):
    """A likelihood must not be chosen by measurement noise."""
    assert probe(shallow, seed=seed).mode == probe(shallow, seed=0).mode


def test_a_mode_must_be_decisively_faster_to_displace_exact(shallow):
    """At Legendre sizes the modes differ by less than the timing spread."""
    c = probe(shallow)
    exact = c.row("exact")
    for r in c.rows:
        if r.mode == c.mode and r.mode != "exact":
            # whatever won had to beat exact's best with its own worst
            assert r.ms_worst < exact.ms * (1 - plselect.SPEED_MARGIN)
    # and the timing range is reported, so the noise is visible
    assert all(r.ms_worst >= r.ms > 0 for r in c.rows)


def test_precision_failure_disqualifies_regardless_of_speed(shallow):
    """A mode that fails the precision bar cannot be chosen, however fast."""
    c = probe(shallow, sd_factor=0.0, sd_floor=0.0)
    assert c.mode == "exact"
    assert all(not r.qualifies for r in c.rows if r.mode != "exact")
    assert "precision" in c.reason


def test_a_large_basis_is_where_the_probe_earns_its_keep():
    """With 30 basis columns exact collapses, and the probe must notice.

    This is the case the hard-coded rule in profile.py cannot cover, because
    the crossover depends on the machine.
    """
    fx = make_target(k_true=0.032, order=29, n_periods=12)
    c = probe(fx)
    exact, chosen = c.row("exact"), c.row(c.mode)
    assert c.mode != "exact", c.describe()
    assert chosen.ms_worst < exact.ms, c.describe()
    assert chosen.qualifies


def test_probe_cost_is_small(shallow):
    """It runs before every fit, so it has to stay cheap."""
    t0 = time.perf_counter()
    c = probe(shallow)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, elapsed          # measured ~2.5s; generous bound
    assert c.elapsed <= elapsed + 0.5


# -- the explicit path -------------------------------------------------

def test_fixed_choice_does_not_probe():
    c = plselect.fixed_choice("ratio")
    assert c.mode == "ratio" and not c.probed
    assert "--PL=ratio" in c.describe()
    assert c.rows == []


def test_calibration_falls_back_when_the_point_is_not_a_mode(shallow):
    """A flat or ascending direction must not produce a degenerate ball."""
    with mx.stream(mx.cpu):
        ref = likelihood.ProfiledTransitLogProb(
            shallow["layout"], shallow["centering"], shallow["epoch_data"],
            shallow["orders"], profile_mode="exact", dtype=mx.float64,
            **shallow["build_kwargs"])
    # a point far from the mode, where the ball mostly improves the density
    u_bad = np.asarray(shallow["u_map"], dtype=np.float64) + 2.0
    width, drop = plselect._calibrate_width(
        ref, u_bad, shallow["transform"], shallow["layout"].dim)
    assert np.isfinite(width) and 1e-4 <= width <= 1.0
