"""Initialization: the template sweep must find displaced transits.

This is the test that matters for turin's reason to exist. hurin's headline
failure was chains settling into a secondary timing mode and reporting a
confident wrong answer (KOI-5162.01: two chains, R-hat <= 1.01, sitting
35.6 log-units below the global optimum). The fix is to seed each epoch at
its own posterior peak, so the test injects transits that are genuinely
displaced and checks the sweep recovers them.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from turin import likelihood, model as M, params, prep, profile, seeding

P_REF, EPOCH, DUR_H = 9.3456, 120.5, 4.2
K_T, B_T, Q1_T, Q2_T = 0.11, 0.30, 0.30, 0.225
T14_T = DUR_H / 24.0
YERR = 1.5e-4
EPH = {"period": P_REF, "epoch": EPOCH, "duration": DUR_H, "depth": 12000.0}


def build_dataset(true_dtau, seed=5, trend=True, yerr=YERR, decoy=None):
    """A light curve whose epoch i transit sits at its predicted time + dtau_i.

    ``decoy``, if given, injects a second identical transit-shaped dip that
    many days away in every epoch -- hurin's real failure mode, where a
    variability dip competes with the true transit for the timing seed.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(100.0, 100.0 + 9.0 * P_REF, 29.4 / 1440.0)
    tw, fw, ew = prep.extract_near_transit_data(
        t, np.ones_like(t), np.full(t.size, yerr), P_REF, EPOCH, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_REF, EPOCH, DUR_H)
    cen = prep.centering_constants(ed, EPH)
    n_ep = ed["n_epochs"]
    dtau = np.broadcast_to(np.asarray(true_dtau, dtype=np.float64),
                           (n_ep,)).copy()

    grid = M.build_grid(cen, prep.supersample_offsets(29.4 / 1440, 7),
                        dtype=mx.float64, exp_time=29.4 / 1440)
    col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        mid = M.mid_times_ttv(grid, mx.array(dtau[None, :], dtype=mx.float64))
        f = np.array(M.transit_flux(
            grid, mid=mid, k=col(K_T), b=col(B_T), T14=col(T14_T),
            q1=col(Q1_T), q2=col(Q2_T), period=col(P_REF)),
            dtype=np.float64)[0]
        if decoy is not None:
            mid2 = M.mid_times_ttv(
                grid, mx.array((dtau + float(decoy))[None, :],
                               dtype=mx.float64))
            f = f * np.array(M.transit_flux(
                grid, mid=mid2, k=col(K_T), b=col(B_T), T14=col(T14_T),
                q1=col(Q1_T), q2=col(Q2_T), period=col(P_REF)),
                dtype=np.float64)[0]

    x = (ed["times_padded"] - ed["epoch_centers"][:, None]) / ed["half_window"]
    base = 1.0 + (2e-4 * x - 1.5e-4 * (1.5 * x**2 - 0.5) if trend else 0.0)
    obs = f * base + rng.normal(0.0, yerr, f.shape)
    ed = dict(ed)
    ed["flux_padded"] = np.where(ed["mask"] > 0, obs, 1.0)
    return ed, cen, dtau, n_ep


def sweep(ed, cen, tau_half, dtype=mx.float64, **kw):
    orders = np.full(ed["n_epochs"], 2)
    ctx = mx.stream(mx.cpu) if dtype == mx.float64 else _null()
    with ctx:
        design = profile.build_design(ed, orders, dtype=dtype)
        return seeding.template_sweep_taus(
            ed, cen, design, tau_half=tau_half, k=K_T, b=B_T, T14=T14_T,
            q1=Q1_T, q2=Q2_T, period=P_REF, num_resample=7,
            exposure_time=29.4 / 1440, dtype=dtype, **kw)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def test_sweep_recovers_zero_ttv():
    ed, cen, dtau, n_ep = build_dataset(0.0)
    seeds, gap, scores, grids = sweep(ed, cen, tau_half=0.05)
    assert seeds.shape == (n_ep,)
    # a cadence is 0.0204 d; seeds should land within a fraction of that
    assert np.abs(seeds).max() < 0.01, seeds


@pytest.mark.parametrize("shift", [0.03, -0.035])
def test_sweep_recovers_a_uniformly_displaced_transit(shift):
    """The case a zero-TTV initialization gets wrong."""
    ed, cen, dtau, n_ep = build_dataset(shift)
    seeds, gap, scores, grids = sweep(ed, cen, tau_half=0.06)
    np.testing.assert_allclose(seeds, dtau, atol=0.008)
    # and the seed genuinely beats the zero-TTV start it replaces
    for i in range(n_ep):
        j_seed = int(np.argmin(np.abs(grids[i] - seeds[i])))
        j_zero = int(np.argmin(np.abs(grids[i])))
        assert scores[i, j_seed] > scores[i, j_zero] + 5.0


def test_sweep_recovers_per_epoch_displacements():
    """Each epoch is seeded independently, which is the point of TTV mode."""
    rng = np.random.default_rng(2)
    ed, cen, dtau, n_ep = build_dataset(np.zeros(9))
    true = rng.uniform(-0.03, 0.03, ed["n_epochs"])
    ed, cen, dtau, n_ep = build_dataset(true)
    seeds, gap, _, _ = sweep(ed, cen, tau_half=0.05)
    np.testing.assert_allclose(seeds, dtau, atol=0.01)


def test_seeds_stay_inside_the_prior():
    ed, cen, dtau, n_ep = build_dataset(0.0)
    half = 0.004                     # much tighter than the true displacement
    seeds, _, _, _ = sweep(ed, cen, tau_half=half)
    assert np.all(np.abs(seeds) <= 0.95 * half + 1e-12)


def test_edge_margin_keeps_the_template_on_real_data():
    """Seeds must not be placed where the template hangs off the window.

    hurin measured a +39 log-unit spurious win from exactly this, so the grid
    is clipped inward by 0.55*T14 from the first and last real point.
    """
    ed, cen, dtau, n_ep = build_dataset(0.0)
    _, _, _, grids = sweep(ed, cen, tau_half=1.0)   # prior far wider than data
    lo, hi = seeding._epoch_time_extent(ed)
    margin = seeding.EDGE_MARGIN_T14 * T14_T
    for i in range(n_ep):
        assert grids[i].min() >= lo[i] + margin - 1e-9
        assert grids[i].max() <= hi[i] - margin + 1e-9


#: A clean, well-detected transit should beat any rival peak by at least this.
RIVAL_MIN = 10.0


def test_rival_gap_is_infinite_when_no_rival_can_exist():
    """With a prior narrower than one duration there is no second mode."""
    ed, cen, dtau, n_ep = build_dataset(0.0)
    _, gap, _, _ = sweep(ed, cen, tau_half=0.06)   # 0.06 d < T14 = 0.175 d
    assert np.all(np.isinf(gap))


#: Timing prior wide enough that a rival peak can exist (> T14, << P/2).
WIDE = 0.6


@pytest.mark.parametrize("yerr", [1.5e-4, 1e-3, 6e-3])
def test_a_decoy_dip_collapses_the_rival_gap(yerr):
    """The diagnostic must respond to genuine competition, at any SNR.

    This is hurin's actual failure mode: a second dip that the timing seed
    could latch onto instead of the transit. Measured, the decoy shrinks the
    gap by 7x to 40x depending on SNR, so an order of magnitude is a safe
    claim at all three noise levels.
    """
    clean = _finite_gap(build_dataset(0.0, yerr=yerr))
    decoy = _finite_gap(build_dataset(0.0, yerr=yerr, decoy=0.4))
    assert np.median(decoy) < 0.25 * np.median(clean), (
        f"clean {np.median(clean):.1f} vs decoy {np.median(decoy):.1f}")


def test_rival_gap_falls_below_the_warning_threshold_when_modes_compete():
    """At the SNR where two dips are genuinely tied, the warning must fire."""
    ed, cen, _, n_ep = build_dataset(0.0, yerr=6e-3, decoy=0.4)
    seeds, gap, _, _ = sweep(ed, cen, tau_half=WIDE)
    finite = gap[np.isfinite(gap)]
    assert finite.size == n_ep
    assert np.median(finite) < seeding.RIVAL_GAP_WARN, finite
    # with neither dip dominating, the tie-break keeps the seed near the
    # predicted time rather than committing to the decoy
    assert np.abs(seeds).max() < 0.1, seeds


def test_rival_gap_matches_its_definition():
    """gap = score(seed) - best score more than one T14 away."""
    ed, cen, _, n_ep = build_dataset(0.0, decoy=0.4)
    seeds, gap, scores, grids = sweep(ed, cen, tau_half=WIDE)
    for i in range(n_ep):
        j = int(np.argmin(np.abs(grids[i] - seeds[i])))
        far = np.abs(grids[i] - seeds[i]) > T14_T
        assert far.any()
        np.testing.assert_allclose(gap[i], scores[i, j] - scores[i][far].max(),
                                   rtol=1e-12)


def _finite_gap(dataset):
    ed, cen = dataset[0], dataset[1]
    _, gap, _, _ = sweep(ed, cen, tau_half=WIDE)
    finite = gap[np.isfinite(gap)]
    assert finite.size == ed["n_epochs"]
    return finite


def test_fp32_sweep_agrees_with_fp64():
    ed, cen, dtau, n_ep = build_dataset(0.02)
    s64, _, _, _ = sweep(ed, cen, tau_half=0.05, dtype=mx.float64)
    s32, _, _, _ = sweep(ed, cen, tau_half=0.05, dtype=mx.float32)
    # both should pick the same basin; the grid spacing is ~2e-4 d
    np.testing.assert_allclose(s32, s64, atol=1e-3)


# -- MAP ---------------------------------------------------------------

def test_find_map_recovers_injected_parameters():
    ed, cen, dtau, n_ep = build_dataset(0.0)
    orders = np.full(n_ep, 2)
    layout = params.lineph_layout(EPH)
    target, transform, lp, _ = likelihood.build_target(
        layout, cen, ed, orders, num_resample=7, exposure_time=29.4 / 1440,
        n_chains_hint=128, fp64=False)

    v0 = seeding.initial_model_vector(layout, EPH)
    rng = np.random.default_rng(1)
    u0 = (transform.from_model_np(v0)[None, :]
          + 0.3 * rng.standard_normal((128, layout.dim)))
    res = seeding.find_map(target, u0.astype(np.float32), n_iter=300)

    assert np.isfinite(res.log_prob_best)
    v = transform.model_np(res.u_best[None, :])[0]
    truth = dict(k=K_T, beta=B_T / (1 + K_T), T14=T14_T)
    for name, want in truth.items():
        got = v[layout.index(name)]
        assert abs(got - want) < 0.25 * abs(want) + 0.02, (
            f"{name}: MAP {got} vs truth {want}")
    # the MAP must beat every start it came from
    assert res.log_prob_best >= np.max(res.log_prob_all) - 1e-6
    assert res.ball(16).shape == (16, layout.dim)


def test_default_shape_start_avoids_the_grazing_boundary():
    """hurin's lesson: the prior median puts b at exactly 1.0."""
    for b_prior in ("transiting", "nongrazing", "box"):
        layout = params.lineph_layout(EPH, b_prior=b_prior)
        s = seeding.default_shape_start(EPH, layout)
        b = M.impact_parameter(s["beta"], s["k"], b_prior)
        assert 0.05 < b < 0.6, (b_prior, b)
        assert 0.0 < s["beta"] < 1.0
        assert 0.0 < s["T14"] < layout.hi[layout.index("T14")]
        # k from the archive depth
        np.testing.assert_allclose(s["k"], np.sqrt(12000.0 / 1e6), rtol=1e-6)


def test_initial_model_vector_shapes_and_clipping():
    ed, cen, dtau, n_ep = build_dataset(0.0)
    lin = params.lineph_layout(EPH)
    v = seeding.initial_model_vector(lin, EPH, dP=0.001, dtau0=-0.002)
    assert v.shape == (7,)
    assert v[0] == 0.001 and v[1] == -0.002

    # --ld=collapsed: the same vector without q1, q2, built by name
    col = params.lineph_layout(EPH, ld="collapsed")
    vc = seeding.initial_model_vector(col, EPH, dP=0.001, dtau0=-0.002)
    assert col.names == params.LINEPH_BASE_COLLAPSED
    np.testing.assert_array_equal(vc, v[:5])

    ttv = params.ttv_layout(EPH, cen, tau_half=0.01)
    v2 = seeding.initial_model_vector(ttv, EPH, dtau=np.full(n_ep, 0.5))
    assert v2.shape == (5 + n_ep,)
    # clipped strictly inside the box, since the bound is infinity in u space
    assert np.all(np.abs(v2[5:]) < 0.01)
    with pytest.raises(ValueError, match="dtau has shape"):
        seeding.initial_model_vector(ttv, EPH, dtau=np.zeros(n_ep + 1))
