"""The batched profile solve, against independent references.

The unrolled Cholesky is checked against ``np.linalg.solve`` on the same
normal equations (float64), the ``ratio`` mode against hurin's own
``profile_detrend_epoch``, and the ``exact`` mode against the definition it
claims to implement: it must actually minimize the weighted chi-squared.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

from turin import model as M
from turin import prep, profile

_HURIN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hurin"))

P_REF, EPOCH, DUR_H = 9.3456, 120.5, 4.2
K, B, T14, Q1, Q2 = 0.08, 0.35, DUR_H / 24.0, 0.30, 0.225


@pytest.fixture(scope="module")
def hurin_tf():
    if not os.path.isdir(_HURIN):
        pytest.skip("hurin clone not found beside turin")
    if _HURIN not in sys.path:
        sys.path.insert(0, _HURIN)
    try:
        import hurin.transit_fit as tf
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot import hurin.transit_fit: {exc}")
    return tf


@pytest.fixture(scope="module")
def segmented():
    """Segmented epochs from a synthetic curve with per-epoch trends."""
    rng = np.random.default_rng(31337)
    t = np.arange(100.0, 160.0, 29.4 / 1440.0)
    t = t[(t < 128.0) | (t > 130.2)]          # a gap, so epochs differ in length
    phase = (t - EPOCH + 0.5 * P_REF) % P_REF - 0.5 * P_REF
    flux = 1.0 + rng.normal(0.0, 2e-4, t.size)
    flux += 4e-4 * np.sin(2 * np.pi * t / 5.1)   # something for the polynomial
    flux[np.abs(phase) < 0.5 * T14] -= 1.2e-3
    ferr = np.full(t.size, 2e-4)
    ferr[::7] = 3e-4                            # heteroscedastic
    tw, fw, ew = prep.extract_near_transit_data(
        t, flux, ferr, P_REF, EPOCH, DUR_H)
    ed = prep.segment_epochs(tw, fw, ew, P_REF, EPOCH, DUR_H)
    assert ed["n_epochs"] >= 4
    return ed


def transit_on(segmented, centering, n_chains=3, dtype=mx.float64, seed=0):
    """A batch of transit flux DEVIATIONS (f - 1) over the segmented grid."""
    rng = np.random.default_rng(seed)
    grid = M.build_grid(centering, prep.supersample_offsets(0.0, 1), dtype=dtype)
    col = lambda v: mx.array(np.asarray(v, dtype=np.float64).reshape(-1, 1),
                             dtype=dtype)
    ks = np.clip(K + rng.normal(0, 0.005, n_chains), 0.02, 0.3)
    return M.transit_flux_dev(
        grid,
        mid=mx.zeros((n_chains, grid.n_epochs), dtype=dtype),
        k=col(ks), b=col(np.full(n_chains, B)), T14=col(np.full(n_chains, T14)),
        q1=col(np.full(n_chains, Q1)), q2=col(np.full(n_chains, Q2)),
        period=col(np.full(n_chains, P_REF)),
    )


@pytest.fixture(scope="module")
def centering(segmented):
    return prep.centering_constants(
        segmented, {"period": P_REF, "epoch": EPOCH})


@pytest.mark.parametrize("mode", profile.PROFILE_MODES)
@pytest.mark.parametrize("order", [0, 1, 2, 5])
def test_matches_numpy_normal_equations(segmented, centering, mode, order):
    """The unrolled Cholesky must agree with np.linalg.solve in float64."""
    n_epochs = segmented["n_epochs"]
    orders = np.full(n_epochs, order)
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        got = np.array(profile.solve_coefficients(design, f, mode),
                       dtype=np.float64)
    ref = profile.solve_coefficients_np(
        segmented, orders, 1.0 + np.array(f, dtype=np.float64), mode)
    assert got.shape == ref.shape
    # hybrid converges to the exact normal equations rather than solving them
    # outright, so it gets the tolerance its refinement count earns
    rtol = 1e-5 if mode == "hybrid" else 1e-9
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=1e-12)
    # inactive columns must be exactly zero, not merely small
    if design.n_cols > order + 1:
        assert np.all(got[..., order + 1:] == 0.0)


def test_mixed_orders_across_epochs(segmented, centering):
    """Per-epoch orders are the real use case: CV picks a different K each."""
    n_epochs = segmented["n_epochs"]
    orders = np.arange(n_epochs) % 4
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        got = np.array(profile.solve_coefficients(design, f, "exact"),
                       dtype=np.float64)
    ref = profile.solve_coefficients_np(
        segmented, orders, 1.0 + np.array(f, dtype=np.float64), "exact")
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)
    for e in range(n_epochs):
        assert np.all(got[:, e, orders[e] + 1:] == 0.0), e


def test_exact_mode_actually_minimizes_chi_squared(segmented, centering):
    """The defining property: no perturbation of c can lower the chi-squared."""
    orders = np.full(segmented["n_epochs"], 2)
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, n_chains=1, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        c = profile.solve_coefficients(design, f, "exact")

        def chi2(coeffs):
            r = profile.residual_dev(design, f, coeffs) * design.inv_sigma
            return float(mx.sum(design.mask * r * r))

        best = chi2(c)
        rng = np.random.default_rng(4)
        scale = float(mx.max(mx.abs(c))) or 1.0
        for _ in range(12):
            pert = mx.array(rng.normal(0, 0.02 * scale, c.shape),
                            dtype=mx.float64)
            assert chi2(c + pert) >= best * (1 - 1e-12)

        # and the gradient of chi^2 w.r.t. c vanishes at the solution
        def chi2_mx(cc):
            r = profile.residual_dev(design, f, cc) * design.inv_sigma
            return mx.sum(design.mask * r * r)

        g = mx.grad(chi2_mx)(c)
        assert float(mx.max(mx.abs(g))) < 1e-6 * max(1.0, best)


_HURIN_PY = "/Users/dkipping/miniconda3/envs/hurin/bin/python"


@pytest.mark.skipif(not os.path.exists(_HURIN_PY),
                    reason="hurin conda env not installed")
def test_ratio_mode_matches_hurin_profile_detrend(segmented, centering, tmp_path):
    """ratio mode must reproduce hurin's own solve, coefficient for coefficient.

    hurin's ``profile_detrend_epoch`` is JAX (and float32), so it runs in the
    hurin environment via a subprocess rather than being imported here.
    """
    n_epochs = segmented["n_epochs"]
    max_order = 2
    orders = np.full(n_epochs, max_order)
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, n_chains=1, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        got = np.array(profile.solve_coefficients(design, f, "ratio"),
                       dtype=np.float64)[0]
        model_mine = np.array(
            profile.detrended_model(design, f, "ratio")[0], dtype=np.float64)[0]

    inp = tmp_path / "in.npz"
    out = tmp_path / "out.npz"
    np.savez(inp, f=1.0 + np.array(f, dtype=np.float64)[0],
             flux=segmented["flux_padded"], ferr=segmented["ferr_padded"],
             times=segmented["times_padded"],
             centers=segmented["epoch_centers"], mask=segmented["mask"],
             half_window=np.array(segmented["half_window"]))
    code = (
        "import numpy as np, sys\n"
        f"sys.path.insert(0, {str(_HURIN)!r})\n"
        "import hurin.transit_fit as tf\n"
        f"d = np.load({str(inp)!r})\n"
        f"K = {max_order}\n"
        "cs, ms = [], []\n"
        "for e in range(d['mask'].shape[0]):\n"
        "    m, c = tf.profile_detrend_epoch(\n"
        "        d['f'][e], d['flux'][e], d['ferr'][e], d['times'][e],\n"
        "        d['centers'][e], float(d['half_window']), d['mask'][e], K, K)\n"
        "    cs.append(np.asarray(c, dtype=np.float64))\n"
        "    ms.append(np.asarray(m, dtype=np.float64))\n"
        f"np.savez({str(out)!r}, c=np.array(cs), m=np.array(ms))\n"
    )
    r = subprocess.run([_HURIN_PY, "-c", code], capture_output=True,
                       text=True, timeout=900)
    if r.returncode != 0:
        pytest.skip(f"hurin profile call failed: {r.stderr[-400:]}")
    ref = np.load(out)

    # hurin solves this in float32 on a normal matrix of scale 1/sigma^2 ~ 1e7,
    # so its coefficients carry ~1e-8 absolute noise. That, not turin, sets the
    # tolerance; 1e-7 is still four orders below a physically meaningful
    # baseline coefficient (~1e-3).
    np.testing.assert_allclose(got, ref["c"], rtol=3e-5, atol=1e-7)
    real = segmented["mask"] > 0
    np.testing.assert_allclose(model_mine[real], ref["m"][real],
                               rtol=0, atol=2e-7)


@pytest.mark.parametrize("depth", [1e-4, 1e-3, 1e-2])
def test_hybrid_converges_to_exact_geometrically_in_depth(depth):
    """Each refinement step must contract the error by ~the transit depth.

    That is the claim the mode rests on: ratio's static factor is a
    preconditioner whose error is O(depth), so refinement is geometric and a
    fixed, small number of steps suffices.
    """
    n_ep, n_pts = 4, 90
    t = np.linspace(-0.4, 0.4, n_pts)
    rng = np.random.default_rng(3)
    ed = dict(times_padded=np.tile(t, (n_ep, 1)),
              flux_padded=1.0 + rng.normal(0, 2e-4, (n_ep, n_pts)),
              ferr_padded=np.full((n_ep, n_pts), 2e-4),
              mask=np.ones((n_ep, n_pts)), epoch_centers=np.zeros(n_ep),
              half_window=0.5, n_epochs=n_ep, max_pts=n_pts)
    prof = np.where(np.abs(t) < 0.08, -depth, 0.0)

    with mx.stream(mx.cpu):
        design = profile.build_design(ed, np.full(n_ep, 2), dtype=mx.float64)
        f_dev = mx.array(np.tile(prof, (n_ep, 1))[None], dtype=mx.float64)
        ref = np.array(profile.solve_coefficients(design, f_dev, "exact"),
                       dtype=np.float64)
        scale = np.abs(ref).max()
        errs = []
        for k in (0, 1, 2):
            old = profile.N_REFINE
            profile.N_REFINE = k
            try:
                got = np.array(
                    profile.solve_coefficients(design, f_dev, "hybrid"),
                    dtype=np.float64)
            finally:
                profile.N_REFINE = old
            errs.append(np.abs(got - ref).max() / scale)

    # Each step contracts by ~the depth, but never below float64 round-off
    # on these quantities (~1e-11), so the bound carries that floor.
    for a, b in zip(errs, errs[1:]):
        assert b < max(a * depth * 3.0, 1e-11), errs
    assert errs[0] > errs[-1] * 100, errs
    # and the shipped default is at or below float32's own precision
    with mx.stream(mx.cpu):
        got = np.array(profile.solve_coefficients(design, f_dev, "hybrid"),
                       dtype=np.float64)
    assert np.abs(got - ref).max() / scale < 1.2e-7, profile.N_REFINE


def test_hybrid_beats_ratio_at_every_depth(segmented, centering):
    """hybrid must be strictly more accurate than the ratio form it refines."""
    orders = np.full(segmented["n_epochs"], 2)
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, n_chains=1, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        ref = np.array(profile.solve_coefficients(design, f, "exact"))
        hyb = np.array(profile.solve_coefficients(design, f, "hybrid"))
        rat = np.array(profile.solve_coefficients(design, f, "ratio"))
    e_h = np.abs(hyb - ref).max()
    e_r = np.abs(rat - ref).max()
    assert e_h < 1e-3 * e_r, (e_h, e_r)


def test_exact_and_ratio_differ_by_order_depth(segmented, centering):
    """The two profiles agree to O(depth) -- the documented approximation."""
    orders = np.full(segmented["n_epochs"], 2)
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, n_chains=1, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        c_e = np.array(profile.solve_coefficients(design, f, "exact"))
        c_r = np.array(profile.solve_coefficients(design, f, "ratio"))
        depth = -float(mx.min(f))
    assert depth > 1e-3
    d = np.abs(c_e - c_r).max()
    assert d > 0, "the two modes should not be identical"
    assert d < 10 * depth * np.abs(c_e).max() + 1e-6


def test_log_likelihood_matches_direct_float64_computation(segmented, centering):
    """The recentred sum plus the host constant must equal the plain formula."""
    orders = np.full(segmented["n_epochs"], 2)
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, n_chains=3, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        term, model, _ = profile.log_likelihood(design, f, "exact")
        logl = np.array(term, dtype=np.float64) + design.log_const
        model = np.array(model, dtype=np.float64)

    y = segmented["flux_padded"]
    err = segmented["ferr_padded"]
    m = segmented["mask"] > 0
    for ch in range(3):
        r = (y[m] - model[ch][m]) / err[m]
        direct = (-0.5 * np.sum(r**2) - np.sum(np.log(err[m]))
                  - 0.5 * m.sum() * np.log(2 * np.pi))
        np.testing.assert_allclose(logl[ch], direct, rtol=1e-10)


def test_recentring_identity_holds_exactly(segmented, centering):
    """The graph term is exactly ``N/2 - chi2/2``, with ``-N/2`` on the host.

    That identity is what lets the float32 sum be O(sqrt(N)) for a good fit
    instead of O(N/2), whose ulp would rival the ~1-unit Metropolis scale.
    (The magnitude of the term itself tracks the fit quality, so it is the
    identity, not the size, that can be asserted on synthetic data.)
    """
    orders = np.full(segmented["n_epochs"], 2)
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, n_chains=1, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        term, model, _ = profile.log_likelihood(design, f, "exact")
        model = np.array(model, dtype=np.float64)[0]

    n = design.n_real
    assert n > 500
    m = segmented["mask"] > 0
    r = (segmented["flux_padded"][m] - model[m]) / segmented["ferr_padded"][m]
    chi2 = float(np.sum(r**2))
    np.testing.assert_allclose(float(term), 0.5 * n - 0.5 * chi2, rtol=1e-10)
    # the -N/2 offset is carried on the host in float64, alongside the usual
    # Gaussian normalization (which is large and positive for sigma << 1)
    err = segmented["ferr_padded"][m]
    normalization = -np.sum(np.log(err)) - 0.5 * n * np.log(2 * np.pi)
    np.testing.assert_allclose(design.log_const + 0.5 * n, normalization,
                               rtol=1e-12)


def test_fp32_agrees_with_fp64_on_the_likelihood(segmented, centering):
    """float32 must hold the log-likelihood to well under the Metropolis scale."""
    orders = np.full(segmented["n_epochs"], 2)
    with mx.stream(mx.cpu):
        f64 = transit_on(segmented, centering, n_chains=4, dtype=mx.float64)
        d64 = profile.build_design(segmented, orders, dtype=mx.float64)
        t64 = np.array(profile.log_likelihood(d64, f64, "exact")[0],
                       dtype=np.float64)
    f32 = transit_on(segmented, centering, n_chains=4, dtype=mx.float32)
    d32 = profile.build_design(segmented, orders, dtype=mx.float32)
    t32 = np.array(profile.log_likelihood(d32, f32, "exact")[0], dtype=np.float64)
    err = np.abs(t32 - t64).max()
    assert err < 0.1, f"fp32 log-likelihood error {err} exceeds the OK band"


def test_epoch_block_selection_partitions_the_sum(segmented, centering):
    """Chunking over epoch blocks must be exact, not approximate."""
    n_epochs = segmented["n_epochs"]
    orders = np.arange(n_epochs) % 3
    with mx.stream(mx.cpu):
        f = transit_on(segmented, centering, n_chains=2, dtype=mx.float64)
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        whole = np.array(profile.log_likelihood(design, f, "exact")[0],
                         dtype=np.float64)
        cut = n_epochs // 2
        part = np.zeros(2)
        for lo, hi in ((0, cut), (cut, n_epochs)):
            sub = design.select(lo, hi)
            part += np.array(
                profile.log_likelihood(sub, f[:, lo:hi], "exact")[0],
                dtype=np.float64)
    np.testing.assert_allclose(part, whole, rtol=1e-11)


def test_gradient_flows_through_the_solve(segmented, centering):
    """The envelope theorem is not assumed: autodiff goes through the Cholesky."""
    orders = np.full(segmented["n_epochs"], 2)
    with mx.stream(mx.cpu):
        design = profile.build_design(segmented, orders, dtype=mx.float64)
        grid = M.build_grid(centering, prep.supersample_offsets(0.0, 1),
                            dtype=mx.float64)

        def logl(v):
            col = lambda i: v[i].reshape(1, 1)
            b = M.impact_parameter(col(1), col(0), "transiting")
            f = M.transit_flux_dev(
                grid, mid=M.mid_times_lineph(grid, col(4), col(5)),
                k=col(0), b=b, T14=col(2), q1=col(3),
                q2=mx.array([[Q2]], dtype=mx.float64),
                period=grid.P_ref + col(4))
            return profile.log_likelihood(design, f, "exact")[0].sum()

        v0 = np.array([K, B / (1 + K), T14, Q1, 0.0, 0.0])
        g = np.array(mx.grad(logl)(mx.array(v0, dtype=mx.float64)))
        assert np.all(np.isfinite(g))
        assert np.abs(g[:4]).min() > 0, "shape parameters must have gradient"

        for i in (0, 2, 3):
            h = 1e-6 * max(1.0, abs(v0[i]))
            vp, vm = v0.copy(), v0.copy()
            vp[i] += h
            vm[i] -= h
            fd = (float(logl(mx.array(vp, dtype=mx.float64)))
                  - float(logl(mx.array(vm, dtype=mx.float64)))) / (2 * h)
            rel = abs(g[i] - fd) / max(1e-12, abs(fd))
            assert rel < 2e-4, f"param {i}: analytic {g[i]} vs fd {fd}"


def test_build_design_caps_orders_and_validates():
    """An epoch with too few points must not get an underdetermined system."""
    ed = {
        "times_padded": np.array([[0.0, 0.1, 0.2, 0.0]]),
        "flux_padded": np.array([[1.0, 1.0, 1.0, 1.0]]),
        "ferr_padded": np.array([[1e-4, 1e-4, 1e-4, 1e10]]),
        "mask": np.array([[1.0, 1.0, 1.0, 0.0]]),
        "epoch_centers": np.array([0.1]),
        "half_window": 0.5,
        "n_epochs": 1,
        "max_pts": 4,
    }
    with mx.stream(mx.cpu):
        d = profile.build_design(ed, np.array([5]), dtype=mx.float64)
    assert d.orders[0] == 2, "order capped to n_real - 1"
    with pytest.raises(ValueError, match="orders has shape"):
        profile.build_design(ed, np.array([1, 2]))
