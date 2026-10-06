"""The MLX forward model, against independent references.

Four checks, in decreasing order of authority:

1. fp64 ``circular`` against MetalPlanet's own batman-style frontend, which
   is verified against a 30-digit mpmath oracle to 2e-16. This is the real
   correctness test: it pins turin's geometry, limb darkening and exposure
   wiring all at once.
2. fp32 against the same, to MetalPlanet's documented fp32 floor.
3. Gradients by exact structural invariants plus finite differences. Note
   FD is a *weak* reference here: MLX's fp64 ``sin``/``cos`` are only
   float32-accurate, so the model is a staircase at ~1e-8 and a small-h FD
   differentiates the staircase. The invariants are exact and are what
   actually pins the gradient.
4. ``chord`` + the ``hurin`` LD map against hurin/jaxoplanet, to hurin's own
   float32 floor. Skipped when the hurin env is absent.
"""

from __future__ import annotations

import functools
import math
import os
import subprocess
import tempfile

import mlx.core as mx
import numpy as np
import pytest
import metalplanet
from metalplanet.api import TransitModel, TransitParams

from turin import model as M
from turin import prep

P_REF, T14, K, B = 9.3456, 0.175, 0.08, 0.35
Q1, Q2 = 0.30, 0.225
U1, U2 = M.limb_dark_coeffs_np(Q1, Q2)
TIMES = np.linspace(-0.25, 0.25, 601)

_HURIN_PY = "/Users/dkipping/miniconda3/envs/hurin/bin/python"
_HURIN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "hurin"))


def one_epoch_grid(times=TIMES, n=0.0, d=0.0, n_sub=1, exp_time=0.0,
                   dtype=mx.float64, n_gl=M.N_GL):
    centering = dict(times_centered=np.asarray(times)[None, :],
                     n_arr=np.array([float(n)]), d_arr=np.array([float(d)]),
                     P_ref=P_REF, tau0_ref=0.0)
    # exp_time drives BOTH routes: the fallback's sub-exposure offsets and
    # the tau kernel's in-kernel contact rule
    return M.build_grid(centering, prep.supersample_offsets(exp_time, n_sub),
                        dtype=dtype, exp_time=exp_time, n_gl=n_gl)


def turin_flux(*, grid, k=K, b=B, T14_=T14, q1=Q1, q2=Q2, period=P_REF,
               mid=0.0, geometry="circular"):
    dt = grid.dtype
    col = lambda v: mx.array([[float(v)]], dtype=dt)
    f = M.transit_flux(grid, mid=col(mid), k=col(k), b=col(b), T14=col(T14_),
                       q1=col(q1), q2=col(q2), period=col(period),
                       geometry=geometry)
    return np.array(f, dtype=np.float64)[0, 0]


def metalplanet_reference(n_sub=1, exp_time=0.0):
    """MetalPlanet's frontend at the a/R* and inclination turin implies."""
    with mx.stream(mx.cpu):
        aRs = float(M.a_over_rstar(*[mx.array([[v]], dtype=mx.float64)
                                     for v in (T14, P_REF, K, B)])[0, 0])
    pars = TransitParams()
    pars.t0, pars.per, pars.rp, pars.a = 0.0, P_REF, K, aRs
    pars.inc = math.degrees(math.acos(B / aRs))
    pars.ecc, pars.w = 0.0, 90.0
    pars.u, pars.limb_dark = [float(U1), float(U2)], "quadratic"
    kwargs = dict(dtype=mx.float64)
    if n_sub > 1:
        kwargs.update(supersample_factor=n_sub, exp_time=exp_time)
    return TransitModel(pars, TIMES, **kwargs).light_curve(pars), aRs


def test_fp64_circular_matches_metalplanet_frontend():
    ref, aRs = metalplanet_reference()
    with mx.stream(mx.cpu):
        got = turin_flux(grid=one_epoch_grid())
    assert 1.0 < aRs < 200.0
    assert np.abs(got - ref).max() < 1e-14, "fp64 must be exact to round-off"
    assert 1 - got.min() > 1e-3, "sanity: there should be a visible transit"


def test_fp32_circular_matches_to_metalplanet_fp32_floor():
    ref, _ = metalplanet_reference()
    got = turin_flux(grid=one_epoch_grid(times=TIMES.astype(np.float32),
                                         dtype=mx.float32))
    assert np.abs(got - ref).max() < 5e-7


def metalplanet_contact(n_gl, exp_time, k=K, b=B):
    """MetalPlanet's frontend with the same contact rule turin now uses."""
    with mx.stream(mx.cpu):
        aRs = float(M.a_over_rstar(*[mx.array([[v]], dtype=mx.float64)
                                     for v in (T14, P_REF, k, b)])[0, 0])
    pars = TransitParams()
    pars.t0, pars.per, pars.rp, pars.a = 0.0, P_REF, k, aRs
    pars.inc = math.degrees(math.acos(b / aRs))
    pars.ecc, pars.w = 0.0, 90.0
    pars.u, pars.limb_dark = [float(U1), float(U2)], "quadratic"
    return TransitModel(pars, TIMES, dtype=mx.float64, exp_time=exp_time,
                        integration="contact", n_gl=n_gl).light_curve(pars)


def test_exposure_integration_matches_metalplanet_contact_rule():
    """turin's in-kernel integration is MetalPlanet's, so it must agree exactly."""
    exp = 29.4 / 1440
    ref = metalplanet_contact(M.N_GL, exp)
    with mx.stream(mx.cpu):
        got = turin_flux(grid=one_epoch_grid(n_sub=1, exp_time=exp))
    assert np.abs(got - ref).max() < 1e-14


def test_contact_integration_is_far_more_accurate_than_supersampling():
    """Why turin switched: the route it replaced was wrong by ~2% of a depth.

    Both are compared against a high-order float64 contact reference, which
    is neither of the rules being judged -- scoring supersampling against a
    supersampled reference would flatter it.
    """
    exp = 29.4 / 1440
    ref = metalplanet_contact(12, exp)
    with mx.stream(mx.cpu):
        contact = turin_flux(grid=one_epoch_grid(n_sub=1, exp_time=exp))
        old = M.HAS_TAU_KERNEL
        M.HAS_TAU_KERNEL = False
        try:
            supersampled = turin_flux(
                grid=one_epoch_grid(n_sub=8, exp_time=exp))
        finally:
            M.HAS_TAU_KERNEL = old

    err_contact = np.abs(contact - ref).max()
    err_super = np.abs(supersampled - ref).max()
    assert err_contact < 1e-7, err_contact          # below the float32 floor
    assert err_super > 100 * err_contact, (err_contact, err_super)
    # and the old error was a sizeable fraction of the transit depth
    depth = 1.0 - ref.min()
    assert err_super > 0.005 * depth


def test_the_fallback_route_still_matches_metalplanet_supersampling():
    """Without the tau kernel turin supersamples, and must do so exactly.

    This is the route ``geometry="chord"`` always takes, and the one any
    machine without the kernel falls back to.
    """
    exp = 29.4 / 1440
    for n_sub in (7, 15):
        with mx.stream(mx.cpu):
            aRs = float(M.a_over_rstar(*[mx.array([[v]], dtype=mx.float64)
                                         for v in (T14, P_REF, K, B)])[0, 0])
            pars = TransitParams()
            pars.t0, pars.per, pars.rp, pars.a = 0.0, P_REF, K, aRs
            pars.inc = math.degrees(math.acos(B / aRs))
            pars.ecc, pars.w = 0.0, 90.0
            pars.u, pars.limb_dark = [float(U1), float(U2)], "quadratic"
            ref = TransitModel(pars, TIMES, dtype=mx.float64,
                               supersample_factor=n_sub,
                               exp_time=exp).light_curve(pars)
            old = M.HAS_TAU_KERNEL
            M.HAS_TAU_KERNEL = False
            try:
                got = turin_flux(grid=one_epoch_grid(n_sub=n_sub,
                                                     exp_time=exp))
            finally:
                M.HAS_TAU_KERNEL = old
        assert np.abs(got - ref).max() < 1e-14, n_sub


def test_the_period_gradient_survives_exposure_integration():
    """The reason for the switch, pinned.

    Differentiating a supersampled kinked integrand amplifies its error: the
    route turin used before got ``dF/d(period)`` wrong by ~100x *and* with
    the wrong sign once exposure integration was on. The reference is a
    float64 high-order contact model differentiated by central differences,
    independent of either route.
    """
    exp, n_arr, dtau0 = 29.4 / 1440, 3.0, 0.001

    def reference(dP):
        per = P_REF + dP
        with mx.stream(mx.cpu):
            aRs = float(M.a_over_rstar(
                *[mx.array([[v]], dtype=mx.float64)
                  for v in (T14, per, K, B)])[0, 0])
        pars = TransitParams()
        pars.t0 = dtau0 + n_arr * dP
        pars.per, pars.rp, pars.a = per, K, aRs
        pars.inc = math.degrees(math.acos(B / aRs))
        pars.ecc, pars.w = 0.0, 90.0
        pars.u, pars.limb_dark = [float(U1), float(U2)], "quadratic"
        f = TransitModel(pars, TIMES, dtype=mx.float64, exp_time=exp,
                         integration="contact", n_gl=16).light_curve(pars)
        return float(np.sum(np.asarray(f, dtype=np.float64) ** 2))

    h = 1e-6
    fd = (reference(h) - reference(-h)) / (2 * h)

    def analytic(use_kernel):
        def scalar(v):
            # n_gl pinned: this tests the route, not the production N_GL,
            # whose cost/accuracy trade is judged on logL (see model.N_GL).
            # The kernel's bare period partial is the slowest quantity to
            # converge: 11.6% off at five nodes, inside 5% from nine.
            grid = one_epoch_grid(n=n_arr, n_sub=8, exp_time=exp, n_gl=9)
            col = lambda i: v[i].reshape(1, 1)
            b = M.impact_parameter(col(3), col(2), "transiting")
            f = M.transit_flux(
                grid, mid=M.mid_times_lineph(grid, col(0), col(1)),
                k=col(2), b=b, T14=col(4), q1=col(5), q2=col(6),
                period=grid.P_ref + col(0))
            return mx.sum(f * f)

        v0 = np.array([0.0, dtau0, K, B / (1 + K), T14, Q1, Q2])
        old = M.HAS_TAU_KERNEL
        M.HAS_TAU_KERNEL = use_kernel
        try:
            with mx.stream(mx.cpu):
                return float(np.array(mx.grad(scalar)(
                    mx.array(v0, dtype=mx.float64)))[0])
        finally:
            M.HAS_TAU_KERNEL = old

    assert abs(analytic(True) - fd) / abs(fd) < 0.05
    # and the route it replaced really was that bad
    bad = analytic(False)
    assert abs(bad - fd) / abs(fd) > 1.0
    assert np.sign(bad) != np.sign(fd)


def test_exposure_time_alone_decides_whether_smearing_happens():
    """The tau kernel integrates from exp_time; n_sub only feeds the fallback.

    This is a cleaner contract than the one it replaced, where the
    sub-exposure count both chose the quadrature *and* decided whether
    integration happened at all.
    """
    exp = 29.4 / 1440
    with mx.stream(mx.cpu):
        instant = turin_flux(grid=one_epoch_grid(n_sub=1, exp_time=0.0))
        # n_sub is irrelevant on the kernel path: both integrate
        smeared_1 = turin_flux(grid=one_epoch_grid(n_sub=1, exp_time=exp))
        smeared_8 = turin_flux(grid=one_epoch_grid(n_sub=8, exp_time=exp))

    np.testing.assert_array_equal(smeared_1, smeared_8)
    # and smearing a 29-minute exposure across a 4.2 h transit does something
    assert np.abs(smeared_1 - instant).max() > 1e-5
    assert (1 - instant.min()) > (1 - smeared_1.min())     # blunted ingress

    # on the fallback route n_sub is what chooses the quadrature, so there
    # n_sub=1 genuinely means instantaneous
    old = M.HAS_TAU_KERNEL
    M.HAS_TAU_KERNEL = False
    try:
        with mx.stream(mx.cpu):
            fb_1 = turin_flux(grid=one_epoch_grid(n_sub=1, exp_time=exp))
            fb_8 = turin_flux(grid=one_epoch_grid(n_sub=8, exp_time=exp))
    finally:
        M.HAS_TAU_KERNEL = old
    np.testing.assert_array_equal(fb_1, instant)
    assert np.abs(fb_8 - instant).max() > 1e-5


def test_geometry_difference_is_second_order_in_T14_over_P():
    """chord and circular agree to O((T14/P)^2) -- and disagree above it."""
    with mx.stream(mx.cpu):
        circ = turin_flux(grid=one_epoch_grid(), geometry="circular")
        chord = turin_flux(grid=one_epoch_grid(), geometry="chord")
    d = np.abs(circ - chord).max()
    # well above the fp32 floor, so the choice is not cosmetic ...
    assert d > 1e-6
    # ... but still a small perturbation on a 7e-3 transit
    assert d < 1e-4


def test_kipping_map_and_its_host_replica():
    """Kipping (2013), and the float64 replica must match the MLX path."""
    u1, u2 = M.limb_dark_coeffs_np(Q1, Q2)
    np.testing.assert_allclose([u1, u2],
                               [2 * math.sqrt(Q1) * Q2,
                                math.sqrt(Q1) * (1 - 2 * Q2)], rtol=1e-14)
    # u1 + u2 = sqrt(q1), so q1 alone fixes the intensity at the limb
    np.testing.assert_allclose(u1 + u2, math.sqrt(Q1), rtol=1e-14)
    with mx.stream(mx.cpu):
        u = M.limb_dark_coeffs(mx.array(Q1, dtype=mx.float64),
                               mx.array(Q2, dtype=mx.float64))
        np.testing.assert_allclose([float(u[0]), float(u[1])], [u1, u2],
                                   rtol=1e-14)


@pytest.mark.parametrize("q1,q2", [(0.3, 0.225), (0.3, 0.45), (0.96, 0.93),
                                   (0.5, 0.0), (0.5, 1.0), (0.99, 0.99)])
def test_kipping_square_maps_into_physical_profiles(q1, q2):
    """Every (q1, q2) in the unit square must give a valid intensity profile.

    That is the whole point of Kipping (2013), and it is what makes the map
    compatible with the polynomial (Agol et al.) formulation MetalPlanet
    implements: the quadratic law is that law's N=2 case.
    """
    u1, u2 = M.limb_dark_coeffs_np(q1, q2)
    mu = np.linspace(0.0, 1.0, 2001)
    I = 1.0 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    assert I.min() >= 0.0, (u1, u2, I.min())          # never negative
    assert np.all(np.diff(I) >= -1e-12)               # brightest at centre
    np.testing.assert_allclose(I[0], 1.0 - math.sqrt(q1), atol=1e-12)


def test_quadratic_law_is_the_n2_case_of_the_polynomial_law():
    """MetalPlanet's 'quadratic' and 'polynomial' must agree at the same u.

    If they did not, Kipping's (q1, q2) -- which parameterizes the *quadratic*
    law -- would not be the right thing to feed this model.
    """
    from metalplanet.api import TransitModel, TransitParams

    tt = np.linspace(-0.2, 0.2, 401)
    for u1, u2 in ((0.35, 0.22), (0.61, 0.36), (1.82, -0.84)):
        out = []
        for law in ("quadratic", "polynomial"):
            p = TransitParams()
            p.t0, p.per, p.rp, p.a, p.inc = 0.0, 9.3456, 0.09, 17.4, 88.85
            p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [u1, u2], law
            out.append(TransitModel(p, tt, dtype=mx.float64).light_curve(p))
        assert np.abs(out[0] - out[1]).max() < 1e-14, (u1, u2)


def test_impact_parameter_maps():
    k = 0.08
    assert M.impact_parameter(0.0, k, "transiting") == 0.0
    assert M.impact_parameter(1.0, k, "transiting") == pytest.approx(1 + k)
    assert M.impact_parameter(1.0, k, "nongrazing") == pytest.approx(1 - k)
    assert M.impact_parameter(1.0, k, "box") == pytest.approx(2.0)
    with pytest.raises(ValueError, match="unknown b_prior"):
        M.impact_parameter(0.5, k, "nope")


def test_a_over_rstar_matches_the_closed_form_derivative():
    """The gradient reference is the analytic dervative, not finite differences.

    MLX's fp64 sin is only float32-accurate, so an FD of this function is
    meaningless below ~1e-7; the closed form is exact.
    """
    with mx.stream(mx.cpu):
        def aRs(T):
            return M.a_over_rstar(T, mx.array(P_REF, dtype=mx.float64),
                                  mx.array(K, dtype=mx.float64),
                                  mx.array(B, dtype=mx.float64))
        g = float(mx.grad(aRs)(mx.array(T14, dtype=mx.float64)))

    sa, ca = math.sin(math.pi * T14 / P_REF), math.cos(math.pi * T14 / P_REF)
    num = (1 + K) ** 2 - B**2
    a = math.sqrt(B**2 + num / sa**2)
    closed = -(num / sa**3) * ca * (math.pi / P_REF) / a
    assert abs(g - closed) / abs(closed) < 1e-6


@pytest.mark.parametrize("geometry", M.GEOMETRIES)
@pytest.mark.parametrize("n_epoch_number", [1.0, 5.0, 12.0])
def test_period_gradient_is_exactly_the_epoch_number_times_tau0(
        geometry, n_epoch_number):
    """An exact structural invariant, and the real gradient test.

    Under a linear ephemeris a single epoch's mid-time is
    ``dtau0 + n dP``, so any functional of the model must satisfy
    ``dF/d(dP) = n dF/d(dtau0)``. In ``chord`` mode ``dP`` enters *only*
    that way, so the identity is exact; in ``circular`` mode the period also
    scales the phase, so it holds to the size of that extra term.
    """
    def scalar(v):
        grid = one_epoch_grid(n=n_epoch_number)
        col = lambda i: v[i].reshape(1, 1)
        dP, dtau0 = col(0), col(1)
        b = M.impact_parameter(col(3), col(2), "transiting")
        mid = M.mid_times_lineph(grid, dP, dtau0)
        f = M.transit_flux(grid, mid=mid, k=col(2), b=b, T14=col(4),
                           q1=col(5), q2=col(6), period=grid.P_ref + dP,
                           geometry=geometry)
        return mx.sum(f * f)

    v0 = np.array([0.0, 0.001, K, B / (1 + K), T14, Q1, Q2])
    with mx.stream(mx.cpu):
        g = np.array(mx.grad(scalar)(mx.array(v0, dtype=mx.float64)))

    assert abs(g[1]) > 1e-6, "gradient should not be trivially zero"
    ratio = g[0] / g[1]
    tol = 1e-12 if geometry == "chord" else 2e-2
    assert abs(ratio - n_epoch_number) / n_epoch_number < tol


GRAZING_EXP = 29.4 / 1440


@functools.lru_cache(maxsize=None)
def _grazing_reference_gradient(k, b):
    """(d/dk, d/db) of sum(f^2), central-differenced on MetalPlanet's frontend.

    The frontend is the same contact rule turin uses (so this is *not* an
    independent implementation of the contact geometry), differenced with
    its split points free to move. At ``n_gl=32`` that estimate is converged:
    against ``n_gl=128`` it agrees to <=1.3e-7, and since the exact integral
    does not depend on where it is split, the converged moving-split FD is
    the true derivative. Cached per (k, b): both ``n_gl`` parametrizations
    of the test share it.
    """
    def ref(k_, b_):
        f = metalplanet_contact(32, GRAZING_EXP, k=k_, b=b_)
        return float(np.sum(np.asarray(f, dtype=np.float64) ** 2))

    h = 1e-6
    return np.array([(ref(k + h, b) - ref(k - h, b)) / (2 * h),
                     (ref(k, b + h) - ref(k, b - h)) / (2 * h)])


@pytest.mark.parametrize("k,b", [
    (0.0604, 0.9322),   # KOI-448.02, turin-defaults mean: inside 1-k = 0.9396
    (0.30, 0.68),       # just inside 1-k = 0.70; the inner pair nearly gone
    (0.30, 0.72),       # grazing: the inner pair collapsed
    (0.30, 1.10),       # deeply grazing, b > 1
    (0.50, 0.95),       # large planet, well past 1-k = 0.50
])
@pytest.mark.parametrize("n_gl,tol", [(16, 1e-5), (None, 5e-4)])
def test_grazing_gradient_matches_a_finite_differenced_reference(k, b, n_gl, tol):
    """d/dk and d/db through the contact rule, at and past the grazing edge.

    What this pins, precisely:

    1. **Finite, non-zero gradients through the collapsed-pair branch.** When
       ``b >= 1 - k`` the contact clip collapses the inner pair and ``sqrt``
       / ``arcsin`` sit at an infinite derivative; MetalPlanet 0.8.1 guards
       that with a ``stop_gradient`` in ``exposure.contact_offsets``, which
       ``flux_dev_from_tau`` and the frontend share. Remove that guard and
       the three collapsed-pair cases here go NaN (measured). The
       pre-existing awkward-geometry test runs at ``exp_time=0`` and never
       builds a contact, so it cannot see this.
    2. **Gradient accuracy against a converged reference**, on both sides
       of the edge. The tolerances are set from turin's own convergence: the
       frozen-split error falls ~8x per doubling of ``n_gl`` (near the edge,
       d/db: 6.5e-5 at 5, 1.2e-5 at 9, 2.3e-6 at 16, 3.0e-7 at 32), so
       ``5e-4`` at the production ``N_GL`` and ``1e-5`` at 16 leave ~8x and
       ~4x margin. Grazing itself is the *easy* regime (~1e-9): the hard one
       is just inside the edge, where the inner pair survives as two narrow
       sub-intervals. KOI-448.02's turin-defaults posterior straddles it.

    What this does **not** pin, and should not: ``flux_dev_from_tau``'s own
    detachment of the split points (``metal.py``). That is MetalPlanet's
    internal choice, and it is invisible to an accuracy test by
    construction -- splits that move track the kink, so a moving-split
    gradient agrees with the truth *better* at finite ``n_gl`` than the
    frozen one does. A reviewer confirmed removing it passes every case.
    """
    n_gl = M.N_GL if n_gl is None else n_gl
    fd = _grazing_reference_gradient(k, b)
    assert np.all(np.abs(fd) > 1e-6), (k, b, fd)      # reference is non-degenerate

    grid = one_epoch_grid(exp_time=GRAZING_EXP, n_gl=n_gl, dtype=mx.float64)

    def scalar(v):
        col = lambda i: v[i].reshape(1, 1)
        f = M.transit_flux(grid, mid=mx.zeros((1, 1), dtype=mx.float64),
                           k=col(0), b=col(1), T14=col(2), q1=col(3),
                           q2=col(4), period=col(5))
        return mx.sum(f * f)

    v0 = np.array([k, b, T14, Q1, Q2, P_REF], dtype=np.float64)
    with mx.stream(mx.cpu):
        g = np.array(mx.grad(scalar)(mx.array(v0, dtype=mx.float64)),
                     dtype=np.float64)

    assert np.all(np.isfinite(g)), (k, b, n_gl, g)
    assert np.all(np.abs(g[:2]) > 1e-6), (k, b, g)    # not silently zeroed
    rel = np.abs(g[:2] - fd) / np.abs(fd)
    assert np.all(rel < tol), (k, b, n_gl, rel, g[:2], fd)


def _three_epoch_grid(dtype, exp_time=29.4 / 1440):
    """Three epochs with different predicted offsets, so the phase-order
    permutation interleaves them -- a single sorted epoch would leave it the
    identity and test nothing."""
    times = np.stack([TIMES, TIMES[::-1] * 0.9, TIMES * 1.1])
    centering = dict(times_centered=times, n_arr=np.array([-1.0, 0.0, 1.0]),
                     d_arr=np.array([0.012, -0.021, 0.004]),
                     P_ref=P_REF, tau0_ref=0.0)
    return M.build_grid(centering, prep.supersample_offsets(exp_time, 1),
                        dtype=dtype, exp_time=exp_time)


@pytest.mark.parametrize("k,b", [
    (0.11, 0.42),     # full transit
    (0.11, 0.95),     # grazing: b > 1 - k = 0.89, the inner contacts gone
    (0.30, 1.15),     # deeply grazing, b > 1
])
@pytest.mark.parametrize("dtype,tol", [(mx.float64, 4e-11), (mx.float32, 2e-6)])
def test_vertex_flux_devs_are_the_three_vertex_laws(dtype, tol, k, b):
    """``vertex_flux_devs`` (one ld_basis launch, phase-ordered) against
    two independent references.

    Each vertex against MetalPlanet's scalar kernel called at exactly that
    ``(u1, u2)`` on the points in plain *storage* order -- no gathers -- so
    the phase-order round trip is checked too. Not via turin's q route:
    ``q_to_u`` clamps ``q1 >= 1e-12``, so ``q1 = 0`` gives ``u1 = 1e-6``, not
    the ``(0, 0)`` vertex (measured 2.8e-9 off, which is the clamp, not a
    bug). Then the convex combination at a general law against turin's own
    ``transit_flux_dev``, through the q route where no clamp is active.

    Grazing geometries are included because ``--ld=collapsed``'s real-target
    acceptance ran KOI-448.02 under ``--nongrazing``, so nothing else in
    turin exercises the ld_basis route with the inner contacts collapsed.
    """
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    grid = _three_epoch_grid(dtype)
    assert grid.order is not None and not np.array_equal(
        np.array(grid.order), np.arange(grid.order.size))
    col = lambda v: mx.array([[float(v)]], dtype=dtype)
    mid = mx.array(np.array([[0.002, -0.001, 0.0005]]), dtype=dtype)
    geo = dict(k=col(k), b=col(b), T14=col(T14), period=col(P_REF))
    with mx.stream(stream):
        verts = [np.array(f, dtype=np.float64)
                 for f in M.vertex_flux_devs(grid, mid=mid, **geo)]

        def single(q1, q2):
            return np.array(M.transit_flux_dev(grid, mid=mid, q1=col(q1),
                                               q2=col(q2), **geo),
                            dtype=np.float64)

        tau = M.time_from_mid(grid, mid).reshape(1, -1)      # storage order
        a = M.a_over_rstar(geo["T14"], geo["period"], geo["k"], geo["b"])

        def at_u(u1, u2):
            dev = metalplanet.flux_dev_from_tau(
                tau, geo["period"], a, geo["b"], geo["k"], col(u1), col(u2),
                exp_time=grid.exp_time, integration="contact", n_gl=grid.n_gl)
            return np.array(dev, dtype=np.float64).reshape(verts[0].shape)

        for got, (u1, u2) in zip(verts, ((0.0, 0.0), (2.0, -1.0), (0.0, 1.0))):
            assert np.max(np.abs(got - at_u(u1, u2))) <= tol

        # a general law is the convex combination with omega_j proportional
        # to lambda_j * F*_j, lambda = (1 - sqrt q1, sqrt q1 q2, sqrt q1 (1-q2))
        q1, q2 = 0.37, 0.61
        r = math.sqrt(q1)
        lam = np.array([1 - r, r * q2, r * (1 - q2)])
        om = lam * np.array(M.VERTEX_FSTAR)
        om /= om.sum()
        mix = sum(w * f for w, f in zip(om, verts))
        assert np.max(np.abs(mix - single(q1, q2))) <= tol


def test_gradients_are_finite_and_nonzero_in_fp32_at_awkward_geometry():
    """Grazing, near-zero impact parameter and tiny q1 must not produce NaN."""
    for beta, k, q1 in ((0.999, 0.3, 1e-10), (1e-6, 0.08, 0.5),
                        (0.5, 0.5, 1.0)):
        def scalar(v):
            grid = one_epoch_grid(times=TIMES.astype(np.float32),
                                  dtype=mx.float32)
            col = lambda i: v[i].reshape(1, 1)
            b = M.impact_parameter(col(3), col(2), "transiting")
            mid = M.mid_times_lineph(grid, col(0), col(1))
            f = M.transit_flux(grid, mid=mid, k=col(2), b=b, T14=col(4),
                               q1=col(5), q2=col(6),
                               period=grid.P_ref + col(0))
            return mx.sum(f * f)

        v0 = np.array([0.0, 0.0, k, beta, T14, q1, 0.25], dtype=np.float32)
        val, g = mx.value_and_grad(scalar)(mx.array(v0, dtype=mx.float32))
        g = np.array(g)
        assert np.isfinite(float(val)), (beta, k, q1)
        assert np.all(np.isfinite(g)), (beta, k, q1, g)


def test_batching_over_chains_is_independent():
    """Row i of a batched call must equal a single-chain call at row i."""
    rng = np.random.default_rng(7)
    n_chains = 5
    grid = one_epoch_grid(times=TIMES.astype(np.float32), dtype=mx.float32)
    ks = rng.uniform(0.05, 0.12, n_chains)
    bs = rng.uniform(0.0, 0.6, n_chains)
    batched = np.array(M.transit_flux(
        grid,
        mid=mx.zeros((n_chains, 1), dtype=mx.float32),
        k=mx.array(ks.reshape(-1, 1), dtype=mx.float32),
        b=mx.array(bs.reshape(-1, 1), dtype=mx.float32),
        T14=mx.full((n_chains, 1), T14, dtype=mx.float32),
        q1=mx.full((n_chains, 1), Q1, dtype=mx.float32),
        q2=mx.full((n_chains, 1), Q2, dtype=mx.float32),
        period=mx.full((n_chains, 1), P_REF, dtype=mx.float32),
    ), dtype=np.float64)
    for i in range(n_chains):
        single = turin_flux(grid=grid, k=ks[i], b=bs[i])
        np.testing.assert_allclose(batched[i, 0], single, atol=1e-7)


def test_ttv_mid_times_shift_each_epoch_independently():
    """TTV mode: epoch i's transit moves with dtau_i and nothing else."""
    times = np.stack([TIMES, TIMES, TIMES])
    centering = dict(times_centered=times, n_arr=np.array([-3.0, 0.0, 4.0]),
                     d_arr=np.zeros(3), P_ref=P_REF, tau0_ref=0.0)
    grid = M.build_grid(centering, prep.supersample_offsets(0.0, 1),
                        dtype=mx.float64)
    shift = 0.01
    dtau = mx.array([[0.0, shift, 0.0]], dtype=mx.float64)
    with mx.stream(mx.cpu):
        mid = M.mid_times_ttv(grid, dtau)
        np.testing.assert_allclose(np.array(mid)[0], [0.0, shift, 0.0])
        col = lambda v: mx.array([[float(v)]], dtype=mx.float64)
        f = np.array(M.transit_flux(
            grid, mid=mid, k=col(K), b=col(B), T14=col(T14), q1=col(Q1),
            q2=col(Q2), period=col(P_REF)), dtype=np.float64)[0]
    # epochs 0 and 2 are untouched and identical; epoch 1 is displaced
    np.testing.assert_allclose(f[0], f[2], atol=1e-14)
    assert np.abs(f[1] - f[0]).max() > 1e-4
    # and the displacement is exactly a translation of the same curve
    with mx.stream(mx.cpu):
        ref = turin_flux(grid=one_epoch_grid(), mid=shift)
    np.testing.assert_allclose(f[1], ref, atol=1e-14)


def test_log10_rho_round_trips_through_a_over_rstar():
    """The reported density must come from the geometry actually fitted."""
    with mx.stream(mx.cpu):
        aRs = float(M.a_over_rstar(*[mx.array([[v]], dtype=mx.float64)
                                     for v in (T14, P_REF, K, B)])[0, 0])
    G_SI, P_sec = 6.67430e-11, P_REF * 86400.0
    expected = np.log10((3.0 * np.pi / (G_SI * P_sec**2)) * aRs**3)
    got = M.log10_rho_from_T14(T14, P_REF, K, B)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # a sun-like density is order 3.1 in log10 kg/m^3
    assert 2.0 < got < 4.5


def _hurin_flux(q1, q2):
    """Evaluate hurin's own transit model in the hurin environment."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "hf.npy")
        code = (
            "import numpy as np, sys\n"
            f"sys.path.insert(0, {_HURIN_DIR!r})\n"
            "import hurin.transit_fit as tf\n"
            "import hurin\n"
            f"t = np.linspace(-0.25, 0.25, 601)\n"
            f"np.save({out!r}, np.asarray(tf.transit_model("
            f"t, {P_REF}, 0.0, {B}, {K}, {T14}, {q1}, {q2}),"
            " dtype=np.float64))\n"
            "print(hurin.__version__)\n"
        )
        r = subprocess.run([_HURIN_PY, "-c", code], capture_output=True,
                           text=True, timeout=900)
        if r.returncode != 0:
            pytest.skip(f"hurin model call failed: {r.stderr[-400:]}")
        return np.load(out), r.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(not os.path.exists(_HURIN_PY),
                    reason="hurin conda env not installed")
def test_chord_geometry_reproduces_hurin_jaxoplanet():
    """turin must be able to reproduce hurin's model, for parity validation.

    Both packages now use the same limb darkening (hurin adopted the correct
    Kipping map in 0.1.68), so ``geometry="chord"`` is the whole of parity.
    hurin runs jaxoplanet in float32 -- x64 is never enabled -- so ~2e-7 is
    its own floor and the target tolerance.
    """
    q1_h, q2_h = 0.30, 0.45
    hurin_f, version = _hurin_flux(q1_h, q2_h)
    if tuple(int(p) for p in version.split(".")[:3]) < (0, 1, 68):
        pytest.skip(f"hurin {version} predates the limb-darkening fix; "
                    "turin no longer carries the old map")

    with mx.stream(mx.cpu):
        got = turin_flux(grid=one_epoch_grid(), q1=q1_h, q2=q2_h,
                         geometry="chord")
        circ = turin_flux(grid=one_epoch_grid(), q1=q1_h, q2=q2_h,
                          geometry="circular")
    assert np.abs(got - hurin_f).max() < 1e-6, (
        f"parity mode must match hurin {version}")
    # and turin's default still differs, now purely by the chord
    # approximation: O((T14/P)^2), well above the float32 floor
    d = np.abs(circ - hurin_f).max()
    assert 1e-6 < d < 1e-4, d
