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

import math
import os
import subprocess
import tempfile

import mlx.core as mx
import numpy as np
import pytest
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
                   dtype=mx.float64):
    centering = dict(times_centered=np.asarray(times)[None, :],
                     n_arr=np.array([float(n)]), d_arr=np.array([float(d)]),
                     P_ref=P_REF, tau0_ref=0.0)
    return M.build_grid(centering, prep.supersample_offsets(exp_time, n_sub),
                        dtype=dtype)


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


@pytest.mark.parametrize("n_sub,exp_time", [(7, 29.4 / 1440), (15, 29.4 / 1440)])
def test_exposure_integration_matches_metalplanet_supersampling(n_sub, exp_time):
    ref, _ = metalplanet_reference(n_sub=n_sub, exp_time=exp_time)
    with mx.stream(mx.cpu):
        got = turin_flux(grid=one_epoch_grid(n_sub=n_sub, exp_time=exp_time))
        inst = turin_flux(grid=one_epoch_grid())
    assert np.abs(got - ref).max() < 1e-14
    # smearing must actually do something at Kepler long cadence
    assert (1 - inst.min()) - (1 - got.min()) > 1e-6


def test_supersample_offsets_average_to_the_instantaneous_model():
    """n_sub=1 and exp_time=0 must be the same graph, not merely close."""
    with mx.stream(mx.cpu):
        a = turin_flux(grid=one_epoch_grid(n_sub=1, exp_time=0.0))
        b = turin_flux(grid=one_epoch_grid(n_sub=1, exp_time=29.4 / 1440))
    np.testing.assert_array_equal(a, b)


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
