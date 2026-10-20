"""``--mle``'s TTV search: the exact timing step.

Reference: each epoch's float64 term on a dense grid over its whole timing
box, with shape and limb darkening held -- the per-epoch maximum that the
step claims to reach, found without its grid, its zoom or its float32
screen. The data put a second, identical dip 0.07 d from every transit
(``test_seeding.build_dataset``'s decoy), so an epoch has two near-equal
peaks and a local search from the predicted time would stop at whichever is
nearer. ``fit_ttv`` itself is exercised end to end, with injected TTVs, in
``test_pipeline.py``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from turin import likelihood, mle, params
from tests.test_seeding import EPH, K_T, B_T, T14_T, build_dataset

TAU_HALF = 0.12
SHAPE = np.array([K_T, B_T / (1 + K_T), T14_T, 0.30, 0.225])


def _targets(ed, cen, n_ep):
    lay = params.ttv_layout(EPH, cen, TAU_HALF)
    _, _, lo, hi = likelihood.build_target(
        lay, cen, ed, np.full(n_ep, 2), num_resample=7,
        exposure_time=29.4 / 1440, n_chains_hint=512, likelihood_only=True)
    return lay, lo, hi


def _terms64(hi, V):
    with mx.stream(mx.cpu):
        return np.array(hi.epoch_log_lik(mx.array(V, dtype=mx.float64)),
                        dtype=np.float64)


def _brute_best(hi, lay, v, n=1500):
    """Each epoch's best float64 term over a dense grid of its own box."""
    s, E = lay.n_shape, lay.n_epochs
    grid = np.linspace(lay.lo[s], lay.hi[s], n + 2)[1:-1]
    V = np.tile(v, (n, 1))
    V[:, s:] = grid[:, None]               # every epoch at grid point g
    return _terms64(hi, V).max(axis=0)     # (E,)


@pytest.mark.parametrize("decoy", [None, 0.07])
def test_timing_step_puts_every_epoch_at_its_best_time(decoy):
    _, _, _, n_ep = build_dataset(0.0)
    rng = np.random.default_rng(3)
    true = 0.03 * rng.uniform(-1, 1, n_ep)
    # the decoy case is noisy enough for its two peaks to trade places
    ed, cen, _, _ = build_dataset(true, yerr=2.5e-3 if decoy else 5e-4,
                                  decoy=decoy)
    lay, lo, hi = _targets(ed, cen, n_ep)
    v0 = np.r_[SHAPE, np.zeros(n_ep)]       # the linear-ephemeris times

    step = mle.timing_step(lo, hi, lay, v0)
    got = _terms64(hi, step.v[None])[0]
    start = _terms64(hi, v0[None])[0]
    best = _brute_best(hi, lay, v0)
    # never lower; at least the dense grid's best, everywhere
    assert np.all(got >= start - 1e-9)
    np.testing.assert_array_less(best - 1e-4, got)
    assert np.all(step.gain >= 0) and not step.flat.any()
    np.testing.assert_allclose(step.gain, np.where(step.moved, got - start, 0),
                               atol=1e-8)
    # nothing left to gain: a second step is the certificate
    again = mle.timing_step(lo, hi, lay, step.v)
    assert not again.moved.any()
    if decoy is None:                       # one dip: the injected time
        np.testing.assert_array_less(np.abs(step.v[lay.n_shape:] - true),
                                     4.0 / 1440)


def test_timing_step_leaves_shape_and_flat_epochs_alone():
    """Only the times move; an epoch with no transit data (its term flat
    over the box) is flagged and keeps its time."""
    _, _, _, n_ep = build_dataset(0.0)
    ed, cen, _, _ = build_dataset(0.01, yerr=2.5e-3)
    ed = dict(ed)
    ed["ferr_padded"] = np.array(ed["ferr_padded"], dtype=np.float64)
    ed["ferr_padded"][0] = 1e3              # epoch 0 carries no information
    lay, lo, hi = _targets(ed, cen, n_ep)
    v0 = np.r_[SHAPE, np.zeros(n_ep)]
    step = mle.timing_step(lo, hi, lay, v0)
    np.testing.assert_array_equal(step.v[:lay.n_shape], SHAPE)
    assert step.flat[0] and not step.moved[0] and step.v[lay.n_shape] == 0.0
    assert step.moved[1:].all()
