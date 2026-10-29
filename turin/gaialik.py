"""The log-density of a Gaia fit: bands jointly, one baseline per band.

Gaia's epoch photometry is a few dozen points per band (G, BP, RP) over
years, often one per transit. turin's per-epoch Legendre baseline would
absorb a one-point epoch whole, so here each band has **one** baseline,
``1 + L c`` in a coordinate across the whole light curve
(:func:`turin.prep.gaia_basis_x`), profiled exactly as always (flux space,
no Occam factor) over all of that band's rows
(:func:`turin.profile.solve_shared_coefficients`). Rows are (transit, band)
pairs, so the per-epoch tau machinery is reused unchanged.

Each band has its own limb darkening (a different passband) and its own
**error scale** ``s_b = exp(lns_b)``: Gaia's catalogue errors run small and
by a band-dependent factor (MetalLS: robust chi median 2.85), and taking them
as exact would make every width too narrow. With ``r = (y - model) /
(s sigma)`` the band's log-likelihood is ``sum 0.5 (1 - r)(1 + r) - N_b ln
s_b`` plus the float64 constant (``-N/2 - sum ln sigma - N/2 ln 2 pi``), the
same recentred form as :func:`turin.profile.chi2_terms`. The baseline solve
does not depend on ``s_b``: a band's weights all scale by ``1/s_b^2``, and a
band's coefficients are its own. A profiled baseline biases ``s_b`` low by
about K/N (a few per cent at K <= 3 and ~50 points), turin's usual
profile-likelihood convention.

One epoch block always (a Gaia star is ~150 points).

MLX's float64 ``exp`` is accurate only to ~4e-7 relative (measured on the
CPU stream; its ``log`` is exact), which would put a float32-sized error
into the float64 reference wherever ``s_b`` enters, so float64 takes
:func:`_exp` instead.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import model as _model
from . import params as _params
from . import profile as _profile
from .likelihood import ProfiledTransitLogProb, _prep_offsets


def _exp(x):
    """``exp(x)``, accurate in float64 too (MLX's float64 ``exp`` errs by
    ~4e-7 relative): ``exp(x/8)`` by its Taylor series to degree 14, then
    squared three times. For ``|x| <= 4`` (``lns``'s box is ``[-1.2, 3.4]``)
    the series' truncation is below 1e-17 and the squarings multiply the
    rounding by eight. float32 uses ``mx.exp``."""
    if x.dtype != mx.float64:
        return mx.exp(x)
    y = x / 8.0
    term = mx.ones_like(y)
    acc = mx.ones_like(y)
    for n in range(1, 15):
        term = term * y / n
        acc = acc + term
    for _ in range(3):
        acc = acc * acc
    return acc


class GaiaLogProb(ProfiledTransitLogProb):
    """Model-space log-density of a Gaia layout (:func:`turin.params.gaia_layout`),
    ``(n_chains, dim) -> (n_chains,)``; anvil's signature, as its parent."""

    def __init__(self, layout, centering, epoch_data, orders, *,
                 num_resample=1, exposure_time=0.0, profile_mode="exact",
                 geometry="circular", noise=None, n_chains_hint=512,
                 dtype=mx.float32, prior=True, **_ignored):
        if not layout.bands:
            raise ValueError("GaiaLogProb needs a Gaia layout (gaia_layout)")
        if profile_mode != "exact":
            raise ValueError("a Gaia fit solves each band's baseline exactly "
                             f"(--PL=exact), not {profile_mode!r}")
        if geometry != "circular":
            raise ValueError("a Gaia fit needs geometry='circular'")
        from .data.gaia import BANDS
        from .prep import gaia_basis_x

        self.layout = layout
        self.mode = layout.mode
        self.profile_mode = "exact"
        self.geometry = geometry
        self.noise = None
        self.dtype = dtype
        self.dim = layout.dim
        self.ld = layout.ld
        self.law = layout.law
        self.prior = prior

        codes = np.asarray(epoch_data["band"], dtype=np.int64)
        want = [BANDS.index(b) for b in layout.bands]
        if sorted(set(codes.tolist())) != sorted(want):
            raise ValueError(f"the layout's bands {layout.bands} are not the "
                             "data's")
        # each row's band as an index into layout.bands
        row_band = np.array([want.index(c) for c in codes], dtype=np.int64)
        self.row_band = row_band
        grid = _model.build_grid(
            centering, _prep_offsets(0.0, 1), dtype=dtype,
            exp_time=exposure_time, mask=epoch_data["mask"],
            exp_epoch=epoch_data["exposure"], band_epoch=row_band)
        orders = np.asarray(orders, dtype=int)
        band_orders = np.array([int(orders[row_band == j][0])
                                for j in range(len(want))])
        design = _profile.build_shared_design(
            epoch_data, band_orders, row_band, gaia_basis_x(epoch_data),
            dtype=dtype)
        self.grid = grid
        self.design = design
        self.blocks = [None]
        self.block_size = grid.n_epochs
        self.n_epochs = grid.n_epochs
        self.n_real = design.n_real
        self.log_const = design.log_const
        self.n_sub = 1
        self.max_pts = grid.max_pts
        self._row_band = mx.array(row_band.astype(np.int32))
        self._n_band = mx.array(design.n_real_group.astype(np.float64),
                                dtype=dtype)

    def unpack(self, v):
        lay = self.layout
        col = lambda name: v[:, lay.index(name):lay.index(name) + 1]
        k, beta, T14 = col("k"), col("beta"), col("T14")
        dP, dtau0 = col("dP"), col("dtau0")
        q_bands = tuple(
            tuple(col(_params.gaia_ld_name(n, band))
                  for n in _model.ld_names(lay.law))
            for band in lay.bands)
        lns = mx.concatenate([col(f"lns_{band}") for band in lay.bands],
                             axis=1)
        b = _model.impact_parameter(beta, k, lay.b_prior)
        return dict(dP=dP, dtau0=dtau0, dtau=None, k=k, beta=beta, b=b,
                    T14=T14, q_bands=q_bands, lns=lns, period=lay.P_ref + dP)

    def mid_times(self, grid, p, lo=0, hi=None):
        return _model.mid_times_lineph(grid, p["dP"], p["dtau0"])

    def log_prior(self, p):
        if not self.prior:
            return mx.zeros(p["k"].shape[:1], dtype=p["k"].dtype)
        term = _params.bk_log_prior(p["beta"], p["k"], self.layout.b_prior)
        if self.layout.rhostar:
            term = term + _params.rhostar_log_prior(
                p["T14"], p["period"], p["k"], p["b"], self.layout.rhostar)
        return term

    def flux_dev(self, p):
        return _model.transit_flux_dev_bands(
            self.grid, mid=self.mid_times(self.grid, p), k=p["k"], b=p["b"],
            T14=p["T14"], period=p["period"], q_bands=p["q_bands"],
            law=self.law)

    def _terms(self, p):
        """Per-row log-likelihood terms (recentred, error scales included),
        ``(n_chains, n_rows)``, and the coefficients."""
        f_dev = self.flux_dev(p)
        c = _profile.solve_shared_coefficients(self.design, f_dev)
        resid = _profile.residual_dev(self.design, f_dev, c)
        inv_s = _exp(-mx.take(p["lns"], self._row_band, axis=1))  # (C, E)
        r = resid * self.design.inv_sigma * inv_s[:, :, None]
        rows = mx.sum(self.design.mask * 0.5 * (1.0 - r) * (1.0 + r), axis=-1)
        return rows, c

    def __call__(self, v):
        p = self.unpack(v)
        rows, _ = self._terms(p)
        # -N_b ln s_b, the error scales' normalisation
        norm = mx.sum(p["lns"] * self._n_band[None, :], axis=1)
        return mx.sum(rows, axis=-1) - norm + self.log_prior(p)

    def epoch_log_lik(self, v):
        raise ValueError("a Gaia fit's bands share baselines across every "
                         "transit, so its likelihood does not factor by epoch")

    def full_model(self, v, *, ld=None):
        """Detrended model and coefficients over every row: ``(model,
        coeffs)``, ``(n_chains, n_rows, max_pts)`` and ``(n_chains, n_rows,
        n_cols)``. ``ld`` is not taken (each band's is in ``v``)."""
        if ld is not None:
            raise ValueError("a Gaia layout carries every band's limb "
                             "darkening; pass it in v")
        p = self.unpack(v)
        f_dev = self.flux_dev(p)
        c = _profile.solve_shared_coefficients(self.design, f_dev)
        return (1.0 + f_dev) * _profile.baseline(self.design, c), c

    def noise_scales(self, v):
        """Each band's error scale ``s_b``, ``(n_chains, n_bands)`` (host)."""
        return np.exp(np.asarray(self.unpack(v)["lns"], dtype=np.float64))

