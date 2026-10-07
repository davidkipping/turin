"""Host-side data conditioning: windows, epochs, float64 centering, Legendre CV.

Everything here is pure NumPy and runs once per fit, before anything touches
the GPU. That split is deliberate: the GPU is float32, whose resolution at
absolute BKJD/BTJD magnitudes (~1500 d) is 5-10 s, so every
large-minus-large subtraction has to happen here, in float64
(``centering_constants``). What reaches MLX is per-epoch time residuals and
exact small integers.

Ported from hurin 0.1.67: ``extract_near_transit_data``,
``segment_epochs``, ``_numpy_legendre_matrix``, ``_centering_constants`` and
``optimize_legendre_orders`` from ``hurin/transit_fit.py``; ``prepare_data``
from ``hurin/batch.py``. The numerical functions keep hurin's signatures and
semantics so the ports can be tested against it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.ephemeris import get_ephemeris, get_other_planet_ephemerides
from .data.lightcurve import get_lightcurve
from .data.preprocessing import sigma_clip

#: Default window half-width, in transit durations, around each predicted
#: transit centre (hurin's n_durations).
N_DURATIONS = 5.0
#: Highest Legendre order the cross-validation considers.
MAX_ORDER = 5


def _predicted_transit_times(time, period, epoch):
    """Predicted transit times spanning the data, inclusive of both ends."""
    t_min, t_max = np.nanmin(time), np.nanmax(time)
    n_lo = int(np.ceil((t_min - epoch) / period))
    n_hi = int(np.floor((t_max - epoch) / period))
    return epoch + np.arange(n_lo, n_hi + 1) * period


def extract_near_transit_data(time, flux, flux_err, period, epoch,
                              duration_hours, n_durations=N_DURATIONS):
    """Clip data to within ``n_durations`` transit durations of a transit.

    Discards out-of-transit baseline far from any transit event, which is
    what keeps short-cadence data tractable.
    """
    half_window = n_durations * duration_hours / 24.0
    transit_times = _predicted_transit_times(time, period, epoch)

    keep = np.zeros(len(time), dtype=bool)
    for tt in transit_times:
        keep |= np.abs(time - tt) <= half_window
    return time[keep], flux[keep], flux_err[keep]


#: An epoch is fitted when at least this many points fall inside its
#: feasible transit zone (``|t - tc| <= T14/2 + tau_shift_max``). An epoch
#: with only a point or two in transit is kept on purpose: its timing
#: posterior comes back close to its prior, which grid-Gibbs samples
#: correctly, rather than the epoch silently vanishing.
MIN_IN_TRANSIT = 1


def window_durations(eph, ttv_max_days=0.0, log=None):
    """Half-width of each epoch's data window, in transit durations.

    N_DURATIONS normally; wider only when a declared TTV amplitude needs it,
    so --TTVmax <= 3.5 T14 gives the same windows as a default run. Refuses a
    TTV amplitude of half the period or more, where epochs would overlap.
    """
    if not ttv_max_days:
        return N_DURATIONS
    period, T14 = eph["period"], eph["duration"] / 24.0
    if ttv_max_days >= 0.5 * period:
        raise SystemExit(
            f"turin: --TTVmax={ttv_max_days * 1440:.0f} min is at least half "
            f"the period ({period:.4f} d); the epochs would overlap")
    n = max(N_DURATIONS, ttv_max_days / T14 + 1.5)
    if log:
        if n * T14 > 0.45 * period:
            log(f"  warning: windows of {n:.1f} durations span more than 45% "
                "of the period; epochs may be poorly separated")
        log(f"  --TTVmax widened the windows to {n:.2f} durations")
    return n


def segment_epochs(time, flux, flux_err, period, epoch, duration_hours,
                   n_durations=N_DURATIONS, min_pts=4, tau_shift_max=0.0):
    """Segment windowed data into per-epoch padded 2D arrays.

    Each point is assigned to its *nearest* predicted transit centre, which
    avoids double-counting when windows overlap for short-period planets.
    An epoch is kept only if it has at least ``min_pts`` points in the wide
    window and at least one point inside the feasible transit zone,
    ``|t - tc| <= 0.5*T14 + tau_shift_max`` — i.e. it actually covers a
    transit. ``tau_shift_max`` (days) widens that test for declared-TTV
    systems (--TTVmax), where the transit may sit away from the prediction.

    Padded slots carry neutral values: flux 1.0, error 1e10, time at the
    epoch centre, mask 0.

    Returns a dict with ``times_padded``, ``flux_padded``, ``ferr_padded``,
    ``mask`` (all ``(n_epochs, max_pts)``), ``epoch_centers``,
    ``half_window``, ``n_epochs``, ``max_pts`` and ``n_in_transit`` (points
    in each kept epoch's transit zone).
    """
    dur_days = duration_hours / 24.0
    half_window = n_durations * dur_days
    time = np.asarray(time, dtype=np.float64)
    if time.size == 0:
        raise ValueError("No epochs with sufficient data points")

    # Each point's nearest predicted transit, by number, in closed form (the
    # centres are epoch + n*P): O(N), with ceil(x - 1/2) breaking an exact tie
    # toward the earlier centre. The transit list spans every number that
    # occurs, so every point's own nearest transit is in it. It used to be
    # the transits whose *centres* fall inside the data's span, with points
    # clipped to the nearest listed one: a window that starts just after an
    # uncovered transit then joined the previous epoch, a whole period away
    # (KOI-5749.01: 22 points 282 d from their epoch's centre, at Legendre
    # x ~ -490, in the fit). hurin's segment_epochs does the same.
    n_pt = np.ceil((time - epoch) / period - 0.5).astype(np.int64)
    n_lo, n_hi = int(n_pt.min()), int(n_pt.max())
    transit_times = epoch + np.arange(n_lo, n_hi + 1) * period
    nearest = n_pt - n_lo

    half_transit = 0.5 * dur_days + tau_shift_max
    epoch_groups = []
    epoch_centers = []
    n_in_transit = []
    for i, tt in enumerate(transit_times):
        # nearest centre AND inside its window: a point can only belong to
        # an epoch it is near, whatever produced the input
        mask_i = (nearest == i) & (np.abs(time - tt) <= half_window)
        if np.sum(mask_i) < min_pts:
            continue
        t_epoch = time[mask_i]
        n_in = int(np.sum(np.abs(t_epoch - tt) <= half_transit))
        if n_in < MIN_IN_TRANSIT:
            continue
        epoch_groups.append((t_epoch, flux[mask_i], flux_err[mask_i]))
        epoch_centers.append(tt)
        n_in_transit.append(n_in)

    n_epochs = len(epoch_groups)
    if n_epochs == 0:
        raise ValueError("No epochs with sufficient data points")

    max_pts = max(len(g[0]) for g in epoch_groups)
    epoch_centers = np.array(epoch_centers, dtype=np.float64)

    times_padded = np.zeros((n_epochs, max_pts))
    flux_padded = np.ones((n_epochs, max_pts))
    ferr_padded = np.full((n_epochs, max_pts), 1e10)
    mask = np.zeros((n_epochs, max_pts))

    for i, (t_i, f_i, e_i) in enumerate(epoch_groups):
        n_i = len(t_i)
        times_padded[i, :n_i] = t_i
        flux_padded[i, :n_i] = f_i
        ferr_padded[i, :n_i] = e_i
        mask[i, :n_i] = 1.0
        times_padded[i, n_i:] = epoch_centers[i]

    return {
        "times_padded": times_padded,
        "flux_padded": flux_padded,
        "ferr_padded": ferr_padded,
        "mask": mask,
        "epoch_centers": epoch_centers,
        "half_window": half_window,
        "n_epochs": n_epochs,
        "max_pts": max_pts,
        "n_in_transit": np.array(n_in_transit, dtype=int),
    }


def legendre_matrix(x, order):
    """Legendre design matrix at ``x`` in [-1, 1], columns P0..P_order.

    Bonnet recurrence: P_n = ((2n-1) x P_{n-1} - (n-1) P_{n-2}) / n.
    """
    cols = [np.ones_like(x)]
    if order >= 1:
        cols.append(x)
    for n in range(2, order + 1):
        cols.append(((2 * n - 1) * x * cols[-1] - (n - 1) * cols[-2]) / n)
    return np.column_stack(cols)


def centering_constants(epoch_data, eph):
    """Precompute, in float64, the epoch-centred time representation.

    The GPU is float32: at BKJD/BTJD magnitudes its resolution is 5-10 s,
    and MetalPlanet measures 1.7e-4 in flux after 1,000 orbits without
    centring. Every large-minus-large subtraction therefore happens here,
    so only O(1) values enter the float32 graph.

    Returns ``times_centered`` (times minus their epoch centre),
    ``centers_abs``, ``n_arr`` (integer epoch number against the input
    ephemeris), ``d_arr`` (``tau0_ref + n*P_ref - center``, order 0), and
    the reference ``P_ref``/``tau0_ref``.
    """
    centers = np.asarray(epoch_data["epoch_centers"], dtype=np.float64)
    P_ref = float(eph["period"])
    tau0_ref = float(eph["epoch"])
    n_arr = np.round((centers - tau0_ref) / P_ref)
    d_arr = tau0_ref + n_arr * P_ref - centers
    times_centered = (np.asarray(epoch_data["times_padded"], dtype=np.float64)
                      - centers[:, None])
    return {
        "times_centered": times_centered,
        "centers_abs": centers,
        "n_arr": n_arr,
        "d_arr": d_arr,
        "P_ref": P_ref,
        "tau0_ref": tau0_ref,
    }


def optimize_legendre_orders(time, flux, flux_err, period, epoch,
                             duration_hours, n_durations=N_DURATIONS,
                             max_order=MAX_ORDER, n_folds=10, log=None,
                             tau_shift_max=0.0):
    """Select each epoch's Legendre order by cross-validation on OOT data.

    Note this is cross-validation, not BIC. Per epoch: the out-of-transit
    points (``|t - tc| > T14 + tau_shift_max`` — the full duration, wider
    than the half-duration used for epoch inclusion) are split into
    ``n_folds`` *contiguous* segments; for each order K the held-out
    log-likelihood is accumulated over folds, and the order maximising it
    relative to K=0 wins. Epochs with fewer than ``2*n_folds`` OOT points
    take order 0.

    Returns ``orders``, ``scores`` (delta-logL per epoch per order),
    ``total_scores`` (median over epochs) and ``epoch_counts``.
    """
    if log is None:
        def log(msg):
            pass

    log("Windowing near-transit data...")
    tw, fw, ew = extract_near_transit_data(
        time, flux, flux_err, period, epoch, duration_hours,
        n_durations=n_durations,
    )

    log("Segmenting into epochs...")
    epoch_data = segment_epochs(
        tw, fw, ew, period, epoch, duration_hours,
        n_durations=n_durations, tau_shift_max=tau_shift_max,
    )
    return cv_orders(epoch_data, duration_hours, max_order=max_order,
                     n_folds=n_folds, log=log, tau_shift_max=tau_shift_max)


def cv_orders(epoch_data, duration_hours, *, max_order=MAX_ORDER, n_folds=10,
              log=None, tau_shift_max=0.0):
    """The CV of :func:`optimize_legendre_orders` on an existing
    segmentation, so the orders line up with exactly the epochs fitted."""
    if log is None:
        def log(msg):
            pass

    n_epochs = epoch_data["n_epochs"]
    T14_days = duration_hours / 24.0

    orders = np.zeros(n_epochs, dtype=int)
    scores = np.zeros((n_epochs, max_order + 1))
    epoch_counts = np.zeros(n_epochs, dtype=int)

    log(f"Running CV on {n_epochs} epochs (K=0..{max_order}, "
        f"{n_folds} folds)...")

    for i in range(n_epochs):
        m = epoch_data["mask"][i].astype(bool)
        t_i = epoch_data["times_padded"][i][m]
        f_i = epoch_data["flux_padded"][i][m]
        e_i = epoch_data["ferr_padded"][i][m]
        tc = epoch_data["epoch_centers"][i]
        hw = epoch_data["half_window"]
        epoch_counts[i] = len(t_i)

        oot_idx = np.where(np.abs(t_i - tc) > T14_days + tau_shift_max)[0]
        if len(oot_idx) < n_folds * 2:
            orders[i] = 0
            continue

        x_i = (t_i - tc) / hw
        folds = np.array_split(oot_idx, n_folds)

        logL = np.zeros(max_order + 1)
        for K in range(max_order + 1):
            for fold_m in range(n_folds):
                val_idx = folds[fold_m]
                train_idx = np.concatenate(
                    [folds[j] for j in range(n_folds) if j != fold_m])
                if len(train_idx) < K + 1 or len(val_idx) == 0:
                    continue

                L_train = legendre_matrix(x_i[train_idx], K)
                w_train = 1.0 / e_i[train_idx] ** 2
                LtW = L_train.T * w_train[None, :]
                LtWL = LtW @ L_train + 1e-10 * np.eye(K + 1)
                c = np.linalg.solve(LtWL, LtW @ (f_i[train_idx] - 1.0))

                L_val = legendre_matrix(x_i[val_idx], K)
                resid = (f_i[val_idx] - 1.0 - L_val @ c) / e_i[val_idx]
                logL[K] += -0.5 * np.sum(resid ** 2)

        scores[i] = logL - logL[0]
        orders[i] = int(np.argmax(scores[i]))

    dist = ", ".join(f"K={k}: {int(np.sum(orders == k))}"
                     for k in range(max_order + 1) if np.sum(orders == k) > 0)
    log(f"CV complete. Order distribution: {dist}")

    return {
        "orders": orders,
        "scores": scores,
        "total_scores": np.median(scores, axis=0),
        "epoch_counts": epoch_counts,
    }


def supersample_offsets(exposure_time, n_sub):
    """Sub-exposure time offsets, batman's endpoint-inclusive convention.

    ``n_sub=1`` gives a single node at the exposure centre, i.e. no
    integration.
    """
    n_sub = max(1, int(n_sub))
    if n_sub == 1 or not exposure_time:
        return np.zeros(1, dtype=np.float64)
    return np.linspace(-0.5 * exposure_time, 0.5 * exposure_time, n_sub)


def recommended_resample(eph, cadence_days, flux_err):
    """Sub-exposure count from Kipping (2010) Eq. 40, or None.

    N = sqrt(delta * I / (0.8 * tau_ingress * sigma_formal)) with the
    ingress duration from the archive where available, else the b=0
    approximation tau ~ T14 k/(1+k) with k = sqrt(delta).

    Unlike hurin, N is not forced odd: that was a jaxoplanet requirement,
    and turin integrates over its own nodes.
    """
    tau_ingress = eph.get("ingress")
    delta_rel = eph["depth"] / 1e6
    if (tau_ingress is None or tau_ingress <= 0) and delta_rel > 0:
        k = np.sqrt(delta_rel)
        tau_ingress = eph["duration"] * k / (1.0 + k)
    if not tau_ingress or tau_ingress <= 0 or delta_rel <= 0:
        return None

    sigma_formal = float(np.nanmedian(flux_err))
    if not np.isfinite(sigma_formal) or sigma_formal <= 0:
        return None

    I_hours = cadence_days * 24.0
    N_raw = np.sqrt(delta_rel * I_hours / (0.8 * tau_ingress * sigma_formal))
    return max(1, int(np.ceil(N_raw)))


@dataclass
class PreparedData:
    """Cleaned light curve plus the ephemeris the fit will actually use."""

    target: str
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    #: input ephemeris with ``epoch`` recentred on the median occupied epoch
    eph: dict
    #: the archive ephemeris as queried, before recentring
    eph_nea: dict
    cadence_days: float
    exposure_time: float
    num_resample: int
    #: the per-epoch segmentation every downstream step uses (CV, both fit
    #: modes), built once against ``eph``; see :func:`segment_epochs`
    epoch_data: dict
    #: half-width of each epoch window, in transit durations
    n_durations: float
    mission: str = ""
    notes: list = field(default_factory=list)

    @property
    def n_occupied(self):
        return self.epoch_data["n_epochs"]


def prepare_data(target, ttv_max_days=0.0, sc_override=False, log=print):
    """Download, clean and condition one target's light curve.

    Quality filter, sigma clip, other-planet masking, occupied-epoch count,
    and the median-epoch recentring that decorrelates P from tau0 in the
    posterior. Ported from hurin's ``batch.prepare_data``.

    ``ttv_max_days`` widens the occupied-epoch test for declared-TTV
    systems, where transits may sit far from the predicted centres.
    """
    def _log(msg):
        if log:
            log(msg)

    _log(f"[{target}] Downloading light curve...")
    lc = get_lightcurve(target, progress=lambda m: _log(f"  {m}"),
                        sc_override=sc_override)

    _log(f"[{target}] Querying ephemeris...")
    eph = get_ephemeris(target)
    other_ephs = get_other_planet_ephemerides(target)

    time_raw = lc["time"]
    flux_raw = lc["flux"]
    flux_err_raw = lc["flux_err"]
    notes = []

    quality = lc.get("quality")
    if quality is not None:
        good = quality == 0
        _log(f"  Quality filter removed {int(np.sum(~good))} points")
        time_raw, flux_raw, flux_err_raw = (
            time_raw[good], flux_raw[good], flux_err_raw[good])

    cadence = float(np.nanmedian(np.diff(time_raw)))
    time, flux, flux_err, n_outliers, clip_skipped = sigma_clip(
        time_raw, flux_raw, flux_err_raw, eph["duration"], cadence)
    if clip_skipped:
        notes.append(f"sigma clip skipped: {clip_skipped}")
        _log(f"  Sigma clip skipped: {clip_skipped}; "
             f"removed {n_outliers} non-finite points")
    else:
        _log(f"  Sigma clip removed {n_outliers} outliers")

    if other_ephs:
        keep = np.ones(len(time), dtype=bool)
        for oeph in other_ephs:
            half_dur = (oeph["duration"] / 24.0) / 2.0
            for ot in _predicted_transit_times(
                    time, oeph["period"], oeph["epoch"]):
                keep &= np.abs(time - ot) > half_dur
        _log(f"  Masked {int(np.sum(~keep))} points from other planets")
        time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

    # The epochs, segmented once here and carried on PreparedData: the CV
    # and every fit mode use this same epoch_data, so the occupied count,
    # the recentring and the fit cannot disagree. Recentring needs the
    # occupied epochs first, so a pass against the archive epoch finds them;
    # recentring shifts by whole periods, so the final pass keeps the same
    # epochs and only re-anchors their centres.
    n_dur = window_durations(eph, ttv_max_days, log=_log)

    def epochs_for(ephem):
        tw, fw, ew = extract_near_transit_data(
            time, flux, flux_err, ephem["period"], ephem["epoch"],
            ephem["duration"], n_durations=n_dur)
        try:
            return segment_epochs(tw, fw, ew, ephem["period"],
                                  ephem["epoch"], ephem["duration"],
                                  n_durations=n_dur,
                                  tau_shift_max=ttv_max_days)
        except ValueError:
            raise ValueError(
                f"{target}: no occupied transit epochs in the data") from None

    occupied = epochs_for(eph)["epoch_centers"]

    # recentre on the median occupied epoch, snapped to an integer number of
    # periods from the archive epoch: this decorrelates P from tau0
    n_median = round((float(np.median(occupied)) - eph["epoch"]) / eph["period"])
    eph_fit = dict(eph)
    eph_fit["epoch"] = eph["epoch"] + n_median * eph["period"]
    epoch_data = epochs_for(eph_fit)
    assert epoch_data["n_epochs"] == len(occupied)
    _log(f"  Found {epoch_data['n_epochs']} occupied epochs (points in "
         f"transit: {', '.join(map(str, epoch_data['n_in_transit']))})")

    cadence_days = float(lc.get("cadence_days") or cadence)
    n_resam = recommended_resample(eph, cadence_days, flux_err) or 1

    _log(f"  Recentered epoch: {eph_fit['epoch']:.5f} "
         f"(NEA: {eph['epoch']:.5f})")
    _log(f"  Period: {eph['period']:.6f} d, "
         f"Duration: {eph['duration']:.3f} h, cadence {cadence_days * 1440:.1f} min")
    _log(f"  Sub-exposure nodes (fallback route only): {n_resam}")

    return PreparedData(
        target=target,
        time=time,
        flux=flux,
        flux_err=flux_err,
        eph=eph_fit,
        eph_nea=eph,
        cadence_days=cadence_days,
        exposure_time=cadence_days,
        num_resample=n_resam,
        epoch_data=epoch_data,
        n_durations=n_dur,
        mission=str(lc.get("mission", "")),
        notes=notes,
    )
