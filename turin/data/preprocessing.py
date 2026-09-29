"""Ported verbatim from hurin 0.1.67 (hurin/preprocessing.py).

Pure NumPy / lightkurve / requests: no JAX, nothing to rewrite for the GPU.
Keep changes here minimal and upstream-traceable.
"""


import numpy as np
from scipy.ndimage import median_filter


def sigma_clip(time, flux, flux_err, duration_hours, cadence_days):
    """Moving-median 4-sigma outlier filter using MAD-based sigma.

    Window size is 1/3 of the transit duration in cadences, rounded to the
    nearest odd integer and floored at 3. Sigma is estimated as 1.4826 * MAD.

    Non-finite points (NaN time, flux, or flux_err) are always removed — the
    quality flag filter does not catch NaN-flux cadences, and a NaN reaching
    the fit poisons the whole epoch's likelihood. The clip itself is skipped
    when the moving median has memorized the flux — sigma non-finite, zero,
    or far below the point-to-point scatter — because the "4-sigma" cut would
    then remove good data rather than outliers.

    Returns (time, flux, flux_err, n_removed, skip_reason) where
    skip_reason is None when the clip ran, else a short human-readable
    explanation for the frontend to display (truthy, so legacy
    ``if clip_skipped:`` checks keep working).
    """
    finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    n_nonfinite = int(np.sum(~finite))
    time, flux, flux_err = time[finite], flux[finite], flux_err[finite]

    if time.size < 10 or not np.isfinite(cadence_days) or cadence_days <= 0:
        return (time, flux, flux_err, n_nonfinite,
                f"too little data for outlier detection (n={time.size})")

    cadence_hours = cadence_days * 24.0
    window = duration_hours / 3.0 / cadence_hours
    window = int(np.round(window))
    if window % 2 == 0:
        window += 1
    window = max(window, 3)

    med = median_filter(flux, size=window)
    resid = flux - med
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    sigma = 1.4826 * mad

    # Window-independent scatter estimate: a healthy moving-median sigma is
    # comparable to the point-to-point scatter. Sigma far below it means the
    # median memorized the data (short-transit windows on Kepler LC), where
    # the cut would silently delete good cadences instead of outliers.
    sigma_floor = 1.4826 * np.nanmedian(np.abs(np.diff(flux))) / np.sqrt(2.0)
    degenerate = not np.isfinite(sigma) or sigma <= 0
    if np.isfinite(sigma_floor) and sigma_floor > 0:
        degenerate = degenerate or sigma < 0.5 * sigma_floor
    if degenerate:
        return (time, flux, flux_err, n_nonfinite,
                f"transit too short for reliable outlier detection at this "
                f"cadence ({window}-point window)")

    mask = np.abs(resid) < 4.0 * sigma
    n_removed = n_nonfinite + int(np.sum(~mask))
    return time[mask], flux[mask], flux_err[mask], n_removed, None
