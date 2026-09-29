"""Corner, phase-fold and O-C figures, in hurin's styles.

All matplotlib, all float64, all post-sampling: nothing here is on the hot
path, so the profile coefficients are recomputed on the CPU at full precision
rather than being carried through the sampler (hurin's lesson -- recording an
``(n_epochs, order+1)`` array per draw dominated both memory and
post-processing on short-period targets).
"""

from __future__ import annotations

import numpy as np

from . import __version__
from . import outputs as _outputs

#: hurin's per-mode colours, kept so figures from the two packages are
#: visually comparable.
MODE_COLOURS = {"lineph": "#1f77b4", "ttv": "#e67e22"}
#: 1, 1.5 and 2 sigma, as corner wants them (1 - exp(-s^2/2)).
SIGMA_LEVELS = tuple(1.0 - np.exp(-0.5 * s**2) for s in (1.0, 1.5, 2.0))


def _pdf_metadata(title):
    return {"Creator": f"turin {__version__}",
            "Title": title,
            "Subject": _outputs.launch_command()}


def _save(fig, path, title, log=None):
    fig.savefig(path, bbox_inches="tight", metadata=_pdf_metadata(title))
    import matplotlib.pyplot as plt

    plt.close(fig)
    if log:
        import os

        log(f"    wrote {os.path.basename(path)}")
    return path


def corner_plot(path, draws, labels, *, title="", colour="#808080", log=None):
    """Corner plot with 1/1.5/2-sigma contours, as hurin draws it."""
    import corner
    import matplotlib

    matplotlib.use("Agg", force=False)
    draws = np.asarray(draws, dtype=np.float64)
    # drop parameters with no spread: corner cannot bin them and they carry
    # no information (a fully profiled or prior-pinned quantity)
    keep = [i for i in range(draws.shape[1]) if np.std(draws[:, i]) > 0]
    if len(keep) < 2:
        return None
    fig = corner.corner(
        draws[:, keep], labels=[labels[i] for i in keep],
        levels=SIGMA_LEVELS, quantiles=[0.16, 0.5, 0.84],
        show_titles=True, title_fmt=".4g", color=colour,
        range=[0.999] * len(keep),
        plot_datapoints=False, fill_contours=True,
    )
    if title:
        fig.suptitle(title, y=1.01, fontsize=11)
    return _save(fig, path, title or "corner", log)


def fold_plot(path, *, epoch_data, mid_times, baseline, model_grid_t,
              model_grid_f, period, title="", colour="#1f77b4",
              n_bins_from=None, log=None):
    """Phase-folded transit with the baseline divided out.

    Each epoch is folded on *its own* mid-transit time, which is what makes
    the same function serve both fit modes: a LinEph fit simply has mid-times
    that lie on a straight line.

    ``baseline`` is the per-point ``1 + L c`` at the maximum-likelihood
    coefficients; dividing by it is what "detrended" means here.
    """
    import matplotlib.pyplot as plt

    mask = np.asarray(epoch_data["mask"]) > 0
    t = np.asarray(epoch_data["times_padded"], dtype=np.float64)
    f = np.asarray(epoch_data["flux_padded"], dtype=np.float64)
    e = np.asarray(epoch_data["ferr_padded"], dtype=np.float64)
    g = np.asarray(baseline, dtype=np.float64)
    mid = np.asarray(mid_times, dtype=np.float64)

    dt = (t - mid[:, None])[mask]
    y = (f / np.where(g != 0, g, 1.0))[mask]
    ye = (e / np.where(g != 0, g, 1.0))[mask]
    order = np.argsort(dt)
    dt, y, ye = dt[order], y[order], ye[order]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(dt * 24.0, y, ".", ms=2.2, color="0.75", zorder=1,
            label="detrended data")

    n_bin = int(n_bins_from or max(1, mask.shape[0]))
    if n_bin > 1 and dt.size > n_bin:
        nb = dt.size // n_bin
        bt = np.array([dt[i * n_bin:(i + 1) * n_bin].mean() for i in range(nb)])
        by = np.array([y[i * n_bin:(i + 1) * n_bin].mean() for i in range(nb)])
        bs = np.array([y[i * n_bin:(i + 1) * n_bin].std()
                       / max(1, np.sqrt(n_bin)) for i in range(nb)])
        ax.errorbar(bt * 24.0, by, yerr=bs, fmt="o", ms=3.4, color="k",
                    lw=0.9, capsize=0, zorder=3, label=f"binned x{n_bin}")

    ax.plot(np.asarray(model_grid_t) * 24.0, np.asarray(model_grid_f), "-",
            color=colour, lw=1.6, zorder=4, label="best fit")

    resid = y - np.interp(dt, model_grid_t, model_grid_f)
    rms_ppm = float(np.std(resid) * 1e6)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("relative flux")
    ax.set_title(f"{title}   (residual RMS {rms_ppm:.0f} ppm)" if title
                 else f"residual RMS {rms_ppm:.0f} ppm", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    lo, hi = np.percentile(y, [0.5, 99.5])
    pad = 2.0 * np.std(resid)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, path, title or "fold", log)


def oc_plot(path, epochs, tmid, tmid_err, *, title="", colour="#e67e22",
            log=None):
    """Observed-minus-calculated diagram against a refitted linear ephemeris.

    The reference is a robust fit to the measured times, not the archive
    ephemeris: against the archive the diagram shows the archive's error.
    """
    import matplotlib.pyplot as plt

    epochs = np.asarray(epochs, dtype=np.float64)
    tmid = np.asarray(tmid, dtype=np.float64)
    err = np.asarray(tmid_err, dtype=np.float64)
    P, tau0, kept = _outputs.fit_linear_ephemeris(epochs, tmid)
    oc_min = (tmid - (tau0 + P * epochs)) * 24.0 * 60.0

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
    ax.errorbar(epochs[kept], oc_min[kept], yerr=err[kept] * 1440.0, fmt="o",
                ms=4, color=colour, lw=1.0, capsize=0, zorder=3)
    if (~kept).any():
        ax.errorbar(epochs[~kept], oc_min[~kept], yerr=err[~kept] * 1440.0,
                    fmt="o", ms=4, mfc="none", color="0.5", lw=1.0,
                    capsize=0, zorder=2, label="clipped from the fit")
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_xlabel("epoch")
    ax.set_ylabel("O - C (minutes)")
    ax.set_title(title or f"P = {P:.6f} d, tau0 = {tau0:.5f}", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, path, title or "O-C", log)


def model_grid(lp, v_row, *, n=1000, span_durations=3.0, T14=None):
    """A densely sampled transit model for overlaying on a fold plot.

    Evaluates the model on its own fine time grid at one parameter vector,
    in float64 on the CPU, with a single synthetic epoch.
    """
    import mlx.core as mx

    from . import model as _model
    from . import prep as _prep

    T14 = float(T14 if T14 is not None else v_row[lp.layout.index("T14")])
    tt = np.linspace(-span_durations * T14, span_durations * T14, n)
    centering = dict(times_centered=tt[None, :], n_arr=np.zeros(1),
                     d_arr=np.zeros(1), P_ref=lp.layout.P_ref,
                     tau0_ref=lp.layout.tau0_ref)
    with mx.stream(mx.cpu):
        grid = _model.build_grid(
            centering,
            _prep.supersample_offsets(lp.grid.sub_offsets.shape[0] > 1
                                      and float(np.ptp(np.array(
                                          lp.grid.sub_offsets))) or 0.0,
                                      lp.grid.n_sub),
            dtype=mx.float64)
        p = lp.unpack(mx.array(np.asarray(v_row, dtype=np.float64)[None, :],
                               dtype=mx.float64))
        f = _model.transit_flux(
            grid, mid=mx.zeros((1, 1), dtype=mx.float64), k=p["k"], b=p["b"],
            T14=p["T14"], q1=p["q1"], q2=p["q2"], period=p["period"],
            geometry=lp.geometry)
        return tt, np.array(f, dtype=np.float64)[0, 0]
