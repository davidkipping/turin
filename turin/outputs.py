"""Output products, in hurin's formats, plus resume state.

Filenames and columns deliberately match hurin exactly, so existing analysis
scripts keep working: 7 products for a LinEph fit, 9 for a TTV fit. Every one
carries provenance -- turin's version and the exact command that produced it.

One hurin convention is load-bearing and easy to get wrong: the version
comment goes on **line 2** of a CSV, *after* the column header, because
``np.genfromtxt(names=True)`` treats a leading comment line as the header row.
Likewise the version member of a chains tarball is appended *after* the CSV,
because readers take ``getmembers()[0]``.
"""

from __future__ import annotations

import io
import os
import pickle
import re
import sys
import tarfile
from dataclasses import dataclass, field

import numpy as np

from . import MODEL_REV, __version__
from . import model as _model

#: Filename tag for the current run, set by the CLI (--tag). Namespaces a run
#: into an independent auto-resume lineage.
_RUN_TAG = None


def set_run_tag(tag):
    """Set the filename tag. ``None`` clears it."""
    global _RUN_TAG
    if tag is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError(f"--tag must be [A-Za-z0-9_-]+, got {tag!r}")
    _RUN_TAG = tag
    return _RUN_TAG


def tag_part():
    """``.<tag>`` inserted before every product extension, or ``""``."""
    return f".{_RUN_TAG}" if _RUN_TAG else ""


def launch_command():
    """The command that produced a product, for provenance stamping."""
    return " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])


def provenance():
    """The stamp on line 2 of every CSV: version, model revision, command."""
    return f"# turin {__version__} rev{MODEL_REV} | {launch_command()}"


def product_path(outdir, target, mode, name, ext):
    return os.path.join(outdir,
                        f"{target}_{mode}_{name}{tag_part()}.{ext}")


def _write_csv(path, header, rows, log=None):
    """Write a CSV with the header first and provenance on line 2."""
    with open(path, "w") as fh:
        fh.write(header + "\n")
        fh.write(provenance() + "\n")
        for row in rows:
            fh.write(row + "\n")
    if log:
        log(f"    wrote {os.path.basename(path)}")
    return path


def _fmt(v):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else f"{v:.10e}"


# -- posterior summaries -----------------------------------------------

def summarize(draws, names, rhat=None, ess_bulk=None, ess_tail=None):
    """Median, std and 16th/84th percentiles per parameter, plus diagnostics.

    ``draws``: ``(n_samples, dim)`` in physical units.
    """
    out = {}
    for i, name in enumerate(names):
        vals = np.asarray(draws[:, i], dtype=np.float64)
        lo, med, hi = np.percentile(vals, [16, 50, 84])
        out[name] = {
            "median": float(med), "std": float(np.std(vals)),
            "lo": float(lo), "hi": float(hi),
            "rhat": None if rhat is None else float(rhat[i]),
            "ess_bulk": None if ess_bulk is None else float(ess_bulk[i]),
            "ess_tail": None if ess_tail is None else float(ess_tail[i]),
        }
    return out


def export_summary(outdir, target, mode, summary, *, extra=None, log=None):
    """(i) Parameter statistics and convergence diagnostics."""
    rows = []
    for name, e in summary.items():
        ess_b = "" if e["ess_bulk"] is None else f"{e['ess_bulk']:.0f}"
        ess_t = "" if e["ess_tail"] is None else f"{e['ess_tail']:.0f}"
        rhat = "" if e["rhat"] is None else f"{e['rhat']:.10f}"
        rows.append(f"{name},{_fmt(e['median'])},{_fmt(e['std'])},"
                    f"{_fmt(e['lo'])},{_fmt(e['hi'])},{rhat},{ess_b},{ess_t}")
    for name, e in (extra or {}).items():
        rows.append(f"{name},{_fmt(e['median'])},{_fmt(e['std'])},"
                    f"{_fmt(e['lo'])},{_fmt(e['hi'])},,,")
    return _write_csv(
        product_path(outdir, target, mode, "summary", "csv"),
        "Parameter,Median,Std,16th,84th,R-hat,Bulk_ESS,Tail_ESS", rows, log)


#: Rows formatted per chunk when streaming the chains CSV to disk.
_CSV_CHUNK_ROWS = 50_000


def export_chains(outdir, target, mode, columns, arrays, *, thin=1,
                  n_total=None, log=None):
    """(ii) The joint posterior, gzipped CSV inside a tarball.

    The version member is appended *after* the CSV: hurin's readers take
    ``getmembers()[0]`` to find the data.

    ``thin`` > 1 means ``arrays`` is a systematic subsample (every
    ``thin``-th draw of every chain) of ``n_total`` draws; the provenance
    line says so, so a reader never mistakes it for the full set. The CSV is
    formatted in chunks into a temporary file rather than built as one
    string: at the draw cap that string was ~1.6 GB and took ~4 minutes.
    """
    path = product_path(outdir, target, mode, "chains", "csv.tar.gz")
    data = np.column_stack([np.asarray(a, dtype=np.float64) for a in arrays])
    note = provenance()
    if thin > 1:
        note += (f" | thinned 1/{thin}: {len(data)} of "
                 f"{n_total if n_total is not None else '?'} draws")
    member = os.path.basename(path).replace(".tar.gz", "")
    fmt = ",".join(["%.10e"] * data.shape[1])

    tmp = path + ".csv.tmp"
    try:
        with open(tmp, "w") as fh:
            fh.write(",".join(columns) + "\n")
            fh.write(note + "\n")
            for s in range(0, len(data), _CSV_CHUNK_ROWS):
                np.savetxt(fh, data[s:s + _CSV_CHUNK_ROWS], fmt=fmt)
        with tarfile.open(path, "w:gz") as tar:
            tar.add(tmp, arcname=member)
            ver = (f"turin {__version__} rev{MODEL_REV}\n"
                   f"{launch_command()}\n").encode()
            vinfo = tarfile.TarInfo(name="turin_version.txt")
            vinfo.size = len(ver)
            tar.addfile(vinfo, io.BytesIO(ver))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if log:
        log(f"    wrote {os.path.basename(path)} "
            f"({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def export_logrho(outdir, target, mode, log10_rho, *, log=None):
    """(iii) The derived stellar-density posterior, one column."""
    rows = [f"{v:.10e}" for v in np.asarray(log10_rho, dtype=np.float64)]
    return _write_csv(product_path(outdir, target, mode, "logrho", "csv"),
                      "log10_rho", rows, log)


def export_lcdata(outdir, target, mode, epoch_data, model, *, log=None):
    """(iv) The data that was fitted, with the best-fit model beside it.

    Padded slots are dropped and the rows sorted by time, as hurin does.
    """
    mask = np.asarray(epoch_data["mask"]) > 0
    t = np.asarray(epoch_data["times_padded"], dtype=np.float64)[mask]
    f = np.asarray(epoch_data["flux_padded"], dtype=np.float64)[mask]
    e = np.asarray(epoch_data["ferr_padded"], dtype=np.float64)[mask]
    m = np.asarray(model, dtype=np.float64)[mask]
    order = np.argsort(t)
    rows = [f"{t[i]:.10e},{f[i]:.10e},{e[i]:.10e},{m[i]:.10e}" for i in order]
    return _write_csv(product_path(outdir, target, mode, "lcdata", "csv"),
                      "time,flux,flux_err,model_flux", rows, log)


def export_ttv_times(outdir, target, mode, rows, *, log=None):
    """(viii) Per-epoch transit times and their O-C, with fit diagnostics."""
    out = []
    for r in rows:
        out.append(
            f"{int(r['epoch'])},{r['tmid']:.10e},{r['tmid_err']:.10e},"
            f"{r['ttv_min']:.10e},{r['ttv_err_min']:.10e},"
            f"{r['snr']:.3f},{int(r['npts'])},{r['chi2']:.2f}")
    return _write_csv(product_path(outdir, target, mode, "times", "csv"),
                      "epoch,tmid,tmid_err,ttv_min,ttv_err_min,snr,npts,chi2",
                      out, log)


# -- derived quantities ------------------------------------------------

def derived_b(draws, names, b_prior):
    """The impact parameter for every draw, under the layout's b prior.

    ``beta`` is sampled as a fraction of a k-dependent bound, so recovering
    ``b`` needs to know which bound: getting this wrong would misreport the
    geometry and, through it, the density.
    """
    idx = {n: i for i, n in enumerate(names)}
    return _model.impact_parameter(draws[:, idx["beta"]], draws[:, idx["k"]],
                                   b_prior)


def log10_rho_draws(draws, names, b_prior, *, P_ref=None):
    """log10(rho*) for every draw, from the fitted duration and geometry.

    Uses the same Seager & Mallen-Ornelas inversion the ``circular`` model
    fits with, so the reported density is consistent with the likelihood. For
    a TTV fit the period is not sampled, so ``P_ref`` supplies it.
    """
    idx = {n: i for i, n in enumerate(names)}
    if "dP" in idx:
        period = draws[:, idx["dP"]]
    elif P_ref is not None:
        period = np.full(len(draws), float(P_ref))
    else:
        raise ValueError("a TTV fit needs P_ref to derive log10_rho")
    return _model.log10_rho_from_T14(
        draws[:, idx["T14"]], period, draws[:, idx["k"]],
        derived_b(draws, names, b_prior))


def fit_linear_ephemeris(epochs, times, *, n_iter=3, clip=5.0):
    """Robust linear fit to measured transit times, for the O-C reference.

    Iterative MAD clipping, as hurin does: the reference should be the times
    themselves, not the archive ephemeris, or the diagram shows the archive's
    error rather than the planet's.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    keep = np.ones(times.size, dtype=bool)
    P = tau0 = np.nan
    for _ in range(n_iter):
        if keep.sum() < 3:
            break
        P, tau0 = np.polyfit(epochs[keep], times[keep], 1)
        resid = times - (tau0 + P * epochs)
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if not np.isfinite(mad) or mad <= 0:
            break
        keep = np.abs(resid) < clip * 1.4826 * mad
    return float(P), float(tau0), keep


# -- resume state ------------------------------------------------------

@dataclass
class ResumeState:
    """What turin needs to continue, or to refuse to.

    The guard fields exist because silently mixing them would produce a
    posterior that is not a posterior for anything: hurin learned this with
    its (b, k) parameterizations, which are not mutually resumable.
    """

    target: str
    mode: str
    turin_version: str
    launch_command: str
    tag: str | None
    # guards: a mismatch against the CLI must refuse, naming the stored value
    b_prior: str
    profile_mode: str
    geometry: str
    sampler: str
    n_chains: int
    ttv_max: float | None
    n_durations: float
    legendre_orders: np.ndarray
    exposure_time: float
    num_resample: int
    # progress
    n_samples_done: int
    done: bool
    #: The likelihood revision these chains were sampled under; see
    #: :data:`turin.MODEL_REV`. Defaulted so a state written before this
    #: field existed reads back as revision 1, which is what it was.
    model_rev: int = 1
    ml_params: dict = field(default_factory=dict)
    #: anvil's own resumable state, when the installed version supports it
    anvil_state_path: str | None = None
    #: last positions in unconstrained space, the fallback restart point
    last_u: np.ndarray | None = None
    #: whether grid-Gibbs moved the epoch times ("on"/"off"). Defaulted to
    #: "off" because every state written before it existed was sampled
    #: without it; LinEph records "off" since it has no epoch times.
    gibbsgrid: str = "off"

    GUARDS = ("b_prior", "profile_mode", "geometry", "sampler", "ttv_max",
              "gibbsgrid")

    def check_model_rev(self, log=None):
        """Refuse to continue chains sampled under an older likelihood.

        The guards above catch the *user* asking for a different model. This
        catches the model changing underneath an unchanged command line --
        a correction landing between one run and the next.

        One exception, following hurin: a **finished LinEph** state is never
        continued, and is only read for the maximum-likelihood shapes that
        seed the TTV template sweep. Approximate shapes from a slightly older
        likelihood are fine for an initialization, so it passes with a note.
        """
        from . import MODEL_REV

        if self.model_rev == MODEL_REV:
            return
        if self.mode == "lineph" and self.done:
            if log:
                log(f"  note: this {self.mode} state predates model revision "
                    f"{MODEL_REV} (it is rev {self.model_rev}), but it is "
                    "complete and only seeds the TTV initialization, so it is "
                    "used as-is")
            return
        raise SystemExit(
            f"{self.target} {self.mode}: these chains were sampled under "
            f"likelihood revision {self.model_rev}, but this turin is "
            f"revision {MODEL_REV} -- the log-density has changed since, so "
            "the chains cannot be continued. Re-run with --fresh.")

    def check(self, **cli):
        """Raise if the CLI disagrees with this state on any guarded field."""
        for name in self.GUARDS:
            if name not in cli or cli[name] is None:
                continue
            stored, given = getattr(self, name), cli[name]
            if isinstance(stored, float) or isinstance(given, float):
                same = (stored is not None and given is not None
                        and abs(float(stored) - float(given)) < 1e-12)
            else:
                same = stored == given
            if not same:
                raise SystemExit(
                    f"{self.target} {self.mode}: this run was made with "
                    f"{name}={stored!r}, but you asked for {given!r}. "
                    f"Use --{name.replace('_', '-')}={stored!r} to extend it, "
                    f"or --fresh to start over.")


def resume_path(outdir, target, mode):
    return product_path(outdir, target, mode, "resume", "pkl")


def save_resume(outdir, target, mode, state, *, log=None):
    path = resume_path(outdir, target, mode)
    with open(path, "wb") as fh:
        pickle.dump(state, fh)
    if log:
        log(f"    wrote {os.path.basename(path)}")
    return path


def load_resume(outdir, target, mode):
    """Load resume state, or ``None`` if this lineage has not run yet."""
    path = resume_path(outdir, target, mode)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


#: Extensions turin writes, longest first so ".csv.tar.gz" wins over ".csv".
_PRODUCT_EXTS = (".csv.tar.gz", ".csv", ".pdf", ".pkl", ".npz")


def _product_stem(filename, prefix):
    """``summary`` or ``summary.r2`` for one of our products, else ``None``.

    Only the part after ``{target}_{mode}_`` is inspected: the target name
    itself contains a dot (``KOI-1.01``), so testing the whole filename for a
    tag separator matches every untagged product too.
    """
    if not filename.startswith(prefix):
        return None
    rest = filename[len(prefix):]
    for ext in _PRODUCT_EXTS:
        if rest.endswith(ext):
            return rest[: -len(ext)]
    return None


def clear_products(outdir, target, mode=None):
    """Remove this lineage's products, for --fresh.

    A differently tagged run is an independent lineage and is left alone.
    """
    removed = []
    if not os.path.isdir(outdir):
        return removed
    for m in ((mode,) if mode else ("lineph", "ttv")):
        prefix = f"{target}_{m}_"
        for fn in sorted(os.listdir(outdir)):
            stem = _product_stem(fn, prefix)
            if stem is None:
                continue
            if _RUN_TAG:
                if not stem.endswith(f".{_RUN_TAG}"):
                    continue
            elif "." in stem:          # a tagged product, not ours
                continue
            os.remove(os.path.join(outdir, fn))
            removed.append(fn)
    return removed
