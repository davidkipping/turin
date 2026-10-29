"""Gaia epoch photometry: DR3 over DataLink, or any release from a local CSV.

A Gaia light curve is a few dozen field-of-view transits per band (G, BP,
RP) over ~3 years (DR3; ~5.5 in DR4), in pairs 106.5 min apart from the two
fields of view and clusters a few days long. Nothing here resembles a
Kepler quarter: there is no MAST product, no archive ephemeris (a candidate's
comes from the search that found it, on the command line), and the times
are Gaia's own, **BJD(TCB) - 2455197.5** ("Gaia time"), kept as they are.

The reader keeps every band, drops what Gaia itself rejects
(``rejected_by_photometry`` for the whole transit,
``variability_flag_<band>_reject`` per band) and non-finite or non-positive
fluxes and errors, and divides each band by its median flux. Outlier
clipping is the fit's conditioning, in ``prep``. MetalLS
(``metalls/ingest.py``, ``benchmarks/fetch_gaia_cadences.py``) reads the
same columns the same way; turin does not import it (it pulls in mlx and
its whole search).
"""

from __future__ import annotations

import csv
import io
import os
import pickle
import re
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np

#: Gaia time is BJD(TCB) minus this.
GAIA_TIME_OFFSET = 2455197.5
#: Band labels, in the order of the integer codes stored per point.
BANDS = ("G", "BP", "RP")
#: Releases turin can name. Only DR3 has an archive reader; DR4's format is
#: to be confirmed at its release (December 2026), so until then its data
#: come in through ``--lcfile``.
RELEASES = ("DR3", "DR4")
ARCHIVE_RELEASES = ("DR3",)
#: Each band's integration, in days. G averages a field-of-view transit's 9
#: astrometric-field CCD crossings (~4.4 s each, ~40 s from first to last;
#: MetalLS's figure); BP and RP are one CCD each. All are far below a
#: transit's ingress, so the boxcar these stand for is a small correction.
EXPOSURE_DAYS = {"G": 40.0 / 86400, "BP": 4.4 / 86400, "RP": 4.4 / 86400}

DATALINK_URL = "https://gea.esac.esa.int/data-server/data"
DATALINK_TIMEOUT = 300           # seconds per request
DATALINK_RETRY_DELAYS = (10, 30, 60, 120)

TARGET_RE = re.compile(r"^GAIA(DR\d+)?-(\d+)$", re.IGNORECASE)

# DataLink CSV columns per band: time, flux, flux error, variability reject
_DR3_COLUMNS = (
    ("g_transit_time", "g_transit_flux", "g_transit_flux_error",
     "variability_flag_g_reject"),
    ("bp_obs_time", "bp_flux", "bp_flux_error", "variability_flag_bp_reject"),
    ("rp_obs_time", "rp_flux", "rp_flux_error", "variability_flag_rp_reject"),
)


def is_gaia(target) -> bool:
    """Whether ``target`` names a Gaia source (any release)."""
    return bool(target) and TARGET_RE.match(str(target).strip()) is not None


def parse_target(target: str) -> dict:
    """``GaiaDR3-<source_id>`` -> ``{type, release, source_id, name}``.

    The release is part of the name: DR4 reprocesses every epoch, and some
    source ids change between releases, so a DR3 fit and a DR4 fit of one
    star are different light curves with different products.
    """
    m = TARGET_RE.match(str(target).strip())
    if not m:
        raise ValueError(f"Cannot parse Gaia target {target!r}; expected "
                         "GaiaDR3-<source_id>")
    if not m.group(1):
        raise ValueError(f"Gaia target {target!r} needs its release: "
                         f"GaiaDR3-{m.group(2)}")
    release = m.group(1).upper()
    if release not in RELEASES:
        raise ValueError(f"Gaia {release} is not a release turin knows "
                         f"({', '.join(RELEASES)})")
    sid = int(m.group(2))
    return {"type": "gaia", "release": release, "source_id": sid,
            "name": f"Gaia{release}-{sid}"}


def _flag(value) -> bool:
    return str(value).strip().lower() == "true"


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_datalink_csv(text: str) -> dict:
    """One source's DataLink ``EPOCH_PHOTOMETRY`` CSV -> raw point arrays.

    Returns ``time`` (Gaia time), ``flux``, ``flux_err`` (e-/s, as
    published), ``band`` (int8 codes into :data:`BANDS`) and ``transit_id``,
    sorted by time, with Gaia's own rejections and unusable values removed.
    A field-of-view transit carries up to one point per band, each at its
    own time (BP and RP ~25-30 s from G).
    """
    t, f, e, b, tid = [], [], [], [], []
    lines = (ln for ln in io.StringIO(text) if not ln.startswith("#"))
    for row in csv.DictReader(lines):
        if _flag(row.get("rejected_by_photometry")):
            continue
        transit = int(row["transit_id"])
        for code, (tc, fc, ec, vc) in enumerate(_DR3_COLUMNS):
            if _flag(row.get(vc)):
                continue
            tt, ff, ee = (_number(row.get(c)) for c in (tc, fc, ec))
            if not (np.isfinite(tt) and np.isfinite(ff) and np.isfinite(ee)
                    and ff > 0 and ee > 0):
                continue
            t.append(tt), f.append(ff), e.append(ee), b.append(code)
            tid.append(transit)
    return _sorted(t, f, e, b, tid)


def _sorted(t, f, e, b, tid=None):
    t = np.asarray(t, dtype=np.float64)
    o = np.argsort(t, kind="stable")
    out = {
        "time": t[o],
        "flux": np.asarray(f, dtype=np.float64)[o],
        "flux_err": np.asarray(e, dtype=np.float64)[o],
        "band": np.asarray(b, dtype=np.int8)[o],
    }
    if tid is not None:
        out["transit_id"] = np.asarray(tid, dtype=np.int64)[o]
    return out


def normalise_bands(lc: dict) -> dict:
    """Divide each band's flux and error by that band's median flux."""
    flux = np.array(lc["flux"], dtype=np.float64)
    err = np.array(lc["flux_err"], dtype=np.float64)
    band = np.asarray(lc["band"])
    for code in np.unique(band):
        m = band == code
        med = float(np.median(flux[m]))
        flux[m] /= med
        err[m] /= med
    return {**lc, "flux": flux, "flux_err": err}


def band_code(label) -> int:
    """A band label (``G``, ``BP``, ``RP``, any case, or 0/1/2) as its code."""
    s = str(label).strip().upper()
    if s in BANDS:
        return BANDS.index(s)
    if s in ("0", "1", "2"):
        return int(s)
    raise ValueError(f"unknown Gaia band {label!r}; expected one of "
                     f"{', '.join(BANDS)} (or 0, 1, 2)")


def read_lcfile(path: str) -> dict:
    """A light curve from a local CSV: the route for releases turin has no
    archive reader for (DR4 until its format is public), or for photometry
    prepared elsewhere.

    Columns, by header name: ``time`` (Gaia time, BJD(TCB) - 2455197.5),
    ``flux`` and ``flux_err`` (any unit: each band is normalised by its
    median), and optionally ``band`` (``G``/``BP``/``RP`` or 0/1/2; all G
    without it). Lines starting with ``#`` are comments.
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    missing = {"time", "flux", "flux_err"} - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}"
                         " (needs time, flux, flux_err, optional band)")
    t, f, e, b = [], [], [], []
    for row in rows:
        tt, ff, ee = (_number(row[c]) for c in ("time", "flux", "flux_err"))
        if not (np.isfinite(tt) and np.isfinite(ff) and np.isfinite(ee)
                and ff > 0 and ee > 0):
            continue
        t.append(tt), f.append(ff), e.append(ee)
        b.append(band_code(row["band"]) if row.get("band") not in (None, "")
                 else 0)
    if not t:
        raise ValueError(f"{path}: no usable rows (finite, positive flux and "
                         "error)")
    return _sorted(t, f, e, b)


def _post(url, params, progress=None):
    """POST with retries on transient failures (the ESA archive, like the
    Exoplanet Archive, has maintenance windows)."""
    data = urllib.parse.urlencode(params).encode()
    last = None
    for attempt, delay in enumerate((*DATALINK_RETRY_DELAYS, None)):
        try:
            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=DATALINK_TIMEOUT) as r:
                return r.read()
        except Exception as err:          # noqa: BLE001 -- retried, then raised
            last = err
        if delay is None:
            break
        if progress:
            progress(f"Gaia archive unavailable ({last}); retrying in "
                     f"{delay}s (attempt {attempt + 2}/"
                     f"{len(DATALINK_RETRY_DELAYS) + 1})")
        time.sleep(delay)
    raise RuntimeError(f"Gaia DataLink request failed: {last}")


def download_dr3(source_id: int, progress=None) -> dict:
    """One source's DR3 epoch photometry over DataLink, parsed.

    ``DATA_STRUCTURE=DATAMODEL_STANDARD``: the archive refuses
    ``INDIVIDUAL`` (MetalLS's request) since October 2026. For one source and
    ``FORMAT=CSV`` it returns the CSV itself, with the same columns; a zip
    holding it is accepted too.
    """
    if progress:
        progress(f"Downloading Gaia DR3 {source_id} epoch photometry "
                 "(DataLink)...")
    raw = _post(DATALINK_URL, {
        "RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY", "RELEASE": "Gaia DR3",
        "FORMAT": "CSV", "DATA_STRUCTURE": "DATAMODEL_STANDARD",
        "ID": f"Gaia DR3 {source_id}"}, progress=progress)
    if raw[:2] == b"PK":
        zf = zipfile.ZipFile(io.BytesIO(raw))
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Gaia DataLink returned {len(names)} files for "
                             f"DR3 {source_id}; expected one")
        raw = zf.read(names[0])
    text = raw.decode()
    if "transit_id" not in text.split("\n", 1)[0] and not text.startswith("#"):
        raise ValueError(f"Gaia DataLink returned no epoch photometry for "
                         f"DR3 {source_id} (has_epoch_photometry false, or "
                         "not a DR3 source id)")
    return parse_datalink_csv(text)


def _cache_path(info: dict) -> str:
    from .lightcurve import CACHE_DIR     # one cache directory for all data
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{info['name']}.pkl")


def get_lightcurve(target, progress=None, lcfile=None) -> dict:
    """A Gaia source's light curve, in the shape ``prep`` reads.

    Returns ``time``, ``flux``, ``flux_err`` (each band normalised by its
    median), ``band`` (int8 codes into :data:`BANDS`), ``mission``
    (``gaia_dr3`` ...) and ``source`` (``datalink`` or the file read). With
    ``lcfile`` the points come from that CSV (:func:`read_lcfile`) and
    nothing is cached; otherwise from the release's archive, cached.
    """
    info = parse_target(target)
    mission = f"gaia_{info['release'].lower()}"
    if lcfile:
        if progress:
            progress(f"Reading {lcfile}")
        raw, source = read_lcfile(lcfile), os.path.abspath(lcfile)
    else:
        if info["release"] not in ARCHIVE_RELEASES:
            raise ValueError(
                f"turin has no archive reader for Gaia {info['release']} yet "
                "(its format is confirmed only at release); pass its "
                "photometry with --lcfile")
        path = _cache_path(info)
        if os.path.exists(path):
            if progress:
                progress("Loading from cache...")
            with open(path, "rb") as fh:
                return pickle.load(fh)
        raw, source = download_dr3(info["source_id"], progress), "datalink"
    if raw["time"].size == 0:
        raise ValueError(f"{info['name']}: no usable epoch photometry")
    lc = {**normalise_bands(raw), "mission": mission, "source": source}
    if not lcfile:
        with open(_cache_path(info), "wb") as fh:
            pickle.dump(lc, fh)
    return lc


#: IAU 2006 Resolution B3: TDB = TCB - L_B (JD_TCB - T0) 86400 s + TDB0.
_L_B = 1.550519768e-8
_T0_JD = 2443144.5003725
_TDB0_S = -6.55e-5


def gaia_time_to_tdb(t_gaia):
    """Gaia time (BJD_TCB - 2455197.5) -> the same instant as BJD_TDB - 2455197.5.

    TCB runs ahead of TDB at the rate L_B: 18.6 s at the start of 2015,
    gaining 0.49 s a year (21.0 s at the start of 2020). Ephemerides from other surveys (TESS, the Exoplanet Archive) are
    BJD_TDB, so a Gaia mid-time compared with them needs this.
    """
    t = np.asarray(t_gaia, dtype=np.float64)
    return t - _L_B * (t + GAIA_TIME_OFFSET - _T0_JD) + _TDB0_S / 86400.0
