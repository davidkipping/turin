"""Ported verbatim from hurin 0.1.67 (hurin/lightcurve.py).

Pure NumPy / lightkurve / requests: no JAX, nothing to rewrite for the GPU.
Keep changes here minimal and upstream-traceable.
"""


import os
import pickle
import re

import lightkurve as lk
import numpy as np

#: Light-curve pickle cache. hurin kept this inside the repo; turin uses a
#: user cache directory so the working tree stays clean, overridable by the
#: CLI's --cache-dir (or $TURIN_CACHE_DIR) via set_cache_dir().
CACHE_DIR = os.environ.get(
    "TURIN_CACHE_DIR",
    os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "turin"
    ),
)


def set_cache_dir(path):
    """Point the light-curve cache at ``path`` (CLI --cache-dir)."""
    global CACHE_DIR
    CACHE_DIR = os.path.abspath(os.path.expanduser(path))
    return CACHE_DIR

# Cadence preference order (exposure time in seconds)
# Default: prefer moderate cadences (skip ultra-short 20s)
_TESS_DEFAULT_PREF = [120.0, 600.0, 1800.0, 200.0]
# SC Override: prefer short cadence, including 20s
_TESS_SC_PREF = [120.0, 20.0, 200.0, 600.0, 1800.0]


def parse_target(target: str) -> dict:
    """Parse a KOI or TOI string into its components.

    Returns dict with keys: type ("koi" or "toi"), number (int), planet (int),
    and the original cleaned string.
    """
    target = target.strip().upper()

    m = re.match(r"KOI[- ]?(\d+)\.(\d+)", target)
    if m:
        return {
            "type": "koi",
            "number": int(m.group(1)),
            "planet": int(m.group(2)),
            "name": f"KOI-{m.group(1)}.{m.group(2)}",
        }

    m = re.match(r"TOI[- ]?(\d+)\.(\d+)", target)
    if m:
        return {
            "type": "toi",
            "number": int(m.group(1)),
            "planet": int(m.group(2)),
            "name": f"TOI-{m.group(1)}.{m.group(2)}",
        }

    raise ValueError(f"Cannot parse target: {target!r}. Expected format: KOI-174.01 or TOI-123.01")


def _cache_path(target_info: dict, sc_override: bool = False) -> str:
    """Return the cache file path for a target."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = target_info["name"].replace(" ", "_")
    suffix = "_sc" if sc_override else ""
    return os.path.join(CACHE_DIR, f"{safe_name}{suffix}.pkl")


def get_lightcurve(target, progress=None, sc_override=False):
    """Download and return the stitched PDC light curve for a KOI or TOI.

    Args:
        target: KOI or TOI identifier string.
        progress: Optional callback ``progress(message)`` called with status
            strings during download.
        sc_override: If True, prefer shortest available cadence (SC for
            Kepler, 20s/2min for TESS). Default False uses LC for Kepler
            and 2min>10min>30min for TESS.

    Returns dict with keys: time, flux, flux_err, quality (numpy arrays),
        mission (str), cadence_days (float).
    """
    info = parse_target(target)
    cache_file = _cache_path(info, sc_override=sc_override)

    if os.path.exists(cache_file):
        if progress:
            progress("Loading from cache...")
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        problem = _mixed_target_problem(cached["time"])
        if problem is None:
            return cached
        # a cache written before the one-star filter existed
        if progress:
            progress(f"Cached light curve is invalid ({problem}); "
                     "downloading again")

    star = _catalog_id(info, progress=progress)
    if info["type"] == "koi":
        if sc_override:
            result = _download_kepler_sc(info, star, progress=progress)
        else:
            result = _download(info, star, author="Kepler", cadence="long",
                               mission="kepler", label="quarter",
                               progress=progress)
    else:
        result = _download_tess(info, star, sc_override=sc_override,
                                progress=progress)

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


def _catalog_id(info, progress=None):
    """The host star's catalogue number: KIC for a KOI, TIC for a TOI.

    Looked up in the NASA Exoplanet Archive, which is where the target's
    ephemeris comes from too. Returns an int, or None if the archive has no
    entry (the download then refuses to guess between stars).
    """
    from .ephemeris import _tap_get, _toi_to_archive_number

    if info["type"] == "koi":
        # every planet in a KOI system shares its host's kepid
        query = ("select kepid from cumulative where "
                 f"kepoi_name='K{info['number']:05d}.01'")
        label = "KIC"
    else:
        query = ("select tid from toi where "
                 f"toi={_toi_to_archive_number(info['name'])}")
        label = "TIC"
    try:
        lines = _tap_get(query).text.strip().split("\n")
        value = lines[1].strip() if len(lines) >= 2 else ""
        star = int(float(value)) if value else None
    except Exception:
        star = None
    if progress and star is not None:
        progress(f"Host star: {label} {star}")
    return star


def _target_number(target_name):
    """MAST's target_name as a catalogue number: 'kplr008423344' -> 8423344."""
    digits = re.sub(r"\D", "", str(target_name)).lstrip("0")
    return int(digits) if digits else None


def _restrict_to_star(search, star, info, progress=None):
    """Keep only the search rows for the target's own star.

    A name search can return a neighbouring star's light curves as well, all
    at distance 0: "KOI-7592.01" returns KIC 8423344 (the KOI, 14 quarters)
    and KIC 8423352 (18 quarters). Downloading every row stitches two stars
    together. Rows are therefore matched to the archive's catalogue number,
    and a search spanning several stars with no number to match refuses
    rather than guessing.
    """
    names = [str(x) for x in search.table["target_name"]]
    numbers = [_target_number(n) for n in names]
    distinct = sorted(set(numbers) - {None})
    if star is not None:
        keep = np.where(np.array(numbers, dtype=object) == star)[0]
        if keep.size == 0:
            raise ValueError(
                f"MAST returned light curves for {distinct} but not for "
                f"{info['name']}'s host star {star}")
        if len(distinct) > 1 and progress:
            others = [n for n in distinct if n != star]
            progress(f"Dropped {len(names) - keep.size} light curve(s) of "
                     f"neighbouring star(s) {others}")
        return search[keep]
    if len(distinct) > 1:
        raise ValueError(
            f"MAST returned light curves for several stars {distinct} and "
            f"the archive has no catalogue number for {info['name']} to "
            "choose between them")
    return search


def _lacks_star(search, star):
    """True when the search has rows, the host star is known, and none of
    the rows are the host's: the name resolved to a neighbour only."""
    if search is None or len(search) == 0 or star is None:
        return False
    return star not in {_target_number(n) for n in search.table["target_name"]}


def _mixed_target_problem(time):
    """Why a stitched light curve cannot be one star's, or None.

    One star's cadences never repeat a timestamp; two stars observed in the
    same quarters do, all over.
    """
    t = np.asarray(time, dtype=np.float64)
    t = t[np.isfinite(t)]
    n_dup = int(t.size - np.unique(np.round(t, 6)).size)
    if n_dup:
        return f"{n_dup} repeated timestamps, so more than one star"
    return None


def _download(info, star, author, cadence, mission, label, progress=None):
    """Download light curve one quarter/sector at a time."""
    kwargs = {"author": author}
    if cadence is not None:
        kwargs["cadence"] = cadence

    if progress:
        progress(f"Searching MAST for {info['name']}...")

    try:
        search = lk.search_lightcurve(info["name"], **kwargs)
    except TimeoutError:
        if info.get("type") != "koi":
            raise
        if progress:
            progress(f"MAST timed out on {info['name']}, retrying with KIC identifier...")
        search = None

    # KOI fallback: search by KIC if the KOI name did not resolve, timed
    # out, or resolved only to a neighbouring star
    if info.get("type") == "koi" and star is not None and (
            search is None or len(search) == 0 or _lacks_star(search, star)):
        if progress:
            progress(f"Retrying MAST search with KIC {star}...")
        search = lk.search_lightcurve(f"KIC {star}", **kwargs)

    if search is None or len(search) == 0:
        raise ValueError(f"No {mission.title()} light curves found for {info['name']}")
    search = _restrict_to_star(search, star, info, progress)

    total = len(search)
    lc_list = []
    for i, row in enumerate(search):
        if progress:
            progress(f"Downloading {label} {i + 1} of {total}...")
        lc_list.append(row.download())

    return _stitch_and_pack(lc_list, mission, progress)


def _download_kepler_sc(info, star, progress=None):
    """Download Kepler data preferring SC over LC per quarter."""
    if progress:
        progress(f"Searching MAST for {info['name']} (all cadences)...")

    try:
        search = lk.search_lightcurve(info["name"], author="Kepler")
    except TimeoutError:
        if progress:
            progress(f"MAST timed out on {info['name']}, retrying with KIC identifier...")
        search = None

    # KOI fallback: search by KIC if the KOI name did not resolve, timed
    # out, or resolved only to a neighbouring star
    if star is not None and (search is None or len(search) == 0
                             or _lacks_star(search, star)):
        if progress:
            progress(f"Retrying MAST search with KIC {star}...")
        search = lk.search_lightcurve(f"KIC {star}", author="Kepler")

    if search is None or len(search) == 0:
        raise ValueError(f"No Kepler light curves found for {info['name']}")
    search = _restrict_to_star(search, star, info, progress)

    search_table = search.table
    exptimes = np.array(search_table["t_exptime"], dtype=float)

    # Parse quarter numbers from mission column (e.g. "Kepler Quarter 05")
    quarters = []
    for row in search_table:
        m = re.search(r"Quarter\s+(\d+)", str(row.get("mission", "")))
        quarters.append(int(m.group(1)) if m else -1)
    quarters = np.array(quarters)

    # For each quarter, pick SC if available, else LC
    selected_indices = []
    unique_quarters = sorted(set(quarters))
    for q in unique_quarters:
        if q < 0:
            continue
        q_mask = quarters == q
        q_indices = np.where(q_mask)[0]
        sc_indices = [j for j in q_indices if exptimes[j] < 120.0]
        if sc_indices:
            selected_indices.extend(sc_indices)
        else:
            selected_indices.extend(q_indices.tolist())
    selected_indices.sort()

    total = len(selected_indices)
    lc_list = []
    for count, idx in enumerate(selected_indices):
        cad_label = "SC" if exptimes[idx] < 120.0 else "LC"
        if progress:
            progress(f"Downloading quarter {count + 1} of {total} ({cad_label})...")
        lc_list.append(search[idx].download())

    return _stitch_and_pack(lc_list, "kepler", progress)


def _pick_best_per_group(exptimes, groups, group_ids, pref_order, label_name):
    """Select the best cadence per group (quarter/sector) given a preference.

    Parameters
    ----------
    exptimes : array of float
        Exposure times for each search result row.
    groups : array of int
        Group ID (quarter or sector number) for each row.
    group_ids : list of int
        Sorted unique group IDs to process.
    pref_order : list of float
        Cadence preference (seconds), first = most preferred.
    label_name : str
        "quarter" or "sector" for progress messages.

    Returns
    -------
    selected_indices : list of int
        Indices into the search results to download.
    cad_labels : list of str
        Human-readable cadence label per selected index.
    """
    # Tolerance for matching cadence (±30%)
    def _matches(exptime, target_cad):
        return abs(exptime - target_cad) < 0.3 * target_cad

    selected_indices = []
    cad_labels = []

    for gid in group_ids:
        if gid < 0:
            continue
        g_mask = groups == gid
        g_indices = np.where(g_mask)[0]

        chosen = None
        for pref_cad in pref_order:
            matches = [j for j in g_indices if _matches(exptimes[j], pref_cad)]
            if matches:
                chosen = matches
                break
        if chosen is None:
            # Fallback: take whatever is available
            chosen = g_indices.tolist()

        for j in chosen:
            et = exptimes[j]
            if et < 30:
                cad_labels.append("20s")
            elif et < 150:
                cad_labels.append("2min")
            elif et < 300:
                cad_labels.append("200s")
            elif et < 900:
                cad_labels.append("10min")
            else:
                cad_labels.append("30min")
        selected_indices.extend(chosen)

    selected_indices.sort()
    # Re-sort cad_labels to match sorted indices
    idx_label = dict(zip([s for s in selected_indices], cad_labels))
    cad_labels = [idx_label.get(i, "?") for i in selected_indices]

    return selected_indices, cad_labels


def _download_tess(info, star, sc_override=False, progress=None):
    """Download TESS data with per-sector cadence preference."""
    if progress:
        progress(f"Searching MAST for {info['name']} (all cadences)...")

    search = lk.search_lightcurve(info["name"], author="SPOC")
    if len(search) == 0:
        raise ValueError(f"No TESS light curves found for {info['name']}")
    search = _restrict_to_star(search, star, info, progress)

    search_table = search.table
    exptimes = np.array(search_table["t_exptime"], dtype=float)

    # Parse sector numbers from mission column (e.g. "TESS Sector 01")
    sectors = []
    for row in search_table:
        m = re.search(r"Sector\s+(\d+)", str(row.get("mission", "")))
        sectors.append(int(m.group(1)) if m else -1)
    sectors = np.array(sectors)

    pref = _TESS_SC_PREF if sc_override else _TESS_DEFAULT_PREF
    unique_sectors = sorted(set(sectors))
    selected_indices, cad_labels = _pick_best_per_group(
        exptimes, sectors, unique_sectors, pref, "sector")

    total = len(selected_indices)
    lc_list = []
    for count, idx in enumerate(selected_indices):
        if progress:
            progress(f"Downloading sector {count + 1} of {total} "
                     f"({cad_labels[count]})...")
        lc_list.append(search[idx].download())

    return _stitch_and_pack(lc_list, "tess", progress)


def _stitch_and_pack(lc_list, mission, progress=None):
    """Stitch light curves and return the standard result dict."""
    if progress:
        progress("Stitching light curves...")

    # Concatenate quality flags before stitching (stitch may not preserve them)
    quality_arrays = []
    for single_lc in lc_list:
        if hasattr(single_lc, "quality") and single_lc.quality is not None:
            quality_arrays.append(np.array(single_lc.quality.value, dtype=np.int32))
        else:
            quality_arrays.append(np.zeros(len(single_lc.time), dtype=np.int32))
    quality = np.concatenate(quality_arrays)

    lc_collection = lk.LightCurveCollection(lc_list)
    lc = lc_collection.stitch()

    time = np.array(lc.time.value, dtype=np.float64)
    problem = _mixed_target_problem(time)
    if problem is not None:
        raise ValueError(f"stitched light curve is not one star's: {problem}")
    cadence_days = float(np.median(np.diff(time))) if len(time) > 1 else 0.0

    return {
        "time": time,
        "flux": np.array(lc.flux.value, dtype=np.float64),
        "flux_err": np.array(lc.flux_err.value, dtype=np.float64),
        "quality": quality,
        "mission": mission,
        "cadence_days": cadence_days,
    }
