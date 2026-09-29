"""Ported verbatim from hurin 0.1.67 (hurin/ephemeris.py).

Pure NumPy / lightkurve / requests: no JAX, nothing to rewrite for the GPU.
Keep changes here minimal and upstream-traceable.
"""


import io
import re
import sys
import time

import numpy as np
import requests

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TAP_TIMEOUT = 30  # seconds per request
TAP_RETRY_DELAYS = (10, 30, 60, 120, 300)  # backoff between retries (~8.7 min total)

# Time system offsets
BKJD_OFFSET = 2454833.0  # BJD = BKJD + 2454833.0
BTJD_OFFSET = 2457000.0  # BJD = BTJD + 2457000.0

# In-memory caches for ephemeris queries
_eph_cache = {}
_other_eph_cache = {}


def _tap_get(query: str) -> requests.Response:
    """GET a TAP query with retries on transient failures.

    The archive's TAP service returns 503 for minutes at a time during
    maintenance windows; without retries one blip kills a batch run at
    startup. Retries cover server errors (5xx), timeouts, and connection
    drops; client errors (4xx) raise immediately.
    """
    for attempt, delay in enumerate((*TAP_RETRY_DELAYS, None)):
        try:
            resp = requests.get(
                TAP_URL, params={"query": query, "format": "csv"},
                timeout=TAP_TIMEOUT)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as err:
            last_err = err
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code < 500:
                raise
            last_err = err
        if delay is None:
            raise last_err
        print(f"[ephemeris] Exoplanet Archive unavailable ({last_err}); "
              f"retrying in {delay}s "
              f"(attempt {attempt + 2}/{len(TAP_RETRY_DELAYS) + 1})",
              file=sys.stderr, flush=True)
        time.sleep(delay)


def _safe_float(val: str) -> float:
    """Convert string to float, returning NaN for empty/missing values."""
    val = val.strip()
    if val == "" or val.lower() == "nan":
        return float("nan")
    return float(val)


def _parse_csv_rows(text: str) -> list[dict]:
    """Parse CSV response text into a list of dicts."""
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        values = line.split(",")
        rows.append(dict(zip(header, values)))
    return rows


def _parse_target_id(target: str) -> dict:
    """Extract target type, system number, and archive ID from a target string.

    Returns dict with keys: type ('koi'|'toi'), system (int), archive_id (str).
    """
    target = target.strip().upper()
    m_koi = re.match(r"KOI-(\d+)\.(\d+)", target)
    m_toi = re.match(r"TOI-(\d+)\.(\d+)", target)
    if m_koi:
        system = int(m_koi.group(1))
        archive_id = f"K{system:05d}.{m_koi.group(2)}"
        return {"type": "koi", "system": system, "archive_id": archive_id}
    elif m_toi:
        system = int(m_toi.group(1))
        planet = m_toi.group(2)
        archive_id = f"{system}.{planet}"
        return {"type": "toi", "system": system, "archive_id": archive_id}
    else:
        raise ValueError(f"Unknown target format: {target}")


def _koi_to_archive_name(koi_name: str) -> str:
    """Convert 'KOI-174.01' to 'K00174.01' for the Exoplanet Archive."""
    m = re.match(r"KOI-(\d+)\.(\d+)", koi_name)
    if not m:
        raise ValueError(f"Invalid KOI name: {koi_name}")
    return f"K{int(m.group(1)):05d}.{m.group(2)}"


def _toi_to_archive_number(toi_name: str) -> str:
    """Convert 'TOI-700.01' to '700.01' for the Exoplanet Archive."""
    m = re.match(r"TOI-(\d+\.\d+)", toi_name)
    if not m:
        raise ValueError(f"Invalid TOI name: {toi_name}")
    return m.group(1)


def get_ephemeris(target: str) -> dict:
    """Query the NASA Exoplanet Archive for transit ephemeris.

    Returns dict with keys:
        period: orbital period in days
        epoch: transit epoch in the same time system as lightkurve
                (BKJD for Kepler, BTJD for TESS)
        depth: transit depth in ppm
        duration: transit duration in hours
    """
    target = target.strip().upper()
    if target in _eph_cache:
        return _eph_cache[target]

    if target.startswith("KOI"):
        result = _query_koi(target)
    elif target.startswith("TOI"):
        result = _query_toi(target)
    else:
        raise ValueError(f"Unknown target type: {target}")

    _eph_cache[target] = result
    return result


def _query_koi(target: str) -> dict:
    """Query the cumulative KOI table."""
    archive_name = _koi_to_archive_name(target)
    query = (
        f"select kepoi_name,koi_period,koi_time0bk,koi_depth,koi_duration,koi_ingress "
        f"from cumulative where kepoi_name='{archive_name}'"
    )
    resp = _tap_get(query)

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        raise ValueError(f"No ephemeris found for {target} (archive name: {archive_name})")

    header = lines[0].split(",")
    values = lines[1].split(",")
    row = dict(zip(header, values))

    ingress = _safe_float(row.get("koi_ingress", ""))
    return {
        "period": float(row["koi_period"]),
        "epoch": float(row["koi_time0bk"]),  # Already in BKJD
        "depth": float(row["koi_depth"]),
        "duration": float(row["koi_duration"]),
        "ingress": ingress if not np.isnan(ingress) else None,  # hours
    }


def _query_toi(target: str) -> dict:
    """Query the TOI table."""
    toi_number = _toi_to_archive_number(target)
    query = (
        f"select toi,pl_orbper,pl_tranmid,pl_trandep,pl_trandurh "
        f"from toi where toi={toi_number}"
    )
    resp = _tap_get(query)

    lines = resp.text.strip().split("\n")
    if len(lines) < 2:
        raise ValueError(f"No ephemeris found for {target}")

    header = lines[0].split(",")
    values = lines[1].split(",")
    row = dict(zip(header, values))

    # Convert epoch from BJD to BTJD (lightkurve's TESS time system)
    epoch_bjd = float(row["pl_tranmid"])
    epoch_btjd = epoch_bjd - BTJD_OFFSET

    return {
        "period": float(row["pl_orbper"]),
        "epoch": epoch_btjd,
        "depth": float(row["pl_trandep"]),
        "duration": float(row["pl_trandurh"]),
        "ingress": None,  # Not available from TOI table
    }


def get_other_planet_ephemerides(target: str) -> list[dict]:
    """Query the NASA Exoplanet Archive for all OTHER planets in the same system.

    Returns a list of dicts with keys: period, epoch, depth, duration.
    Planets with missing period, epoch, or duration are silently skipped.
    """
    target = target.strip().upper()
    if target in _other_eph_cache:
        return _other_eph_cache[target]

    info = _parse_target_id(target)

    if info["type"] == "koi":
        result = _query_other_koi(info)
    else:
        result = _query_other_toi(info)

    _other_eph_cache[target] = result
    return result


def _query_other_koi(info: dict) -> list[dict]:
    """Query other KOI planets in the same system."""
    system = info["system"]
    archive_id = info["archive_id"]
    prefix = f"K{system:05d}.%"
    query = (
        f"select kepoi_name,koi_period,koi_time0bk,koi_depth,koi_duration "
        f"from cumulative where kepoi_name like '{prefix}' "
        f"and kepoi_name!='{archive_id}'"
    )
    resp = _tap_get(query)

    results = []
    for row in _parse_csv_rows(resp.text):
        period = _safe_float(row["koi_period"])
        epoch = _safe_float(row["koi_time0bk"])
        depth = _safe_float(row["koi_depth"])
        duration = _safe_float(row["koi_duration"])
        if np.isnan(period) or np.isnan(epoch) or np.isnan(duration):
            continue
        results.append({
            "period": period,
            "epoch": epoch,  # Already in BKJD
            "depth": depth,
            "duration": duration,
        })
    return results


def _query_other_toi(info: dict) -> list[dict]:
    """Query other TOI planets in the same system."""
    system = info["system"]
    archive_id = info["archive_id"]
    query = (
        f"select toi,pl_orbper,pl_tranmid,pl_trandep,pl_trandurh "
        f"from toi where toi>={system} and toi<{system + 1} "
        f"and toi!={archive_id}"
    )
    resp = _tap_get(query)

    results = []
    for row in _parse_csv_rows(resp.text):
        period = _safe_float(row["pl_orbper"])
        epoch_bjd = _safe_float(row["pl_tranmid"])
        depth = _safe_float(row["pl_trandep"])
        duration = _safe_float(row["pl_trandurh"])
        if np.isnan(period) or np.isnan(epoch_bjd) or np.isnan(duration):
            continue
        results.append({
            "period": period,
            "epoch": epoch_bjd - BTJD_OFFSET,
            "depth": depth,
            "duration": duration,
        })
    return results
