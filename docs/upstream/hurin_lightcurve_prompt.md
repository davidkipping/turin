# Task: hurin stitches a neighbouring star's light curves into the target's

**Who is asking.** `turin` (github.com/davidkipping/turin), hurin's GPU
successor, ported hurin's `lightcurve.py`. Running turin on the first ten
rows of a Kepler target list found that one target's light curve was two
stars interleaved. The cause is in the ported code, and hurin's current
`hurin/lightcurve.py` (`50c566a`) has it unchanged. This is a correctness
bug: the fit runs to completion on the wrong data and nothing flags it.

**Deliverable discipline.** As before: tests beside the existing ones, a
VERSIONS row, push to `https://github.com/davidkipping/hurin`, and say what
landed. turin has already fixed its copy (see "turin's fix").

## What happens

`_download` (and `_download_kepler_sc`, `_download_tess`) call
`lk.search_lightcurve(info["name"], ...)` and then download **every row**
of the result. A name search can return a neighbouring star's light curves
too, at distance 0, so both stars are stitched into one light curve.

Measured, live MAST, 2026-10-03:

```python
>>> r = lk.search_lightcurve("KOI-7592.01", author="Kepler", cadence="long")
>>> len(r), collections.Counter(str(x) for x in r.table["target_name"])
32, Counter({'kplr008423352': 18, 'kplr008423344': 14})
```

Both rows report `distance` 0.0 arcsec, so a distance cut cannot separate
them. The NASA Exoplanet Archive says KOI-7592 is **KIC 8423344**, the star
with **fewer** files:

```
select kepid from cumulative where kepoi_name='K07592.01'   ->  8423344
```

Stitching all 32 files gives 120,694 cadences with **54,276 repeated
timestamps** and 14 places where time runs backwards. The second star has
per-point errors of ~460-600 ppm against the KOI's ~125 ppm, so hurin's
4-sigma moving-median clip then removes ~8,000 points (7.5% of the light
curve, against 0.1-1% on every other target in the list), and what is left
is still two stars. Restricted to KIC 8423344 alone: 55,190 cadences,
monotonic, no repeats.

On the other nine targets in the list, the search returned one star, so
they were unaffected. The bug needs a neighbour close enough for the
name search to pick up, so it is target-dependent and silent when it hits.

## The ask

1. **Restrict every search to the host star's own catalogue number** before
   downloading: KIC for a KOI (`cumulative.kepid`, which hurin already
   queries in `_koi_to_kic`), TIC for a TOI (`toi.tid`). MAST's
   `target_name` encodes it (`kplr008423344` for Kepler, the bare TIC
   number for TESS SPOC). Apply it in all three download paths.
2. **Refuse rather than guess**: if the search spans several stars and the
   archive has no catalogue number, raise, naming the stars.
3. **Guard the stitch**: one star's cadences never repeat a timestamp, so a
   stitched light curve with repeated timestamps should raise rather than
   be cached.
4. **Invalidate bad caches**: a cache written before the fix holds the mixed
   light curve. hurin's cache for any affected target should be detected
   (the repeated-timestamp test does it) and re-downloaded.

## turin's fix, for reference

turin 0.1.26, `turin/data/lightcurve.py`: `_catalog_id` (archive lookup,
once per download), `_restrict_to_star` (filter by `target_name`, refuse
when ambiguous), `_mixed_target_problem` (repeated-timestamp test, used both
after stitching and when loading a cache). Tests in
`tests/test_cli_outputs.py` replay the KOI-7592.01 search offline.
