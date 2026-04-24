"""Google Earth Engine — nighttime lights lookup.

Importable module: `get_ntl_summary(iso3, year)` returns a short string for
use in LLM prompts, or None if unavailable. GEE is lazy-initialized on first
call so importing the module is cheap.

Run as a script to batch-annotate `reviewed_flagged.csv` with NTL radiance
and write `flags_with_NTL.csv` (preserves the original workflow).
"""

import logging

log = logging.getLogger(__name__)

_ee = None
_initialized = False
_cache: dict[tuple[str, int], str | None] = {}


def _init_gee() -> bool:
    """Lazy-initialize Earth Engine. Returns True on success."""
    global _ee, _initialized
    if _initialized:
        return _ee is not None
    _initialized = True
    try:
        import ee
        try:
            ee.Initialize()
        except Exception:
            ee.Authenticate(auth_mode="localhost")
            ee.Initialize(project="ntl-data-493519")
        _ee = ee
        return True
    except Exception as e:
        log.warning("Earth Engine init failed: %s", e)
        _ee = None
        return False


def get_ntl_intensity(iso_code: str, year: int) -> float | None:
    """Mean nighttime-light radiance for one (country, year). None if unavailable.

    Uses VIIRS for 2014+, DMSP for 1992–2013.
    """
    if year < 1992 or not _init_gee():
        return None
    ee = _ee
    try:
        countries = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM0")
        country_geom = countries.filter(ee.Filter.eq("shapeGroup", iso_code)).geometry()
        start, end = f"{year}-01-01", f"{year}-12-31"
        if year >= 2014:
            col = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            band = "avg_rad"
        else:
            col = ee.ImageCollection("NOAA/DMSP-OLS/NIGHTTIME_LIGHTS")
            band = "stable_lights"
        img = col.filterDate(start, end).select(band).mean()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=country_geom,
            scale=10000,
            maxPixels=int(1e13),
            bestEffort=True,
        )
        return stats.get(band).getInfo()
    except Exception as e:
        log.warning("NTL fetch failed for %s %d: %s", iso_code, year, e)
        return None


def get_ntl_summary(iso3: str, year: int) -> str | None:
    """Short summary string for LLM prompts, or None if unavailable. Cached.

    Includes year-over-year % change when the prior year uses the same sensor
    (VIIRS or DMSP). The 2013→2014 boundary is skipped since the two sensors
    aren't directly comparable.
    """
    key = (iso3, year)
    if key in _cache:
        return _cache[key]
    val = get_ntl_intensity(iso3, year)
    if val is None:
        _cache[key] = None
        return None
    source = "VIIRS" if year >= 2014 else "DMSP"

    prior_val = None
    same_sensor = year - 1 >= 1992 and (year - 1 >= 2014) == (year >= 2014)
    if same_sensor:
        prior_val = get_ntl_intensity(iso3, year - 1)

    if prior_val is not None and prior_val != 0:
        delta = (val - prior_val) / prior_val * 100
        summary = f"mean radiance {val:.2f} ({source}); {delta:+.1f}% vs {year - 1}"
    else:
        summary = f"mean radiance {val:.2f} ({source})"

    _cache[key] = summary
    return summary


if __name__ == "__main__":
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _init_gee()
    print("Earth Engine Initialized Successfully!")

    df = pd.read_csv("reviewed_flagged.csv")
    print("Extracting satellite data. This may take a moment depending on your dataset size...")

    MAX_WORKERS = 10
    ntl_values = [None] * len(df)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for index, row in df.iterrows():
            iso_code = row["country_code"]
            year = row["year"]
            future = executor.submit(get_ntl_intensity, iso_code, year)
            futures[future] = (index, iso_code, year)

        for future in as_completed(futures):
            index, iso_code, year = futures[future]
            result = future.result()
            ntl_values[index] = result
            print(f"{iso_code} {year}: {result}")

    df["NTL_Radiance"] = ntl_values
    df.to_csv("flags_with_NTL.csv", index=False)
    print("Finished! Saved as 'flags_with_NTL.csv'")
