"""UN Comtrade lookup — annual total trade flows for a country/year.

Used to enrich LLM integrity assessments with trade context. Uses the free
public preview endpoint (no API key required).
"""

import logging

import requests

log = logging.getLogger(__name__)

_REPORTERS_URL = "https://comtradeapi.un.org/files/v1/app/reference/Reporters.json"
_DATA_URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

_reporters: dict[str, str] | None = None
_summary_cache: dict[tuple[str, int], str | None] = {}


def _load_reporters() -> dict[str, str]:
    """Build ISO3 → Comtrade reporter code mapping (cached module-level)."""
    global _reporters
    if _reporters is not None:
        return _reporters
    try:
        r = requests.get(_REPORTERS_URL, timeout=10)
        r.raise_for_status()
        data = r.json().get("results", [])
        _reporters = {}
        for item in data:
            iso3 = item.get("reporterCodeIsoAlpha3")
            code = item.get("reporterCode")
            if not iso3 or not code or item.get("entryExpiredDate"):
                continue
            _reporters[iso3] = str(code)
    except Exception as e:
        log.warning("Comtrade reporter list fetch failed: %s", e)
        _reporters = {}
    return _reporters


def get_trade_summary(iso3: str, year: int) -> str | None:
    """Return a one-line summary of total exports/imports/balance, or None.

    Caches per (iso3, year). Network failures degrade to None.
    """
    key = (iso3, year)
    if key in _summary_cache:
        return _summary_cache[key]

    reporters = _load_reporters()
    reporter_code = reporters.get(iso3)
    if reporter_code is None:
        _summary_cache[key] = None
        return None

    try:
        r = requests.get(
            _DATA_URL,
            params={
                "reporterCode": reporter_code,
                "period": year,
                "partnerCode": 0,
                "cmdCode": "TOTAL",
                "flowCode": "M,X",
            },
            timeout=15,
        )
        r.raise_for_status()
        rows = r.json().get("data", []) or []
    except Exception as e:
        log.warning("Comtrade fetch failed for %s %d: %s", iso3, year, e)
        _summary_cache[key] = None
        return None

    totals: dict[str, float] = {}
    for row in rows:
        # Comtrade splits rows by mode-of-transport; motCode=0 is the all-modes aggregate.
        if str(row.get("motCode")) != "0":
            continue
        flow = row.get("flowCode")
        val = row.get("primaryValue")
        if flow in ("M", "X") and val is not None:
            totals[flow] = float(val)

    parts = []
    if "X" in totals:
        parts.append(f"total exports ${totals['X']/1e9:.2f}B")
    if "M" in totals:
        parts.append(f"total imports ${totals['M']/1e9:.2f}B")
    if "X" in totals and "M" in totals:
        parts.append(f"trade balance ${(totals['X']-totals['M'])/1e9:.2f}B")

    summary = "; ".join(parts) if parts else None
    _summary_cache[key] = summary
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for cc, yr in [("USA", 2020), ("BRA", 2019), ("CHN", 2018)]:
        print(f"{cc} {yr}: {get_trade_summary(cc, yr)}")
