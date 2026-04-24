"""Phase B — two-pass pair discovery.

For each candidate pair of indicators, fetches the time-series across a
diverse 30-country reference panel, computes a per-country Pearson r, and
keeps pairs that show a genuinely reproducible cross-country correlation.
Survivors land in the ``useful_pairs`` table, which the worker reads when
seeding jobs — replacing the combinatorial explosion from
``itertools.combinations``.

Run:

    uv run python pair_discovery.py             # over the hardcoded INDICATORS
    uv run python pair_discovery.py --dry-run   # print but don't write
"""

import argparse
import itertools
import logging
import time

import numpy as np

import database as db
from data_ingestion import INDICATORS, fetch_indicator

logging.getLogger("shelved_cache").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


REFERENCE_PANEL: list[str] = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "BRA", "IND", "CAN", "RUS",
    "KOR", "AUS", "ESP", "MEX", "IDN", "NLD", "TUR", "SAU", "POL", "SWE",
    "NOR", "ARG", "ZAF", "NGA", "EGY", "THA", "VNM", "PHL", "CHL", "COL",
]

# Keep pairs whose global r is strong and agreed on by most of the panel.
MIN_GLOBAL_R = 0.5
MIN_COUNTRY_R = 0.4
MIN_SUPPORT_RATIO = 0.6
MIN_OVERLAPPING_YEARS = 10

SLEEP_BETWEEN_API_CALLS = 0.5


def _fetch_cached(country_code: str, indicator_code: str) -> list[tuple[int, float]]:
    """Shares the worker's SQLite cache. Writes back API results and negative
    fetches so a future worker run doesn't repeat the call."""
    if db.has_indicator_data(country_code, indicator_code):
        return db.get_indicator_data(country_code, indicator_code)
    if db.has_fetch_log(country_code, indicator_code) == 0:
        return []
    df = fetch_indicator(country_code, indicator_code)
    rows = list(zip(df["year"].tolist(), df["value"].tolist()))
    if rows:
        db.store_indicator_data(country_code, indicator_code, rows)
    db.record_fetch_attempt(country_code, indicator_code, had_data=bool(rows))
    time.sleep(SLEEP_BETWEEN_API_CALLS)
    return rows


def _prefetch_panel(
    indicator_codes: list[str], panel: list[str]
) -> dict[tuple[str, str], dict[int, float]]:
    """Load every (panel_country, indicator) series once up front."""
    series: dict[tuple[str, str], dict[int, float]] = {}
    total = len(indicator_codes) * len(panel)
    done = 0
    for ind in indicator_codes:
        for country in panel:
            rows = _fetch_cached(country, ind)
            series[(country, ind)] = {int(y): float(v) for y, v in rows}
            done += 1
            if done % 50 == 0:
                log.info("Prefetch %d/%d (%s/%s: %d years)",
                         done, total, country, ind, len(rows))
    return series


def _country_pearson(
    s1: dict[int, float], s2: dict[int, float], min_years: int
) -> float | None:
    """Pearson r over the overlapping-years intersection of two series.
    Returns None if fewer than ``min_years`` overlap or either series is
    constant (variance 0)."""
    overlap_years = sorted(set(s1) & set(s2))
    if len(overlap_years) < min_years:
        return None
    a = np.array([s1[y] for y in overlap_years])
    b = np.array([s2[y] for y in overlap_years])
    if a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(r):
        return None
    return r


def classify_pair(
    ind_a: str,
    ind_b: str,
    series: dict[tuple[str, str], dict[int, float]],
    panel: list[str],
) -> dict:
    """Compute the per-country r distribution for one pair across the panel."""
    per_country: list[tuple[str, float]] = []
    for country in panel:
        r = _country_pearson(
            series.get((country, ind_a), {}),
            series.get((country, ind_b), {}),
            MIN_OVERLAPPING_YEARS,
        )
        if r is not None:
            per_country.append((country, r))

    if not per_country:
        return {
            "ind_a": ind_a, "ind_b": ind_b,
            "global_r": 0.0, "support_count": 0,
            "panel_size": len(panel),
            "countries_with_data": 0,
            "passes": False,
        }

    rs = np.array([r for _, r in per_country])
    global_r = float(np.median(rs))
    sign = np.sign(global_r) if global_r != 0 else 0
    support_count = int(((np.abs(rs) >= MIN_COUNTRY_R) & (np.sign(rs) == sign)).sum())

    passes = (
        abs(global_r) >= MIN_GLOBAL_R
        and support_count >= int(MIN_SUPPORT_RATIO * len(panel))
    )
    return {
        "ind_a": ind_a,
        "ind_b": ind_b,
        "global_r": global_r,
        "support_count": support_count,
        "panel_size": len(panel),
        "countries_with_data": len(per_country),
        "passes": passes,
    }


def discover(
    indicator_codes: list[str],
    panel: list[str] = REFERENCE_PANEL,
) -> list[dict]:
    """Prefetch panel data, then classify every combinatorial pair."""
    indicator_codes = sorted(set(indicator_codes))
    log.info("Pair discovery: %d indicators × %d panel countries",
             len(indicator_codes), len(panel))
    series = _prefetch_panel(indicator_codes, panel)

    pairs = list(itertools.combinations(indicator_codes, 2))
    log.info("Classifying %d pairs...", len(pairs))
    return [classify_pair(a, b, series, panel) for a, b in pairs]


def _format_row(result: dict) -> str:
    marker = "KEEP" if result["passes"] else "drop"
    return (
        f"  [{marker}] r={result['global_r']:+.3f}  "
        f"support={result['support_count']:>2}/{result['panel_size']}  "
        f"(data in {result['countries_with_data']:>2} countries)  "
        f"{result['ind_a']}  ×  {result['ind_b']}"
    )


def write_survivors(results: list[dict]) -> int:
    """Clear the useful_pairs table and persist the passing results.
    Returns the number of surviving pairs."""
    db.clear_useful_pairs()
    kept = [r for r in results if r["passes"]]
    for r in kept:
        db.upsert_useful_pair(
            r["ind_a"], r["ind_b"],
            global_r=r["global_r"],
            support_count=r["support_count"],
            panel_size=r["panel_size"],
        )
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B pair discovery — find indicator pairs worth running jobs on",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the ranked list without touching the useful_pairs table",
    )
    parser.add_argument(
        "--all-useful", action="store_true",
        help="Use every indicator with status='useful' from the DB, not just the hardcoded set",
    )
    args = parser.parse_args()

    db.init_db()

    if args.all_useful:
        with db._connect() as conn:
            indicator_codes = [
                row["code"]
                for row in conn.execute(
                    "SELECT code FROM indicators WHERE status='useful' ORDER BY code"
                )
            ]
    else:
        indicator_codes = list(INDICATORS.keys())

    if len(indicator_codes) < 2:
        log.error("Need at least 2 indicators to form pairs; got %d", len(indicator_codes))
        return

    results = discover(indicator_codes)
    results.sort(key=lambda r: (-int(r["passes"]), -abs(r["global_r"])))

    print("\n=== Pair discovery results ===")
    for r in results:
        print(_format_row(r))
    kept = sum(1 for r in results if r["passes"])
    print(f"\n{kept} / {len(results)} pairs passed  "
          f"(|global_r|>={MIN_GLOBAL_R}, "
          f"support>={int(MIN_SUPPORT_RATIO * len(REFERENCE_PANEL))}/{len(REFERENCE_PANEL)})")

    if args.dry_run:
        print("\n--dry-run: not writing to useful_pairs")
        return

    n = write_survivors(results)
    print(f"\nWrote {n} rows to useful_pairs.")


if __name__ == "__main__":
    main()
