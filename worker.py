"""Background worker: fetches World Bank data, runs correlation analysis, gets LLM assessments."""

import logging

# Suppress noisy "Key not in persistent cache" warnings from shelved_cache
# (triggered when wbdata loads and expires its persistent cache at import time)
logging.getLogger("shelved_cache").setLevel(logging.ERROR)

import time

import pandas as pd

import database as db
from correlation_analysis import analyze_correlation, build_peer_change_frame
from data_ingestion import INDICATORS, fetch_indicator, get_country_list, get_indicators_by_topic
from llm_integrity import assess_integrity

# Per-pair peer frames are expensive (one merge over ~200 countries). The
# scheduler in `run()` processes jobs ordered by (ind1, ind2, country), so
# caching the last-built frame covers >99% of the sequential workload
# without growing memory. Cleared whenever the pair changes.
_PEER_FRAME_CACHE: dict[tuple[str, str], pd.DataFrame] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MIN_OVERLAPPING_YEARS = 5
SLEEP_BETWEEN_API_CALLS = 0.5
SLEEP_WHEN_IDLE = 300  # 5 minutes

# Usefulness gating: once an indicator has been tried against at least
# MIN_COUNTRIES_BEFORE_JUDGING countries, if more than NOT_USEFUL_RATIO of
# those attempts returned no data, the indicator is marked not useful and
# all pending jobs referencing it are skipped.
MIN_COUNTRIES_BEFORE_JUDGING = 20
NOT_USEFUL_RATIO = 0.5


def _check_indicator_usefulness(indicator_code: str) -> None:
    """Mark indicator as not useful if the empty-fetch ratio crosses threshold."""
    if not db.is_indicator_useful(indicator_code):
        return
    stats = db.get_indicator_fetch_stats(indicator_code)
    tried, empty = stats["tried"], stats["empty"]
    if tried < MIN_COUNTRIES_BEFORE_JUDGING:
        return
    if empty / tried > NOT_USEFUL_RATIO:
        skipped = db.mark_indicator_not_useful(indicator_code)
        log.warning(
            "Indicator %s marked not useful: %d/%d countries returned no data (%d pending jobs skipped)",
            indicator_code, empty, tried, skipped,
        )


def _fetch_and_cache(country_code: str, indicator_code: str) -> list[tuple[int, float]]:
    """Fetch indicator data from API if not cached, return (year, value) list."""
    if db.has_indicator_data(country_code, indicator_code):
        return db.get_indicator_data(country_code, indicator_code)

    # Negative cache: we've tried this (country, indicator) before and got nothing.
    if db.has_fetch_log(country_code, indicator_code) == 0:
        return []

    log.info("  Fetching %s / %s from API", country_code, indicator_code)
    df = fetch_indicator(country_code, indicator_code)
    rows = list(zip(df["year"].tolist(), df["value"].tolist()))
    if rows:
        db.store_indicator_data(country_code, indicator_code, rows)
    db.record_fetch_attempt(country_code, indicator_code, had_data=bool(rows))
    _check_indicator_usefulness(indicator_code)
    time.sleep(SLEEP_BETWEEN_API_CALLS)
    return rows


def _build_merged_df(
    country_code: str, data_1: list[tuple[int, float]], data_2: list[tuple[int, float]]
) -> pd.DataFrame | None:
    """Merge two indicator series on year. Returns None if too few overlapping years."""
    df1 = pd.DataFrame(data_1, columns=["year", "value_1"])
    df2 = pd.DataFrame(data_2, columns=["year", "value_2"])
    merged = df1.merge(df2, on="year")
    if len(merged) < MIN_OVERLAPPING_YEARS:
        return None
    merged["country_id"] = country_code
    return merged.sort_values("year").reset_index(drop=True)


def _countries_for_pair(ind1: str, ind2: str) -> list[str]:
    """Every country that has at least one pending/in-flight job for this
    pair. Used to decide who's in the peer frame."""
    with db._connect() as conn:
        rows = conn.execute(
            """SELECT DISTINCT country_code FROM analysis_jobs
               WHERE indicator_1=? AND indicator_2=?""",
            (ind1, ind2),
        ).fetchall()
    return [r["country_code"] for r in rows]


def _ensure_peer_frame(ind1: str, ind2: str) -> pd.DataFrame:
    """Return (and cache) the peer frame for a pair. On first call for a
    new pair, drops any stale cache, prefetches both indicators for every
    country that has a job for this pair so the peer frame is complete,
    then builds the adjusted YoY-product frame."""
    key = (ind1, ind2)
    if key in _PEER_FRAME_CACHE:
        return _PEER_FRAME_CACHE[key]

    # Drop the previous pair's frame — only one pair is "active" at a time.
    _PEER_FRAME_CACHE.clear()

    countries = _countries_for_pair(ind1, ind2)
    log.info(
        "Peer frame for %s × %s: prefetching %d countries",
        ind1, ind2, len(countries),
    )

    rows: list[pd.DataFrame] = []
    for cc in countries:
        data_1 = _fetch_and_cache(cc, ind1)
        data_2 = _fetch_and_cache(cc, ind2)
        merged = _build_merged_df(cc, data_1, data_2)
        if merged is not None:
            rows.append(merged)

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        peer_frame = build_peer_change_frame(combined)
    else:
        peer_frame = pd.DataFrame(columns=["country_id", "year", "change_product"])

    log.info(
        "Peer frame ready: %d countries with data, %d (country, year) rows",
        peer_frame["country_id"].nunique() if not peer_frame.empty else 0,
        len(peer_frame),
    )
    _PEER_FRAME_CACHE[key] = peer_frame
    return peer_frame


def _get_indicator_name(code: str) -> str:
    """Look up indicator name from DB, fall back to INDICATORS dict or code."""
    with db._connect() as conn:
        row = conn.execute("SELECT name FROM indicators WHERE code=?", (code,)).fetchone()
        if row:
            return row["name"]
    return INDICATORS.get(code, code)


def _process_job(job: dict) -> None:
    """Run a single analysis job end-to-end."""
    job_id = job["id"]
    country = job["country_code"]
    ind1 = job["indicator_1"]
    ind2 = job["indicator_2"]
    ind1_name = _get_indicator_name(ind1)
    ind2_name = _get_indicator_name(ind2)

    # Skip if either indicator has been marked not useful since seeding.
    if not db.is_indicator_useful(ind1) or not db.is_indicator_useful(ind2):
        db.update_job_status(
            job_id, "skipped", error_message="indicator marked not useful"
        )
        log.info("Job %d: skipped — indicator marked not useful", job_id)
        return

    log.info("Job %d: %s — %s vs %s", job_id, country, ind1, ind2)

    # Fetch data
    db.update_job_status(job_id, "fetching")
    db.update_heartbeat("working", job_id)

    data_1 = _fetch_and_cache(country, ind1)
    data_2 = _fetch_and_cache(country, ind2)

    if not data_1 or not data_2:
        db.update_job_status(job_id, "done", total_years=0, flagged_count=0)
        log.info("  Skipped — no data for one or both indicators")
        return

    merged = _build_merged_df(country, data_1, data_2)
    if merged is None:
        db.update_job_status(job_id, "done", total_years=0, flagged_count=0)
        log.info("  Skipped — fewer than %d overlapping years", MIN_OVERLAPPING_YEARS)
        return

    # Correlation analysis with peer-year filter (Phase C)
    db.update_job_status(job_id, "analyzing")
    peer_frame = _ensure_peer_frame(ind1, ind2)
    result = analyze_correlation(merged, peer_frame=peer_frame)

    flagged = result[result["integrity_flag"]].copy()
    pearson_r = result["expected_correlation"].iloc[0] if len(result) > 0 else None

    # Store flagged items
    flagged_ids = []
    for _, row in flagged.iterrows():
        peer_z = row.get("peer_z", float("nan"))
        shock = row.get("global_shock_fraction", float("nan"))
        item_id = db.store_flagged_item(
            job_id=job_id,
            country_code=country,
            indicator_1=ind1,
            indicator_2=ind2,
            year=int(row["year"]),
            value_1=float(row["value_1"]),
            value_2=float(row["value_2"]),
            expected_correlation=float(row["expected_correlation"]),
            statistical_confidence=float(row["confidence_score"]),
            peer_z=None if pd.isna(peer_z) else float(peer_z),
            global_shock_fraction=None if pd.isna(shock) else float(shock),
        )
        flagged_ids.append(item_id)

    # LLM assessment
    if flagged_ids:
        db.update_job_status(job_id, "assessing")
        for item_id, (_, row) in zip(flagged_ids, flagged.iterrows()):
            try:
                assessment = assess_integrity(
                    {
                        "country_id": country,
                        "year": int(row["year"]),
                        "value_1": float(row["value_1"]),
                        "value_2": float(row["value_2"]),
                        "expected_correlation": float(row["expected_correlation"]),
                        "indicator_1": ind1,
                        "indicator_2": ind2,
                    },
                    indicator_1_name=ind1_name,
                    indicator_2_name=ind2_name,
                )
                db.update_flagged_item_assessment(
                    item_id,
                    is_anomaly=assessment.is_anomaly,
                    confidence=assessment.confidence_score,
                    explanation=assessment.explanation,
                )
            except Exception as e:
                log.warning("  LLM assessment failed for item %d: %s", item_id, e)

    db.update_job_status(
        job_id,
        "done",
        pearson_r=pearson_r,
        total_years=len(merged),
        flagged_count=len(flagged_ids),
    )
    log.info("  Done — %d years, %d flagged", len(merged), len(flagged_ids))


def seed_jobs(pairs: list[tuple[str, str]], country_codes: list[str]) -> int:
    """Create pending jobs for each useful pair × country. Returns count."""
    count = 0
    log.info(
        "Seeding jobs: %d useful pairs × %d countries = %d jobs",
        len(pairs), len(country_codes), len(pairs) * len(country_codes),
    )
    for ind1, ind2 in pairs:
        for country in country_codes:
            db.get_or_create_job(country, ind1, ind2)
            count += 1
    return count


def run(use_topics: bool = False, reseed: bool = False) -> None:
    """Main worker loop."""
    db.init_db()

    if reseed:
        counts = db.wipe_job_pipeline()
        log.warning(
            "Reseed requested — wiped pipeline: %d reviews, %d flagged_items, %d analysis_jobs",
            counts["reviews"], counts["flagged_items"], counts["analysis_jobs"],
        )

    if db.has_any_jobs():
        log.info("Jobs already seeded — skipping indicator/country/job seeding (pass --reseed to force)")
    else:
        if use_topics:
            log.info("Discovering indicators by topic...")
            try:
                topic_indicators = get_indicators_by_topic([3])  # Economy & Growth
                log.info("Found %d indicators from World Bank topics", len(topic_indicators))
                db.upsert_indicators(topic_indicators, topic="Economy & Growth")
            except Exception as e:
                log.warning("Topic discovery failed, using hardcoded indicators: %s", e)
            db.upsert_indicators(INDICATORS, topic="Hardcoded")
        else:
            db.upsert_indicators(INDICATORS, topic="Hardcoded")

        # Pair selection: only run jobs for pairs that pair_discovery validated.
        pairs = db.get_useful_pairs()
        if not pairs:
            log.error(
                "useful_pairs is empty — run `uv run python pair_discovery.py` "
                "before starting the worker. Skipping job seeding."
            )
            return

        log.info("Fetching country list...")
        countries = get_country_list()
        country_codes = list(countries.keys())
        log.info("Found %d countries", len(country_codes))

        seed_jobs(pairs, country_codes)

    # Main loop
    log.info("Starting main loop...")
    while True:
        job = db.get_next_pending_job()
        if job is None:
            log.info("No pending jobs. Sleeping %d seconds...", SLEEP_WHEN_IDLE)
            db.update_heartbeat("idle")
            time.sleep(SLEEP_WHEN_IDLE)
            continue

        try:
            _process_job(job)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error("Job %d failed: %s", job["id"], e, exc_info=True)
            db.update_job_status(job["id"], "error", error_message=str(e))

        db.update_heartbeat("working")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Background integrity monitor worker")
    parser.add_argument(
        "--topics", action="store_true",
        help="Discover indicators from World Bank topics (many more pairs, much slower)",
    )
    parser.add_argument(
        "--reseed", action="store_true",
        help="Force re-seeding of indicators and jobs even if already present in the database",
    )
    args = parser.parse_args()

    try:
        run(use_topics=args.topics, reseed=args.reseed)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        db.update_heartbeat("stopped")
