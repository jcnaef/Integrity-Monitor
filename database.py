"""SQLite database layer for the Cross-Indicator Integrity Monitor."""

import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).parent / "integrity_monitor.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS indicators (
    code   TEXT PRIMARY KEY,
    name   TEXT NOT NULL,
    topic  TEXT,
    status TEXT NOT NULL DEFAULT 'useful'
);

CREATE TABLE IF NOT EXISTS indicator_data (
    country_code   TEXT NOT NULL,
    indicator_code TEXT NOT NULL REFERENCES indicators(code),
    year           INTEGER NOT NULL,
    value          REAL NOT NULL,
    fetched_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (country_code, indicator_code, year)
);

CREATE TABLE IF NOT EXISTS indicator_fetch_log (
    country_code   TEXT NOT NULL,
    indicator_code TEXT NOT NULL,
    had_data       INTEGER NOT NULL,
    fetched_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (country_code, indicator_code)
);

CREATE TABLE IF NOT EXISTS analysis_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code  TEXT NOT NULL,
    indicator_1   TEXT NOT NULL,
    indicator_2   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',
    pearson_r     REAL,
    total_years   INTEGER,
    flagged_count INTEGER,
    error_message TEXT,
    started_at    TEXT,
    completed_at  TEXT,
    UNIQUE (country_code, indicator_1, indicator_2)
);

CREATE TABLE IF NOT EXISTS flagged_items (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                 INTEGER NOT NULL REFERENCES analysis_jobs(id),
    country_code           TEXT NOT NULL,
    indicator_1            TEXT NOT NULL,
    indicator_2            TEXT NOT NULL,
    year                   INTEGER NOT NULL,
    value_1                REAL NOT NULL,
    value_2                REAL NOT NULL,
    expected_correlation   REAL NOT NULL,
    statistical_confidence REAL NOT NULL,
    peer_z                 REAL,
    global_shock_fraction  REAL,
    llm_is_anomaly         INTEGER,
    llm_confidence         REAL,
    llm_explanation        TEXT,
    assessed_at            TEXT,
    exported_at            TEXT,
    UNIQUE (country_code, indicator_1, indicator_2, year)
);

CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flagged_item_id INTEGER NOT NULL UNIQUE REFERENCES flagged_items(id),
    status          TEXT NOT NULL,
    note            TEXT DEFAULT '',
    reviewed_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS worker_status (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    last_heartbeat TEXT,
    current_job_id INTEGER,
    state          TEXT DEFAULT 'idle'
);

CREATE TABLE IF NOT EXISTS regime_data (
    country_code TEXT NOT NULL,
    year         INTEGER NOT NULL,
    regime_type  INTEGER NOT NULL,
    regime_label TEXT NOT NULL,
    PRIMARY KEY (country_code, year)
);
CREATE INDEX IF NOT EXISTS ix_regime_country_year
    ON regime_data (country_code, year);

CREATE TABLE IF NOT EXISTS useful_pairs (
    indicator_1   TEXT NOT NULL,
    indicator_2   TEXT NOT NULL,
    global_r      REAL NOT NULL,
    support_count INTEGER NOT NULL,
    panel_size    INTEGER NOT NULL,
    discovered_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (indicator_1, indicator_2)
);
"""

REGIME_LABELS = {
    0: "Closed Autocracy",
    1: "Electoral Autocracy",
    2: "Electoral Democracy",
    3: "Liberal Democracy",
}


def _connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply lightweight ALTER TABLE migrations for existing databases."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(indicators)").fetchall()}
    if "status" not in cols:
        conn.execute(
            "ALTER TABLE indicators ADD COLUMN status TEXT NOT NULL DEFAULT 'useful'"
        )

    flagged_cols = {
        r["name"] for r in conn.execute("PRAGMA table_info(flagged_items)").fetchall()
    }
    if "peer_z" not in flagged_cols:
        conn.execute("ALTER TABLE flagged_items ADD COLUMN peer_z REAL")
    if "global_shock_fraction" not in flagged_cols:
        conn.execute("ALTER TABLE flagged_items ADD COLUMN global_shock_fraction REAL")


def init_db(db_path: Path = DEFAULT_DB) -> None:
    """Create all tables and seed worker_status row."""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.execute(
            "INSERT OR IGNORE INTO worker_status (id, state) VALUES (1, 'idle')"
        )


# ── Indicators ──────────────────────────────────────────────────────────


def upsert_indicators(
    indicators: dict[str, str],
    topic: str | None = None,
    db_path: Path = DEFAULT_DB,
) -> None:
    """Insert or update indicator codes and names."""
    with _connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO indicators (code, name, topic)
               VALUES (?, ?, ?)
               ON CONFLICT(code) DO UPDATE SET name=excluded.name, topic=excluded.topic""",
            [(code, name, topic) for code, name in indicators.items()],
        )


# ── Indicator Data ──────────────────────────────────────────────────────


def has_indicator_data(
    country_code: str, indicator_code: str, db_path: Path = DEFAULT_DB
) -> bool:
    """Check whether we already have cached data for this country+indicator."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM indicator_data WHERE country_code=? AND indicator_code=? LIMIT 1",
            (country_code, indicator_code),
        ).fetchone()
        return row is not None


def store_indicator_data(
    country_code: str,
    indicator_code: str,
    rows: list[tuple[int, float]],
    db_path: Path = DEFAULT_DB,
) -> None:
    """Store (year, value) rows for a country+indicator. Upserts on conflict."""
    with _connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO indicator_data (country_code, indicator_code, year, value)
               VALUES (?, ?, ?, ?)
               ON CONFLICT DO UPDATE SET value=excluded.value, fetched_at=datetime('now')""",
            [(country_code, indicator_code, year, value) for year, value in rows],
        )


def get_indicator_data(
    country_code: str, indicator_code: str, db_path: Path = DEFAULT_DB
) -> list[tuple[int, float]]:
    """Return cached (year, value) pairs sorted by year."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT year, value FROM indicator_data WHERE country_code=? AND indicator_code=? ORDER BY year",
            (country_code, indicator_code),
        ).fetchall()
        return [(r["year"], r["value"]) for r in rows]


# ── Analysis Jobs ───────────────────────────────────────────────────────


def get_or_create_job(
    country_code: str,
    indicator_1: str,
    indicator_2: str,
    db_path: Path = DEFAULT_DB,
) -> int:
    """Return the job id, creating a pending job if one doesn't exist."""
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO analysis_jobs (country_code, indicator_1, indicator_2)
               VALUES (?, ?, ?)""",
            (country_code, indicator_1, indicator_2),
        )
        row = conn.execute(
            "SELECT id FROM analysis_jobs WHERE country_code=? AND indicator_1=? AND indicator_2=?",
            (country_code, indicator_1, indicator_2),
        ).fetchone()
        return row["id"]


def update_job_status(
    job_id: int,
    status: str,
    db_path: Path = DEFAULT_DB,
    *,
    pearson_r: float | None = None,
    total_years: int | None = None,
    flagged_count: int | None = None,
    error_message: str | None = None,
) -> None:
    """Update the status (and optional result fields) of a job."""
    with _connect(db_path) as conn:
        conn.execute(
            """UPDATE analysis_jobs
               SET status=?,
                   pearson_r=COALESCE(?, pearson_r),
                   total_years=COALESCE(?, total_years),
                   flagged_count=COALESCE(?, flagged_count),
                   error_message=COALESCE(?, error_message),
                   started_at=CASE WHEN ?='fetching' THEN datetime('now') ELSE started_at END,
                   completed_at=CASE WHEN ? IN ('done','error','skipped') THEN datetime('now') ELSE completed_at END
               WHERE id=?""",
            (status, pearson_r, total_years, flagged_count, error_message, status, status, job_id),
        )


def get_next_pending_job(db_path: Path = DEFAULT_DB) -> dict | None:
    """Return the next pending job as a dict, or None if queue is empty."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM analysis_jobs WHERE status='pending' ORDER BY id LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def has_any_jobs(db_path: Path = DEFAULT_DB) -> bool:
    """Return True if analysis_jobs has at least one row."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT 1 FROM analysis_jobs LIMIT 1").fetchone()
        return row is not None


# ── Indicator Usefulness ────────────────────────────────────────────────


def has_fetch_log(
    country_code: str, indicator_code: str, db_path: Path = DEFAULT_DB
) -> int | None:
    """Return the had_data flag for a prior fetch, or None if never tried."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT had_data FROM indicator_fetch_log WHERE country_code=? AND indicator_code=?",
            (country_code, indicator_code),
        ).fetchone()
        return row["had_data"] if row else None


def record_fetch_attempt(
    country_code: str,
    indicator_code: str,
    had_data: bool,
    db_path: Path = DEFAULT_DB,
) -> None:
    """Log a World Bank API fetch attempt for (country, indicator)."""
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO indicator_fetch_log (country_code, indicator_code, had_data)
               VALUES (?, ?, ?)
               ON CONFLICT DO UPDATE
               SET had_data=excluded.had_data, fetched_at=datetime('now')""",
            (country_code, indicator_code, 1 if had_data else 0),
        )


def get_indicator_fetch_stats(
    indicator_code: str, db_path: Path = DEFAULT_DB
) -> dict:
    """Return {tried, empty} counts computed from indicator_fetch_log."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS tried,
                      COALESCE(SUM(CASE WHEN had_data=0 THEN 1 ELSE 0 END), 0) AS empty
               FROM indicator_fetch_log WHERE indicator_code=?""",
            (indicator_code,),
        ).fetchone()
        return {"tried": row["tried"], "empty": row["empty"]}


def is_indicator_useful(
    indicator_code: str, db_path: Path = DEFAULT_DB
) -> bool:
    """Return True if indicator is unknown or still marked 'useful'."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM indicators WHERE code=?", (indicator_code,)
        ).fetchone()
        return row is None or row["status"] == "useful"


def mark_indicator_not_useful(
    indicator_code: str, db_path: Path = DEFAULT_DB
) -> int:
    """Mark an indicator not useful and cancel pending jobs referencing it.

    Returns the number of pending jobs that were skipped.
    """
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE indicators SET status='not_useful' WHERE code=?",
            (indicator_code,),
        )
        cur = conn.execute(
            """UPDATE analysis_jobs
               SET status='skipped',
                   error_message='indicator ' || ? || ' marked not useful',
                   completed_at=datetime('now')
               WHERE status='pending' AND (indicator_1=? OR indicator_2=?)""",
            (indicator_code, indicator_code, indicator_code),
        )
        return cur.rowcount


# ── Flagged Items ───────────────────────────────────────────────────────


def store_flagged_item(
    job_id: int,
    country_code: str,
    indicator_1: str,
    indicator_2: str,
    year: int,
    value_1: float,
    value_2: float,
    expected_correlation: float,
    statistical_confidence: float,
    peer_z: float | None = None,
    global_shock_fraction: float | None = None,
    db_path: Path = DEFAULT_DB,
) -> int:
    """Insert a flagged data point. Returns the new row id."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO flagged_items
               (job_id, country_code, indicator_1, indicator_2, year,
                value_1, value_2, expected_correlation, statistical_confidence,
                peer_z, global_shock_fraction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            (job_id, country_code, indicator_1, indicator_2, year,
             value_1, value_2, expected_correlation, statistical_confidence,
             peer_z, global_shock_fraction),
        )
        if cur.lastrowid:
            return cur.lastrowid
        # Already exists — fetch the id
        row = conn.execute(
            "SELECT id FROM flagged_items WHERE country_code=? AND indicator_1=? AND indicator_2=? AND year=?",
            (country_code, indicator_1, indicator_2, year),
        ).fetchone()
        return row["id"]


def update_flagged_item_assessment(
    item_id: int,
    is_anomaly: bool,
    confidence: float,
    explanation: str,
    db_path: Path = DEFAULT_DB,
) -> None:
    """Store the LLM assessment for a flagged item."""
    with _connect(db_path) as conn:
        conn.execute(
            """UPDATE flagged_items
               SET llm_is_anomaly=?, llm_confidence=?, llm_explanation=?, assessed_at=datetime('now')
               WHERE id=?""",
            (int(is_anomaly), confidence, explanation, item_id),
        )


# Active-learning priority score computed in SQL. Expression is shared
# between get_unreviewed_items (for ORDER BY + as a returned column) and
# the count helper (so "N of M" shown in the dashboard is consistent).
#
# Rewards, in order:
#   1.0  LLM/stats disagreement (flags worth a human eye)
#   ≤0.5 LLM uncertainty — peaks at llm_confidence = 0.5
#   ≤0.3 peer-unique move (low global_shock_fraction)
_PRIORITY_EXPR = """(
    CASE
        WHEN (f.llm_is_anomaly = 1 AND f.statistical_confidence < 0.5)
          OR (f.llm_is_anomaly = 0 AND f.statistical_confidence >= 0.7)
            THEN 1.0
        ELSE 0.0
    END
    + 0.5 * (1.0 - ABS(COALESCE(f.llm_confidence, 0.5) - 0.5) * 2)
    + 0.3 * (1.0 - COALESCE(f.global_shock_fraction, 0.0))
)"""

# "High-value" = NOT (LLM assessed AND LLM agrees with stats direction AND
# both are confident). Unassessed items (NULL llm_is_anomaly) are always
# considered high-value so they reach the reviewer.
_HIGH_VALUE_FILTER = """NOT (
    f.llm_is_anomaly IS NOT NULL
    AND f.llm_is_anomaly = CASE
        WHEN f.statistical_confidence >= 0.7 THEN 1 ELSE 0 END
    AND f.llm_confidence >= 0.85
    AND f.statistical_confidence >= 0.85
)"""


def _build_review_filters(
    *,
    country_codes: list[str] | None,
    indicator_pairs: list[tuple[str, str]] | None,
    min_confidence: float | None,
    show_reviewed: bool,
    high_value_only: bool,
) -> tuple[list[str], list]:
    """Assemble WHERE-clause fragments and params for the review queries."""
    conditions: list[str] = []
    params: list = []

    if not show_reviewed:
        conditions.append("f.id NOT IN (SELECT flagged_item_id FROM reviews)")

    if country_codes:
        placeholders = ",".join("?" * len(country_codes))
        conditions.append(f"f.country_code IN ({placeholders})")
        params.extend(country_codes)

    if indicator_pairs:
        pair_conds = []
        for ind1, ind2 in indicator_pairs:
            pair_conds.append("(f.indicator_1=? AND f.indicator_2=?)")
            params.extend([ind1, ind2])
        conditions.append(f"({' OR '.join(pair_conds)})")

    if min_confidence is not None:
        conditions.append("f.llm_confidence >= ?")
        params.append(min_confidence)

    if high_value_only:
        conditions.append(_HIGH_VALUE_FILTER)

    return conditions, params


def get_unreviewed_items(
    db_path: Path = DEFAULT_DB,
    *,
    country_codes: list[str] | None = None,
    indicator_pairs: list[tuple[str, str]] | None = None,
    min_confidence: float | None = None,
    show_reviewed: bool = False,
    high_value_only: bool = True,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    """Return flagged items for the review queue ordered by active-learning
    priority (see ``_PRIORITY_EXPR``).

    ``high_value_only`` (default True) hides flags where the LLM and the
    statistical detector already agree at high confidence — those don't
    need human attention.
    """
    conditions, params = _build_review_filters(
        country_codes=country_codes,
        indicator_pairs=indicator_pairs,
        min_confidence=min_confidence,
        show_reviewed=show_reviewed,
        high_value_only=high_value_only,
    )
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""SELECT f.*,
                       i1.name AS indicator_1_name,
                       i2.name AS indicator_2_name,
                       r.status AS review_status,
                       r.note AS review_note,
                       {_PRIORITY_EXPR} AS priority
                FROM flagged_items f
                LEFT JOIN indicators i1 ON f.indicator_1 = i1.code
                LEFT JOIN indicators i2 ON f.indicator_2 = i2.code
                LEFT JOIN reviews r ON f.id = r.flagged_item_id
                {where}
                ORDER BY priority DESC, f.statistical_confidence DESC, f.id
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
        return [dict(r) for r in rows]


def get_unreviewed_count(
    db_path: Path = DEFAULT_DB,
    *,
    country_codes: list[str] | None = None,
    indicator_pairs: list[tuple[str, str]] | None = None,
    min_confidence: float | None = None,
    high_value_only: bool = False,
) -> int:
    """Count unreviewed items, optionally honoring the same filters the
    review queue uses. ``high_value_only=False`` (default) returns the raw
    queue size — matches the dashboard's "showing X of Y" denominator."""
    conditions, params = _build_review_filters(
        country_codes=country_codes,
        indicator_pairs=indicator_pairs,
        min_confidence=min_confidence,
        show_reviewed=False,
        high_value_only=high_value_only,
    )
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM flagged_items f {where}",
            params,
        ).fetchone()
        return row["cnt"]


# ── Reviews ─────────────────────────────────────────────────────────────


def submit_review(
    flagged_item_id: int,
    status: str,
    note: str = "",
    db_path: Path = DEFAULT_DB,
) -> None:
    """Record a review decision (validated / dismissed / edited)."""
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO reviews (flagged_item_id, status, note)
               VALUES (?, ?, ?)
               ON CONFLICT(flagged_item_id) DO UPDATE
               SET status=excluded.status, note=excluded.note, reviewed_at=datetime('now')""",
            (flagged_item_id, status, note),
        )


# ── Dashboard Stats ────────────────────────────────────────────────────


def get_dashboard_stats(db_path: Path = DEFAULT_DB) -> dict:
    """Return summary stats for the dashboard header."""
    with _connect(db_path) as conn:
        jobs_done = conn.execute(
            "SELECT COUNT(*) AS cnt FROM analysis_jobs WHERE status='done'"
        ).fetchone()["cnt"]
        total_flagged = conn.execute(
            "SELECT COUNT(*) AS cnt FROM flagged_items"
        ).fetchone()["cnt"]
        reviewed = conn.execute(
            "SELECT COUNT(*) AS cnt FROM reviews"
        ).fetchone()["cnt"]
        ws = conn.execute(
            "SELECT last_heartbeat, state FROM worker_status WHERE id=1"
        ).fetchone()
        return {
            "jobs_completed": jobs_done,
            "total_flagged": total_flagged,
            "reviewed_count": reviewed,
            "worker_state": ws["state"] if ws else "unknown",
            "last_heartbeat": ws["last_heartbeat"] if ws else None,
        }


# ── Worker Status ──────────────────────────────────────────────────────


def update_heartbeat(
    state: str = "working",
    current_job_id: int | None = None,
    db_path: Path = DEFAULT_DB,
) -> None:
    """Update the worker heartbeat timestamp and state."""
    with _connect(db_path) as conn:
        conn.execute(
            """UPDATE worker_status
               SET last_heartbeat=datetime('now'), state=?, current_job_id=?
               WHERE id=1""",
            (state, current_job_id),
        )


# ── Export ──────────────────────────────────────────────────────────────


def get_unexported_items(db_path: Path = DEFAULT_DB) -> list[dict]:
    """Return flagged items that haven't been exported yet."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """SELECT f.*, i1.name AS indicator_1_name, i2.name AS indicator_2_name
               FROM flagged_items f
               LEFT JOIN indicators i1 ON f.indicator_1 = i1.code
               LEFT JOIN indicators i2 ON f.indicator_2 = i2.code
               WHERE f.exported_at IS NULL"""
        ).fetchall()
        return [dict(r) for r in rows]


def mark_as_exported(item_ids: list[int], db_path: Path = DEFAULT_DB) -> None:
    """Set exported_at timestamp on given flagged item IDs."""
    if not item_ids:
        return
    with _connect(db_path) as conn:
        placeholders = ",".join("?" * len(item_ids))
        conn.execute(
            f"UPDATE flagged_items SET exported_at=datetime('now') WHERE id IN ({placeholders})",
            item_ids,
        )


# ── Regime Data ────────────────────────────────────────────────────────


def upsert_regime_data(
    rows: list[tuple[str, int, int]], db_path: Path = DEFAULT_DB
) -> int:
    """Insert or update (country_code, year, regime_type) rows. Label is derived.

    Returns the number of rows written.
    """
    if not rows:
        return 0
    payload = [
        (cc, int(yr), int(rt), REGIME_LABELS.get(int(rt), "Unknown"))
        for cc, yr, rt in rows
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            """INSERT INTO regime_data (country_code, year, regime_type, regime_label)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(country_code, year) DO UPDATE
               SET regime_type=excluded.regime_type,
                   regime_label=excluded.regime_label""",
            payload,
        )
    return len(payload)


def get_regime(
    country_code: str, year: int, db_path: Path = DEFAULT_DB
) -> dict | None:
    """Return exact regime record for (country, year) or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT country_code, year, regime_type, regime_label "
            "FROM regime_data WHERE country_code=? AND year=?",
            (country_code, year),
        ).fetchone()
        return dict(row) if row else None


def get_regime_with_fallback(
    country_code: str, year: int, db_path: Path = DEFAULT_DB
) -> dict | None:
    """Return regime for (country, year), carrying forward the last known
    regime for years beyond dataset coverage. Adds ``source_year`` and
    ``carried_forward`` to the returned dict."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """SELECT country_code, year AS source_year, regime_type, regime_label
               FROM regime_data
               WHERE country_code=? AND year<=?
               ORDER BY year DESC LIMIT 1""",
            (country_code, year),
        ).fetchone()
        if not row:
            return None
        out = dict(row)
        out["year"] = year
        out["carried_forward"] = out["source_year"] != year
        return out


def regime_row_count(db_path: Path = DEFAULT_DB) -> int:
    """Return total rows in regime_data. Useful for loader smoke tests."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM regime_data").fetchone()
        return row["cnt"]


# ── Useful Pairs ───────────────────────────────────────────────────────


def upsert_useful_pair(
    indicator_1: str,
    indicator_2: str,
    global_r: float,
    support_count: int,
    panel_size: int,
    db_path: Path = DEFAULT_DB,
) -> None:
    """Record a pair that passed discovery. Stores with indicators sorted so
    (ind1, ind2) matches the canonical order used when seeding jobs."""
    a, b = sorted([indicator_1, indicator_2])
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO useful_pairs
                 (indicator_1, indicator_2, global_r, support_count, panel_size, discovered_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(indicator_1, indicator_2) DO UPDATE SET
                 global_r=excluded.global_r,
                 support_count=excluded.support_count,
                 panel_size=excluded.panel_size,
                 discovered_at=excluded.discovered_at""",
            (a, b, float(global_r), int(support_count), int(panel_size)),
        )


def get_useful_pairs(db_path: Path = DEFAULT_DB) -> list[tuple[str, str]]:
    """Return list of (indicator_1, indicator_2) tuples that passed discovery."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT indicator_1, indicator_2 FROM useful_pairs "
            "ORDER BY ABS(global_r) DESC"
        ).fetchall()
        return [(r["indicator_1"], r["indicator_2"]) for r in rows]


def get_useful_pairs_detailed(db_path: Path = DEFAULT_DB) -> list[dict]:
    """Return full useful_pairs rows — for CLI display and dashboards."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM useful_pairs ORDER BY ABS(global_r) DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def clear_useful_pairs(db_path: Path = DEFAULT_DB) -> int:
    """Wipe the useful_pairs table. Returns number of rows removed."""
    with _connect(db_path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM useful_pairs").fetchone()[0]
        conn.execute("DELETE FROM useful_pairs")
        return before


def wipe_job_pipeline(db_path: Path = DEFAULT_DB) -> dict:
    """Drop all queued work and human feedback. Used by ``worker.py --reseed``
    when the pair catalog changes and the old flags/reviews no longer map to
    current pairs. Returns row counts removed per table."""
    with _connect(db_path) as conn:
        counts = {
            "reviews": conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0],
            "flagged_items": conn.execute("SELECT COUNT(*) FROM flagged_items").fetchone()[0],
            "analysis_jobs": conn.execute("SELECT COUNT(*) FROM analysis_jobs").fetchone()[0],
        }
        conn.execute("DELETE FROM reviews")
        conn.execute("DELETE FROM flagged_items")
        conn.execute("DELETE FROM analysis_jobs")
    return counts


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mktemp(suffix=".db"))
    print(f"Testing with temp DB: {tmp}")

    init_db(tmp)

    # Indicators
    upsert_indicators({"NY.GDP.MKTP.KD.ZG": "GDP Growth", "FP.CPI.TOTL.ZG": "Inflation"}, topic="Economy", db_path=tmp)
    print("Upserted 2 indicators")

    # Indicator data
    store_indicator_data("USA", "NY.GDP.MKTP.KD.ZG", [(2020, 2.3), (2021, 5.7)], db_path=tmp)
    assert has_indicator_data("USA", "NY.GDP.MKTP.KD.ZG", db_path=tmp)
    assert not has_indicator_data("USA", "FAKE.IND", db_path=tmp)
    data = get_indicator_data("USA", "NY.GDP.MKTP.KD.ZG", db_path=tmp)
    assert data == [(2020, 2.3), (2021, 5.7)]
    print(f"Indicator data: {data}")

    # Jobs
    job_id = get_or_create_job("USA", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG", db_path=tmp)
    print(f"Created job: {job_id}")
    update_job_status(job_id, "done", db_path=tmp, pearson_r=0.85, total_years=10, flagged_count=2)
    assert get_next_pending_job(db_path=tmp) is None  # no more pending

    # Flagged items (with Phase C peer fields)
    item_id = store_flagged_item(
        job_id, "USA", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG",
        2020, 2.3, 3.1, 0.85, 0.92,
        peer_z=1.7, global_shock_fraction=0.3,
        db_path=tmp,
    )
    print(f"Flagged item: {item_id}")
    update_flagged_item_assessment(item_id, True, 0.88, "Likely anomaly", db_path=tmp)
    # Default high_value_only filter hides this (LLM + stats agree at high confidence).
    assert get_unreviewed_items(db_path=tmp) == []
    items = get_unreviewed_items(db_path=tmp, high_value_only=False)
    assert items[0]["peer_z"] == 1.7 and items[0]["global_shock_fraction"] == 0.3
    assert "priority" in items[0]

    # Review queue (items present under high_value_only=False; hidden by default)
    items = get_unreviewed_items(db_path=tmp, high_value_only=False)
    assert len(items) == 1
    assert items[0]["llm_explanation"] == "Likely anomaly"
    assert get_unreviewed_count(db_path=tmp) == 1  # raw count
    # Same item hidden under high-value filter (LLM and stats agree, both confident).
    assert get_unreviewed_count(db_path=tmp, high_value_only=True) == 0

    # Review
    submit_review(item_id, "validated", "Confirmed issue", db_path=tmp)
    assert get_unreviewed_count(db_path=tmp) == 0

    # Stats
    stats = get_dashboard_stats(db_path=tmp)
    print(f"Stats: {stats}")
    assert stats["jobs_completed"] == 1
    assert stats["total_flagged"] == 1
    assert stats["reviewed_count"] == 1

    # Export
    unexported = get_unexported_items(db_path=tmp)
    assert len(unexported) == 1
    mark_as_exported([item_id], db_path=tmp)
    assert len(get_unexported_items(db_path=tmp)) == 0

    # Heartbeat
    update_heartbeat("working", job_id, db_path=tmp)

    # Regime data
    n = upsert_regime_data(
        [("USA", 2000, 3), ("USA", 2020, 3), ("ARG", 1999, 2)], db_path=tmp
    )
    assert n == 3
    assert get_regime("USA", 2000, db_path=tmp)["regime_label"] == "Liberal Democracy"
    # Exact match
    r = get_regime_with_fallback("USA", 2020, db_path=tmp)
    assert r["regime_type"] == 3 and r["carried_forward"] is False
    # Carry-forward (2024 beyond dataset max year of 2020 for USA)
    r = get_regime_with_fallback("USA", 2024, db_path=tmp)
    assert r["regime_type"] == 3 and r["carried_forward"] is True
    assert r["source_year"] == 2020
    # Unknown country → None
    assert get_regime_with_fallback("ZZZ", 2020, db_path=tmp) is None
    assert regime_row_count(db_path=tmp) == 3

    # Useful pairs
    upsert_useful_pair("NY.GDP.MKTP.CD", "EG.USE.ELEC.KH.PC", 0.78, 22, 30, db_path=tmp)
    upsert_useful_pair("FP.CPI.TOTL.ZG", "FM.LBL.BMNY.ZG", 0.55, 19, 30, db_path=tmp)
    # Sort-order canonicalization: inserting reversed pair updates same row.
    upsert_useful_pair("EG.USE.ELEC.KH.PC", "NY.GDP.MKTP.CD", 0.80, 23, 30, db_path=tmp)
    pairs = get_useful_pairs(db_path=tmp)
    assert len(pairs) == 2, f"expected 2 pairs, got {pairs}"
    assert all(a < b for a, b in pairs), "pairs should be stored sorted"
    detailed = get_useful_pairs_detailed(db_path=tmp)
    assert detailed[0]["global_r"] == 0.80  # updated value won
    removed = clear_useful_pairs(db_path=tmp)
    assert removed == 2 and get_useful_pairs(db_path=tmp) == []

    # Pipeline wipe — 1 review, 1 flagged item, 1 job from earlier in the test.
    counts = wipe_job_pipeline(db_path=tmp)
    assert counts == {"reviews": 1, "flagged_items": 1, "analysis_jobs": 1}, counts
    assert get_unreviewed_count(db_path=tmp) == 0
    counts_again = wipe_job_pipeline(db_path=tmp)
    assert counts_again == {"reviews": 0, "flagged_items": 0, "analysis_jobs": 0}

    tmp.unlink()
    print("\nAll tests passed!")
