# Cross-Indicator Integrity Monitor — Implementation Plan

## Context
CS-4400 final project. Build a HITL agent that compares conflicting indicators (e.g., GDP growth vs. electricity consumption) across countries. A **background worker** continuously fetches World Bank data, runs correlation analysis, and gets LLM assessments for flagged points. The **Streamlit dashboard** is a review queue where the user approves/dismisses flags. Data is persisted in SQLite and exported to a remote server via REST API (no duplicates). The architecture supports future Apriori analysis to discover which indicator pairs most reliably predict data integrity issues.

---

## Module Structure

```
final/
├── main.py                 # Streamlit review dashboard (REWRITE)
├── database.py             # SQLite schema + CRUD (NEW)
├── worker.py               # Background data pipeline (NEW)
├── data_ingestion.py       # World Bank API pulls (MODIFY — add topic-based indicator discovery)
├── correlation_analysis.py # Cross-indicator correlation + outlier detection (NO CHANGES)
├── llm_integrity.py        # LangChain LLM integrity assessment (MODIFY — lazy init, optional name params)
├── ckan_export.py          # CKAN export + server push (MODIFY — incremental export, REST POST)
└── pyproject.toml
```

---

## SQLite Schema (`database.py` — NEW)

```sql
CREATE TABLE indicators (
    code  TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    topic TEXT
);

CREATE TABLE indicator_data (
    country_code   TEXT NOT NULL,
    indicator_code TEXT NOT NULL REFERENCES indicators(code),
    year           INTEGER NOT NULL,
    value          REAL NOT NULL,
    fetched_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (country_code, indicator_code, year)
);

CREATE TABLE analysis_jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code  TEXT NOT NULL,
    indicator_1   TEXT NOT NULL,
    indicator_2   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending|fetching|analyzing|assessing|done|error
    pearson_r     REAL,
    total_years   INTEGER,
    flagged_count INTEGER,
    error_message TEXT,
    started_at    TEXT,
    completed_at  TEXT,
    UNIQUE (country_code, indicator_1, indicator_2)
);

CREATE TABLE flagged_items (
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
    llm_is_anomaly         INTEGER,       -- 0 or 1, NULL until LLM assesses
    llm_confidence         REAL,
    llm_explanation        TEXT,
    assessed_at            TEXT,
    exported_at            TEXT,           -- NULL until sent to server
    UNIQUE (country_code, indicator_1, indicator_2, year)
);

CREATE TABLE reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flagged_item_id INTEGER NOT NULL UNIQUE REFERENCES flagged_items(id),
    status          TEXT NOT NULL,  -- validated|dismissed|edited
    note            TEXT DEFAULT '',
    reviewed_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE worker_status (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    last_heartbeat TEXT,
    current_job_id INTEGER,
    state          TEXT DEFAULT 'idle'  -- idle|working|stopped
);
```

Use WAL mode (`PRAGMA journal_mode=WAL`) for concurrent read/write.

Module exposes: `init_db`, `upsert_indicators`, `has_indicator_data`, `store_indicator_data`, `get_indicator_data`, `get_or_create_job`, `update_job_status`, `get_next_pending_job`, `store_flagged_item`, `update_flagged_item_assessment`, `get_unreviewed_items`, `get_unreviewed_count`, `submit_review`, `get_dashboard_stats`, `update_heartbeat`, `get_unexported_items`, `mark_as_exported`.

---

## Step 1 — Data Ingestion (`data_ingestion.py`) — MODIFY

- Keep all existing functions and the hardcoded `INDICATORS` dict as fallback.
- Add `get_indicators_by_topic(topic_ids: list[int]) -> dict[str, str]` using `wbdata.get_indicators(topic=id)` for each topic. Used by the worker on startup to build an expanded indicator catalog (~30-50 indicators from economic topics).

## Step 2 — Correlation Analysis (`correlation_analysis.py`) — NO CHANGES

Interface already works: takes `DataFrame(year, value_1, value_2, country_id)`, returns with `expected_correlation`, `integrity_flag`, `confidence_score`.

## Step 3 — LLM Integrity Flagging (`llm_integrity.py`) — MODIFY

- Accept optional `indicator_1_name` / `indicator_2_name` kwargs in `assess_integrity()` so the worker can pass names from the DB instead of relying on the hardcoded `INDICATORS` dict.
- Lazy-initialize the LLM chain (don't create Ollama connection on import — only when first called).

## Step 4 — CKAN Export (`ckan_export.py`) — MODIFY

- Add `get_unexported_items(db_path) -> pd.DataFrame` — queries flagged items where `exported_at IS NULL`, returns CKAN-schema DataFrame.
- Add `export_to_server(db_path, server_url: str) -> int` — fetches unexported items, POSTs them as JSON batch to server REST API, marks successful items with `exported_at`. Returns count sent.
- Add `mark_as_exported(db_path, item_ids: list[int])` — sets `exported_at` on given IDs.
- Keep existing `build_ckan_dataset()` and `export_csv()` for local use.

## Step 5 — Background Worker (`worker.py`) — NEW

**Startup:**
1. Call `get_indicators_by_topic()` to build expanded indicator catalog. Store in `indicators` table.
2. Fetch country list via `get_country_list()`.
3. Generate all unique indicator pairs (`itertools.combinations`). For each pair × country, seed `analysis_jobs` with `pending` status.

**Main loop (one job at a time):**
1. Pick next pending job from DB (`get_next_pending_job`).
2. Fetch each indicator's time series (check cache via `has_indicator_data`). Reuse `data_ingestion.fetch_indicator()`. Rate limit: `time.sleep(0.5)` between API calls.
3. Merge on year. Skip if < 5 overlapping years.
4. Run `correlation_analysis.analyze_correlation()`.
5. Store flagged items in DB.
6. For each flagged item, run `llm_integrity.assess_integrity()` with indicator names from DB.
7. Mark job `done`. Update heartbeat. Sleep briefly, repeat.
8. When no pending jobs remain, sleep 5 minutes and re-check.

Handle errors with retry/skip. Log progress. Graceful `KeyboardInterrupt` handling.

## Step 6 — Streamlit Dashboard (`main.py`) — REWRITE

### Layout

**Stats bar (top):** Four `st.metric` columns — jobs completed, total flagged, reviewed count, worker status (alive if heartbeat < 60s).

**Sidebar:**
- Filter by country (multi-select, populated from DB)
- Filter by indicator pair (multi-select, populated from DB)
- LLM confidence threshold slider
- "Show already-reviewed" toggle (default off)
- Server URL text input
- "Export to Server" button — POSTs unexported items, shows count sent
- "Download CSV" button — downloads only unexported items

**Main area — Review Queue:**
- Paginated list of unreviewed flagged items (10/page)
- Each item as `st.expander`: country, year, indicator names + values, expected correlation, statistical confidence, LLM assessment (anomaly, confidence, explanation)
- Action buttons: Validate Flag, Dismiss Flag, Save Note
- On action: `submit_review()` → `st.rerun()` to remove from queue
- Manual "Refresh" button to pick up new items from worker

---

## Build Order

| Phase | Task | Files | Verify |
|-------|------|-------|--------|
| 1 | Create database module with schema + CRUD | `database.py` | Run standalone, insert/read sample data |
| 2 | Add `get_indicators_by_topic()` | `data_ingestion.py` | Call it, print expanded indicator list |
| 3 | Update LLM module (lazy init, optional name params) | `llm_integrity.py` | Existing `__main__` test still works |
| 4 | Create worker | `worker.py` | Run it, watch jobs complete in SQLite |
| 5 | Add incremental export + server push | `ckan_export.py` | POST unexported items, verify no duplicates |
| 6 | Rewrite dashboard | `main.py` | Run alongside worker, review flags end-to-end |

---

## Verification

1. Start worker: `python worker.py` — observe DB populating (`sqlite3 integrity_monitor.db "SELECT count(*) FROM analysis_jobs"`)
2. Start dashboard: `streamlit run main.py` — see stats updating, flagged items in queue
3. Review items — confirm they leave the queue, reappear with "Show reviewed" toggle
4. Export to server — confirm only unexported items are POSTed, `exported_at` set, re-click sends 0 (no duplicates)
5. Download CSV — confirm only unexported items included
6. Stop/restart worker — confirm it resumes (no duplicate work)
