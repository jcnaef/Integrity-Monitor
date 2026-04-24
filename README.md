# Cross-Indicator Integrity Monitor

A human-in-the-loop agent that detects potential data integrity issues in
World Bank country indicators by cross-checking pairs of indicators against
their expected correlations. A background worker continuously fetches data,
runs statistical analysis, and asks a local LLM to assess each suspicious
point. Flagged items land in a Streamlit review queue where a human
validates or dismisses them, and reviewed results can be exported as CSV or
pushed to a remote REST server.

CS-4400 final project.

## Architecture

```
  ┌──────────────┐     World Bank API
  │  worker.py   │ ◄──── (wbdata)
  └──────┬───────┘
         │  fetch → correlate → LLM assess
         ▼
  ┌────────────────────┐      ┌──────────────────┐
  │ integrity_monitor  │◄────►│   main.py        │
  │     .db (SQLite)   │      │  (Streamlit UI)  │
  └────────┬───────────┘      └──────────────────┘
           │
           ▼
  ┌────────────────────┐
  │ ckan_export.py     │ ──► CSV / POST to server
  │ export_reviewed.py │
  │ ntl_api.py         │ ──► Google Earth Engine (NTL)
  └────────────────────┘
```

The worker and the dashboard are decoupled — they only share the SQLite
database (WAL mode), so the UI stays responsive while analysis runs in the
background.

## Module overview

| File | Purpose |
|------|---------|
| `main.py` | Streamlit dashboard: stats bar, filterable review queue, export controls |
| `worker.py` | Background pipeline: seeds jobs, fetches data, runs correlation + LLM assessment |
| `database.py` | SQLite schema and CRUD layer |
| `data_ingestion.py` | World Bank API wrapper (`wbdata`), indicator catalog, topic discovery |
| `correlation_analysis.py` | Pearson correlation + z-score outlier detection on year-over-year deltas |
| `llm_integrity.py` | LangChain + Ollama structured-output integrity assessment |
| `ckan_export.py` | CKAN-schema DataFrame, incremental CSV / REST export |
| `export_reviewed.py` | CLI to dump all reviewed items to CSV |
| `ntl_api.py` | Post-hoc enrichment with VIIRS/DMSP nighttime-lights radiance via Google Earth Engine |

## Data flow

1. **Seed** — on first run the worker loads its indicator catalog (either the
   hardcoded `INDICATORS` dict or an expanded list discovered from a World
   Bank topic), fetches the country list, and inserts one pending job per
   `(country, indicator_1, indicator_2)` tuple.
2. **Fetch & cache** — per job the worker pulls each indicator time-series
   from the World Bank API (rate-limited), caches it in `indicator_data`,
   and records empty fetches in `indicator_fetch_log` so the same
   unavailable pair is never requested twice. Indicators that come back
   empty for more than 50% of 20+ countries are marked `not_useful` and all
   pending jobs referencing them are skipped.
3. **Analyze** — `correlation_analysis.analyze_correlation()` computes the
   Pearson r for the series, then flags years whose year-over-year change
   moves against the correlation sign by more than 1.5 standard deviations.
4. **Assess** — each flagged row is sent to a local Ollama model
   (`qwen3.5:4b` by default) via LangChain structured output; the model
   returns `is_anomaly`, `confidence_score`, and a short explanation.
5. **Review** — the Streamlit dashboard shows unreviewed items with filters
   (country, indicator pair, LLM confidence). The user validates,
   dismisses, or annotates each one.
6. **Export** — reviewed/unexported items can be streamed to a remote REST
   endpoint or downloaded as CSV. `exported_at` timestamps prevent
   duplicates.

## Database schema

Defined in `database.py`:

- `indicators` — catalog + usefulness status
- `indicator_data` — cached time-series values
- `indicator_fetch_log` — negative cache for empty API results
- `analysis_jobs` — one row per `(country, ind1, ind2)` pair with status
- `flagged_items` — suspicious data points with statistical + LLM metadata
- `reviews` — human decisions (validated / dismissed / edited)
- `worker_status` — singleton row for heartbeat + current state

## Requirements

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended)
- [Ollama](https://ollama.com/) running locally, with a model available for
  `llm_integrity.py` (defaults to `qwen3.5:4b`)
- For `ntl_api.py` only: a Google Earth Engine account

Install dependencies:

```bash
uv sync
```

## Running

### 1. Start the background worker

```bash
uv run python worker.py                 # hardcoded ~11-indicator catalog
uv run python worker.py --topics        # expanded catalog from WB topic 3 (Economy & Growth)
uv run python worker.py --reseed        # force re-seeding of jobs
```

The worker is resumable — killing it (`Ctrl+C`) and restarting it picks up
where it left off. Existing jobs are not re-seeded unless you pass
`--reseed`.

### 2. Launch the dashboard

```bash
uv run streamlit run main.py
```

The dashboard shows live stats, the current review queue, and sidebar
controls for filtering, server URL, and export. Worker liveness is
indicated by the heartbeat (< 60s since last update = alive).

### 3. Export reviewed items

- **From the UI:** use the Download CSV button (unexported flagged items
  only) or Export to Server (POSTs unexported items as JSON to the
  configured URL and stamps them as exported).
- **From the CLI:** `uv run python export_reviewed.py reviewed_flagged.csv`
  dumps all reviewed items with their decisions.

### 4. Optional: nighttime-lights enrichment

After exporting reviewed flags, `ntl_api.py` attaches VIIRS / DMSP
nighttime-lights radiance per `(country, year)` using Google Earth Engine:

```bash
uv run python ntl_api.py   # reads reviewed_flagged.csv → writes flags_with_NTL.csv
```

On first run this will trigger a browser-based GEE authentication flow.

## Testing individual modules

Most modules have a `__main__` block that exercises them against live data:

```bash
uv run python database.py             # in-memory schema + CRUD smoke test
uv run python data_ingestion.py       # fetches GDP/Inflation for USA, GBR
uv run python correlation_analysis.py # runs analysis on the above
uv run python llm_integrity.py        # asks Ollama to assess a fixed row
uv run python ckan_export.py          # builds + writes sample_output.csv
```

## Configuration knobs

- `worker.py`: `MIN_OVERLAPPING_YEARS`, `SLEEP_BETWEEN_API_CALLS`,
  `SLEEP_WHEN_IDLE`, and the usefulness-gating thresholds
  (`MIN_COUNTRIES_BEFORE_JUDGING`, `NOT_USEFUL_RATIO`).
- `correlation_analysis.py`: `threshold` argument to `analyze_correlation`
  controls flag aggressiveness (default 1.5 σ).
- `llm_integrity.py`: model name and temperature in `_get_chain()`.
- `main.py`: default server URL (`http://localhost:8000/api/flags`) and
  `PAGE_SIZE`.

## File layout

```
final/
├── main.py
├── worker.py
├── database.py
├── data_ingestion.py
├── correlation_analysis.py
├── llm_integrity.py
├── ckan_export.py
├── export_reviewed.py
├── ntl_api.py
├── integrity_monitor.db     # created on first run
├── pyproject.toml
└── PLAN.md                  # original design doc
```
