# Cross-Indicator Integrity Monitor

A human-in-the-loop agent that detects potential data integrity issues in
World Bank country indicators by cross-checking pairs of indicators against
their expected correlations. A background worker continuously fetches data,
runs a peer-aware statistical analysis, and asks a local LLM to assess each
suspicious point with political-regime, trade, and nighttime-lights context.
Flagged items land in a Streamlit review queue (ranked by informativeness)
where a human validates or dismisses them, and reviewed results can be
exported as CSV or pushed to a remote REST server.

CS-4400 final project.

## Architecture

```
  World Bank API                 V-Dem / OWID         UN Comtrade       Google Earth Engine
  (wbdata)                       regimes CSV          trade totals      (VIIRS/DMSP NTL)
       │                              │                    │                     │
       ▼                              ▼                    │                     │
  ┌──────────────┐              ┌──────────────┐           │                     │
  │pair_discovery│──► useful_   │ regime_data  │           │                     │
  │    .py       │    pairs     │    .py       │           │                     │
  └──────────────┘    table     └──────────────┘           │                     │
       │                              │                    │                     │
       ▼                              ▼                    ▼                     ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                               worker.py                                          │
  │    seed from useful_pairs → fetch → peer-aware correlate → LLM assess            │
  └────────────────────────────────┬─────────────────────────────────────────────────┘
                                   │
                                   ▼
  ┌────────────────────┐      ┌──────────────────┐
  │ integrity_monitor  │◄────►│   main.py        │
  │     .db (SQLite)   │      │  (Streamlit UI,  │
  └────────┬───────────┘      │  active-learning │
           │                  │   ranked queue)  │
           ▼                  └──────────────────┘
  ┌────────────────────┐
  │ ckan_export.py     │ ──► CSV / POST to server
  │ export_reviewed.py │
  │ ntl_api.py         │ ──► Post-hoc NTL enrichment on exported CSV
  └────────────────────┘
```

The worker and the dashboard are decoupled — they only share the SQLite
database (WAL mode), so the UI stays responsive while analysis runs in the
background. `pair_discovery.py` and `regime_data.py` are one-shot loaders
that must run before the worker on a fresh database.

## Module overview

| File | Purpose |
|------|---------|
| `main.py` | Streamlit dashboard: stats bar, filterable + priority-ranked review queue, high-value toggle, export controls |
| `worker.py` | Background pipeline: seeds jobs from `useful_pairs`, fetches data, runs peer-aware correlation + LLM assessment |
| `pair_discovery.py` | Phase B: two-pass pair discovery over a 30-country reference panel → `useful_pairs` |
| `regime_data.py` | Phase A: loads V-Dem Regimes-of-the-World classification (via Our World in Data) |
| `database.py` | SQLite schema and CRUD layer |
| `data_ingestion.py` | World Bank API wrapper (`wbdata`), indicator catalog, topic discovery |
| `correlation_analysis.py` | Phase C: Pearson correlation + own-history z-score + peer-year robust z filter |
| `llm_integrity.py` | LangChain + Ollama structured-output integrity assessment, enriched with regime / trade / NTL |
| `comtrade_api.py` | UN Comtrade lookup — annual trade flow summary for `(country, year)` used in LLM prompt |
| `ntl_api.py` | VIIRS/DMSP nighttime-lights radiance via Google Earth Engine — used both at assess time (electricity indicators) and as post-hoc CSV enrichment |
| `ckan_export.py` | CKAN-schema DataFrame, incremental CSV / REST export |
| `export_reviewed.py` | CLI to dump all reviewed items to CSV |

## Data flow

1. **Discover useful pairs** — `pair_discovery.py` walks every combinatorial
   pair in the indicator catalog across a 30-country reference panel,
   computes per-country Pearson r, and keeps pairs with `|global_r| ≥ 0.5`
   and same-sign support in ≥60% of the panel. Survivors are written to
   the `useful_pairs` table. Replaces the old `itertools.combinations`
   explosion so the worker only runs jobs for pairs with a genuinely
   reproducible cross-country relationship.
2. **Seed** — on first run the worker loads its indicator catalog, reads
   `useful_pairs`, fetches the country list, and inserts one pending job
   per `(country, indicator_1, indicator_2)` tuple. Worker refuses to seed
   if `useful_pairs` is empty.
3. **Fetch & cache** — per job the worker pulls each indicator time-series
   from the World Bank API (rate-limited), caches it in `indicator_data`,
   and records empty fetches in `indicator_fetch_log` so the same
   unavailable pair is never requested twice. Indicators that come back
   empty for more than 50% of 20+ countries are marked `not_useful` and
   all pending jobs referencing them are skipped.
4. **Build peer frame (once per pair)** — for the active pair, the worker
   merges every country's cached series and computes the sign-adjusted
   year-over-year change product. This becomes the `peer_frame` used for
   every country's analysis — rebuilt only when the pair rotates.
5. **Analyze** — `correlation_analysis.analyze_correlation()` computes
   Pearson r and flags candidate years where the country's YoY change
   moves against the correlation sign by >1.5 σ (own history). Each
   candidate is then checked against the peer frame: `peer_z` (MAD-based
   robust z of own YoY product vs. peers that year) must clear
   `peer_threshold=1.0`, otherwise the flag is dropped as a shared shock.
   `global_shock_fraction` (peers moving the same direction) is also
   persisted for the review UI.
6. **Assess** — each surviving flag is sent to a local Ollama model
   (`qwen3.5:4b` by default) via LangChain structured output. The prompt
   now includes three enrichment channels:
   - **Regime context** from `regime_data` (V-Dem RoW, with last-known
     carry-forward for years past the CSV's coverage).
   - **Trade context** from UN Comtrade (annual total import/export
     summary) via `comtrade_api.py`.
   - **Nighttime-lights context** from Google Earth Engine VIIRS/DMSP,
     only when the pair involves an electricity indicator.
   The model returns `is_anomaly`, `confidence_score`, and a short
   explanation.
7. **Review** — the Streamlit dashboard shows unreviewed items
   priority-ordered: LLM/stats disagreement, mid-range LLM confidence
   (0.5 ± 0.1), and low `global_shock_fraction` all push items up. A
   **"High-value reviews only"** toggle (on by default) hides items
   where LLM and statistics agree at high confidence, so human time is
   spent where the system is uncertain or the two signals conflict.
   Each card carries badges (`LLM/stats disagree`, `uncertain`,
   `peer-unique`, etc.) that explain why it bubbled up.
8. **Export** — reviewed/unexported items can be streamed to a remote
   REST endpoint or downloaded as CSV. `exported_at` timestamps prevent
   duplicates.

## Database schema

Defined in `database.py`:

- `indicators` — catalog + usefulness status
- `indicator_data` — cached time-series values
- `indicator_fetch_log` — negative cache for empty API results
- `useful_pairs` — output of `pair_discovery.py` (`global_r`, `support_count`, `panel_size`)
- `regime_data` — V-Dem RoW regime classification, keyed by `(country_code, year)`
- `analysis_jobs` — one row per `(country, ind1, ind2)` pair with status
- `flagged_items` — suspicious data points with statistical + peer-year metadata (`peer_z`, `global_shock_fraction`) + LLM assessment
- `reviews` — human decisions (validated / dismissed / edited)
- `worker_status` — singleton row for heartbeat + current state

## Requirements

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended)
- [Ollama](https://ollama.com/) running locally, with a model available for
  `llm_integrity.py` (defaults to `qwen3.5:4b`)
- For `ntl_api.py` only: a Google Earth Engine account
- Internet access for UN Comtrade + OWID regime CSV on first run (both
  cache locally afterwards — regime CSV is bundled as
  `political_regime.csv`)

Install dependencies:

```bash
uv sync
```

## Running

### 0. One-shot setup on a fresh database

```bash
uv run python regime_data.py             # populate V-Dem regime_data table
uv run python pair_discovery.py          # populate useful_pairs (a few min, mostly cached after)
uv run python pair_discovery.py --dry-run   # preview the ranked list without writing
```

Both are idempotent. Rerun `pair_discovery.py` whenever the indicator
catalog changes.

### 1. Start the background worker

```bash
uv run python worker.py                 # uses hardcoded ~11-indicator catalog
uv run python worker.py --topics        # expanded catalog from WB topic 3 (Economy & Growth)
uv run python worker.py --reseed        # wipe pipeline + reseed from useful_pairs
```

The worker is resumable — killing it (`Ctrl+C`) and restarting it picks up
where it left off. Existing jobs are not re-seeded unless you pass
`--reseed`. `--reseed` wipes `analysis_jobs`, `flagged_items`, and
`reviews` (any snapshot you want to keep should be exported first).
The worker will refuse to seed jobs if `useful_pairs` is empty.

### 2. Launch the dashboard

```bash
uv run streamlit run main.py
```

The dashboard shows live stats, a priority-ranked review queue, and
sidebar controls for filtering, the high-value-only toggle, server URL,
and export. Worker liveness is indicated by the heartbeat (< 60s since
last update = alive).

### 3. Export reviewed items

- **From the UI:** use the Download CSV button (unexported flagged items
  only) or Export to Server (POSTs unexported items as JSON to the
  configured URL and stamps them as exported).
- **From the CLI:** `uv run python export_reviewed.py reviewed_flagged.csv`
  dumps all reviewed items with their decisions.

### 4. Optional: nighttime-lights enrichment of the exported CSV

After exporting reviewed flags, `ntl_api.py` attaches VIIRS / DMSP
nighttime-lights radiance per `(country, year)` using Google Earth Engine:

```bash
uv run python ntl_api.py   # reads reviewed_flagged.csv → writes flags_with_NTL.csv
```

On first run this will trigger a browser-based GEE authentication flow.
Note: NTL is also consulted *inline* during LLM assessment when a flag's
indicator pair involves electricity.

## Testing individual modules

Most modules have a `__main__` block that exercises them against live data:

```bash
uv run python database.py             # in-memory schema + CRUD smoke test
uv run python data_ingestion.py       # fetches GDP/Inflation for USA, GBR
uv run python correlation_analysis.py # own-history vs peer-aware comparison
uv run python pair_discovery.py --dry-run
uv run python regime_data.py --status # coverage summary of the regime table
uv run python llm_integrity.py        # asks Ollama to assess a fixed row
uv run python ckan_export.py          # builds + writes sample_output.csv
```

## Configuration knobs

- `worker.py`: `MIN_OVERLAPPING_YEARS`, `SLEEP_BETWEEN_API_CALLS`,
  `SLEEP_WHEN_IDLE`, and the usefulness-gating thresholds
  (`MIN_COUNTRIES_BEFORE_JUDGING`, `NOT_USEFUL_RATIO`).
- `pair_discovery.py`: `MIN_GLOBAL_R` (default 0.5), `MIN_COUNTRY_R`
  (0.4), `MIN_SUPPORT_RATIO` (0.6), `MIN_OVERLAPPING_YEARS` (10), and
  the 30-entry `REFERENCE_PANEL`.
- `correlation_analysis.py`: `threshold` (own-history z, default 1.5 σ)
  and `peer_threshold` (MAD-based peer z, default 1.0; set to 0 to
  disable peer filtering for regression tests).
- `llm_integrity.py`: model name and temperature in `_get_chain()`.
- `main.py`: default server URL (`http://localhost:8000/api/flags`),
  `PAGE_SIZE`, and the high-value-only default.

## Roadmap

See `APRIORI_PLAN.md` for the full plan. Phases A–D are implemented.
**Phase E — Apriori classifier from scratch** is the remaining piece:
mine class association rules from reviewed flags
(`apriori_classifier.py`, stratified k-fold CV) to predict
validated vs. dismissed from features like regime, indicator pair,
peer-shock fraction, and LLM verdict.

## File layout

```
final/
├── main.py
├── worker.py
├── pair_discovery.py
├── regime_data.py
├── database.py
├── data_ingestion.py
├── correlation_analysis.py
├── llm_integrity.py
├── comtrade_api.py
├── ntl_api.py
├── ckan_export.py
├── export_reviewed.py
├── political_regime.csv        # bundled OWID regime CSV fallback
├── integrity_monitor.db        # created on first run
├── pyproject.toml
├── PLAN.md                     # original design doc
└── APRIORI_PLAN.md             # follow-on plan (phases A–E)
```
