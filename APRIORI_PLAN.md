# Apriori + Data-Quality Optimization Plan

Follow-on to `PLAN.md`. Five phases total; **Phase A is already shipped**,
Phases B–E are queued. Each phase is independently shippable and tests
cleanly before the next one starts.

## Status

| Phase | Status | Summary |
|---|---|---|
| A — Regime enrichment | **DONE** | V-Dem RoW data loaded via `regime_data.py`, wired into LLM prompt, last-known-regime carry-forward for 2024+ |
| B — Two-pass pair discovery + clean reseed | pending | |
| C — Peer-year deviation in correlation analysis | pending | |
| D — Active learning review queue | pending | |
| E — Apriori classifier from scratch | pending | |

---

## Open decisions (resolve before starting B)

1. **Pair-discovery thresholds.** Proposed: `|global_r| ≥ 0.5` AND ≥60% of
   the ~30-country reference panel has `|country_r| ≥ 0.4` with the same
   sign as `global_r`. Looser alternative: `0.4 / 50%`. Conservative is
   better for clean training data but may drop useful pairs.
2. **Orphan flags on reseed.** Planned: delete `analysis_jobs` for pairs no
   longer in `useful_pairs`, leave `flagged_items` + `reviews` intact (so
   150 reviewed rows survive). Orphan flagged items will have dangling
   `job_id`. Acceptable vs. also pruning flagged items?
3. **Target class for Apriori.** Planned: `review_status` (validated vs.
   dismissed). Alternative: `llm_is_anomaly`, or a composite "true anomaly"
   (LLM says anomaly AND human validated).

---

## Phase B — Two-pass pair discovery + clean reseed

Replace the combinatorial `itertools.combinations(indicators, 2)` seeding
with a discovery step that keeps only pairs with a genuinely reproducible
cross-country correlation.

### Schema (`database.py`)

```sql
CREATE TABLE useful_pairs (
    indicator_1   TEXT NOT NULL,
    indicator_2   TEXT NOT NULL,
    global_r      REAL NOT NULL,       -- median of per-country r across panel
    support_count INTEGER NOT NULL,    -- panel countries agreeing (same sign, |r|≥0.4)
    panel_size    INTEGER NOT NULL,
    discovered_at TEXT NOT NULL,
    PRIMARY KEY (indicator_1, indicator_2)
);
```

CRUD: `upsert_useful_pair`, `get_useful_pairs() -> list[tuple]`,
`clear_useful_pairs`.

### Reference panel

30 diverse economies (not just OECD — need autocracies, small economies,
resource states):

```python
REFERENCE_PANEL = [
    "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "BRA", "IND", "CAN", "RUS",
    "KOR", "AUS", "ESP", "MEX", "IDN", "NLD", "TUR", "SAU", "POL", "SWE",
    "NOR", "ARG", "ZAF", "NGA", "EGY", "THA", "VNM", "PHL", "CHL", "COL",
]
```

### New module: `pair_discovery.py`

1. For every combinatorial pair in the indicator catalog:
   - For every panel country, fetch both series (reuse `_fetch_and_cache`
     logic so SQLite cache is shared with worker).
   - Need ≥10 overlapping years — else skip that country for this pair.
   - Compute Pearson `r`.
2. Classify the pair:
   - `global_r = median(per_country_r)`
   - `support_count = #countries where |r| ≥ 0.4 AND sign(r) == sign(global_r)`
   - Keep if `|global_r| ≥ 0.5` AND `support_count ≥ 0.6 × panel_size`.
3. Write survivors to `useful_pairs`.

CLI: `uv run python pair_discovery.py` prints the ranked list before writing.

### Worker changes (`worker.py`)

Replace:

```python
pairs = list(itertools.combinations(sorted(indicator_codes), 2))
```

with:

```python
pairs = db.get_useful_pairs()   # already filtered
if not pairs:
    log.warning("No useful pairs — run `python pair_discovery.py` first")
    return
```

### Clean reseed

New flag `--reseed` behavior:

```sql
DELETE FROM analysis_jobs;  -- preserve flagged_items, reviews
```

Then re-seed from `useful_pairs × countries`. `flagged_items.job_id` may
point to deleted jobs; that's acceptable (see open decision #2).

### Verification

- `SELECT * FROM useful_pairs` returns sensible classics: GDP×electricity,
  GDP×CO₂, inflation×M2, exports×FX reserves.
- Worker seeds far fewer jobs than before (was `N_indicators × (N-1)/2 × N_countries`).
- Existing reviews survive: `SELECT COUNT(*) FROM reviews` unchanged.

### Cost

~30 panel × 55 pairs × 2 fetches ≈ 3,300 potential API calls, but almost
all cached from prior worker runs. First discovery pass: a few minutes.
Cached in `useful_pairs` forever unless indicator catalog changes.

---

## Phase C — Peer-year deviation

Current detector: z-score of a country's YoY against *its own* history.
Catches 2008, 2020 everywhere as "suspicious." New detector also requires
that *peer countries didn't move the same way*, so global shocks drop out.

### Schema migration (`database.py`)

```sql
ALTER TABLE flagged_items ADD COLUMN peer_z REAL;
ALTER TABLE flagged_items ADD COLUMN global_shock_fraction REAL;
```

Store the worker's computed values; expose via `get_unreviewed_items`.

### `correlation_analysis.py`

New signature:

```python
def analyze_correlation(
    df: pd.DataFrame,
    threshold: float = 1.5,
    peer_threshold: float = 1.0,
    peer_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ...
```

`peer_frame` columns: `(country_id, year, change_product)` — all countries'
adjusted YoY-product for the same pair. For each candidate flag at
`(country_c, year_y)`:

- `own_z` = existing own-history z-score
- `peer_median, peer_mad` = median and MAD of `peer_frame[year == year_y].change_product`
  excluding country_c
- `peer_z = (country_c_value - peer_median) / peer_mad` (robust z)
- `global_shock_fraction` = fraction of peers with sign matching country_c
  (when most peers moved the same way, it's a global shock)

**Flag only when** `own_z ≥ threshold AND peer_z ≥ peer_threshold`.

Emit both `peer_z` and `global_shock_fraction` as columns on the returned
DataFrame so the worker can persist them.

### Worker changes (`worker.py`)

New helper `_build_peer_frame(ind1, ind2) -> pd.DataFrame` that pulls
cached data for *all* countries for this pair, merges on year, computes
`change_product`. Cache the peer frame per-pair across jobs (computed
once per pair per worker session) — avoid rebuilding for every country.

`_process_job` passes `peer_frame` into `analyze_correlation`. Before
storing, reads `peer_z` and `global_shock_fraction` off each flagged row
and stores them.

### Verification

- Rerun on a country-year you know is a global shock (e.g., USA 2008 GDP×
  unemployment): should no longer flag — `global_shock_fraction` high,
  `peer_z` low.
- Rerun on a known real anomaly (ARG 1999 FDI×electricity, already in
  the reviewed CSV): should still flag, with `peer_z` > 1.

---

## Phase D — Active learning review queue

Re-rank the review queue by informativeness. Hide obvious items by
default so human time is spent only on flags where the system is
uncertain or tools disagree.

### `database.py` — `get_unreviewed_items` extension

Add computed `priority` column to the query:

```sql
-- agreement_disagreement: 1 when LLM verdict conflicts with statistical high-confidence
-- uncertainty: peaks at llm_confidence 0.5
-- peer signal: global_shock_fraction low = unusual against peers = more interesting
priority =
    CASE
        WHEN (llm_is_anomaly=1 AND statistical_confidence<0.5)
          OR (llm_is_anomaly=0 AND statistical_confidence>=0.7) THEN 1.0
        ELSE 0.0
    END
  + 0.5 * (1.0 - ABS(llm_confidence - 0.5) * 2)
  + 0.3 * (1.0 - COALESCE(global_shock_fraction, 0.0))
```

Order `get_unreviewed_items` by `priority DESC, statistical_confidence DESC`.

New filter param `high_value_only: bool = True`:

```sql
-- Hide items where LLM and stats both agree AND both confident
AND NOT (
    llm_is_anomaly = CASE WHEN statistical_confidence >= 0.7 THEN 1 ELSE 0 END
    AND llm_confidence >= 0.85
    AND statistical_confidence >= 0.85
)
```

### `main.py` — dashboard changes

- Sidebar toggle: **"High-value reviews only"** (default on).
- Per-item badge in the expander:
  - ⚡ LLM/stats disagree
  - 🤔 uncertain (llm_confidence near 0.5)
  - ✓ agreement (hidden by default)
- Count display: "Showing 12 of 47 unreviewed (high-value only)".

No auto-accept. User can always toggle to see the full queue.

### Verification

- Open dashboard, top items should visibly involve LLM/stats disagreement
  or mid-range confidence.
- Toggle off → full queue returns.
- Item count drops ~60-70% under default filter.

---

## Phase E — Apriori classifier from scratch

The original goal. Implementation only sensible *after* B–D have improved
training data quality.

### Target class (open decision #3)

Default: `review_status` ∈ {validated, dismissed}.

### New module: `apriori_classifier.py`

#### 1. Transaction builder — `build_transactions(df) -> list[frozenset[str]]`

Each reviewed flag → a transaction (set of `feature=bucket` strings).

| Feature family | Bucketing |
|---|---|
| `country` | keep code: `country=ARG` |
| `ind1`, `ind2` | keep code: `ind1=NY.GDP.MKTP.CD` |
| `decade` | `decade=1990s`, `decade=2000s`, ... |
| `corr_sign` | `corr=positive` / `corr=negative` / `corr=near_zero` (|r|<0.1) |
| `stat_conf` | tertile: low/med/high |
| `llm` | `llm=anomaly` / `llm=ok` |
| `llm_conf` | tertile: low/med/high |
| `regime` | from `regime_data`: `regime=electoral_autocracy` etc. |
| `peer` (Phase C) | `peer_moved_together=yes/no` from `global_shock_fraction` |
| `ntl` (optional) | tertile of NTL radiance |
| **class** | `class=validated` / `class=dismissed` |

Discretizer uses `pd.qcut` with tertile edges from the training fold only.
Save edges so test-fold encoding is consistent.

#### 2. Apriori — frequent itemset mining

Classic level-wise:

```
L1 = {items with support ≥ min_support}
k = 2
while L(k-1) non-empty:
    Ck = F(k-1) ⨝ F(k-1) join on shared (k-2)-prefix
    prune candidates whose (k-1)-subsets are not all in L(k-1)
    scan transactions, count support of each candidate
    Lk = {c in Ck with support ≥ min_support}
    k += 1
return ⋃ Lk
```

Storage: `dict[frozenset[str], int]` mapping itemset → support count.

Helpers: `_generate_candidates(prev, k)`, `_has_infrequent_subset(c, prev)`.

With ~150 rows use `min_support` as a count (e.g. 5–10), not a fraction.

#### 3. Class association rules — `generate_rules(itemsets, class_items, min_conf)`

Only rules of form `antecedent → class=X`:

```python
for I in itemsets:
    class_items_in_I = I & class_items
    if len(class_items_in_I) != 1: continue
    antecedent = I - class_items_in_I
    if not antecedent: continue
    conf = support(I) / support(antecedent)
    if conf >= min_conf:
        emit Rule(antecedent, next(iter(class_items_in_I)), support(I), conf)
```

Rank by (conf desc, support desc, len asc) — CBA ordering.

#### 4. Classifier — `classify(rules, transaction, default_class)`

Walk ranked rules, return first rule whose antecedent ⊆ transaction.
Return both the predicted class *and* the firing rule so the dashboard
can show "why."

#### 5. Evaluation — `cross_validate(df, k=5)`

Stratified k-fold (write it yourself, ~15 lines). Per fold:
- Fit discretizer on train only
- Mine rules on train
- Encode + classify test
- Track accuracy, per-class precision/recall, confusion matrix

Print top N rules for sanity checks, e.g.:

```
[0.93 conf | 12 sup] {country=ARG, llm=anomaly} → class=validated
[0.87 conf | 18 sup] {regime=closed_autocracy, corr=negative} → class=validated
```

#### 6. CLI

```bash
uv run python apriori_classifier.py \
    --input reviewed_flagged.csv \
    --min-support 5 \
    --min-confidence 0.7 \
    --evaluate
```

Flags:
- `--top-rules N` — print top N rules
- `--predict FLAG_ID` — pull one flag from DB, encode, classify, show firing rule
- `--target {review_status, llm_is_anomaly}` — switch target class

### Optional optimizations to consider during Phase E

- **Class-balanced mining.** If validated/dismissed is skewed, mine rules
  per-class to avoid the majority class dominating. Alternative: rank by
  **lift** instead of confidence.
- **CBA rule pruning.** Drop rules whose antecedent is a superset of a
  higher-confidence rule's antecedent.
- **Coverage constraint.** Keep only rules needed to cover every training
  row at least once (CBA's database coverage). Produces a minimal
  interpretable rule set.

### Verification

- Print top 10 rules — should be human-readable and intuitive.
- 5-fold CV accuracy beats majority-class baseline by a meaningful margin.
- Confusion matrix doesn't collapse to one class.
- Predicting on a held-out flag returns a sensible rule + class.

---

## Build order & effort

| Phase | Effort | Files touched |
|---|---|---|
| B | 2–3h | `database.py`, `pair_discovery.py` (new), `worker.py` |
| C | 2–3h | `database.py`, `correlation_analysis.py`, `worker.py` |
| D | 1h | `database.py`, `main.py` |
| E | 3–4h | `apriori_classifier.py` (new) |

B → C → D → E is the dependency order. B and C can technically run in
parallel but both touch `worker.py`, so sequential is simpler.

---

## Pre-flight checklist before Phase B

- [ ] Confirm thresholds (`0.5/60%` vs `0.4/50%`)
- [ ] Confirm orphan-flags policy on reseed
- [ ] `git commit` current state so the reseed can be backed out if needed
- [ ] Snapshot `integrity_monitor.db` (copy the file aside) before running `--reseed`
