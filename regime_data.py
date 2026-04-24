"""V-Dem Regimes-of-the-World loader.

Populates the ``regime_data`` SQLite table with V-Dem's four-level regime
classification (Closed Autocracy / Electoral Autocracy / Electoral Democracy
/ Liberal Democracy), keyed by ISO-3 country code and year.

Primary source is Our World in Data's cleaned extract of V-Dem's ``v2x_regime``
variable, which is a stable public CSV keyed by country and year.

Usage
-----
    uv run python regime_data.py            # download + populate (idempotent)
    uv run python regime_data.py --status   # print coverage summary
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import pandas as pd
import requests

import database as db

log = logging.getLogger(__name__)

# OWID's political-regime dataset. V-Dem ``v2x_regime`` mapped to 0..3.
_OWID_URL = "https://ourworldindata.org/grapher/political-regime.csv"

# Bundled fallback path (optional — populated if present).
_LOCAL_CSV = Path(__file__).parent / "political_regime.csv"


def _fetch_csv() -> pd.DataFrame:
    """Return the regime dataset as a DataFrame.

    Tries the bundled CSV first (so reruns are offline-friendly), then falls
    back to downloading from OWID. Any IO error from the download bubbles up.
    """
    if _LOCAL_CSV.exists():
        log.info("Reading bundled regime CSV: %s", _LOCAL_CSV)
        return pd.read_csv(_LOCAL_CSV)

    log.info("Downloading regime CSV from %s", _OWID_URL)
    resp = requests.get(
        _OWID_URL,
        params={"v": "1", "csvType": "full", "useColumnShortNames": "true"},
        headers={"User-Agent": "integrity-monitor/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    # Cache locally for next run.
    _LOCAL_CSV.write_bytes(resp.content)
    log.info("Cached CSV to %s", _LOCAL_CSV)
    return pd.read_csv(io.BytesIO(resp.content))


def _identify_regime_column(df: pd.DataFrame) -> str:
    """Find the column holding integer regime codes 0..3.

    OWID short column names shift between releases (``political_regime``,
    ``regime_row_owid``, ...). We pick the first numeric column whose
    non-null values fall entirely within {0,1,2,3}.
    """
    reserved = {"Entity", "Code", "Year", "year", "entity", "code"}
    for col in df.columns:
        if col in reserved:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        unique = set(series.round().astype(int).unique())
        if unique and unique.issubset({0, 1, 2, 3}):
            return col
    raise ValueError(
        f"No regime column found in CSV. Columns: {list(df.columns)}"
    )


def _normalize(df: pd.DataFrame) -> list[tuple[str, int, int]]:
    """Normalize the raw CSV into (iso3, year, regime_type) tuples."""
    # OWID's canonical columns are Entity, Code, Year. The short-name variant
    # uses lowercase ``country``/``year``. Normalise both.
    code_col = "Code" if "Code" in df.columns else "code"
    year_col = "Year" if "Year" in df.columns else "year"
    regime_col = _identify_regime_column(df)

    subset = df[[code_col, year_col, regime_col]].copy()
    subset.columns = ["code", "year", "regime"]
    # OWID aggregates (World, continents) have no ISO-3 code → drop those.
    subset = subset.dropna(subset=["code", "year", "regime"])
    subset = subset[subset["code"].str.len() == 3]
    subset["year"] = subset["year"].astype(int)
    subset["regime"] = subset["regime"].round().astype(int)

    return list(
        zip(subset["code"].tolist(), subset["year"].tolist(), subset["regime"].tolist())
    )


def load(db_path: Path = db.DEFAULT_DB) -> int:
    """Download (or read bundled) regime CSV, upsert rows. Returns row count."""
    db.init_db(db_path)
    df = _fetch_csv()
    rows = _normalize(df)
    written = db.upsert_regime_data(rows, db_path=db_path)
    log.info("Upserted %d regime rows into %s", written, db_path)
    return written


def status(db_path: Path = db.DEFAULT_DB) -> None:
    """Print a short coverage summary."""
    import sqlite3

    db.init_db(db_path)
    total = db.regime_row_count(db_path)
    print(f"Total rows: {total}")
    if total == 0:
        print("(empty — run `uv run python regime_data.py` to populate)")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    year_row = conn.execute(
        "SELECT MIN(year) AS lo, MAX(year) AS hi FROM regime_data"
    ).fetchone()
    country_count = conn.execute(
        "SELECT COUNT(DISTINCT country_code) AS n FROM regime_data"
    ).fetchone()["n"]
    by_regime = conn.execute(
        """SELECT regime_label, COUNT(*) AS n
           FROM regime_data GROUP BY regime_label ORDER BY n DESC"""
    ).fetchall()
    conn.close()

    print(f"Year range:    {year_row['lo']} – {year_row['hi']}")
    print(f"Countries:     {country_count}")
    print("Distribution:")
    for r in by_regime:
        print(f"  {r['regime_label']:<22s} {r['n']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description="Load V-Dem regime data")
    parser.add_argument("--status", action="store_true", help="Print coverage and exit")
    args = parser.parse_args()

    if args.status:
        status()
    else:
        n = load()
        print(f"\nUpserted {n} regime rows.")
        status()
