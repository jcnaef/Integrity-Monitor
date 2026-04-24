"""Export all reviewed flagged items to CSV."""

import csv
import sys
from pathlib import Path

from database import DEFAULT_DB, _connect

QUERY = """
SELECT
    f.id,
    f.country_code,
    f.indicator_1,
    i1.name AS indicator_1_name,
    f.indicator_2,
    i2.name AS indicator_2_name,
    f.year,
    f.value_1,
    f.value_2,
    f.expected_correlation,
    f.statistical_confidence,
    f.llm_is_anomaly,
    f.llm_confidence,
    f.llm_explanation,
    f.assessed_at,
    r.status        AS review_status,
    r.note          AS review_note,
    r.reviewed_at
FROM flagged_items f
JOIN reviews r    ON r.flagged_item_id = f.id
LEFT JOIN indicators i1 ON i1.code = f.indicator_1
LEFT JOIN indicators i2 ON i2.code = f.indicator_2
ORDER BY r.reviewed_at
"""


def export(out_path: Path, db_path: Path = DEFAULT_DB) -> int:
    with _connect(db_path) as conn:
        rows = conn.execute(QUERY).fetchall()
    if not rows:
        print("No reviewed items to export.")
        return 0
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(rows[0].keys())
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reviewed_flagged.csv")
    n = export(out)
    print(f"Wrote {n} rows to {out}")
