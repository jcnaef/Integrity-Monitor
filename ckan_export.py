"""Build and export a CKAN-compatible dataset from analyzed indicator data."""

from pathlib import Path

import pandas as pd
import requests

from database import DEFAULT_DB, get_unexported_items, mark_as_exported


def build_ckan_dataset(
    df: pd.DataFrame, indicator_1: str, indicator_2: str
) -> pd.DataFrame:
    """Map the analyzed DataFrame to the required CKAN schema.

    Parameters
    ----------
    df : DataFrame
        Output of analyze_correlation — must have columns: country_id,
        value_1, value_2, expected_correlation, integrity_flag,
        confidence_score.
    indicator_1, indicator_2 : str
        Indicator codes used for the pair (e.g. "NY.GDP.MKTP.KD.ZG").

    Returns
    -------
    DataFrame with CKAN schema columns.
    """
    indicator_pair = f"{indicator_1}__{indicator_2}"
    return pd.DataFrame(
        {
            "country_id": df["country_id"],
            "indicator_pair": indicator_pair,
            "reported_value_1": df["value_1"],
            "reported_value_2": df["value_2"],
            "expected_correlation": df["expected_correlation"],
            "integrity_flag": df["integrity_flag"],
            "confidence_score": df["confidence_score"],
        }
    )


def export_csv(df: pd.DataFrame, path: str) -> None:
    """Write a CKAN dataset DataFrame to CSV."""
    df.to_csv(path, index=False)


def build_ckan_metadata(indicator_1: str, indicator_2: str) -> dict:
    """Return CKAN-compatible datapackage metadata."""
    return {
        "name": "cross-indicator-integrity-monitor",
        "title": "Cross-Indicator Integrity Monitor",
        "description": (
            f"Integrity analysis comparing {indicator_1} and {indicator_2} "
            "across countries. Flagged data points indicate potential "
            "misalignment between reported indicator values."
        ),
        "resources": [
            {
                "name": "integrity_flags",
                "path": "integrity_flags.csv",
                "format": "csv",
                "schema": {
                    "fields": [
                        {"name": "country_id", "type": "string"},
                        {"name": "indicator_pair", "type": "string"},
                        {"name": "reported_value_1", "type": "number"},
                        {"name": "reported_value_2", "type": "number"},
                        {"name": "expected_correlation", "type": "number"},
                        {"name": "integrity_flag", "type": "boolean"},
                        {"name": "confidence_score", "type": "number"},
                    ]
                },
            }
        ],
    }


def get_unexported_dataframe(db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Query flagged items where exported_at IS NULL, return as CKAN-schema DataFrame."""
    items = get_unexported_items(db_path=db_path)
    if not items:
        return pd.DataFrame(columns=[
            "country_id", "indicator_pair", "year",
            "reported_value_1", "reported_value_2",
            "expected_correlation", "integrity_flag",
            "confidence_score", "llm_explanation",
        ])
    rows = []
    for item in items:
        rows.append({
            "id": item["id"],
            "country_id": item["country_code"],
            "indicator_pair": f"{item['indicator_1']}__{item['indicator_2']}",
            "year": item["year"],
            "reported_value_1": item["value_1"],
            "reported_value_2": item["value_2"],
            "expected_correlation": item["expected_correlation"],
            "integrity_flag": bool(item.get("llm_is_anomaly")),
            "confidence_score": item.get("statistical_confidence"),
            "llm_explanation": item.get("llm_explanation", ""),
        })
    return pd.DataFrame(rows)


def export_to_server(server_url: str, db_path: Path = DEFAULT_DB) -> int:
    """POST unexported flagged items as JSON to the server REST API.

    Returns the number of items successfully sent.
    """
    df = get_unexported_dataframe(db_path=db_path)
    if df.empty:
        return 0

    item_ids = df["id"].tolist()
    payload = df.drop(columns=["id"]).to_dict(orient="records")

    resp = requests.post(server_url, json=payload, timeout=30)
    resp.raise_for_status()

    mark_as_exported(item_ids, db_path=db_path)
    return len(item_ids)


if __name__ == "__main__":
    from correlation_analysis import analyze_correlation
    from data_ingestion import fetch_multi_country

    ind1 = "NY.GDP.MKTP.KD.ZG"
    ind2 = "FP.CPI.TOTL.ZG"

    print("Fetching data ...")
    raw = fetch_multi_country(["USA", "GBR"], ind1, ind2)

    print("Analyzing ...")
    analyzed = analyze_correlation(raw)

    ckan_df = build_ckan_dataset(analyzed, ind1, ind2)
    print("\nCKAN dataset sample:")
    print(ckan_df.head(10).to_string(index=False))
    print(f"\nColumns: {list(ckan_df.columns)}")
    print(f"Rows: {len(ckan_df)}")

    export_csv(ckan_df, "sample_output.csv")
    print("\nExported to sample_output.csv")

    meta = build_ckan_metadata(ind1, ind2)
    print(f"\nMetadata: {meta['title']}")
