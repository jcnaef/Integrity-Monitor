"""Cross-indicator correlation analysis and outlier detection."""

import numpy as np
import pandas as pd


def analyze_correlation(
    df: pd.DataFrame, threshold: float = 1.5
) -> pd.DataFrame:
    """Analyze correlation between two indicators and flag mismatches.

    Parameters
    ----------
    df : DataFrame with columns: year, value_1, value_2, country_id
        (output of fetch_multi_country).
    threshold : float
        Z-score threshold for flagging a data point. Lower values flag more
        aggressively.

    Returns
    -------
    DataFrame with added columns:
        expected_correlation - Pearson r for that country's indicator pair
        integrity_flag       - True when the year deviates from expected pattern
        confidence_score     - 0-1, how extreme the deviation is
    """
    results = []

    for country_id, group in df.groupby("country_id"):
        group = group.sort_values("year").copy()

        # Pearson correlation for the full series
        corr = group["value_1"].corr(group["value_2"])
        group["expected_correlation"] = corr

        # Year-over-year changes
        delta_1 = group["value_1"].diff()
        delta_2 = group["value_2"].diff()

        # Product of changes: negative means they moved in opposite directions
        change_product = delta_1 * delta_2

        # If expected correlation is positive, opposite moves (negative product)
        # are suspicious. If negative, same-direction moves (positive product)
        # are suspicious. We flip the sign so that "suspicious" is always
        # negative in the adjusted product.
        if corr >= 0:
            adjusted = change_product
        else:
            adjusted = -change_product

        # Z-score of the adjusted product to find outliers
        mean = adjusted.mean()
        std = adjusted.std()
        if std == 0 or np.isnan(std):
            z_scores = pd.Series(0.0, index=group.index)
        else:
            z_scores = (adjusted - mean) / std

        # Flag rows where z-score is below -threshold (i.e. moves against
        # the expected correlation pattern). First row has no diff, skip it.
        group["integrity_flag"] = z_scores < -threshold
        group.loc[group.index[0], "integrity_flag"] = False

        # Confidence score: how extreme the deviation is, normalized to 0-1
        # using the absolute value of negative z-scores, capped at 1.
        raw_confidence = (-z_scores).clip(lower=0) / (threshold * 2)
        group["confidence_score"] = raw_confidence.clip(upper=1.0)
        group.loc[~group["integrity_flag"], "confidence_score"] = 0.0

        results.append(group)

    if not results:
        return df.assign(
            expected_correlation=np.nan,
            integrity_flag=False,
            confidence_score=0.0,
        )

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    from data_ingestion import fetch_multi_country

    print("Fetching GDP Growth vs Inflation for USA, GBR ...")
    df = fetch_multi_country(
        ["USA", "GBR"], "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG"
    )

    print("Running correlation analysis ...")
    result = analyze_correlation(df, threshold=1.5)

    for country_id, group in result.groupby("country_id"):
        flagged = group[group["integrity_flag"]]
        print(f"\n{country_id}: Pearson r = {group['expected_correlation'].iloc[0]:.3f}, "
              f"{len(flagged)} flags out of {len(group)} rows")
        if not flagged.empty:
            print(flagged[["year", "value_1", "value_2", "confidence_score"]].to_string(index=False))
