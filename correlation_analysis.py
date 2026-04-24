"""Cross-indicator correlation analysis and outlier detection.

Phase C adds a *peer-year* deviation check: a candidate flag only survives
if the country moved unusually *and* its peers didn't move the same way
that year. This cleans out global shocks (2008 GFC, 2020 COVID) that the
own-history z-score alone will flag everywhere.
"""

import numpy as np
import pandas as pd


def _per_country_adjusted_product(group: pd.DataFrame, corr: float) -> pd.Series:
    """Year-over-year change product, sign-flipped so "suspicious against
    the pair's expected sign" is always negative in the returned series."""
    delta_1 = group["value_1"].diff()
    delta_2 = group["value_2"].diff()
    change_product = delta_1 * delta_2
    return change_product if corr >= 0 else -change_product


def build_peer_change_frame(multi_country_df: pd.DataFrame) -> pd.DataFrame:
    """Produce the ``peer_frame`` expected by ``analyze_correlation``.

    Given a DataFrame containing (country_id, year, value_1, value_2) rows
    for *many* countries, returns (country_id, year, change_product) using
    the same adjusted YoY product each country's analysis uses.

    Callers that already compute this can bypass this helper; it exists so
    the worker can build a peer frame once per pair instead of per job.
    """
    frames: list[pd.DataFrame] = []
    for country_id, group in multi_country_df.groupby("country_id"):
        g = group.sort_values("year").copy()
        corr = g["value_1"].corr(g["value_2"])
        adjusted = _per_country_adjusted_product(g, 0.0 if np.isnan(corr) else corr)
        frames.append(
            pd.DataFrame(
                {
                    "country_id": country_id,
                    "year": g["year"].values,
                    "change_product": adjusted.values,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["country_id", "year", "change_product"])
    return pd.concat(frames, ignore_index=True).dropna(subset=["change_product"])


def _peer_suspicion_z(own: float, peers: np.ndarray) -> float:
    """How many robust z-units more *suspicious* the country was than its
    peers that year. Positive when own's adjusted YoY product is lower
    (more against-correlation) than the peer median; near zero for a
    shared shock; negative if peers moved more suspiciously than own.
    Returns NaN for <3 peers or zero-MAD peer distribution."""
    if len(peers) < 3:
        return float("nan")
    median = float(np.median(peers))
    mad = float(np.median(np.abs(peers - median)))
    if mad == 0:
        return float("nan")
    # 1.4826 makes MAD-based z comparable to a normal z.
    return (median - own) / (1.4826 * mad)


def analyze_correlation(
    df: pd.DataFrame,
    threshold: float = 1.5,
    peer_threshold: float = 1.0,
    peer_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Analyze correlation between two indicators and flag mismatches.

    Parameters
    ----------
    df : DataFrame with columns: year, value_1, value_2, country_id.
    threshold : float
        Own-history z-score threshold (lower = more aggressive).
    peer_threshold : float
        Minimum robust peer-year z-score required to keep a flag. Setting
        it to 0 disables the peer-aware filter (matches pre-Phase-C
        behavior, for regression tests).
    peer_frame : DataFrame | None
        (country_id, year, change_product) for every country running the
        same indicator pair. When supplied, a candidate flag is kept only
        if the country's ``change_product`` is a peer-year outlier too.
        When None, ``peer_z`` and ``global_shock_fraction`` are NaN and no
        peer-based filtering happens.

    Returns
    -------
    DataFrame with added columns: expected_correlation, integrity_flag,
    confidence_score, peer_z, global_shock_fraction.
    """
    use_peer_filter = peer_frame is not None and not peer_frame.empty
    results: list[pd.DataFrame] = []

    for country_id, group in df.groupby("country_id"):
        group = group.sort_values("year").copy()

        corr = group["value_1"].corr(group["value_2"])
        group["expected_correlation"] = corr

        adjusted = _per_country_adjusted_product(
            group, 0.0 if np.isnan(corr) else corr
        )

        mean = adjusted.mean()
        std = adjusted.std()
        if std == 0 or np.isnan(std):
            z_scores = pd.Series(0.0, index=group.index)
        else:
            z_scores = (adjusted - mean) / std

        # Own-history detector: candidate flag when the adjusted product
        # moves suspiciously against the expected correlation sign.
        candidate = (z_scores < -threshold).fillna(False)
        candidate.iloc[0] = False  # first row has no YoY delta
        group["integrity_flag"] = candidate.copy()

        # Peer-year comparison + global-shock fraction.
        peer_z_col = pd.Series(np.nan, index=group.index, dtype=float)
        shock_col = pd.Series(np.nan, index=group.index, dtype=float)

        if use_peer_filter:
            for idx in group.index[candidate]:
                year = int(group.at[idx, "year"])
                own_change = float(adjusted.at[idx])
                if np.isnan(own_change):
                    group.at[idx, "integrity_flag"] = False
                    continue

                peers = peer_frame[
                    (peer_frame["year"] == year)
                    & (peer_frame["country_id"] != country_id)
                ]["change_product"].to_numpy(dtype=float)
                peers = peers[~np.isnan(peers)]

                pz = _peer_suspicion_z(own_change, peers)
                # Same-sign fraction describes how broadly the movement was
                # shared. High = global shock (deprioritize); low = unusual
                # within the peer group (interesting).
                if len(peers) >= 3 and own_change != 0:
                    own_sign = np.sign(own_change)
                    shared = float(np.mean(np.sign(peers) == own_sign))
                else:
                    shared = float("nan")

                peer_z_col.at[idx] = pz
                shock_col.at[idx] = shared

                # Drop the flag when own wasn't meaningfully more
                # suspicious than peers. NaN peer_z (too few peers or
                # zero MAD) → keep conservatively.
                if not np.isnan(pz) and pz < peer_threshold:
                    group.at[idx, "integrity_flag"] = False

        group["peer_z"] = peer_z_col
        group["global_shock_fraction"] = shock_col

        # Confidence score: own-history deviation, gated on the final flag.
        raw_confidence = (-z_scores).clip(lower=0) / (threshold * 2)
        group["confidence_score"] = raw_confidence.clip(upper=1.0)
        group.loc[~group["integrity_flag"], "confidence_score"] = 0.0

        results.append(group)

    if not results:
        return df.assign(
            expected_correlation=np.nan,
            integrity_flag=False,
            confidence_score=0.0,
            peer_z=np.nan,
            global_shock_fraction=np.nan,
        )

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    from data_ingestion import fetch_multi_country

    print("Fetching GDP Growth vs Inflation for USA, GBR ...")
    df = fetch_multi_country(
        ["USA", "GBR"], "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG"
    )

    print("Running correlation analysis (no peer frame — own-history only) ...")
    result = analyze_correlation(df, threshold=1.5)

    for country_id, group in result.groupby("country_id"):
        flagged = group[group["integrity_flag"]]
        print(f"\n{country_id}: Pearson r = {group['expected_correlation'].iloc[0]:.3f}, "
              f"{len(flagged)} flags out of {len(group)} rows")
        if not flagged.empty:
            print(flagged[[
                "year", "value_1", "value_2", "confidence_score",
                "peer_z", "global_shock_fraction",
            ]].to_string(index=False))

    print("\n--- Peer-aware run (peer frame from the same 2-country sample) ---")
    peer_frame = build_peer_change_frame(df)
    result2 = analyze_correlation(df, threshold=1.5, peer_frame=peer_frame)
    for country_id, group in result2.groupby("country_id"):
        flagged = group[group["integrity_flag"]]
        print(f"{country_id}: {len(flagged)} flags (was "
              f"{int(result[result['country_id']==country_id]['integrity_flag'].sum())})")
