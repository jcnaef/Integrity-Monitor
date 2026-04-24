"""World Bank API data ingestion."""

import pandas as pd
import wbdata

# Hardcoded indicator dict for the sidebar dropdown.
INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP Growth (annual %)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of labor force)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    "GC.DOD.TOTL.GD.ZS": "Central government debt, total (% of GDP)",
    "NE.EXP.GNFS.ZS": "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS": "Imports of goods and services (% of GDP)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "SP.POP.GROW": "Population growth (annual %)",
    "PA.NUS.FCRF": "Official exchange rate (LCU per US$)",
    "EG.USE.ELEC.KH.PC": "Electric power consumption (kWh per capita)",
}


def get_indicators_by_topic(topic_ids: list[int]) -> dict[str, str]:
    """Fetch indicators from the World Bank API for the given topic IDs.

    Returns {indicator_code: indicator_name} for all indicators in those topics.
    Indicators that return no data for 5 test countries are filtered out.
    """
    test_countries = ["USA", "GBR", "CHN", "BRA", "ZAF"]
    result: dict[str, str] = {}
    for topic_id in topic_ids:
        indicators = wbdata.get_indicators(topic=topic_id)
        for ind in indicators:
            code, name = ind["id"], ind["name"]
            has_data = False
            for cc in test_countries:
                df = fetch_indicator(cc, code)
                if not df.empty:
                    has_data = True
                    break
            if has_data:
                result[code] = name
    return result


def get_country_list() -> dict[str, str]:
    """Return {country_code: country_name} for all non-aggregate countries."""
    countries = wbdata.get_countries()
    return {
        c["id"]: c["name"]
        for c in countries
        if c["region"]["id"] != "NA"  # filter out aggregates
    }


def get_indicator_list() -> dict[str, str]:
    """Return {indicator_code: indicator_name}."""
    return INDICATORS


def fetch_indicator(country_code: str, indicator_code: str) -> pd.DataFrame:
    """Pull a time-series for one country + indicator from the World Bank API.

    Returns a DataFrame with columns: year (int), value (float).
    Rows with missing values are dropped.
    """
    try:
        raw = wbdata.get_dataframe(
            {indicator_code: "value"}, country=country_code
        )
    except (TypeError, ValueError, RuntimeError):
        return pd.DataFrame(columns=["year", "value"])
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["year", "value"])
    df = raw.reset_index()
    df = df.rename(columns={"date": "year"})
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("year").reset_index(drop=True)
    return df[["year", "value"]]


def fetch_indicator_pair(
    country_code: str, indicator_1: str, indicator_2: str
) -> pd.DataFrame:
    """Pull two indicators for one country and merge on year.

    Returns a DataFrame with columns: year, value_1, value_2.
    Rows where either value is missing are dropped.
    """
    df1 = fetch_indicator(country_code, indicator_1)
    df2 = fetch_indicator(country_code, indicator_2)
    merged = df1.merge(df2, on="year", suffixes=("_1", "_2"))
    return merged.reset_index(drop=True)


def fetch_multi_country(
    country_codes: list[str], indicator_1: str, indicator_2: str
) -> pd.DataFrame:
    """Fetch an indicator pair for multiple countries.

    Returns a combined DataFrame with an added country_id column.
    """
    frames = []
    for code in country_codes:
        df = fetch_indicator_pair(code, indicator_1, indicator_2)
        df["country_id"] = code
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["year", "value_1", "value_2", "country_id"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("--- Country list (first 5) ---")
    countries = get_country_list()
    for code, name in list(countries.items())[:5]:
        print(f"  {code}: {name}")

    print("\n--- GDP Growth for USA ---")
    df = fetch_indicator("USA", "NY.GDP.MKTP.KD.ZG")
    print(df.head(10))
    print(f"  ... {len(df)} rows total")

    print("\n--- Indicator Pair: GDP Growth vs Inflation for USA ---")
    pair_df = fetch_indicator_pair("USA", "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG")
    print(pair_df.head(10))
    print(f"  ... {len(pair_df)} rows total")

    print("\n--- Multi-country: GDP Growth vs Inflation (USA, GBR) ---")
    multi_df = fetch_multi_country(
        ["USA", "GBR"], "NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG"
    )
    print(multi_df.head(10))
    print(f"  ... {len(multi_df)} rows total")
