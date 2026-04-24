"""LLM-based integrity flagging using LangChain structured output."""

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.globals import set_debug
import database as db
from comtrade_api import get_trade_summary
from data_ingestion import INDICATORS
from ntl_api import get_ntl_summary

set_debug(False)

class IntegrityAssessment(BaseModel):
    """Structured LLM response for a flagged data point."""

    is_anomaly: bool = Field(
        description="Whether this is a genuine data integrity issue"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the assessment (0.0-1.0)"
    )
    explanation: str = Field(
        description="Rationale for the assessment in 2-3 short sentences, max 60 words."
    )


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an economic data integrity analyst. Assess whether a "
        "flagged data point represents a genuine data integrity issue or "
        "has a plausible explanation.",
    ),
    (
        "human",
        "For {country} in {year}:\n"
        "- {indicator_1_name}: {value_1}\n"
        "- {indicator_2_name}: {value_2}\n"
        "- Expected correlation between these indicators: {expected_correlation:.3f}\n"
        "- Political regime (V-Dem): {regime_context}\n"
        "- Trade context (UN Comtrade): {trade_context}\n"
        "- Nighttime lights (satellite): {ntl_context}\n\n"
        "This data point has been flagged because the indicators appear to "
        "conflict with their expected correlation pattern.\n\n"
        "Assess whether this is a genuine data integrity issue or has a "
        "plausible explanation. Keep the explanation to 2-3 short sentences "
        "(max 60 words). Be concise and direct — no preamble, no restating "
        "the question.",
    ),
])

_chain = None


def _mentions_electricity(name: str) -> bool:
    """True when an indicator name refers to electricity (gates NTL context)."""
    n = name.lower()
    return "electric" in n or "electricity" in n or "power consumption" in n


def _get_chain():
    global _chain
    if _chain is None:
        llm = ChatOllama(
            model="qwen3.5:4b",
            temperature=0.0,
            reasoning=False,
        )
        _chain = _PROMPT | llm.with_structured_output(
            IntegrityAssessment, method="function_calling")
    return _chain


def assess_integrity(
    row: dict,
    *,
    indicator_1_name: str | None = None,
    indicator_2_name: str | None = None,
) -> IntegrityAssessment:
    """Assess a single flagged data point using the LLM.

    Parameters
    ----------
    row : dict with keys: country_id, year, value_1, value_2,
          expected_correlation, indicator_1 (code), indicator_2 (code)
    indicator_1_name : optional display name (falls back to INDICATORS dict or code)
    indicator_2_name : optional display name (falls back to INDICATORS dict or code)

    Returns
    -------
    IntegrityAssessment
    """
    name_1 = indicator_1_name or INDICATORS.get(row["indicator_1"], row["indicator_1"])
    name_2 = indicator_2_name or INDICATORS.get(row["indicator_2"], row["indicator_2"])
    trade_context = get_trade_summary(row["country_id"], row["year"]) or "unavailable"
    if _mentions_electricity(name_1) or _mentions_electricity(name_2):
        ntl_context = get_ntl_summary(row["country_id"], row["year"]) or "unavailable"
    else:
        ntl_context = "not applicable"
    regime = db.get_regime_with_fallback(row["country_id"], row["year"])
    if regime is None:
        regime_context = "unavailable"
    elif regime["carried_forward"]:
        regime_context = f"{regime['regime_label']} (last known {regime['source_year']})"
    else:
        regime_context = regime["regime_label"]
    return _get_chain().invoke({
        "country": row["country_id"],
        "year": row["year"],
        "indicator_1_name": name_1,
        "indicator_2_name": name_2,
        "value_1": row["value_1"],
        "value_2": row["value_2"],
        "expected_correlation": row["expected_correlation"],
        "regime_context": regime_context,
        "trade_context": trade_context,
        "ntl_context": ntl_context,
    })


if __name__ == "__main__":
    # Test with a hardcoded flagged row
    test_row = {
        "country_id": "USA",
        "year": 2020,
        "value_1": -3.4,   # GDP Growth
        "value_2": 1.23,   # Inflation
        "expected_correlation": -0.15,
        "indicator_1": "NY.GDP.MKTP.KD.ZG",
        "indicator_2": "FP.CPI.TOTL.ZG",
    }
    print("Assessing hardcoded flagged row ...")
    result = assess_integrity(test_row)
    print(f"  is_anomaly: {result.is_anomaly}")
    print(f"  confidence: {result.confidence_score}")
    print(f"  explanation: {result.explanation}")
