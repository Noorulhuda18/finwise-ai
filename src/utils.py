"""
utils.py
--------
Small, reusable helper functions - most importantly a ROBUST JSON parser
that never allows a malformed LLM response to crash the Streamlit app.

Functions:
- safe_parse_json()   -> tries hard to turn raw LLM text into a Python dict
- get_default_analysis() -> a safe fallback structure if parsing fails
- format_currency()   -> small display helper for money values
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple


def safe_parse_json(raw_text: str) -> Tuple[Dict[str, Any] | None, str | None]:
    """
    Safely parse a JSON object out of raw LLM text.

    Handles common LLM quirks:
    - Accidental ```json ... ``` or ``` ... ``` code fences
    - Extra text/prose before or after the JSON object
    - Leading/trailing whitespace

    Args:
        raw_text: the raw string returned by the LLM.

    Returns:
        A tuple (parsed_dict, error_message).
        - On success: (dict, None)
        - On failure: (None, "human readable error message")
        This function NEVER raises - callers can safely use the result
        without wrapping this call in their own try/except.
    """
    if not raw_text or not raw_text.strip():
        return None, "The AI returned an empty response."

    cleaned = raw_text.strip()

    # 1. Remove ```json ... ``` or ``` ... ``` fences if present.
    fence_pattern = r"^```(?:json)?\s*(.*?)\s*```$"
    fence_match = re.match(fence_pattern, cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Sometimes the fence is only at the start or only at the end.
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    # 2. Try direct parsing first.
    try:
        return json.loads(cleaned), None
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Try to extract the outermost {...} block if there is extra prose
    #    around the JSON object.
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(candidate), None
        except (json.JSONDecodeError, ValueError) as exc:
            return None, f"Could not parse the AI's JSON response: {exc}"

    return None, "No valid JSON object could be found in the AI's response."


def get_default_analysis() -> Dict[str, Any]:
    """
    Return a safe, empty-but-valid analysis structure.

    Used as a fallback when JSON parsing fails, so the Streamlit dashboard
    always has something safe to render instead of crashing.
    """
    return {
        "financial_summary": "The AI response could not be parsed. Please try generating the analysis again.",
        "financial_health_score": 0,
        "spending_analysis": [],
        "risk_level": "UNKNOWN",
        "top_priorities": [],
        "budget_recommendations": [],
        "savings_strategy": [],
        "next_month_action_plan": [],
    }


def format_currency(amount: float, symbol: str = "$") -> str:
    """Format a number as a currency string, e.g. 1234.5 -> '$1,234.50'."""
    try:
        return f"{symbol}{amount:,.2f}"
    except (TypeError, ValueError):
        return f"{symbol}0.00"


def validate_analysis_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all expected keys exist in a parsed analysis dict, filling in
    safe defaults for any that are missing. This protects the UI from
    KeyErrors if the model omits a field.
    """
    defaults = get_default_analysis()
    for key, default_value in defaults.items():
        if key not in data:
            data[key] = default_value
    return data
