"""
config.py
---------
Centralised, non-secret configuration for FinWise AI.

This file holds:
- App metadata (name, tagline, disclaimer text)
- Dropdown / selectbox options used by the Streamlit form
- Default model + cache settings
- Educational score bands

IMPORTANT: This file must NEVER contain an actual API key.
API keys are handled only in `app.py` via `st.session_state` or a `.env` file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

APP_NAME = "FinWise AI"
APP_TAGLINE = "Your AI-Powered Personal Financial Analysis & Smart Budget Assistant"

EDUCATIONAL_DISCLAIMER = (
    "**Educational Prototype Notice:** FinWise AI is built for learning purposes only. "
    "It does **not** provide guaranteed investment advice, does **not** execute financial "
    "transactions, does **not** connect to real bank accounts, and does **not** guarantee "
    "any financial outcome. Please consult a qualified financial professional before making "
    "important financial decisions."
)

# ---------------------------------------------------------------------------
# LLM model options
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.4

# ---------------------------------------------------------------------------
# Cache options
# ---------------------------------------------------------------------------

CACHE_OPTIONS = ["No Cache", "InMemoryCache", "SQLiteCache"]
DEFAULT_CACHE_OPTION = "InMemoryCache"
SQLITE_CACHE_PATH = ".finwise_cache.db"

# ---------------------------------------------------------------------------
# Financial goal options
# ---------------------------------------------------------------------------

FINANCIAL_GOALS = [
    "Save money",
    "Emergency fund",
    "Pay off debt",
    "Vacation",
    "Start a business",
    "Improve budgeting",
]

# ---------------------------------------------------------------------------
# Currency options
# ---------------------------------------------------------------------------

CURRENCIES = {
    "USD": "$",
    "PKR": "Rs",
    "EUR": "€",
    "GBP": "£",
    "AUD": "A$",
    "CAD": "C$",
    "INR": "₹",
    "AED": "AED",
}
DEFAULT_CURRENCY = "USD"

# ---------------------------------------------------------------------------
# Expense categories (label -> internal key)
# ---------------------------------------------------------------------------

EXPENSE_CATEGORIES = {
    "Housing / Rent": "housing",
    "Food": "food",
    "Transportation": "transportation",
    "Utilities": "utilities",
    "Education": "education",
    "Healthcare": "healthcare",
    "Entertainment": "entertainment",
    "Loan / Debt": "debt",
    "Other": "other",
}

# ---------------------------------------------------------------------------
# Educational score bands (used for BOTH Python and AI scores)
# ---------------------------------------------------------------------------

SCORE_BANDS = [
    (80, 100, "Strong"),
    (60, 79, "Generally Healthy"),
    (40, 59, "Needs Improvement"),
    (0, 39, "High Attention"),
]


def get_score_band(score: float) -> str:
    """Return the educational label for a given 0-100 score."""
    score = max(0, min(100, score))
    for low, high, label in SCORE_BANDS:
        if low <= score <= high:
            return label
    return "Unknown"
