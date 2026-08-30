"""
prompts.py
----------
All LangChain prompt engineering lives here:

- SYSTEM_PROMPT_TEXT         : shared safety + role instructions for the AI
- JSON_SCHEMA_INSTRUCTIONS   : the exact JSON structure the AI must return
- FINANCIAL_PROMPT_TEMPLATE  : a reusable `PromptTemplate` (single string)
- FINANCIAL_CHAT_TEMPLATE    : a reusable `ChatPromptTemplate` (structured JSON analysis)
- NARRATIVE_CHAT_TEMPLATE    : a `ChatPromptTemplate` used for the streamed narrative summary
- demo_message_types()       : a small demo of SystemMessage / HumanMessage / AIMessage
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# 1. Shared system instructions (AI safety rules)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEXT = """You are FinWise AI, an educational personal-finance assistant embedded in a
Streamlit prototype application. Follow these safety rules at all times:

1. You are an EDUCATIONAL assistant only - you are not a licensed financial advisor.
2. Never claim that any financial outcome, investment return, or prediction is guaranteed.
3. Never claim certainty about future investment returns or market performance.
4. You do not execute financial transactions of any kind.
5. You do not have access to, and must never pretend to have access to, the user's real bank accounts.
6. Do not present risk levels or forecasts as guaranteed facts - always frame them as educational
   observations based only on the numbers provided.
7. Always encourage the user to consult a qualified financial professional for significant decisions.
8. Base your entire analysis strictly on the numbers the user provided - never invent income,
   expenses, account balances, or other financial information that was not given to you.
9. Clearly distinguish between an "observation" (what the numbers show) and a "recommendation"
   (an educational suggestion) - do not blend the two.
10. Do not claim to be human, do not claim to be a licensed financial advisor, and do not claim
    any regulatory certification.
"""

# ---------------------------------------------------------------------------
# 2. Structured JSON schema instructions
# ---------------------------------------------------------------------------

JSON_SCHEMA_INSTRUCTIONS = """Return your ENTIRE response as a single valid JSON object and NOTHING else.
Do not include Markdown code fences (no ```json). Do not include any text before or after the JSON.

Use EXACTLY this structure and these keys:

{{
  "financial_summary": "string - a short, plain-language educational summary of the user's situation",
  "financial_health_score": 0,
  "spending_analysis": [
    {{
      "category": "string - expense category name",
      "observation": "string - a factual observation based on the numbers",
      "recommendation": "string - an educational suggestion related to this category"
    }}
  ],
  "risk_level": "LOW, MEDIUM, or HIGH",
  "top_priorities": ["string", "string"],
  "budget_recommendations": ["string", "string"],
  "savings_strategy": ["string", "string"],
  "next_month_action_plan": ["string", "string"]
}}

The "financial_health_score" must be an integer between 0 and 100, using these educational bands:
80-100 Strong, 60-79 Generally Healthy, 40-59 Needs Improvement, below 40 High Attention.
"""

# ---------------------------------------------------------------------------
# 3. PromptTemplate - reusable single-string template
# ---------------------------------------------------------------------------
# Required variables: monthly_income, total_expenses, remaining_income, savings,
# savings_ratio, expense_ratio, financial_goal, expense_breakdown

FINANCIAL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=[
        "monthly_income",
        "total_expenses",
        "remaining_income",
        "savings",
        "savings_ratio",
        "expense_ratio",
        "financial_goal",
        "expense_breakdown",
    ],
    template="""Analyze the following user's monthly financial snapshot for educational purposes only.

Monthly income: {monthly_income}
Total expenses: {total_expenses}
Remaining income after expenses: {remaining_income}
Current monthly savings: {savings}
Savings ratio (% of income saved): {savings_ratio}%
Expense ratio (% of income spent): {expense_ratio}%
Financial goal: {financial_goal}

Expense breakdown by category:
{expense_breakdown}

""" + JSON_SCHEMA_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# 4. ChatPromptTemplate - System + Human messages for the structured analysis
# ---------------------------------------------------------------------------

FINANCIAL_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEXT),
        (
            "human",
            """Here is my monthly financial snapshot. Python has already performed the
calculations below - please analyze them, you do not need to recompute them.

Monthly income: {monthly_income}
Total expenses: {total_expenses}
Remaining income after expenses: {remaining_income}
Current monthly savings: {savings}
Savings ratio (% of income saved): {savings_ratio}%
Expense ratio (% of income spent): {expense_ratio}%
Financial goal: {financial_goal}

Expense breakdown by category:
{expense_breakdown}

"""
            + JSON_SCHEMA_INSTRUCTIONS,
        ),
    ]
)


# ---------------------------------------------------------------------------
# 5. ChatPromptTemplate - used for the streamed, human-readable narrative
# ---------------------------------------------------------------------------

NARRATIVE_CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEXT),
        (
            "human",
            """Based on this financial snapshot, write a short, warm, encouraging educational
narrative (4-6 sentences, plain text, NO JSON, NO Markdown headers) that walks the user
through their situation and 2-3 practical next steps tied to their goal of "{financial_goal}".

Monthly income: {monthly_income}
Total expenses: {total_expenses}
Remaining income: {remaining_income}
Savings ratio: {savings_ratio}%
Expense ratio: {expense_ratio}%
Preliminary Python financial health score: {preliminary_score}/100

Remember: this is educational only, never guarantee outcomes, and encourage professional
advice for significant decisions.""",
        ),
    ]
)


# ---------------------------------------------------------------------------
# 6. Demonstration of SystemMessage / HumanMessage / AIMessage
# ---------------------------------------------------------------------------

def demo_message_types(financial_summary_line: str) -> list:
    """
    Build a small demonstration conversation showing how SystemMessage,
    HumanMessage, and AIMessage represent a LangChain conversation.

    This is purely illustrative (used in an expander in the UI) and is not
    required for the main analysis pipeline to function.

    Args:
        financial_summary_line: a one-line summary of the user's numbers,
            used to make the demo feel connected to the user's real data.

    Returns:
        A list of LangChain message objects: [SystemMessage, HumanMessage, AIMessage].
    """
    system_msg = SystemMessage(
        content="You are FinWise AI, an educational financial assistant. Never guarantee outcomes."
    )
    human_msg = HumanMessage(
        content=f"Here is my financial snapshot: {financial_summary_line}. What should I focus on?"
    )
    ai_msg = AIMessage(
        content=(
            "Based on your numbers, focus first on building a small emergency buffer, then "
            "review your largest expense category. This is educational guidance only - please "
            "consult a financial professional for major decisions."
        )
    )
    return [system_msg, human_msg, ai_msg]


def format_expense_breakdown(expenses: dict) -> str:
    """
    Turn a {category: amount} dict into a readable multi-line string for prompts.

    Example:
        Housing / Rent: 1200
        Food: 400
    """
    lines = [f"- {category}: {amount}" for category, amount in expenses.items()]
    return "\n".join(lines) if lines else "- No expenses provided"
