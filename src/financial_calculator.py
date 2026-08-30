"""
financial_calculator.py
------------------------
ALL deterministic financial maths lives here. Nothing in this file talks
to the LLM. Given the same inputs, these functions ALWAYS return the same
outputs - that is what makes them "deterministic" as opposed to the
AI-generated insights, which can vary between calls.

Functions:
- calculate_totals()            -> total expenses, remaining income, ratios
- calculate_preliminary_score() -> rule-based 0-100 heuristic score
- run_financial_calculations()  -> convenience wrapper that returns everything
"""

from __future__ import annotations

from typing import Dict, Any


def calculate_totals(monthly_income: float, expenses: Dict[str, float], savings: float) -> Dict[str, float]:
    """
    Compute total expenses, remaining income, savings ratio, and expense ratio.

    Guards against division by zero when monthly_income is 0.

    Args:
        monthly_income: user's gross monthly income.
        expenses: dict mapping expense category -> amount.
        savings: user's current monthly savings amount.

    Returns:
        A dict with total_expenses, remaining_income, savings_ratio, expense_ratio.
    """
    total_expenses = sum(expenses.values())
    remaining_income = monthly_income - total_expenses

    if monthly_income > 0:
        savings_ratio = (savings / monthly_income) * 100
        expense_ratio = (total_expenses / monthly_income) * 100
    else:
        # Division-by-zero guard: with no income, ratios are undefined.
        # We report 0 so the UI can still render safely.
        savings_ratio = 0.0
        expense_ratio = 0.0

    return {
        "total_expenses": round(total_expenses, 2),
        "remaining_income": round(remaining_income, 2),
        "savings_ratio": round(savings_ratio, 2),
        "expense_ratio": round(expense_ratio, 2),
    }


def calculate_preliminary_score(
    monthly_income: float,
    expenses: Dict[str, float],
    savings: float,
    totals: Dict[str, float],
) -> int:
    """
    Compute a rule-based "Preliminary Financial Health Score" from 0 to 100.

    This score is calculated ENTIRELY in Python and is independent from
    whatever score the AI later produces. It exists so users can compare
    a simple, transparent heuristic against the AI's more nuanced view.

    Scoring logic (documented clearly, weights sum to 100):

    1. Savings ratio component (max 35 points)
       - Rewards saving a healthy share of income.
       - 20%+ savings ratio -> full 35 points, scaled linearly below that.

    2. Remaining income component (max 25 points)
       - Rewards having money left over after expenses.
       - Negative remaining income -> 0 points.
       - Remaining income >= 20% of monthly income -> full 25 points.

    3. Expense ratio component (max 25 points)
       - Penalises spending a very high share of income.
       - Expense ratio <= 50% -> full 25 points, scaled down to 0 at 100%+.

    4. Debt burden component (max 15 points)
       - Penalises heavy loan/debt payments relative to income.
       - Debt ratio <= 10% -> full 15 points, scaled down to 0 at 40%+.

    If monthly_income is 0, we cannot meaningfully score anything, so we
    return 0 to avoid a division by zero and to reflect an incomplete
    financial picture.

    Returns:
        An integer financial health score between 0 and 100.
    """
    if monthly_income <= 0:
        return 0

    savings_ratio = totals["savings_ratio"]
    remaining_income = totals["remaining_income"]
    expense_ratio = totals["expense_ratio"]
    debt_amount = expenses.get("debt", 0.0)
    debt_ratio = (debt_amount / monthly_income) * 100

    # 1. Savings ratio component (0-35)
    savings_points = min(35.0, (savings_ratio / 20.0) * 35.0)
    savings_points = max(0.0, savings_points)

    # 2. Remaining income component (0-25)
    remaining_pct_of_income = (remaining_income / monthly_income) * 100
    if remaining_pct_of_income <= 0:
        remaining_points = 0.0
    else:
        remaining_points = min(25.0, (remaining_pct_of_income / 20.0) * 25.0)

    # 3. Expense ratio component (0-25) - lower expense ratio is better
    if expense_ratio <= 50:
        expense_points = 25.0
    elif expense_ratio >= 100:
        expense_points = 0.0
    else:
        # Linearly scale down from 25 points at 50% to 0 points at 100%
        expense_points = 25.0 * (100 - expense_ratio) / 50.0

    # 4. Debt burden component (0-15) - lower debt ratio is better
    if debt_ratio <= 10:
        debt_points = 15.0
    elif debt_ratio >= 40:
        debt_points = 0.0
    else:
        debt_points = 15.0 * (40 - debt_ratio) / 30.0

    total_score = savings_points + remaining_points + expense_points + debt_points
    return int(round(max(0.0, min(100.0, total_score))))


def run_financial_calculations(
    monthly_income: float,
    expenses: Dict[str, float],
    savings: float,
) -> Dict[str, Any]:
    """
    Convenience wrapper: run all Python calculations in one call.

    Returns a single dict containing totals plus the preliminary score,
    ready to be fed into the LangChain prompts.
    """
    totals = calculate_totals(monthly_income, expenses, savings)
    preliminary_score = calculate_preliminary_score(monthly_income, expenses, savings, totals)

    return {
        "monthly_income": round(monthly_income, 2),
        "savings": round(savings, 2),
        "total_expenses": totals["total_expenses"],
        "remaining_income": totals["remaining_income"],
        "savings_ratio": totals["savings_ratio"],
        "expense_ratio": totals["expense_ratio"],
        "preliminary_score": preliminary_score,
        "expenses": expenses,
    }
