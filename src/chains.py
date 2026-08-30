"""
chains.py
---------
Builds the ChatOpenAI model and the reusable LLMChain used for the
structured JSON financial analysis, plus a streaming generator used for
the narrative "typing effect" recommendation.

Functions:
- get_llm()                  -> constructs a ChatOpenAI instance
- build_financial_chain()    -> builds a reusable chain (prompt | llm) for JSON analysis
- run_financial_analysis()   -> invokes the chain and returns raw text
- stream_recommendations()   -> generator that yields narrative text chunks
"""

from __future__ import annotations

from typing import Any, Dict, Iterator

from langchain_openai import ChatOpenAI

from src.prompts import FINANCIAL_CHAT_TEMPLATE, NARRATIVE_CHAT_TEMPLATE, format_expense_breakdown
from src.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


def get_llm(api_key: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, streaming: bool = False) -> ChatOpenAI:
    """
    Construct a ChatOpenAI instance.

    Args:
        api_key: the OpenAI API key supplied by the user (session) or .env.
        model: the model name, e.g. "gpt-4o-mini".
        temperature: sampling temperature.
        streaming: whether this instance will be used with .stream().

    Returns:
        A configured ChatOpenAI client.

    Note: the api_key is passed directly to the client and is never logged,
    printed, or stored anywhere else by this function.
    """
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
        streaming=streaming,
    )


def build_financial_chain(llm: ChatOpenAI):
    """
    Build a reusable "LLMChain"-style pipeline for the structured JSON
    financial analysis using the modern LangChain Expression Language
    (prompt | llm), which replaces the legacy LLMChain class.

    Args:
        llm: a configured ChatOpenAI instance (non-streaming is fine here).

    Returns:
        A runnable chain: FINANCIAL_CHAT_TEMPLATE | llm
    """
    return FINANCIAL_CHAT_TEMPLATE | llm


def run_financial_analysis(llm: ChatOpenAI, calculations: Dict[str, Any], financial_goal: str) -> str:
    """
    Run the reusable financial analysis chain and return the raw text
    response from the model (expected to be a JSON string).

    Args:
        llm: a configured ChatOpenAI instance.
        calculations: the dict returned by financial_calculator.run_financial_calculations().
        financial_goal: the user's selected financial goal.

    Returns:
        The raw string content of the AI's response.
    """
    chain = build_financial_chain(llm)

    inputs = {
        "monthly_income": calculations["monthly_income"],
        "total_expenses": calculations["total_expenses"],
        "remaining_income": calculations["remaining_income"],
        "savings": calculations["savings"],
        "savings_ratio": calculations["savings_ratio"],
        "expense_ratio": calculations["expense_ratio"],
        "financial_goal": financial_goal,
        "expense_breakdown": format_expense_breakdown(calculations["expenses"]),
    }

    response = chain.invoke(inputs)
    return response.content


def stream_recommendations(llm: ChatOpenAI, inputs: Dict[str, Any]) -> Iterator[str]:
    """
    Generator that streams a narrative financial recommendation chunk by chunk.

    Intended to be used directly with `st.write_stream(stream_recommendations(llm, inputs))`.

    Args:
        llm: a ChatOpenAI instance constructed with streaming=True.
        inputs: dict with keys matching NARRATIVE_CHAT_TEMPLATE's variables
            (monthly_income, total_expenses, remaining_income, savings_ratio,
            expense_ratio, preliminary_score, financial_goal).

    Yields:
        String chunks of the narrative recommendation as they arrive.
    """
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)

    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
