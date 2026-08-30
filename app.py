"""
app.py
------
FinWise AI - main Streamlit application.

Run with:
    streamlit run app.py

Architecture (see README.md for full details):
    User Input -> Python Calculations -> Preliminary Score -> LangChain Prompt
    -> OpenAI ChatOpenAI -> Structured JSON -> Safe JSON Parsing -> Dashboard
    -> Streaming Recommendations
"""

from __future__ import annotations

import os
import json
import openai
import streamlit as st
from dotenv import load_dotenv

from src import config
from src.financial_calculator import run_financial_calculations
from src.chains import get_llm, run_financial_analysis, stream_recommendations
from src.cache_manager import configure_cache
from src.utils import safe_parse_json, validate_analysis_structure, format_currency
from src.prompts import demo_message_types

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinWise AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()  # loads OPENAI_API_KEY from .env into os.environ, if present


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Set up every session-state key FinWise AI relies on, if not already set."""
    defaults = {
        "session_api_key": None,          # API key entered manually this session
        "api_setup_complete": False,      # True once a valid key source is confirmed
        "financial_inputs": None,         # last submitted form values
        "calculations": None,             # last Python calculation results
        "latest_analysis": None,          # last parsed AI JSON analysis
        "raw_ai_response": None,          # last raw AI text (for debugging)
        "selected_model": config.DEFAULT_MODEL,
        "selected_cache": config.DEFAULT_CACHE_OPTION,
        "selected_currency": config.DEFAULT_CURRENCY,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# API key resolution helpers
# ---------------------------------------------------------------------------

def get_active_api_key() -> str | None:
    """
    Resolve the API key to use, following this priority:
    1. Session-provided key (entered manually in this session)
    2. .env / environment variable OPENAI_API_KEY
    3. None (no key available)
    """
    if st.session_state.get("session_api_key"):
        return st.session_state["session_api_key"]
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    return None


def reset_api_key() -> None:
    """Remove the session-provided API key and lock the app again."""
    st.session_state["session_api_key"] = None
    st.session_state["api_setup_complete"] = False


def reset_session() -> None:
    """Clear all session data, including the API key, back to defaults."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


# ---------------------------------------------------------------------------
# Educational disclaimer (rendered on every screen)
# ---------------------------------------------------------------------------

def render_disclaimer() -> None:
    st.info(config.EDUCATIONAL_DISCLAIMER, icon="ℹ️")


# ---------------------------------------------------------------------------
# Screen 1: API Key Setup / Welcome screen
# ---------------------------------------------------------------------------

def render_api_key_setup() -> None:
    st.title(f"💰 {config.APP_NAME}")
    st.subheader(config.APP_TAGLINE)
    render_disclaimer()

    st.markdown("---")
    st.markdown("## 🔑 Enter your OpenAI API Key to continue")
    st.markdown(
        "FinWise AI needs an OpenAI API key to communicate with the language model that "
        "powers your personalized financial insights."
    )

    st.warning(
        "**Security notice:** Never share your API key with anyone. FinWise AI keeps your "
        "key only in this browser session's memory - it is never saved to disk, never "
        "written to any file, and never shown again after you submit it.",
        icon="⚠️",
    )

    with st.form("api_key_form", clear_on_submit=False):
        entered_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is stored only in this session and is never displayed again.",
        )
        submitted = st.form_submit_button("Continue to FinWise AI", use_container_width=True)

    if submitted:
        if not entered_key or not entered_key.strip():
            st.error("Please enter a non-empty API key before continuing.")
        else:
            st.session_state["session_api_key"] = entered_key.strip()
            st.session_state["api_setup_complete"] = True
            st.rerun()

    with st.expander("Where do I get an OpenAI API key?"):
        st.markdown(
            "1. Go to [platform.openai.com](https://platform.openai.com) and sign in.\n"
            "2. Open the **API Keys** section of your account settings.\n"
            "3. Click **Create new secret key** and copy it.\n"
            "4. Paste it into the field above. It typically starts with `sk-`."
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(api_key_source: str) -> None:
    with st.sidebar:
        st.markdown(f"## 💰 {config.APP_NAME}")
        st.caption(config.APP_TAGLINE)

        st.markdown("---")
        st.markdown(
            "**About:** FinWise AI combines deterministic Python financial calculations "
            "with LangChain + OpenAI to generate educational budgeting insights."
        )

        st.markdown("---")
        st.caption(config.EDUCATIONAL_DISCLAIMER)

        st.markdown("---")
        st.markdown("### 🔐 API Key Status")
        if api_key_source == "session":
            st.success("API Key: Connected ✓ (this session)")
        elif api_key_source == "env":
            st.success("API Key: Connected ✓ (.env)")
        else:
            st.error("API Key: Not connected")

        if st.button("Change / Reset API Key", use_container_width=True):
            reset_api_key()
            st.rerun()

        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        st.session_state["selected_model"] = st.selectbox(
            "OpenAI Model",
            options=config.AVAILABLE_MODELS,
            index=config.AVAILABLE_MODELS.index(st.session_state["selected_model"])
            if st.session_state["selected_model"] in config.AVAILABLE_MODELS
            else 0,
        )

        st.markdown("### 🗄️ Caching")
        st.session_state["selected_cache"] = st.selectbox(
            "Cache Mode",
            options=config.CACHE_OPTIONS,
            index=config.CACHE_OPTIONS.index(st.session_state["selected_cache"]),
            help=(
                "InMemoryCache: fastest, cleared on restart. "
                "SQLiteCache: saved to disk, survives restarts. "
                "No Cache: always calls the API."
            ),
        )
        configure_cache(st.session_state["selected_cache"])

        st.markdown("---")
        if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
            reset_session()
            st.rerun()


# ---------------------------------------------------------------------------
# Main financial input form
# ---------------------------------------------------------------------------

def render_financial_form() -> dict | None:
    st.markdown("## 📋 Your Monthly Financial Snapshot")
    st.caption("Fill in your numbers below. Nothing here is sent anywhere except to the AI model for analysis.")

    with st.form("financial_form"):
        col1, col2 = st.columns(2)
        with col1:
            monthly_income = st.number_input(
                "Monthly Income", min_value=0.0, step=100.0, value=0.0, format="%.2f"
            )
        with col2:
            currency = st.selectbox(
                "Currency",
                options=list(config.CURRENCIES.keys()),
                index=list(config.CURRENCIES.keys()).index(st.session_state["selected_currency"]),
            )

        st.markdown("#### 💸 Monthly Expenses")
        expense_cols = st.columns(3)
        expenses = {}
        category_items = list(config.EXPENSE_CATEGORIES.items())
        for idx, (label, key) in enumerate(category_items):
            col = expense_cols[idx % 3]
            with col:
                expenses[key] = st.number_input(label, min_value=0.0, step=10.0, value=0.0, format="%.2f", key=f"exp_{key}")

        col3, col4 = st.columns(2)
        with col3:
            savings = st.number_input(
                "Current Monthly Savings", min_value=0.0, step=50.0, value=0.0, format="%.2f"
            )
        with col4:
            financial_goal = st.selectbox("Financial Goal", options=config.FINANCIAL_GOALS)

        submitted = st.form_submit_button("🚀 Generate AI Financial Analysis", use_container_width=True)

    if submitted:
        if monthly_income <= 0:
            st.error("Please enter a monthly income greater than 0 to generate an analysis.")
            return None

        st.session_state["selected_currency"] = currency
        return {
            "monthly_income": monthly_income,
            "expenses": expenses,
            "savings": savings,
            "financial_goal": financial_goal,
            "currency": currency,
        }

    return None


# ---------------------------------------------------------------------------
# Financial overview metrics
# ---------------------------------------------------------------------------

def render_overview_metrics(calculations: dict, currency_symbol: str) -> None:
    st.markdown("## 📊 Financial Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monthly Income", format_currency(calculations["monthly_income"], currency_symbol))
    col2.metric("Total Expenses", format_currency(calculations["total_expenses"], currency_symbol))
    col3.metric(
        "Remaining Balance",
        format_currency(calculations["remaining_income"], currency_symbol),
        delta=f"{calculations['remaining_income']:.2f}",
    )
    col4.metric("Current Savings", format_currency(calculations["savings"], currency_symbol))


# ---------------------------------------------------------------------------
# AI Analysis Dashboard
# ---------------------------------------------------------------------------

def render_score_section(preliminary_score: int, analysis: dict) -> None:
    st.markdown("## 🧠 AI Analysis Dashboard")
    render_disclaimer()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🐍 Python Preliminary Score")
        st.caption("Calculated deterministically in Python, independent of the AI.")
        st.progress(preliminary_score / 100)
        st.metric("Preliminary Score", f"{preliminary_score}/100", help=config.get_score_band(preliminary_score))
        st.caption(f"Band: **{config.get_score_band(preliminary_score)}**")

    with col2:
        ai_score = analysis.get("financial_health_score", 0)
        try:
            ai_score = int(ai_score)
        except (TypeError, ValueError):
            ai_score = 0
        ai_score = max(0, min(100, ai_score))

        st.markdown("#### 🤖 AI Financial Health Score")
        st.caption("Generated by the language model based on your data - may differ from the Python score.")
        st.progress(ai_score / 100)
        st.metric("AI Score", f"{ai_score}/100", help=config.get_score_band(ai_score))
        st.caption(f"Band: **{config.get_score_band(ai_score)}**")

    st.caption(
        "🔎 Note: the Python Preliminary Score and the AI Financial Health Score are computed "
        "independently and may not match - this is expected and helps illustrate the difference "
        "between rule-based and AI-based analysis."
    )


def render_risk_level(risk_level: str) -> None:
    st.markdown("### ⚠️ Risk Level")
    risk_level = (risk_level or "UNKNOWN").upper()
    caption = "This is an educational observation, not a guaranteed prediction."
    if risk_level == "LOW":
        st.success(f"Risk Level: **LOW** — {caption}")
    elif risk_level == "MEDIUM":
        st.warning(f"Risk Level: **MEDIUM** — {caption}")
    elif risk_level == "HIGH":
        st.error(f"Risk Level: **HIGH** — {caption}")
    else:
        st.info(f"Risk Level: **{risk_level}** — {caption}")


def render_analysis_tabs(analysis: dict) -> None:
    tabs = st.tabs(
        ["📝 Summary", "🔍 Spending Analysis", "🎯 Priorities & Budget", "💡 Savings & Action Plan"]
    )

    with tabs[0]:
        st.markdown("#### Financial Summary")
        st.info(analysis.get("financial_summary", "No summary available."))
        render_risk_level(analysis.get("risk_level", "UNKNOWN"))

    with tabs[1]:
        st.markdown("#### Spending Analysis by Category")
        spending_analysis = analysis.get("spending_analysis", [])
        if not spending_analysis:
            st.warning("No spending analysis was returned.")
        for item in spending_analysis:
            with st.expander(f"📁 {item.get('category', 'Category')}"):
                st.markdown(f"**Observation:** {item.get('observation', 'N/A')}")
                st.markdown(f"**Recommendation:** {item.get('recommendation', 'N/A')}")

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Top Priorities")
            priorities = analysis.get("top_priorities", [])
            if priorities:
                for p in priorities:
                    st.markdown(f"- {p}")
            else:
                st.caption("No priorities returned.")
        with col2:
            st.markdown("#### 💰 Budget Recommendations")
            budget_recs = analysis.get("budget_recommendations", [])
            if budget_recs:
                for b in budget_recs:
                    st.markdown(f"- {b}")
            else:
                st.caption("No budget recommendations returned.")

    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏦 Savings Strategy")
            savings_strategy = analysis.get("savings_strategy", [])
            if savings_strategy:
                for s in savings_strategy:
                    st.markdown(f"- {s}")
            else:
                st.caption("No savings strategy returned.")
        with col2:
            st.markdown("#### ✅ Next Month Action Plan")
            action_plan = analysis.get("next_month_action_plan", [])
            if action_plan:
                for a in action_plan:
                    st.markdown(f"- {a}")
            else:
                st.caption("No action plan returned.")


# ---------------------------------------------------------------------------
# Main dashboard flow (after form submission)
# ---------------------------------------------------------------------------

def run_analysis_pipeline(api_key: str, financial_inputs: dict) -> None:
    """Run the full Python -> LangChain -> JSON pipeline and store results in session state."""
    calculations = run_financial_calculations(
        monthly_income=financial_inputs["monthly_income"],
        expenses=financial_inputs["expenses"],
        savings=financial_inputs["savings"],
    )
    st.session_state["calculations"] = calculations

    try:
        llm = get_llm(api_key=api_key, model=st.session_state["selected_model"], streaming=False)
        with st.spinner("FinWise AI is analyzing your finances..."):
            raw_response = run_financial_analysis(
                llm=llm,
                calculations=calculations,
                financial_goal=financial_inputs["financial_goal"],
            )
        st.session_state["raw_ai_response"] = raw_response

        parsed, error = safe_parse_json(raw_response)
        if error or parsed is None:
            st.session_state["latest_analysis"] = None
            st.error(f"The AI response could not be parsed as JSON: {error}")
            with st.expander("🔧 Debug: raw AI response"):
                st.code(raw_response or "(empty response)")
        else:
            st.session_state["latest_analysis"] = validate_analysis_structure(parsed)

    except openai.AuthenticationError:
        st.error("Your OpenAI API key was rejected. Please check it and try again via 'Change / Reset API Key'.")
    except openai.RateLimitError:
        st.error("OpenAI rate limit reached. Please wait a moment and try again.")
    except openai.APIConnectionError:
        st.error("Could not connect to OpenAI. Please check your internet connection and try again.")
    except openai.APIError as exc:
        st.error(f"OpenAI API returned an error: {exc}")
    except Exception as exc:  # noqa: BLE001 - final safety net so the app never crashes
        st.error(f"An unexpected error occurred while generating the analysis: {exc}")


def render_streaming_section(api_key: str) -> None:
    calculations = st.session_state.get("calculations")
    financial_inputs = st.session_state.get("financial_inputs")
    if not calculations or not financial_inputs:
        return

    st.markdown("## ✍️ Personalized Narrative Recommendation")
    st.caption("Streamed live from the AI model for a natural, typing-style effect.")

    if st.button("Generate Streaming Recommendation"):
        try:
            streaming_llm = get_llm(api_key=api_key, model=st.session_state["selected_model"], streaming=True)
            stream_inputs = {
                "monthly_income": calculations["monthly_income"],
                "total_expenses": calculations["total_expenses"],
                "remaining_income": calculations["remaining_income"],
                "savings_ratio": calculations["savings_ratio"],
                "expense_ratio": calculations["expense_ratio"],
                "preliminary_score": calculations["preliminary_score"],
                "financial_goal": financial_inputs["financial_goal"],
            }
            st.write_stream(stream_recommendations(streaming_llm, stream_inputs))
        except openai.AuthenticationError:
            st.error("Your OpenAI API key was rejected. Please check it and try again.")
        except openai.RateLimitError:
            st.error("OpenAI rate limit reached. Please wait a moment and try again.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"An unexpected error occurred while streaming: {exc}")


def render_message_demo() -> None:
    calculations = st.session_state.get("calculations")
    if not calculations:
        return
    with st.expander("🧩 Developer Demo: SystemMessage / HumanMessage / AIMessage"):
        st.caption(
            "This demonstrates how a LangChain conversation is represented internally "
            "using message objects (for learning purposes only - not used for the real analysis)."
        )
        summary_line = (
            f"income={calculations['monthly_income']}, "
            f"expenses={calculations['total_expenses']}, "
            f"savings_ratio={calculations['savings_ratio']}%"
        )
        messages = demo_message_types(summary_line)
        for msg in messages:
            role = msg.__class__.__name__
            st.markdown(f"**{role}:**")
            st.code(msg.content, language="text")


# ---------------------------------------------------------------------------
# Main app flow
# ---------------------------------------------------------------------------

def main() -> None:
    active_key = get_active_api_key()

    # Determine the source of the active key for the sidebar status display.
    if st.session_state.get("session_api_key"):
        key_source = "session"
    elif os.getenv("OPENAI_API_KEY"):
        key_source = "env"
    else:
        key_source = None

    if not active_key:
        render_api_key_setup()
        return

    st.session_state["api_setup_complete"] = True
    render_sidebar(key_source)

    st.title(f"💰 {config.APP_NAME}")
    st.subheader(config.APP_TAGLINE)
    render_disclaimer()
    st.markdown("---")

    financial_inputs = render_financial_form()
    if financial_inputs is not None:
        st.session_state["financial_inputs"] = financial_inputs
        run_analysis_pipeline(active_key, financial_inputs)

    calculations = st.session_state.get("calculations")
    analysis = st.session_state.get("latest_analysis")
    inputs = st.session_state.get("financial_inputs")

    if calculations and inputs:
        st.markdown("---")
        currency_symbol = config.CURRENCIES.get(inputs.get("currency", "USD"), "$")
        render_overview_metrics(calculations, currency_symbol)

        if analysis:
            st.markdown("---")
            render_score_section(calculations["preliminary_score"], analysis)
            st.markdown("---")
            render_analysis_tabs(analysis)
            st.markdown("---")
            render_streaming_section(active_key)
            st.markdown("---")
            render_message_demo()
        else:
            st.warning(
                "No AI analysis is available yet. Submit the form above to generate one, "
                "or check the debug section if a previous attempt failed."
            )


if __name__ == "__main__":
    main()
