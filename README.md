https://finwise-ai-tkanejsa6kgun9nqqqfhwg.streamlit.app/
# 💰 FinWise AI

### AI-Powered Personal Financial Analysis & Smart Budget Assistant

FinWise AI is a beginner-friendly, educational Streamlit application that combines deterministic
Python financial calculations with a LangChain + OpenAI powered assistant to produce personalized,
structured budgeting insights.

> **⚠️ Educational Prototype Notice**
> FinWise AI is built for **learning purposes only**. It does **not** provide guaranteed investment
> advice, does **not** execute financial transactions, does **not** connect to real bank accounts, and
> does **not** guarantee any financial outcome. Always consult a qualified financial professional
> before making important financial decisions.

---

## 1. Project Overview

Users enter their monthly income, expenses (across 9 categories), current savings, and a financial
goal. Python performs deterministic calculations (totals, ratios, a preliminary score), and a
LangChain-orchestrated OpenAI model then analyzes those numbers to produce a structured financial
health assessment, a risk level, and actionable, streamed recommendations.

## 2. Features

- 🔑 In-app OpenAI API key setup screen (session-only, never persisted)
- 📋 Professional financial input form (income, 9 expense categories, savings, goal, currency)
- 🧮 Deterministic Python calculations, fully separated from AI output
- 🤖 LangChain `ChatOpenAI` integration with a configurable model
- 📝 Reusable `PromptTemplate` and `ChatPromptTemplate`
- 🧩 `SystemMessage` / `HumanMessage` / `AIMessage` demonstration
- 🔗 Reusable chain for structured financial analysis
- 📦 Strict JSON output schema with safe, crash-proof parsing
- ✍️ Streamed narrative recommendations (`llm.stream()` + `st.write_stream()`)
- 🗄️ Switchable caching: No Cache / `InMemoryCache` / `SQLiteCache`
- 📊 Modern dashboard: metrics, progress bars, tabs, expanders, alert boxes
- 🔐 Security-first API key handling throughout

## 3. Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| UI | Streamlit |
| LLM framework | LangChain (`langchain`, `langchain-openai`, `langchain-community`, `langchain-core`) |
| Model provider | OpenAI (default `gpt-4o-mini`) |
| Secrets | `python-dotenv` (optional `.env` file) |
| Caching | `InMemoryCache` + `SQLiteCache` |

## 4. Architecture

```
User Input
   ↓
Python Financial Calculations   (src/financial_calculator.py)
   ↓
Rule-Based Preliminary Score    (src/financial_calculator.py)
   ↓
LangChain Prompt                (src/prompts.py)
   ↓
OpenAI ChatOpenAI               (src/chains.py)
   ↓
Structured JSON
   ↓
Safe JSON Parsing               (src/utils.py)
   ↓
Streamlit Dashboard             (app.py)
   ↓
Streaming Recommendations       (src/chains.py -> app.py)
```

Python calculations and LLM-generated insights are **strictly separated** — the app always shows
both a **Python Preliminary Score** and a distinct **AI Financial Health Score** side by side.

## 5. Project Structure

```
finwise_ai/
│
├── app.py                 # Streamlit UI - run this file
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py               # settings + form options
│   ├── prompts.py               # PromptTemplate + ChatPromptTemplate + JSON schema
│   ├── financial_calculator.py  # deterministic maths - no AI
│   ├── chains.py                # ChatOpenAI, chain, streaming
│   ├── cache_manager.py         # InMemoryCache + SQLiteCache
│   └── utils.py                 # safe JSON parsing + helpers
│
└── docs/
    └── FinTech_AI_Assignment.pdf
```

## 6. Installation

```bash
# 1. Clone or download this project, then move into the folder
cd finwise_ai
```

## 7. Virtual Environment Setup

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 8. API Key Setup (.env method)

1. Create an account at [platform.openai.com](https://platform.openai.com) and generate an API key.
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and set:
   ```
   OPENAI_API_KEY=sk-your-real-key-here
   ```
4. `.env` is already listed in `.gitignore` — **never commit your real key**.

## 9. In-App API Key Setup

If no `.env` key is found, FinWise AI shows a welcome screen asking you to paste your OpenAI API key
directly into a password-style field. This key:

- Is stored **only** in `st.session_state` for the current browser session.
- Is **never** written to disk, a database, a log, or a generated file.
- Is **never** displayed again after submission (the sidebar only shows "API Key: Connected ✓").
- Can be removed at any time using **"Change / Reset API Key"** in the sidebar.

**Key resolution priority:** session-provided key → `.env` key → API key setup screen.

## 10. How to Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

## 11. Python Calculations

All deterministic maths lives in `src/financial_calculator.py`:

```python
total_expenses    = sum(all_expense_categories)
remaining_income  = monthly_income - total_expenses
savings_ratio     = (savings / monthly_income) * 100
expense_ratio     = (total_expenses / monthly_income) * 100
```

A **Preliminary Financial Health Score (0–100)** is also computed here using a transparent, weighted
heuristic across savings ratio (35 pts), remaining income (25 pts), expense ratio (25 pts), and debt
burden (15 pts). Division by zero is guarded whenever income is 0.

## 12. PromptTemplate

`src/prompts.py` defines `FINANCIAL_PROMPT_TEMPLATE`, a reusable `PromptTemplate` with the required
variables: `monthly_income`, `total_expenses`, `remaining_income`, `savings`, `savings_ratio`,
`expense_ratio`, `financial_goal`, `expense_breakdown`.

## 13. ChatPromptTemplate

`FINANCIAL_CHAT_TEMPLATE` combines a **System Message** (FinWise AI's role and safety rules) with a
**Human Message** (the user's dynamically inserted financial data) to request the structured JSON
analysis. A second template, `NARRATIVE_CHAT_TEMPLATE`, is used for the streamed narrative summary.

## 14. SystemMessage / HumanMessage / AIMessage

`src/prompts.py` includes `demo_message_types()`, which builds a small illustrative conversation using
`SystemMessage`, `HumanMessage`, and `AIMessage`. You can view this demo in an expander at the bottom
of the dashboard after generating an analysis.

## 15. LLMChain

`src/chains.py` builds a reusable chain (`FINANCIAL_CHAT_TEMPLATE | llm`) using modern LangChain
Expression Language, which is the current recommended replacement for the legacy `LLMChain` class
while serving the same purpose: piping the prompt template's output into the model.

## 16. Structured JSON

The AI is instructed to return **only** raw JSON (no Markdown fences) matching this schema:

```json
{
  "financial_summary": "",
  "financial_health_score": 0,
  "spending_analysis": [
    { "category": "", "observation": "", "recommendation": "" }
  ],
  "risk_level": "",
  "top_priorities": [],
  "budget_recommendations": [],
  "savings_strategy": [],
  "next_month_action_plan": []
}
```

`src/utils.py` provides `safe_parse_json()`, which strips accidental code fences, trims stray text
around the JSON object, and never raises — a friendly error plus a debug expander is shown instead if
parsing fails, and the app keeps running.

## 17. Streaming

`src/chains.py` exposes `stream_recommendations()`, a generator that yields text chunks from
`llm.stream()`. The Streamlit UI renders it live with `st.write_stream()` for a natural typing effect.

## 18. InMemoryCache

Stored in RAM via LangChain's `InMemoryCache`. Fastest option; cleared when the app restarts. Good for
quickly repeating the same analysis within one session.

## 19. SQLiteCache

Stored on disk (`.finwise_cache.db`) via LangChain's `SQLiteCache`. Slightly slower than in-memory but
**survives app restarts**, so identical requests avoid unnecessary API calls even across sessions.

Both are configured in `src/cache_manager.py` via `set_llm_cache(...)`, selectable from the sidebar.
**The user's API key is never written to the cache** — only prompts and completions are cached.

## 20. Python Score vs AI Score

FinWise AI **always displays both scores separately and labeled**:

- **Python Preliminary Score** — a transparent, rule-based heuristic computed with plain arithmetic.
- **AI Financial Health Score** — generated by the language model's holistic read of your situation.

They may differ — this is expected and is a deliberate teaching point about the difference between
rule-based and AI-based analysis. Both use the same educational bands:

| Score | Label |
|---|---|
| 80–100 | Strong |
| 60–79 | Generally Healthy |
| 40–59 | Needs Improvement |
| Below 40 | High Attention |

## 21. Security Considerations

- The API key is **never** hard-coded, printed, logged, saved to SQLite, written to any file, or
  shown in the UI after submission.
- The key lives only in `st.session_state` for the current session, or is read from environment
  variables via `.env` (which is git-ignored).
- Errors are caught and shown as friendly messages that never leak the key or other sensitive details.
- `.gitignore` excludes `.env`, `*.db` cache files, `__pycache__/`, and virtual environments.

## 22. Testing

Manually test these scenarios (see `docs/FinTech_AI_Assignment.pdf` for full detail):

| # | Scenario | Expect |
|---|---|---|
| 1 | Income 8000, expenses ~2000 | High score, LOW risk, growth-focused tips |
| 2 | Income 2000, expenses ~2600 | Negative remaining income, HIGH risk, cost-cutting |
| 3 | Income 5000, debt 2500 | High debt burden, MEDIUM/HIGH risk |
| 4 | Income 4000, savings 1200 | ~30% savings ratio, strong score, LOW risk |
| 5 | Income 3000, expenses 3000 | Remaining = 0, MEDIUM/HIGH risk |
| 6 | No API key present | Setup screen appears; app is locked |
| 7 | Invalid API key | Friendly error; app does not crash; key never shown |
| 8 | Same inputs submitted twice with caching on | Second run uses cache, no extra API call |

## 23. Limitations

- This is a prototype: it does not connect to real bank accounts or execute any transactions.
- AI-generated content can vary between runs and should be treated as educational, not authoritative.
- Currency values are for display only; no live exchange-rate conversion is performed.
- Caching is local to the machine running the app (SQLite file) or the running process (in-memory).

## 24. Educational Disclaimer

FinWise AI is an **educational prototype only**. It does not provide guaranteed investment advice, does
not execute financial transactions, does not connect to real bank accounts, and does not guarantee any
financial outcome. Always consult a qualified, licensed financial professional for decisions that
matter to your real finances.
