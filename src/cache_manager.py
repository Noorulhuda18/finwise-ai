"""
cache_manager.py
----------------
Configures LangChain's global LLM cache.

FinWise AI supports three modes:
- "No Cache"       : every request calls the OpenAI API.
- "InMemoryCache"   : cached in RAM for this run only (fastest, lost on restart).
- "SQLiteCache"     : cached to a local .db file (slightly slower, survives restarts).

IMPORTANT: The cache stores prompts and completions ONLY. The user's API key
is NEVER written to the cache, because the API key is never part of the
prompt text sent to the model.
"""

from __future__ import annotations

from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

from src.config import SQLITE_CACHE_PATH


def configure_cache(cache_option: str) -> None:
    """
    Set LangChain's global cache based on the user's selection.

    Args:
        cache_option: one of "No Cache", "InMemoryCache", "SQLiteCache".
            Any unrecognised value disables caching (safe default).
    """
    if cache_option == "InMemoryCache":
        set_llm_cache(InMemoryCache())
    elif cache_option == "SQLiteCache":
        set_llm_cache(SQLiteCache(database_path=SQLITE_CACHE_PATH))
    else:
        # "No Cache" or any unknown value -> disable caching entirely.
        set_llm_cache(None)
