# NBA Guide RAG

Pipeline: PDF -> chunks -> Gemini embeddings -> ChromaDB (./data), plus pilot stat tables -> SQLite (nba_stats.db),
queried by a LangChain agent (main.py) that uses both.

## Rules (always)
- Never read, print, log or edit .env. Never hardcode keys.
- Two keys: GEMINI_EMBED_API_KEY (paid, embeddings only), GEMINI_LLM_API_KEY (free tier, chat LLM only).
- Make NO live Gemini calls (embeddings or LLM) without asking me first. Tests must be fully offline/mocked.
- Never modify ./data, never run embed_data.py --reset, never run full ingestion without asking.
- Never write to nba_stats.db until I say "load".
- Python 3.10+, full type hints, no bare except, logging not print (except CLI output).
- Before finishing any task run: `mypy .` (plain, no --follow-imports=skip), `ruff check .`, `pytest -q`, and paste the real output.
- Work one numbered step at a time and stop for my "continue".

## Stack (installed)
chromadb 1.5.9, google-genai 2.30.1, langchain 1.4.6 (use langchain.agents.create_agent), langchain-google-genai 4.4.2,
langgraph 1.2.16, pdfplumber 0.11.10, pandas 3.0.6. Do not use LangChain's Chroma wrapper or embeddings classes;
use embed_texts() from embed_data.py so ingestion and query embeddings cannot drift.