# NBA Guide RAG

Pipeline: PDF -> chunks -> Gemini embeddings -> ChromaDB (./data), plus pilot stat tables -> SQLite (nba_stats.db),
queried by a LangChain agent (main.py) that uses both.

## Rules (always)
- Never read, print, log or edit .env. Never hardcode keys.
- Keys: embeddings use GEMINI_EMBED_API_KEY (paid Gemini key, embeddings only). The chat model uses NVIDIA_API_KEY
  with LLM_PROVIDER=nvidia (default model nvidia/nemotron-3.5-super-vl-120b-a12b via ChatNVIDIA); Gemini is an
  optional fallback (LLM_PROVIDER=gemini with GEMINI_LLM_API_KEY, free tier).
- Make NO live Gemini calls (embeddings or LLM) without asking me first. Tests must be fully offline/mocked.
- Never modify ./data, never run embed_data.py --reset, never run full ingestion without asking.
- Never write to nba_stats.db until I say "load".
- Python 3.10+, full type hints, no bare except, logging not print (except CLI output).
- Before finishing any task run: `mypy .` (plain, no --follow-imports=skip) and `ruff check .`, and paste the real output.
- The old test suite and eval files (test_*.py except test_sqlsafe.py, conftest.py, eval_retrieval.py,
  eval_questions.yaml, smoke_test.py) were removed in the cleanup and live in git tag `pre-cleanup`
  (restore one with `git checkout pre-cleanup -- <file>`). Dev tools are in requirements-dev.txt.
- Work one numbered step at a time and stop for my "continue".

## Stack (installed)
chromadb 1.5.9, google-genai 2.30.1, langchain 1.4.6 (use langchain.agents.create_agent), langchain-google-genai 4.4.2,
langgraph 1.2.16, pdfplumber 0.11.10, pandas 3.0.6. Do not use LangChain's Chroma wrapper or embeddings classes;
use embed_texts() from embed_data.py so ingestion and query embeddings cannot drift.