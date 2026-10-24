# Evaluation design

Design for the NBA Guide RAG evaluation harness. Nothing here is implemented yet; each later step cites a section.

## 1. Goals and claims

The published README numbers should support three claims, each backed by one experiment on the same gold set:

1. **Retrieval quality:** recall@k / MRR / nDCG@k of `search_guide`, and what each retrieval option adds.
2. **Faithfulness:** answers are supported by the retrieved context or SQL results (no hallucination).
3. **Calibrated abstention:** on questions the guide cannot answer, the agent says so instead of inventing an answer.

## 2. Gold set (120 items)

| Type | Count | Answer path | Gold label |
|------|------:|-------------|-----------|
| narrative | 51 | `search_guide` | `expected_pages`, `gold_answer` |
| numeric | 44 | `run_sql` | `gold_sql` (+ `gold_answer`) |
| mixed | 7 | both tools | `expected_pages`, `retrieval_pages`, `gold_sql`, `gold_answer` |
| unanswerable | 18 | neither (must abstain) | `gold_answer: null` |

Caveats stated up front:
- SQLite holds 5 tables: `draft_2026`, `all_nba_by_player`, `all_nba_by_season`, `nba_finals` and `awards` (the award lists on PDF pp. 100-103). The numeric items are limited to them.
- Award-list questions are answered with `run_sql` on `awards`, so 14 items first drafted as narrative or mixed were retyped as numeric (some join `awards` with another table). They are not scored on `search_guide` retrieval: award lists are hard to retrieve because the extracted text puts each list away from its heading, and SQL answers them exactly.
- For mixed items, `retrieval_pages` lists only the text pages `search_guide` should find; the table pages are reached through `run_sql`.
- Unanswerable items are NBA questions plausible enough to tempt a guess (e.g. a stat the guide never publishes, or a future event), not off-topic ones. The guide covers every season since the league began, so "a season it does not cover" will not be used. The off-topic guard is a separate feature.
- Only items with `verified: true` are scored. Drafted items start as `verified: false`.

### Item schema (extends `eval_questions.yaml`)

```yaml
- id: n17
  type: narrative            # narrative | numeric | mixed | unanswerable
  question: "..."
  gold_answer: "..."         # short, self-contained; null for unanswerable
  expected_pages: [102]      # PDF page index (same numbering as Chroma `page` and SQLite `source_page`)
  gold_sql: null             # numeric/mixed only; read-only SELECT
  note: "where in the guide the answer is"
  verified: false            # set to true by the user after checking against the PDF
```

### Authoring and verification workflow
1. I draft items from local text search of the PDF and from read-only SQLite queries. This step makes no API calls, so it needs no LLM drafting and costs nothing.
2. Every item carries the page and quoted source text in `note`, so checking one means reading one page.
3. The user reviews in batches, fixes or deletes items, and flips `verified: true`.
4. Numeric `gold_sql` is executed read-only to produce `gold_answer`, so those cannot drift from the DB.

## 3. Metrics

### Retrieval (no LLM; deterministic)
A chunk is relevant if its `page` is in `expected_pages`. Computed over the top-k hits for k in {4, 8, 10}:
- **hit@k:** any relevant chunk in top-k.
- **recall@k:** fraction of `expected_pages` covered by top-k pages.
- **MRR:** reciprocal rank of the first relevant chunk.
- **nDCG@k:** binary-relevance gain, so rank position counts.

Scored on narrative and mixed items only (the ones with `expected_pages`). Page-level labels are coarse, since a page holds several chunks. This is a known limitation and is stated in the README.

### End-to-end answers (run the real agent)
Recorded per item: final answer, every tool call, retrieved contexts, SQL text and rows.
- **Faithfulness (Ragas):** are the answer's claims supported by the retrieved contexts / SQL rows? Narrative and mixed items.
- **Answer correctness:** answer vs `gold_answer` (Ragas factual correctness, judged by the LLM).
- **SQL execution match:** the agent's SQL result set equals `gold_sql`'s (order-insensitive unless the question asks for a ranking). Deterministic, no LLM.
- **Abstention accuracy:** on unanswerable items, the answer declines; on answerable items, it does not decline. Reported as precision and recall of abstention, so over-refusal is visible too.
- **Tool routing:** the agent used the expected tool(s). Deterministic.

### Judging
- **Primary judge:** the NVIDIA model through LangChain, wrapped for Ragas.
- **Cross-check judge:** Gemini on the free-tier `GEMINI_LLM_API_KEY`, run on the same outputs. Report both and the agreement rate (Cohen's kappa on pass/fail verdicts).
- The primary judge is the same model family as the generator, which can inflate scores. The cross-check is the mitigation, and the README says so.
- Free-tier limits mean the cross-check is rate-limited and may be slow. The runner batches with backoff and caches verdicts so a re-run never repeats a call.

## 4. Configurations compared (Step 6)

Same gold set, same judge, one variable at a time against a baseline of `k=8, ACRONYM_MODE=fallback`:

| Axis | Values |
|------|--------|
| Acronym expansion | off / fallback / both |
| k | 4 / 8 / 10 |
| Reranker | none / local cross-encoder (fetch top-20, rerank to k) |
| Embeddings | Gemini `gemini-embedding-001` (paid, `./data`) / local `BAAI/bge-base-en-v1.5` (`./data_bge`) |

Both embedding stores hold the same 8,280 chunks (same chunk ids). bge-base truncates at 512 tokens, which cuts the tail of 1,208 chunks (mostly number-heavy tables); this is reported as a limitation of that backend.

The reranker does not exist yet. A new module will wrap a local cross-encoder behind an opt-in flag, off by default so current behavior is unchanged. Model and dependency choice come at Step 6 and will be confirmed with the user first, because it adds a torch/sentence-transformers install.

## 5. Cost and safety controls

- Query embeddings use the paid key. They are cached on disk keyed by (model, text), so each distinct query is embedded once ever, and only after the user approves the live run. A fully cached run is free.
- Ragas judge and agent runs need LLM calls. Before each live run I will report the call count and ask for approval.
- All runs write only to `eval_results/` (new, git-ignored for raw outputs; the summary tables are committed).
- Nothing touches `./data` or `nba_stats.db` writes. SQL is executed through the existing read-only path in `sqlsafe.py`.
- All unit tests mock the embedding client, the agent and the judges.

## 6. Planned files

| File | Purpose |
|------|---------|
| `eval_questions.yaml` | gold set (restored from `pre-cleanup`, then extended) |
| `eval_metrics.py` | pure metric functions (hit/recall/MRR/nDCG, SQL match, abstention) |
| `eval_retrieval.py` | retrieval runner with embedding cache |
| `eval_answers.py` | end-to-end runner + Ragas / Gemini judging |
| `rerank.py` | local cross-encoder wrapper (Step 6) |
| `eval_report.py` | JSON results to markdown tables and a chart |
| `test_eval_*.py` | offline tests |
| `README.md` | "Evaluation" section with real numbers |

## 7. Known limitations (to be repeated in the README)

- A single document (the 2026-27 NBA Guide), so results may not generalize to other corpora or document styles.
- 120 items gives wide confidence intervals; differences of a few points between configs are noise. The report will include bootstrap 95% CIs.
- Page-level relevance, LLM-based judging, and a same-family primary judge.
- Gold set is authored by the same people who built the system, so it may over-represent the question styles it handles well.
