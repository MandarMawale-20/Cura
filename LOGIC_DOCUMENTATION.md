# Logic Documentation

This document serves as a technical supplement to the README.md. It stays close to the code and only describes behavior that is actually implemented.

## Request Flow

The frontend sends `message`, `conversation_id`, and optional `user_context` to `POST /chat`. From there, `backend/app.py` trims the message and rejects empty input immediately. If a conversation already exists, the backend reuses it and updates the stored user context when one is provided. If not, it creates a new conversation and saves that context with it.

Guardrails run before retrieval. Emergency patterns take priority and return a fixed emergency message straight away. Diagnosis requests are handled the same way, with a fixed refusal response and no call to retrieval or the LLM.

For normal questions, the backend loads conversation history, reads any stored user context, retrieves supporting chunks, builds the prompt, and calls Groq. The user message and assistant reply are then written back to conversation history.

## Guardrails

`backend/guardrails.py` uses pattern matching rather than a classifier. That is deliberate: the goal is to catch obvious emergency language and direct diagnosis requests quickly, not to infer intent from a model. The emergency response and diagnosis refusal are both fixed messages, and the checks short-circuit in `backend/app.py` before any LLM call happens.

The patterns are intentionally narrow. They cover phrases like chest pain, trouble breathing, overdose, suicidal intent, and requests such as "do I have..." or "diagnose me." That keeps the behavior predictable, which matters more here than trying to be clever. A disclaimer is attached to every response, guardrail-triggered or not.

## Retrieval

Retrieval lives in `backend/rag.py`. ChromaDB is the local store, and embeddings come from `SentenceTransformerEmbeddingFunction` with `all-MiniLM-L6-v2`. Semantic search always runs first.

If the query names a disease that appears in `data/diseases.json`, the matching disease profile chunks are merged into the candidate set. That exact match path is additive only. It never filters the semantic results out, which is important because the code is meant to widen recall, not narrow it. This matters more than it sounds: most of the dataset, especially Q&A pairs and first-aid entries, carries no disease metadata at all, so a filter-based approach would drop a large part of the knowledge base.

When `ENABLE_KEYWORD_RERANK` is on, the already-fetched pool is re-sorted by keyword overlap. A local match is considered good either when an exact disease match exists or when the best semantic distance is within `SIMILARITY_DISTANCE_THRESHOLD`. If the local match is weak and Tavily is configured, the backend falls back to live search. Those queries are limited to `who.int`, `medlineplus.gov`, `cdc.gov`, and `fda.gov`. If Tavily is unavailable or the request fails, the backend keeps the local results instead of breaking the request.

Sources are tagged as `(indexed)` for local chunks and `(live)` for Tavily results so it is obvious where the answer came from.

## Prompt Assembly

`backend/prompts.py` builds the final message list for Groq, and no two requests build the same prompt. The system prompt is static: it keeps the model in general educational territory, tells it not to diagnose or prescribe, asks it to stay concise, and tells it to be honest when retrieved material does not cover the question.

Everything else in the prompt is assembled per request. Conversation history is appended before the current question, so follow-ups stay coherent. Retrieved chunks are presented explicitly as background reference material, with a clear boundary marker separating them from the actual conversation, not as part of the user's own words. That separation matters because some of the retrieved material comes from doctor-patient style transcripts, and those lines must not be mistaken for current user input. Optional user context is included once per conversation and framed as context to adjust suggestions with, not as a medical record to diagnose from.

## Response Generation

Groq's `llama-3.3-70b-versatile` receives the assembled message list with a low temperature and a token cap, which keeps the answers focused and consistent. The raw text response is returned as-is, with citations and a disclaimer attached afterward rather than being baked into the model output.

Groq was chosen mainly for latency. A chat interface depends on turnaround time, and the model choice here keeps the response loop fast enough to feel conversational while still using a simple OpenAI-style client.

## Conversation Storage

`backend/memory.py` tries Mongo first. It pings the database at import time, and if that succeeds, conversations are stored in the `conversations` collection. If anything in that startup path fails, the backend silently switches to an in-memory dictionary.

That fallback is intentional. It keeps the app usable during local development without turning a missing Mongo instance into a hard failure. User context and message history are both stored per conversation in either backend.

## Frontend

`frontend/streamlit_app.py` keeps the UI intentionally plain. Session state tracks the message list, the active conversation ID, and the optional user context. The sidebar carries the disclaimer and emergency warning, and the first prompt can collect context before the conversation starts. Each chat turn posts to the backend, then renders the answer, disclaimer, and sources.

## Data Ingestion

`backend/ingest.py` reads `data/rag_dataset.jsonl` and writes it into Chroma. Existing entries are deleted before re-indexing so reruns do not duplicate data. Records are inserted in batches controlled by `INGEST_BATCH_SIZE`. `data/diseases.json` is loaded separately for query-time exact disease-name matching. If the dataset file is missing, the script prints a message and exits.

## Configuration

The main environment variables come from `backend/config.py`.

`GROQ_API_KEY` and `GROQ_MODEL` control the LLM call. `MONGO_URI` and `MONGO_DB_NAME` control persistence. `CHROMA_DIR` and `CHROMA_COLLECTION` control the vector store. `TOP_K`, `HISTORY_TURNS`, and `SIMILARITY_DISTANCE_THRESHOLD` shape retrieval and context size. `TAVILY_API_KEY` enables live fallback, and `TAVILY_DOMAINS` are fixed in code.

## Assumptions

- Queries are assumed to be in English; no language detection or translation is implemented.
- The knowledge base is a small demo dataset, not comprehensive medical coverage. It is sized to prove the retrieval pipeline end to end, not to be exhaustive.
- Pattern-based guardrails are treated as sufficient for this scope. A production system in this domain would likely add a classifier-backed layer.
- No authentication layer was built, since the assignment marked it optional. Sessions are identified by a generated UUID rather than a logged-in user.
- Mongo is treated as optional infrastructure by design, so local review does not depend on a database being available first.
