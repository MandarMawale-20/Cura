# HealthAssist — Healthcare Information Chatbot

A RAG-powered healthcare chatbot built with Streamlit, FastAPI, Groq (Llama 3.3 70B), and ChromaDB. It answers general health questions using a curated knowledge base, refuses to diagnose, and escalates emergency-sounding queries instead of responding directly.

## Architecture

```
Streamlit UI → FastAPI → Guardrails → ChromaDB Retrieval → Prompt Builder → Groq LLM → Response + Citations
                                ↓                    ↓
                        MongoDB (conversation      Tavily fallback (who.int, medlineplus.gov,
                        history + user_context)     cdc.gov, fda.gov) when local match is weak
```

If MongoDB isn't running, the backend automatically falls back to in-memory conversation storage so the app still works locally without extra setup.

At session start, the user can optionally share relevant context (e.g. "has diabetes"), stored once per conversation and folded into every prompt afterward without being treated as a diagnosis input.

Retrieval first checks ChromaDB. If the closest match isn't similar enough (cosine distance above `SIMILARITY_DISTANCE_THRESHOLD`), it falls back to a live, domain-restricted Tavily search instead of answering from a weak local match. Citations are tagged `(indexed)` or `(live)` so the source is always clear. If `TAVILY_API_KEY` isn't set, the app simply skips the fallback and uses whatever local match it has.

## Folder Structure

```
healthcare-chatbot/
├── backend/
│   ├── app.py          # FastAPI routes
│   ├── config.py        # Environment configuration
│   ├── models.py         # Request/response schemas
│   ├── guardrails.py     # Emergency & diagnosis-refusal checks
│   ├── rag.py             # ChromaDB retrieval
│   ├── prompts.py         # System prompt + prompt assembly
│   ├── llm.py               # Groq API client
│   ├── memory.py           # Conversation history (Mongo or in-memory)
│   └── ingest.py             # Loads data/*.json into ChromaDB
├── frontend/
│   └── streamlit_app.py      # Chat UI
├── data/                       # Source health documents (JSON)
├── vector_store/                 # ChromaDB persistent storage (generated)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Clone and enter the project**
   ```
   cd healthcare-chatbot
   ```

2. **Create a virtual environment and install dependencies**
   ```
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```
   cp .env.example .env
   ```
   Then edit `.env` and add your Groq API key (get one at [console.groq.com](https://console.groq.com)). Optionally add a `TAVILY_API_KEY` (from [tavily.com](https://tavily.com)) to enable the live web fallback — the app works fine without it.

4. **MongoDB (optional)**
   If you have MongoDB running locally, conversation history will persist there. If not, the app still works — history is kept in memory for the session.

## Running the App

**1. Index the knowledge base** (run once, or after editing `data/`)
```
cd backend
python ingest.py
```

**2. Start the backend**
```
cd backend
uvicorn app:app --reload --port 8000
```

**3. Start the frontend** (in a new terminal)
```
cd frontend
streamlit run streamlit_app.py
```

Open the Streamlit URL printed in the terminal (usually `http://localhost:8501`).

## Adding the Knowledge Base

This project expects two files in `data/` (not included — see `data/README.md`):

- **`rag_dataset.jsonl`** — one record per line: `{"id": ..., "document": ..., "metadata": {...}}`. `metadata.type` is one of `qa`, `intent`, `disease_profile`, `description`, `symptoms`, `diet`, `precaution`, `workout`. Only disease-related types carry a `disease` field — that's expected, not a bug.
- **`diseases.json`** — a flat array of known disease names, used at query time to boost exact-name matches on top of semantic search (never to filter results out).

Run once (or after replacing the dataset):
```
cd backend
python ingest.py
```
It batches inserts in groups of 500 and re-indexes cleanly on reruns.

## Notes

- Retrieval is additive by design: semantic search always runs across the whole collection; an exact disease-name match (if the query names one) is merged on top of it, never used to filter results out. A cheap keyword-overlap re-rank runs on the fetched pool only (toggle via `ENABLE_KEYWORD_RERANK`), since `qa`/`intent` chunks carry no `disease` metadata to boost on.
- This project is for educational/demo purposes and is not a certified medical tool.
