# Cura

Cura is a small healthcare Q&A app built with Streamlit, FastAPI, Groq, ChromaDB, and a Mongo-backed conversation store. It is not just a thin LLM wrapper. Retrieval is additive, with semantic search plus exact disease-name matching, emergency and diagnosis checks short-circuit before any LLM call, and weak local matches can fall back to a live Tavily search.

The app also accepts optional per-session user context, so someone can add a condition or other relevant detail once and have it carried through the rest of the conversation. Mongo is used when it is available, but if it is not running the backend silently drops back to in-memory history and keeps going.

It answers common symptoms, general diseases, healthy lifestyle habits, nutrition and diet, preventive healthcare, and first aid. It does not diagnose or prescribe, and every health response carries the same disclaimer.

## Tech Stack

| Layer | What it uses |
| --- | --- |
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq API with `llama-3.3-70b-versatile` |
| Vector DB | ChromaDB |
| Memory | MongoDB, with automatic in-memory fallback |
| Live fallback | Tavily search limited to `who.int`, `medlineplus.gov`, `cdc.gov`, `fda.gov` |

Groq was picked mainly for latency. A chat app lives or dies on response time, and Groq's Llama 3.3 70B endpoint keeps the round trip fast enough to feel conversational while still fitting the OpenAI-style client used here.

## Architecture

```text
Streamlit UI
  -> FastAPI /chat
  -> guardrails
  -> RAG retrieval: Chroma semantic search + exact disease match
  -> Tavily fallback when local retrieval is weak
  -> prompt builder
  -> Groq
  -> answer + citations

Conversation state
  -> MongoDB
  -> in-memory fallback when Mongo is unavailable
```

## Setup

```bash
git clone <repo-url> cura
cd cura
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` with at least `GROQ_API_KEY`. Add `TAVILY_API_KEY` if you want live fallback, and `MONGO_URI` / `MONGO_DB_NAME` if you want Mongo. Without those two, the app still runs with live fallback disabled and in-memory history.

## Run

```bash
cd backend
python ingest.py
uvicorn app:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```

`backend/ingest.py` expects `data/rag_dataset.jsonl` and `data/diseases.json`. Re-run it after changing the dataset.

## Known Limitations

- Small demo dataset.
- English only.
- Guardrails are pattern based, not ML based.
- No auth. The assignment treats it as optional.

For deeper implementation detail, see [LOGIC_DOCUMENTATION.md](LOGIC_DOCUMENTATION.md).
