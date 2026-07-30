import json
import os
import re

import chromadb
from chromadb.utils import embedding_functions

import config

_client = chromadb.PersistentClient(path=config.CHROMA_DIR)

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_collection = _client.get_or_create_collection(
    name=config.CHROMA_COLLECTION,
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

_tavily_client = None
if config.TAVILY_API_KEY:
    from tavily import TavilyClient
    _tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)

_known_diseases = []
if os.path.exists(config.DISEASES_FILE):
    with open(config.DISEASES_FILE, "r", encoding="utf-8") as f:
        _known_diseases = json.load(f)


def _label_for(meta: dict) -> tuple:
    doc_type = meta.get("type", "reference")

    if doc_type == "disease_profile" or meta.get("disease"):
        title = meta.get("disease", "Disease Profile")
    elif doc_type == "intent":
        title = meta.get("topic", "First-Aid Guidance")
    else:
        title = "Health Q&A"

    source_name = meta.get("source", "Medical Reference Dataset")
    return title, source_name


def _match_known_disease(query: str):
    q = query.lower()
    for disease in _known_diseases:
        if disease.lower() in q:
            return disease
    return None


def _keyword_overlap(query: str, text: str) -> int:
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    return len(query_words & text_words)


def _semantic_search(query: str, top_k: int):
    if _collection.count() == 0:
        return [], None

    results = _collection.query(query_texts=[query], n_results=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    best_distance = distances[0] if distances else None
    hits = list(zip(documents, metadatas, distances))
    return hits, best_distance


def _exact_disease_chunks(disease: str):
    result = _collection.get(
        where={"$and": [{"disease": disease}, {"type": "disease_profile"}]}
    )
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    # Direct disease-name hits are treated as exact matches, not nearest-neighbor guesses.
    return [(doc, meta, 0.0) for doc, meta in zip(documents, metadatas)]


def _query_local(query: str, top_k: int):
    semantic_hits, best_distance = _semantic_search(query, top_k)

    matched_disease = _match_known_disease(query)
    exact_hits = _exact_disease_chunks(matched_disease) if matched_disease else []

    combined = exact_hits + semantic_hits

    seen, deduped = set(), []
    for doc, meta, dist in combined:
        if doc not in seen:
            seen.add(doc)
            deduped.append((doc, meta, dist))

    if config.ENABLE_KEYWORD_RERANK:
        # Re-rank only the fetched pool; the full collection is too broad for this.
        deduped.sort(key=lambda item: _keyword_overlap(query, item[0]), reverse=True)

    top = deduped[:top_k]

    chunks = []
    for doc, meta, dist in top:
        title, source_name = _label_for(meta)
        chunks.append({
            "content": doc,
            "title": title,
            "origin": f"{source_name} (indexed)",
            "url": None,
            "distance": dist,
        })

    good_match = matched_disease is not None or (
        best_distance is not None and best_distance <= config.SIMILARITY_DISTANCE_THRESHOLD
    )

    return chunks, good_match


def _query_tavily(query: str, top_k: int):
    if _tavily_client is None:
        return []

    try:
        response = _tavily_client.search(
            query=query,
            include_domains=config.TAVILY_DOMAINS,
            max_results=top_k,
        )
    except Exception:
        return []

    chunks = []
    for result in response.get("results", []):
        chunks.append({
            "content": result.get("content", ""),
            "title": result.get("title", "Web result"),
            "origin": "(live)",
            "url": result.get("url"),
            "distance": None,
        })

    return chunks


def retrieve(query: str, top_k: int = None):
    top_k = top_k or config.TOP_K

    local_chunks, good_match = _query_local(query, top_k)
    if good_match:
        return local_chunks

    live_chunks = _query_tavily(query, top_k)
    return live_chunks if live_chunks else local_chunks


def collection_size() -> int:
    return _collection.count()
