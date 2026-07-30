import json
import os

import chromadb
from chromadb.utils import embedding_functions

import config


def load_jsonl(path: str):
    ids, docs, metas = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(row["id"])
            docs.append(row["document"])
            metas.append(row["metadata"])
    return ids, docs, metas


def main():
    if not os.path.exists(config.RAG_DATASET_FILE):
        print(f"Dataset not found: {config.RAG_DATASET_FILE}")
        return

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = load_jsonl(config.RAG_DATASET_FILE)
    if not ids:
        print("No records found in dataset.")
        return

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    batch_size = config.INGEST_BATCH_SIZE
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=docs[i:i + batch_size],
            metadatas=metas[i:i + batch_size],
        )
        print(f"Indexed {min(i + batch_size, len(ids))}/{len(ids)}")

    print(f"Done. {len(ids)} chunks indexed into '{config.CHROMA_COLLECTION}'.")

    if os.path.exists(config.DISEASES_FILE):
        with open(config.DISEASES_FILE, "r", encoding="utf-8") as f:
            diseases = json.load(f)
        print(f"Found {len(diseases)} known diseases at {config.DISEASES_FILE} for exact-match retrieval.")
    else:
        print(f"Warning: {config.DISEASES_FILE} not found. Exact disease matching will be disabled at query time.")


if __name__ == "__main__":
    main()
