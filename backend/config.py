import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "healthcare_chatbot")

CHROMA_DIR = os.getenv("CHROMA_DIR", "../vector_store")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "healthcare_kb")

TOP_K = int(os.getenv("TOP_K", "3"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "5"))

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_DOMAINS = ["who.int", "medlineplus.gov", "cdc.gov", "fda.gov"]


SIMILARITY_DISTANCE_THRESHOLD = float(os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "0.5"))

RAG_DATASET_FILE = os.getenv("RAG_DATASET_FILE", "../data/rag_dataset.jsonl")
DISEASES_FILE = os.getenv("DISEASES_FILE", "../data/diseases.json")
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "500"))

ENABLE_KEYWORD_RERANK = os.getenv("ENABLE_KEYWORD_RERANK", "true").lower() == "true"
