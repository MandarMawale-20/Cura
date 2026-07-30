import uuid
from datetime import datetime, timezone

import config

_mongo_available = False
_conversations_collection = None
_local_store = {}

try:
    from pymongo import MongoClient

    _mongo_client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=1500)
    _mongo_client.admin.command("ping")
    _db = _mongo_client[config.MONGO_DB_NAME]
    _conversations_collection = _db["conversations"]
    _mongo_available = True
except Exception:
    _mongo_available = False


def create_conversation(user_context: str = "") -> str:
    conversation_id = str(uuid.uuid4())

    if _mongo_available:
        _conversations_collection.insert_one({
            "conversation_id": conversation_id,
            "user_context": user_context,
            "messages": [],
            "created_at": datetime.now(timezone.utc),
        })
    else:
        _local_store[conversation_id] = {"user_context": user_context, "messages": []}

    return conversation_id


def get_user_context(conversation_id: str) -> str:
    if _mongo_available:
        doc = _conversations_collection.find_one({"conversation_id": conversation_id})
        return doc.get("user_context", "") if doc else ""

    return _local_store.get(conversation_id, {}).get("user_context", "")


def set_user_context(conversation_id: str, user_context: str):
    if _mongo_available:
        _conversations_collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"user_context": user_context}},
            upsert=True,
        )
    else:
        _local_store.setdefault(conversation_id, {"user_context": "", "messages": []})
        _local_store[conversation_id]["user_context"] = user_context


def add_message(conversation_id: str, role: str, content: str):
    entry = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc),
    }

    if _mongo_available:
        _conversations_collection.update_one(
            {"conversation_id": conversation_id},
            {"$push": {"messages": entry}},
            upsert=True,
        )
    else:
        _local_store.setdefault(conversation_id, {"user_context": "", "messages": []})
        _local_store[conversation_id]["messages"].append(entry)


def get_history(conversation_id: str, limit: int = None) -> list:
    limit = limit or config.HISTORY_TURNS

    if _mongo_available:
        doc = _conversations_collection.find_one({"conversation_id": conversation_id})
        messages = doc["messages"] if doc else []
    else:
        messages = _local_store.get(conversation_id, {}).get("messages", [])

    recent = messages[-limit:] if limit else messages
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def is_mongo_connected() -> bool:
    return _mongo_available
