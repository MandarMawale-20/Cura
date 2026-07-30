from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_context: Optional[str] = None


class Source(BaseModel):
    title: str
    origin: str
    url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: List[Source] = []
    disclaimer: str
    is_emergency: bool = False


class HistoryMessage(BaseModel):
    role: str
    content: str
