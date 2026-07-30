from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import guardrails
import llm
import memory
import prompts
import rag
from models import ChatRequest, ChatResponse, Source

app = FastAPI(title="Healthcare Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mongo_connected": memory.is_mongo_connected(),
        "knowledge_base_size": rag.collection_size(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if request.conversation_id:
        conversation_id = request.conversation_id
        if request.user_context is not None:
            memory.set_user_context(conversation_id, request.user_context)
    else:
        conversation_id = memory.create_conversation(request.user_context or "")

    if guardrails.check_emergency(message):
        memory.add_message(conversation_id, "user", message)
        memory.add_message(conversation_id, "assistant", guardrails.EMERGENCY_MESSAGE)
        return ChatResponse(
            answer=guardrails.EMERGENCY_MESSAGE,
            conversation_id=conversation_id,
            sources=[],
            disclaimer=guardrails.DISCLAIMER,
            is_emergency=True,
        )

    if guardrails.check_diagnosis_request(message):
        memory.add_message(conversation_id, "user", message)
        memory.add_message(conversation_id, "assistant", guardrails.DIAGNOSIS_MESSAGE)
        return ChatResponse(
            answer=guardrails.DIAGNOSIS_MESSAGE,
            conversation_id=conversation_id,
            sources=[],
            disclaimer=guardrails.DISCLAIMER,
            is_emergency=False,
        )

    history = memory.get_history(conversation_id)
    user_context = memory.get_user_context(conversation_id)
    chunks = rag.retrieve(message)
    llm_messages = prompts.build_messages(message, chunks, history, user_context)
    answer = llm.generate_response(llm_messages)

    memory.add_message(conversation_id, "user", message)
    memory.add_message(conversation_id, "assistant", answer)

    sources = [
        Source(title=c["title"], origin=c["origin"], url=c.get("url"))
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        sources=sources,
        disclaimer=guardrails.DISCLAIMER,
        is_emergency=False,
    )


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    history = memory.get_history(conversation_id, limit=None)
    return {"conversation_id": conversation_id, "messages": history}
