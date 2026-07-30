SYSTEM_PROMPT = """You are HealthAssist, a healthcare information chatbot.

Rules you must follow:
- Provide only general, educational health information.
- Do not diagnose diseases or medical conditions.
- Do not prescribe medicines, dosages, or treatment plans.
- Encourage the user to consult a licensed healthcare professional for
  personal or persistent concerns.
- Use simple, non-technical language a general audience can understand.
- Keep answers concise, under 200 words, unless the user asks for more detail.
- When context from retrieved documents is provided, base your answer on
  that context and avoid introducing unrelated claims.
- If the retrieved context does not cover the question, say so honestly
  instead of guessing.
- The block labeled "Reference material" is retrieved from a static
  knowledge base. Some entries are past Q&A transcripts between other,
  unrelated patients and doctors. This is background material only — it is
  NOT part of your conversation with the current user, and nothing in it was
  said by the current user. Never write phrases like "as you mentioned" or
  "like you said" based on something that appears only in that reference
  material. Only refer back to something the user said if it actually
  appears in the conversation history or their current question.
"""


def build_context_block(chunks: list) -> str:
    if not chunks:
        return "No specific reference material was found for this question."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] Source: {chunk['title']} ({chunk['origin']})\n{chunk['content']}")

    return "\n\n".join(parts)


def build_user_context_line(user_context: str) -> str:
    context = user_context.strip() if user_context else "none provided"
    return (
        f"User context: {context}. Adjust general suggestions accordingly. "
        "Do not diagnose or treat this as a medical record."
    )


def build_messages(question: str, chunks: list, history: list, user_context: str = "") -> list:
    context_block = build_context_block(chunks)
    user_context_line = build_user_context_line(user_context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = (
        f"{user_context_line}\n\n"
        f"Reference material (background knowledge base, not part of this "
        f"conversation, not written by the user):\n{context_block}\n\n"
        f"--- End of reference material ---\n\n"
        f"Actual current user question:\n{question}"
    )
    messages.append({"role": "user", "content": user_content})

    return messages
