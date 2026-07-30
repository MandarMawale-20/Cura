from groq import Groq

import config

_client = Groq(api_key=config.GROQ_API_KEY)


def generate_response(messages: list) -> str:
    completion = _client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=500,
    )
    return completion.choices[0].message.content.strip()
