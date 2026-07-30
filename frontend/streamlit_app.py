import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="HealthAssist", page_icon="H", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7f8fa 0%, #ffffff 30%, #ffffff 100%);
        }

        .main .block-container {
            max-width: 920px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            padding: 1.25rem 1.5rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .app-header h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
            color: #0f172a;
        }

        .app-header p {
            margin: 0.5rem 0 0;
            color: #475569;
            font-size: 0.98rem;
        }

        .app-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .app-pill {
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #f8fafc;
            color: #334155;
            font-size: 0.82rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }

        section[data-testid="stSidebar"] .stButton button {
            width: 100%;
        }

        .stChatMessage {
            border-radius: 16px;
        }

        .stTextInput input,
        .stChatInput input {
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "user_context" not in st.session_state:
    st.session_state.user_context = None

with st.sidebar:
    st.title("HealthAssist")
    st.caption("General healthcare information, not a substitute for medical advice.")
    st.warning(
        "In a medical emergency, call your local emergency number immediately. "
        "This assistant cannot help with urgent or life-threatening situations."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.session_state.user_context = None
            st.rerun()

    st.markdown(
        """
        <div class="app-header">
            <h1>HealthAssist</h1>
            <p>General health answers with retrieval, guardrails, and optional session context.</p>
            <div class="app-meta">
                <span class="app-pill">RAG search</span>
                <span class="app-pill">Emergency guardrails</span>
                <span class="app-pill">Mongo fallback</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
        st.info(
            "Add optional context once, then ask a question. The assistant will keep it in"
            " mind for the rest of the conversation."
        )
    st.session_state.user_context = st.text_input(
        "Any conditions you'd like me to consider? (optional)",
        value=st.session_state.user_context or "",
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['title']}** ({src['origin']})")

user_input = st.chat_input("Ask a general health question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": user_input,
                        "conversation_id": st.session_state.conversation_id,
                        "user_context": st.session_state.user_context,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                st.session_state.conversation_id = data["conversation_id"]
                answer = data["answer"]

                if data.get("is_emergency"):
                    st.error(answer)
                else:
                    st.markdown(answer)

                st.caption(data["disclaimer"])

                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"- **{src['title']}** ({src['origin']})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend: {e}")
