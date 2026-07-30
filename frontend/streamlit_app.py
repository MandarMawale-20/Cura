import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="HealthAssist", page_icon="🩺", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "user_context" not in st.session_state:
    st.session_state.user_context = None

with st.sidebar:
    st.title("🩺 HealthAssist")
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

st.title("Healthcare Assistant")

if not st.session_state.messages:
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

user_input = st.chat_input("Ask a general health question...")

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
