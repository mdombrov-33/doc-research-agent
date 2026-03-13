import json
import uuid

import requests
import streamlit as st

from src.config import get_settings

settings = get_settings()

OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-5"]
OPENROUTER_MODELS = [
    "deepseek/deepseek-v3.2",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.6",
    "google/gemini-2.5-flash",
    "x-ai/grok-4.1-fast",
    "moonshotai/kimi-k2.5",
    "openai/gpt-oss-120b",
]


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def upload_file(uploaded_file):
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    try:
        response = requests.post(f"{settings.API_URL}/api/upload", files=files, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None


def stream_query(question: str, model: str | None, top_k: int):
    """Generator that yields tokens from the SSE stream and stores final metadata."""
    st.session_state.stream_meta = {}

    with requests.post(
        f"{settings.API_URL}/api/stream",
        json={
            "question": question,
            "session_id": st.session_state.session_id,
            "model": model,
            "top_k": top_k,
        },
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = json.loads(line[6:])
            if "error" in data:
                yield f"\n\n*Error: {data['error']}*"
                return
            if "token" in data:
                yield data["token"]
            if data.get("done"):
                st.session_state.stream_meta = data
                return


def fetch_eval_stats() -> dict | None:
    try:
        r = requests.get(f"{settings.API_URL}/api/evaluation/stats", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def render_eval_stats():
    stats = fetch_eval_stats()
    if not stats or stats.get("total_queries", 0) == 0:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries", stats["total_queries"])
    c2.metric("Retrieval precision", f"{stats['avg_retrieval_precision']:.0%}")
    c3.metric("Web search rate", f"{stats['web_search_rate']:.0%}")
    c4.metric("Avg latency", f"{stats['avg_latency_ms'] / 1000:.1f}s")
    st.divider()


def main():
    st.set_page_config(page_title="Document Agent", layout="wide")

    init_session_state()

    st.title("Document Research Agent")
    render_eval_stats()
    st.markdown(
        "Upload documents using the sidebar, then ask questions about their content. "
        "The agent searches your documents first — if it doesn't find enough relevant information, "
        "it automatically falls back to web search."
    )
    st.divider()

    with st.sidebar:
        st.header("Upload Documents")
        uploaded_file = st.file_uploader("Select file", type=["pdf", "docx", "txt"])

        if uploaded_file and st.button("Upload"):
            with st.spinner("Processing..."):
                result = upload_file(uploaded_file)
                if result:
                    st.success(
                        f"Uploaded {result['filename']} — {result['chunks_created']} chunks created"
                    )

        st.divider()
        st.subheader("Configuration")

        all_models = OPENAI_MODELS + OPENROUTER_MODELS
        default_model = settings.get_llm_model()
        default_index = all_models.index(default_model) if default_model in all_models else 0
        selected_model = st.selectbox("Model", all_models, index=default_index)

        top_k = st.slider("Top-K results", min_value=3, max_value=15, value=5, step=1)

        st.text(f"Backend: {settings.API_URL}")
        st.caption(f"Session: {st.session_state.session_id[:8]}...")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(f"Sources: {message['sources']}")

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer = st.write_stream(stream_query(prompt, selected_model, top_k))
            meta = st.session_state.get("stream_meta", {})
            sources = meta.get("sources_count", 0)
            if sources:
                st.caption(f"Sources: {sources}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
