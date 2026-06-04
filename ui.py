import json
import uuid

import requests
import streamlit as st

from src.config import get_settings

settings = get_settings()

MODELS: dict[str, str] = {
    "anthropic/claude-opus-4.7": "Claude Opus 4.7",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "google/gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "openai/gpt-5.4-mini": "GPT 5.4 Mini",
    "openai/gpt-5.5": "GPT 5.5",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
}


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
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Queries", stats["total_queries"])
    c2.metric("Avg latency", f"{stats['avg_latency_ms'] / 1000:.1f}s")
    c3.metric("Retrieval precision", f"{stats['avg_retrieval_precision']:.0%}")
    c4.metric("Web search rate", f"{stats['web_search_rate']:.0%}")
    c5.metric("Avg docs retrieved", f"{stats['avg_docs_retrieved']:.1f}")
    c6.metric("Avg docs relevant", f"{stats['avg_docs_relevant']:.1f}")
    st.divider()


def main():
    st.set_page_config(page_title="Document Agent", layout="wide")

    init_session_state()

    st.title("Document Research Agent")
    render_eval_stats()
    st.markdown(
        "Upload documents and ask questions. The agent always searches your documents — "
        "for queries needing current information, it also runs a live web search in parallel."
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

        st.subheader("Configuration")

        default_model = settings.get_llm_model()
        model_keys = list(MODELS.keys())
        default_index = model_keys.index(default_model) if default_model in model_keys else 0
        selected_model = st.selectbox(
            "Model",
            options=model_keys,
            index=default_index,
            format_func=lambda x: MODELS[x],
        )

        top_k = st.slider("Top-K results", min_value=3, max_value=15, value=5, step=1)

        st.caption(f"Backend: {settings.API_URL}")
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
            sources_meta = meta.get("sources", [])
            if sources_meta:
                with st.expander(f"Sources ({sources})"):
                    for s in sources_meta:
                        if s["source"] == "web":
                            st.caption("🌐 Web Search")
                        else:
                            st.caption(
                                f"📄 {s['filename']} · chunk {s['chunk_index']} · "
                                f"{s['chunk_length']} chars"
                            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
