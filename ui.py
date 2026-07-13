import json
import uuid

import requests
import streamlit as st

from src.config import SUPPORTED_MODELS, get_settings

settings = get_settings()

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


def fetch_monitoring_stats() -> dict | None:
    try:
        r = requests.get(f"{settings.API_URL}/api/monitoring/stats", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def render_monitoring_stats():
    stats = fetch_monitoring_stats()
    if not stats or stats.get("total_queries", 0) == 0:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries", stats["total_queries"])
    c2.metric("Avg latency", f"{stats['avg_latency_ms'] / 1000:.1f}s")
    c3.metric("Web search rate", f"{stats['web_search_rate']:.0%}")
    c4.metric("Avg sources returned", f"{stats['avg_sources_retrieved']:.1f}")
    st.divider()


def main():
    st.set_page_config(page_title="Document Agent", layout="wide")

    init_session_state()

    st.title("Document Research Agent")
    render_monitoring_stats()
    st.markdown(
        "Upload documents and ask questions. The agent searches your documents first, and "
        "falls back to a live web search on its own when they don't cover the question."
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

        default_model = settings.LLM_MODEL
        model_keys = list(SUPPORTED_MODELS)
        default_index = model_keys.index(default_model) if default_model in model_keys else 0
        selected_model = st.selectbox(
            "Model",
            options=model_keys,
            index=default_index,
            format_func=lambda x: SUPPORTED_MODELS[x],
        )

        top_k = st.slider("Top-K results", min_value=3, max_value=15, value=10, step=1)
        st.caption("Per retrieval call — the agent may call retrieve multiple times.")

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
                        if s["source_type"] == "web":
                            st.markdown(f"🌐 [{s['title']}]({s['url']})")
                        else:
                            location = f" · page {s['page']}" if s.get("page") else ""
                            st.caption(f"📄 {s['title']}{location}")
                        st.caption(s["excerpt"])

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
