import json
import uuid

import requests
import streamlit as st

from src.config import get_settings

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


def stream_query(question: str):
    """Generator that yields tokens from the SSE stream and stores final metadata."""
    st.session_state.stream_meta = {}

    with requests.post(
        f"{settings.API_URL}/api/stream",
        json={"question": question, "session_id": st.session_state.session_id},
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


def main():
    st.set_page_config(page_title="Document Agent", layout="wide")

    init_session_state()

    st.title("Document Research Agent")
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
            answer = st.write_stream(stream_query(prompt))
            meta = st.session_state.get("stream_meta", {})
            sources = meta.get("sources_count", 0)
            if sources:
                st.caption(f"Sources: {sources}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
