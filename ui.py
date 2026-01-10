import requests
import streamlit as st

from src.config import get_settings

settings = get_settings()


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def upload_file(uploaded_file):
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    try:
        response = requests.post(f"{settings.API_URL}/api/upload", files=files, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None


def query_documents(question: str):
    try:
        response = requests.post(
            f"{settings.API_URL}/api/query", json={"question": question}, timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Query failed: {e}")
        return None


def main():
    st.set_page_config(page_title="RAG Security Agent", layout="wide")

    init_session_state()

    st.title("Document Research Agent")

    with st.sidebar:
        st.header("Upload Documents")
        uploaded_file = st.file_uploader("Select file", type=["pdf", "docx", "txt"])

        if uploaded_file and st.button("Upload"):
            with st.spinner("Processing..."):
                result = upload_file(uploaded_file)
                if result:
                    st.success(
                        f"Uploaded {result['filename']} - {result['chunks_created']} chunks created"
                    )

        st.divider()
        st.subheader("Configuration")
        st.text(f"Backend: {settings.API_URL}")
        st.text("Search: Hybrid (BM25 + Vector)")
        st.text(f"Threshold: {settings.RELEVANCE_THRESHOLD} docs")

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
            with st.spinner("Processing..."):
                result = query_documents(prompt)

                if result:
                    answer = result.get("answer", "No answer generated")
                    sources = result.get("sources_count", 0)

                    st.markdown(answer)
                    st.caption(f"Sources: {sources}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    st.error("Failed to get response")


if __name__ == "__main__":
    main()
