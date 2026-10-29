
import hashlib

import streamlit as st

from document_loader import load_pdf
from retriever import Retriever
from generator import generate_answer


st.set_page_config(
    page_title="DocuMind - Chat with PDFs",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind")
st.caption("Your private AI assistant for asking questions about PDF documents.")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    if current_hash != st.session_state.file_hash:
        st.session_state.retriever = None
        st.session_state.file_hash = None
        st.session_state.chat_history = []

        try:
            with st.spinner("Reading PDF and building the search index..."):
                pages = load_pdf(uploaded_file)
                retriever = Retriever()
                chunk_count = retriever.index_pages(pages)

                st.session_state.retriever = retriever
                st.session_state.file_hash = current_hash
                st.session_state.document_name = uploaded_file.name
                st.session_state.chunk_count = chunk_count

            st.success(
                f"PDF processed successfully! Created {chunk_count} text chunks."
            )

        except Exception as error:
            st.error(f"Could not process this PDF: {error}")

    if st.session_state.retriever is not None:
        st.info(f"Document: {uploaded_file.name}")

        with st.sidebar:
            st.header("Document information")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(
                f"**Text chunks:** {st.session_state.chunk_count}"
            )
            st.caption("Your document is processed locally using Ollama.")

            if st.button("Clear chat history"):
                st.session_state.chat_history = []
                st.rerun()

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("View source passages"):
                        for source in message["sources"]:
                            st.markdown(
                                f"**Page {source['page']}** "
                                f"(similarity: {source['score']:.3f})"
                            )
                            st.write(source["text"])

        question = st.chat_input(
            "Ask a question about your PDF..."
        )

        if question:
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching the document and generating an answer..."):
                        sources = st.session_state.retriever.retrieve(
                            question,
                            top_k=3
                        )

                        answer = generate_answer(question, sources)

                    st.markdown(answer)

                    if sources:
                        with st.expander("View source passages"):
                            for source in sources:
                                st.markdown(
                                    f"**Page {source['page']}** "
                                    f"(similarity: {source['score']:.3f})"
                                )
                                st.write(source["text"])

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as error:
                    st.error(f"Something went wrong: {error}")

else:
    st.info("Upload a PDF above to start asking questions.")

    st.markdown(
        """
        ### How to use DocuMind
        1. Upload a text-based PDF.
        2. Wait for the document to be processed.
        3. Ask questions in the chat box.
        4. Check the source passages to see which pages were retrieved.

        **Privacy:** This app uses a local Ollama server for embeddings
        and answer generation. The app code does not send your PDF to
        a hosted AI API.
        """
    )
