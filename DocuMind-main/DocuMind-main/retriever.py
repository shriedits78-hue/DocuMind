
from chunking import chunk_pages
from embeddings import get_embeddings, get_embedding
from vector_store import VectorStore


class Retriever:
    """Build and search an in-memory index of document chunks."""

    def __init__(self):
        self.vector_store = VectorStore()

    def index_pages(self, pages, chunk_size=500, overlap=80):
        """Split PDF pages, embed the chunks, and index them."""
        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap=overlap
        )

        if not chunks:
            raise ValueError("No text chunks were created from the PDF.")

        embeddings = get_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        self.vector_store = VectorStore()
        self.vector_store.add(chunks, embeddings)

        return len(chunks)

    def retrieve(self, question, top_k=3):
        """Find the most relevant chunks for a question."""
        if not question or not question.strip():
            return []

        query_embedding = get_embedding(question)

        return self.vector_store.search(
            query_embedding,
            top_k=top_k
        )
