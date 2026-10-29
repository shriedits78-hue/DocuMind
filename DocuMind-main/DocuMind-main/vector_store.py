
import numpy as np


class VectorStore:
    """A lightweight in-memory store for document chunks and embeddings."""

    def __init__(self):
        self.chunks = []
        self.embeddings = None

    def add(self, chunks, embeddings):
        """Store chunks and their corresponding embedding vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        if self.embeddings is not None:
            if vectors.shape[1] != self.embeddings.shape[1]:
                raise ValueError("Embedding dimensions do not match.")

            self.embeddings = np.vstack([self.embeddings, vectors])
        else:
            self.embeddings = vectors

        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):
        """Return the most similar chunks using cosine similarity."""
        if self.embeddings is None or not self.chunks:
            return []

        if top_k <= 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimensions do not match.")

        query_norm = np.linalg.norm(query)
        vector_norms = np.linalg.norm(self.embeddings, axis=1)

        if query_norm == 0:
            return []

        similarities = (
            self.embeddings @ query
        ) / np.maximum(vector_norms * query_norm, 1e-12)

        best_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in best_indices:
            result = dict(self.chunks[int(index)])
            result["score"] = float(similarities[index])
            results.append(result)

        return results
