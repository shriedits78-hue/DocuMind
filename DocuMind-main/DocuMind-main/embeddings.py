
import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


def get_embeddings(texts):
    """
    Convert a list of text strings into embedding vectors
    using the locally running Ollama server.
    """
    if not texts:
        return []

    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings", [])

    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama returned an unexpected number of embeddings."
        )

    return embeddings


def get_embedding(text):
    """Return one embedding vector for a single text."""
    return get_embeddings([text])[0]
