
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "qwen2.5:0.5b"


def generate_answer(question, retrieved_chunks):
    """Generate an answer using retrieved document passages."""

    if not retrieved_chunks:
        return "I couldn't find relevant information in the uploaded document."

    context_parts = []

    for chunk in retrieved_chunks:
        page = chunk.get("page", "Unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Page {page}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are DocuMind, a PDF question-answering assistant.

Answer the user's question using ONLY the information in the
document context below.

Rules:
- RAG means Retrieval-Augmented Generation in this application.
- Do not confuse it with Responsible AI Governance.
- If the context does not contain the answer, say:
  "I couldn't find the answer in the provided document."
- Do not invent facts or use outside knowledge.
- Give a clear, direct answer in simple English.
- Mention relevant page numbers when useful.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 180
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()
    answer = data.get("response", "").strip()

    if not answer:
        return "The AI model returned an empty answer. Please try again."

    return answer
