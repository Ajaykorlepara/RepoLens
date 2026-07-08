"""
Retrieval-augmented generation: pulls relevant chunks from Chroma, builds a
prompt with conversation history, and asks Groq to answer with citations.
"""
from typing import Dict, List

from groq import Groq

from config import settings
from ingest import get_chroma_client
from search import hybrid_search

_groq_client = None

# In-memory conversation history, keyed by repo_id.
# Simple by design: it resets if the server restarts. Good enough for a
# single-user local tool; swap for a database if you need it to persist.
_conversations: Dict[str, List[Dict]] = {}


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://console.groq.com/keys"
            )
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def get_history(repo_id: str) -> List[Dict]:
    return _conversations.setdefault(repo_id, [])


def retrieve_context(repo_id: str, question: str) -> List[Dict]:
    """Pull the top-k most relevant chunks for this repo, combining semantic
    (embedding) search with keyword (BM25) search."""
    client = get_chroma_client()
    try:
        client.get_collection(repo_id)
    except Exception:
        raise RuntimeError("This repository hasn't been indexed yet. Call /index first.")

    return hybrid_search(repo_id, question, top_k=settings.top_k)


def _source_label(c: Dict) -> str:
    if c.get("line_start") and c.get("line_end"):
        return f"{c['file']} (lines {c['line_start']}-{c['line_end']})"
    return c["file"]


def build_messages(question: str, contexts: List[Dict], history: List[Dict]) -> List[Dict]:
    context_block = "\n\n".join(f"[Source: {_source_label(c)}]\n{c['text']}" for c in contexts)

    system_message = (
        "You are a helpful assistant that answers questions about a specific "
        "codebase using only the source excerpts provided below each question. "
        "Always mention which file(s) your answer is based on. "
        "If the excerpts don't contain enough information to answer, say so "
        "honestly instead of guessing."
    )

    messages = [{"role": "system", "content": system_message}]

    # Keep the last few turns for conversational memory, without letting the
    # prompt grow unbounded.
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({
        "role": "user",
        "content": f"Source excerpts:\n\n{context_block}\n\nQuestion: {question}",
    })
    return messages


def ask_question(repo_id: str, question: str) -> Dict:
    contexts = retrieve_context(repo_id, question)
    history = get_history(repo_id)
    messages = build_messages(question, contexts, history)

    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    history.append({"question": question, "answer": answer})

    citations = [
        {
            "file": c["file"],
            "line_start": c.get("line_start"),
            "line_end": c.get("line_end"),
            "kind": c.get("kind", "text"),
            "text": c["text"],
        }
        for c in contexts
    ]
    sources = sorted({_source_label(c) for c in contexts})
    return {"answer": answer, "sources": sources, "citations": citations}
