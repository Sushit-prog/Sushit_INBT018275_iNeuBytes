"""
Groq LLM call logic.

Takes a user query, retrieves relevant context from the knowledge base via
the RAG module, builds a system prompt, and calls the Groq API to produce a
mentor-style conversational answer.
"""

import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------------------------
# Configuration — loaded once at module level
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

# Defer the Groq client creation so that deployments without the key set at
# import time (e.g. Hugging Face Spaces, where env vars are injected at
# container runtime) can still start up and surface a friendly error message.
_client: Groq | None = None
if GROQ_API_KEY:
    _client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"
MAX_TOKENS = 500
TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an AI/ML career mentor chatbot for students learning AI engineering. "
    "Your role is to guide, explain, and encourage — not to write code for them. "
    "Answer using the provided context when it is relevant to the question. "
    "If the context does not cover the question, answer from your general AI/ML "
    "knowledge but clearly note that it is outside the provided knowledge base. "
    "Be encouraging but honest — do not give vague or generic advice. "
    "Keep answers concise, practical, and focused on actionable guidance. "
    "Avoid essay-length responses."
)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def build_prompt(user_message: str, context: str) -> list[dict[str, str]]:
    """Build a message list suitable for Groq's chat completions API.

    Parameters
    ----------
    user_message:
        The raw text the user typed.
    context:
        Retrieved and formatted context string from ``format_context()``,
        or a fallback message if nothing was retrieved.

    Returns
    -------
    list[dict[str, str]]
        A list of ``{"role": …, "content": …}`` message dicts.
    """
    user_content = (
        f"Retrieved context:\n{context}\n\n"
        f"Student's question:\n{user_message}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def ask_groq(user_message: str) -> dict[str, Any]:
    """Run the full RAG pipeline for *user_message* and return the LLM reply.

    Steps
    -----
    1. Validate that *user_message* is non-empty.
    2. Retrieve relevant chunks from ChromaDB via ``src.rag.query_vectorstore``.
    3. Format the retrieved chunks via ``src.rag.format_context``.
    4. Build a prompt with ``build_prompt()``.
    5. Call the Groq chat completions API.
    6. Return the reply text together with the source filenames.

    Returns
    -------
    dict
        On success: ``{"success": True, "response": str, "sources": list[str]}``
        On failure: ``{"success": False, "error": str}``
    """
    # --- Validate input ----------------------------------------------------
    if not user_message or not user_message.strip():
        return {"success": False, "error": "Message cannot be empty."}

    # --- Retrieve + format context -----------------------------------------
    from src.rag import format_context, query_vectorstore  # noqa: PLC0415

    results = query_vectorstore(user_message, top_k=3)
    context = format_context(results)

    # Collect source filenames for the response metadata
    sources: list[str] = list({r["source"] for r in results})

    # --- Build prompt ------------------------------------------------------
    messages = build_prompt(user_message, context)

    # --- Check that the Groq client is configured --------------------------
    if _client is None:
        return {
            "success": False,
            "error": (
                "GROQ_API_KEY is not set. "
                "Please set the GROQ_API_KEY environment variable. "
                "Get a key at https://console.groq.com/keys"
            ),
        }

    # --- Call Groq API -----------------------------------------------------
    try:
        completion = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": f"Groq API call failed: {exc}",
        }

    reply = completion.choices[0].message.content or ""

    return {
        "success": True,
        "response": reply,
        "sources": sources,
    }
