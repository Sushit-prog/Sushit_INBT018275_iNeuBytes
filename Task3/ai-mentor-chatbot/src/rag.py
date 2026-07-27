"""
Retrieval-Augmented Generation (RAG) logic.

Handles embedding queries with sentence-transformers, connecting to a
persistent ChromaDB collection, and retrieving the most relevant chunks
for a user query. No LLM calls live here — those belong in chatbot.py.
"""

import os
from typing import Any

import chromadb
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration — loaded once at module level
# ---------------------------------------------------------------------------

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "mentor_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load the embedding model once so it is reused across all calls.
# (SentenceTransformer uses caching internally for the same model.)
_embedding_model: SentenceTransformer = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def get_vectorstore() -> PersistentClient:
    """Return a :class:`chromadb.PersistentClient` connected to
    ``CHROMA_DB_PATH``.

    Safe to call multiple times — each call returns a new lightweight
    client handle to the same on-disk database.
    """
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def query_vectorstore(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Embed *query* and retrieve the *top_k* most similar chunks from the
    ``mentor_knowledge_base`` Chroma collection.

    Returns a list of dictionaries:
    .. code-block:: python

        [
            {
                "text":   "<chunk text>",
                "source": "<filename of the source document>",
                "score":  <cosine distance as a float>,
            },
            ...
        ]

    If the collection does not exist or is empty, a warning is printed and
    an empty list is returned. All Chroma query errors are caught gracefully
    so the caller never receives an exception from this function.
    """
    try:
        client = get_vectorstore()
    except Exception as exc:
        print(f"[WARNING] Failed to connect to ChromaDB at '{CHROMA_DB_PATH}': {exc}")
        return []

    # Try to get the collection — it may not exist if ingest.py hasn't run yet
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        print(
            f"[WARNING] Collection '{COLLECTION_NAME}' not found. "
            f"Run `python src/ingest.py` first."
        )
        return []

    # Check whether the collection has data
    count = collection.count()
    if count == 0:
        print(f"[WARNING] Collection '{COLLECTION_NAME}' is empty. "
              f"Run `python src/ingest.py` to populate it.")
        return []

    # Embed the query
    try:
        query_embedding = _embedding_model.encode(query, show_progress_bar=False).tolist()
    except Exception as exc:
        print(f"[WARNING] Failed to embed query: {exc}")
        return []

    # Run similarity search
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        print(f"[WARNING] Chroma query failed: {exc}")
        return []

    # Parse results into a clean list of dicts
    retrieved: list[dict[str, Any]] = []

    # Chroma returns lists-of-lists (one inner list per query — we have one query)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return []

    for doc_text, meta, dist in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc_text,
            "source": meta.get("source", "unknown") if meta else "unknown",
            "score": dist,
        })

    return retrieved


def format_context(results: list[dict[str, Any]]) -> str:
    """Format a list of retrieved chunks into a single string block suitable
    for injecting into an LLM prompt.

    Each chunk is separated by a horizontal rule and its source filename is
    noted. If *results* is empty a message indicating that no relevant context
    was found is returned.
    """
    if not results:
        return "[No relevant context found in the knowledge base.]"

    blocks: list[str] = []
    for i, item in enumerate(results, 1):
        blocks.append(
            f"--- Begin chunk {i} (source: {item['source']}) ---\n"
            f"{item['text']}\n"
            f"--- End chunk {i} ---"
        )

    return "\n\n".join(blocks)
