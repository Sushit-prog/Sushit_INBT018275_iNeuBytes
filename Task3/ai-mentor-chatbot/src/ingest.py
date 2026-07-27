"""
Script to build the Chroma vector store from markdown files in
data/knowledge_base/.

Run standalone:
    python src/ingest.py

Expected content: one or more .md files in data/knowledge_base/ with
mentorship / educational content. Each file is split into chunks, embedded
with sentence-transformers, and stored in a persistent ChromaDB collection.
"""

import os
import glob
from pathlib import Path
from typing import Any

import chromadb
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "mentor_knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Resolve the knowledge base directory relative to this file's location
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_markdown_files(directory_path: str) -> list[dict[str, str]]:
    """Read all ``.md`` files from *directory_path*.

    Returns a list of dictionaries, each containing:
        - ``filename``: the base name of the file (e.g. ``"01_topic.md"``)
        - ``content``: the full text content of the file

    If the directory does not exist or no ``.md`` files are found, an empty
    list is returned.
    """
    md_dir = Path(directory_path)
    if not md_dir.is_dir():
        print(f"[WARNING] Directory not found: {md_dir}")
        return []

    md_files = sorted(glob.glob(str(md_dir / "*.md")))
    if not md_files:
        print(f"[WARNING] No .md files found in {md_dir}")
        return []

    documents: list[dict[str, str]] = []
    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as fh:
            content = fh.read()
        documents.append({"filename": Path(filepath).name, "content": content})
        print(f"  Loaded: {Path(filepath).name}")

    return documents


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split *text* into overlapping chunks of roughly *chunk_size* characters.

    The function first attempts to split on paragraph boundaries (``\\n\\n``).
    If a resulting paragraph is still longer than *chunk_size*, it falls back
    to character-level splitting at the nearest sentence or space boundary.

    *overlap* characters from the end of each chunk are prepended to the next
    chunk to preserve context across chunk boundaries.
    """
    paragraphs = text.split("\n\n")
    raw_chunks: list[str] = []
    buffer: list[str] = []

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        # If the paragraph fits within the current buffer, append and continue
        proposed = " ".join(buffer + [stripped]) if buffer else stripped
        if len(proposed) <= chunk_size:
            buffer.append(stripped)
        else:
            # Flush the current buffer as a chunk
            if buffer:
                raw_chunks.append(" ".join(buffer))
                buffer = []
            # If the paragraph alone exceeds chunk_size, split it further
            if len(stripped) > chunk_size:
                raw_chunks.extend(_split_by_chars(stripped, chunk_size))
            else:
                buffer.append(stripped)

    # Flush any remaining buffer
    if buffer:
        raw_chunks.append(" ".join(buffer))

    # Apply overlap between consecutive chunks
    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            # Take the tail of the previous chunk, starting at a word boundary
            # so the overlap doesn't begin mid-word.
            prev = raw_chunks[i - 1]
            if len(prev) >= overlap:
                candidate = len(prev) - overlap
                # Word-boundary: adjust ``candidate`` backward to the nearest
                # preceding whitespace within 50 chars, then start just after it.
                _start = max(0, candidate - 50)
                adj = candidate
                for pos in range(candidate - 1, _start - 1, -1):
                    if prev[pos] in (" ", "\n"):
                        adj = pos + 1
                        break
                prev_tail = prev[adj:]
            else:
                prev_tail = prev
            overlapped.append(prev_tail + chunk)

    return overlapped


def _split_by_chars(text: str, chunk_size: int) -> list[str]:
    """Split *text* into fixed-size chunks, ending on complete words.

    When computing a chunk boundary, the function adjusts the end index
    backward to the nearest preceding whitespace (space or newline) so that
    chunks never cut words in half.  A lookback window of 50 characters
    limits how far back we search; if no whitespace is found within the
    window the chunk is hard-cut at the original boundary.  This prevents
    infinite loops on unusually long tokens such as URLs.
    """
    LOOKBACK = 50
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Word-boundary: adjust ``end`` backward to the nearest whitespace
        # within ``LOOKBACK`` characters.  If none is found, keep hard cut.
        search_from = max(start, end - LOOKBACK)
        for pos in range(end - 1, search_from - 1, -1):
            if text[pos] in (" ", "\n"):
                end = pos
                break

        chunks.append(text[start:end].strip())
        start = end
    return chunks


def ingest_all() -> None:
    """Main pipeline: load, chunk, embed, and store documents in ChromaDB.

    Steps:
        1. Load all ``.md`` files from ``data/knowledge_base/``.
        2. Chunk each file's content.
        3. Initialise the embedding model and Chroma persistent client.
        4. Delete any existing collection with the target name (to avoid
           duplication on re-runs).
        5. Create a fresh collection and add all chunks with metadata.
        6. Print a summary of the operation.
    """
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Ingesting knowledge base into ChromaDB")
    print("=" * 60)
    print(f"\nKnowledge base directory: {KNOWLEDGE_BASE_DIR}")

    documents = load_markdown_files(str(KNOWLEDGE_BASE_DIR))
    if not documents:
        print("\n[WARNING] No documents to ingest. Exiting.")
        return

    print(f"\nLoaded {len(documents)} file(s).")

    # ------------------------------------------------------------------
    # 2. Chunk
    # ------------------------------------------------------------------
    all_chunks: list[str] = []
    chunk_metadata: list[dict[str, Any]] = []
    chunk_ids: list[str] = []

    for doc in documents:
        chunks = chunk_text(doc["content"], DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({"source": doc["filename"], "chunk_index": i})
            chunk_ids.append(f"{doc['filename']}_{i}")

    print(f"Created {len(all_chunks)} chunk(s) from {len(documents)} file(s).\n")

    # ------------------------------------------------------------------
    # 3. Embedding model
    # ------------------------------------------------------------------
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.\n")

    # ------------------------------------------------------------------
    # 4. Chroma client
    # ------------------------------------------------------------------
    print(f"Chroma persistence path: {CHROMA_DB_PATH}")
    client: PersistentClient = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Drop existing collection so re-runs don't duplicate
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except NotFoundError:
        # Collection didn't exist on first run — that's fine
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Created collection '{COLLECTION_NAME}'.\n")

    # ------------------------------------------------------------------
    # 5. Embed & add
    # ------------------------------------------------------------------
    BATCH_SIZE = 32
    total = len(all_chunks)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_chunks = all_chunks[batch_start:batch_end]
        batch_ids = chunk_ids[batch_start:batch_end]
        batch_metadata = chunk_metadata[batch_start:batch_end]

        # SentenceTransformer.encode() returns numpy array → convert to list
        embeddings = model.encode(batch_chunks, show_progress_bar=False).tolist()

        collection.add(
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_metadata,
            ids=batch_ids,
        )
        print(f"  Indexed {batch_end}/{total} chunk(s)")

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Ingestion complete.")
    print(f"  Files processed:   {len(documents)}")
    print(f"  Chunks created:    {total}")
    print(f"  Collection:        {COLLECTION_NAME}")
    print(f"  Embedding model:   {EMBEDDING_MODEL_NAME}")
    print(f"  Chroma path:       {CHROMA_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    ingest_all()
