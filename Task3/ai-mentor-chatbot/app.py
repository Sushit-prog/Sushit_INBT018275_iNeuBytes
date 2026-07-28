"""
Flask entrypoint for the AI Mentor Chatbot.

Run with:
    python app.py

This module initialises the Flask application, registers API routes, and
serves the front-end static files. The chatbot uses RAG (Retrieval-Augmented
Generation) to answer user queries based on a knowledge base.
"""

import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.chatbot import ask_groq


app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)


# ---------------------------------------------------------------------------
# Startup initialisation
# ---------------------------------------------------------------------------

def _init_model() -> None:
    """Pre-load the SentenceTransformer embedding model at startup so the
    first user request doesn't trigger a slow load (or OOM crash).

    We deliberately **skip** auto-ingestion of the knowledge base here because
    the ingestion pipeline (load + chunk + embed + index) pushes Render's free
    tier (512 MB) over the memory limit.  The chatbot answers questions using
    Groq's general knowledge even when the Chroma collection is empty, and the
    knowledge base can be populated manually by visiting ``/ingest``.
    """
    from src.rag import warmup  # noqa: PLC0415

    try:
        warmup()
    except Exception as exc:
        print(f"[WARNING] Failed to pre-load embedding model: {exc}")


print("Initialising AI Mentor Chatbot...")
_init_model()
print("Startup complete.\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the front-end ``index.html``."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    """Health-check endpoint — returns ``{"status": "ok"}``."""
    return jsonify({"status": "ok"}), 200


@app.route("/ingest", methods=["POST"])
def ingest_kb():
    """Trigger knowledge-base ingestion on demand.

    Returns immediately, ingestion runs in the background so the chat UI
    stays responsive.  Check the service logs for progress.
    """
    import threading

    def _run() -> None:
        from src.ingest import ingest_all  # noqa: PLC0415
        from src.rag import get_embedding_model, warmup  # noqa: PLC0415
        try:
            print("[/ingest] Starting background ingestion...")
            warmup()
            model = get_embedding_model()
            ingest_all(model=model)
            print("[/ingest] Ingestion complete.")
        except Exception as exc:
            print(f"[/ingest] Ingestion failed: {exc}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"status": "ingestion started"}), 202


@app.route("/respond", methods=["POST"])
def respond():
    """Handle a chat message from the user.

    Expects a JSON body with a ``"message"`` field containing the user's
    question. Returns the AI mentor's reply together with source filenames.
    """
    try:
        # --- Validate JSON body --------------------------------------------------
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Missing 'message' field in request body"}), 400

        message = data.get("message")
        if message is None:
            return jsonify({"error": "Missing 'message' field in request body"}), 400

        stripped = message.strip()
        if not stripped:
            return jsonify({"error": "Message cannot be empty"}), 400

        # --- Call the chatbot ----------------------------------------------------
        result = ask_groq(stripped)

        if not result["success"]:
            return jsonify({"error": result["error"]}), 500

        return jsonify({
            "response": result["response"],
            "sources": result["sources"],
        }), 200

    except Exception:
        # Catch-all for truly unexpected errors (e.g. import failures)
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
