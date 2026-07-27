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
    app.run(host="0.0.0.0", port=port, debug=True)
