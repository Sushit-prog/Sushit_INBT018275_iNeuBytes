#!/bin/bash
# ---------------------------------------------------------------------------
# Startup script for Hugging Face Spaces (Docker)
#
# Runs knowledge base ingestion once (on first deploy or after a fresh build)
# and then starts the Flask app.  Uses a sentinel file (.ingestion_done) to
# skip re-ingestion on subsequent container restarts.
# ---------------------------------------------------------------------------

set -e

echo "=========================================="
echo " AI Mentor Chatbot — Hugging Face Spaces"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Knowledge base ingestion (first run only)
# ---------------------------------------------------------------------------
if [ ! -f ".ingestion_done" ]; then
    echo ""
    echo "[INFO] First launch — running knowledge base ingestion..."
    echo ""

    python src/ingest.py

    # Mark ingestion as done so subsequent restarts skip this step
    touch .ingestion_done

    echo ""
    echo "[INFO] Ingestion complete."
    echo ""
else
    echo ""
    echo "[INFO] Knowledge base already ingested — skipping."
    echo ""
fi

# ---------------------------------------------------------------------------
# 2. Start Flask
# ---------------------------------------------------------------------------
echo ""
echo "[INFO] Starting Flask server..."
echo ""

exec python app.py
