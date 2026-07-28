---
title: AI Mentor Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# AI Mentor Chatbot

A RAG-powered chatbot that answers programming/mentorship questions using a
local knowledge base and the Groq LLM API. Built with Flask, ChromaDB, and
sentence-transformers.

## Quick Start (Local)

1.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate    # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2.  Create a `.env` file with your Groq API key:
    ```bash
    echo "GROQ_API_KEY=your_key_here" > .env
    ```

3.  Build the vector store:
    ```bash
    python src/ingest.py
    ```

4.  Run the app:
    ```bash
    python app.py
    ```

5.  Open `http://127.0.0.1:5000` in a browser.

## Project Structure

```
├── app.py                  # Flask entrypoint
├── Dockerfile              # Container image (Hugging Face Spaces ready)
├── start.sh                # Startup script (auto-ingest on first run)
├── requirements.txt        # Python dependencies
├── data/
│   └── knowledge_base/     # Markdown files ingested into ChromaDB
├── src/
│   ├── ingest.py           # Builds the Chroma vector store
│   ├── rag.py              # Retrieval logic (ChromaDB + embeddings)
│   └── chatbot.py          # Groq LLM call logic
├── static/
│   ├── index.html          # Chat front-end
│   ├── style.css           # Styles
│   └── script.js           # Front-end logic
└── README.md
```

## Deploy to Hugging Face Spaces

1.  Push this repo to GitHub.
2.  Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
3.  Choose **Docker** as the SDK.
4.  Connect your GitHub repository.
5.  In the Space **Settings → Repository Secrets**, add:
    - `GROQ_API_KEY` — your Groq API key from [console.groq.com](https://console.groq.com/keys)
6.  The app will build and start automatically.

## Usage

Open the Space URL in a browser and start chatting! The first startup may take
30–60 seconds while the embedding model loads and the knowledge base is indexed.
Subsequent restarts will be faster.
