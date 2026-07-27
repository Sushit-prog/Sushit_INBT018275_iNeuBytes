# AI Mentor Chatbot

A RAG-powered chatbot that answers programming/mentorship questions using a
local knowledge base and the Groq LLM API.

## Project Structure

```
Task3/ai-mentor-chatbot/
├── app.py                  # Flask entrypoint
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── data/
│   └── knowledge_base/     # Place markdown files here (ingested into ChromaDB)
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Builds the Chroma vector store from knowledge_base/
│   ├── rag.py              # Retrieval logic (ChromaDB + embeddings)
│   └── chatbot.py          # Groq LLM call logic
├── static/
│   ├── index.html          # Chat front-end
│   ├── style.css           # Styles
│   └── script.js           # Front-end logic
├── tests/
│   └── postman_collection.json  # API test collection
└── README.md
```

## Setup

1.  Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate    # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2.  Copy `.env.example` to `.env` and add your Groq API key:
    ```bash
    cp .env.example .env
    # Edit .env: GROQ_API_KEY=your_key_here
    ```

3.  Place markdown knowledge documents in `data/knowledge_base/`.

4.  Build the vector store:
    ```bash
    python src/ingest.py
    ```

5.  Run the app:
    ```bash
    python app.py
    ```

## Usage

Open `http://127.0.0.1:5000` in a browser and start chatting.
