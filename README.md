# iNeuBytes AI Engineering Internship

*Three end-to-end machine learning projects — from CNNs and sentiment analysis
to a production RAG chatbot.*

**Intern:** Sushit &nbsp;|&nbsp; **Course:** AIINB10626

[Task 1 — CIFAR-10 CNN](#task-1--cifar-10-cnn-classification) ·
[Task 2 — Sentiment Analysis](#task-2--imdb-sentiment-analysis) ·
[Task 3 — AI Mentor Chatbot](#task-3--ai-mentor-chatbot) ·
[License](#license)

---

## Overview

This repository contains three independent projects completed during the AI
Engineering internship at iNeuBytes. Each task lives in its own directory with
self-contained source code, dependencies, and documentation.

| Task | Directory | Domain | Stack |
|------|-----------|--------|-------|
| **1** | [`Task1/cifar10-cnn/`](Task1/cifar10-cnn) | Image Classification | TensorFlow / Keras |
| **2** | [`Task2/sentiment-analysis/`](Task2/sentiment-analysis) | NLP / Sentiment | scikit-learn, TensorFlow |
| **3** | [`Task3/ai-mentor-chatbot/`](Task3/ai-mentor-chatbot) | RAG Chatbot (Production) | Flask, ChromaDB, Groq |

---

## Task 1 — CIFAR-10 CNN Classification

**Folder:** [`Task1/cifar10-cnn/`](Task1/cifar10-cnn)

Compares two convolutional neural network architectures on the CIFAR-10 dataset
(60,000 32x32 colour images, 10 classes).

### Models

| Model | Architecture | Regularization | Test Accuracy | Parameters |
|-------|-------------|----------------|--------------|------------|
| **Traditional CNN** (AlexNet-style) | 3× Conv blocks (64→128→256) + 2× Dense | None | 74.57% | 3,376,970 |
| **Custom CNN** | 3× Conv blocks with 3 layers each + BN + Dropout + Augmentation | BatchNorm, Dropout, LR scheduling, data augmentation | **85.57%** | 4,030,218 |

### Key Result

The custom CNN improved test accuracy by **+11.00 percentage points** over the
baseline (requirement: +3pp), while reducing the train/validation gap from ~22
points to ~1.6 points — indicating the regularization meaningfully reduced
overfitting.

### Pipeline

```bash
cd Task1/cifar10-cnn
pip install -r requirements.txt
python main.py --part both
```

Supports `--part a`, `--part b`, or `--part both` (default). Results and figures
are written to `outputs/`.

---

## Task 2 — IMDB Sentiment Analysis

**Folder:** [`Task2/sentiment-analysis/`](Task2/sentiment-analysis)

Compares classical machine learning models against a deep-learning LSTM for
binary sentiment classification on the IMDB movie review dataset (50,000
balanced reviews).

### Models

| Model | Category | Accuracy | F1-score | Training Time |
|-------|----------|----------|----------|--------------|
| **Logistic Regression** | Classical ML | **0.9000** | **0.9008** | ~3 seconds |
| **LinearSVC** | Classical ML | 0.8919 | 0.8918 | ~5 seconds |
| **LSTM** | Deep Learning | 0.8882 | 0.8903 | ~351 seconds |

### Key Insight

The simpler Logistic Regression beat the LSTM by ~1 point in F1-score while
training **100× faster**. This demonstrates that a well-tuned linear model with
TF-IDF features is often sufficient for text classification when the dataset
is under ~100k samples — a central lesson of the project.

### Pipeline

```bash
cd Task2/sentiment-analysis
pip install -r requirements.txt
# Place IMDB Dataset.csv in data/
python main.py
```

The orchestrator runs all stages sequentially: data preparation, Part A (ML
baselines), Part B (LSTM), and the final comparison table.

---

## Task 3 — AI Mentor Chatbot

**Folder:** [`Task3/ai-mentor-chatbot/`](Task3/ai-mentor-chatbot)

A Retrieval-Augmented Generation (RAG) chatbot that answers AI/ML career and
learning questions. Uses a local knowledge base of markdown documents,
sentence embeddings with ChromaDB for retrieval, and the Groq LLM API
for generation.

**Live demo:** [ai-mentor-chatbot-aovu.onrender.com](https://ai-mentor-chatbot-aovu.onrender.com)

### Architecture

```
User ──> Flask ──> RAG (ChromaDB + fastembed) ──> Groq API ──> Response
                          │
                     knowledge_base/
                     (9 markdown files)
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web server | Flask + gunicorn | HTTP API + static file serving |
| Embeddings | fastembed (ONNX Runtime) | Lightweight text-to-vector (BAAI/bge-small-en-v1.5) |
| Vector store | ChromaDB 0.5.23 | Persistent similarity search |
| LLM | Groq (llama-3.3-70b-versatile) | Answer generation |
| Frontend | Vanilla HTML/CSS/JS | Chat interface |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves chat UI |
| `GET` | `/health` | Health check (returns `{"status": "ok"}`) |
| `POST` | `/respond` | Send a message (+ RAG context) |
| `POST` | `/ingest` | Trigger knowledge base ingestion (background) |

### Run Locally

```bash
cd Task3/ai-mentor-chatbot
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
python src/ingest.py          # Build vector store
python app.py                 # Start server on :5000
```

### Deploy

The app is deployed on **Render** (free tier). The Dockerfile and Procfile are
included for containerized deployment. Key environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key (get at console.groq.com) |
| `PORT` | No | 5000 | Server port (Render sets automatically) |
| `CHROMA_DB_PATH` | No | `./chroma_db` | ChromaDB persistence directory |

### Performance Characteristics

- **Memory:** ~210 MB at idle (embedding model + Flask + ChromaDB) — fits on
  Render's free 512 MB tier.
- **Embedding model:** BAAI/bge-small-en-v1.5 via fastembed (ONNX Runtime) —
  ~90 MB loaded vs ~400 MB with PyTorch-based alternatives.
- **First request latency:** ~1-2 seconds (model pre-loaded at startup).
- **Ingestion:** Triggered manually via `POST /ingest`; runs in a background
  thread to avoid blocking the chat UI.

---

## Repository Structure

```
.
├── README.md                      # This file
├── .gitignore
│
├── Task1/cifar10-cnn/             # CIFAR-10 CNN Classification
│   ├── main.py                    # CLI entry point
│   ├── compare_results.py         # Standalone comparison script
│   ├── requirements.txt
│   ├── src/
│   │   ├── config.py              # Hyperparameters, seeds, paths
│   │   ├── data.py                # CIFAR-10 loading + 90/10 val split
│   │   ├── models.py              # Traditional CNN + Custom CNN builders
│   │   ├── train.py               # Shared training runner
│   │   ├── evaluate.py            # Metrics + comparison logic
│   │   └── visualize.py           # Architecture diagrams + plots
│   └── outputs/                   # Generated figures + metrics
│
├── Task2/sentiment-analysis/      # IMDB Sentiment Analysis
│   ├── main.py                    # Pipeline orchestrator
│   ├── requirements.txt
│   ├── src/
│   │   ├── data_prep.py           # Load, clean, split, save
│   │   ├── part_a_ml.py           # TF-IDF + LR + LinearSVC
│   │   ├── part_b_lstm.py         # Tokenizer + LSTM
│   │   ├── generate_diagrams.py   # Architecture diagrams
│   │   └── utils.py               # Comparison table builder
│   └── outputs/                   # Metrics CSVs + figures
│
└── Task3/ai-mentor-chatbot/       # AI Mentor Chatbot
    ├── app.py                     # Flask entrypoint (production)
    ├── Dockerfile                 # Container image
    ├── Procfile                   # Render start command
    ├── requirements.txt
    ├── src/
    │   ├── chatbot.py             # Groq LLM integration
    │   ├── rag.py                 # Embedding + retrieval logic
    │   ├── ingest.py              # Vector store builder
    │   └── build_arxiv_subset.py  # Dataset utility
    ├── static/                    # Frontend (HTML/CSS/JS)
    ├── data/knowledge_base/       # 9 markdown documents
    └── tests/                     # Postman collection
```

---

## Tech Stack Summary

| Area | Technologies |
|------|-------------|
| Deep Learning | TensorFlow, Keras |
| Classical ML | scikit-learn (LR, SVC, TF-IDF) |
| NLP | NLTK, TF-IDF, Tokenizer |
| Backend | Flask, gunicorn |
| Vector Search | ChromaDB |
| Embeddings | fastembed (ONNX) |
| LLM | Groq API |
| Frontend | Vanilla JS, CSS |
| Deployment | Docker, Render |

---

## Limitations

- **Task 1:** Models trained CPU-only (no GPU). Training times are
  correspondingly higher than GPU-accelerated equivalents.
- **Task 2:** LSTM trained with early stopping at epoch 3 (patience=2). A
  deeper model with pre-trained embeddings and GPU training would likely
  outperform the classical baselines.
- **Task 3:** Knowledge base ingestion is memory-intensive and is not run
  automatically at startup on Render's free tier. The chatbot answers using
  Groq's general knowledge until `/ingest` is called manually.

---

## License

This project is completed as part of an internship with iNeuBytes. All code is
provided for educational purposes.
