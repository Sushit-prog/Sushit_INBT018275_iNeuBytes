# AI/ML Tool Comparison Guide

## LangChain vs LangGraph
LangChain is best for simple, linear chains (prompt to model to output). LangGraph is better for stateful, multi-step, or multi-agent workflows where you need conditional branching, loops, or persistent memory across steps.

## Vector Databases
- Chroma: lightweight, easy local setup, good for small to medium projects and prototyping
- FAISS: fast similarity search, no persistence layer built in by default, good for research and benchmarking
- Pinecone: managed cloud service, better for production scale but requires an external account and has usage limits on free tier

## LLM Hosting Options
- Groq: extremely fast inference, generous free tier, good for latency-sensitive apps like chatbots
- OpenAI API: high quality, well-documented, but paid with no meaningful free tier for sustained use
- Ollama: runs models locally, no API costs, but requires more RAM/compute than most free-tier laptops have for larger models

## Embedding Models
- all-MiniLM-L6-v2: lightweight, fast, good default for CPU-only environments and small-to-medium document sets
- OpenAI text-embedding models: higher quality but require API calls and cost per use

## Choosing for Resource-Constrained Environments
When working with limited RAM and CPU-only hardware, prioritize API-based LLM inference (Groq) over local model hosting, and lightweight embedding models over large ones. This keeps both development and deployment feasible on free-tier hosting.
