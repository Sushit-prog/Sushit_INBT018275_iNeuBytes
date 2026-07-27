# Understanding RAG (Retrieval-Augmented Generation)

## What RAG Solves
LLMs have a fixed knowledge cutoff and can hallucinate facts. RAG grounds responses in real documents by retrieving relevant context before generating an answer, instead of relying purely on the model's training data.

## How RAG Works
1. Documents are split into chunks
2. Each chunk is converted into a vector embedding (using models like all-MiniLM-L6-v2)
3. Embeddings are stored in a vector database (Chroma, FAISS, Pinecone)
4. At query time, the user's question is embedded and compared to stored chunks using similarity search
5. The most relevant chunks are retrieved and inserted into the LLM prompt as context
6. The LLM generates an answer grounded in that context

## Naive RAG vs Corrective RAG (CRAG)
Naive RAG retrieves top-k chunks and trusts them blindly. CRAG adds an evaluation step: an LLM or classifier scores each retrieved chunk for relevance before use, and can trigger a fallback (like a web search or query rewrite) if retrieval quality is poor.

## Common Pitfalls
- Chunking documents too large (loses precision) or too small (loses context)
- Using a mismatched embedding model between ingestion and query time
- Not handling the case where no relevant documents are found
- Forgetting to cap context length, causing prompt overflow
