# Common Beginner Mistakes in AI/ML Learning

## 1. Notebook-Only Projects
Building models only in Jupyter notebooks with no deployment. Employers want to see a live, usable application, not just a .ipynb file. Always aim to wrap your model in an API and deploy it.

## 2. Skipping Error Handling
Backends that crash on empty input or malformed requests signal inexperience. Always validate input and return clear JSON error messages instead of raw stack traces.

## 3. Oversized Models for Free Hosting
Free platforms like Render or Railway have memory limits (usually 512MB). Trying to deploy large models causes deployment failures. Prefer lightweight models, quantization, or API-based LLMs (like Groq) instead of hosting massive weights yourself.

## 4. No README or Documentation
A GitHub repo with no README, unclear setup instructions, or missing requirements.txt makes projects unusable by reviewers. Every project should be runnable from a clean clone.

## 5. Committing Secrets to GitHub
Never commit API keys, passwords, or tokens. Always use environment variables and .env files (excluded via .gitignore).

## 6. Chasing Tools Instead of Fundamentals
Jumping between frameworks (LangChain today, CrewAI tomorrow) without understanding core concepts like embeddings, retrieval, and prompt design leads to shallow knowledge that breaks down in interviews.

## 7. Ignoring Evaluation
Building a model or chatbot without testing edge cases (empty input, adversarial prompts, malformed data) means bugs surface only in production or during interviews.
