FROM python:3.11.9-slim

WORKDIR /app

# Layer caching: install Python deps before copying source
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Render and most platforms set PORT via environment variable.
# Default 10000 for Render, 8080 for Cloud Run, 7860 for HF Spaces.
EXPOSE 10000

# Use gunicorn for production stability (vs Flask dev server)
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120

