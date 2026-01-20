FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Download models at build time so workers don't hit network on first job.
# bart-large-mnli is the zero-shot classifier.
# distilbert-base-uncased-finetuned-sst-2-english gives us a sentiment/risk tone baseline.
RUN python -c "\
from transformers import pipeline; \
pipeline('zero-shot-classification', model='facebook/bart-large-mnli'); \
pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')"

COPY . .
