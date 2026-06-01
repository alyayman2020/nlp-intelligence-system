# syntax=docker/dockerfile:1
# ---- builder ---------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install --prefix=/install .

# Pre-download NLTK corpora into the image so the container is offline-ready.
RUN python -c "import sys; sys.path.insert(0,'/install/lib/python3.11/site-packages')" || true
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m nltk.downloader -d /usr/share/nltk_data stopwords wordnet omw-1.4

# ---- runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/usr/share/nltk_data \
    MODELS_DIR=/app/models \
    DATA_DIR=/app/data

# Non-root user
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data
COPY src ./src

# Models are mounted at runtime (-v ./models:/app/models).
RUN mkdir -p /app/models /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "nlp_system.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
