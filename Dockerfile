# Base image pinned by digest (python:3.12-slim, Python 3.12.15, pulled 2026-10-05).
FROM python:3.12-slim@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VAR_DIR=/var/lib/demo-rag \
    CORPUS_PATH=/app/data/corpus.jsonl \
    EVALS_DIR=/app/evals

WORKDIR /app

RUN useradd --system --uid 10001 --user-group --create-home --home-dir /home/app app \
    && mkdir -p /var/lib/demo-rag \
    && chown app:app /var/lib/demo-rag

COPY requirements.lock .
RUN pip install --require-hashes --no-deps -r requirements.lock

COPY app ./app
COPY data ./data
COPY evals ./evals
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

USER app

# Index, embedding model cache and the cost-guard SQLite live here.
VOLUME ["/var/lib/demo-rag"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
