# Multi-stage build: keeps the final image lean
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Final image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY shared/        ./shared/
COPY 01-basic-rag/  ./01-basic-rag/
COPY 02-chunking-strategies/ ./02-chunking-strategies/
COPY 03-advanced-rag/ ./03-advanced-rag/
COPY app.py         ./app.py

# Non-root user for security
RUN useradd -m appuser
USER appuser

EXPOSE 8080

# Default: run the HTTP API server
CMD ["python", "app.py"]
