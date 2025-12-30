# ==============================================================================
# Docling Knowledge Base ExApp for Nextcloud
# Multi-stage build for AI-powered document processing and RAG
# ==============================================================================

# Build stage
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies (including cmake for llama-cpp-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpoppler-cpp-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Production stage
# ==============================================================================

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="Docling Knowledge Base for Nextcloud"
LABEL org.opencontainers.image.description="AI-powered document processing and knowledge base with RAG capabilities"
LABEL org.opencontainers.image.vendor="Docling Nextcloud Team"
LABEL org.opencontainers.image.licenses="CC-BY-NC-SA-4.0"
LABEL org.opencontainers.image.authors="Ihsan Mokhlis"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpoppler-cpp0v5 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create directories
WORKDIR /app
RUN mkdir -p /app/data /app/data/chroma /app/cache && \
    chown -R appuser:appuser /app

# Copy application
COPY --chown=appuser:appuser ex_app/lib/ /app/
COPY --chown=appuser:appuser healthcheck.sh /app/
COPY --chown=appuser:appuser start.sh /app/

RUN chmod +x /app/healthcheck.sh /app/start.sh

USER appuser

# Environment for models and data
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache
ENV DOCLING_CACHE_DIR=/app/cache
ENV CHROMA_PERSIST_DIRECTORY=/app/data/chroma
ENV LLM_CACHE_DIR=/app/cache/models

# LLM Configuration - Uses embedded local LLM by default (no external services!)
ENV LLM_MODEL=qwen2-0.5b
ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
# Set to "true" to disable LLM features
ENV DISABLE_LLM=false

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Volumes for persistent data
VOLUME ["/app/data", "/app/cache"]

CMD ["/app/start.sh"]
