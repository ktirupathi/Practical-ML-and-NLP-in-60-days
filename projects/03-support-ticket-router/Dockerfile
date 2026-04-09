# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate prefix (layer-caching friendly)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Download NLTK stopwords corpus during build so the container is self-contained
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
# Copy NLTK data
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

# Copy application source code
COPY src/ ./src/
COPY app.py .

# Create directories for artifacts and logs
RUN mkdir -p artifacts logs

EXPOSE 8000

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check: polls the root endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
