# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY app.py .

# Create directories for artifacts and logs at runtime
RUN mkdir -p artifacts logs

# Expose the port that uvicorn will listen on
EXPOSE 8000

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check: hit the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the FastAPI application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
