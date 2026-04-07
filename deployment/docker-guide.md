# Docker Guide for ML Applications

This guide covers everything you need to containerize a Python ML/NLP application — from writing an optimized Dockerfile to running a full stack with docker-compose including Redis, a model server, and monitoring.

**Used in:** Day 27 — Docker for ML, Day 30 — Testing and CI/CD

---

## Table of Contents

- [Docker Basics for ML](#docker-basics-for-ml)
- [Writing an Optimized Dockerfile](#writing-an-optimized-dockerfile)
- [Multi-Stage Builds](#multi-stage-builds)
- [Docker Compose for ML Stack](#docker-compose-for-ml-stack)
- [Volume Mounts for Models](#volume-mounts-for-models)
- [Environment Variables](#environment-variables)
- [Complete Working Example](#complete-working-example)
- [Docker Hub Push Guide](#docker-hub-push-guide)
- [Common Commands Reference](#common-commands-reference)

---

## Docker Basics for ML

### Why Docker for ML?

| Problem | Docker Solution |
|---------|----------------|
| "Works on my machine" | Identical environment everywhere |
| Dependency conflicts | Isolated container per service |
| Python version mismatch | Pin exact version in Dockerfile |
| CUDA / GPU driver issues | Use official NVIDIA base images |
| Model deployment complexity | Package model + API + dependencies together |
| Reproducibility | `docker pull image:sha256-hash` is deterministic |

### Key concepts

```
Image     = Blueprint (built from Dockerfile)
Container = Running instance of an image
Layer     = Each RUN/COPY instruction adds a layer (cached separately)
Volume    = Persistent storage that survives container restarts
Registry  = Docker Hub / GHCR / ECR — stores images
```

### Install Docker

```bash
# Linux (Ubuntu/Debian)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker run hello-world
```

---

## Writing an Optimized Dockerfile

### The naive approach (what NOT to do)

```dockerfile
# BAD — don't do this
FROM python:3.11
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

Problems: large image, no layer caching, root user, rebuilds everything on any change.

### Optimized single-stage Dockerfile

```dockerfile
# ── Base image ─────────────────────────────────────────────────────────────────
# Use slim variant: ~50MB vs ~900MB for full python image
FROM python:3.11-slim

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
    # ↑ Clean apt cache in the SAME RUN layer to avoid bloating the image

# ── Non-root user (security best practice) ────────────────────────────────────
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# ── Install Python dependencies BEFORE copying source code ────────────────────
# This layer is cached as long as requirements.txt doesn't change.
# Source code changes won't invalidate this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ───────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Switch to non-root user ────────────────────────────────────────────────────
USER appuser

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# ── Start command ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### .dockerignore (critical for build speed)

```
# .dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.git/
.gitignore
.env
*.env
venv/
.venv/
env/
.pytest_cache/
.mypy_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
data/                 # Don't bake large data files into image
notebooks/            # Jupyter notebooks not needed in production
tests/                # Can include for CI builds, exclude for prod
*.log
.DS_Store
Thumbs.db
```

---

## Multi-Stage Builds

Multi-stage builds produce a lean production image by separating the build environment from the runtime environment. Essential when using compiled dependencies (e.g., numpy, torch).

```dockerfile
# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Builder — install dependencies with full build tools
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build tools for compiling C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create a virtual environment inside the builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Runtime — copy only the venv and app code
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# Copy the pre-built virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Minimal runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

Build and check size difference:

```bash
docker build -t ml-api:single-stage -f Dockerfile.single .
docker build -t ml-api:multi-stage  -f Dockerfile.multi  .
docker images | grep ml-api
# ml-api  multi-stage    ~450MB
# ml-api  single-stage   ~900MB
```

---

## Docker Compose for ML Stack

A production-grade `docker-compose.yml` for a complete ML application:

```yaml
# docker-compose.yml
version: "3.9"

services:
  # ── ML API ──────────────────────────────────────────────────────────────────
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime          # Use the runtime stage from multi-stage build
    image: ml-api:latest
    container_name: ml-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - API_KEY=${API_KEY}
      - MODEL_PATH=/app/models/classifier.pkl
      - VECTORIZER_PATH=/app/models/vectorizer.pkl
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./models:/app/models:ro    # Read-only model mount
      - ./logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - ml-network
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G
        reservations:
          memory: 512M

  # ── Redis (caching + rate limiting) ─────────────────────────────────────────
  redis:
    image: redis:7-alpine
    container_name: ml-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ml-network

  # ── Nginx reverse proxy ──────────────────────────────────────────────────────
  nginx:
    image: nginx:alpine
    container_name: ml-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    networks:
      - ml-network

  # ── Prometheus monitoring ────────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:latest
    container_name: ml-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - ml-network

  # ── Grafana dashboard ────────────────────────────────────────────────────────
  grafana:
    image: grafana/grafana:latest
    container_name: ml-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - ml-network

networks:
  ml-network:
    driver: bridge

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

Run the full stack:

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f api

# Scale the API horizontally
docker compose up -d --scale api=3

# Stop everything
docker compose down

# Stop and remove volumes
docker compose down -v
```

---

## Volume Mounts for Models

Models can be large (BERT = 400 MB, LLaMA = 7 GB+). Never bake them into the image.

### Strategy 1: Bind mount from host (development)

```bash
docker run -v $(pwd)/models:/app/models:ro ml-api:latest
```

### Strategy 2: Named Docker volume (production)

```bash
# Create volume
docker volume create ml-models

# Copy model artifacts into volume
docker run --rm \
  -v $(pwd)/models:/source \
  -v ml-models:/target \
  alpine cp -r /source/. /target/

# Use in container
docker run -v ml-models:/app/models:ro ml-api:latest
```

### Strategy 3: Download on startup (CI/CD friendly)

```dockerfile
# In Dockerfile — download model at build time (baked in)
RUN python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = 'ProsusAI/finbert'
AutoTokenizer.from_pretrained(model_name, cache_dir='/app/model_cache')
AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='/app/model_cache')
print('Model cached successfully')
"
```

### Strategy 4: S3/GCS download on container startup

```bash
# entrypoint.sh
#!/bin/bash
set -e

echo "Downloading model from S3..."
aws s3 cp s3://my-bucket/models/classifier.pkl /app/models/classifier.pkl
aws s3 cp s3://my-bucket/models/vectorizer.pkl /app/models/vectorizer.pkl
echo "Model download complete."

exec "$@"
```

```dockerfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Environment Variables

### .env file (never commit to git)

```bash
# .env
API_KEY=your-production-api-key-here
DEBUG=false
MODEL_PATH=/app/models/classifier.pkl
VECTORIZER_PATH=/app/models/vectorizer.pkl
MODEL_VERSION=v2
REDIS_URL=redis://redis:6379/0
GRAFANA_PASSWORD=secure-grafana-password
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

### Reference in docker-compose.yml

```yaml
services:
  api:
    env_file:
      - .env            # Load all vars from .env
    environment:
      - NODE_ENV=production   # Override specific vars
```

### Pass at runtime

```bash
docker run \
  -e API_KEY="$(cat /run/secrets/api-key)" \
  -e DEBUG=false \
  ml-api:latest
```

---

## Complete Working Example

Build and run the full ML API with one command:

```bash
# 1. Clone and navigate
git clone https://github.com/ktirupathi/practical-ml-and-nlp-in-60-days.git
cd practical-ml-and-nlp-in-60-days

# 2. Train and save model (generates models/ artifacts)
python scripts/train_model.py

# 3. Copy .env template
cp .env.example .env
# Edit .env with your API_KEY

# 4. Build image
docker build -t ml-api:latest .

# 5. Run with docker compose (API + Redis)
docker compose up -d

# 6. Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: $(grep API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"text": "Revenue increased by 25% year over year"}'

# 7. View logs
docker compose logs -f api
```

### Nginx configuration

`nginx/nginx.conf`:

```nginx
events { worker_connections 1024; }

http {
    upstream api_backend {
        server api:8000;
        # Add more api servers here when scaling
        # server api_2:8000;
    }

    server {
        listen 80;
        server_name _;

        # Rate limiting at nginx level (belt + suspenders)
        limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;

        location /api/ {
            limit_req zone=api burst=5 nodelay;
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 120s;
        }

        location /health {
            proxy_pass http://api_backend;
            access_log off;   # Don't log health checks
        }
    }
}
```

---

## Docker Hub Push Guide

```bash
# 1. Create free account at hub.docker.com
# 2. Log in
docker login
# Enter your Docker Hub username and password

# 3. Tag your image with your username
docker tag ml-api:latest yourusername/ml-sentiment-api:v1.0.0
docker tag ml-api:latest yourusername/ml-sentiment-api:latest

# 4. Push to Docker Hub
docker push yourusername/ml-sentiment-api:v1.0.0
docker push yourusername/ml-sentiment-api:latest

# 5. Anyone can now pull your image
docker pull yourusername/ml-sentiment-api:latest

# 6. Automate with GitHub Actions
# .github/workflows/docker-publish.yml
```

GitHub Actions workflow for automated Docker Hub push:

```yaml
# .github/workflows/docker-publish.yml
name: Build and Push Docker Image

on:
  push:
    branches: [main]
    tags: ['v*.*.*']

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ secrets.DOCKER_USERNAME }}/ml-sentiment-api
          tags: |
            type=semver,pattern={{version}}
            type=sha,prefix=sha-

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Common Commands Reference

```bash
# Build
docker build -t my-image:tag .
docker build --target runtime -t my-image:prod .   # specific stage
docker build --no-cache -t my-image:fresh .         # bypass cache

# Run
docker run -d -p 8000:8000 --name my-container my-image:tag
docker run -it --rm my-image:tag /bin/bash          # interactive shell
docker run --gpus all my-image:gpu-tag              # GPU access

# Inspect
docker ps                        # running containers
docker ps -a                     # all containers (including stopped)
docker images                    # local images
docker inspect my-container      # detailed container info
docker stats                     # live resource usage

# Logs
docker logs my-container
docker logs -f my-container      # follow
docker logs --tail=100 my-container

# Exec into running container
docker exec -it my-container /bin/bash
docker exec my-container cat /app/logs/app.log

# Cleanup
docker rm my-container                      # remove stopped container
docker rmi my-image:tag                     # remove image
docker system prune -f                      # remove all unused resources
docker system prune -af --volumes           # nuclear option — remove everything

# Volumes
docker volume ls
docker volume inspect ml-models
docker volume rm ml-models

# Networks
docker network ls
docker network inspect ml-network
```
