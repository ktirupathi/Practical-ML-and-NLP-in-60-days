# Day 27: Docker for ML

## Learning Objectives

- Write a Dockerfile that packages an ML application with all its dependencies
- Use multi-stage builds to keep production images small and secure
- Orchestrate multiple services (API + database + dashboard) with docker-compose
- Mount volumes for data persistence and bind-mount code for development
- Follow best practices for caching pip installs and reducing image layers

## Key Concepts

### Why Containerize ML Applications

"It works on my machine" is the classic failure mode when moving an ML model from a data
scientist's laptop to a server. Docker eliminates this by packaging the application, its
Python dependencies, system libraries, and even the model artifact into a single image that
runs identically everywhere. Containers also provide process isolation, resource limits, and
a clean deployment unit for orchestrators like Kubernetes.

### Multi-Stage Builds

ML Docker images can grow large because of heavy dependencies (NumPy, scikit-learn, PyTorch).
Multi-stage builds let you compile or install dependencies in a temporary "builder" stage and
then copy only the runtime artifacts into a slim final image. This can cut image sizes by
50--80%. The builder stage can also run tests or linters before producing the final image,
acting as a lightweight CI gate.

### docker-compose for Local Development

A production ML system often involves more than one container: an API server, a model
registry, a database, and perhaps a dashboard. `docker-compose.yml` lets you define all
these services declaratively, wire up networks and volumes, and bring the entire stack up
with `docker-compose up`. This gives every team member a one-command reproducible environment.

## Practical Example

```python
# --- Dockerfile (multi-stage) ---
"""
# Stage 1: Builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app.py model.joblib ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# --- docker-compose.yml ---
"""
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/model.joblib
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      retries: 3

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
"""

# --- Python script to build and verify the container programmatically ---
import subprocess
import requests
import time

def build_and_test():
    # Build the image
    subprocess.run(
        ["docker", "build", "-t", "ml-api:latest", "."],
        check=True,
    )

    # Run the container in the background
    container = subprocess.run(
        ["docker", "run", "-d", "-p", "8000:8000", "ml-api:latest"],
        capture_output=True, text=True, check=True,
    )
    container_id = container.stdout.strip()

    try:
        # Wait for the service to start
        for _ in range(10):
            try:
                resp = requests.get("http://localhost:8000/health")
                if resp.status_code == 200:
                    print("Health check passed:", resp.json())
                    break
            except requests.ConnectionError:
                time.sleep(1)

        # Test a prediction
        resp = requests.post(
            "http://localhost:8000/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]},
        )
        print("Prediction:", resp.json())
    finally:
        subprocess.run(["docker", "stop", container_id], check=True)

if __name__ == "__main__":
    build_and_test()
```

## Resources

- [Docker official Python guide](https://docs.docker.com/language/python/)
- [Multi-stage builds documentation](https://docs.docker.com/build/building/multi-stage/)
- [docker-compose reference](https://docs.docker.com/compose/compose-file/)

## Up Next

**Day 28 -- Project: Resume Screening:** Apply everything from Days 21-27 to build an end-to-end AI resume screening system.
