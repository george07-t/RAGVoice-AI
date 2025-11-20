# syntax=docker/dockerfile:1

# Use the official Python base image
ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install build dependencies required for Python packages with native extensions
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Create a new directory for our application code
WORKDIR /app

# Copy requirements.txt first for better layer caching
COPY requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for RAG system
RUN . venv/bin/activate && \
    pip install --no-cache-dir faiss-cpu sentence-transformers tiktoken pytz

# Copy all application files into the container
COPY . .

# Create necessary directories
RUN mkdir -p vector_store documents && \
    chown -R appuser:appuser /app

# Switch to the non-privileged user
USER appuser

# Run the application using the virtual environment
CMD ["/app/venv/bin/python", "src/agent.py", "start"]