# Multi-stage optimized Dockerfile for Emergency Supply Chain Swarm
FROM python:3.12-slim AS base

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and install editable package
COPY . .
RUN pip install --no-cache-dir --no-deps --no-build-isolation -e .

# Pre-train / verify ML models during build stage
RUN python ml/train_models.py

# Expose ports: 8501 for Streamlit UI, 8000 for FastAPI API
EXPOSE 8501 8000

# Healthcheck for Streamlit service
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: Launch Streamlit Live Command Center
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

