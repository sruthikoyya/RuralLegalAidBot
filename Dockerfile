# ─────────────────────────────────────────────────────────────────────────────
# RuralLegalAidBot v2 — Dockerfile
# Base: Python 3.10 + CUDA 11.8 + cuDNN 8
# ─────────────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── OS Dependencies ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-tel \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python Dependencies ───────────────────────────────────────────────────────
# Install PyTorch with CUDA 11.8 first (must come before other packages)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy Project Files ────────────────────────────────────────────────────────
COPY . .

# Create data directories
RUN mkdir -p data/legal_docs data/chroma_db data/temp

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health Check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Start ─────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
