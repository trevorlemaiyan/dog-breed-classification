# Production CPU-Optimized Dockerfile for Dog Breed Classification API
# Compatible with Hugging Face Spaces (Port 7860) and Container Hosts (Render, AWS, GCP)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# Install minimal OS dependencies for OpenCV, PIL, and health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY --chown=user:user . .

# Ensure model weights or baseline checkpoint structure is present
RUN python scripts/download_weights.py || true

# Switch to non-root user
USER user

EXPOSE 7860

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
