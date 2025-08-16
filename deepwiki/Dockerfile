# DeepWiki Service Docker Image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone DeepWiki repository (when available)
# For now, we'll use our custom implementation
# RUN git clone https://github.com/AsyncFuncAI/deepwiki-open.git /app/deepwiki

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service implementation
COPY service.py .
COPY config.yaml .
COPY wiki_engine.py .

# Set environment variables
ENV SERVICE_NAME=deepwiki
ENV SERVICE_PORT=8011
ENV WIKI_DATA_PATH=/data/wiki

# Create data directory
RUN mkdir -p /data/wiki

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8011/health || exit 1

# Expose port
EXPOSE 8011

# Run service
CMD ["python", "service.py"]