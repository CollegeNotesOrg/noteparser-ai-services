# RagFlow Service Docker Image
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

# Clone RagFlow repository
RUN git clone https://github.com/infiniflow/ragflow.git /app/ragflow

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service wrapper
COPY service.py .
COPY config.yaml .

# Set environment variables
ENV PYTHONPATH=/app/ragflow:$PYTHONPATH
ENV SERVICE_NAME=ragflow
ENV SERVICE_PORT=8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Expose port
EXPOSE 8010

# Run service
CMD ["python", "service.py"]