# Configuration

This guide covers the configuration options for NoteParser AI Services.

## Environment Variables

The AI services can be configured using environment variables. Create a `.env` file in the root directory:

```bash
# Core Service Configuration
RAGFLOW_PORT=8010
DEEPWIKI_PORT=8011

# Additional Services (Experimental - Not yet documented)
# DOLPHIN_PORT=8012
# LANGEXTRACT_PORT=8013

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5434
POSTGRES_DB=noteparser_ai
POSTGRES_USER=noteparser
POSTGRES_PASSWORD=your_password_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_PASSWORD=your_redis_password

# Vector Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your_api_key

WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080

# AI Model Configuration
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_hf_token

# Service Features
ENABLE_RAGFLOW=true
ENABLE_DEEPWIKI=true
ENABLE_VECTOR_DB=true
ENABLE_CACHING=true

# Performance Settings
MAX_WORKERS=4
REQUEST_TIMEOUT=30
MAX_RETRIES=3
BATCH_SIZE=32
```

## Docker Compose Configuration

### Development Mode

For development, use the `docker-compose.yml` with development overrides:

```yaml
version: '3.8'

services:
  ragflow:
    environment:
      - DEBUG=true
      - LOG_LEVEL=debug
    volumes:
      - ./ragflow:/app
      - ./data/ragflow:/data
    ports:
      - "8010:8010"

  deepwiki:
    environment:
      - DEBUG=true
      - LOG_LEVEL=debug
    volumes:
      - ./deepwiki:/app
      - ./data/deepwiki:/data
    ports:
      - "8011:8011"
```

### Production Mode

For production deployment, use environment-specific configurations:

```yaml
version: '3.8'

services:
  ragflow:
    environment:
      - DEBUG=false
      - LOG_LEVEL=info
      - WORKERS=4
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Service-Specific Configuration

### RagFlow Service

Configure RagFlow in `ragflow/config.yaml`:

```yaml
service:
  name: ragflow
  version: 1.0.0
  port: 8010

embedding:
  model: text-embedding-ada-002
  dimension: 1536
  batch_size: 100

indexing:
  chunk_size: 500
  chunk_overlap: 50
  max_chunks: 1000

retrieval:
  top_k: 5
  similarity_threshold: 0.7
  rerank: true
```

### DeepWiki Service

Configure DeepWiki in `deepwiki/config.yaml`:

```yaml
service:
  name: deepwiki
  version: 1.0.0
  port: 8011

knowledge_graph:
  backend: neo4j
  max_nodes: 10000
  max_relationships: 50000

wiki:
  auto_link: true
  extract_entities: true
  generate_summaries: true
```

## Database Configuration

### PostgreSQL with pgvector

Initialize the database with vector support:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

### Redis Cache Configuration

Configure Redis for caching:

```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Vector Database Configuration

### Qdrant

Configure Qdrant in `data/qdrant/config.yaml`:

```yaml
storage:
  storage_path: ./storage
  wal:
    wal_capacity_mb: 32
    wal_segments_ahead: 0

service:
  http_port: 6333
  grpc_port: 6334
  host: 0.0.0.0

cluster:
  enabled: false
```

### Weaviate

Configure Weaviate schema:

```json
{
  "class": "Document",
  "properties": [
    {
      "name": "title",
      "dataType": ["text"]
    },
    {
      "name": "content",
      "dataType": ["text"]
    },
    {
      "name": "category",
      "dataType": ["text"]
    }
  ],
  "vectorizer": "text2vec-openai"
}
```

## Security Configuration

### API Authentication

Enable API key authentication:

```yaml
security:
  enable_auth: true
  api_key_header: X-API-Key
  jwt_secret: your_jwt_secret_here
  token_expiry: 3600
```

### SSL/TLS Configuration

For production, enable SSL:

```yaml
ssl:
  enabled: true
  cert_file: /path/to/cert.pem
  key_file: /path/to/key.pem
  ca_file: /path/to/ca.pem
```

## Performance Tuning

### Connection Pooling

```yaml
database:
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  pool_recycle: 3600
```

### Caching Strategy

```yaml
cache:
  type: redis
  ttl: 3600
  max_entries: 10000
  compression: true
```

## Logging Configuration

Configure logging levels and outputs:

```yaml
logging:
  level: INFO
  format: json
  output:
    - type: console
      level: INFO
    - type: file
      path: /var/log/noteparser-ai
      level: DEBUG
      max_size: 100MB
      max_files: 10
    - type: syslog
      host: localhost
      port: 514
      level: WARNING
```

## Monitoring Configuration

### Prometheus Metrics

```yaml
metrics:
  enabled: true
  port: 9090
  path: /metrics
  include:
    - requests_total
    - request_duration
    - active_connections
    - cache_hits
    - database_queries
```

### Health Checks

```yaml
health:
  endpoint: /health
  checks:
    - type: database
      timeout: 5s
    - type: redis
      timeout: 2s
    - type: disk_space
      threshold: 90
```

## Next Steps

- [Deployment Guide](deployment.md) - Deploy to production
- [API Reference](api-reference.md) - API documentation
- [Installation Guide](installation.md) - Installation instructions
- [Quick Start](quickstart.md) - Get started quickly
