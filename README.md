# NoteParser AI Services 🤖

**Advanced AI backend services for intelligent document processing and knowledge organization**

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![AI Services](https://img.shields.io/badge/AI-powered-orange.svg)](https://github.com/CollegeNotesOrg/noteparser-ai-services)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the AI microservices backend for [NoteParser](https://github.com/CollegeNotesOrg/noteparser), enabling intelligent document processing, semantic search, and knowledge organization for academic materials.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- 8GB+ RAM (16GB recommended for all services)
- Optional: NVIDIA GPU for accelerated inference

### Setup

1. **Clone this repository:**
```bash
git clone https://github.com/CollegeNotesOrg/noteparser-ai-services.git
cd noteparser-ai-services
```

2. **Set environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start services:**
```bash
docker-compose up -d
```

4. **Verify health:**
```bash
curl http://localhost:8010/health  # RagFlow
curl http://localhost:8011/health  # DeepWiki
```

## 📦 Services

### RagFlow (Port 8010)
RAG-based document indexing and Q&A service.

**Endpoints:**
- `POST /index` - Index a document
- `POST /query` - Query indexed documents
- `POST /insights` - Extract insights from content
- `GET /stats` - Get service statistics

**Example:**
```python
import requests

# Index document
response = requests.post('http://localhost:8010/index', json={
    'content': 'Your document content here',
    'metadata': {'title': 'Document Title'}
})

# Query
response = requests.post('http://localhost:8010/query', json={
    'query': 'What is machine learning?',
    'k': 5
})
```

### DeepWiki (Port 8011)
AI-powered wiki system with knowledge organization.

**Endpoints:**
- `POST /article` - Create wiki article
- `PUT /article/{id}` - Update article
- `GET /article/{id}` - Get article
- `POST /search` - Search wiki
- `POST /ask` - Ask AI assistant
- `GET /graph` - Get link graph

**Example:**
```python
# Create article
response = requests.post('http://localhost:8011/article', json={
    'title': 'Introduction to AI',
    'content': 'AI content here',
    'metadata': {'tags': ['AI', 'intro']}
})

# Ask assistant
response = requests.post('http://localhost:8011/ask', json={
    'question': 'What is deep learning?'
})
```

## 🗄️ Vector Databases

### Qdrant (Port 6333)
High-performance vector similarity search.

**Web UI:** http://localhost:6333/dashboard

### Weaviate (Port 8080)
GraphQL-based vector search engine.

**GraphQL Console:** http://localhost:8080/v1/graphql

## 🧠 Model Management

### Triton Inference Server (Ports 8001-8003)
Serves deep learning models for inference.

**Model Repository:** `./models/`

### MinIO (Port 9000)
S3-compatible object storage for models.

**Console:** http://localhost:9001
**Credentials:** minioadmin/minioadmin

## 📊 Development Tools

### Jupyter Lab (Port 8888)
Interactive notebook environment.

**Access:** http://localhost:8888
**Token:** noteparser

## 🔧 Configuration

### Environment Variables

Create `.env` file:
```env
# API Keys
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
HUGGINGFACE_TOKEN=your-token-here

# Service Configuration
RAGFLOW_CHUNK_SIZE=1000
DEEPWIKI_AI_ENHANCEMENT=true

# Database
POSTGRES_PASSWORD=secure-password
```

### Service Configuration

Each service has its own `config.yaml`:

**ragflow/config.yaml:**
```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
vector_db: faiss
chunk_size: 1000
```

**deepwiki/config.yaml:**
```yaml
ai_model: gpt-3.5-turbo
wiki_features:
  auto_link: true
  versioning: true
```

## 🚀 Deployment

### Development
```bash
docker-compose up
```

### Production
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### With GPU Support
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## 📈 Monitoring

### Metrics
Each service exposes Prometheus metrics at `/metrics`.

### Logs
```bash
docker-compose logs -f ragflow
docker-compose logs -f deepwiki
```

### Health Checks
```bash
./scripts/health-check.sh
```

## 🧪 Testing

### Unit Tests
```bash
docker-compose run --rm ragflow pytest tests/
docker-compose run --rm deepwiki pytest tests/
```

### Integration Tests
```bash
./scripts/integration-test.sh
```

### Load Testing
```bash
docker run --rm -v $(pwd)/tests:/tests \
  grafana/k6 run /tests/load-test.js
```

## 📝 API Documentation

Interactive API docs available at:
- RagFlow: http://localhost:8010/docs
- DeepWiki: http://localhost:8011/docs

## 🔐 Security

### Authentication
Set `ENABLE_AUTH=true` and configure JWT tokens.

### HTTPS
Use the included nginx configuration with SSL certificates.

### Network Isolation
Services communicate on internal Docker network.

## 🐛 Troubleshooting

### Service won't start
```bash
# Check logs
docker-compose logs [service-name]

# Restart service
docker-compose restart [service-name]
```

### Out of memory
Increase Docker memory allocation or reduce service replicas.

### GPU not detected
Ensure NVIDIA Docker runtime is installed:
```bash
nvidia-docker run --rm nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 📚 Adding New Services

1. Create service directory:
```bash
mkdir -p newservice
```

2. Add Dockerfile and service.py

3. Update docker-compose.yml

4. Add client in noteparser integration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## 📄 License

MIT License - See LICENSE file

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/CollegeNotesOrg/noteparser-ai-services/issues)
- Email: suryanshss1011@gmail.com