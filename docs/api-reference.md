# API Reference

Complete API documentation for NoteParser AI Services.

## RagFlow Service API

**Base URL:** `http://localhost:8010`

### Authentication

Currently, authentication is optional. When enabled, include API key in headers:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8010/health
```

### Health Check

#### GET /health

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "ragflow"
}
```

### Document Indexing

#### POST /index

Index a document for semantic search and retrieval.

**Request:**
```json
{
  "content": "Document content to be indexed",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "course": "Course Code",
    "tags": ["tag1", "tag2"],
    "source": "file_path_or_url"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "chunks_indexed": 5,
  "document_ids": ["uuid1", "uuid2", "uuid3"],
  "duration_seconds": 1.234,
  "metadata": {
    "title": "Document Title",
    "author": "Author Name"
  }
}
```

**Error Response:**
```json
{
  "status": "error",
  "error": "Error message",
  "code": 400
}
```

### Document Querying

#### POST /query

Query indexed documents using natural language.

**Request:**
```json
{
  "query": "What is machine learning?",
  "k": 5,
  "filters": {
    "course": "CS101",
    "tags": ["ML"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "query": "What is machine learning?",
  "response": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "content": "Relevant text chunk from document",
      "metadata": {
        "title": "ML Basics",
        "author": "Professor"
      },
      "similarity_score": 0.95
    }
  ],
  "num_sources": 3,
  "duration_seconds": 0.87
}
```

### Insight Extraction

#### POST /insights

Extract key insights, summaries, and concepts from content.

**Request:**
```json
{
  "content": "Long document content...",
  "insight_types": ["summary", "key_points", "entities", "questions"]
}
```

**Response:**
```json
{
  "status": "success",
  "insights": {
    "summary": "Document summary...",
    "key_points": [
      "Key point 1",
      "Key point 2"
    ],
    "entities": {
      "people": ["Einstein", "Newton"],
      "concepts": ["relativity", "gravity"],
      "organizations": ["MIT", "Stanford"]
    },
    "questions": [
      "What is the main concept?",
      "How does this relate to previous work?"
    ]
  },
  "duration_seconds": 2.1
}
```

### Statistics

#### GET /stats

Get service statistics and metrics.

**Response:**
```json
{
  "total_documents": 1234,
  "total_chunks": 5678,
  "total_queries": 9876,
  "avg_query_time": 0.45,
  "storage_used_mb": 256,
  "last_indexed": "2025-01-15T10:30:00Z"
}
```

## DeepWiki Service API

**Base URL:** `http://localhost:8011`

### Health Check

#### GET /health

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "deepwiki"
}
```

### Article Management

#### POST /article

Create a new wiki article.

**Request:**
```json
{
  "title": "Neural Networks",
  "content": "# Neural Networks\n\nNeural networks are computational models...",
  "metadata": {
    "tags": ["AI", "ML", "deep-learning"],
    "author": "Student Name",
    "course": "CS229",
    "difficulty": "intermediate"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "article_id": "neural-networks",
  "title": "Neural Networks",
  "version": 1,
  "links_created": 3,
  "concepts_extracted": ["neural", "networks", "computation"],
  "duration_seconds": 2.15
}
```

#### GET /article/{article_id}

Retrieve a specific article.

**Response:**
```json
{
  "id": "neural-networks",
  "title": "Neural Networks",
  "content": "# Neural Networks\n\n...",
  "metadata": {
    "tags": ["AI", "ML"],
    "author": "Student"
  },
  "links": [
    {
      "target": "machine-learning",
      "relevance": 0.95,
      "type": "concept"
    }
  ],
  "version": 1,
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z"
}
```

#### PUT /article/{article_id}

Update an existing article.

**Request:**
```json
{
  "content": "Updated content...",
  "metadata": {
    "tags": ["AI", "ML", "updated"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "article_id": "neural-networks",
  "version": 2,
  "changes": {
    "content_changed": true,
    "metadata_changed": true,
    "links_updated": 2
  }
}
```

### Search

#### POST /search

Search wiki articles by content, title, or concepts.

**Request:**
```json
{
  "query": "machine learning algorithms",
  "limit": 10,
  "search_type": "content",
  "filters": {
    "tags": ["ML"],
    "author": "specific-author"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "query": "machine learning algorithms",
  "results": [
    {
      "article_id": "ml-algorithms",
      "title": "Machine Learning Algorithms",
      "score": 0.92,
      "snippet": "Machine learning algorithms are computational procedures..."
    }
  ],
  "result_count": 5,
  "ai_summary": "The search results cover various machine learning algorithms including supervised, unsupervised, and reinforcement learning approaches...",
  "duration_seconds": 1.45
}
```

### AI Assistant

#### POST /ask

Ask the AI assistant questions about wiki content.

**Request:**
```json
{
  "question": "What is the difference between supervised and unsupervised learning?",
  "context_articles": ["machine-learning", "supervised-learning"],
  "use_all_context": false
}
```

**Response:**
```json
{
  "status": "success",
  "question": "What is the difference between supervised and unsupervised learning?",
  "answer": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs, while unsupervised learning finds patterns in data without labeled examples...",
  "related_articles": [
    {
      "article_id": "supervised-learning",
      "title": "Supervised Learning",
      "relevance": 0.95
    }
  ],
  "context_used": true,
  "confidence": 0.88,
  "duration_seconds": 1.8
}
```

### Knowledge Graph

#### GET /graph

Get the knowledge graph for visualization.

**Query Parameters:**
- `article_id` (optional): Focus on specific article
- `depth` (default: 2): Graph traversal depth
- `min_relevance` (default: 0.5): Minimum link relevance

**Response:**
```json
{
  "nodes": [
    {
      "id": "neural-networks",
      "title": "Neural Networks",
      "type": "article",
      "concepts": ["neural", "networks"],
      "centrality": 0.85
    }
  ],
  "edges": [
    {
      "source": "neural-networks",
      "target": "machine-learning",
      "relevance": 0.92,
      "type": "concept",
      "concepts": ["learning", "algorithms"]
    }
  ],
  "stats": {
    "total_nodes": 50,
    "total_edges": 120,
    "clusters": 5
  }
}
```

#### GET /similar/{article_id}

Find articles similar to a given article.

**Query Parameters:**
- `limit` (default: 5): Number of similar articles to return

**Response:**
```json
{
  "article_id": "neural-networks",
  "similar_articles": [
    {
      "article_id": "deep-learning",
      "title": "Deep Learning",
      "similarity_score": 0.88,
      "common_concepts": ["neural", "learning", "networks"]
    }
  ],
  "algorithm": "concept_similarity",
  "duration_seconds": 0.23
}
```

### Export/Import

#### GET /export

Export wiki data in various formats.

**Query Parameters:**
- `format`: json, xml, markdown
- `articles`: comma-separated list of article IDs (optional)

**Response:**
```json
{
  "format": "json",
  "data": {
    "articles": [...],
    "links": [...],
    "metadata": {...}
  },
  "exported_at": "2025-01-15T10:30:00Z"
}
```

#### POST /import

Import wiki data.

**Request:**
```json
{
  "format": "json",
  "data": {
    "articles": [...],
    "links": [...]
  },
  "merge_strategy": "update_existing"
}
```

### Metrics

#### GET /metrics

Get Prometheus metrics (when monitoring is enabled).

**Response:** Prometheus format metrics

```
# HELP deepwiki_articles_total Total number of articles
# TYPE deepwiki_articles_total counter
deepwiki_articles_total 150

# HELP deepwiki_queries_duration_seconds Query duration
# TYPE deepwiki_queries_duration_seconds histogram
deepwiki_queries_duration_seconds_bucket{le="0.1"} 45
```

## Error Codes

### Common HTTP Status Codes

- `200` - Success
- `400` - Bad Request (invalid JSON, missing fields)
- `401` - Unauthorized (invalid API key)
- `404` - Not Found (article not found)
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error
- `503` - Service Unavailable (service overloaded)

### Error Response Format

```json
{
  "status": "error",
  "error": "Detailed error message",
  "code": 400,
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "uuid"
}
```

## Rate Limiting

Both services implement rate limiting:

- **Development**: 100 requests/minute per IP
- **Production**: 1000 requests/minute per API key

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642253400
```

## Pagination

For endpoints returning lists (search results, etc.):

**Request:**
```json
{
  "query": "search term",
  "limit": 20,
  "offset": 40
}
```

**Response:**
```json
{
  "results": [...],
  "pagination": {
    "total": 150,
    "limit": 20,
    "offset": 40,
    "has_next": true,
    "has_prev": true
  }
}
```

## WebSocket API (DeepWiki)

For real-time collaboration features:

**Connection:** `ws://localhost:8011/ws`

**Events:**
```javascript
// Connect
socket.on('connected', (data) => {
  console.log('Connected to DeepWiki');
});

// Article updates
socket.emit('article_update', {
  article_id: 'neural-networks',
  content: 'Updated content...'
});

socket.on('article_updated', (data) => {
  console.log('Article updated:', data);
});
```

## SDK Examples

### Python

```python
import asyncio
from noteparser.integration.service_client import ServiceClientManager

async def main():
    # Initialize clients
    manager = ServiceClientManager()
    
    # Check health
    health = await manager.health_check_all()
    print(f"Services health: {health}")
    
    # Use RagFlow
    ragflow = manager.get_client("ragflow")
    result = await ragflow.index_document(
        content="Machine learning is...",
        metadata={"title": "ML Intro"}
    )
    
    # Use DeepWiki
    deepwiki = manager.get_client("deepwiki")
    article = await deepwiki.create_article(
        title="Neural Networks",
        content="Neural networks are...",
        metadata={"tags": ["AI"]}
    )
    
    await manager.close_all()

asyncio.run(main())
```

### JavaScript

```javascript
const axios = require('axios');

// RagFlow example
async function indexDocument() {
  const response = await axios.post('http://localhost:8010/index', {
    content: 'Document content...',
    metadata: { title: 'Document Title' }
  });
  return response.data;
}

// DeepWiki example
async function createArticle() {
  const response = await axios.post('http://localhost:8011/article', {
    title: 'New Article',
    content: '# Content',
    metadata: { tags: ['tag1'] }
  });
  return response.data;
}
```

### cURL Examples

```bash
# Index document in RagFlow
curl -X POST http://localhost:8010/index \
  -H "Content-Type: application/json" \
  -d '{"content": "AI content", "metadata": {"title": "AI Basics"}}'

# Query documents
curl -X POST http://localhost:8010/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AI?", "k": 3}'

# Create wiki article
curl -X POST http://localhost:8011/article \
  -H "Content-Type: application/json" \
  -d '{"title": "AI Overview", "content": "AI is...", "metadata": {"tags": ["AI"]}}'

# Search wiki
curl -X POST http://localhost:8011/search \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "limit": 5}'
```