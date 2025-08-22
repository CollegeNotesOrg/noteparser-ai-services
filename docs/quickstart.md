# Quick Start Guide

Get NoteParser AI Services running in minutes with this step-by-step guide.

## Prerequisites

Before you begin, ensure you have:

- **Docker Desktop** with 8GB+ memory allocated
- **Git** for cloning repositories
- **curl** for testing endpoints (optional)

!!! tip "System Requirements"
    - **Memory**: 8GB RAM minimum (16GB recommended)
    - **Storage**: 5GB free disk space
    - **OS**: Linux, macOS, or Windows with WSL2

## Step 1: Clone Repository

```bash
# Clone the AI services repository
git clone https://github.com/CollegeNotesOrg/noteparser-ai-services.git
cd noteparser-ai-services
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional for basic setup)
nano .env
```

!!! info "Environment Configuration"
    For basic testing, the default configuration works out of the box. For production use or AI features, you'll need to add API keys:
    
    ```bash
    OPENAI_API_KEY=your_openai_key_here
    ANTHROPIC_API_KEY=your_anthropic_key_here
    ```

## Step 3: Start Services

```bash
# Start all services in the background
docker-compose up -d

# Watch the startup process (optional)
docker-compose logs -f
```

This will start:

- **RagFlow** (RAG processing) on port 8010
- **DeepWiki** (knowledge organization) on port 8011  
- **PostgreSQL** database on port 5434
- **Redis** cache on port 6380
- **Qdrant** vector database on port 6333

## Step 4: Verify Services

Check that all services are healthy:

```bash
# Check service status
docker-compose ps

# Test RagFlow health
curl http://localhost:8010/health

# Test DeepWiki health  
curl http://localhost:8011/health
```

Expected responses:
```json
{"status": "healthy", "service": "ragflow"}
{"status": "healthy", "service": "deepwiki"}
```

## Step 5: Test Basic Functionality

### Test RagFlow (Document Processing)

```bash
# Index a sample document
curl -X POST http://localhost:8010/index \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    "metadata": {"title": "ML Introduction", "author": "AI Assistant"}
  }'
```

Expected response:
```json
{
  "status": "success",
  "chunks_indexed": 1,
  "document_ids": ["uuid-here"],
  "duration_seconds": 0.234
}
```

```bash
# Query the indexed document
curl -X POST http://localhost:8010/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 3
  }'
```

### Test DeepWiki (Knowledge Organization)

```bash
# Create a wiki article
curl -X POST http://localhost:8011/article \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Artificial Intelligence",
    "content": "# Artificial Intelligence\n\nArtificial Intelligence (AI) is the simulation of human intelligence in machines...",
    "metadata": {"tags": ["AI", "technology"], "author": "Student"}
  }'
```

Expected response:
```json
{
  "status": "success",
  "article_id": "artificial-intelligence",
  "title": "Artificial Intelligence",
  "version": 1,
  "links_created": 0
}
```

```bash
# Search wiki articles
curl -X POST http://localhost:8011/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "limit": 5
  }'
```

## Integration with NoteParser

Once your AI services are running, you can integrate them with the main NoteParser library:

### Install NoteParser with AI Support

```bash
pip install noteparser[ai]
```

### Use with Python

```python
import asyncio
from noteparser import NoteParser
from noteparser.integration.service_client import ServiceClientManager

async def main():
    # Initialize service manager
    manager = ServiceClientManager()
    
    # Check service health
    health = await manager.health_check_all()
    print(f"Services health: {health}")
    
    # Use RagFlow for document processing
    ragflow = manager.get_client("ragflow")
    
    # Index a document
    result = await ragflow.index_document(
        content="Your document content here",
        metadata={"title": "My Document", "course": "CS101"}
    )
    print(f"Indexed: {result}")
    
    # Query documents
    query_result = await ragflow.query(
        query="What are the main concepts?",
        k=5
    )
    print(f"Query result: {query_result}")
    
    # Use DeepWiki for knowledge organization
    deepwiki = manager.get_client("deepwiki")
    
    # Create wiki article
    article = await deepwiki.create_article(
        title="New Concept",
        content="Detailed explanation of the concept...",
        metadata={"tags": ["concept", "important"]}
    )
    print(f"Created article: {article}")
    
    # Search wiki
    search_result = await deepwiki.search("concept")
    print(f"Search results: {search_result}")
    
    # Clean up
    await manager.close_all()

# Run the example
asyncio.run(main())
```

## Common Commands

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a specific service
docker-compose restart ragflow

# View logs
docker-compose logs -f ragflow
docker-compose logs -f deepwiki

# Check resource usage
docker stats
```

### Data Management

```bash
# Backup database
docker exec ai-services-postgres pg_dump -U noteparser noteparser > backup.sql

# Clear Redis cache
docker exec ai-services-redis redis-cli FLUSHALL

# View vector database
curl http://localhost:6333/collections
```

### Development

```bash
# Run services locally (for development)
docker-compose up -d postgres redis qdrant
cd ragflow && python service.py
cd deepwiki && python service.py

# Run tests
docker-compose exec ragflow pytest tests/
docker-compose exec deepwiki pytest tests/
```

## Troubleshooting

### Services Won't Start

**Check Docker resources:**
```bash
docker system df
docker system prune  # Clean up unused resources
```

**Check port conflicts:**
```bash
# See what's using the ports
lsof -i :8010
lsof -i :8011

# Kill conflicting processes if needed
kill -9 <PID>
```

### Memory Issues

**Increase Docker memory:**
- Docker Desktop → Settings → Resources → Memory → 8GB+

**Reduce service resources:**
```yaml
# In docker-compose.yml
services:
  ragflow:
    deploy:
      resources:
        limits:
          memory: 1G
```

### Connection Issues

**Check service logs:**
```bash
docker-compose logs ragflow
docker-compose logs deepwiki
```

**Test connectivity:**
```bash
# Test from inside container
docker exec ragflow ping postgres
docker exec deepwiki ping redis
```

### Performance Issues

**Monitor resource usage:**
```bash
docker stats
htop
```

**Check service metrics:**
```bash
curl http://localhost:8010/stats
curl http://localhost:8011/metrics
```

## Next Steps

Now that your AI services are running:

1. **[Read the API Reference](api-reference.md)** - Explore all available endpoints
2. **[Configuration Guide](configuration.md)** - Configure services for production
3. **[Deploy to Production](deployment.md)** - Production deployment guides
4. **[API Reference](api-reference.md)** - Complete API documentation

!!! success "You're Ready!"
    Your NoteParser AI Services are now running and ready to process documents and organize knowledge. Start building intelligent applications with semantic search and AI-powered insights!

## Example Applications

### Academic Research Assistant

```python
# Process research papers
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]

for paper_path in papers:
    # Parse with NoteParser
    parser = NoteParser()
    result = parser.parse_to_markdown(paper_path)
    
    # Index in RagFlow
    await ragflow.index_document(
        content=result["content"],
        metadata={
            "title": result["metadata"]["title"],
            "authors": result["metadata"]["authors"],
            "year": result["metadata"]["year"]
        }
    )
    
    # Create wiki entry
    await deepwiki.create_article(
        title=result["metadata"]["title"],
        content=result["content"],
        metadata=result["metadata"]
    )

# Query across all papers
insights = await ragflow.query("What are the latest trends in AI?")
```

### Course Material Organizer

```python
# Process lecture slides and readings
course_materials = {
    "week1": ["intro_slides.pptx", "reading1.pdf"],
    "week2": ["ml_basics.pptx", "reading2.pdf"]
}

for week, materials in course_materials.items():
    for material_path in materials:
        result = parser.parse_to_markdown(material_path)
        
        # Index for search
        await ragflow.index_document(
            content=result["content"],
            metadata={
                "course": "CS229",
                "week": week,
                "type": "lecture" if "slides" in material_path else "reading"
            }
        )

# Enable student Q&A
student_question = "What is gradient descent?"
answer = await deepwiki.ask_assistant(student_question)
```

Happy building! 🚀