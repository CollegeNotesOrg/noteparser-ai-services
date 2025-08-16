.PHONY: help build up down logs health test clean deploy

# Variables
COMPOSE_FILE = docker-compose.yml
PROJECT_NAME = noteparser-ai-services

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all service images
	docker-compose build

build-service: ## Build specific service (usage: make build-service SERVICE=ragflow)
	docker-compose build $(SERVICE)

up: ## Start all services
	docker-compose up -d

up-service: ## Start specific service (usage: make up-service SERVICE=ragflow)
	docker-compose up -d $(SERVICE)

down: ## Stop all services
	docker-compose down

down-clean: ## Stop all services and remove volumes
	docker-compose down -v

logs: ## View logs from all services
	docker-compose logs -f

logs-service: ## View logs from specific service (usage: make logs-service SERVICE=ragflow)
	docker-compose logs -f $(SERVICE)

health: ## Check health of all services
	@echo "Checking service health..."
	@curl -s http://localhost:8010/health | jq '.' || echo "RagFlow: DOWN"
	@curl -s http://localhost:8011/health | jq '.' || echo "DeepWiki: DOWN"
	@curl -s http://localhost:6333/health | jq '.' || echo "Qdrant: DOWN"
	@curl -s http://localhost:8080/v1/.well-known/ready || echo "Weaviate: DOWN"

restart: ## Restart all services
	docker-compose restart

restart-service: ## Restart specific service (usage: make restart-service SERVICE=ragflow)
	docker-compose restart $(SERVICE)

ps: ## Show running services
	docker-compose ps

stats: ## Show container statistics
	docker stats --no-stream

test: ## Run tests for all services
	@echo "Running tests..."
	docker-compose run --rm ragflow pytest tests/ -v
	docker-compose run --rm deepwiki pytest tests/ -v

test-service: ## Test specific service (usage: make test-service SERVICE=ragflow)
	docker-compose run --rm $(SERVICE) pytest tests/ -v

clean: ## Clean up temporary files and unused images
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

deploy: ## Deploy to production
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

metrics: ## View service metrics
	@echo "RagFlow metrics:"
	@curl -s http://localhost:8010/metrics | head -20
	@echo "\nDeepWiki metrics:"
	@curl -s http://localhost:8011/metrics | head -20

backup: ## Backup service data
	@echo "Backing up service data..."
	@mkdir -p backups
	docker-compose exec postgres pg_dump -U noteparser ragflow > backups/ragflow_$$(date +%Y%m%d_%H%M%S).sql
	docker-compose exec postgres pg_dump -U noteparser deepwiki > backups/deepwiki_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup completed!"

restore: ## Restore from backup (usage: make restore BACKUP=backups/ragflow_20240101.sql SERVICE=ragflow)
	@echo "Restoring $(SERVICE) from $(BACKUP)..."
	docker-compose exec -T postgres psql -U noteparser $(SERVICE) < $(BACKUP)
	@echo "Restore completed!"

shell: ## Open shell in service container (usage: make shell SERVICE=ragflow)
	docker-compose exec $(SERVICE) /bin/bash

jupyter: ## Open Jupyter Lab
	@echo "Opening Jupyter Lab..."
	@echo "URL: http://localhost:8888"
	@echo "Token: noteparser"

minio: ## Open MinIO console
	@echo "Opening MinIO console..."
	@echo "URL: http://localhost:9001"
	@echo "Username: minioadmin"
	@echo "Password: minioadmin"

init: ## Initialize services with sample data
	@echo "Initializing services..."
	./scripts/init-services.sh

benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	./scripts/benchmark.sh

monitor: ## Open monitoring dashboards
	@echo "Service endpoints:"
	@echo "RagFlow: http://localhost:8010"
	@echo "DeepWiki: http://localhost:8011"
	@echo "Qdrant: http://localhost:6333/dashboard"
	@echo "Weaviate: http://localhost:8080"
	@echo "Jupyter: http://localhost:8888 (token: noteparser)"
	@echo "MinIO: http://localhost:9001 (minioadmin/minioadmin)"

dev: build up logs ## Development workflow

prod: ## Production deployment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

gpu: ## Start with GPU support
	docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d