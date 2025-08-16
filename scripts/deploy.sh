#!/bin/bash

# NoteParser AI Services Deployment Script
# This script deploys and configures all AI services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_MEM" -lt 8 ]; then
        log_warning "Less than 8GB RAM detected. Services may run slowly."
    fi
    
    log_info "Requirements check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env file..."
        cat > .env << EOF
# AI Service Configuration
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_TOKEN=

# Service Ports
RAGFLOW_PORT=8010
DEEPWIKI_PORT=8011
QDRANT_PORT=6333
WEAVIATE_PORT=8080

# Database
POSTGRES_USER=noteparser
POSTGRES_PASSWORD=noteparser

# Redis
REDIS_PASSWORD=

# Monitoring
ENABLE_MONITORING=true
EOF
        log_warning "Please edit .env file and add your API keys"
    fi
    
    # Create necessary directories
    mkdir -p models notebooks data/{ragflow,deepwiki,qdrant,weaviate}
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    log_info "Environment setup completed"
}

build_services() {
    log_info "Building service images..."
    
    docker-compose build --parallel
    
    if [ $? -eq 0 ]; then
        log_info "Build completed successfully"
    else
        log_error "Build failed"
        exit 1
    fi
}

start_infrastructure() {
    log_info "Starting infrastructure services..."
    
    # Start databases first
    docker-compose up -d postgres redis
    sleep 10
    
    # Start vector databases
    docker-compose up -d qdrant weaviate
    sleep 5
    
    # Start MinIO for model storage
    docker-compose up -d minio
    
    log_info "Infrastructure services started"
}

start_ai_services() {
    log_info "Starting AI services..."
    
    # Start RagFlow
    docker-compose up -d ragflow
    sleep 5
    
    # Start DeepWiki
    docker-compose up -d deepwiki
    sleep 5
    
    # Start other services if configured
    if [ -d "$PROJECT_ROOT/dolphin" ]; then
        docker-compose up -d dolphin
    fi
    
    if [ -d "$PROJECT_ROOT/langextract" ]; then
        docker-compose up -d langextract
    fi
    
    log_info "AI services started"
}

start_monitoring() {
    log_info "Starting monitoring services..."
    
    # Start Jupyter for development
    docker-compose up -d jupyter
    
    # Start Nginx reverse proxy
    docker-compose up -d nginx
    
    log_info "Monitoring services started"
}

health_check() {
    log_info "Performing health checks..."
    
    sleep 10
    
    # Check RagFlow
    if curl -s http://localhost:8010/health > /dev/null; then
        log_info "✓ RagFlow is healthy"
    else
        log_warning "✗ RagFlow is not responding"
    fi
    
    # Check DeepWiki
    if curl -s http://localhost:8011/health > /dev/null; then
        log_info "✓ DeepWiki is healthy"
    else
        log_warning "✗ DeepWiki is not responding"
    fi
    
    # Check Qdrant
    if curl -s http://localhost:6333/health > /dev/null; then
        log_info "✓ Qdrant is healthy"
    else
        log_warning "✗ Qdrant is not responding"
    fi
    
    # Check Weaviate
    if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null; then
        log_info "✓ Weaviate is healthy"
    else
        log_warning "✗ Weaviate is not responding"
    fi
}

initialize_services() {
    log_info "Initializing services with sample data..."
    
    # Wait for services to be ready
    sleep 10
    
    # Initialize RagFlow
    curl -X POST http://localhost:8010/index \
        -H "Content-Type: application/json" \
        -d '{
            "content": "Welcome to NoteParser AI Services. This is a sample document for testing.",
            "metadata": {"title": "Welcome", "type": "sample"}
        }' > /dev/null 2>&1
    
    # Initialize DeepWiki
    curl -X POST http://localhost:8011/article \
        -H "Content-Type: application/json" \
        -d '{
            "title": "Getting Started",
            "content": "# Getting Started\n\nWelcome to DeepWiki, your AI-powered knowledge base.",
            "metadata": {"tags": ["intro", "guide"]}
        }' > /dev/null 2>&1
    
    log_info "Services initialized"
}

show_summary() {
    echo ""
    echo "=========================================="
    echo "   NoteParser AI Services Deployed!"
    echo "=========================================="
    echo ""
    echo "Service Endpoints:"
    echo "  • RagFlow API: http://localhost:8010"
    echo "  • DeepWiki API: http://localhost:8011"
    echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
    echo "  • Weaviate GraphQL: http://localhost:8080/v1/graphql"
    echo "  • Jupyter Lab: http://localhost:8888 (token: noteparser)"
    echo "  • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
    echo ""
    echo "Next Steps:"
    echo "  1. Edit .env file to add API keys"
    echo "  2. Run 'make logs' to view service logs"
    echo "  3. Run 'make health' to check service health"
    echo "  4. Visit service endpoints to start using"
    echo ""
    echo "Documentation: $PROJECT_ROOT/README.md"
    echo ""
}

# Main deployment flow
main() {
    log_info "Starting NoteParser AI Services deployment..."
    
    check_requirements
    setup_environment
    build_services
    start_infrastructure
    start_ai_services
    start_monitoring
    health_check
    initialize_services
    show_summary
    
    log_info "Deployment completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    "start")
        start_infrastructure
        start_ai_services
        start_monitoring
        ;;
    "stop")
        log_info "Stopping all services..."
        docker-compose down
        ;;
    "restart")
        log_info "Restarting all services..."
        docker-compose restart
        ;;
    "health")
        health_check
        ;;
    "clean")
        log_warning "Removing all services and volumes..."
        docker-compose down -v
        ;;
    *)
        main
        ;;
esac