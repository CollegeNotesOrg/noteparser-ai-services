# Installation Guide

Comprehensive installation guide for NoteParser AI Services across different environments and platforms.

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores, 2.0 GHz
- **Memory**: 4GB RAM  
- **Storage**: 10GB free space
- **Network**: Internet connection for AI models

### Recommended Requirements

- **CPU**: 4+ cores, 2.5+ GHz
- **Memory**: 8GB+ RAM (16GB for production)
- **Storage**: 50GB+ SSD storage
- **Network**: High-speed internet for model downloads

### Supported Platforms

- **Linux**: Ubuntu 20.04+, CentOS 8+, Debian 11+
- **macOS**: 10.15+ (Intel & Apple Silicon)
- **Windows**: Windows 10+ with WSL2

## Docker Installation

### Install Docker Desktop

=== "macOS"

    ```bash
    # Download Docker Desktop for Mac
    # Visit: https://www.docker.com/products/docker-desktop
    
    # Or install via Homebrew
    brew install --cask docker
    
    # Start Docker Desktop
    open -a Docker
    ```

=== "Windows"

    ```bash
    # Enable WSL2
    wsl --install
    
    # Download Docker Desktop for Windows
    # Visit: https://www.docker.com/products/docker-desktop
    
    # Restart system after installation
    ```

=== "Linux (Ubuntu)"

    ```bash
    # Update system
    sudo apt update && sudo apt upgrade -y
    
    # Install dependencies
    sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    newgrp docker
    
    # Test installation
    docker run hello-world
    ```

### Configure Docker Resources

**Increase memory allocation (important!):**

=== "Docker Desktop"

    1. Open Docker Desktop
    2. Go to Settings → Resources → Memory
    3. Increase to 8GB+ (16GB recommended)
    4. Click "Apply & Restart"

=== "Linux"

    ```bash
    # Check available memory
    free -h
    
    # No specific configuration needed on Linux
    # Docker uses host resources directly
    ```

## Quick Installation

### Option 1: One-Command Install

```bash
curl -fsSL https://raw.githubusercontent.com/CollegeNotesOrg/noteparser-ai-services/main/install.sh | bash
```

This script will:
- Clone the repository
- Set up environment files
- Start all services
- Verify installation

### Option 2: Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/CollegeNotesOrg/noteparser-ai-services.git
cd noteparser-ai-services

# 2. Copy environment template
cp .env.example .env

# 3. Start services
docker-compose up -d

# 4. Verify installation
./scripts/health-check.sh
```

## Detailed Installation Steps

### Step 1: Clone Repository

```bash
# Clone with HTTPS
git clone https://github.com/CollegeNotesOrg/noteparser-ai-services.git

# Or clone with SSH (if you have SSH key configured)
git clone git@github.com:CollegeNotesOrg/noteparser-ai-services.git

# Navigate to directory
cd noteparser-ai-services
```

### Step 2: Environment Configuration

Create and configure your environment file:

```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env  # or vim, code, etc.
```

**Basic configuration (.env):**
```bash
# AI Model APIs (optional for basic testing)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_TOKEN=

# Service Ports
RAGFLOW_PORT=8010
DEEPWIKI_PORT=8011
QDRANT_PORT=6333
WEAVIATE_PORT=8080

# Database Configuration
POSTGRES_USER=noteparser
POSTGRES_PASSWORD=noteparser
POSTGRES_DB=noteparser

# Redis Configuration
REDIS_PASSWORD=

# Environment
DEBUG=false
LOAD_SAMPLE_DATA=true
ENABLE_MONITORING=true
```

### Step 3: Start Services

```bash
# Start all services
docker-compose up -d

# View startup logs (optional)
docker-compose logs -f

# Check status
docker-compose ps
```

Expected output:
```
NAME                   IMAGE                             COMMAND                  SERVICE    CREATED        STATUS                    PORTS
ai-services-postgres   pgvector/pgvector:pg16            "docker-entrypoint.s…"   postgres   2 minutes ago  Up 2 minutes              0.0.0.0:5434->5432/tcp
ai-services-redis      redis:7-alpine                    "docker-entrypoint.s…"   redis      2 minutes ago  Up 2 minutes              0.0.0.0:6380->6379/tcp
deepwiki-service       noteparser-ai-services-deepwiki   "python service.py"      deepwiki   2 minutes ago  Up 2 minutes (healthy)    0.0.0.0:8011->8011/tcp
ragflow-service        noteparser-ai-services-ragflow    "python service.py"      ragflow    2 minutes ago  Up 2 minutes (healthy)    0.0.0.0:8010->8010/tcp
```

### Step 4: Verify Installation

```bash
# Check service health
curl http://localhost:8010/health
curl http://localhost:8011/health

# Expected responses:
# {"status": "healthy", "service": "ragflow"}
# {"status": "healthy", "service": "deepwiki"}

# Run health check script
./scripts/health-check.sh
```

## Installation Options

### Development Installation

For active development with hot reload:

```bash
# Start infrastructure only
docker-compose up -d postgres redis qdrant weaviate

# Install dependencies locally
cd ragflow
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

cd ../deepwiki
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run services locally
cd ragflow && python service.py &
cd deepwiki && python service.py &
```

### Production Installation

For production deployment:

```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or use specific production configuration
cp .env.production .env
docker-compose up -d
```

### Minimal Installation

For testing with minimal resources:

```bash
# Use minimal configuration
docker-compose -f docker-compose.minimal.yml up -d

# This starts only:
# - PostgreSQL
# - Redis  
# - RagFlow
# - DeepWiki
```

## Platform-Specific Instructions

### macOS Installation

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker Desktop
brew install --cask docker

# Install Git (if needed)
brew install git

# Clone and start services
git clone https://github.com/CollegeNotesOrg/noteparser-ai-services.git
cd noteparser-ai-services
docker-compose up -d
```

### Windows (WSL2) Installation

```powershell
# Enable WSL2
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart computer, then set WSL2 as default
wsl --set-default-version 2

# Install Ubuntu
wsl --install -d Ubuntu

# Open Ubuntu terminal and follow Linux instructions
```

### Linux Server Installation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Reboot or re-login for group changes
sudo reboot

# Clone and start services
git clone https://github.com/CollegeNotesOrg/noteparser-ai-services.git
cd noteparser-ai-services
docker-compose up -d
```

## Custom Installation

### Using Custom Ports

Edit `docker-compose.yml` to change ports:

```yaml
services:
  ragflow:
    ports:
      - "9010:8010"  # Custom port 9010 instead of 8010
  
  deepwiki:
    ports:
      - "9011:8011"  # Custom port 9011 instead of 8011
```

### Using External Database

To use an external PostgreSQL database:

```yaml
# In docker-compose.yml, remove the postgres service
# and update environment variables:

services:
  ragflow:
    environment:
      - DATABASE_URL=postgresql://user:pass@external-host:5432/dbname
  
  deepwiki:
    environment:
      - DATABASE_URL=postgresql://user:pass@external-host:5432/dbname
```

### Using Custom Models

To use custom AI models:

```bash
# Create models directory
mkdir -p ./models/custom

# Add your model files
cp /path/to/your/model/* ./models/custom/

# Update service configuration
# Edit ragflow/config.yaml or deepwiki/config.yaml
```

## Troubleshooting Installation

### Common Issues

**Port already in use:**
```bash
# Check what's using the port
lsof -i :8010
lsof -i :8011

# Kill the process or change ports in docker-compose.yml
```

**Docker daemon not running:**
```bash
# On Linux
sudo systemctl start docker
sudo systemctl enable docker

# On macOS/Windows
# Start Docker Desktop application
```

**Permission denied:**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended)
sudo docker-compose up -d
```

**Out of disk space:**
```bash
# Clean up Docker
docker system prune -af
docker volume prune

# Check available space
df -h
```

**Memory issues:**
```bash
# Increase Docker memory (Docker Desktop)
# Settings → Resources → Memory → 8GB+

# Check system memory
free -h  # Linux
vm_stat  # macOS
```

### Diagnostic Commands

```bash
# Check Docker installation
docker --version
docker-compose --version

# Check container status
docker-compose ps
docker-compose logs

# Check resource usage
docker stats

# Check network connectivity
docker exec ragflow-service ping postgres
docker exec deepwiki-service ping redis

# Test database connection
docker exec ai-services-postgres pg_isready -U noteparser

# Test Redis connection
docker exec ai-services-redis redis-cli ping
```

### Log Analysis

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs ragflow
docker-compose logs deepwiki

# Follow logs in real-time
docker-compose logs -f

# View last 50 lines
docker-compose logs --tail=50

# Filter by time
docker-compose logs --since="1h"
```

## Post-Installation Setup

### API Keys Configuration

If you plan to use AI features, configure API keys:

```bash
# Edit .env file
nano .env

# Add your API keys
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Restart services to apply changes
docker-compose restart
```

### Load Sample Data

```bash
# Load sample documents and wiki articles
docker-compose exec ragflow python scripts/load_sample_data.py
docker-compose exec deepwiki python scripts/load_sample_data.py
```

### Create Admin User

```bash
# Create admin user for DeepWiki
docker-compose exec deepwiki python scripts/create_admin.py --username admin --email admin@example.com
```

### Test Integration

```bash
# Install NoteParser client
pip install noteparser[ai]

# Test integration
python scripts/test_integration.py
```

## Next Steps

After successful installation:

1. **[Configure Services](configuration.md)** - Customize service settings
2. **[Set up Monitoring](operations/monitoring.md)** - Add production monitoring
3. **[API Testing](api-reference.md)** - Test the APIs
4. **[Deploy to Production](deployment.md)** - Production deployment

## Uninstallation

To completely remove NoteParser AI Services:

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: This deletes all data!)
docker-compose down -v

# Remove images
docker rmi $(docker images "noteparser*" -q)

# Remove repository
cd .. && rm -rf noteparser-ai-services
```

## Getting Help

If you encounter issues during installation:

- 📚 Check the [Troubleshooting Guide](operations/troubleshooting.md)
- 💬 Ask questions in [GitHub Discussions](https://github.com/CollegeNotesOrg/noteparser/discussions)
- 🐛 Report bugs in [GitHub Issues](https://github.com/CollegeNotesOrg/noteparser-ai-services/issues)
- 📧 Email support: suryanshss1011@gmail.com