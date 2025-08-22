# Deployment Guide

Comprehensive guide for deploying NoteParser AI Services in different environments.

## Local Development

**Prerequisites:**
- Docker Desktop with 8GB+ memory allocated
- Git
- Text editor or IDE

**Steps:**

1. **Clone repository:**
   ```bash
   git clone https://github.com/CollegeNotesOrg/noteparser-ai-services.git
   cd noteparser-ai-services
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify deployment:**
   ```bash
   ./scripts/health-check.sh
   ```

### Development with Hot Reload

For active development with code changes:

```bash
# Start infrastructure only
docker-compose up -d postgres redis qdrant

# Run services locally
cd ragflow && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python service.py

# In another terminal
cd deepwiki && python -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
python service.py
```

## Production Deployment

### Docker Swarm

**Initialize Swarm:**
```bash
docker swarm init
```

**Deploy Stack:**
```bash
# Use production compose file
docker stack deploy -c docker-compose.prod.yml noteparser-ai

# Check deployment
docker stack services noteparser-ai
docker stack ps noteparser-ai
```

**Scale Services:**
```bash
docker service scale noteparser-ai_ragflow=3
docker service scale noteparser-ai_deepwiki=3
```

### Kubernetes

**Generate Manifests:**
```bash
# Install kompose
curl -L https://github.com/kubernetes/kompose/releases/download/v1.28.0/kompose-linux-amd64 -o kompose
chmod +x kompose && sudo mv kompose /usr/local/bin

# Convert docker-compose to k8s
kompose convert -f docker-compose.yml
```

**Deploy to Cluster:**
```bash
# Apply manifests
kubectl apply -f .

# Check deployment
kubectl get pods
kubectl get services
```

**Production Kubernetes Configuration:**

Create `k8s-production.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ragflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ragflow
  template:
    metadata:
      labels:
        app: ragflow
    spec:
      containers:
      - name: ragflow
        image: noteparser/ragflow:latest
        ports:
        - containerPort: 8010
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8010
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8010
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ragflow-service
spec:
  selector:
    app: ragflow
  ports:
  - port: 8010
    targetPort: 8010
  type: LoadBalancer
```

## Cloud Deployment

### AWS ECS

**Using AWS Copilot:**

```bash
# Install Copilot
curl -Lo copilot https://github.com/aws/copilot-cli/releases/latest/download/copilot-linux
chmod +x copilot && sudo mv copilot /usr/local/bin

# Initialize application
copilot app init noteparser-ai

# Initialize environment
copilot env init --name production

# Initialize services
copilot svc init --name ragflow --svc-type "Backend Service"
copilot svc init --name deepwiki --svc-type "Backend Service"

# Deploy
copilot svc deploy --name ragflow --env production
copilot svc deploy --name deepwiki --env production
```

**Manual ECS Deployment:**

1. **Create task definition:**
   ```json
   {
     "family": "ragflow",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "containerDefinitions": [
       {
         "name": "ragflow",
         "image": "your-registry/ragflow:latest",
         "portMappings": [
           {
             "containerPort": 8010,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "DATABASE_URL",
             "value": "postgresql://..."
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/ragflow",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

2. **Create service:**
   ```bash
   aws ecs create-service \
     --cluster noteparser-ai \
     --service-name ragflow \
     --task-definition ragflow:1 \
     --desired-count 2 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
   ```

### Google Cloud Run

**Deploy RagFlow:**
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/ragflow ./ragflow

# Deploy to Cloud Run
gcloud run deploy ragflow \
  --image gcr.io/PROJECT_ID/ragflow \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --set-env-vars DATABASE_URL="postgresql://..."
```

**Deploy DeepWiki:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/deepwiki ./deepwiki
gcloud run deploy deepwiki \
  --image gcr.io/PROJECT_ID/deepwiki \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Azure Container Instances

**Deploy with Azure CLI:**
```bash
# Create resource group
az group create --name noteparser-ai --location eastus

# Deploy RagFlow
az container create \
  --resource-group noteparser-ai \
  --name ragflow \
  --image your-registry/ragflow:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8010 \
  --environment-variables DATABASE_URL="postgresql://..."

# Deploy DeepWiki
az container create \
  --resource-group noteparser-ai \
  --name deepwiki \
  --image your-registry/deepwiki:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8011
```

## Database Deployment

### Managed Database Services

**PostgreSQL:**

- **AWS RDS:**
  ```bash
  aws rds create-db-instance \
    --db-instance-identifier noteparser-postgres \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --master-username noteparser \
    --master-user-password your-password \
    --allocated-storage 100 \
    --vpc-security-group-ids sg-xxx
  ```

- **Google Cloud SQL:**
  ```bash
  gcloud sql instances create noteparser-postgres \
    --database-version=POSTGRES_13 \
    --tier=db-f1-micro \
    --region=us-central1
  ```

- **Azure Database:**
  ```bash
  az postgres server create \
    --resource-group noteparser-ai \
    --name noteparser-postgres \
    --location eastus \
    --admin-user noteparser \
    --admin-password your-password \
    --sku-name GP_Gen5_2
  ```

**Redis:**

- **AWS ElastiCache:**
  ```bash
  aws elasticache create-cache-cluster \
    --cache-cluster-id noteparser-redis \
    --cache-node-type cache.t3.micro \
    --engine redis \
    --num-cache-nodes 1
  ```

- **Google Memorystore:**
  ```bash
  gcloud redis instances create noteparser-redis \
    --size=1 \
    --region=us-central1
  ```

## Environment Configuration

### Environment Variables

**Required Variables:**
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0

# Service Configuration
RAGFLOW_PORT=8010
DEEPWIKI_PORT=8011
DEBUG=false
```

**Optional Variables:**
```bash
# Authentication
ENABLE_AUTH=true
JWT_SECRET=your_jwt_secret
API_KEY=your_api_key

# Performance
WORKERS=4
BATCH_SIZE=32
CACHE_TTL=3600

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### Configuration Files

**production.env:**
```bash
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4
ENABLE_AUTH=true
RATE_LIMIT_ENABLED=true
CACHE_TTL=3600
```

**staging.env:**
```bash
DEBUG=true
LOG_LEVEL=DEBUG
WORKERS=2
ENABLE_AUTH=false
RATE_LIMIT_ENABLED=false
LOAD_SAMPLE_DATA=true
```

## Security Configuration

### SSL/TLS Setup

**Generate Certificates:**
```bash
# Self-signed (development only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Let's Encrypt (production)
certbot certonly --standalone -d your-domain.com
```

**Nginx Configuration:**
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location /ragflow/ {
        proxy_pass http://ragflow:8010/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /deepwiki/ {
        proxy_pass http://deepwiki:8011/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Authentication Setup

**JWT Configuration:**
```yaml
# config/auth.yaml
jwt:
  secret: "your-secret-key"
  expiration: 3600
  algorithm: "HS256"

api_keys:
  - name: "production"
    key: "prod-key-xxxx"
    permissions: ["read", "write"]
  - name: "readonly"
    key: "readonly-key-xxxx"
    permissions: ["read"]
```

### Network Security

**Docker Network Isolation:**
```yaml
# docker-compose.prod.yml
services:
  ragflow:
    networks:
      - internal
      - external
  
  postgres:
    networks:
      - internal  # No external access

networks:
  internal:
    driver: bridge
    internal: true
  external:
    driver: bridge
```

**Firewall Rules:**
```bash
# Only allow specific ports
ufw allow 443/tcp  # HTTPS
ufw allow 22/tcp   # SSH
ufw deny 8010/tcp  # Block direct service access
ufw deny 8011/tcp
ufw enable
```

## Monitoring and Logging

### Prometheus Monitoring

**prometheus.yml:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ragflow'
    static_configs:
      - targets: ['ragflow:8010']
    metrics_path: '/metrics'
    
  - job_name: 'deepwiki'
    static_configs:
      - targets: ['deepwiki:8011']
    metrics_path: '/metrics'
```

### Centralized Logging

**Fluentd Configuration:**
```yaml
# fluentd.conf
<source>
  @type docker
  container_names ["ragflow", "deepwiki"]
  tag docker.*
</source>

<match docker.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name docker-logs
</match>
```

### Health Checks

**Kubernetes Health Checks:**
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8010
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8010
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Performance Optimization

### Resource Allocation

**Production Resources:**
```yaml
services:
  ragflow:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
      replicas: 3
      
  deepwiki:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
      replicas: 2
```

### Database Optimization

**PostgreSQL Tuning:**
```sql
-- postgresql.conf optimizations
shared_buffers = '256MB'
effective_cache_size = '1GB'
maintenance_work_mem = '64MB'
checkpoint_completion_target = 0.9
wal_buffers = '16MB'
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

**Redis Optimization:**
```bash
# redis.conf
maxmemory 1gb
maxmemory-policy allkeys-lru
save ""  # Disable persistence for cache-only usage
tcp-keepalive 300
timeout 300
```

### Load Balancing

**Nginx Load Balancer:**
```nginx
upstream ragflow_backend {
    least_conn;
    server ragflow1:8010;
    server ragflow2:8010;
    server ragflow3:8010;
}

upstream deepwiki_backend {
    least_conn;
    server deepwiki1:8011;
    server deepwiki2:8011;
}

server {
    listen 80;
    
    location /ragflow/ {
        proxy_pass http://ragflow_backend/;
    }
    
    location /deepwiki/ {
        proxy_pass http://deepwiki_backend/;
    }
}
```

## Backup and Recovery

### Database Backup

**PostgreSQL Backup:**
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="noteparser"

# Create backup
pg_dump -h postgres -U noteparser -d $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/backup_$DATE.sql.gz s3://your-backup-bucket/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

**Redis Backup:**
```bash
# Create Redis backup
redis-cli --rdb /backups/redis_backup_$(date +%Y%m%d).rdb
```

### Disaster Recovery

**Recovery Procedures:**

1. **Database Recovery:**
   ```bash
   # Restore PostgreSQL
   gunzip -c backup_20250115.sql.gz | psql -h postgres -U noteparser -d noteparser
   
   # Restore Redis
   redis-cli --rdb restore.rdb
   ```

2. **Service Recovery:**
   ```bash
   # Kubernetes
   kubectl apply -f k8s-manifests/
   
   # Docker Swarm
   docker stack deploy -c docker-compose.prod.yml noteparser-ai
   ```

## Troubleshooting

### Common Deployment Issues

**Service Won't Start:**
```bash
# Check logs
docker-compose logs ragflow
kubectl logs deployment/ragflow

# Check resources
docker stats
kubectl top pods

# Check network connectivity
docker exec ragflow ping postgres
kubectl exec -it ragflow-pod -- ping postgres
```

**Database Connection Issues:**
```bash
# Test database connectivity
docker exec postgres pg_isready
psql -h localhost -p 5434 -U noteparser -d noteparser

# Check credentials
echo $DATABASE_URL
```

**Memory Issues:**
```bash
# Check memory usage
free -h
docker stats
kubectl top nodes

# Optimize memory settings
# Reduce worker processes, batch sizes
```

**Performance Issues:**
```bash
# Check metrics
curl http://localhost:9090/metrics

# Profile application
docker exec ragflow python -m cProfile -o profile.out service.py

# Database performance
EXPLAIN ANALYZE SELECT * FROM documents WHERE ...;
```

## Deployment Checklist

### Pre-deployment Tasks

!!! checklist "Before Deployment"

    - [ ] Environment variables configured
    - [ ] Database migrations applied  
    - [ ] SSL certificates valid
    - [ ] Health checks configured
    - [ ] Monitoring setup
    - [ ] Backup procedures in place

### Post-deployment Verification

!!! checklist "After Deployment"

    - [ ] Health checks passing
    - [ ] Metrics being collected
    - [ ] Logs being aggregated  
    - [ ] Performance baselines established
    - [ ] Backup procedures tested
    - [ ] Load testing completed

### Security Requirements

!!! checklist "Security"

    - [ ] API authentication enabled
    - [ ] Network isolation configured
    - [ ] Firewall rules applied
    - [ ] SSL/TLS certificates valid
    - [ ] Secrets properly managed
    - [ ] Container images scanned
