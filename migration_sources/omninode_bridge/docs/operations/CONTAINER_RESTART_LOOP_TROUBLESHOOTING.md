# Container Restart Loop Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures for container restart loops in the OmniNode Bridge ecosystem. Container restart loops can occur due to various factors including database connection failures, Kafka connectivity issues, configuration problems, and resource constraints.

## Common Causes and Solutions

### 1. Database Connection Failures

#### Symptoms
- PostgreSQL client containers repeatedly restarting
- Logs showing connection timeouts or authentication failures
- Database connection pool exhaustion

#### Root Causes
```bash
# Common error patterns in logs
ERROR - Database connection failure during initialization
ERROR - Failed to connect to PostgreSQL after all retry attempts
ERROR - Database authentication failed during initialization
ERROR - Database connection pool exhausted during initialization
```

#### Diagnosis Steps
```bash
# Check PostgreSQL container status
docker ps | grep postgres
docker logs omninode-bridge-postgres --tail 50

# Test database connectivity
docker exec omninode-bridge-postgres pg_isready -h localhost -p 5432

# Check connection pool utilization
curl -s http://localhost:8001/health | jq '.infrastructure_health.postgres'
```

#### Solutions

**1. Fix Database Configuration**
```bash
# Verify database environment variables
docker exec omninode-bridge-hook-receiver env | grep POSTGRES

# Common fixes in docker-compose.yml or .env
POSTGRES_HOST=omninode-bridge-postgres  # Use Docker service name
POSTGRES_PORT=5432                      # Internal Docker port
POSTGRES_PASSWORD=your-secure-password  # Must be set
POSTGRES_SSL_ENABLED=false             # For development
```

**2. Adjust Connection Pool Settings**
```bash
# For high-load environments, increase pool sizes
POSTGRES_POOL_MIN_SIZE=10
POSTGRES_POOL_MAX_SIZE=50
POSTGRES_QUERY_TIMEOUT=60
POSTGRES_ACQUIRE_TIMEOUT=10
```

**3. Enable Connection Resilience**
```bash
# Enable enhanced error handling and retries
POSTGRES_LEAK_DETECTION=true
POSTGRES_POOL_EXHAUSTION_THRESHOLD=80.0
```

### 2. Kafka/RedPanda Connectivity Issues

#### Symptoms
- Services failing to connect to Kafka on startup
- Workflow coordinator restart loops
- Event processing failures

#### Root Causes
```bash
# Common Kafka error patterns
ERROR - Kafka consumer timeout
ERROR - Network error starting Kafka consumer
ERROR - Failed to connect to Kafka broker
```

#### Diagnosis Steps
```bash
# Check RedPanda container status
docker ps | grep redpanda
docker logs omninode-bridge-redpanda --tail 50

# Test Kafka connectivity
docker exec omninode-bridge-redpanda kafka-topics --bootstrap-server localhost:9092 --list

# Check Kafka client configuration
curl -s http://localhost:8006/health | jq '.infrastructure_health.kafka'
```

#### Solutions

**1. Fix Kafka Bootstrap Servers**
```bash
# Use correct Docker service names
KAFKA_BOOTSTRAP_SERVERS=omninode-bridge-redpanda:9092  # Internal Docker network
# OR for external access:
KAFKA_BOOTSTRAP_SERVERS=localhost:29092                # External port mapping
```

**2. Verify Topic Configuration**
```bash
# Ensure topics exist
KAFKA_WORKFLOW_TOPIC=dev.omninode_bridge.onex.workflows.v1
KAFKA_TASK_EVENTS_TOPIC=dev.omninode_bridge.onex.task-events.v1
```

### 3. Environment Variable Configuration Issues

#### Symptoms
- Services failing validation on startup
- API key authentication errors
- SSL/TLS configuration failures

#### Root Causes
```bash
# Configuration validation errors
ERROR - API_KEY environment variable is required for authentication
ERROR - Database password must be provided
ERROR - SSL configuration error
```

#### Diagnosis Steps
```bash
# Check environment variable configuration
docker exec omninode-bridge-hook-receiver env | grep -E "(API_KEY|POSTGRES|KAFKA)"

# Validate configuration
docker exec omninode-bridge-hook-receiver python -c "
from omninode_bridge.config.environment_config import DatabaseConfig, KafkaConfig
try:
    db = DatabaseConfig()
    kafka = KafkaConfig()
    print('✅ Configuration valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

#### Solutions

**1. Set Required Environment Variables**
```bash
# Required for all environments
API_KEY=your-secure-api-key-here
POSTGRES_PASSWORD=your-secure-database-password

# Required for production
JWT_SECRET=your-jwt-secret-minimum-32-characters-long
WEBHOOK_SIGNING_SECRET=your-webhook-signing-secret
```

**2. Environment-Specific Configuration**
```bash
# Development
ENVIRONMENT=development
SECURITY_MODE=permissive
POSTGRES_SSL_ENABLED=false

# Production
ENVIRONMENT=production
SECURITY_MODE=compliance
POSTGRES_SSL_ENABLED=true
POSTGRES_SSL_MODE=require
```

### 4. Resource Constraint Issues

#### Symptoms
- Services killed by OOM killer
- CPU throttling causing timeouts
- Disk space exhaustion

#### Diagnosis Steps
```bash
# Check container resource usage
docker stats --no-stream

# Check Docker system resources
docker system df
docker system events --since 10m

# Check host system resources
df -h
free -h
top
```

#### Solutions

**1. Increase Container Resources**
```yaml
# docker-compose.yml
services:
  hook-receiver:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
```

**2. Optimize Application Settings**
```bash
# Reduce resource usage
POSTGRES_POOL_MAX_SIZE=20          # Reduce from default 50
KAFKA_BUFFER_MEMORY=16777216       # Reduce from default 33554432
WORKER_PROCESSES=1                 # Reduce for development
```

### 5. Service Dependency Issues

#### Symptoms
- Services starting before dependencies are ready
- Intermittent connection failures
- Race conditions during startup

#### Solutions

**1. Add Health Checks and Dependencies**
```yaml
# docker-compose.yml
services:
  hook-receiver:
    depends_on:
      postgres:
        condition: service_healthy
      redpanda:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
```

**2. Add Startup Delays**
```bash
# In service startup scripts
sleep 30  # Wait for dependencies
```

## Systematic Troubleshooting Procedure

### Step 1: Identify the Restarting Service
```bash
# Monitor container restarts
watch 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# Check restart counts
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(Restarting|Exited)"
```

### Step 2: Examine Container Logs
```bash
# Get recent logs from restarting container
docker logs <container_name> --tail 100

# Follow logs in real-time
docker logs <container_name> --follow

# Check logs with timestamps
docker logs <container_name> --timestamps --since 5m
```

### Step 3: Check Health Status
```bash
# Check service health endpoints
curl -s http://localhost:8001/health | jq '.'  # Hook Receiver
curl -s http://localhost:8006/health | jq '.'  # Workflow Coordinator

# Check infrastructure health
curl -s http://localhost:8001/health | jq '.infrastructure_health'
```

### Step 4: Validate Configuration
```bash
# Check environment variables
docker exec <container_name> env | grep -E "(POSTGRES|KAFKA|API)"

# Test configuration loading
docker exec <container_name> python -c "
from omninode_bridge.config.environment_config import DatabaseConfig
print('Config loaded successfully')
"
```

### Step 5: Test Dependencies
```bash
# Test PostgreSQL connectivity
docker exec omninode-bridge-postgres pg_isready

# Test Kafka connectivity
docker exec omninode-bridge-redpanda kafka-topics --bootstrap-server localhost:9092 --list

# Test network connectivity between containers
docker exec omninode-bridge-hook-receiver ping omninode-bridge-postgres
```

### Step 6: Apply Targeted Fixes

Based on the root cause identified:

**Database Issues:**
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Reset connection pools
docker-compose restart hook-receiver workflow-coordinator
```

**Kafka Issues:**
```bash
# Restart RedPanda
docker-compose restart redpanda

# Clear consumer groups if needed
docker exec omninode-bridge-redpanda kafka-consumer-groups --bootstrap-server localhost:9092 --delete --group workflow_coordinator
```

**Configuration Issues:**
```bash
# Update .env file with correct values
# Restart affected services
docker-compose up -d --force-recreate
```

## Prevention Strategies

### 1. Robust Health Checks
```yaml
# Comprehensive health check configuration
healthcheck:
  test: |
    curl -f http://localhost:8001/health &&
    python -c "import requests; assert requests.get('http://localhost:8001/health').json()['status'] == 'healthy'"
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 90s
```

### 2. Graceful Startup Ordering
```yaml
# Proper service dependencies
services:
  postgres:
    # Database starts first

  redpanda:
    depends_on:
      - postgres
    # Message queue starts after database

  hook-receiver:
    depends_on:
      postgres:
        condition: service_healthy
      redpanda:
        condition: service_started
    # Application services start last
```

### 3. Enhanced Error Handling
```python
# Example: Robust initialization with retries
async def initialize_with_retries(max_retries=5, base_delay=2.0):
    for attempt in range(max_retries):
        try:
            await initialize_services()
            logger.info("Services initialized successfully")
            return
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to initialize after {max_retries} attempts: {e}")
                raise

            delay = base_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Initialization attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)
```

### 4. Resource Monitoring
```bash
# Add resource monitoring to health checks
curl -s http://localhost:8001/health | jq '{
  status: .status,
  memory_usage: .system_metrics.memory_usage_percent,
  cpu_usage: .system_metrics.cpu_usage_percent,
  disk_usage: .system_metrics.disk_usage_percent
}'
```

### 5. Configuration Validation
```bash
# Pre-deployment configuration validation
docker-compose config  # Validate docker-compose.yml
python -m omninode_bridge.config.validation  # Validate application config
```

## Emergency Procedures

### Quick Recovery Steps
```bash
# 1. Stop all services
docker-compose down

# 2. Clean up any stuck containers
docker container prune -f

# 3. Clean up networks
docker network prune -f

# 4. Restart with fresh state
docker-compose up -d

# 5. Monitor startup
docker-compose logs -f
```

### Backup Recovery
```bash
# If issues persist, restore from backup
./scripts/restore-from-backup.sh

# Or reset to clean state
docker-compose down -v  # WARNING: Destroys data
docker-compose up -d
```

### Escalation Criteria

Escalate to senior engineers if:
- Restart loops persist after applying standard fixes
- Multiple services are affected simultaneously
- Data corruption is suspected
- Security incidents are involved
- Customer-facing services are down for >30 minutes

## Monitoring and Alerting

### Key Metrics to Monitor
```bash
# Container restart frequency
docker events --filter type=container --filter event=restart

# Service health status
curl -s http://localhost:8001/health | jq '.status'

# Resource utilization
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

### Automated Alerts
```yaml
# Example monitoring configuration
alerts:
  - name: "Container Restart Loop"
    condition: "container_restarts > 3 in 5m"
    action: "notify_oncall"

  - name: "Service Health Degraded"
    condition: "health_check_failures > 2 in 2m"
    action: "run_auto_recovery"
```

This guide provides comprehensive procedures for identifying, diagnosing, and resolving container restart loops in the OmniNode Bridge ecosystem. Use it systematically to ensure rapid resolution and prevent recurrence.
