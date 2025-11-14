# OmniBase Infrastructure Deployment Guide

**Version:** 1.0.0
**Date:** 2025-11-14
**Status:** Production-Ready

---

## 🎯 Overview

This guide covers deploying the complete OmniBase infrastructure stack with all bridge nodes, adapters, and supporting services.

---

## 📦 Infrastructure Services

### Core Services

1. **PostgreSQL** (port 5435)
   - Database for infrastructure state storage
   - Reducer state persistence
   - Intent queue storage

2. **RedPanda** (Kafka-compatible streaming)
   - Event bus for all node communication
   - Ports: 9092 (Kafka), 29092 (external), 9644 (admin), 8082 (proxy)
   - Topics auto-created for ONEX infrastructure events

3. **Consul** (port 8500)
   - Service discovery and registration
   - Health check aggregation
   - KV store for configuration

### Adapter Nodes

4. **PostgreSQL Adapter** (port 8081)
   - Message bus bridge to PostgreSQL
   - Connection pooling (10-50 connections)
   - Circuit breaker integration

5. **Consul Adapter** (port 8082)
   - Message bus bridge to Consul
   - Service registration/deregistration
   - Health check coordination

### Optional Services

6. **RedPanda UI** (port 8080, dev profile)
   - Web interface for topic management
   - Message inspection and debugging

7. **Kafka Adapter** (port 8083)
   - Event streaming coordination
   - Producer pool management

8. **Hook Node** (port 8084)
   - Webhook notifications (Slack, Discord, etc.)
   - Infrastructure alerting

---

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- GitHub Personal Access Token (for private dependencies)

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and set your values
nano .env
```

**Required environment variables:**
```bash
POSTGRES_PASSWORD=your_secure_password
GITHUB_TOKEN=ghp_your_github_token
```

### 2. Start Infrastructure

```bash
# Use the startup script (recommended)
./start-infrastructure.sh

# Or manually with docker-compose
docker-compose -f docker-compose.infrastructure.yml up -d
```

### 3. Verify Services

```bash
# Check service status
docker-compose -f docker-compose.infrastructure.yml ps

# Check service health
curl http://localhost:8500/v1/agent/health/service/name/consul  # Consul
curl http://localhost:8081/health                               # PostgreSQL Adapter
curl http://localhost:8082/health                               # Consul Adapter
```

---

## 📋 Service Configuration

### PostgreSQL Configuration

**Environment Variables:**
```bash
POSTGRES_PASSWORD=your_password      # Required
POSTGRES_ADAPTER_PORT=8081          # Default: 8081
```

**Connection Details:**
- Host: `localhost`
- Port: `5435`
- Database: `omnibase_infrastructure`
- User: `postgres`

**Tables Created:**
- `infrastructure_state` - Reducer state storage
- `infrastructure_intents` - Intent queue

### RedPanda Configuration

**Environment Variables:**
```bash
REDPANDA_PORT=9092                          # Kafka API
REDPANDA_EXTERNAL_PORT=29092                # External access
REDPANDA_ADMIN_PORT=9644                    # Admin API
REDPANDA_PROXY_PORT=8082                    # HTTP Proxy
REDPANDA_UI_PORT=8080                       # UI (dev)
REDPANDA_LOG_LEVEL=info                     # Log level
REDPANDA_DEFAULT_PARTITIONS=3               # Default partitions
REDPANDA_DEFAULT_REPLICATION_FACTOR=1       # Replication factor
REDPANDA_LOG_RETENTION_MS=604800000         # 7 days
REDPANDA_SEGMENT_MS=86400000                # 24 hours
```

**Topics Created (ONEX Infrastructure):**

**Command Topics:**
- `dev.omnibase.onex.cmd.postgres-execute-query.v1`
- `dev.omnibase.onex.cmd.postgres-health-check.v1`

**Event Topics:**
- `dev.omnibase.onex.evt.postgres-query-completed.v1`
- `dev.omnibase.onex.evt.postgres-query-failed.v1`
- `dev.omnibase.onex.evt.postgres-connection-established.v1`
- `dev.omnibase.onex.evt.postgres-connection-failed.v1`

**Query-Response Topics:**
- `dev.omnibase.onex.qrs.postgres-health-requests.v1`
- `dev.omnibase.onex.qrs.postgres-health-replies.v1`

**Retry/DLT Topics:**
- `dev.omnibase.onex.rty.postgres-execute-query.v1`
- `dev.omnibase.onex.dlt.postgres-execute-query.v1`

**Audit/Metrics Topics:**
- `dev.omnibase.onex.aud.postgres-operations.v1`
- `dev.omnibase.onex.met.postgres-performance.v1`

### Consul Configuration

**Environment Variables:**
```bash
CONSUL_PORT=8500                    # HTTP API
CONSUL_DNS_PORT=8600                # DNS interface
CONSUL_LOG_LEVEL=INFO               # Log level
CONSUL_ADAPTER_PORT=8082            # Adapter port
```

**Access:**
- UI: http://localhost:8500
- API: http://localhost:8500/v1/
- DNS: localhost:8600

---

## 🔧 Service Management

### Start Services

```bash
# All services
docker-compose -f docker-compose.infrastructure.yml up -d

# Specific service
docker-compose -f docker-compose.infrastructure.yml up -d postgres

# With development profile (includes RedPanda UI)
docker-compose -f docker-compose.infrastructure.yml --profile development up -d
```

### Stop Services

```bash
# Graceful shutdown
docker-compose -f docker-compose.infrastructure.yml down

# With volume removal (CAUTION: deletes all data)
docker-compose -f docker-compose.infrastructure.yml down -v
```

### View Logs

```bash
# All services
docker-compose -f docker-compose.infrastructure.yml logs -f

# Specific service
docker-compose -f docker-compose.infrastructure.yml logs -f postgres-adapter

# Last 100 lines
docker-compose -f docker-compose.infrastructure.yml logs --tail=100 postgres
```

### Restart Services

```bash
# Restart all
docker-compose -f docker-compose.infrastructure.yml restart

# Restart specific service
docker-compose -f docker-compose.infrastructure.yml restart postgres-adapter
```

---

## 🔍 Health Checks

All services include health checks with automatic retries:

### PostgreSQL
```bash
docker exec omnibase-infra-postgres pg_isready -U postgres -d omnibase_infrastructure
```

### RedPanda
```bash
docker exec omnibase-infra-redpanda rpk cluster info
```

### Consul
```bash
docker exec omnibase-infra-consul consul members
```

### Adapters
```bash
curl http://localhost:8081/health  # PostgreSQL Adapter
curl http://localhost:8082/health  # Consul Adapter
```

---

## 🐛 Troubleshooting

### Service Won't Start

**Check logs:**
```bash
docker-compose -f docker-compose.infrastructure.yml logs [service-name]
```

**Common issues:**
- Port already in use: Change port in .env
- GitHub token invalid: Update GITHUB_TOKEN in .env
- Missing .env: Copy from .env.example

### Connection Issues

**PostgreSQL:**
```bash
# Test connection
docker exec -it omnibase-infra-postgres psql -U postgres -d omnibase_infrastructure

# Check connections
docker exec omnibase-infra-postgres psql -U postgres -c "SELECT * FROM pg_stat_activity;"
```

**RedPanda:**
```bash
# List topics
docker exec omnibase-infra-redpanda rpk topic list

# Describe topic
docker exec omnibase-infra-redpanda rpk topic describe dev.omnibase.onex.evt.postgres-query-completed.v1
```

**Consul:**
```bash
# Check catalog
docker exec omnibase-infra-consul consul catalog services

# Check health
docker exec omnibase-infra-consul consul operator raft list-peers
```

### Adapter Health

**Check adapter logs:**
```bash
docker-compose -f docker-compose.infrastructure.yml logs -f postgres-adapter
docker-compose -f docker-compose.infrastructure.yml logs -f consul-adapter
```

**Common adapter issues:**
- Circuit breaker open: Check service health
- Connection pool exhausted: Increase pool size in config
- Event bus unavailable: Verify RedPanda is healthy

---

## 📊 Monitoring

### Service Status Dashboard

```bash
# Quick status
docker-compose -f docker-compose.infrastructure.yml ps

# Detailed stats
docker stats omnibase-infra-postgres omnibase-infra-redpanda omnibase-infra-consul
```

### RedPanda UI (Development)

Access: http://localhost:8080

Features:
- Topic management
- Message inspection
- Consumer group monitoring
- Schema registry

### Consul UI

Access: http://localhost:8500

Features:
- Service catalog
- Health checks
- KV store browser
- Events

---

## 🔐 Security

### Secrets Management

**Docker Secrets (Recommended for Production):**
```bash
# Set secrets via environment
export GITHUB_TOKEN=ghp_your_token
export POSTGRES_PASSWORD=your_password
```

**Files:**
- `.env` - Never commit to git (in .gitignore)
- `.env.example` - Template for configuration

### Network Isolation

All services run on isolated `omnibase-network` bridge network.

**Service Communication:**
- Internal: Via Docker network (omnibase-infra-postgres:5432)
- External: Via published ports (localhost:5435)

### TLS/SSL

For production:
- Enable PostgreSQL SSL
- Configure RedPanda TLS
- Use Consul TLS/HTTPS

---

## 📈 Scaling

### Horizontal Scaling

**RedPanda:**
```yaml
# Add more brokers in docker-compose.yml
redpanda-2:
  image: redpandadata/redpanda:v24.2.7
  # ... configuration
```

**Adapters:**
```bash
# Scale adapter instances
docker-compose -f docker-compose.infrastructure.yml up -d --scale postgres-adapter=3
```

### Vertical Scaling

**Resource Limits:**
```yaml
# Add to service in docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

---

## 🎯 Production Deployment

### Recommended Configuration

**PostgreSQL:**
- Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
- Enable automated backups
- Configure read replicas

**RedPanda:**
- Use RedPanda Cloud or self-managed cluster
- 3+ brokers for production
- Replication factor: 3

**Consul:**
- 3+ server nodes
- Enable ACLs
- Configure TLS

**Adapters:**
- Deploy with Kubernetes/ECS
- Configure auto-scaling
- Use health checks for load balancing

### CI/CD Integration

```bash
# Build
docker-compose -f docker-compose.infrastructure.yml build

# Test
docker-compose -f docker-compose.infrastructure.yml run --rm postgres-adapter pytest

# Deploy
docker-compose -f docker-compose.infrastructure.yml up -d
```

---

## 📖 Additional Resources

- [Docker Compose Reference](https://docs.docker.com/compose/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [RedPanda Documentation](https://docs.redpanda.com/)
- [Consul Documentation](https://www.consul.io/docs)
- [ONEX Architecture Guide](./COMPLETE_MIGRATION_SUMMARY.md)

---

## ✅ Deployment Checklist

- [ ] Copy `.env.example` to `.env`
- [ ] Set `GITHUB_TOKEN` in `.env`
- [ ] Set `POSTGRES_PASSWORD` in `.env`
- [ ] Review and adjust port configurations
- [ ] Run `./start-infrastructure.sh`
- [ ] Verify all services are healthy
- [ ] Check adapter health endpoints
- [ ] Verify topics are created in RedPanda
- [ ] Test database connection
- [ ] Test Consul service discovery
- [ ] Review logs for any errors
- [ ] Set up monitoring/alerting (production)
- [ ] Configure backups (production)
- [ ] Enable TLS/SSL (production)

---

**Infrastructure is production-ready!** 🚀
