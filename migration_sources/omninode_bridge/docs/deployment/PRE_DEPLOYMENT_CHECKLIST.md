# Pre-Deployment Checklist - OmniNode Bridge

**Production Deployment Readiness Checklist**
**Version**: 1.0
**Last Updated**: October 25, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Infrastructure Readiness](#infrastructure-readiness)
3. [PostgreSQL Setup](#postgresql-setup)
4. [Application Configuration](#application-configuration)
5. [Security Configuration](#security-configuration)
6. [Kafka/Redpanda Setup](#kafkaredpanda-setup)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Backup and Recovery](#backup-and-recovery)
9. [Final Verification](#final-verification)

---

## Overview

This checklist ensures all prerequisites are met before deploying OmniNode Bridge to production environments.

**Deployment Phases**:
1. Pre-deployment preparation (this checklist)
2. Infrastructure provisioning
3. Application deployment
4. Post-deployment validation
5. Production monitoring

---

## Infrastructure Readiness

### Compute Resources

- [ ] **CPU**: Minimum 4 cores allocated
- [ ] **Memory**: Minimum 8GB RAM allocated
- [ ] **Storage**: Minimum 100GB SSD storage
- [ ] **Network**: Low-latency network connectivity (<10ms)

### Operating System

- [ ] **OS Version**: Ubuntu 22.04 LTS or RHEL 8+ (or compatible)
- [ ] **Kernel**: Linux kernel 5.10+ for optimal container performance
- [ ] **System Updates**: All security patches applied
- [ ] **Time Sync**: NTP/chrony configured for accurate timestamps

### Container Runtime

- [ ] **Docker**: Version 24.0+ installed and configured
- [ ] **Docker Compose**: Version 2.20+ installed
- [ ] **Container Registry Access**: Pull access to required images
- [ ] **Resource Limits**: Docker daemon configured with appropriate limits

### Network Configuration

- [ ] **Firewall Rules**: Required ports opened (5432, 9092, 8053, etc.)
- [ ] **DNS Resolution**: Hostname resolution working correctly
- [ ] **Load Balancer**: Configured if using multiple instances
- [ ] **SSL/TLS Certificates**: Valid certificates for HTTPS endpoints

---

## PostgreSQL Setup

### Database Installation

- [ ] **PostgreSQL Version**: PostgreSQL 15+ installed
- [ ] **PostgreSQL Extensions**: Required extensions available:
  - [ ] `uuid-ossp` installed
  - [ ] `pg_stat_statements` installed (optional but recommended)
- [ ] **Extension Packages**: `postgresql-contrib` package installed

### Database Creation

- [ ] **Database Created**: `omninode_bridge` database exists
- [ ] **Database User Created**: Application user created with appropriate privileges
- [ ] **User Password**: Strong password set (minimum 16 characters, alphanumeric + symbols)
- [ ] **Connection Limits**: Max connections configured (recommended: 100+)

### Extension Setup

Choose one of the following approaches:

**Option A: Superuser Pre-Creates Extensions (Recommended)**

- [ ] Extensions created by DBA or superuser:
  ```sql
  -- Connect as superuser
  CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
  CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
  ```
- [ ] Application user granted usage on schema:
  ```sql
  GRANT USAGE ON SCHEMA public TO omninode_app;
  ```
- [ ] Extensions verified:
  ```sql
  SELECT extname, extversion FROM pg_extension
  WHERE extname IN ('uuid-ossp', 'pg_stat_statements');
  ```

**Option B: Temporary Superuser for Migration (Development Only)**

- [ ] Application user temporarily granted superuser:
  ```sql
  ALTER USER omninode_app WITH SUPERUSER;
  ```
- [ ] Migrations run with superuser privileges
- [ ] Superuser privileges revoked after migration:
  ```sql
  ALTER USER omninode_app WITH NOSUPERUSER;
  ```

**Option C: Automated Script (Recommended for CI/CD)**

- [ ] Script executed: `bash deployment/scripts/setup_postgres_extensions.sh`
- [ ] Script output verified for success
- [ ] Extensions confirmed created

### Database Configuration

- [ ] **Connection Pooling**: `max_connections = 100` (or appropriate for workload)
- [ ] **Memory Configuration**:
  - [ ] `shared_buffers = 2GB` (25% of RAM)
  - [ ] `effective_cache_size = 6GB` (50-75% of RAM)
  - [ ] `work_mem = 16MB` (per-query memory)
  - [ ] `maintenance_work_mem = 512MB` (maintenance operations)
- [ ] **Performance Tuning**:
  - [ ] `random_page_cost = 1.1` (for SSDs)
  - [ ] `effective_io_concurrency = 200` (for SSDs)
  - [ ] `max_wal_size = 4GB`
  - [ ] `checkpoint_completion_target = 0.9`
- [ ] **Logging Configuration**:
  - [ ] `log_min_duration_statement = 1000` (log queries >1s)
  - [ ] `log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '`

### Migration Execution

- [ ] **Backup Created**: Full database backup before migrations
- [ ] **Migration Scripts Validated**: All scripts tested on staging
- [ ] **Extensions Pre-Created**: Extensions created before running migrations
- [ ] **Migrations Run**: All migrations executed successfully (see [Database Migrations](../../migrations/README.md))
- [ ] **Critical Migration 005**: Node registrations table created
  ```bash
  # Verify migration 005 applied
  psql -U postgres -d omninode_bridge -c "\d node_registrations"
  ```
- [ ] **Schema Verified**: All required tables exist:
  - [ ] `workflow_executions`
  - [ ] `workflow_steps`
  - [ ] `fsm_transitions`
  - [ ] `bridge_states`
  - [ ] **`node_registrations`** (CRITICAL for dual registration system)
  - [ ] `metadata_stamps`
- [ ] **Indexes Verified**: All performance indexes created
- [ ] **Rollback Plan**: Rollback procedures documented and tested

### Verification Commands

Run these commands to verify PostgreSQL setup:

```bash
# Check PostgreSQL version
psql -U postgres -c "SELECT version();"

# Verify extensions
psql -U postgres -d omninode_bridge -c "SELECT extname, extversion FROM pg_extension WHERE extname IN ('uuid-ossp', 'pg_stat_statements');"

# Check table count
psql -U omninode_app -d omninode_bridge -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"

# Verify indexes
psql -U omninode_app -d omninode_bridge -c "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';"

# Test UUID generation
psql -U omninode_app -d omninode_bridge -c "SELECT uuid_generate_v4();"

# Check connection limit
psql -U postgres -c "SHOW max_connections;"
```

---

## Application Configuration

### Environment Variables

- [ ] **Database Connection**:
  - [ ] `POSTGRES_HOST` configured
  - [ ] `POSTGRES_PORT` configured (default: 5432)
  - [ ] `POSTGRES_DATABASE` set to `omninode_bridge`
  - [ ] `POSTGRES_USER` set to application user
  - [ ] `POSTGRES_PASSWORD` set securely (use secrets management)
- [ ] **Kafka Configuration**:
  - [ ] `KAFKA_BOOTSTRAP_SERVERS` configured (e.g., `localhost:9092`)
  - [ ] `KAFKA_ENABLE_LOGGING` set to `true`
- [ ] **Service Configuration**:
  - [ ] `SERVICE_PORT` configured (default: 8053)
  - [ ] `LOG_LEVEL` set appropriately (INFO for production, DEBUG for troubleshooting)
  - [ ] `DEBUG` set to `false` for production
- [ ] **Connection Pool**:
  - [ ] `DB_POOL_MIN_SIZE` configured (default: 10)
  - [ ] `DB_POOL_MAX_SIZE` configured (default: 50)

### Secrets Management

- [ ] **Passwords**: Never hardcoded, stored in secure secrets management
- [ ] **API Keys**: Rotated regularly, stored securely
- [ ] **Certificates**: Valid and accessible, not committed to version control
- [ ] **Encryption Keys**: Generated securely, backed up securely

### Configuration Files

- [ ] **Docker Compose**: Production docker-compose.yml configured
- [ ] **Service Configs**: All service configurations validated
- [ ] **Volume Mounts**: Persistent volumes configured for data
- [ ] **Restart Policies**: Set to `always` or `unless-stopped`

---

## Security Configuration

### Network Security

- [ ] **Firewall Rules**: Only required ports exposed
- [ ] **SSL/TLS**: HTTPS enabled for all external endpoints
- [ ] **Database SSL**: PostgreSQL SSL connections enforced
- [ ] **Internal Communication**: Service-to-service communication secured

### Access Control

- [ ] **Database User**: Minimal privileges (no superuser in production)
- [ ] **Application User**: Non-root user for application processes
- [ ] **File Permissions**: Restricted permissions on config files (600 or 640)
- [ ] **Container Security**: Containers run as non-root user

### Compliance

- [ ] **Password Policy**: Strong passwords enforced
- [ ] **Audit Logging**: Database and application audit logs enabled
- [ ] **Data Encryption**: Encryption at rest enabled for sensitive data
- [ ] **Security Scanning**: Container images scanned for vulnerabilities

---

## Kafka/Redpanda Setup

### Installation

- [ ] **Kafka/Redpanda**: Installed and running
- [ ] **Version**: Kafka 3.0+ or Redpanda 22.0+
- [ ] **Broker Configuration**: Adequate broker resources allocated

### Topic Configuration

- [ ] **Topics Created**: All 13 required topics exist
- [ ] **Topic Settings**:
  - [ ] Replication factor: 3 (for production)
  - [ ] Partitions: Configured based on throughput requirements
  - [ ] Retention: Set to appropriate duration (7 days default)
- [ ] **Consumer Groups**: Configured for event processing

### Network Configuration

- [ ] **Hostname Resolution**: Broker hostnames resolvable
  ```bash
  # Add to /etc/hosts if needed (Kafka/Redpanda only)
  echo "127.0.0.1 omninode-bridge-redpanda" | sudo tee -a /etc/hosts
  ```
- [ ] **Port Access**: Broker ports accessible (9092, 29092)
- [ ] **Network Latency**: Low latency between brokers and application (<5ms)

### Verification

```bash
# Check Kafka/Redpanda is running
docker ps | grep -i redpanda

# List topics
docker exec omninode-bridge-redpanda rpk topic list

# Verify topic configuration
docker exec omninode-bridge-redpanda rpk topic describe <topic_name>
```

---

## Monitoring and Observability

### Metrics Collection

- [ ] **Prometheus**: Deployed and scraping application metrics
- [ ] **Grafana**: Dashboard configured for visualization
- [ ] **Service Metrics**: `/metrics` endpoint accessible
- [ ] **Database Metrics**: PostgreSQL metrics exported

### Logging

- [ ] **Log Aggregation**: Centralized logging configured (e.g., ELK, Loki)
- [ ] **Log Retention**: Appropriate retention policy set
- [ ] **Structured Logging**: JSON logging enabled
- [ ] **Log Levels**: Production log level set to INFO

### Alerting

- [ ] **Critical Alerts**: Configured for service downtime
- [ ] **Performance Alerts**: Configured for degraded performance
- [ ] **Error Rate Alerts**: Configured for increased error rates
- [ ] **Database Alerts**: Configured for connection pool exhaustion

### Health Checks

- [ ] **Liveness Probe**: Configured for container orchestration
- [ ] **Readiness Probe**: Configured for load balancer
- [ ] **Health Endpoint**: `/health` endpoint returns 200 OK
- [ ] **Dependency Checks**: Database and Kafka connectivity verified

---

## Backup and Recovery

### Backup Strategy

- [ ] **Database Backups**: Automated daily backups configured
- [ ] **Backup Retention**: 7 daily, 4 weekly, 12 monthly backups
- [ ] **Backup Storage**: Backups stored in multiple geographic locations
- [ ] **Backup Encryption**: Backups encrypted at rest
- [ ] **WAL Archiving**: Continuous WAL archiving for point-in-time recovery

### Recovery Testing

- [ ] **Restore Procedure**: Documented step-by-step
- [ ] **Restore Test**: Successful restore test completed
- [ ] **RTO**: Recovery Time Objective defined (target: 1 hour)
- [ ] **RPO**: Recovery Point Objective defined (target: 5 minutes)
- [ ] **Failover Plan**: Documented and tested

### Disaster Recovery

- [ ] **DR Site**: Disaster recovery environment provisioned
- [ ] **DR Sync**: Data replication to DR site configured
- [ ] **DR Test**: Full DR failover test completed
- [ ] **Runbooks**: DR runbooks documented and accessible

---

## Final Verification

### Pre-Deployment Smoke Tests

- [ ] **Database Connectivity**: Application can connect to database
  ```bash
  python -c "from omninode_bridge.infrastructure.postgres_client import PostgresConnectionManager; import asyncio; asyncio.run(PostgresConnectionManager(config).initialize())"
  ```
- [ ] **Migration Status**: All migrations completed successfully
  ```bash
  psql -U omninode_app -d omninode_bridge -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
  ```
- [ ] **Extension Status**: All required extensions created
  ```bash
  psql -U omninode_app -d omninode_bridge -c "SELECT extname FROM pg_extension WHERE extname IN ('uuid-ossp', 'pg_stat_statements');"
  ```
- [ ] **Kafka Connectivity**: Application can publish events to Kafka
  ```bash
  # Check Kafka topics
  docker exec omninode-bridge-redpanda rpk topic list
  ```
- [ ] **Health Check**: `/health` endpoint returns healthy status
  ```bash
  curl http://localhost:8053/health
  ```
- [ ] **Metrics Endpoint**: `/metrics` endpoint returns Prometheus metrics
  ```bash
  curl http://localhost:8053/metrics
  ```

### Performance Baseline

- [ ] **Load Test**: Successful load test with expected throughput
- [ ] **Performance Metrics**: Baseline metrics captured
- [ ] **Resource Usage**: CPU, memory, disk usage within acceptable limits
- [ ] **Query Performance**: Database query performance meets targets (<50ms p95)

### Documentation

- [ ] **Deployment Guide**: Complete deployment documentation available
- [ ] **Runbooks**: Operational runbooks created
- [ ] **Architecture Diagrams**: Updated architecture diagrams
- [ ] **Configuration Reference**: All configuration options documented
- [ ] **Troubleshooting Guide**: Common issues and solutions documented

### Rollback Plan

- [ ] **Rollback Procedure**: Step-by-step rollback documented
- [ ] **Rollback Test**: Rollback procedure tested on staging
- [ ] **Rollback Time**: Estimated rollback time calculated
- [ ] **Data Migration Rollback**: Database rollback procedure documented

### Sign-Off

- [ ] **Development Team**: Deployment package approved
- [ ] **Operations Team**: Infrastructure ready
- [ ] **Security Team**: Security review completed
- [ ] **Management**: Deployment authorized

---

## Related Documentation

- [Database Migrations](../../migrations/README.md) - PostgreSQL extension requirements and migration procedures
- [Database Guide](../database/DATABASE_GUIDE.md) - Complete database schema and operations guide
- [Secure Deployment Guide](./SECURE_DEPLOYMENT_GUIDE.md) - Security best practices for production
- [Setup Guide](../SETUP.md) - Development environment setup
- [Operations Guide](../operations/OPERATIONS_GUIDE.md) - Production operations and monitoring

---

**Document Version**: 1.0
**Maintained By**: omninode_bridge team
**Last Review**: October 25, 2025
**Next Review**: November 25, 2025
