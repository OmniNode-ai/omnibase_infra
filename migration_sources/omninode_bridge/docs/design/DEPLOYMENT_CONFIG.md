# Docker Compose Deployment Configuration - Automated Stamping MVP

**Status**: Design Document
**Version**: 1.0.0
**Date**: October 2025

## Table of Contents

1. [Services Overview](#services-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Complete Docker Compose Structure](#complete-docker-compose-structure)
4. [Service Configurations](#service-configurations)
5. [Network Configuration](#network-configuration)
6. [Volume Configuration](#volume-configuration)
7. [Environment Variables](#environment-variables)
8. [Startup Scripts](#startup-scripts)
9. [Health Check Scripts](#health-check-scripts)
10. [Development vs Production](#development-vs-production)
11. [Resource Limits](#resource-limits)
12. [Monitoring Configuration](#monitoring-configuration)
13. [Backup & Recovery](#backup--recovery)
14. [Testing Stack](#testing-stack)
15. [Integration Steps](#integration-steps)

---

## 1. Services Overview

### Complete Service Inventory

```yaml
# Infrastructure Services (Existing)
Infrastructure:
  - redpanda: Kafka-compatible event streaming (Redpanda v24.2.7)
  - postgres: Database with pgvector extension (PostgreSQL 15)
  - consul: Service discovery and configuration (Consul 1.17)
  - vault: Secrets management (Vault 1.15)
  - redpanda-topic-manager: Automatic topic provisioning

# Application Services (Existing)
Existing:
  - metadata-stamping: O.N.E. v0.1 protocol support (Port 8057)
  - onextree: Project structure intelligence (Port 8058)
  - hook-receiver: HTTP webhook interface (Port 8001)
  - model-metrics: AI Lab integration (Port 8005)

# New MVP Services (To Add)
New:
  - stamping-consumer: Event processor for automated stamping
  - intelligence-enrichment: OnexTree intelligence integration
  - prometheus: Metrics aggregation and monitoring
  - grafana: Visualization and dashboards
  - alertmanager: Alert routing and management

# Development Tools (Optional)
Development:
  - redpanda-ui: Web interface for Kafka topics (Port 3839)
```

### Service Dependencies

```
Infrastructure Layer:
  ├── consul (no deps)
  ├── vault (no deps)
  ├── postgres (no deps)
  └── redpanda (no deps)
      └── redpanda-topic-manager (depends: redpanda)

Application Layer:
  ├── metadata-stamping (depends: postgres, redpanda, consul)
  ├── onextree (no deps)
  ├── hook-receiver (depends: postgres, redpanda)
  ├── model-metrics (depends: postgres, redpanda)
  ├── stamping-consumer (depends: postgres, redpanda, metadata-stamping)
  └── intelligence-enrichment (depends: redpanda, onextree, metadata-stamping)

Monitoring Layer:
  ├── prometheus (scrapes: all application services)
  ├── alertmanager (depends: prometheus)
  └── grafana (depends: prometheus)
```

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      External Clients                                │
│                 (Git Hooks, CI/CD, Manual Triggers)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Hook Receiver (8001)                            │
│         HTTP API → Kafka Events (hook-received.v1)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Redpanda (Kafka)                                │
│    Topics: hook-received, stamp-created, batch-processed            │
└────────┬────────────────────────────────┬───────────────────────────┘
         │                                │
         ▼                                ▼
┌──────────────────┐            ┌──────────────────────┐
│ Stamping Consumer│            │Intelligence Enrichment│
│  (Event Processor)│            │   (OnexTree Client)   │
│                  │            │                      │
│ - Consumes events│            │ - Enriches metadata  │
│ - Calls metadata │            │ - Adds tree context  │
│   stamping API   │            │ - Publishes enhanced │
│ - Handles batches│            │   events            │
└────────┬─────────┘            └──────────┬───────────┘
         │                                │
         ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Metadata Stamping Service (8057)                        │
│         BLAKE3 Hashing + Database Persistence                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PostgreSQL (5436)                               │
│          Tables: metadata_stamps, hash_metrics                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      Monitoring Stack                                │
│  Prometheus (9090) ← Grafana (3000) ← Alertmanager (9093)          │
│  Scrapes: All services on /metrics endpoints                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Docker Compose Structure

### File: `deployment/docker-compose.mvp.yml`

```yaml
# Docker Compose MVP Configuration
# OmniNode Bridge - Automated Stamping Stack
# Version: 1.0.0

version: '3.9'

# Network Configuration
networks:
  omninode-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1

# Volume Configuration
volumes:
  # Infrastructure volumes
  postgres-data:
    driver: local
  redpanda-data:
    driver: local
  consul-data:
    driver: local
  consul-config:
    driver: local
  vault-data:
    driver: local
  vault-logs:
    driver: local

  # Application volumes
  onextree-cache:
    driver: local
  consumer-state:
    driver: local

  # Monitoring volumes
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
  alertmanager-data:
    driver: local

# Secrets Configuration
secrets:
  postgres_password:
    environment: "POSTGRES_PASSWORD"
  api_key:
    environment: "API_KEY"
  github_token:
    environment: "GITHUB_TOKEN"

# Services Configuration
services:

  # =========================================================================
  # INFRASTRUCTURE SERVICES
  # =========================================================================

  # Consul - Service Discovery
  consul:
    image: hashicorp/consul:1.17
    container_name: omninode-mvp-consul
    hostname: consul
    networks:
      - omninode-network
    ports:
      - "${CONSUL_PORT:-28500}:8500"
      - "${CONSUL_DNS_PORT:-28600}:8600/udp"
    environment:
      CONSUL_BIND_INTERFACE: eth0
      CONSUL_CLIENT_INTERFACE: eth0
    command: >
      consul agent
      -server
      -bootstrap-expect=1
      -datacenter=omninode-mvp
      -data-dir=/consul/data
      -config-dir=/consul/config
      -ui
      -client=0.0.0.0
      -bind=0.0.0.0
      -log-level=${CONSUL_LOG_LEVEL:-info}
    volumes:
      - consul-data:/consul/data
      - consul-config:/consul/config
    healthcheck:
      test: ["CMD", "consul", "members"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
        reservations:
          cpus: '0.1'
          memory: 128M

  # PostgreSQL - Database
  postgres:
    image: pgvector/pgvector:pg15
    container_name: omninode-mvp-postgres
    hostname: postgres
    networks:
      - omninode-network
    secrets:
      - postgres_password
    environment:
      POSTGRES_DB: omninode_bridge
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
      # Performance tuning
      POSTGRES_SHARED_BUFFERS: 256MB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB
      POSTGRES_MAINTENANCE_WORK_MEM: 64MB
      POSTGRES_CHECKPOINT_COMPLETION_TARGET: 0.9
      POSTGRES_WAL_BUFFERS: 16MB
      POSTGRES_DEFAULT_STATISTICS_TARGET: 100
      POSTGRES_RANDOM_PAGE_COST: 1.1
      POSTGRES_EFFECTIVE_IO_CONCURRENCY: 200
      POSTGRES_WORK_MEM: 4MB
      POSTGRES_MIN_WAL_SIZE: 1GB
      POSTGRES_MAX_WAL_SIZE: 4GB
    ports:
      - "${POSTGRES_PORT:-5436}:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/postgres-init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d omninode_bridge"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  # Redpanda - Kafka-compatible Event Streaming
  redpanda:
    image: redpandadata/redpanda:v24.2.7
    container_name: omninode-mvp-redpanda
    hostname: redpanda
    networks:
      - omninode-network
    ports:
      - "${REDPANDA_PORT:-29092}:9092"
      - "${REDPANDA_EXTERNAL_PORT:-29102}:29092"
      - "${REDPANDA_ADMIN_PORT:-29654}:9644"
      - "${REDPANDA_PROXY_PORT:-28092}:8082"
    command:
      - redpanda
      - start
      - --kafka-addr=internal://0.0.0.0:9092,external://0.0.0.0:29092
      - --advertise-kafka-addr=internal://redpanda:9092,external://localhost:29092
      - --pandaproxy-addr=internal://0.0.0.0:8082,external://0.0.0.0:8083
      - --advertise-pandaproxy-addr=internal://redpanda:8082,external://localhost:8083
      - --schema-registry-addr=internal://0.0.0.0:8081,external://0.0.0.0:8084
      - --rpc-addr=redpanda:33145
      - --advertise-rpc-addr=redpanda:33145
      - --mode=dev-container
      - --smp=2
      - --memory=2G
      - --default-log-level=${REDPANDA_LOG_LEVEL:-info}
    environment:
      REDPANDA_AUTO_CREATE_TOPICS_ENABLED: "true"
      REDPANDA_DEFAULT_TOPIC_PARTITIONS: ${REDPANDA_DEFAULT_PARTITIONS:-3}
      REDPANDA_DEFAULT_TOPIC_REPLICATIONS: ${REDPANDA_DEFAULT_REPLICATION_FACTOR:-1}
      REDPANDA_TOPIC_CLEANUP_POLICY: delete
      REDPANDA_LOG_RETENTION_MS: ${REDPANDA_LOG_RETENTION_MS:-604800000}
      REDPANDA_SEGMENT_MS: ${REDPANDA_SEGMENT_MS:-86400000}
    volumes:
      - redpanda-data:/var/lib/redpanda/data
    healthcheck:
      test: ["CMD", "rpk", "cluster", "health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  # Redpanda Topic Manager - Topic Provisioning
  redpanda-topic-manager:
    image: redpandadata/redpanda:v24.2.7
    container_name: omninode-mvp-topic-manager
    networks:
      - omninode-network
    depends_on:
      redpanda:
        condition: service_healthy
    entrypoint: ["/bin/sh"]
    command:
      - -c
      - |
        echo '=== OmniNode MVP Topic Manager ==='
        echo 'Waiting for Redpanda cluster...'
        rpk cluster info --brokers redpanda:9092

        echo 'Creating MVP infrastructure topics...'

        # Stamping workflow topics
        rpk topic create dev.omninode.mvp.evt.stamp-requested.v1 --brokers redpanda:9092 -p 3 -r 1 || true
        rpk topic create dev.omninode.mvp.evt.stamp-created.v1 --brokers redpanda:9092 -p 3 -r 1 || true
        rpk topic create dev.omninode.mvp.evt.batch-processed.v1 --brokers redpanda:9092 -p 3 -r 1 || true
        rpk topic create dev.omninode.mvp.evt.enrichment-completed.v1 --brokers redpanda:9092 -p 3 -r 1 || true

        # Hook receiver topics
        rpk topic create dev.omninode.mvp.evt.hook-received.v1 --brokers redpanda:9092 -p 3 -r 1 || true
        rpk topic create dev.omninode.mvp.evt.hook-validated.v1 --brokers redpanda:9092 -p 3 -r 1 || true

        # DLQ topics
        rpk topic create dev.omninode.mvp.dlq.stamp-failed.v1 --brokers redpanda:9092 -p 3 -r 1 || true
        rpk topic create dev.omninode.mvp.dlq.enrichment-failed.v1 --brokers redpanda:9092 -p 3 -r 1 || true

        # Metrics and monitoring
        rpk topic create dev.omninode.mvp.met.performance.v1 --brokers redpanda:9092 -p 3 -r 1 || true
        rpk topic create dev.omninode.mvp.met.consumer-lag.v1 --brokers redpanda:9092 -p 3 -r 1 || true

        echo 'Topic creation completed!'
        echo 'Starting topic monitoring service...'

        while true; do
          sleep 60
          echo "=== Topic Status Check $(date) ==="
          rpk topic list --brokers redpanda:9092 | grep "dev.omninode.mvp" || echo "No MVP topics found"
        done
    healthcheck:
      test: ["CMD", "rpk", "topic", "list", "--brokers", "redpanda:9092"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.2'
          memory: 128M

  # =========================================================================
  # APPLICATION SERVICES
  # =========================================================================

  # Metadata Stamping Service
  metadata-stamping:
    build:
      context: ..
      dockerfile: docker/metadata-stamping/Dockerfile
      target: ${BUILD_TARGET:-production}
      secrets:
        - gh_pat
    image: omninode-mvp/metadata-stamping:latest
    container_name: omninode-mvp-metadata-stamping
    hostname: metadata-stamping
    networks:
      - omninode-network
    depends_on:
      postgres:
        condition: service_healthy
      redpanda:
        condition: service_healthy
      consul:
        condition: service_healthy
      redpanda-topic-manager:
        condition: service_healthy
    ports:
      - "8057:8053"
      - "9091:9090"
    environment:
      # Service configuration
      METADATA_STAMPING_SERVICE_HOST: 0.0.0.0
      METADATA_STAMPING_SERVICE_PORT: 8053
      METADATA_STAMPING_SERVICE_NAME: metadata-stamping-mvp
      METADATA_STAMPING_LOG_LEVEL: ${LOG_LEVEL:-info}

      # Database
      METADATA_STAMPING_DB_HOST: postgres
      METADATA_STAMPING_DB_PORT: 5432
      METADATA_STAMPING_DB_NAME: omninode_bridge
      METADATA_STAMPING_DB_USER: postgres
      METADATA_STAMPING_DB_PASSWORD: ${POSTGRES_PASSWORD}
      METADATA_STAMPING_DB_POOL_MIN_SIZE: 10
      METADATA_STAMPING_DB_POOL_MAX_SIZE: 50

      # Performance tuning
      METADATA_STAMPING_HASH_GENERATOR_POOL_SIZE: 100
      METADATA_STAMPING_HASH_GENERATOR_MAX_WORKERS: 8
      METADATA_STAMPING_ENABLE_BATCH_OPERATIONS: "true"

      # Kafka integration
      METADATA_STAMPING_ENABLE_EVENTS: "true"
      METADATA_STAMPING_KAFKA_BOOTSTRAP_SERVERS: redpanda:9092
      METADATA_STAMPING_KAFKA_ENABLE_DLQ: "true"
      METADATA_STAMPING_KAFKA_MAX_RETRY_ATTEMPTS: 3

      # Service discovery
      METADATA_STAMPING_ENABLE_REGISTRY: "true"
      METADATA_STAMPING_CONSUL_HOST: consul
      METADATA_STAMPING_CONSUL_PORT: 8500

      # Metrics
      METADATA_STAMPING_ENABLE_PROMETHEUS_METRICS: "true"
      METADATA_STAMPING_PROMETHEUS_PORT: 9090

      # Security
      API_KEY: ${API_KEY}

    volumes:
      - ../logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8053/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 45s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    profiles:
      - mvp
      - production

  # OnexTree Service - Project Structure Intelligence
  onextree:
    build:
      context: ..
      dockerfile: docker/onextree/Dockerfile
    image: omninode-mvp/onextree:latest
    container_name: omninode-mvp-onextree
    hostname: onextree
    networks:
      - omninode-network
    ports:
      - "8058:8058"
    environment:
      LOG_LEVEL: ${LOG_LEVEL:-info}
      PYTHONPATH: /app/src
      ONEXTREE_CACHE_DIR: /cache
      ONEXTREE_MAX_DEPTH: 10
      ONEXTREE_ENABLE_METRICS: "true"
    volumes:
      - ../logs:/app/logs
      - onextree-cache:/cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8058/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.2'
          memory: 256M
    profiles:
      - mvp
      - production

  # Hook Receiver - HTTP Webhook Interface
  hook-receiver:
    build:
      context: ..
      dockerfile: docker/hook-receiver/Dockerfile
    image: omninode-mvp/hook-receiver:latest
    container_name: omninode-mvp-hook-receiver
    hostname: hook-receiver
    networks:
      - omninode-network
    depends_on:
      postgres:
        condition: service_healthy
      redpanda:
        condition: service_healthy
    ports:
      - "8001:8001"
    environment:
      # Service configuration
      HOOK_RECEIVER_HOST: 0.0.0.0
      HOOK_RECEIVER_PORT: 8001
      LOG_LEVEL: ${LOG_LEVEL:-info}

      # Kafka
      KAFKA_BOOTSTRAP_SERVERS: redpanda:9092

      # PostgreSQL
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_DATABASE: omninode_bridge
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}

      # Security
      API_KEY: ${API_KEY}
      WEBHOOK_SIGNING_SECRET: ${WEBHOOK_SIGNING_SECRET}
      ENABLE_WEBHOOK_SIGNATURE_VERIFICATION: "true"

      # Rate limiting
      RATE_LIMIT_ENABLED: "true"
      RATE_LIMIT_RPM: 100
      RATE_LIMIT_BURST: 20

    volumes:
      - ../logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.2'
          memory: 256M
    profiles:
      - mvp
      - production

  # =========================================================================
  # NEW MVP SERVICES
  # =========================================================================

  # Stamping Consumer - Event Processor
  stamping-consumer:
    build:
      context: ..
      dockerfile: docker/stamping-consumer/Dockerfile
    image: omninode-mvp/stamping-consumer:latest
    container_name: omninode-mvp-stamping-consumer
    hostname: stamping-consumer
    networks:
      - omninode-network
    depends_on:
      redpanda:
        condition: service_healthy
      postgres:
        condition: service_healthy
      metadata-stamping:
        condition: service_healthy
    environment:
      # Service configuration
      CONSUMER_SERVICE_NAME: stamping-consumer-mvp
      LOG_LEVEL: ${LOG_LEVEL:-info}

      # Kafka consumer configuration
      KAFKA_BOOTSTRAP_SERVERS: redpanda:9092
      KAFKA_GROUP_ID: stamping-consumer-mvp
      KAFKA_AUTO_OFFSET_RESET: earliest
      KAFKA_ENABLE_AUTO_COMMIT: "false"
      KAFKA_MAX_POLL_RECORDS: 100
      KAFKA_SESSION_TIMEOUT_MS: 30000
      KAFKA_HEARTBEAT_INTERVAL_MS: 10000

      # Consumer topics
      KAFKA_INPUT_TOPICS: dev.omninode.mvp.evt.hook-received.v1,dev.omninode.mvp.evt.stamp-requested.v1
      KAFKA_OUTPUT_TOPIC: dev.omninode.mvp.evt.stamp-created.v1
      KAFKA_DLQ_TOPIC: dev.omninode.mvp.dlq.stamp-failed.v1

      # Processing configuration
      CONSUMER_BATCH_SIZE: 50
      CONSUMER_MAX_WORKERS: 10
      CONSUMER_PROCESSING_TIMEOUT_SEC: 60
      CONSUMER_RETRY_ATTEMPTS: 3
      CONSUMER_RETRY_BACKOFF_MS: 1000

      # Metadata stamping service
      METADATA_STAMPING_URL: http://metadata-stamping:8053
      METADATA_STAMPING_API_KEY: ${API_KEY}
      METADATA_STAMPING_TIMEOUT_SEC: 30

      # PostgreSQL (for state tracking)
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_DATABASE: omninode_bridge
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}

      # Performance monitoring
      CONSUMER_ENABLE_METRICS: "true"
      CONSUMER_METRICS_PORT: 9092

    volumes:
      - ../logs:/app/logs
      - consumer-state:/app/state
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9092/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.2'
          memory: 256M
    profiles:
      - mvp
      - production

  # Intelligence Enrichment Service - OnexTree Integration
  intelligence-enrichment:
    build:
      context: ..
      dockerfile: docker/intelligence-enrichment/Dockerfile
    image: omninode-mvp/intelligence-enrichment:latest
    container_name: omninode-mvp-intelligence-enrichment
    hostname: intelligence-enrichment
    networks:
      - omninode-network
    depends_on:
      redpanda:
        condition: service_healthy
      onextree:
        condition: service_healthy
      metadata-stamping:
        condition: service_healthy
    environment:
      # Service configuration
      ENRICHMENT_SERVICE_NAME: intelligence-enrichment-mvp
      LOG_LEVEL: ${LOG_LEVEL:-info}

      # Kafka consumer configuration
      KAFKA_BOOTSTRAP_SERVERS: redpanda:9092
      KAFKA_GROUP_ID: intelligence-enrichment-mvp
      KAFKA_AUTO_OFFSET_RESET: earliest

      # Consumer topics
      KAFKA_INPUT_TOPIC: dev.omninode.mvp.evt.stamp-created.v1
      KAFKA_OUTPUT_TOPIC: dev.omninode.mvp.evt.enrichment-completed.v1
      KAFKA_DLQ_TOPIC: dev.omninode.mvp.dlq.enrichment-failed.v1

      # OnexTree service
      ONEXTREE_URL: http://onextree:8058
      ONEXTREE_TIMEOUT_SEC: 10
      ONEXTREE_ENABLE_CACHE: "true"
      ONEXTREE_CACHE_TTL_SEC: 300

      # Enrichment configuration
      ENRICHMENT_ENABLE_TREE_CONTEXT: "true"
      ENRICHMENT_ENABLE_PATTERN_DETECTION: "true"
      ENRICHMENT_MAX_TREE_DEPTH: 5

      # Performance
      ENRICHMENT_BATCH_SIZE: 20
      ENRICHMENT_MAX_WORKERS: 5
      ENRICHMENT_ENABLE_METRICS: "true"
      ENRICHMENT_METRICS_PORT: 9093

    volumes:
      - ../logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9093/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.2'
          memory: 256M
    profiles:
      - mvp
      - production

  # =========================================================================
  # MONITORING SERVICES
  # =========================================================================

  # Prometheus - Metrics Aggregation
  prometheus:
    image: prom/prometheus:v2.48.0
    container_name: omninode-mvp-prometheus
    hostname: prometheus
    networks:
      - omninode-network
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro
      - prometheus-data:/prometheus
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.2'
          memory: 512M
    profiles:
      - monitoring
      - production

  # Alertmanager - Alert Routing
  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: omninode-mvp-alertmanager
    hostname: alertmanager
    networks:
      - omninode-network
    ports:
      - "9093:9093"
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager-data:/alertmanager
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.1'
          memory: 128M
    profiles:
      - monitoring
      - production

  # Grafana - Visualization Dashboards
  grafana:
    image: grafana/grafana:10.2.2
    container_name: omninode-mvp-grafana
    hostname: grafana
    networks:
      - omninode-network
    ports:
      - "3000:3000"
    environment:
      # Admin configuration
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}

      # Server configuration
      GF_SERVER_ROOT_URL: http://localhost:3000
      GF_INSTALL_PLUGINS: redis-datasource

      # Auth configuration
      GF_AUTH_ANONYMOUS_ENABLED: "false"

      # Alerting
      GF_UNIFIED_ALERTING_ENABLED: "true"

    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.2'
          memory: 256M
    profiles:
      - monitoring
      - production

  # =========================================================================
  # DEVELOPMENT TOOLS
  # =========================================================================

  # Redpanda UI - Kafka Management Interface
  redpanda-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: omninode-mvp-redpanda-ui
    networks:
      - omninode-network
    depends_on:
      redpanda:
        condition: service_healthy
    ports:
      - "${REDPANDA_UI_PORT:-3839}:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: omninode-mvp-redpanda
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: redpanda:9092
      KAFKA_CLUSTERS_0_SCHEMAREGISTRY: http://redpanda:8081
      SERVER_SERVLET_CONTEXT_PATH: /
      DYNAMIC_CONFIG_ENABLED: "true"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8080/actuator/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    profiles:
      - development
```

---

## 4. Service Configurations

### 4.1 Stamping Consumer Service

**Purpose**: Consumes events from hook-receiver and stamp-requested topics, processes them through metadata-stamping service.

**Dockerfile**: `docker/stamping-consumer/Dockerfile`

```dockerfile
# Multi-stage Dockerfile for stamping consumer service
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# System dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    git

# Install Poetry
ENV POETRY_VERSION=2.1.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Production stage
FROM base as production

# Copy Poetry configuration
COPY pyproject.toml poetry.lock ./

# Install production dependencies
RUN --mount=type=secret,id=gh_pat \
    --mount=type=cache,target=/tmp/poetry_cache \
    export GIT_TOKEN=$(cat /run/secrets/gh_pat 2>/dev/null || echo "") && \
    if [ -n "$GIT_TOKEN" ]; then \
        git config --global url."https://${GIT_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi && \
    poetry install --only=main && \
    if [ -n "$GIT_TOKEN" ]; then \
        git config --global --unset url."https://github.com/".insteadOf || true; \
    fi

# Copy source code
COPY src/ ./src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash consumer && \
    chown -R consumer:consumer /app && \
    mkdir -p /app/state /app/logs && \
    chown -R consumer:consumer /app/state /app/logs

USER consumer

EXPOSE 9092

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9092/health || exit 1

# Run consumer
CMD ["python", "-m", "omninode_bridge.consumers.stamping_consumer"]
```

**Key Features**:
- Batch processing (50 events per batch)
- Horizontal scaling (3 replicas)
- DLQ integration for failed events
- Circuit breaker for metadata-stamping API
- Prometheus metrics on port 9092

---

### 4.2 Intelligence Enrichment Service

**Purpose**: Consumes stamp-created events, enriches them with OnexTree intelligence, publishes enrichment-completed events.

**Dockerfile**: `docker/intelligence-enrichment/Dockerfile`

```dockerfile
# Multi-stage Dockerfile for intelligence enrichment service
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim as base

# System dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    git

# Install Poetry
ENV POETRY_VERSION=2.1.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Production stage
FROM base as production

# Copy Poetry configuration
COPY pyproject.toml poetry.lock ./

# Install production dependencies
RUN --mount=type=secret,id=gh_pat \
    --mount=type=cache,target=/tmp/poetry_cache \
    poetry install --only=main

# Copy source code
COPY src/ ./src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash enrichment && \
    chown -R enrichment:enrichment /app && \
    mkdir -p /app/logs && \
    chown -R enrichment:enrichment /app/logs

USER enrichment

EXPOSE 9093

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:9093/health || exit 1

# Run enrichment service
CMD ["python", "-m", "omninode_bridge.services.intelligence_enrichment"]
```

**Key Features**:
- OnexTree HTTP client with circuit breaker
- In-memory caching (TTL: 300s)
- Pattern detection capabilities
- Tree context enrichment (max depth: 5)
- Prometheus metrics on port 9093

---

## 5. Network Configuration

### Network Design

```yaml
networks:
  omninode-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1
    driver_opts:
      com.docker.network.bridge.name: br-omninode-mvp
      com.docker.network.bridge.enable_icc: "true"
      com.docker.network.bridge.enable_ip_masquerade: "true"
```

**Service Hostnames** (Internal DNS):
- `consul` (172.28.0.x)
- `postgres` (172.28.0.x)
- `redpanda` (172.28.0.x)
- `metadata-stamping` (172.28.0.x)
- `onextree` (172.28.0.x)
- `hook-receiver` (172.28.0.x)
- `stamping-consumer` (172.28.0.x)
- `intelligence-enrichment` (172.28.0.x)
- `prometheus` (172.28.0.x)
- `grafana` (172.28.0.x)

**Port Mapping Strategy**:
- Infrastructure: 28xxx (Consul: 28500, Redpanda: 29092)
- Application: 80xx (Metadata: 8057, OnexTree: 8058, Hook: 8001)
- Monitoring: 90xx (Prometheus: 9090, Grafana: 3000)
- Metrics: 90xx (Service metrics: 9091-9093)

---

## 6. Volume Configuration

### Volume Persistence Strategy

```yaml
volumes:
  # Infrastructure (Critical - Always persist)
  postgres-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/omninode-mvp/data/postgres

  redpanda-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/omninode-mvp/data/redpanda

  # Monitoring (Important - Persist in production)
  prometheus-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/omninode-mvp/data/prometheus

  grafana-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /opt/omninode-mvp/data/grafana

  # Cache (Optional - Can be ephemeral)
  onextree-cache:
    driver: local

  consumer-state:
    driver: local
```

**Volume Backup Priority**:
1. **Critical**: `postgres-data`, `redpanda-data`
2. **Important**: `prometheus-data`, `grafana-data`
3. **Optional**: `onextree-cache`, `consumer-state`

---

## 7. Environment Variables

### Complete `.env.mvp` File

```bash
# ========================================
# OmniNode MVP Environment Configuration
# ========================================

# Environment
ENVIRONMENT=production
LOG_LEVEL=info
SERVICE_VERSION=1.0.0

# ========================================
# Security
# ========================================
API_KEY=your-secure-api-key-change-in-production
WEBHOOK_SIGNING_SECRET=your-webhook-secret-change-in-production
GRAFANA_PASSWORD=your-grafana-password-change-in-production

# PostgreSQL
POSTGRES_PASSWORD=your-postgres-password-change-in-production

# ========================================
# Infrastructure Ports
# ========================================
CONSUL_PORT=28500
CONSUL_DNS_PORT=28600
POSTGRES_PORT=5436
REDPANDA_PORT=29092
REDPANDA_EXTERNAL_PORT=29102
REDPANDA_ADMIN_PORT=29654
REDPANDA_PROXY_PORT=28092
REDPANDA_UI_PORT=3839

# ========================================
# Redpanda Configuration
# ========================================
REDPANDA_LOG_LEVEL=info
REDPANDA_DEFAULT_PARTITIONS=3
REDPANDA_DEFAULT_REPLICATION_FACTOR=1
REDPANDA_LOG_RETENTION_MS=604800000
REDPANDA_SEGMENT_MS=86400000

# ========================================
# Docker Build Configuration
# ========================================
BUILD_TARGET=production

# ========================================
# Feature Flags
# ========================================
ENABLE_DEBUG_ENDPOINTS=false
RATE_LIMIT_ENABLED=true
RATE_LIMIT_RPM=100
RATE_LIMIT_BURST=20

# ========================================
# Consumer Configuration
# ========================================
KAFKA_GROUP_ID=stamping-consumer-mvp
KAFKA_AUTO_OFFSET_RESET=earliest
CONSUMER_BATCH_SIZE=50
CONSUMER_MAX_WORKERS=10

# ========================================
# Enrichment Configuration
# ========================================
ONEXTREE_ENABLE_CACHE=true
ONEXTREE_CACHE_TTL_SEC=300
ENRICHMENT_MAX_TREE_DEPTH=5

# ========================================
# Monitoring Configuration
# ========================================
PROMETHEUS_RETENTION_DAYS=30
ALERTMANAGER_SLACK_WEBHOOK_URL=
ALERTMANAGER_EMAIL_TO=

# ========================================
# GitHub Integration
# ========================================
GITHUB_TOKEN=your-github-token-for-private-repos
```

---

## 8. Startup Scripts

### 8.1 Complete Startup Script

**File**: `scripts/start-mvp.sh`

```bash
#!/bin/bash
# OmniNode MVP Stack Startup Script
# Usage: ./scripts/start-mvp.sh [--profile <profile>] [--build]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_ROOT}/deployment/docker-compose.mvp.yml"
ENV_FILE="${PROJECT_ROOT}/.env.mvp"

# Default values
PROFILE="production"
BUILD_FLAG=""
DETACHED="-d"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --foreground)
            DETACHED=""
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --profile <profile>   Profile to use (development, mvp, production, monitoring)"
            echo "  --build               Build images before starting"
            echo "  --foreground          Run in foreground (no -d flag)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose v2."
        exit 1
    fi

    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        log_warn ".env.mvp not found. Copying from .env.mvp.example..."
        cp "${ENV_FILE}.example" "$ENV_FILE"
        log_warn "Please edit .env.mvp with your configuration!"
        exit 1
    fi

    log_info "Prerequisites OK"
}

# Add hostname to /etc/hosts if needed
setup_hosts() {
    log_info "Checking /etc/hosts configuration..."

    if ! grep -q "omninode-mvp-redpanda" /etc/hosts; then
        log_warn "Adding redpanda hostname to /etc/hosts (requires sudo)..."
        echo "127.0.0.1 omninode-mvp-redpanda" | sudo tee -a /etc/hosts
    fi

    log_info "Hosts configuration OK"
}

# Start infrastructure services
start_infrastructure() {
    log_info "Starting infrastructure services (consul, postgres, redpanda)..."

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up $DETACHED $BUILD_FLAG \
        consul postgres redpanda redpanda-topic-manager

    log_info "Waiting for infrastructure health checks..."
    "${SCRIPT_DIR}/wait-for-services.sh" consul postgres redpanda
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm \
        metadata-stamping poetry run alembic upgrade head

    log_info "Migrations completed"
}

# Start application services
start_application() {
    log_info "Starting application services..."

    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile "$PROFILE" up $DETACHED $BUILD_FLAG \
        metadata-stamping onextree hook-receiver stamping-consumer intelligence-enrichment

    log_info "Application services started"
}

# Start monitoring services
start_monitoring() {
    if [[ "$PROFILE" == "monitoring" || "$PROFILE" == "production" ]]; then
        log_info "Starting monitoring services..."

        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile monitoring up $DETACHED \
            prometheus alertmanager grafana

        log_info "Monitoring services started"
    fi
}

# Display service URLs
show_urls() {
    log_info "=== OmniNode MVP Stack Started ==="
    echo ""
    echo "Application Services:"
    echo "  Metadata Stamping:    http://localhost:8057"
    echo "  OnexTree:             http://localhost:8058"
    echo "  Hook Receiver:        http://localhost:8001"
    echo ""
    echo "Infrastructure:"
    echo "  Consul UI:            http://localhost:28500"
    echo "  Redpanda UI:          http://localhost:3839"
    echo ""
    if [[ "$PROFILE" == "monitoring" || "$PROFILE" == "production" ]]; then
        echo "Monitoring:"
        echo "  Prometheus:           http://localhost:9090"
        echo "  Alertmanager:         http://localhost:9093"
        echo "  Grafana:              http://localhost:3000 (admin/${GRAFANA_PASSWORD:-admin})"
        echo ""
    fi
    echo "Use 'docker compose -f $COMPOSE_FILE logs -f' to view logs"
    echo "Use './scripts/stop-mvp.sh' to stop all services"
}

# Main execution
main() {
    log_info "Starting OmniNode MVP Stack (profile: $PROFILE)..."

    check_prerequisites
    setup_hosts
    start_infrastructure
    run_migrations
    start_application
    start_monitoring
    show_urls

    log_info "Startup complete!"
}

main
```

### 8.2 Shutdown Script

**File**: `scripts/stop-mvp.sh`

```bash
#!/bin/bash
# OmniNode MVP Stack Shutdown Script
# Usage: ./scripts/stop-mvp.sh [--cleanup]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_ROOT}/deployment/docker-compose.mvp.yml"
ENV_FILE="${PROJECT_ROOT}/.env.mvp"

CLEANUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --cleanup   Remove volumes (WARNING: deletes data)"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_info "Stopping OmniNode MVP Stack..."

# Stop all services
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" --profile mvp --profile production --profile monitoring down

# Cleanup volumes if requested
if [ "$CLEANUP" = true ]; then
    log_warn "Removing volumes (this will delete all data)..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down -v
fi

log_info "Stack stopped successfully"
```

---

## 9. Health Check Scripts

### 9.1 Wait for Services Script

**File**: `scripts/wait-for-services.sh`

```bash
#!/bin/bash
# Wait for services to become healthy
# Usage: ./scripts/wait-for-services.sh <service1> <service2> ...

set -e

# Default services if none specified
if [ $# -eq 0 ]; then
    SERVICES=("consul" "postgres" "redpanda")
else
    SERVICES=("$@")
fi

# Timeout configuration
TIMEOUT=300  # 5 minutes
INTERVAL=5   # Check every 5 seconds

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WAIT]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for specific service
wait_for_service() {
    local service=$1
    local elapsed=0

    log_info "Waiting for $service to become healthy..."

    while [ $elapsed -lt $TIMEOUT ]; do
        if docker compose -f deployment/docker-compose.mvp.yml ps | grep "$service" | grep -q "healthy"; then
            log_info "$service is healthy!"
            return 0
        fi

        log_warn "$service not ready yet (${elapsed}s/${TIMEOUT}s)..."
        sleep $INTERVAL
        elapsed=$((elapsed + INTERVAL))
    done

    log_error "$service failed to become healthy within ${TIMEOUT}s"
    return 1
}

# Main execution
main() {
    log_info "Waiting for ${#SERVICES[@]} service(s) to become healthy..."

    for service in "${SERVICES[@]}"; do
        if ! wait_for_service "$service"; then
            log_error "Service health check failed: $service"
            exit 1
        fi
    done

    log_info "All services are healthy!"
}

main
```

### 9.2 Health Check Verification Script

**File**: `scripts/verify-health.sh`

```bash
#!/bin/bash
# Verify health of all services
# Usage: ./scripts/verify-health.sh

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Health check endpoints
declare -A HEALTH_ENDPOINTS=(
    ["metadata-stamping"]="http://localhost:8057/health"
    ["onextree"]="http://localhost:8058/health"
    ["hook-receiver"]="http://localhost:8001/health"
    ["consul"]="http://localhost:28500/v1/health/node/consul"
)

# Check service health
check_health() {
    local service=$1
    local endpoint=$2

    if curl -f -s -o /dev/null "$endpoint"; then
        log_info "$service is healthy"
        return 0
    else
        log_error "$service is unhealthy"
        return 1
    fi
}

# Main execution
main() {
    echo "=== OmniNode MVP Health Check ==="
    echo ""

    failed=0

    for service in "${!HEALTH_ENDPOINTS[@]}"; do
        if ! check_health "$service" "${HEALTH_ENDPOINTS[$service]}"; then
            failed=$((failed + 1))
        fi
    done

    echo ""
    if [ $failed -eq 0 ]; then
        log_info "All services are healthy!"
        exit 0
    else
        log_error "$failed service(s) failed health check"
        exit 1
    fi
}

main
```

---

## 10. Development vs Production

### 10.1 Development Profile

```yaml
# Usage: docker compose --profile development up
profiles:
  development:
    # Features:
    # - Source code volume mounts for hot reload
    # - Debug logging enabled
    # - Development tools (Redpanda UI)
    # - Reduced resource limits
    # - Single replica for consumers
```

**Development Environment Variables**:
```bash
BUILD_TARGET=development
LOG_LEVEL=debug
ENABLE_DEBUG_ENDPOINTS=true
METADATA_STAMPING_SERVICE_RELOAD=true
```

### 10.2 Production Profile

```yaml
# Usage: docker compose --profile production up
profiles:
  production:
    # Features:
    # - Multi-stage production builds
    # - Optimized logging
    # - Resource limits enforced
    # - Multiple consumer replicas
    # - Monitoring stack enabled
    # - Read-only file systems
    # - Security hardening
```

**Production Environment Variables**:
```bash
BUILD_TARGET=production
LOG_LEVEL=info
ENABLE_DEBUG_ENDPOINTS=false
METADATA_STAMPING_SERVICE_RELOAD=false
```

### 10.3 Profile Comparison

| Feature | Development | Production |
|---------|-------------|------------|
| **Build Target** | development | production |
| **Hot Reload** | ✅ Enabled | ❌ Disabled |
| **Volume Mounts** | Source code | Logs only |
| **Logging** | DEBUG | INFO |
| **Consumer Replicas** | 1 | 3 |
| **Resource Limits** | Soft | Hard |
| **Monitoring** | Optional | Required |
| **Debug Endpoints** | Enabled | Disabled |
| **Redpanda UI** | Enabled | Disabled |

---

## 11. Resource Limits

### 11.1 Production Resource Configuration

```yaml
# Infrastructure Services
consul:
  limits: { cpus: '0.5', memory: 256M }
  reservations: { cpus: '0.1', memory: 128M }

postgres:
  limits: { cpus: '2.0', memory: 2G }
  reservations: { cpus: '0.5', memory: 512M }

redpanda:
  limits: { cpus: '2.0', memory: 2G }
  reservations: { cpus: '0.5', memory: 1G }

# Application Services
metadata-stamping:
  limits: { cpus: '2.0', memory: 2G }
  reservations: { cpus: '0.5', memory: 512M }

onextree:
  limits: { cpus: '1.0', memory: 512M }
  reservations: { cpus: '0.2', memory: 256M }

hook-receiver:
  limits: { cpus: '1.0', memory: 1G }
  reservations: { cpus: '0.2', memory: 256M }

stamping-consumer (per replica):
  limits: { cpus: '1.0', memory: 1G }
  reservations: { cpus: '0.2', memory: 256M }

intelligence-enrichment:
  limits: { cpus: '1.0', memory: 1G }
  reservations: { cpus: '0.2', memory: 256M }

# Monitoring Services
prometheus:
  limits: { cpus: '1.0', memory: 2G }
  reservations: { cpus: '0.2', memory: 512M }

grafana:
  limits: { cpus: '1.0', memory: 1G }
  reservations: { cpus: '0.2', memory: 256M }
```

### 11.2 Total Resource Requirements

**Minimum** (Development):
- CPU: 4 cores
- Memory: 8GB
- Disk: 20GB

**Recommended** (Production):
- CPU: 8+ cores
- Memory: 16GB
- Disk: 100GB (with monitoring retention)

---

## 12. Monitoring Configuration

### 12.1 Prometheus Configuration

**File**: `deployment/monitoring/prometheus.yml`

```yaml
# Prometheus configuration for OmniNode MVP
global:
  scrape_interval: 30s
  scrape_timeout: 10s
  evaluation_interval: 30s
  external_labels:
    cluster: 'omninode-mvp'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093
      timeout: 10s

# Alert rules
rule_files:
  - '/etc/prometheus/alerts.yml'

# Scrape configurations
scrape_configs:
  # Metadata Stamping Service
  - job_name: 'metadata-stamping'
    static_configs:
      - targets:
          - 'metadata-stamping:9090'
    metrics_path: '/metrics'
    scrape_interval: 15s

  # Stamping Consumer (3 replicas)
  - job_name: 'stamping-consumer'
    dns_sd_configs:
      - names:
          - 'stamping-consumer'
        type: 'A'
        port: 9092
    metrics_path: '/metrics'
    scrape_interval: 15s

  # Intelligence Enrichment
  - job_name: 'intelligence-enrichment'
    static_configs:
      - targets:
          - 'intelligence-enrichment:9093'
    metrics_path: '/metrics'
    scrape_interval: 15s

  # OnexTree Service
  - job_name: 'onextree'
    static_configs:
      - targets:
          - 'onextree:8058'
    metrics_path: '/metrics'
    scrape_interval: 30s

  # Hook Receiver
  - job_name: 'hook-receiver'
    static_configs:
      - targets:
          - 'hook-receiver:8001'
    metrics_path: '/metrics'
    scrape_interval: 30s

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets:
          - 'localhost:9090'
```

### 12.2 Alert Rules

**File**: `deployment/monitoring/alerts.yml`

```yaml
# Alert rules for OmniNode MVP
groups:
  - name: service_health
    interval: 30s
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.job }} has been down for more than 2 minutes"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on {{ $labels.job }}"
          description: "{{ $labels.job }} has error rate > 5% for 5 minutes"

  - name: consumer_health
    interval: 30s
    rules:
      - alert: ConsumerLagHigh
        expr: kafka_consumer_lag > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Consumer lag is high"
          description: "Consumer {{ $labels.group_id }} has lag > 1000 for 10 minutes"

      - alert: DLQMessagesPiling
        expr: increase(kafka_dlq_messages_total[1h]) > 100
        labels:
          severity: critical
        annotations:
          summary: "DLQ messages piling up"
          description: "More than 100 DLQ messages in the last hour"

  - name: performance
    interval: 30s
    rules:
      - alert: SlowHashGeneration
        expr: histogram_quantile(0.99, rate(blake3_hash_duration_seconds_bucket[5m])) > 0.002
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "BLAKE3 hash generation is slow"
          description: "P99 hash generation latency > 2ms for 10 minutes"

      - alert: DatabasePoolExhaustion
        expr: postgres_pool_usage_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool near exhaustion"
          description: "Pool usage > 90% for 5 minutes"
```

### 12.3 Alertmanager Configuration

**File**: `deployment/monitoring/alertmanager.yml`

```yaml
# Alertmanager configuration
global:
  resolve_timeout: 5m

  # Slack configuration (optional)
  slack_api_url: '${ALERTMANAGER_SLACK_WEBHOOK_URL}'

# Email configuration (optional)
#  smtp_smarthost: 'smtp.gmail.com:587'
#  smtp_from: 'alerts@omninode.dev'
#  smtp_auth_username: 'alerts@omninode.dev'
#  smtp_auth_password: '${SMTP_PASSWORD}'

# Route configuration
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    # Critical alerts
    - match:
        severity: critical
      receiver: 'critical'
      continue: true

    # Warning alerts
    - match:
        severity: warning
      receiver: 'warning'

# Receivers
receivers:
  - name: 'default'
    # Default receiver (logs only)

  - name: 'critical'
    slack_configs:
      - channel: '#omninode-alerts-critical'
        title: '🚨 Critical Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'

  - name: 'warning'
    slack_configs:
      - channel: '#omninode-alerts-warning'
        title: '⚠️ Warning Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'

# Inhibition rules
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

### 12.4 Grafana Dashboard

**File**: `deployment/monitoring/grafana/dashboards/mvp-overview.json`

```json
{
  "dashboard": {
    "title": "OmniNode MVP Overview",
    "tags": ["omninode", "mvp"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{job}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Consumer Lag",
        "type": "graph",
        "targets": [
          {
            "expr": "kafka_consumer_lag",
            "legendFormat": "{{group_id}} - {{topic}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "BLAKE3 Hash P99 Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, rate(blake3_hash_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ],
        "yaxis": {
          "format": "s"
        }
      },
      {
        "id": 4,
        "title": "Database Pool Usage",
        "type": "gauge",
        "targets": [
          {
            "expr": "postgres_pool_usage_percent",
            "legendFormat": "Pool Usage %"
          }
        ]
      },
      {
        "id": 5,
        "title": "DLQ Messages",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(kafka_dlq_messages_total)",
            "legendFormat": "Total DLQ"
          }
        ]
      },
      {
        "id": 6,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "{{job}} - 5xx"
          }
        ]
      }
    ],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

---

## 13. Backup & Recovery

### 13.1 Backup Script

**File**: `scripts/backup.sh`

```bash
#!/bin/bash
# Backup OmniNode MVP data
# Usage: ./scripts/backup.sh [--destination <path>]

set -e

# Configuration
BACKUP_DIR=${1:-"/opt/omninode-mvp/backups/$(date +%Y%m%d_%H%M%S)"}
COMPOSE_FILE="deployment/docker-compose.mvp.yml"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log_info "Starting backup to $BACKUP_DIR"

# Backup PostgreSQL
log_info "Backing up PostgreSQL database..."
docker compose -f "$COMPOSE_FILE" exec -T postgres \
    pg_dump -U postgres -d omninode_bridge \
    | gzip > "$BACKUP_DIR/postgres_omninode_bridge.sql.gz"

# Backup Redpanda topics
log_info "Backing up Redpanda topic data..."
docker compose -f "$COMPOSE_FILE" exec -T redpanda \
    rpk topic list --brokers localhost:9092 \
    > "$BACKUP_DIR/redpanda_topics.txt"

# Backup volumes
log_info "Backing up Docker volumes..."
docker run --rm \
    -v omninode-mvp_postgres-data:/source:ro \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf /backup/postgres-data.tar.gz -C /source .

docker run --rm \
    -v omninode-mvp_redpanda-data:/source:ro \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf /backup/redpanda-data.tar.gz -C /source .

# Backup Prometheus data
log_info "Backing up Prometheus data..."
docker run --rm \
    -v omninode-mvp_prometheus-data:/source:ro \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf /backup/prometheus-data.tar.gz -C /source .

# Backup Grafana data
log_info "Backing up Grafana data..."
docker run --rm \
    -v omninode-mvp_grafana-data:/source:ro \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf /backup/grafana-data.tar.gz -C /source .

# Backup configuration
log_info "Backing up configuration files..."
cp .env.mvp "$BACKUP_DIR/.env.mvp.backup"
cp "$COMPOSE_FILE" "$BACKUP_DIR/docker-compose.mvp.yml.backup"

# Create backup manifest
log_info "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest.txt" <<EOF
OmniNode MVP Backup Manifest
===========================
Backup Date: $(date)
Backup Location: $BACKUP_DIR

Contents:
- postgres_omninode_bridge.sql.gz (PostgreSQL dump)
- postgres-data.tar.gz (PostgreSQL volume)
- redpanda-data.tar.gz (Redpanda volume)
- redpanda_topics.txt (Topic list)
- prometheus-data.tar.gz (Prometheus volume)
- grafana-data.tar.gz (Grafana volume)
- .env.mvp.backup (Environment configuration)
- docker-compose.mvp.yml.backup (Docker Compose configuration)

Restore with: ./scripts/restore.sh $BACKUP_DIR
EOF

# Display summary
log_info "Backup completed successfully!"
log_info "Backup location: $BACKUP_DIR"
log_info "Backup size: $(du -sh "$BACKUP_DIR" | cut -f1)"
```

### 13.2 Restore Script

**File**: `scripts/restore.sh`

```bash
#!/bin/bash
# Restore OmniNode MVP from backup
# Usage: ./scripts/restore.sh <backup-directory>

set -e

# Configuration
BACKUP_DIR=$1
COMPOSE_FILE="deployment/docker-compose.mvp.yml"

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup-directory>"
    exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Confirmation
log_warn "This will restore from backup: $BACKUP_DIR"
log_warn "Current data will be REPLACED!"
read -p "Continue? (yes/no): " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Restore cancelled"
    exit 1
fi

# Stop services
log_info "Stopping services..."
./scripts/stop-mvp.sh

# Restore PostgreSQL
log_info "Restoring PostgreSQL database..."
docker compose -f "$COMPOSE_FILE" up -d postgres
sleep 10

gunzip -c "$BACKUP_DIR/postgres_omninode_bridge.sql.gz" | \
    docker compose -f "$COMPOSE_FILE" exec -T postgres \
    psql -U postgres -d omninode_bridge

# Restore volumes
log_info "Restoring Docker volumes..."

docker run --rm \
    -v omninode-mvp_postgres-data:/target \
    -v "$BACKUP_DIR":/backup \
    alpine tar xzf /backup/postgres-data.tar.gz -C /target

docker run --rm \
    -v omninode-mvp_redpanda-data:/target \
    -v "$BACKUP_DIR":/backup \
    alpine tar xzf /backup/redpanda-data.tar.gz -C /target

docker run --rm \
    -v omninode-mvp_prometheus-data:/target \
    -v "$BACKUP_DIR":/backup \
    alpine tar xzf /backup/prometheus-data.tar.gz -C /target

docker run --rm \
    -v omninode-mvp_grafana-data:/target \
    -v "$BACKUP_DIR":/backup \
    alpine tar xzf /backup/grafana-data.tar.gz -C /target

# Restore configuration (with confirmation)
if [ -f "$BACKUP_DIR/.env.mvp.backup" ]; then
    log_warn "Restore .env.mvp configuration?"
    read -p "Continue? (yes/no): " -r
    if [[ $REPLY =~ ^yes$ ]]; then
        cp "$BACKUP_DIR/.env.mvp.backup" .env.mvp
    fi
fi

log_info "Restore completed!"
log_info "Start services with: ./scripts/start-mvp.sh"
```

---

## 14. Testing Stack

### 14.1 Test Docker Compose

**File**: `deployment/docker-compose.test.yml`

```yaml
# Test environment for OmniNode MVP
# Isolated from production services
version: '3.9'

networks:
  test-network:
    driver: bridge

services:
  # Test PostgreSQL
  test-postgres:
    image: pgvector/pgvector:pg15
    container_name: omninode-test-postgres
    networks:
      - test-network
    environment:
      POSTGRES_DB: omninode_bridge_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5437:5432"
    tmpfs:
      - /var/lib/postgresql/data

  # Test Redpanda
  test-redpanda:
    image: redpandadata/redpanda:v24.2.7
    container_name: omninode-test-redpanda
    networks:
      - test-network
    command:
      - redpanda
      - start
      - --kafka-addr=internal://0.0.0.0:9092
      - --advertise-kafka-addr=internal://test-redpanda:9092
      - --mode=dev-container
      - --smp=1
    ports:
      - "29093:9092"
    tmpfs:
      - /var/lib/redpanda/data

  # Test metadata stamping service
  test-metadata-stamping:
    build:
      context: ..
      dockerfile: docker/metadata-stamping/Dockerfile
      target: development
    container_name: omninode-test-metadata-stamping
    networks:
      - test-network
    depends_on:
      - test-postgres
      - test-redpanda
    environment:
      METADATA_STAMPING_DB_HOST: test-postgres
      METADATA_STAMPING_DB_PORT: 5432
      METADATA_STAMPING_DB_NAME: omninode_bridge_test
      METADATA_STAMPING_DB_USER: test_user
      METADATA_STAMPING_DB_PASSWORD: test_password
      METADATA_STAMPING_KAFKA_BOOTSTRAP_SERVERS: test-redpanda:9092
      METADATA_STAMPING_LOG_LEVEL: debug
    ports:
      - "8059:8053"
```

### 14.2 Integration Test Script

**File**: `scripts/run-integration-tests.sh`

```bash
#!/bin/bash
# Run integration tests against test stack
# Usage: ./scripts/run-integration-tests.sh

set -e

COMPOSE_FILE="deployment/docker-compose.test.yml"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Start test stack
log_info "Starting test stack..."
docker compose -f "$COMPOSE_FILE" up -d

# Wait for services
log_info "Waiting for services..."
sleep 10

# Run migrations
log_info "Running test migrations..."
docker compose -f "$COMPOSE_FILE" exec -T test-metadata-stamping \
    poetry run alembic upgrade head

# Run integration tests
log_info "Running integration tests..."
poetry run pytest tests/integration/ -v --tb=short

# Cleanup
log_info "Cleaning up test stack..."
docker compose -f "$COMPOSE_FILE" down -v

log_info "Integration tests completed!"
```

---

## 15. Integration Steps

### Complete Integration Checklist

#### Phase 1: Infrastructure Setup
- [ ] Copy `docker-compose.mvp.yml` to `deployment/`
- [ ] Copy `.env.mvp.example` to `.env.mvp`
- [ ] Edit `.env.mvp` with production values
- [ ] Create monitoring configuration files
- [ ] Create backup directories

#### Phase 2: Docker Image Preparation
- [ ] Create `docker/stamping-consumer/Dockerfile`
- [ ] Create `docker/intelligence-enrichment/Dockerfile`
- [ ] Update `pyproject.toml` with consumer dependencies
- [ ] Build all images: `./scripts/build-images.sh`

#### Phase 3: Service Implementation
- [ ] Implement `src/omninode_bridge/consumers/stamping_consumer.py`
- [ ] Implement `src/omninode_bridge/services/intelligence_enrichment/`
- [ ] Add health check endpoints to new services
- [ ] Add Prometheus metrics to new services

#### Phase 4: Database Migrations
- [ ] Create migration for consumer state table
- [ ] Create migration for enrichment cache table
- [ ] Test migrations on test database

#### Phase 5: Testing
- [ ] Unit tests for consumer logic
- [ ] Unit tests for enrichment service
- [ ] Integration tests with test stack
- [ ] Load tests with production-like data

#### Phase 6: Deployment
- [ ] Run startup script: `./scripts/start-mvp.sh --profile production --build`
- [ ] Verify all services healthy
- [ ] Run smoke tests
- [ ] Configure monitoring alerts
- [ ] Set up backup cron jobs

#### Phase 7: Validation
- [ ] Send test webhook to hook-receiver
- [ ] Verify event flows through Kafka
- [ ] Check DLQ for any failures
- [ ] Validate Grafana dashboards
- [ ] Test backup and restore procedures

---

## Summary

This deployment configuration provides:

✅ **Complete MVP Stack** - All services for automated stamping workflow
✅ **Production-Ready** - Resource limits, health checks, monitoring
✅ **Scalable** - Horizontal scaling for consumers
✅ **Observable** - Prometheus + Grafana monitoring
✅ **Resilient** - DLQ handling, circuit breakers, retries
✅ **Maintainable** - Clear separation of concerns, profiles
✅ **Tested** - Integration test stack included

**Next Steps**:
1. Implement stamping consumer service
2. Implement intelligence enrichment service
3. Create Dockerfiles for new services
4. Test deployment with scripts
5. Configure production monitoring
6. Set up automated backups

---

**Document Version**: 1.0.0
**Last Updated**: October 2025
**Maintainer**: OmniNode Bridge Team
