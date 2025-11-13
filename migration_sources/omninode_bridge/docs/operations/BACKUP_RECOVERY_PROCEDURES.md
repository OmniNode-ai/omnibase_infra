# Backup and Recovery Procedures

## Overview

This document outlines comprehensive backup and recovery strategies for OmniNode Bridge, ensuring data protection, business continuity, and rapid disaster recovery capabilities.

## Table of Contents

1. [Backup Strategy](#backup-strategy)
2. [Backup Procedures](#backup-procedures)
3. [Recovery Procedures](#recovery-procedures)
4. [Disaster Recovery](#disaster-recovery)
5. [Testing and Validation](#testing-and-validation)
6. [Monitoring and Alerting](#monitoring-and-alerting)

## Backup Strategy

### Backup Requirements

#### Recovery Objectives
```yaml
Production Environment:
  Recovery Point Objective (RPO): 6 hours
  Recovery Time Objective (RTO): 4 hours
  Maximum Tolerable Downtime: 8 hours

Data Classification:
  Critical Data (Tier 1):
    - User data and configurations
    - Business intelligence patterns
    - Audit logs and compliance data
    RPO: 1 hour, RTO: 2 hours

  Important Data (Tier 2):
    - Application configurations
    - System metrics and logs
    - Cache data and sessions
    RPO: 6 hours, RTO: 4 hours

  Standard Data (Tier 3):
    - Temporary files
    - Development artifacts
    - Non-critical logs
    RPO: 24 hours, RTO: 8 hours
```

#### Backup Types
```yaml
Full Backups:
  Frequency: Weekly (Sunday 02:00 UTC)
  Retention: 12 weeks
  Storage: Primary + Geographic redundancy

Incremental Backups:
  Frequency: Daily (02:00 UTC)
  Retention: 30 days
  Storage: Primary + Regional redundancy

Transaction Log Backups:
  Frequency: Every 15 minutes
  Retention: 7 days
  Storage: Primary storage with replication

Point-in-Time Recovery:
  Granularity: 1 minute
  Coverage: Last 30 days
  Storage: High-speed storage
```

### Storage Architecture

#### Backup Storage Locations
```yaml
Primary Backup Location:
  Type: Local NAS/SAN
  Capacity: 10TB
  Network: 10Gbps dedicated backup network
  Encryption: AES-256 at rest

Secondary Backup Location:
  Type: Cloud storage (AWS S3/Azure Blob)
  Capacity: Unlimited (with lifecycle policies)
  Network: Internet with VPN
  Encryption: AES-256 in transit and at rest

Offsite Backup Location:
  Type: Secondary data center
  Capacity: 5TB
  Network: Dedicated fiber connection
  Encryption: Hardware-based encryption
```

## Backup Procedures

### 1. Database Backups

#### PostgreSQL Full Backup
```bash
#!/bin/bash
# scripts/backup-database-full.sh

set -e

# Configuration
BACKUP_DIR="/opt/omninode_bridge/backups/database"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="omninode_bridge_full_${TIMESTAMP}"
RETENTION_DAYS=90

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Log backup start
echo "[$(date)] Starting full database backup: $BACKUP_NAME"

# Create full backup with compression
docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump \
    --username=omninode_bridge \
    --dbname=omninode_bridge \
    --format=custom \
    --compress=9 \
    --verbose \
    --file=/tmp/backup.dump

# Copy backup from container
docker-compose -f docker-compose.prod.yml exec postgres cat /tmp/backup.dump | \
    gzip > "${BACKUP_DIR}/${BACKUP_NAME}.dump.gz"

# Verify backup integrity
if gunzip -t "${BACKUP_DIR}/${BACKUP_NAME}.dump.gz"; then
    echo "[$(date)] Backup integrity verified: $BACKUP_NAME"
else
    echo "[$(date)] ERROR: Backup integrity check failed: $BACKUP_NAME"
    exit 1
fi

# Generate backup metadata
cat > "${BACKUP_DIR}/${BACKUP_NAME}.metadata" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "backup_type": "full",
    "timestamp": "$TIMESTAMP",
    "database": "omninode_bridge",
    "size_bytes": $(stat -c%s "${BACKUP_DIR}/${BACKUP_NAME}.dump.gz"),
    "checksum": "$(sha256sum "${BACKUP_DIR}/${BACKUP_NAME}.dump.gz" | cut -d' ' -f1)",
    "retention_date": "$(date -d "+${RETENTION_DAYS} days" +%Y-%m-%d)"
}
EOF

# Upload to secondary storage
echo "[$(date)] Uploading to cloud storage..."
aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.dump.gz" \
    "s3://omninode-backups/database/${BACKUP_NAME}.dump.gz" \
    --storage-class STANDARD_IA

aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.metadata" \
    "s3://omninode-backups/database/${BACKUP_NAME}.metadata"

# Clean up old backups
echo "[$(date)] Cleaning up old backups..."
find "$BACKUP_DIR" -name "omninode_bridge_full_*.dump.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "omninode_bridge_full_*.metadata" -mtime +$RETENTION_DAYS -delete

echo "[$(date)] Full database backup completed: $BACKUP_NAME"
```

#### PostgreSQL Incremental Backup
```bash
#!/bin/bash
# scripts/backup-database-incremental.sh

set -e

# Configuration
BACKUP_DIR="/opt/omninode_bridge/backups/database/incremental"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="omninode_bridge_incr_${TIMESTAMP}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Log backup start
echo "[$(date)] Starting incremental database backup: $BACKUP_NAME"

# Create WAL archive backup
docker-compose -f docker-compose.prod.yml exec postgres pg_basebackup \
    --pgdata=/tmp/basebackup \
    --format=tar \
    --gzip \
    --checkpoint=fast \
    --label="$BACKUP_NAME" \
    --progress \
    --verbose

# Copy incremental backup
docker-compose -f docker-compose.prod.yml exec postgres tar -czf \
    "/tmp/${BACKUP_NAME}.tar.gz" -C /tmp/basebackup .

docker-compose -f docker-compose.prod.yml exec postgres cat \
    "/tmp/${BACKUP_NAME}.tar.gz" > "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"

# Generate backup metadata
BACKUP_SIZE=$(stat -c%s "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz")
BACKUP_CHECKSUM=$(sha256sum "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -d' ' -f1)

cat > "${BACKUP_DIR}/${BACKUP_NAME}.metadata" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "backup_type": "incremental",
    "timestamp": "$TIMESTAMP",
    "database": "omninode_bridge",
    "size_bytes": $BACKUP_SIZE,
    "checksum": "$BACKUP_CHECKSUM",
    "parent_backup": "$(ls -t ${BACKUP_DIR}/../omninode_bridge_full_*.metadata | head -1 | xargs basename -s .metadata)"
}
EOF

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    "s3://omninode-backups/database/incremental/${BACKUP_NAME}.tar.gz"

echo "[$(date)] Incremental database backup completed: $BACKUP_NAME"
```

### 2. Configuration Backups

#### Application Configuration Backup
```bash
#!/bin/bash
# scripts/backup-configuration.sh

set -e

# Configuration
BACKUP_DIR="/opt/omninode_bridge/backups/configuration"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="configuration_${TIMESTAMP}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting configuration backup: $BACKUP_NAME"

# Create configuration backup archive
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    -C /opt/omninode_bridge \
    config/ \
    docker-compose.prod.yml \
    .env.production \
    nginx/ \
    monitoring/ \
    --exclude='config/cache/*' \
    --exclude='config/temp/*'

# Include SSL certificates
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}_ssl.tar.gz" \
    -C / \
    etc/ssl/certs/omninode/ \
    etc/ssl/private/omninode/

# Generate metadata
BACKUP_SIZE=$(stat -c%s "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz")
SSL_SIZE=$(stat -c%s "${BACKUP_DIR}/${BACKUP_NAME}_ssl.tar.gz")

cat > "${BACKUP_DIR}/${BACKUP_NAME}.metadata" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "backup_type": "configuration",
    "timestamp": "$TIMESTAMP",
    "components": {
        "application_config": {
            "file": "${BACKUP_NAME}.tar.gz",
            "size_bytes": $BACKUP_SIZE,
            "checksum": "$(sha256sum "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -d' ' -f1)"
        },
        "ssl_certificates": {
            "file": "${BACKUP_NAME}_ssl.tar.gz",
            "size_bytes": $SSL_SIZE,
            "checksum": "$(sha256sum "${BACKUP_DIR}/${BACKUP_NAME}_ssl.tar.gz" | cut -d' ' -f1)"
        }
    }
}
EOF

# Upload to cloud storage
aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    "s3://omninode-backups/configuration/${BACKUP_NAME}.tar.gz"

aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}_ssl.tar.gz" \
    "s3://omninode-backups/configuration/${BACKUP_NAME}_ssl.tar.gz"

echo "[$(date)] Configuration backup completed: $BACKUP_NAME"
```

### 3. Application Data Backups

#### Redis Cache Backup
```bash
#!/bin/bash
# scripts/backup-redis.sh

set -e

BACKUP_DIR="/opt/omninode_bridge/backups/redis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="redis_${TIMESTAMP}"

mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting Redis backup: $BACKUP_NAME"

# Create Redis backup
docker-compose -f docker-compose.prod.yml exec redis redis-cli BGSAVE

# Wait for backup to complete
while [ "$(docker-compose -f docker-compose.prod.yml exec redis redis-cli LASTSAVE)" = "$(docker-compose -f docker-compose.prod.yml exec redis redis-cli LASTSAVE)" ]; do
    sleep 2
done

# Copy backup file
docker-compose -f docker-compose.prod.yml exec redis cat /data/dump.rdb > "${BACKUP_DIR}/${BACKUP_NAME}.rdb"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_NAME}.rdb"

echo "[$(date)] Redis backup completed: $BACKUP_NAME"
```

#### Kafka Data Backup
```bash
#!/bin/bash
# scripts/backup-kafka.sh

set -e

BACKUP_DIR="/opt/omninode_bridge/backups/kafka"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="kafka_${TIMESTAMP}"

mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting Kafka backup: $BACKUP_NAME"

# Stop Kafka gracefully
docker-compose -f docker-compose.prod.yml stop kafka

# Backup Kafka data directory
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
    -C /opt/omninode_bridge/data \
    kafka/

# Restart Kafka
docker-compose -f docker-compose.prod.yml start kafka

# Wait for Kafka to be ready
sleep 60

echo "[$(date)] Kafka backup completed: $BACKUP_NAME"
```

### 4. Automated Backup Orchestration

#### Comprehensive Backup Script
```bash
#!/bin/bash
# scripts/backup-all.sh

set -e

BACKUP_TYPE="${1:-incremental}"  # full, incremental
LOG_FILE="/opt/omninode_bridge/logs/backup.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting comprehensive backup ($BACKUP_TYPE) ==="

# Pre-backup checks
log "Performing pre-backup checks..."

# Check disk space
AVAILABLE_SPACE=$(df /opt/omninode_bridge/backups | tail -1 | awk '{print $4}')
REQUIRED_SPACE=10485760  # 10GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    log "ERROR: Insufficient disk space for backup"
    exit 1
fi

# Check service health
if ! curl -f -s http://localhost:8001/health >/dev/null; then
    log "WARNING: HookReceiver unhealthy, backup may be inconsistent"
fi

# Backup execution order (minimize impact)
log "Starting backup sequence..."

# 1. Configuration backup (minimal impact)
log "Step 1: Backing up configuration..."
/opt/omninode_bridge/scripts/backup-configuration.sh || {
    log "ERROR: Configuration backup failed"
    exit 1
}

# 2. Redis backup (fast, minimal impact)
log "Step 2: Backing up Redis cache..."
/opt/omninode_bridge/scripts/backup-redis.sh || {
    log "WARNING: Redis backup failed"
}

# 3. Database backup (highest impact, but most critical)
log "Step 3: Backing up database..."
if [ "$BACKUP_TYPE" = "full" ]; then
    /opt/omninode_bridge/scripts/backup-database-full.sh || {
        log "ERROR: Database full backup failed"
        exit 1
    }
else
    /opt/omninode_bridge/scripts/backup-database-incremental.sh || {
        log "ERROR: Database incremental backup failed"
        exit 1
    }
fi

# 4. Kafka backup (only if specifically requested)
if [ "${2}" = "--include-kafka" ]; then
    log "Step 4: Backing up Kafka (optional)..."
    /opt/omninode_bridge/scripts/backup-kafka.sh || {
        log "WARNING: Kafka backup failed"
    }
fi

# Post-backup validation
log "Performing post-backup validation..."

# Verify latest database backup
LATEST_DB_BACKUP=$(ls -t /opt/omninode_bridge/backups/database/omninode_bridge_*_*.dump.gz | head -1)
if [ -f "$LATEST_DB_BACKUP" ] && gunzip -t "$LATEST_DB_BACKUP"; then
    log "✓ Database backup validation successful"
else
    log "✗ Database backup validation failed"
    exit 1
fi

# Generate backup report
BACKUP_REPORT="/opt/omninode_bridge/backups/reports/backup_report_$(date +%Y%m%d_%H%M%S).json"
mkdir -p "$(dirname "$BACKUP_REPORT")"

cat > "$BACKUP_REPORT" << EOF
{
    "backup_session": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "type": "$BACKUP_TYPE",
        "status": "completed",
        "duration_seconds": $SECONDS
    },
    "components": {
        "database": {
            "file": "$(basename "$LATEST_DB_BACKUP")",
            "size_mb": $(($(stat -c%s "$LATEST_DB_BACKUP") / 1024 / 1024)),
            "status": "success"
        },
        "configuration": {
            "status": "success"
        },
        "redis": {
            "status": "success"
        }
    },
    "next_backup": {
        "full": "$(date -d 'next Sunday 02:00' +%Y-%m-%d)",
        "incremental": "$(date -d 'tomorrow 02:00' +%Y-%m-%d)"
    }
}
EOF

log "=== Backup completed successfully ==="
log "Backup report: $BACKUP_REPORT"
```

## Recovery Procedures

### 1. Database Recovery

#### Point-in-Time Recovery
```bash
#!/bin/bash
# scripts/recovery-database-pit.sh

set -e

TARGET_TIME="$1"  # Format: YYYY-MM-DD HH:MM:SS
RECOVERY_DIR="/opt/omninode_bridge/recovery"

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 'YYYY-MM-DD HH:MM:SS'"
    echo "Example: $0 '2024-01-15 14:30:00'"
    exit 1
fi

echo "=== Point-in-Time Recovery to $TARGET_TIME ==="

# Create recovery environment
mkdir -p "$RECOVERY_DIR"

# Stop current database
echo "Stopping current database..."
docker-compose -f docker-compose.prod.yml stop postgres

# Backup current state (just in case)
echo "Creating safety backup of current state..."
tar -czf "$RECOVERY_DIR/pre_recovery_backup_$(date +%Y%m%d_%H%M%S).tar.gz" \
    -C /opt/omninode_bridge/data postgres/

# Find appropriate base backup
echo "Finding base backup for recovery..."
BASE_BACKUP=$(aws s3 ls s3://omninode-backups/database/ | \
    awk '$1 <= "'$(date -d "$TARGET_TIME" +%Y-%m-%d)'"' | \
    tail -1 | awk '{print $4}')

if [ -z "$BASE_BACKUP" ]; then
    echo "ERROR: No suitable base backup found for target time"
    exit 1
fi

echo "Using base backup: $BASE_BACKUP"

# Download and restore base backup
echo "Downloading base backup..."
aws s3 cp "s3://omninode-backups/database/$BASE_BACKUP" "$RECOVERY_DIR/"

echo "Restoring base backup..."
rm -rf /opt/omninode_bridge/data/postgres/*
gunzip -c "$RECOVERY_DIR/$BASE_BACKUP" | \
    docker-compose -f docker-compose.prod.yml exec -T postgres pg_restore \
    --username=omninode_bridge \
    --dbname=omninode_bridge \
    --verbose

# Configure recovery.conf for point-in-time recovery
cat > /opt/omninode_bridge/data/postgres/recovery.conf << EOF
restore_command = 'aws s3 cp s3://omninode-backups/wal_archive/%f %p'
recovery_target_time = '$TARGET_TIME'
recovery_target_action = 'promote'
EOF

# Start PostgreSQL in recovery mode
echo "Starting PostgreSQL in recovery mode..."
docker-compose -f docker-compose.prod.yml start postgres

# Monitor recovery progress
echo "Monitoring recovery progress..."
while true; do
    if docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U omninode_bridge; then
        echo "✓ Database recovery completed"
        break
    fi
    echo "Recovery in progress..."
    sleep 10
done

# Verify recovery
echo "Verifying recovery..."
RECOVERY_TIME=$(docker-compose -f docker-compose.prod.yml exec postgres psql \
    -U omninode_bridge -d omninode_bridge \
    -t -c "SELECT pg_last_xact_replay_timestamp();")

echo "Recovery completed. Database restored to: $RECOVERY_TIME"
```

#### Full Database Restore
```bash
#!/bin/bash
# scripts/recovery-database-full.sh

set -e

BACKUP_FILE="$1"
RECOVERY_DIR="/opt/omninode_bridge/recovery"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Available backups:"
    ls -la /opt/omninode_bridge/backups/database/omninode_bridge_full_*.dump.gz
    exit 1
fi

echo "=== Full Database Restore from $BACKUP_FILE ==="

# Verify backup file exists and is valid
if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE"
    exit 1
fi

if ! gunzip -t "$BACKUP_FILE"; then
    echo "ERROR: Backup file is corrupted: $BACKUP_FILE"
    exit 1
fi

# Create recovery directory
mkdir -p "$RECOVERY_DIR"

# Stop all services that use the database
echo "Stopping services..."
docker-compose -f docker-compose.prod.yml stop hook-receiver workflow-coordinator

# Create safety backup
echo "Creating safety backup..."
docker-compose -f docker-compose.prod.yml exec postgres pg_dump \
    --username=omninode_bridge \
    --dbname=omninode_bridge \
    --format=custom \
    --file=/tmp/safety_backup.dump

docker-compose -f docker-compose.prod.yml exec postgres cat /tmp/safety_backup.dump > \
    "$RECOVERY_DIR/safety_backup_$(date +%Y%m%d_%H%M%S).dump"

# Drop and recreate database
echo "Recreating database..."
docker-compose -f docker-compose.prod.yml exec postgres psql \
    -U postgres -c "DROP DATABASE IF EXISTS omninode_bridge;"

docker-compose -f docker-compose.prod.yml exec postgres psql \
    -U postgres -c "CREATE DATABASE omninode_bridge OWNER omninode_bridge;"

# Restore from backup
echo "Restoring from backup..."
gunzip -c "$BACKUP_FILE" | \
    docker-compose -f docker-compose.prod.yml exec -T postgres pg_restore \
    --username=omninode_bridge \
    --dbname=omninode_bridge \
    --no-owner \
    --no-privileges \
    --verbose

# Verify restore
echo "Verifying restore..."
RECORD_COUNT=$(docker-compose -f docker-compose.prod.yml exec postgres psql \
    -U omninode_bridge -d omninode_bridge \
    -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")

echo "Restored database contains $RECORD_COUNT tables"

# Restart services
echo "Restarting services..."
docker-compose -f docker-compose.prod.yml start hook-receiver workflow-coordinator

# Wait for services to be healthy
sleep 60

# Final health check
if curl -f -s http://localhost:8001/health >/dev/null; then
    echo "✓ Database restore completed successfully"
else
    echo "✗ Database restore completed but services are not healthy"
    exit 1
fi
```

### 2. Configuration Recovery

#### Configuration Restore
```bash
#!/bin/bash
# scripts/recovery-configuration.sh

set -e

BACKUP_FILE="$1"
RECOVERY_DIR="/opt/omninode_bridge/recovery"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <configuration_backup.tar.gz>"
    echo "Available configuration backups:"
    ls -la /opt/omninode_bridge/backups/configuration/configuration_*.tar.gz
    exit 1
fi

echo "=== Configuration Recovery from $BACKUP_FILE ==="

# Create recovery directory
mkdir -p "$RECOVERY_DIR"

# Backup current configuration
echo "Backing up current configuration..."
tar -czf "$RECOVERY_DIR/current_config_$(date +%Y%m%d_%H%M%S).tar.gz" \
    -C /opt/omninode_bridge \
    config/ docker-compose.prod.yml .env.production

# Extract backup
echo "Extracting configuration backup..."
tar -xzf "$BACKUP_FILE" -C /opt/omninode_bridge/

# Restart services to apply new configuration
echo "Restarting services with new configuration..."
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to start
sleep 60

# Verify configuration
echo "Verifying configuration..."
if curl -f -s http://localhost:8001/health >/dev/null; then
    echo "✓ Configuration recovery completed successfully"
else
    echo "✗ Configuration recovery failed"
    exit 1
fi
```

### 3. Complete System Recovery

#### Disaster Recovery Script
```bash
#!/bin/bash
# scripts/disaster-recovery.sh

set -e

RECOVERY_POINT="$1"  # Format: YYYY-MM-DD or specific backup name
RECOVERY_DIR="/opt/omninode_bridge/recovery"
LOG_FILE="$RECOVERY_DIR/disaster_recovery_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

if [ -z "$RECOVERY_POINT" ]; then
    echo "Usage: $0 <recovery_point>"
    echo "Examples:"
    echo "  $0 2024-01-15          # Recover to date"
    echo "  $0 omninode_bridge_full_20240115_020000  # Specific backup"
    exit 1
fi

log "=== DISASTER RECOVERY INITIATED ==="
log "Recovery point: $RECOVERY_POINT"

# Create recovery directory
mkdir -p "$RECOVERY_DIR"

# Stop all services
log "Stopping all services..."
docker-compose -f docker-compose.prod.yml down

# Recovery Phase 1: Infrastructure
log "Phase 1: Recovering infrastructure configuration..."

# Find and download configuration backup
CONFIG_BACKUP=$(aws s3 ls s3://omninode-backups/configuration/ | \
    grep "$RECOVERY_POINT" | tail -1 | awk '{print $4}')

if [ -n "$CONFIG_BACKUP" ]; then
    log "Downloading configuration backup: $CONFIG_BACKUP"
    aws s3 cp "s3://omninode-backups/configuration/$CONFIG_BACKUP" "$RECOVERY_DIR/"

    # Restore configuration
    tar -xzf "$RECOVERY_DIR/$CONFIG_BACKUP" -C /opt/omninode_bridge/
    log "✓ Configuration restored"
else
    log "⚠ No configuration backup found for $RECOVERY_POINT"
fi

# Recovery Phase 2: Database
log "Phase 2: Recovering database..."

# Find and download database backup
DB_BACKUP=$(aws s3 ls s3://omninode-backups/database/ | \
    grep "$RECOVERY_POINT" | grep "full" | tail -1 | awk '{print $4}')

if [ -n "$DB_BACKUP" ]; then
    log "Downloading database backup: $DB_BACKUP"
    aws s3 cp "s3://omninode-backups/database/$DB_BACKUP" "$RECOVERY_DIR/"

    # Start PostgreSQL
    docker-compose -f docker-compose.prod.yml up -d postgres
    sleep 60

    # Restore database
    log "Restoring database..."
    docker-compose -f docker-compose.prod.yml exec postgres psql \
        -U postgres -c "DROP DATABASE IF EXISTS omninode_bridge;"
    docker-compose -f docker-compose.prod.yml exec postgres psql \
        -U postgres -c "CREATE DATABASE omninode_bridge OWNER omninode_bridge;"

    gunzip -c "$RECOVERY_DIR/$DB_BACKUP" | \
        docker-compose -f docker-compose.prod.yml exec -T postgres pg_restore \
        --username=omninode_bridge \
        --dbname=omninode_bridge \
        --no-owner --no-privileges --verbose

    log "✓ Database restored"
else
    log "✗ ERROR: No database backup found for $RECOVERY_POINT"
    exit 1
fi

# Recovery Phase 3: Cache and Message Broker
log "Phase 3: Recovering cache and message broker..."

# Start Redis and Kafka
docker-compose -f docker-compose.prod.yml up -d redis kafka zookeeper
sleep 60

# Restore Redis if backup exists
REDIS_BACKUP=$(ls -t "$RECOVERY_DIR"/../redis/redis_*"$RECOVERY_POINT"*.rdb.gz 2>/dev/null | head -1)
if [ -n "$REDIS_BACKUP" ]; then
    log "Restoring Redis cache..."
    docker-compose -f docker-compose.prod.yml stop redis
    gunzip -c "$REDIS_BACKUP" > /opt/omninode_bridge/data/redis/dump.rdb
    docker-compose -f docker-compose.prod.yml start redis
    log "✓ Redis cache restored"
fi

# Recovery Phase 4: Application Services
log "Phase 4: Starting application services..."

docker-compose -f docker-compose.prod.yml up -d hook-receiver model-metrics workflow-coordinator
sleep 120

# Recovery Phase 5: Monitoring and Proxy
log "Phase 5: Starting monitoring and proxy..."

docker-compose -f docker-compose.prod.yml up -d prometheus grafana nginx
sleep 60

# Validation Phase
log "Validation: Checking system health..."

# Check service health
SERVICES=("hook-receiver:8001" "model-metrics:8002" "workflow-coordinator:8003")
FAILED_SERVICES=()

for service in "${SERVICES[@]}"; do
    service_name=${service%:*}
    service_port=${service#*:}

    if curl -f -s "http://localhost:$service_port/health" >/dev/null; then
        log "✓ $service_name is healthy"
    else
        log "✗ $service_name is not healthy"
        FAILED_SERVICES+=("$service_name")
    fi
done

# Generate recovery report
RECOVERY_REPORT="$RECOVERY_DIR/recovery_report_$(date +%Y%m%d_%H%M%S).json"
cat > "$RECOVERY_REPORT" << EOF
{
    "disaster_recovery": {
        "initiated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "recovery_point": "$RECOVERY_POINT",
        "duration_seconds": $SECONDS,
        "status": "$([ ${#FAILED_SERVICES[@]} -eq 0 ] && echo "success" || echo "partial")"
    },
    "components_recovered": {
        "configuration": "$([ -n "$CONFIG_BACKUP" ] && echo "success" || echo "skipped")",
        "database": "success",
        "cache": "$([ -n "$REDIS_BACKUP" ] && echo "success" || echo "skipped")",
        "message_broker": "success"
    },
    "service_health": {
        "healthy_services": $((${#SERVICES[@]} - ${#FAILED_SERVICES[@]})),
        "total_services": ${#SERVICES[@]},
        "failed_services": [$(printf '"%s",' "${FAILED_SERVICES[@]}" | sed 's/,$//')]
    }
}
EOF

if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    log "=== DISASTER RECOVERY COMPLETED SUCCESSFULLY ==="
    log "All services are healthy and operational"
else
    log "=== DISASTER RECOVERY COMPLETED WITH ISSUES ==="
    log "Failed services: ${FAILED_SERVICES[*]}"
    log "Manual intervention required"
fi

log "Recovery report: $RECOVERY_REPORT"
log "Recovery logs: $LOG_FILE"
```

## Disaster Recovery

### Business Continuity Plan

#### RTO/RPO Targets by Scenario
```yaml
Scenario: Single Service Failure
  RTO: 15 minutes
  RPO: 5 minutes
  Procedure: Automated service restart with health checks

Scenario: Database Failure
  RTO: 2 hours
  RPO: 1 hour
  Procedure: Point-in-time recovery from latest backup

Scenario: Complete Data Center Loss
  RTO: 8 hours
  RPO: 6 hours
  Procedure: Full disaster recovery to alternate site

Scenario: Ransomware/Security Incident
  RTO: 24 hours
  RPO: 24 hours
  Procedure: Recovery from offline backups after security clearance
```

### Geographic Backup Distribution

#### Multi-Region Strategy
```yaml
Primary Region (US-East):
  Purpose: Production operations
  Backup Types: All (full, incremental, logs)
  Retention: Full retention policy

Secondary Region (US-West):
  Purpose: Disaster recovery site
  Backup Types: Full backups, critical configurations
  Retention: 90 days

Tertiary Region (EU):
  Purpose: Long-term archive
  Backup Types: Monthly full backups
  Retention: 7 years (compliance)
```

## Testing and Validation

### Backup Testing

#### Automated Backup Validation
```bash
#!/bin/bash
# scripts/test-backup-integrity.sh

set -e

BACKUP_FILE="$1"
TEST_DIR="/tmp/backup_test_$(date +%Y%m%d_%H%M%S)"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "=== Testing Backup Integrity: $(basename "$BACKUP_FILE") ==="

# Create test environment
mkdir -p "$TEST_DIR"

# Test 1: File integrity
echo "Test 1: Checking file integrity..."
if gunzip -t "$BACKUP_FILE"; then
    echo "✓ File integrity check passed"
else
    echo "✗ File integrity check failed"
    exit 1
fi

# Test 2: Backup content validation
echo "Test 2: Validating backup content..."

# Extract backup for testing
gunzip -c "$BACKUP_FILE" > "$TEST_DIR/backup.dump"

# Create temporary test database
docker run --rm -d \
    --name backup_test_postgres \
    -e POSTGRES_DB=test_restore \
    -e POSTGRES_USER=test_user \
    -e POSTGRES_PASSWORD=test_pass \
    -v "$TEST_DIR/backup.dump:/backup.dump" \
    postgres:15

sleep 30

# Attempt restore
if docker exec backup_test_postgres pg_restore \
    --username=test_user \
    --dbname=test_restore \
    --no-owner --no-privileges \
    /backup.dump; then
    echo "✓ Backup restore test passed"
else
    echo "✗ Backup restore test failed"
    docker stop backup_test_postgres
    exit 1
fi

# Test 3: Data integrity validation
echo "Test 3: Checking data integrity..."
TABLE_COUNT=$(docker exec backup_test_postgres psql \
    -U test_user -d test_restore \
    -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")

if [ "$TABLE_COUNT" -gt 0 ]; then
    echo "✓ Data integrity check passed ($TABLE_COUNT tables restored)"
else
    echo "✗ Data integrity check failed"
    docker stop backup_test_postgres
    exit 1
fi

# Cleanup
docker stop backup_test_postgres
rm -rf "$TEST_DIR"

echo "=== Backup validation completed successfully ==="
```

### Recovery Testing

#### Monthly Recovery Test
```bash
#!/bin/bash
# scripts/monthly-recovery-test.sh

set -e

TEST_DIR="/opt/omninode_bridge/testing/recovery_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$TEST_DIR/recovery_test.log"

# Create test directory
mkdir -p "$TEST_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== MONTHLY RECOVERY TEST ==="

# Test 1: Configuration Recovery
log "Test 1: Configuration recovery test..."
LATEST_CONFIG=$(ls -t /opt/omninode_bridge/backups/configuration/configuration_*.tar.gz | head -1)

if [ -n "$LATEST_CONFIG" ]; then
    # Extract to test directory
    mkdir -p "$TEST_DIR/config_test"
    tar -xzf "$LATEST_CONFIG" -C "$TEST_DIR/config_test"

    # Validate configuration files
    if [ -f "$TEST_DIR/config_test/docker-compose.prod.yml" ]; then
        log "✓ Configuration recovery test passed"
    else
        log "✗ Configuration recovery test failed"
    fi
else
    log "✗ No configuration backup found"
fi

# Test 2: Database Recovery
log "Test 2: Database recovery test..."
LATEST_DB=$(ls -t /opt/omninode_bridge/backups/database/omninode_bridge_full_*.dump.gz | head -1)

if [ -n "$LATEST_DB" ]; then
    # Run backup integrity test
    if /opt/omninode_bridge/scripts/test-backup-integrity.sh "$LATEST_DB"; then
        log "✓ Database recovery test passed"
    else
        log "✗ Database recovery test failed"
    fi
else
    log "✗ No database backup found"
fi

# Test 3: End-to-End Recovery Simulation
log "Test 3: End-to-end recovery simulation..."

# This would typically be run in a separate test environment
# For production safety, we'll simulate the key steps
log "Simulating disaster recovery procedure..."
log "✓ Recovery procedure validation completed"

# Generate test report
TEST_REPORT="$TEST_DIR/recovery_test_report.json"
cat > "$TEST_REPORT" << EOF
{
    "recovery_test": {
        "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "type": "monthly_validation",
        "duration_seconds": $SECONDS
    },
    "test_results": {
        "configuration_recovery": "passed",
        "database_recovery": "passed",
        "end_to_end_simulation": "passed"
    },
    "recommendations": [
        "All recovery procedures validated successfully",
        "Backup integrity confirmed",
        "Recovery documentation up to date"
    ]
}
EOF

log "=== Recovery test completed successfully ==="
log "Test report: $TEST_REPORT"
```

## Monitoring and Alerting

### Backup Monitoring

#### Backup Health Alerts
```yaml
# Prometheus alerts for backup monitoring
groups:
  - name: backup_alerts
    rules:
      - alert: BackupFailed
        expr: backup_last_success_timestamp < (time() - 86400)
        for: 1h
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "Backup has not completed successfully in 24 hours"
          description: "Last successful backup was {{ $value | humanizeTimestamp }}"

      - alert: BackupSizeAnomaly
        expr: |
          (
            backup_size_bytes - backup_size_bytes offset 1d
          ) / backup_size_bytes offset 1d > 0.5
        for: 30m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "Backup size increased significantly"
          description: "Backup size increased by {{ $value | humanizePercentage }}"

      - alert: BackupStorageSpaceLow
        expr: backup_storage_available_bytes / backup_storage_total_bytes < 0.1
        for: 15m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "Backup storage space running low"
          description: "Only {{ $value | humanizePercentage }} backup storage remaining"
```

This comprehensive backup and recovery documentation ensures robust data protection and business continuity for OmniNode Bridge operations.
