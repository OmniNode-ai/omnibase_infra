# NodeDistributedLockEffect v1.0.0

**ONEX v2.0 Compliant Distributed Locking Effect Node**

PostgreSQL-backed distributed locking for multi-instance deployments with automatic lease expiration, deadlock detection, and comprehensive monitoring.

## Overview

`NodeDistributedLockEffect` provides production-grade distributed locking capabilities for coordinating work across multiple service instances. Built on PostgreSQL for reliability and ACID guarantees, it supports:

- **Lock Acquisition** with configurable lease duration
- **Lock Release** with ownership validation
- **Lease Extension** to prevent expiration
- **Lock Status Queries** for monitoring
- **Automatic Cleanup** of expired locks

## Features

- ✅ **PostgreSQL-Backed** - ACID guarantees and reliable persistence
- ✅ **Lease Management** - Automatic expiration and cleanup
- ✅ **Deadlock Detection** - Retry with exponential backoff
- ✅ **Ownership Tracking** - Metadata and multi-tenant support
- ✅ **Background Cleanup** - Periodic removal of expired locks
- ✅ **Kafka Events** - Observability through event publishing
- ✅ **Metrics Collection** - Performance and health monitoring
- ✅ **ONEX v2.0 Compliant** - Full contract and manifest support

## Performance Targets

| Metric | Target | Actual (P95) |
|--------|--------|--------------|
| Lock Acquisition | < 50ms | TBD |
| Lock Release | < 10ms | TBD |
| Throughput | 100+ ops/sec | TBD |
| Cleanup (100 locks) | < 100ms | TBD |

## Quick Start

### Installation

```bash
# Install dependencies
poetry add asyncpg omnibase_core

# Or with pip
pip install asyncpg omnibase-core
```

### Basic Usage

```python
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omninode_bridge.nodes.distributed_lock_effect.v1_0_0 import NodeDistributedLockEffect

# 1. Create configuration container
config = {
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_database": "omninode_bridge",
    "postgres_user": "postgres",
    "postgres_password": "password",  # pragma: allowlist secret
    "default_lease_duration": 30.0,
    "cleanup_interval": 60.0,
}

container = ModelContainer(value=config, container_type="config")

# 2. Initialize node
node = NodeDistributedLockEffect(container)
await node.initialize()

# 3. Acquire lock
acquire_contract = ModelContractEffect(
    name="acquire_lock",
    version={"major": 1, "minor": 0, "patch": 0},
    description="Acquire distributed lock",
    node_type="EFFECT",
    input_model="ModelDistributedLockRequest",
    output_model="ModelDistributedLockResponse",
    input_state={
        "operation": "acquire",
        "lock_name": "workflow_processing_lock",
        "owner_id": "orchestrator-node-1",
        "lease_duration": 30.0,
    },
)

response = await node.execute_effect(acquire_contract)

if response.success:
    print(f"✅ Lock acquired: {response.lock_info.lock_name}")
    print(f"   Expires at: {response.lock_info.expires_at}")
else:
    print(f"❌ Lock acquisition failed: {response.error_message}")

# 4. Release lock
release_contract = ModelContractEffect(
    name="release_lock",
    version={"major": 1, "minor": 0, "patch": 0},
    description="Release distributed lock",
    node_type="EFFECT",
    input_model="ModelDistributedLockRequest",
    output_model="ModelDistributedLockResponse",
    input_state={
        "operation": "release",
        "lock_name": "workflow_processing_lock",
        "owner_id": "orchestrator-node-1",
    },
)

response = await node.execute_effect(release_contract)
print(f"Lock released: {response.success}")

# 5. Cleanup
await node.shutdown()
```

## API Reference

### Operations

#### 1. ACQUIRE

Acquire a distributed lock with configurable lease duration.

**Input:**
```python
{
    "operation": "acquire",
    "lock_name": "my_lock",           # Unique lock identifier
    "owner_id": "instance-123",       # Lock owner identifier
    "lease_duration": 30.0,           # Lease duration in seconds
    "max_acquire_attempts": 3,        # Retry attempts
    "owner_metadata": {               # Optional metadata
        "node_id": "orchestrator-1",
        "instance_id": "inst-abc123"
    }
}
```

**Response:**
```python
{
    "success": True,
    "operation": "acquire",
    "lock_info": {
        "lock_name": "my_lock",
        "owner_id": "instance-123",
        "status": "acquired",
        "acquired_at": "2025-10-30T12:00:00Z",
        "expires_at": "2025-10-30T12:00:30Z",
        "lease_duration": 30.0,
        "acquisition_count": 1,
        "extension_count": 0
    },
    "duration_ms": 45.2,
    "acquire_attempts": 1
}
```

#### 2. RELEASE

Release a held distributed lock.

**Input:**
```python
{
    "operation": "release",
    "lock_name": "my_lock",
    "owner_id": "instance-123"
}
```

**Response:**
```python
{
    "success": True,
    "operation": "release",
    "duration_ms": 8.5
}
```

#### 3. EXTEND

Extend lock lease duration before expiration.

**Input:**
```python
{
    "operation": "extend",
    "lock_name": "my_lock",
    "owner_id": "instance-123",
    "extension_duration": 30.0  # Additional seconds
}
```

**Response:**
```python
{
    "success": True,
    "operation": "extend",
    "lock_info": {
        "expires_at": "2025-10-30T12:01:00Z",  # Updated expiration
        "extension_count": 1
    },
    "duration_ms": 12.3
}
```

#### 4. QUERY

Query lock status and metadata.

**Input:**
```python
{
    "operation": "query",
    "lock_name": "my_lock"
}
```

**Response:**
```python
{
    "success": True,
    "operation": "query",
    "lock_info": {
        "lock_name": "my_lock",
        "owner_id": "instance-123",
        "status": "acquired",
        "acquired_at": "2025-10-30T12:00:00Z",
        "expires_at": "2025-10-30T12:00:30Z",
        "lease_duration": 30.0
    },
    "duration_ms": 5.2
}
```

#### 5. CLEANUP

Clean up expired locks (typically called by background task).

**Input:**
```python
{
    "operation": "cleanup",
    "lock_name": "*",  # Cleanup all locks
    "max_age_seconds": 3600.0
}
```

**Response:**
```python
{
    "success": True,
    "operation": "cleanup",
    "cleaned_count": 5,
    "cleaned_lock_names": ["lock1", "lock2", "lock3", "lock4", "lock5"],
    "duration_ms": 85.5
}
```

## Configuration

### Environment Variables

```bash
# PostgreSQL Configuration
POSTGRES_HOST=192.168.86.200
POSTGRES_PORT=5436
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=***
POSTGRES_MIN_CONNECTIONS=5
POSTGRES_MAX_CONNECTIONS=20

# Lock Behavior
DEFAULT_LEASE_DURATION=30.0
CLEANUP_INTERVAL=60.0
MAX_ACQUIRE_ATTEMPTS=3
ACQUIRE_RETRY_DELAY=0.1

# Kafka Event Publishing
KAFKA_BOOTSTRAP_SERVERS=192.168.86.200:9092
KAFKA_TOPIC_LOCK_ACQUIRED=dev.omninode-bridge.lock.acquired.v1
KAFKA_TOPIC_LOCK_RELEASED=dev.omninode-bridge.lock.released.v1
KAFKA_TOPIC_LOCK_EXPIRED=dev.omninode-bridge.lock.expired.v1
```

### Configuration Model

```python
from omninode_bridge.nodes.distributed_lock_effect.v1_0_0.models import ModelDistributedLockConfig

config = ModelDistributedLockConfig(
    postgres_host="localhost",
    postgres_port=5432,
    postgres_database="omninode_bridge",
    default_lease_duration=30.0,
    cleanup_interval=60.0,
    max_acquire_attempts=3,
    acquire_retry_delay=0.1,
)
```

## Database Schema

```sql
CREATE TABLE distributed_locks (
    lock_id VARCHAR(255) PRIMARY KEY,
    owner_id VARCHAR(255) NOT NULL,
    acquired_at BIGINT NOT NULL,
    expires_at BIGINT NOT NULL,
    lease_duration REAL NOT NULL,
    heartbeat_at BIGINT NOT NULL,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'acquired',
    acquisition_count INT DEFAULT 1,
    extension_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_distributed_locks_expires_at ON distributed_locks(expires_at);
CREATE INDEX idx_distributed_locks_owner_id ON distributed_locks(owner_id);
CREATE INDEX idx_distributed_locks_status ON distributed_locks(status);
```

## Testing

### Run Unit Tests

```bash
# All unit tests
pytest tests/test_node.py -v

# Specific test
pytest tests/test_node.py::TestNodeDistributedLockEffect::test_acquire_lock_success -v

# With coverage
pytest tests/test_node.py --cov=. --cov-report=html
```

### Run Integration Tests

```bash
# Requires running PostgreSQL
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DATABASE=test_db

pytest tests/test_integration.py -m integration -v
```

## Deployment

### Docker Compose

```yaml
services:
  distributed-lock-effect:
    image: omninode-bridge/distributed-lock-effect:1.0.0
    ports:
      - "8062:8062"
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_DATABASE: omninode_bridge
      DEFAULT_LEASE_DURATION: "30.0"
      CLEANUP_INTERVAL: "60.0"
    depends_on:
      - postgres
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-lock-effect
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: distributed-lock-effect
        image: omninode-bridge/distributed-lock-effect:1.0.0
        ports:
        - containerPort: 8062
        env:
        - name: POSTGRES_HOST
          value: postgres-service
        - name: DEFAULT_LEASE_DURATION
          value: "30.0"
```

## Monitoring

### Metrics

```python
# Get node metrics
metrics = node.get_metrics()

# Example output
{
    "total_acquires": 150,
    "total_releases": 148,
    "total_extends": 25,
    "total_queries": 50,
    "total_cleanups": 10,
    "failed_acquires": 2,
    "success_rate": 0.9867,
    "avg_acquire_time_ms": 42.5,
    "avg_release_time_ms": 8.2
}
```

### Health Checks

```bash
# Node health
curl http://localhost:8062/health

# PostgreSQL connection health
curl http://localhost:8062/health/postgres

# Active locks count
curl http://localhost:8062/health/locks
```

## Best Practices

### 1. Lease Duration

- **Short tasks** (< 10s): Use 15-30s lease
- **Medium tasks** (10-60s): Use 60-120s lease
- **Long tasks** (> 60s): Use 5-10min lease + periodic extension

### 2. Lock Naming

```python
# ✅ Good: Descriptive, hierarchical names
"workflow.processing.{workflow_id}"
"api.rate_limit.{user_id}"
"cache.invalidation.{key}"

# ❌ Bad: Generic, collision-prone names
"lock1"
"my_lock"
"processing"
```

### 3. Error Handling

```python
try:
    response = await node.execute_effect(acquire_contract)
    if not response.success:
        if response.error_code == "LOCK_TIMEOUT":
            # Lock is held by another instance
            logger.warning(f"Lock busy, retry after {response.retry_after_seconds}s")
            await asyncio.sleep(response.retry_after_seconds)
        elif response.error_code == "DATABASE_ERROR":
            # Database connectivity issue
            logger.error("Database error, escalating to ops")
            raise
except ModelOnexError as e:
    logger.error(f"Lock operation failed: {e}")
    raise
```

### 4. Cleanup Strategy

```python
# Manual cleanup after processing
try:
    # Acquire lock
    acquire_response = await node.execute_effect(acquire_contract)

    # Do work...

finally:
    # Always release lock
    release_response = await node.execute_effect(release_contract)
```

## Troubleshooting

### Lock Acquisition Failures

**Symptom**: `LOCK_TIMEOUT` errors

**Causes**:
- Lock held by another instance
- Deadlock in application logic
- Insufficient retry attempts

**Solutions**:
- Increase `max_acquire_attempts`
- Add exponential backoff delays
- Check for deadlocks in application code

### High Lock Cleanup Count

**Symptom**: Many locks being cleaned up

**Causes**:
- Application crashes before releasing locks
- Lease duration too short for task
- Network issues preventing release

**Solutions**:
- Increase `default_lease_duration`
- Add lock release in finally blocks
- Monitor application health

### Database Connection Pool Exhaustion

**Symptom**: Connection timeout errors

**Causes**:
- High concurrent lock operations
- Slow queries
- Insufficient pool size

**Solutions**:
- Increase `postgres_max_connections`
- Add connection pool monitoring
- Optimize lock operation frequency

## License

Copyright © 2025 OmniNode Team. All rights reserved.

## Support

- **Documentation**: [ONEX Architecture Guide](../../docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md)
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join our Discord community
