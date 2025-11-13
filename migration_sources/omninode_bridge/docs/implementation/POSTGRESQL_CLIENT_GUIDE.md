# PostgreSQL Client Implementation Guide

## Overview

The OmniNode Bridge PostgreSQL client provides high-performance, secure, and resilient database connectivity with advanced features including SSL/TLS support, connection pool optimization, batch operations, and comprehensive error handling.

## Key Features

### 🔒 Security Features
- **SSL/TLS encryption** with certificate validation
- **Connection security validation** based on environment
- **Prepared statement caching** to prevent SQL injection
- **Security audit logging** for compliance
- **Row-level security** support

### ⚡ Performance Features
- **Optimized connection pooling** with leak detection
- **Prepared statement caching** for frequently used queries
- **Batch operation support** with executemany optimization
- **Ultra-high-performance COPY operations** for bulk data
- **Adaptive bulk insert** methods
- **Connection warmup** for faster initial responses

### 🛡️ Resilience Features
- **Comprehensive error handling** with specific exception types
- **Automatic retry logic** with exponential backoff
- **Circuit breaker pattern** integration
- **Graceful degradation** capabilities
- **Resource leak detection** and prevention

## Configuration

### Environment-Based Configuration
```python
from omninode_bridge.services.postgres_client import PostgresClient

# Automatic configuration from environment
client = PostgresClient()
await client.connect()
```

### Environment Variables
```bash
# Basic Connection Settings
POSTGRES_HOST=omninode-bridge-postgres
POSTGRES_PORT=5432
POSTGRES_DATABASE=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password

# SSL/TLS Configuration
POSTGRES_SSL_ENABLED=true
POSTGRES_SSL_MODE=require  # disable, allow, prefer, require, verify-ca, verify-full
POSTGRES_SSL_CHECK_HOSTNAME=true
POSTGRES_SSL_CERT=/path/to/client-cert.pem  # Optional client certificate
POSTGRES_SSL_KEY=/path/to/client-key.pem    # Optional client key
POSTGRES_SSL_CA=/path/to/ca-cert.pem        # Optional CA certificate

# Connection Pool Optimization
POSTGRES_POOL_MIN_SIZE=10
POSTGRES_POOL_MAX_SIZE=50
POSTGRES_QUERY_TIMEOUT=60
POSTGRES_ACQUIRE_TIMEOUT=10
POSTGRES_POOL_EXHAUSTION_THRESHOLD=80.0
POSTGRES_MAX_QUERIES_PER_CONNECTION=50000
POSTGRES_CONNECTION_MAX_AGE=3600
POSTGRES_LEAK_DETECTION=true
```

### Environment-Specific Defaults

#### Development
```bash
POSTGRES_POOL_MIN_SIZE=2
POSTGRES_POOL_MAX_SIZE=10
POSTGRES_SSL_ENABLED=false
POSTGRES_QUERY_TIMEOUT=30
```

#### Production
```bash
POSTGRES_POOL_MIN_SIZE=10
POSTGRES_POOL_MAX_SIZE=50
POSTGRES_SSL_ENABLED=true
POSTGRES_SSL_MODE=require
POSTGRES_QUERY_TIMEOUT=60
POSTGRES_LEAK_DETECTION=true
```

## SSL/TLS Configuration

### Production SSL Setup
```python
# Automatic SSL context creation with enhanced security
client = PostgresClient(
    ssl_enabled=True,
    ssl_cert_path="/path/to/client.pem",
    ssl_key_path="/path/to/client-key.pem",
    ssl_ca_path="/path/to/ca.pem"
)

# SSL context includes:
# - TLS 1.2/1.3 only
# - Strong cipher suites
# - Certificate verification
# - Hostname validation (configurable)
```

### SSL Validation
```python
# Validate SSL connection
ssl_result = await client.validate_ssl_connection()

print(f"SSL Enabled: {ssl_result.ssl_enabled}")
print(f"SSL Version: {ssl_result.ssl_version}")
print(f"Cipher Suite: {ssl_result.cipher_suite}")
```

### Environment-Based SSL Enforcement
```python
# Automatic SSL enforcement in production
# Development: SSL optional
# Staging: SSL preferred
# Production: SSL required with full validation
```

## Connection Management

### Connection Pool Optimization
```python
# High-performance connection pool with monitoring
client = PostgresClient(
    min_size=10,                    # Minimum connections
    max_size=50,                    # Maximum connections
    max_queries_per_connection=50000,  # Rotate after X queries
    connection_max_age_seconds=3600,   # Rotate after 1 hour
    query_timeout_seconds=60,          # Query timeout
    acquire_timeout_seconds=10         # Connection acquisition timeout
)
```

### Pool Health Monitoring
```python
# Get detailed pool metrics
metrics = await client.get_pool_metrics()

print(f"Current Size: {metrics.current_size}")
print(f"Utilization: {metrics.utilization_percent}%")
print(f"Potential Leaks: {metrics.performance_metrics['potential_leaks']}")

# Health check with pool monitoring
health = await client.health_check()
print(f"Pool Health: {health.get('pool_health', 'unknown')}")
```

### Connection Warmup
```python
# Automatic connection pool warmup for better performance
await client.connect()  # Automatically warms up min_size connections

# Manual warmup
await client._warmup_connection_pool()
```

## Performance Optimizations

### Prepared Statement Caching
```python
# Automatic prepared statement caching for better performance
result = await client.execute_query(
    "SELECT * FROM users WHERE id = $1",
    user_id,
    use_prepared_statements=True  # Default: True
)

# Cache metrics
performance = await client.get_performance_metrics()
print(f"Cache Hit Rate: {performance.cache_hit_rate}%")
```

### Batch Operations
```python
from omninode_bridge.services.postgres_client import BatchOperation

# Create batch operations
operations = [
    BatchOperation(
        query="INSERT INTO events (id, data) VALUES ($1, $2)",
        params=(event_id, event_data),
        operation_type="execute"
    ),
    # ... more operations
]

# Execute batch with optimization
results = await client.execute_batch(
    operations,
    use_transaction=True,           # Wrap in transaction
    optimize_similar_queries=True   # Use executemany for identical queries
)
```

### Bulk Insert Operations
```python
# High-performance bulk insert
columns = ["id", "name", "email", "created_at"]
data = [
    (1, "John", "john@example.com", datetime.now()),
    (2, "Jane", "jane@example.com", datetime.now()),
    # ... thousands of rows
]

# Adaptive bulk insert (automatically chooses best method)
success = await client.adaptive_bulk_insert(
    table_name="users",
    columns=columns,
    data=data,
    size_threshold=1000  # Use COPY for >1000 rows, executemany for smaller
)

# Ultra-high-performance COPY for large datasets
success = await client.copy_bulk_insert(
    table_name="users",
    columns=columns,
    data=data,
    format_type="csv"  # or "binary" for maximum performance
)
```

## Error Handling and Resilience

### Comprehensive Error Handling
```python
from asyncpg.exceptions import (
    ConnectionFailureError,
    TooManyConnectionsError,
    InvalidPasswordError
)

try:
    await client.connect()
except ConnectionFailureError as e:
    logger.error(f"Network connectivity issue: {e}")
    # Implement retry logic or fallback
except TooManyConnectionsError as e:
    logger.error(f"Database overloaded: {e}")
    # Scale down operations or use read replica
except InvalidPasswordError as e:
    logger.error(f"Authentication failed: {e}")
    # Check credentials configuration
```

### Circuit Breaker Integration
```python
# Automatic circuit breaker protection
@DATABASE_CIRCUIT_BREAKER()
async def connect(self):
    # Circuit breaker prevents cascade failures
    # Opens on repeated failures
    # Half-opens for health checks
    # Closes when operations succeed
```

### Retry Logic with Exponential Backoff
```python
# Built-in retry logic for transient failures
await client.connect(
    max_retries=5,           # Maximum retry attempts
    base_delay=2.0          # Base delay with exponential backoff
)

# Automatic retry for:
# - Network connectivity issues
# - Temporary database unavailability
# - Connection pool exhaustion
```

## Security Features

### Audit Logging
```python
# Record security events
await client.record_security_event(
    event_type="connection_attempt",
    client_info={"ip": "192.168.1.100", "user_agent": "service"},
    success=True
)

# Record authentication failures
await client.record_security_event(
    event_type="authentication_failure",
    client_info={"ip": "192.168.1.100"},
    success=False,
    error_details="Invalid credentials"
)
```

### Connection Security Validation
```python
# Automatic security validation based on environment
client = PostgresClient()

# Development: Permissive (SSL optional)
# Staging: Enhanced (SSL preferred)
# Production: Strict (SSL required, hostname verification)
```

### Data Validation
```python
from omninode_bridge.services.postgres_client import HookEventData

# Validated data models prevent injection attacks
hook_data = HookEventData(
    id="event-123",
    source="github",
    action="push",
    resource="repository",
    resource_id="repo-456",
    payload={"commits": [...]},  # Size validated
    processing_errors=["error1"]  # Length validated
)

# Safe storage with validation
success = await client.store_hook_event(hook_data)
```

## Monitoring and Observability

### Performance Metrics
```python
# Get comprehensive performance metrics
metrics = await client.get_performance_metrics()

print(f"Average Query Time: {metrics.avg_query_time_ms}ms")
print(f"Cache Hit Rate: {metrics.cache_hit_rate}%")
print(f"Connection Metrics: {metrics.connection_metrics}")
print(f"Pool Efficiency: {metrics.pool_efficiency}")
```

### Health Monitoring
```python
# Detailed health check
health = await client.health_check()

{
    "status": "healthy",
    "connected": True,
    "test_query": True,
    "pool": {
        "size": 15,
        "max_size": 50,
        "utilization_percent": 30.0,
        "potential_leaks": 0
    },
    "pool_health": "normal"
}
```

### Resource Tracking
```python
# Monitor for resource leaks
pool_metrics = await client.get_pool_metrics()

if pool_metrics.performance_metrics['potential_leaks'] > 5:
    logger.warning("Potential connection leaks detected")
    # Implement cleanup or restart procedures
```

## Data Management

### Service Session Management
```python
# Create service session
session_id = uuid.uuid4()
success = await client.create_service_session(
    session_id=session_id,
    service_name="hook_receiver",
    instance_id="instance-1",
    metadata={"version": "1.0.0"}
)

# End service session
success = await client.end_service_session(session_id)

# Get active sessions
sessions = await client.get_active_sessions()
```

### Hook Event Storage
```python
# Store hook events with validation
from omninode_bridge.models.hooks import HookEvent

hook_event = HookEvent(...)
success = await client.store_hook_event(hook_event)

# Batch hook event storage
hook_events = [HookEventData(...) for _ in range(100)]
success = await client.store_hook_events_batch(hook_events)
```

### Event Metrics Recording
```python
# Record processing metrics
success = await client.record_event_metrics(
    event_id=event_id,
    processing_time_ms=45.2,
    kafka_publish_success=True,
    error_message=None
)
```

### Data Cleanup
```python
# Automatic data retention
cleanup_result = await client.cleanup_old_data()

print(f"Total Deleted: {cleanup_result.total_deleted}")
print(f"Operation Time: {cleanup_result.operation_time_ms}ms")

# Cleanup includes:
# - Old processed hook events (configurable retention)
# - Old event metrics
# - Ended service sessions
# - Old security audit logs
# - Old connection metrics
```

## Testing and Development

### Local Development Setup
```python
# Development-friendly configuration
client = PostgresClient(
    host="localhost",
    port=5436,  # Mapped port for external access
    ssl_enabled=False,  # Simplified for development
    min_size=2,         # Minimal resources
    max_size=10
)
```

### Testing with Mocks
```python
# Test configuration
import pytest
from unittest.mock import AsyncMock

@pytest.fixture
async def mock_postgres_client():
    client = PostgresClient()
    client.pool = AsyncMock()
    client._connected = True
    return client

async def test_hook_storage(mock_postgres_client):
    success = await mock_postgres_client.store_hook_event(hook_event)
    assert success
```

### Configuration Validation
```python
# Validate configuration before deployment
from omninode_bridge.config.environment_config import DatabaseConfig

try:
    config = DatabaseConfig()
    print("✅ Database configuration valid")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
```

## Best Practices

### Connection Pool Sizing
```python
# Calculate optimal pool size based on workload
# Rule of thumb:
# - Development: 2-10 connections
# - Staging: 5-25 connections
# - Production: 10-100 connections
# - Consider: concurrent users × average query time / target response time

POSTGRES_POOL_MAX_SIZE = min(
    max(concurrent_users * avg_query_time_ms / target_response_time_ms, 10),
    100  # Don't exceed database limits
)
```

### Query Optimization
```python
# Use prepared statements for frequent queries
result = await client.execute_query(
    "SELECT * FROM users WHERE active = $1 AND created_at > $2",
    True,
    datetime.now() - timedelta(days=30),
    use_prepared_statements=True
)

# Use batch operations for multiple similar queries
operations = [
    BatchOperation(
        query="UPDATE users SET last_seen = NOW() WHERE id = $1",
        params=(user_id,),
        operation_type="execute"
    )
    for user_id in user_ids
]
await client.execute_batch(operations)
```

### Error Recovery
```python
# Implement proper error recovery
async def robust_database_operation():
    try:
        return await client.execute_query("SELECT 1")
    except ConnectionFailureError:
        # Wait and retry
        await asyncio.sleep(5)
        await client.connect()  # Reconnect
        return await client.execute_query("SELECT 1")
    except TooManyConnectionsError:
        # Scale down operations
        await asyncio.sleep(10)
        return None  # Graceful degradation
```

### Resource Cleanup
```python
# Always properly cleanup resources
try:
    await client.connect()
    # Perform operations
    results = await client.execute_query("SELECT * FROM table")
finally:
    # Cleanup is automatic, but explicit cleanup for long-running services
    await client.cleanup()
```

## Migration from Legacy Systems

### From Raw asyncpg
```python
# Before: Manual connection management
import asyncpg

pool = await asyncpg.create_pool(dsn)
async with pool.acquire() as conn:
    result = await conn.fetch("SELECT * FROM users")

# After: Feature-rich client
client = PostgresClient()
await client.connect()
result = await client.fetch_all("SELECT * FROM users")
```

### From SQLAlchemy
```python
# Before: ORM complexity
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(database_url)
# ... complex ORM setup

# After: Direct high-performance access
client = PostgresClient()
await client.connect()
# Direct SQL with full performance optimization
```

## Troubleshooting

### Common Issues

**SSL Connection Failures:**
```python
# Check SSL configuration
ssl_result = await client.validate_ssl_connection()
if not ssl_result.ssl_enabled:
    print("SSL not properly configured")
    print(f"Errors: {ssl_result.validation_errors}")
```

**Connection Pool Exhaustion:**
```python
# Monitor pool utilization
health = await client.health_check()
if health.get('pool_exhaustion_likely'):
    print("Increase POSTGRES_POOL_MAX_SIZE or reduce connection usage")
```

**Performance Issues:**
```python
# Check query performance
metrics = await client.get_performance_metrics()
if metrics.avg_query_time_ms > 100:
    print("Consider query optimization or connection pooling tuning")
```

The OmniNode Bridge PostgreSQL client provides enterprise-grade database connectivity with comprehensive features for security, performance, and resilience. Use this guide to leverage its full capabilities in your applications.
