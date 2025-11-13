# ADR-016: Database Strategy for Multi-Service Architecture

**Status**: Accepted
**Date**: 2024-09-25
**Deciders**: OmniNode Bridge Architecture Team
**Technical Story**: Implementation of database strategy for multi-service architecture with shared PostgreSQL instance

## Context

The multi-service architecture (HookReceiver, ModelMetrics API, WorkflowCoordinator) requires a robust data persistence strategy that can handle:

- Cross-service data consistency requirements
- High-performance concurrent access patterns
- Session management across service boundaries
- Event sourcing and audit trail requirements
- Scalable connection management with resource efficiency
- Security and compliance for sensitive data

Traditional approaches include either database-per-service (microservices pattern) or shared database patterns. Each has significant trade-offs in terms of complexity, consistency, and operational overhead.

## Decision

We adopt a **Shared Database with Service-Specific Schemas** pattern using PostgreSQL with the following architecture:

### Database Architecture
- **Single PostgreSQL Instance**: Shared across all services for consistency and operational simplicity
- **Connection Pooling**: Advanced connection pool management with leak detection and performance optimization
- **Schema Organization**: Logical separation by data domain rather than service boundaries
- **Transactional Consistency**: ACID transactions across service operations when required

### Core Components

#### 1. Advanced Connection Pool Management
```python
PostgresClient(
    min_size=5,          # Minimum pool size
    max_size=25,         # Maximum pool size
    max_queries_per_connection=10000,
    connection_max_age_seconds=3600,
    query_timeout_seconds=30,
    acquire_timeout_seconds=10
)
```

#### 2. Performance Optimization Features
- **Prepared Statement Caching**: Automatic caching of frequently used queries
- **Batch Operation Support**: Optimized batch processing with executemany()
- **Bulk Insert Capabilities**: COPY protocol for high-throughput data loading
- **Connection Warmup**: Pre-established connections for reduced latency
- **Adaptive Query Optimization**: Different strategies based on query patterns

#### 3. Security and Compliance
- **SSL/TLS Encryption**: Comprehensive SSL configuration with certificate validation
- **Connection Security**: Enhanced authentication and authorization
- **Audit Logging**: Security events and access patterns tracked
- **Data Retention**: Automatic cleanup with configurable retention policies

#### 4. Monitoring and Observability
```python
# Comprehensive metrics collection
connection_metrics = {
    "acquired_connections": int,
    "released_connections": int,
    "failed_acquisitions": int,
    "query_count": int,
    "avg_query_time_ms": float,
    "pool_utilization_percent": float,
    "potential_leaks": int
}
```

### Database Schema Design

#### Core Tables
```sql
-- Service session management
CREATE TABLE service_sessions (
    id UUID PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    instance_id VARCHAR(255),
    session_start TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    session_end TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    metadata JSONB DEFAULT '{}'
);

-- Event persistence and debugging
CREATE TABLE hook_events (
    id UUID PRIMARY KEY,
    source VARCHAR(255) NOT NULL,
    action VARCHAR(255) NOT NULL,
    resource VARCHAR(255) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    processed BOOLEAN NOT NULL DEFAULT FALSE,
    processing_errors TEXT[],
    retry_count INTEGER NOT NULL DEFAULT 0
);

-- Performance and audit metrics
CREATE TABLE event_metrics (
    id SERIAL PRIMARY KEY,
    event_id UUID NOT NULL,
    processing_time_ms FLOAT NOT NULL,
    kafka_publish_success BOOLEAN NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Security audit trail
CREATE TABLE security_audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    client_info JSONB,
    success BOOLEAN NOT NULL,
    error_details TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Connection pool monitoring
CREATE TABLE connection_metrics (
    id SERIAL PRIMARY KEY,
    pool_size INTEGER NOT NULL,
    active_connections INTEGER NOT NULL,
    idle_connections INTEGER NOT NULL,
    total_queries BIGINT NOT NULL DEFAULT 0,
    failed_queries INTEGER NOT NULL DEFAULT 0,
    avg_query_time_ms FLOAT,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

#### Performance Indexes
```sql
-- Optimized indexes for common query patterns
CREATE INDEX idx_service_sessions_service_status
    ON service_sessions(service_name, status) WHERE status = 'active';

CREATE INDEX idx_hook_events_processed
    ON hook_events(processed, created_at DESC) WHERE NOT processed;

CREATE INDEX idx_event_metrics_performance
    ON event_metrics(processing_time_ms, kafka_publish_success);

CREATE INDEX idx_security_audit_failed
    ON security_audit_log(created_at DESC) WHERE NOT success;
```

### Data Management Strategy

#### 1. Automatic Data Retention
```sql
-- Automated cleanup function
CREATE OR REPLACE FUNCTION cleanup_old_data() RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean up processed events older than 30 days
    DELETE FROM hook_events
    WHERE processed = TRUE AND created_at < NOW() - INTERVAL '30 days';

    -- Clean up old metrics data older than 90 days
    DELETE FROM event_metrics
    WHERE created_at < NOW() - INTERVAL '90 days';

    -- Clean up ended sessions older than 7 days
    DELETE FROM service_sessions
    WHERE status IN ('ended', 'terminated', 'failed')
    AND session_end < NOW() - INTERVAL '7 days';

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
```

#### 2. Circuit Breaker Integration
```python
@DATABASE_CIRCUIT_BREAKER()
async def connect(self, max_retries: int = 3, base_delay: float = 1.0):
    """Connect with circuit breaker protection and exponential backoff."""
    # Comprehensive connection logic with resilience patterns
```

#### 3. Health Check and Monitoring
```python
async def health_check(self) -> HealthCheckResult:
    """Enhanced health checking with pool monitoring."""
    return {
        "status": "healthy|degraded|unhealthy",
        "connection_pool": {
            "size": pool_size,
            "utilization_percent": utilization,
            "potential_leaks": leak_count
        },
        "query_test": {"result": bool, "response_time_ms": float}
    }
```

### Service Integration Patterns

#### 1. Session Management
```python
# Service startup
session_id = await postgres_client.create_service_session(
    session_id=uuid4(),
    service_name="hook_receiver",
    instance_id=instance_id,
    metadata={"version": "1.0.0"}
)

# Service shutdown
await postgres_client.end_service_session(session_id)
```

#### 2. Event Persistence
```python
# Individual event storage
await postgres_client.store_hook_event(hook_event_data)

# Batch event storage for performance
await postgres_client.store_hook_events_batch(hook_events_list)

# Ultra-high-performance bulk loading
await postgres_client.adaptive_bulk_insert(
    table_name="hook_events",
    columns=["id", "source", "action", "payload"],
    data=bulk_data
)
```

#### 3. Metrics Collection
```python
# Record processing metrics
await postgres_client.record_event_metrics(
    event_id=event_id,
    processing_time_ms=duration,
    kafka_publish_success=success,
    error_message=error if not success else None
)

# Get performance insights
metrics = await postgres_client.get_performance_metrics()
```

## Consequences

### Positive Consequences

- **Operational Simplicity**: Single database instance reduces operational complexity and cost
- **Data Consistency**: ACID transactions enable cross-service consistency when needed
- **Performance Optimization**: Advanced connection pooling and query optimization
- **Comprehensive Monitoring**: Detailed metrics and health monitoring built-in
- **Security**: Enterprise-grade SSL/TLS and audit capabilities
- **Automatic Management**: Self-managing connection pools with leak detection
- **Scalable Performance**: Prepared statements, batch operations, and bulk loading
- **Resilience**: Circuit breaker integration and graceful degradation

### Negative Consequences

- **Service Coupling**: Shared database creates coupling between services
- **Single Point of Failure**: Database outage affects all services
- **Schema Evolution Complexity**: Database changes require coordination across services
- **Resource Contention**: Services compete for database resources
- **Scaling Limitations**: Database becomes bottleneck as services scale
- **Security Blast Radius**: Database compromise affects all services

### Mitigation Strategies

- **Connection Isolation**: Service-specific connection pools prevent resource starvation
- **Circuit Breakers**: Prevent cascading failures from database issues
- **Read Replicas**: Planned for read-heavy workloads and service isolation
- **Backup and Recovery**: Comprehensive backup strategy with point-in-time recovery
- **Monitoring**: Proactive monitoring prevents issues before they impact services
- **Schema Versioning**: Structured approach to database schema evolution

## Implementation Details

### SSL/TLS Configuration
```python
# Enhanced SSL configuration
ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
ssl_context.verify_mode = ssl.CERT_REQUIRED
ssl_context.check_hostname = True
ssl_context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20")
```

### Performance Optimization
```python
# Adaptive bulk insert strategy
if len(data) >= 1000:
    # Use PostgreSQL COPY for large datasets
    await postgres_client.copy_bulk_insert(table, columns, data)
else:
    # Use executemany for smaller datasets
    await postgres_client.bulk_insert(table, columns, data)
```

### Connection Pool Monitoring
```python
# Automatic pool exhaustion detection
if pool_utilization > 85%:
    logger.warning("High connection pool utilization detected")
    degradation_service.record_degradation_event("database", "high_utilization")
```

## Compliance

This database strategy aligns with ONEX standards by:

- **Security First**: Comprehensive SSL/TLS and audit capabilities
- **Observability**: Detailed metrics and monitoring at all levels
- **Resilience**: Circuit breaker integration and graceful degradation
- **Performance**: Optimized connection management and query execution
- **Operational Excellence**: Automated management and maintenance features
- **Data Governance**: Structured retention policies and cleanup procedures

## Migration Strategy

### Phase 1: Foundation
- Deploy PostgreSQL with SSL/TLS configuration
- Implement connection pooling with monitoring
- Create base schema with security enhancements

### Phase 2: Service Integration
- Migrate HookReceiver to shared database
- Implement session management across services
- Deploy monitoring and alerting

### Phase 3: Optimization
- Enable prepared statement caching
- Implement batch operation patterns
- Optimize connection pool parameters based on production metrics

### Phase 4: Advanced Features
- Deploy read replicas for scaling
- Implement advanced monitoring dashboards
- Enable automatic performance tuning

## Related Decisions

- ADR-013: Multi-Service Architecture Pattern
- ADR-014: Event-Driven Architecture with Kafka
- ADR-015: Circuit Breaker Pattern Implementation
- ADR-017: Authentication and Authorization Strategy

## References

- [PostgreSQL High Performance](https://www.postgresql.org/docs/current/performance-tips.html)
- [Connection Pooling Best Practices](https://wiki.postgresql.org/wiki/Number_Of_Database_Connections)
- [Database Security Guide](https://www.postgresql.org/docs/current/security.html)
- [Microservices Data Management](https://microservices.io/patterns/data/)
- [asyncpg Performance Guide](https://magicstack.github.io/asyncpg/current/usage.html#performance)
