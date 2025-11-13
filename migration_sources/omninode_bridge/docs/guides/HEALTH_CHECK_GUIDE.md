# ONEX Health Check Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-02
**Status**: ✅ Implemented

## Overview

The ONEX health check system provides comprehensive health monitoring for bridge nodes with:

- **Component-level health checks** (database, kafka, external services)
- **Status reporting** (HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN)
- **Docker HEALTHCHECK compatibility**
- **Prometheus metrics integration** (optional)
- **Async and sync health check methods**

## Architecture

### Health Status Levels

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"      # All components operational
    DEGRADED = "degraded"    # Some non-critical components failing
    UNHEALTHY = "unhealthy"  # Critical components failing
    UNKNOWN = "unknown"      # Status cannot be determined
```

### Status Resolution Logic

```
Critical component UNHEALTHY → Overall: UNHEALTHY
Any component DEGRADED       → Overall: DEGRADED
All components HEALTHY       → Overall: HEALTHY
No components or all UNKNOWN → Overall: UNKNOWN
```

### Health Check Mixin

The `HealthCheckMixin` provides reusable health check functionality:

```python
from omninode_bridge.nodes.mixins import HealthCheckMixin, HealthStatus

class NodeBridgeOrchestrator(NodeOrchestrator, HealthCheckMixin):
    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)
        self.initialize_health_checks()  # Setup health check system
```

## Usage

### 1. Initialize Health Checks

Call `initialize_health_checks()` in your node's `__init__()` method:

```python
def __init__(self, container: ModelONEXContainer) -> None:
    super().__init__(container)

    # Initialize health check system
    self.initialize_health_checks()
```

### 2. Register Component Checks

Override `_register_component_checks()` to register your node's health checks:

```python
def _register_component_checks(self) -> None:
    """Register component health checks."""
    # Parent class registers node_runtime check
    super()._register_component_checks()

    # Register critical components
    self.register_component_check(
        "database",
        self._check_database_health,
        critical=True,           # Mark as critical
        timeout_seconds=5.0,     # Check timeout
    )

    # Register non-critical components
    self.register_component_check(
        "kafka",
        self._check_kafka_health,
        critical=False,          # Non-critical (graceful degradation)
        timeout_seconds=3.0,
    )
```

### 3. Implement Health Check Functions

Each health check function should return `(status, message, details)`:

```python
async def _check_database_health(
    self,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """
    Check database connection health.

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Attempt database operation
        result = await self.db_client.execute("SELECT 1")

        return (
            HealthStatus.HEALTHY,
            "Database connection healthy",
            {"query_time_ms": 1.5}
        )

    except TimeoutError:
        return (
            HealthStatus.UNHEALTHY,
            "Database query timed out",
            {"timeout_seconds": 5.0}
        )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Database connection failed: {e!s}",
            {"error": str(e)}
        )
```

### 4. Use Health Check Methods

#### Async Comprehensive Health Check

```python
# Perform full health check
health_result = await node.check_health()

# Access results
print(f"Overall Status: {health_result.overall_status.value}")
print(f"Components: {len(health_result.components)}")
print(f"Uptime: {health_result.uptime_seconds}s")

# Check specific component
for component in health_result.components:
    if component.name == "database":
        print(f"Database: {component.status.value}")
        print(f"  Message: {component.message}")
        print(f"  Details: {component.details}")
```

#### Sync Health Status Retrieval

```python
# Get cached health status (non-blocking)
health_status = node.get_health_status()

# Returns dict with health information
print(health_status["overall_status"])  # "healthy", "degraded", etc.
print(health_status["uptime_seconds"])
print(health_status["components"])
```

#### Simple Boolean Check

```python
# Quick health check for routing decisions
if node.is_healthy():
    # Node is operational (healthy or degraded)
    process_request()
else:
    # Node is unhealthy
    reject_request()
```

### 5. Docker HEALTHCHECK Integration

Nodes automatically support Docker health checks via the CLI:

```dockerfile
# Dockerfile health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -m omninode_bridge.nodes.health_check_cli orchestrator
```

The CLI returns exit codes:
- `0`: Healthy or degraded (container is operational)
- `1`: Unhealthy or unknown (container should be restarted)

## Component Health Checks

### Orchestrator Components

```python
def _register_component_checks(self) -> None:
    super()._register_component_checks()

    # Critical: MetadataStampingService
    self.register_component_check(
        "metadata_stamping",
        self._check_metadata_stamping_health,
        critical=True,
        timeout_seconds=5.0,
    )

    # Non-critical: OnexTree intelligence service
    self.register_component_check(
        "onextree",
        self._check_onextree_health,
        critical=False,
        timeout_seconds=3.0,
    )

    # Non-critical: Kafka event publishing
    self.register_component_check(
        "kafka",
        self._check_kafka_health,
        critical=False,
        timeout_seconds=3.0,
    )
```

### Reducer Components

```python
def _register_component_checks(self) -> None:
    super()._register_component_checks()

    # Critical: PostgreSQL state persistence
    self.register_component_check(
        "postgres",
        self._check_postgres_health,
        critical=True,
        timeout_seconds=5.0,
    )

    # Non-critical: Aggregation buffer health
    self.register_component_check(
        "aggregation_buffer",
        self._check_aggregation_buffer_health,
        critical=False,
        timeout_seconds=1.0,
    )
```

## Helper Functions

The mixin provides reusable health check helpers:

### Database Connection Check

```python
from omninode_bridge.nodes.mixins.health_mixin import check_database_connection

async def _check_postgres_health(self):
    return await check_database_connection(
        self.db_client,
        timeout_seconds=3.0
    )
```

### Kafka Connection Check

```python
from omninode_bridge.nodes.mixins.health_mixin import check_kafka_connection

async def _check_kafka_health(self):
    return await check_kafka_connection(
        self.kafka_producer,
        timeout_seconds=3.0
    )
```

### HTTP Service Check

```python
from omninode_bridge.nodes.mixins.health_mixin import check_http_service

async def _check_metadata_stamping_health(self):
    return await check_http_service(
        "http://metadata-stamping:8053",
        timeout_seconds=3.0
    )
```

## HTTP Health Endpoints

If your node exposes HTTP endpoints, add health check routes:

```python
from fastapi import FastAPI, status
from omninode_bridge.nodes.mixins import HealthStatus

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    health_result = await node.check_health()

    return {
        "status": health_result.overall_status.value,
        "components": [c.to_dict() for c in health_result.components],
        "uptime_seconds": health_result.uptime_seconds,
    }, health_result.overall_status.to_http_status()

@app.get("/health/liveness")
async def liveness_check():
    """Kubernetes liveness probe."""
    # Simple check: is the process running?
    return {"status": "alive"}, 200

@app.get("/health/readiness")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check if ready to accept traffic
    if node.is_healthy():
        return {"status": "ready"}, 200
    else:
        return {"status": "not_ready"}, 503
```

## Graceful Degradation

The health check system supports graceful degradation:

```python
# Non-critical components can fail without marking node unhealthy
self.register_component_check(
    "onextree",
    self._check_onextree_health,
    critical=False,  # Node can operate without OnexTree
    timeout_seconds=3.0,
)

# Check if specific service is available
async def execute_workflow(self):
    # Use OnexTree if healthy, skip if degraded
    onextree_available = any(
        c.name == "onextree" and c.status == HealthStatus.HEALTHY
        for c in (await self.check_health()).components
    )

    if onextree_available:
        intelligence = await self.get_onextree_intelligence()
    else:
        logger.info("OnexTree unavailable, skipping intelligence")
        intelligence = None
```

## Monitoring Integration

### Prometheus Metrics (Optional)

The health check system can export Prometheus metrics:

```python
from prometheus_client import Gauge, Counter

# Health status gauge (1=healthy, 0.5=degraded, 0=unhealthy)
health_status_gauge = Gauge(
    'node_health_status',
    'Current health status',
    ['node_id', 'component']
)

# Health check counter
health_check_total = Counter(
    'node_health_checks_total',
    'Total health checks performed',
    ['node_id', 'component', 'status']
)

# Update metrics after health check
async def check_health_with_metrics(self):
    result = await self.check_health()

    for component in result.components:
        # Update gauge
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
        }[component.status]

        health_status_gauge.labels(
            node_id=self.node_id,
            component=component.name
        ).set(status_value)

        # Increment counter
        health_check_total.labels(
            node_id=self.node_id,
            component=component.name,
            status=component.status.value
        ).inc()

    return result
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

# Log health check results
health_result = await node.check_health()

logger.info(
    "Health check completed",
    node_id=node.node_id,
    overall_status=health_result.overall_status.value,
    components_healthy=len([c for c in health_result.components
                           if c.status == HealthStatus.HEALTHY]),
    components_degraded=len([c for c in health_result.components
                            if c.status == HealthStatus.DEGRADED]),
    components_unhealthy=len([c for c in health_result.components
                             if c.status == HealthStatus.UNHEALTHY]),
)
```

## Best Practices

### 1. Critical vs Non-Critical Components

Mark components as critical only if their failure should prevent the node from accepting requests:

```python
# Critical: Required for core functionality
self.register_component_check(
    "database",
    self._check_database_health,
    critical=True,  # Node cannot operate without database
)

# Non-critical: Enhances functionality but not required
self.register_component_check(
    "cache",
    self._check_cache_health,
    critical=False,  # Node can operate with slower performance
)
```

### 2. Appropriate Timeouts

Set timeouts based on expected response times:

```python
# Fast local checks
self.register_component_check(
    "aggregation_buffer",
    self._check_buffer_health,
    timeout_seconds=1.0,  # Should be instant
)

# Network-based checks
self.register_component_check(
    "external_api",
    self._check_api_health,
    timeout_seconds=5.0,  # Allow for network latency
)
```

### 3. Health Check Caching

The mixin automatically caches health check results for 5 seconds by default:

```python
# First call: performs actual health checks
result1 = await node.check_health()  # ~50-100ms

# Subsequent calls within 5 seconds: returns cached result
result2 = await node.check_health()  # <1ms (cached)

# Force fresh check
result3 = await node.check_health(force=True)  # ~50-100ms
```

### 4. Component Health Details

Include useful diagnostic information in health check details:

```python
async def _check_database_health(self):
    try:
        start_time = time.time()
        await self.db_client.execute("SELECT 1")
        query_time_ms = (time.time() - start_time) * 1000

        return (
            HealthStatus.HEALTHY,
            "Database connection healthy",
            {
                "query_time_ms": query_time_ms,
                "pool_size": self.db_client.pool_size,
                "active_connections": self.db_client.active_count,
            }
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Database check failed: {e!s}",
            {
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
```

## Testing Health Checks

### Unit Testing

```python
import pytest
from omninode_bridge.nodes.mixins import HealthStatus

@pytest.mark.asyncio
async def test_health_check_all_healthy():
    """Test health check when all components are healthy."""
    node = create_test_node()

    # Mock all components as healthy
    node._check_database_health = async_mock_return(
        (HealthStatus.HEALTHY, "OK", {})
    )

    result = await node.check_health()

    assert result.overall_status == HealthStatus.HEALTHY
    assert len(result.components) > 0

@pytest.mark.asyncio
async def test_health_check_critical_failure():
    """Test health check when critical component fails."""
    node = create_test_node()

    # Mock critical component as unhealthy
    node._check_database_health = async_mock_return(
        (HealthStatus.UNHEALTHY, "Connection failed", {"error": "timeout"})
    )

    result = await node.check_health()

    assert result.overall_status == HealthStatus.UNHEALTHY
    assert not node.is_healthy()
```

### Integration Testing

```python
@pytest.mark.integration
async def test_health_check_real_components():
    """Test health check with real component connections."""
    # Start test containers
    async with TestContainers(["postgres", "kafka"]) as containers:
        node = create_node_with_real_deps(containers)

        # Wait for components to be ready
        await wait_for_healthy(node, timeout=30.0)

        # Verify health
        result = await node.check_health()
        assert result.overall_status == HealthStatus.HEALTHY

        # Verify all components
        component_names = {c.name for c in result.components}
        assert "postgres" in component_names
        assert "kafka" in component_names
```

## Troubleshooting

### Health Check Never Returns

**Symptom**: `check_health()` hangs indefinitely

**Solution**: Check component timeout configuration:

```python
# Ensure all checks have appropriate timeouts
self.register_component_check(
    "slow_service",
    self._check_slow_service,
    timeout_seconds=10.0,  # Increase if service is slow
)
```

### False Unhealthy Status

**Symptom**: Node reports unhealthy when components are actually working

**Solution**: Verify health check logic and connectivity:

```python
async def _check_database_health(self):
    try:
        # Use appropriate check method
        result = await self.db_client.execute("SELECT 1")  # ✓ Correct
        # not: await self.db_client.connect()  # ✗ Wrong

        return (HealthStatus.HEALTHY, "OK", {})
    except Exception as e:
        # Log error for debugging
        logger.error("Database health check failed", error=str(e))
        return (HealthStatus.UNHEALTHY, str(e), {})
```

### Container Restart Loop

**Symptom**: Docker container continuously restarts

**Solution**: Check Docker health check settings and startup time:

```dockerfile
# Allow sufficient startup time
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -m omninode_bridge.nodes.health_check_cli orchestrator
```

## Reference

### Complete Example: Orchestrator Node

```python
from omninode_bridge.nodes.mixins import (
    HealthCheckMixin,
    HealthStatus,
    check_http_service,
)

class NodeBridgeOrchestrator(NodeOrchestrator, HealthCheckMixin):
    def __init__(self, container: ModelONEXContainer):
        super().__init__(container)

        # Service URLs
        self.metadata_stamping_url = "http://metadata-stamping:8053"
        self.onextree_url = "http://onextree:8080"
        self.kafka_broker = "kafka:9092"

        # Initialize health checks
        self.initialize_health_checks()

    def _register_component_checks(self):
        super()._register_component_checks()

        self.register_component_check(
            "metadata_stamping",
            self._check_metadata_stamping_health,
            critical=True,
            timeout_seconds=5.0,
        )

        self.register_component_check(
            "onextree",
            self._check_onextree_health,
            critical=False,
            timeout_seconds=3.0,
        )

    async def _check_metadata_stamping_health(self):
        return await check_http_service(
            self.metadata_stamping_url,
            timeout_seconds=3.0
        )

    async def _check_onextree_health(self):
        return await check_http_service(
            self.onextree_url,
            timeout_seconds=3.0
        )
```

---

**Version**: 1.0.0
**Status**: ✅ Implemented
**Last Updated**: 2025-10-02
