# Dual Registration Architecture

**Research Date:** October 24, 2025
**Status:** ⚠️ EXPERIMENTAL - Core Implementation Complete, Backend Integrations Pending
**Archon MCP Status:** ❌ Not Accessible (http://localhost:8051)

## Executive Summary

The OmniNode Bridge implements a **dual registration system** where services register in both **Consul** (for service discovery) and **PostgreSQL** (for persistent tool orchestration). This architecture provides:

- **Consul**: Dynamic service discovery, health monitoring, DNS-based lookup
- **PostgreSQL**: Persistent registry, queryable node database, tool orchestration metadata

**Current Status:**
- ✅ Architecture & Models: Complete with cache-only implementation
- ⏳ Backend Integrations: Consul, PostgreSQL, Kafka methods stubbed (not implemented)
- ❌ Database Schema: Migration exists but not applied
- ❌ Service Registration: No services currently registered in either backend
- ❌ Archon MCP Intelligence: Not accessible for RAG research

## ⚠️ Implementation Status

**Completed:**
- ✅ Service architecture and models
- ✅ In-memory cache registration
- ✅ Dual registration orchestration logic
- ✅ Event publishing infrastructure
- ✅ Health monitoring framework

**Pending (TODO):**
- ⏳ Consul service registration (method stubbed)
- ⏳ PostgreSQL database registration (method stubbed)
- ⏳ Kafka event publishing (method stubbed)
- ⏳ Health check updates (method stubbed)

**Usage:** Currently operates in **cache-only mode**. Dual registration to Consul and PostgreSQL will be completed in Phase 3.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Dual Registration System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐     Kafka Events      ┌──────────────────┐   │
│  │  ONEX Nodes  │────────────────────────>│ NodeBridge       │   │
│  │  (Bridge     │                         │ Registry         │   │
│  │  Services)   │<──────────────────────┐ │ (Consumer)       │   │
│  └──────────────┘  Introspection Request │ └────────┬─────────┘   │
│         │                                 │          │             │
│         │ Broadcast NODE_INTROSPECTION   │          │             │
│         │ events (via IntrospectionMixin) │          │             │
│         └────────────────────────────────┘          │             │
│                                                      │             │
│                              ┌───────────────────────┴────────┐   │
│                              │   Dual Registration Logic      │   │
│                              │   (Atomic Transaction)         │   │
│                              └───────────┬────────────────────┘   │
│                                          │                        │
│                      ┌───────────────────┴───────────────────┐   │
│                      │                                        │   │
│         ┌────────────▼──────────┐              ┌──────────▼───────┐
│         │  Consul Service       │              │  PostgreSQL      │
│         │  Discovery            │              │  Tool Registry   │
│         │  (Dynamic)            │              │  (Persistent)    │
│         └────────────┬──────────┘              └──────────┬───────┘
│                      │                                    │        │
│                      │                                    │        │
│         ┌────────────▼──────────┐              ┌──────────▼───────┐
│         │  • Health Checks      │              │  • Capabilities  │
│         │  • Service Tags       │              │  • Endpoints     │
│         │  • DNS Resolution     │              │  • Metadata      │
│         │  • Dynamic Discovery  │              │  • Queryable DB  │
│         └───────────────────────┘              └──────────────────┘
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Deep Dive

### 1. NodeBridgeRegistry (Central Registration Service)

**Location:** `/src/omninode_bridge/nodes/registry/v1_0_0/node.py`

**Responsibilities:**
1. Listen for `NODE_INTROSPECTION` events from Kafka
2. Deserialize `OnexEnvelopeV1` and extract introspection data
3. Perform dual registration in Consul + PostgreSQL
4. Request all nodes to re-broadcast introspection on startup
5. Monitor health of registration services

**Architectural Features (Design Complete, Backend Integration Pending):**
- **Memory Leak Prevention**: TTL cache for offset tracking (10,000 max offsets, 1-hour TTL)
- **Race Condition Prevention**: Cleanup locks for safe background task management
- **Circuit Breaker Pattern**: Robust error recovery with exponential backoff
- **Atomic Registration**: Transactional registration preventing partial state (Saga pattern) - *Architecture defined, awaiting backend implementation*
- **Security-Aware Logging**: Sensitive data masking (passwords, tokens, API keys)
- **Configurable Values**: Environment-based configuration for all magic numbers
- **Performance Monitoring**: Real-time memory usage and performance metrics

**Kafka Topics:**
- **Consumes**: `{env}.omninode_bridge.onex.evt.node-introspection.v1`
- **Produces**: `{env}.omninode_bridge.onex.evt.registry-request-introspection.v1`

**Configuration:**
```yaml
# From service_discovery.yaml
primary_strategy: "dual"  # Register in both Consul and PostgreSQL

backends:
  - consul:
      host: localhost
      port: 8500
      priority: 1
  - postgresql:
      host: localhost
      port: 5432
      table: node_registrations
      priority: 2

atomic_registration:
  enabled: true
  transaction_based: true
  rollback_on_partial_failure: true
```

**Atomic Registration Flow (Saga Pattern):**
```python
async def _atomic_dual_register(introspection: ModelNodeIntrospectionEvent):
    """
    Implements Saga pattern with compensation for rollbacks.
    Either both Consul and PostgreSQL succeed, or both fail.
    """
    try:
        # Phase 1: Register with Consul
        consul_result = await _register_with_consul(introspection)
        if not consul_result.success:
            raise OnexError("Consul registration failed")

        # Phase 2: Register with PostgreSQL
        postgres_result = await _register_with_postgres(introspection)
        if not postgres_result.success:
            # Rollback Consul registration
            await _rollback_consul_registration(node_id)
            raise OnexError("PostgreSQL registration failed")

        return (consul_result, postgres_result)

    except Exception:
        # Compensation: Rollback any successful registrations
        if consul_result.success:
            await _rollback_consul_registration(node_id)
        if postgres_result.success:
            await _rollback_postgres_registration(node_id)
        raise
```

### 2. Consul Service Discovery

**Location:** `/src/omninode_bridge/services/metadata_stamping/registry/consul_client.py`

**Class:** `RegistryConsulClient`

**Purpose:** Register services for dynamic service discovery with health monitoring

**Registration Details:**
```python
# Service registration structure
{
    "name": "{node_type}-{node_id}",  # e.g., "orchestrator-abc123"
    "id": "{node_type}-{node_id}-{instance_id}",
    "address": "localhost",
    "port": 8080,  # Extracted from node endpoints
    "tags": ["omninode", "bridge", "onex-v2.0"],
    "meta": {
        "node_version": "1.0.0",
        "onex_version": "2.0",
        "environment": "development",
        "node_type": "orchestrator",
        "capabilities": "workflow,state_management"
    },
    "checks": [
        {
            "http": "http://localhost:8080/health",
            "interval": "30s",
            "timeout": "5s",
            "deregister_critical_service_after": "90s"
        }
    ]
}
```

**Health Check Integration:**
- Consul actively monitors registered services via HTTP health checks
- Failed health checks → Service marked unhealthy → Removed from DNS
- Health endpoints must respond within 5 seconds
- Critical failures trigger automatic deregistration after 90 seconds

**Current Status:**
```bash
$ curl -s http://localhost:28500/v1/agent/services | python3 -m json.tool
{}  # No services currently registered
```

### 3. PostgreSQL Tool Registry

**Location:** `/migrations/005_create_node_registrations.sql`

**Table Schema:**
```sql
CREATE TABLE IF NOT EXISTS node_registrations (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100) NOT NULL,
    node_version VARCHAR(50) NOT NULL,
    capabilities JSONB DEFAULT '{}',
    endpoints JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    health_status VARCHAR(50) NOT NULL DEFAULT 'UNKNOWN',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_node_registrations_health ON node_registrations(health_status);
CREATE INDEX idx_node_registrations_type ON node_registrations(node_type);
CREATE INDEX idx_node_registrations_last_heartbeat ON node_registrations(last_heartbeat DESC);
CREATE INDEX idx_node_registrations_type_health ON node_registrations(node_type, health_status);
```

**Current Status:**
```bash
$ docker exec omninode-bridge-postgres psql -U postgres -d omninode_bridge -c "\d node_registrations"
Did not find any relation named "node_registrations".
```
**⚠️ Migration not applied** - Table does not exist in database

### 4. IntrospectionMixin (Node Self-Discovery)

**Location:** `/src/omninode_bridge/nodes/mixins/introspection_mixin.py`

**Purpose:** Nodes use this mixin to broadcast their capabilities to Kafka

**Key Features:**
- **Automatic capability detection**: Endpoints, operations, resources
- **FSM state introspection**: For workflow nodes (orchestrator, reducer)
- **Performance metrics**: Resource monitoring with psutil
- **Dynamic configuration discovery**: Non-sensitive config extraction
- **Health check integration**: Health status reporting

**Introspection Data Structure:**
```python
{
    "node_id": "orchestrator-v1_0_0-abc123",
    "node_name": "NodeBridgeOrchestrator",
    "node_type": "orchestrator",
    "node_version": "1.0.0",
    "capabilities": {
        "node_type": "orchestrator",
        "fsm_states": {
            "supported_states": ["PENDING", "PROCESSING", "COMPLETED", "FAILED"],
            "active_workflows": 5,
            "active_states": ["PROCESSING", "COMPLETED"]
        },
        "supported_operations": [
            "execute_orchestration",
            "coordinate_workflow",
            "manage_state",
            "route_operations"
        ],
        "resource_limits": {
            "cpu_cores_available": 8,
            "memory_available_gb": 12.5,
            "current_memory_usage_mb": 245.6
        }
    },
    "endpoints": {
        "health": "http://localhost:8080/health",
        "api": "http://localhost:8080",
        "orchestration": "http://localhost:8080"
    },
    "configuration": {
        "node_ttl_hours": 24,
        "cleanup_interval_hours": 1
    },
    "current_state": {
        "status": "active",
        "uptime_seconds": 3600,
        "workflow_states": {
            "active_workflows": 5,
            "active_states": ["PROCESSING"]
        },
        "health_status": "healthy"
    },
    "introspection_timestamp": "2025-10-24T12:00:00Z",
    "introspection_version": "2.0.0-optimized"
}
```

**Broadcasting Triggers:**
1. **Startup**: Node broadcasts introspection on initialization
2. **Periodic Heartbeat**: Every 30 seconds (configurable)
3. **Registry Request**: When NodeBridgeRegistry requests re-broadcast
4. **Manual Trigger**: Via `publish_introspection(reason="manual")`

**Performance Optimizations:**
- **Cached Imports**: Enum imports cached at module level
- **Lazy Loading**: Heavy operations only performed when needed
- **Memoization**: Results of expensive operations cached (5-minute TTL)
- **Optimized Reflection**: Reduced attribute access overhead
- **Efficient State Discovery**: FSM state discovery optimized with caching

### 5. NodeRegistrationRepository (Database Layer)

**Location:** `/src/omninode_bridge/services/node_registration_repository.py`

**Purpose:** Provides CRUD operations for PostgreSQL registry

**Expected Methods:**
```python
class NodeRegistrationRepository:
    async def create_registration(self, registration_create):
        """Insert new node registration"""

    async def update_registration(self, node_id, registration_update):
        """Update existing node registration"""

    async def delete_node_registration(self, node_id):
        """Remove node from registry"""

    async def get_registration(self, node_id):
        """Retrieve node registration by ID"""

    async def query_registrations(self, filters):
        """Query registrations with filters"""
```

**Note:** Implementation currently uses PostgresClient with connection pooling (10-50 connections)

## Registration Flow

### Startup Registration Flow

```
1. Service Starts
   ↓
2. Initialize IntrospectionMixin
   ↓
3. Gather introspection data (capabilities, endpoints, state)
   ↓
4. Create ModelNodeIntrospectionEvent
   ↓
5. Wrap in OnexEnvelopeV1
   ↓
6. Publish to Kafka topic: node-introspection.v1
   ↓
7. NodeBridgeRegistry consumes event
   ↓
8. Validate introspection data
   ↓
9. Atomic dual registration:
   ├─ 9a. Register in Consul (health checks, tags, metadata)
   │  └─ Success? → Continue
   │  └─ Failure? → Abort (no partial registration)
   │
   └─ 9b. Register in PostgreSQL (capabilities, endpoints, metadata)
      └─ Success? → Complete
      └─ Failure? → Rollback Consul registration
   ↓
10. Update in-memory cache in NodeBridgeRegistry
   ↓
11. Publish confirmation event (optional)
```

### Runtime Updates Flow

```
1. Node state changes (e.g., workflow completes)
   ↓
2. Node detects change via IntrospectionMixin
   ↓
3. Publish NODE_INTROSPECTION event (reason: "state_change")
   ↓
4. NodeBridgeRegistry updates both registries
   ↓
5. Consul health checks continue monitoring
   ↓
6. PostgreSQL updated with latest state
```

### Deregistration Flow

```
1. Node Shutdown
   ↓
2. Publish NODE_DEREGISTRATION event
   ↓
3. NodeBridgeRegistry deregisters from Consul
   ↓
4. Mark inactive in PostgreSQL (soft delete)
   ↓
5. TTL cleanup removes stale nodes after 24 hours
```

## Use Cases

### When to Use Consul

**Best For:**
- **Dynamic service discovery**: Find healthy service instances at runtime
- **DNS-based lookup**: `curl http://orchestrator-service.service.consul:8080/health`
- **Load balancing**: Consul provides built-in DNS round-robin
- **Health monitoring**: Automatic removal of unhealthy services
- **Short-lived services**: Ephemeral containers, auto-scaling groups

**Example:**
```bash
# Discover all healthy orchestrator instances
dig @127.0.0.1 -p 8600 orchestrator-service.service.consul

# Query via HTTP API
curl http://localhost:28500/v1/health/service/orchestrator-service?passing=true
```

### When to Use PostgreSQL Registry

**Best For:**
- **Persistent registry**: Long-term node catalog
- **Complex queries**: "Find all orchestrator nodes with capability X running version > 1.2.0"
- **Tool orchestration**: Dynamic tool selection based on capabilities
- **Analytics**: Track registration history, node lifecycle
- **Metadata-rich queries**: Search by capabilities, endpoints, custom metadata

**Example:**
```sql
-- Find all healthy orchestrator nodes with specific capability
SELECT node_id, endpoints, capabilities
FROM node_registrations
WHERE node_type = 'orchestrator'
  AND health_status = 'HEALTHY'
  AND capabilities @> '{"fsm_states": {"supported_states": ["PROCESSING"]}}'::jsonb
ORDER BY last_heartbeat DESC;
```

### Dual Registration Benefits

**Complementary Strengths:**
1. **Consul**: Real-time health monitoring + dynamic discovery
2. **PostgreSQL**: Rich querying + historical data

**Use Case Example:**
```python
# 1. Query PostgreSQL for nodes with required capability
nodes = await registry.query_nodes({
    "node_type": "orchestrator",
    "capability": "workflow_coordination"
})

# 2. Use Consul to find healthy instances among matched nodes
for node in nodes:
    healthy_instances = await consul.health.service(
        node.node_id,
        passing=True
    )

    # 3. Select instance from Consul (load balanced, health-checked)
    if healthy_instances:
        selected = healthy_instances[0]
        await route_request_to(selected['Service']['Address'], selected['Service']['Port'])
```

## Security Considerations

### Password Management

**Best Practice (RECOMMENDED):**
```bash
# Set PostgreSQL password via environment variable
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:?Password not set}"

# Alternative: Use a secure value directly (NOT recommended for production)
# export POSTGRES_PASSWORD="<YOUR_SECURE_PASSWORD_HERE>"
```

**⚠️ SECURITY BEST PRACTICES:**
- **NEVER** hardcode passwords in code or configuration files
- **ALWAYS** use environment variables or secrets management systems
- **NEVER** commit passwords to version control
- Use strong, unique passwords (minimum 16 characters, mix of letters/numbers/symbols)
- Rotate passwords regularly (every 90 days recommended)

**NodeBridgeRegistry Security:**
- ✅ Always prefers `POSTGRES_PASSWORD` environment variable
- ✅ Warns if using container config for passwords
- ✅ Sanitizes all log output (removes passwords, tokens, secrets)
- ✅ Validates password is configured before connecting

**Logging Security:**
```python
from omninode_bridge.utils.secure_logging import sanitize_log_data

# Automatically masks sensitive fields
log_data = {
    "postgres_host": "localhost",
    "postgres_password": "<REDACTED>",  # Example only - will be masked automatically
    "node_id": "abc123"
}

secure_logger.info("Database connected", **sanitize_log_data(log_data, environment))
```

**Masked Fields:**
- `password`, `pwd`, `secret`, `token`, `api_key`
- `private_key`, `certificate`, `credential`
- `auth`, `authorization`, `bearer`
- `session`, `cookie`, `csrf`, `jwt`, `oauth`

### Consul Security

**Authentication:**
```bash
# Set Consul token via environment variable
export CONSUL_HTTP_TOKEN="consul_acl_token"
```

**Configuration:**
```yaml
consul:
  token: null  # Set via CONSUL_HTTP_TOKEN environment variable
  scheme: http  # Use https in production
  datacenter: dc1
  namespace: omninode-bridge
```

## Performance Characteristics

### NodeBridgeRegistry Performance

**Metrics:**
```python
registration_metrics = {
    "total_registrations": 1523,
    "successful_registrations": 1498,
    "failed_registrations": 25,
    "consul_registrations": 1495,
    "postgres_registrations": 1492,
    "current_nodes_count": 142,
    "memory_usage_mb": 45.2,
    "cleanup_operations_count": 6
}
```

**Targets:**
- **Registration latency**: <100ms per operation
- **Throughput**: 100+ concurrent registrations per second
- **Memory usage**: <512MB under normal load
- **Cache hit rate**: >80% for introspection data

**Circuit Breaker Settings:**
```python
circuit_breaker:
  registration_circuit_breaker:
    failure_threshold: 5  # Open after 5 consecutive failures
    success_threshold: 2  # Close after 2 consecutive successes
    timeout_seconds: 60   # Stay open for 60 seconds
    half_open_max_calls: 3  # Allow 3 test calls in half-open state
```

### TTL and Cleanup

**Node Lifecycle:**
```yaml
node_tracking:
  max_registered_nodes: 10000
  node_ttl_hours: 24  # Nodes expire after 24 hours without heartbeat
  cleanup_interval_hours: 1  # Run cleanup every hour
  cleanup_terminal_nodes: true
  track_last_seen: true
  track_heartbeat: true
```

**Offset Cache (Memory Leak Prevention):**
```yaml
memory_leak_prevention:
  enabled: true
  offset_cache_type: "ttl_cache"
  max_tracked_offsets: 10000  # Max offsets to track
  offset_cache_ttl_seconds: 3600  # 1 hour TTL
  offset_cleanup_interval_seconds: 300  # Cleanup every 5 minutes
```

## Monitoring and Observability

### Health Check Endpoints

**NodeBridgeRegistry Health:**
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "registry_id": "registry-abc123",
  "environment": "development",
  "timestamp": "2025-10-24T12:00:00Z",
  "checks": {
    "background_tasks": {
      "consumer_task_running": true,
      "cleanup_task_running": true,
      "memory_monitor_running": true
    },
    "services": {
      "kafka_client_available": true,
      "consul_client_available": true,
      "postgres_client_available": true,
      "node_repository_available": true
    },
    "circuit_breakers": {
      "registration_circuit": "closed",
      "kafka_circuit": "closed"
    }
  },
  "metrics": {
    "registration": {
      "total_registrations": 1523,
      "successful_registrations": 1498,
      "failed_registrations": 25
    },
    "memory": {
      "nodes_before_cleanup": 150,
      "nodes_after_cleanup": 142,
      "memory_freed_mb": 8.3
    }
  }
}
```

### Metrics Collection

**Prometheus Metrics:**
```
# Registration metrics
omninode_registry_total_registrations{environment="dev"} 1523
omninode_registry_successful_registrations{environment="dev"} 1498
omninode_registry_failed_registrations{environment="dev"} 25

# Backend metrics
omninode_registry_consul_registrations{environment="dev"} 1495
omninode_registry_postgres_registrations{environment="dev"} 1492

# Performance metrics
omninode_registry_registration_latency_ms{quantile="0.95"} 45
omninode_registry_registration_latency_ms{quantile="0.99"} 87

# Memory metrics
omninode_registry_memory_usage_mb{environment="dev"} 45.2
omninode_registry_nodes_count{environment="dev"} 142
```

## Configuration Reference

### Environment Variables

```bash
# Kafka Configuration
KAFKA_BROKER_URL=localhost:29092
KAFKA_ENVIRONMENT=dev  # Prefix for Kafka topics

# Consul Configuration
CONSUL_HOST=localhost
CONSUL_PORT=8500
CONSUL_HTTP_TOKEN=  # Optional ACL token

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=omninode_bridge
POSTGRES_USER=postgres
POSTGRES_PASSWORD=  # REQUIRED - Set via secrets manager

# Registry Configuration
NODE_TTL_HOURS=24
CLEANUP_INTERVAL_HOURS=1
MAX_TRACKED_OFFSETS=10000
OFFSET_CACHE_TTL_SECONDS=3600

# Production Features
CIRCUIT_BREAKER_ENABLED=true
ATOMIC_REGISTRATION_ENABLED=true
MEMORY_MONITORING_ENABLED=true
```

### Contract Configuration

**Location:** `/src/omninode_bridge/nodes/registry/v1_0_0/contracts/service_discovery.yaml`

**Key Settings:**
```yaml
primary_strategy: "dual"

backends:
  - backend_name: "consul"
    enabled: true
    priority: 1

  - backend_name: "postgresql"
    enabled: true
    priority: 2

production_features:
  memory_leak_prevention:
    enabled: true
  race_condition_prevention:
    enabled: true
  circuit_breaker:
    enabled: true
  atomic_registration:
    enabled: true
  security_aware_logging:
    enabled: true
```

## Troubleshooting

### No Services Appearing in Consul

**Check:**
1. Consul is running: `docker ps | grep consul`
2. Consul is accessible: `curl http://localhost:28500/v1/agent/services`
3. NodeBridgeRegistry is running: `docker ps | grep registry`
4. Kafka topics exist: Check introspection topic
5. Services are broadcasting: Check IntrospectionMixin integration

**Debug:**
```bash
# Check Consul services
curl http://localhost:28500/v1/agent/services | python3 -m json.tool

# Check Consul health
curl http://localhost:28500/v1/agent/checks | python3 -m json.tool

# Check NodeBridgeRegistry logs
docker logs omninode-bridge-registry --tail 100
```

### PostgreSQL Table Missing

**Problem:**
```bash
$ docker exec omninode-bridge-postgres psql -U postgres -d omninode_bridge -c "\d node_registrations"
Did not find any relation named "node_registrations".
```

**Solution:**
```bash
# Apply migration 005
docker exec omninode-bridge-postgres psql -U postgres -d omninode_bridge < migrations/005_create_node_registrations.sql

# Verify table exists
docker exec omninode-bridge-postgres psql -U postgres -d omninode_bridge -c "\d node_registrations"
```

### Circuit Breaker Open

**Symptoms:**
- Registration failures with "Circuit breaker open" error
- Health check shows `circuit_breakers.registration_circuit: "open"`

**Diagnosis:**
```python
# Check circuit breaker status
circuit_status = await registry._registration_circuit_breaker.get_status()
print(circuit_status)
# {
#   "state": "open",
#   "failure_count": 5,
#   "last_failure_time": "2025-10-24T12:00:00Z"
# }
```

**Resolution:**
1. Fix underlying issue (Consul/PostgreSQL connectivity)
2. Wait for timeout (60 seconds default)
3. Circuit breaker will transition to half-open
4. 2 successful registrations will close circuit

### Memory Leak Detection

**Symptoms:**
- Memory usage steadily increasing
- Offset cache growing unbounded

**Diagnosis:**
```python
# Check memory metrics
memory_metrics = registry.memory_metrics
print(memory_metrics)

# Check offset cache
cache_metrics = registry._offset_cache.get_metrics()
print(cache_metrics)
```

**Resolution:**
- Enable TTL cache: `offset_cache_type: "ttl_cache"`
- Reduce max offsets: `max_tracked_offsets: 5000`
- Decrease TTL: `offset_cache_ttl_seconds: 1800`
- Enable memory monitoring: `memory_monitoring_enabled: true`

## Future Enhancements

### Planned Features

1. **Multi-Region Support**
   - Cross-datacenter Consul replication
   - PostgreSQL read replicas for geographically distributed queries

2. **Advanced Health Checks**
   - Custom health check scripts
   - Dependency health propagation (e.g., orchestrator depends on database)

3. **Service Mesh Integration**
   - Consul Connect for service-to-service TLS
   - Envoy sidecar proxies for advanced routing

4. **Enhanced Analytics**
   - Registration history tracking
   - Node lifecycle analytics (uptime, failure patterns)
   - Capability evolution tracking

5. **GraphQL Query API**
   - Rich query interface for PostgreSQL registry
   - Subscriptions for real-time registry updates

## References

### Internal Documentation

- **NodeBridgeRegistry**: `/src/omninode_bridge/nodes/registry/v1_0_0/node.py`
- **RegistryConsulClient**: `/src/omninode_bridge/services/metadata_stamping/registry/consul_client.py`
- **IntrospectionMixin**: `/src/omninode_bridge/nodes/mixins/introspection_mixin.py`
- **Service Discovery Contract**: `/src/omninode_bridge/nodes/registry/v1_0_0/contracts/service_discovery.yaml`
- **Migration 005**: `/migrations/005_create_node_registrations.sql`

### External References

- **Consul Documentation**: https://www.consul.io/docs
- **Consul Service Discovery**: https://www.consul.io/docs/discovery
- **Consul Health Checks**: https://www.consul.io/docs/discovery/checks
- **PostgreSQL JSONB**: https://www.postgresql.org/docs/current/datatype-json.html
- **Saga Pattern**: https://microservices.io/patterns/data/saga.html

## Appendix: Complete Registration Code Example

```python
from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry
from omnibase_core.models.core import ModelContainer

# Create container with configuration
container = ModelContainer(config={
    "registry_id": "registry-prod-001",
    "kafka_broker_url": "localhost:29092",
    "consul_host": "localhost",
    "consul_port": 8500,
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_db": "omninode_bridge",
    "postgres_user": "postgres",
    # IMPORTANT: Set POSTGRES_PASSWORD environment variable
    # postgres_password should NOT be in config for security
})

# Initialize registry
registry = NodeBridgeRegistry(container, environment="production")

# Start registry (initializes async services)
await registry.on_startup()

# Registry is now:
# 1. Consuming NODE_INTROSPECTION events from Kafka
# 2. Performing dual registration for each event
# 3. Running periodic cleanup of stale nodes
# 4. Monitoring memory usage

# Check health
health = await registry.health_check()
print(health['status'])  # "healthy"

# Get metrics
metrics = registry.get_metrics()
print(metrics['registration_metrics'])

# Shutdown gracefully
await registry.on_shutdown()
```

---

**Document Version:** 1.0
**Last Updated:** October 24, 2025
**Status:** ⚠️ Implementation Complete, Deployment Pending
