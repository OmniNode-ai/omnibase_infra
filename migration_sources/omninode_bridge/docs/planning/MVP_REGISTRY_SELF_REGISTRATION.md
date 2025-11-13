# MVP Registry Self-Registration

**Status**: ⏳ **PLANNED** (Phase 1a - Pending Implementation)
**Timeline**: 1-2 days
**Priority**: Critical (Foundation for complete network topology)
**Last Updated**: October 29, 2025

## Overview

This document outlines the immediate implementation of **two-way node registration** for the MVP, enabling the registry to register itself in addition to registering other nodes. This establishes complete network topology visibility while acknowledging production security and robustness gaps that will be addressed in post-MVP phases.

### What is Two-Way Registration?

**Current State (One-Way)**:
```
Nodes → [publish introspection] → Registry → [registers nodes] → Consul + PostgreSQL
```

**Target State (Two-Way)**:
```
Nodes → [publish introspection] → Registry → [registers nodes] → Consul + PostgreSQL
   ↑                                   ↓
   └─────── [registers itself] ────────┘
```

The registry becomes **both consumer and producer** of introspection events, enabling:
- Complete network topology visibility (all nodes including registry are discoverable)
- Self-healing capabilities (registry can re-register on failure)
- Consistent node discovery patterns (registry uses same mechanism as other nodes)

## Current Implementation Status

### ✅ Complete

1. **IntrospectionMixin** - Production-ready introspection system
   - Location: `src/omninode_bridge/nodes/mixins/introspection_mixin.py`
   - Features:
     - Automatic capability extraction
     - Kafka event broadcasting (NODE_INTROSPECTION, NODE_HEARTBEAT)
     - Periodic heartbeat with configurable interval (default: 30s)
     - Registry request listener (responds to REGISTRY_REQUEST events)
     - Performance-optimized with caching (TTL-based, configurable)
   - Usage: Mixed into NodeBridgeOrchestrator, NodeBridgeReducer

2. **NodeBridgeRegistry** - Dual registration system
   - Location: `src/omninode_bridge/nodes/registry/v1_0_0/node.py`
   - Features:
     - Consumes NODE_INTROSPECTION events from Kafka
     - Performs dual registration (Consul + PostgreSQL)
     - Production-ready with circuit breakers, TTL cache, atomic registration
     - Comprehensive health checks and metrics
   - Current Gap: **Does not register itself** (one-way only)

3. **Event Infrastructure**
   - Kafka topics configured and operational:
     - `{env}.omninode_bridge.onex.evt.node-introspection.v1`
     - `{env}.omninode_bridge.onex.evt.node-heartbeat.v1`
     - `{env}.omninode_bridge.onex.evt.registry-request-introspection.v1`
   - OnexEnvelopeV1 format for all events

### ⏳ Pending (This PR)

1. **Registry Self-Registration**
   - Registry must publish its own introspection event on startup
   - Registry must register itself in both Consul and PostgreSQL
   - Registry must handle its own heartbeat broadcasting

## MVP Implementation Plan

### Phase 1a: Registry Self-Registration (This PR)

#### 1. Add IntrospectionMixin to NodeBridgeRegistry

**Changes Required**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

from ...mixins.health_mixin import HealthCheckMixin
from ...mixins.introspection_mixin import IntrospectionMixin  # NEW

class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):  # Add IntrospectionMixin
    """
    Production-ready Bridge Registry with self-registration capability.
    """

    def __init__(self, container: ModelONEXContainer, environment: str = "development") -> None:
        super().__init__(container)

        # ... existing initialization ...

        # Initialize introspection system (NEW)
        self.initialize_introspection()

        # Log initialization
        self.secure_logger.info(
            "NodeBridgeRegistry initialized with self-registration capability",
            registry_id=self.registry_id,
            introspection_enabled=True,
        )
```

**Why This Works**:
- Python MRO (Method Resolution Order) ensures IntrospectionMixin methods are available
- IntrospectionMixin is already production-ready with caching, async operations
- No breaking changes to existing registry functionality

#### 2. Publish Introspection on Startup

**Changes Required**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

async def on_startup(self) -> dict[str, Any]:
    """
    Start the registry service with self-registration.
    """
    try:
        self.secure_logger.info("Starting NodeBridgeRegistry service")

        # Initialize health status
        self.health_status = HealthStatus.UNKNOWN
        self.last_health_check = datetime.now(UTC)

        # Initialize services asynchronously
        if not self._health_check_mode:
            await self._initialize_services_async(self._container_for_init)

        # === NEW: Self-Registration ===
        # Broadcast startup introspection
        introspection_success = await self.publish_introspection(reason="startup")
        if not introspection_success:
            self.secure_logger.warning(
                "Failed to broadcast registry introspection on startup - continuing anyway"
            )

        # Register self in Consul and PostgreSQL
        await self._register_self()

        # Start introspection background tasks (heartbeat, registry listener)
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=30,
            enable_registry_listener=True,  # Registry listens for its own requests
        )
        # === END NEW ===

        # Start background tasks for consuming other nodes' introspection
        await self.start_consuming()

        # Start memory monitoring
        if self.config.memory_monitoring_interval_seconds > 0:
            self._memory_monitor_task = asyncio.create_task(
                self._memory_monitor_loop()
            )

        # Request all nodes to re-broadcast introspection
        await self._request_introspection_rebroadcast()

        self.health_status = HealthStatus.HEALTHY
        self.last_health_check = datetime.now(UTC)

        startup_info = {
            "status": "started",
            "registry_id": self.registry_id,
            "self_registered": True,  # NEW
            "environment": self.environment,
            # ... existing fields ...
        }

        return startup_info

    except Exception as e:
        # ... existing error handling ...
```

#### 3. Implement Self-Registration Helper Method

**New Method**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

async def _register_self(self) -> dict[str, Any]:
    """
    Register the registry itself in Consul and PostgreSQL.

    This method extracts registry introspection data and performs
    dual registration just like any other node.

    Returns:
        Self-registration result dictionary

    Raises:
        OnexError: If self-registration fails critically
    """
    try:
        self.secure_logger.info("Registering registry node itself")

        # Extract registry introspection data
        introspection_data = await self.get_introspection_data()

        # Create ModelNodeIntrospectionEvent from introspection data
        from ...orchestrator.v1_0_0.models.model_node_introspection_event import (
            ModelNodeIntrospectionEvent,
        )

        # Build introspection event
        self_introspection = ModelNodeIntrospectionEvent(
            node_id=self.registry_id,
            node_type="registry",
            capabilities=introspection_data.get("capabilities", {}),
            endpoints=introspection_data.get("endpoints", {}),
            metadata={
                **introspection_data.get("metadata", {}),
                "self_registration": True,
                "registry_version": "1.0.0",
            },
            timestamp=datetime.now(UTC),
        )

        # Perform dual registration (reuse existing method)
        registration_result = await self.dual_register(self_introspection)

        self.secure_logger.info(
            "Registry self-registration completed",
            registration_result=registration_result,
            consul_registered=registration_result.get("consul_registered", False),
            postgres_registered=registration_result.get("postgres_registered", False),
        )

        emit_log_event(
            LogLevel.INFO,
            "Registry self-registration completed",
            sanitize_log_data(
                {
                    "registry_id": self.registry_id,
                    "registration_result": registration_result,
                },
                self.environment,
            ),
        )

        return registration_result

    except Exception as e:
        error_context = sanitize_log_data(
            {
                "registry_id": self.registry_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
            self.environment,
        )

        self.secure_logger.error("Registry self-registration failed", **error_context)

        emit_log_event(
            LogLevel.ERROR,
            "Registry self-registration failed",
            error_context,
        )

        # WARNING: In MVP, we continue even if self-registration fails
        # This prevents registry startup from being blocked
        # In production (Phase 1b+), this should be a critical failure
        self.secure_logger.warning(
            "Continuing registry startup despite self-registration failure (MVP mode)",
            registry_id=self.registry_id,
        )

        return {
            "status": "error",
            "error": str(e),
            "consul_registered": False,
            "postgres_registered": False,
        }
```

#### 4. Extract Capabilities for Registry

**Override Method**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

async def get_capabilities(self) -> dict[str, Any]:
    """
    Get registry-specific capabilities.

    Overrides IntrospectionMixin.get_capabilities() to provide
    registry-specific capability information.

    Returns:
        Dictionary describing registry capabilities
    """
    # Get base capabilities from IntrospectionMixin
    capabilities = await super().get_capabilities()

    # Add registry-specific capabilities
    capabilities.update({
        "node_type": "registry",
        "supported_operations": [
            "register_nodes",
            "discover_services",
            "manage_registrations",
            "dual_registration",  # Consul + PostgreSQL
            "introspection_rebroadcast",
        ],
        "dual_registration": {
            "consul_enabled": self.consul_client is not None,
            "postgres_enabled": self.node_repository is not None,
            "atomic_mode": self.config.atomic_registration_enabled,
        },
        "tracking": {
            "registered_nodes_count": len(self.registered_nodes),
            "node_ttl_hours": self.node_ttl_hours,
            "cleanup_interval_hours": self.cleanup_interval_hours,
        },
        "performance": {
            "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
            "offset_tracking_enabled": self.config.offset_tracking_enabled,
            "memory_monitoring_enabled": self.config.memory_monitoring_interval_seconds > 0,
        },
    })

    return capabilities
```

#### 5. Update Envelope Metadata

**Enhanced Introspection Events**:

The introspection events will now include additional metadata for network topology:

```python
# Example of enhanced NODE_INTROSPECTION event payload (no code changes needed)
{
  "envelope_version": "1.0",
  "event_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "NODE_INTROSPECTION",
  "timestamp": "2025-10-28T12:00:00Z",
  "source_service": "omninode-bridge",
  "source_node": "registry-abc123",  # Registry's node_id
  "environment": "production",

  # NEW: Enhanced metadata for network topology
  "metadata": {
    "network_id": "omninode-network-1",      # Logical network identifier
    "deployment_id": "prod-us-west-2-001",   # Deployment instance identifier
    "epoch": "2025-10-28T00:00:00Z",         # Deployment epoch (for blue-green)
    "node_type": "registry",
    "capabilities_count": 8,
    "endpoints_count": 3,
  },

  "payload": {
    "node_id": "registry-abc123",
    "node_type": "registry",
    "capabilities": {
      # ... registry capabilities from get_capabilities() ...
    },
    "endpoints": {
      "health": "http://registry-abc123:8053/health",
      "api": "http://registry-abc123:8053/api/v1",
      "metrics": "http://registry-abc123:9090/metrics",
    },
    # ... rest of introspection payload ...
  }
}
```

**Implementation Notes**:
- `network_id`: Logical network identifier (configured via environment variable `OMNINODE_NETWORK_ID`)
- `deployment_id`: Deployment instance identifier (configured via `OMNINODE_DEPLOYMENT_ID`)
- `epoch`: Deployment epoch timestamp for blue-green deployments (configured via `OMNINODE_DEPLOYMENT_EPOCH`)

These fields are **optional** in MVP and will be added by:
1. Modifying `IntrospectionMixin.publish_introspection()` to read these environment variables
2. Including them in the envelope metadata if configured
3. Defaulting to `None` if not configured (for backward compatibility)

**Configuration**:

```bash
# Environment variables (optional for MVP)
export OMNINODE_NETWORK_ID="omninode-network-1"
export OMNINODE_DEPLOYMENT_ID="prod-us-west-2-001"
export OMNINODE_DEPLOYMENT_EPOCH="2025-10-28T00:00:00Z"
```

#### 6. Graceful Shutdown with Self-Deregistration

**Changes Required**:

```python
# File: src/omninode_bridge/nodes/registry/v1_0_0/node.py

async def on_shutdown(self) -> dict[str, Any]:
    """
    Shutdown the registry service gracefully with self-deregistration.
    """
    async with self._shutdown_lock:
        try:
            self.secure_logger.info("Shutting down NodeBridgeRegistry service")

            # Update health status
            self.health_status = HealthStatus.UNKNOWN
            self.last_health_check = datetime.now(UTC)

            # === NEW: Self-Deregistration ===
            # Broadcast shutdown introspection
            await self.publish_introspection(reason="shutdown")

            # Stop introspection background tasks (heartbeat, registry listener)
            await self.stop_introspection_tasks()
            # === END NEW ===

            # Stop consuming and cleanup tasks
            await self.stop_consuming()

            # Stop memory monitoring task
            if self._memory_monitor_task and not self._memory_monitor_task.done():
                self._memory_monitor_task.cancel()
                try:
                    await self._memory_monitor_task
                except asyncio.CancelledError:
                    pass
                self._memory_monitor_task = None

            # Stop TTL cache
            await self._offset_cache.stop()

            # ... existing cleanup ...

            return shutdown_info

        except Exception as e:
            # ... existing error handling ...
```

## Test Strategy

### Unit Tests

**New Tests Required** (`tests/unit/nodes/registry/test_registry_self_registration.py`):

```python
"""Tests for NodeBridgeRegistry self-registration functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry
from omnibase_core.models.container.model_onex_container import ModelONEXContainer


@pytest.fixture
def mock_container():
    """Create mock container for testing."""
    container = MagicMock(spec=ModelONEXContainer)
    container.value = {
        "registry_id": "test-registry-001",
        "kafka_broker_url": "localhost:29092",
        "consul_host": "localhost",
        "consul_port": 8500,
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_db": "omninode_bridge",
        "postgres_user": "postgres",
    }
    container.get_service = MagicMock(return_value=None)
    return container


@pytest.mark.asyncio
async def test_registry_self_registration_on_startup(mock_container, monkeypatch):
    """Test that registry registers itself on startup."""
    # Arrange
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
    registry = NodeBridgeRegistry(mock_container, environment="test")

    # Mock dependencies
    registry.kafka_client = AsyncMock()
    registry.consul_client = AsyncMock()
    registry.node_repository = AsyncMock()

    # Mock introspection publishing
    with patch.object(registry, 'publish_introspection', new_callable=AsyncMock) as mock_publish:
        with patch.object(registry, '_register_self', new_callable=AsyncMock) as mock_register_self:
            mock_publish.return_value = True
            mock_register_self.return_value = {
                "status": "success",
                "consul_registered": True,
                "postgres_registered": True,
            }

            # Act
            result = await registry.on_startup()

            # Assert
            assert result["status"] == "started"
            assert result["self_registered"] is True
            mock_publish.assert_called_once_with(reason="startup")
            mock_register_self.assert_called_once()


@pytest.mark.asyncio
async def test_registry_get_capabilities(mock_container, monkeypatch):
    """Test that registry capabilities include registry-specific info."""
    # Arrange
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
    registry = NodeBridgeRegistry(mock_container, environment="test")

    # Act
    capabilities = await registry.get_capabilities()

    # Assert
    assert capabilities["node_type"] == "registry"
    assert "register_nodes" in capabilities["supported_operations"]
    assert "dual_registration" in capabilities["supported_operations"]
    assert "dual_registration" in capabilities
    assert "tracking" in capabilities


@pytest.mark.asyncio
async def test_registry_self_registration_failure_continues_startup(mock_container, monkeypatch):
    """Test that registry continues startup even if self-registration fails (MVP mode)."""
    # Arrange
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_password")
    registry = NodeBridgeRegistry(mock_container, environment="test")

    # Mock dependencies
    registry.kafka_client = AsyncMock()
    registry.consul_client = None  # Force failure
    registry.node_repository = None  # Force failure

    # Mock introspection publishing to succeed
    with patch.object(registry, 'publish_introspection', new_callable=AsyncMock) as mock_publish:
        mock_publish.return_value = True

        # Act - should NOT raise exception despite self-registration failure
        result = await registry._register_self()

        # Assert
        assert result["status"] == "error"
        assert result["consul_registered"] is False
        assert result["postgres_registered"] is False
```

### Integration Tests

**New Tests Required** (`tests/integration/test_registry_self_registration_e2e.py`):

```python
"""End-to-end tests for registry self-registration."""

import pytest
import asyncio
from datetime import datetime, UTC

from omninode_bridge.nodes.registry.v1_0_0.node import NodeBridgeRegistry
from omnibase_core.models.container.model_onex_container import ModelONEXContainer


@pytest.mark.integration
@pytest.mark.asyncio
async def test_registry_self_registration_e2e(test_kafka_client, test_consul_client, test_postgres_client):
    """Test complete self-registration flow with real services."""
    # Arrange
    container = ModelONEXContainer()
    container.value = {
        "registry_id": "test-registry-e2e",
        "kafka_broker_url": "localhost:29092",
        "consul_host": "localhost",
        "consul_port": 8500,
        "postgres_host": "localhost",
        "postgres_port": 5432,
        "postgres_db": "omninode_bridge",
        "postgres_user": "postgres",
        "postgres_password": "test_password", # pragma: allowlist secret
    }

    # Register real services
    container.register_service("kafka_client", test_kafka_client)
    container.register_service("consul_client", test_consul_client)
    container.register_service("postgres_client", test_postgres_client)

    registry = NodeBridgeRegistry(container, environment="test")

    try:
        # Act
        await registry.on_startup()

        # Wait for introspection to propagate
        await asyncio.sleep(2)

        # Assert - Check Consul registration
        consul_services = await test_consul_client.get_services()
        assert "test-registry-e2e" in consul_services

        # Assert - Check PostgreSQL registration
        from omninode_bridge.services.node_registration_repository import NodeRegistrationRepository
        repo = NodeRegistrationRepository(test_postgres_client)
        registration = await repo.get_node_registration("test-registry-e2e")
        assert registration is not None
        assert registration.node_type == "registry"

        # Assert - Check introspection event was published to Kafka
        # (Consume from introspection topic and verify)
        messages = await test_kafka_client.consume_messages(
            topic="dev.omninode_bridge.onex.evt.node-introspection.v1",
            group_id="test-consumer",
            max_messages=10,
            timeout_ms=5000,
        )

        # Find registry introspection event
        registry_events = [
            msg for msg in messages
            if "test-registry-e2e" in str(msg.value)
        ]
        assert len(registry_events) > 0, "Registry introspection event not found in Kafka"

    finally:
        # Cleanup
        await registry.on_shutdown()
```

### Manual Testing Checklist

- [ ] Start registry with self-registration enabled
- [ ] Verify introspection event is published to Kafka topic
- [ ] Verify registry is registered in Consul (check Consul UI)
- [ ] Verify registry is registered in PostgreSQL (query `node_registrations` table)
- [ ] Verify heartbeat events are published periodically
- [ ] Verify registry responds to REGISTRY_REQUEST events
- [ ] Stop registry and verify shutdown introspection is published
- [ ] Restart registry and verify it re-registers successfully

## Known Limitations (MVP Mode)

### ⚠️ Security Warnings

**No Trust Model** (Phase 1b will address):
- Introspection events are **unsigned** and **unverified**
- Any node can impersonate the registry by publishing events with registry_id
- No authentication or authorization for Consul registration
- No encryption for introspection data in transit (Kafka uses plaintext)

**Mitigation for MVP**:
- Run in isolated network environment (private VPC)
- Use Kafka ACLs to restrict topic access (optional)
- Monitor for duplicate registry_id registrations (alerting)
- Document security model in deployment guide

**Production Readiness** (Post-MVP):
- Phase 1b will implement SPIFFE-based identity and signed introspection
- Phase 1b will add bootstrap controller with CA for trust
- Phase 2 will add policy engine for authorization

### ⚠️ Robustness Gaps

**No Lease Semantics** (Phase 1c will address):
- Nodes don't have TTL-based leases in Consul
- Stale registrations can remain after node crashes
- Manual cleanup required for orphaned registrations

**Mitigation for MVP**:
- TTL-based cleanup in registry (every `cleanup_interval_hours`)
- Heartbeat monitoring (detect unhealthy nodes)
- Manual deregistration API endpoint (for emergency cleanup)

**No Tombstones** (Phase 1c will address):
- Deleted nodes don't publish tombstone events
- No explicit "node left" notifications
- Topology changes are only detected via TTL expiry

**Mitigation for MVP**:
- Publish shutdown introspection event on graceful shutdown
- Monitor heartbeat gaps to detect node failures
- Use TTL cleanup to remove stale nodes

**No Topology Diff Stream** (Phase 1c will address):
- Consumers must poll for topology changes
- No incremental updates for topology changes
- Full topology refresh required for updates

**Mitigation for MVP**:
- Use heartbeat events to detect topology changes
- Poll Consul service catalog periodically
- Cache topology data with TTL refresh

### ⚠️ Multi-Tenancy Gaps

**No Network Isolation** (Phase 2 will address):
- All nodes share the same Kafka topics
- No namespace separation for different tenants
- Cross-tenant visibility in Consul service discovery

**Mitigation for MVP**:
- Single-tenant deployment only
- Document multi-tenancy limitations
- Plan for namespace support in Phase 2

## Timeline and Milestones

| Milestone | Tasks | Duration | Completion Date |
|-----------|-------|----------|----------------|
| **M1: IntrospectionMixin Integration** | Add mixin to NodeBridgeRegistry, update __init__ | 2 hours | Day 1 AM |
| **M2: Self-Registration Implementation** | Implement `_register_self()`, update `on_startup()` | 3 hours | Day 1 PM |
| **M3: Capabilities Override** | Implement `get_capabilities()` override | 1 hour | Day 1 PM |
| **M4: Unit Tests** | Write comprehensive unit tests | 3 hours | Day 2 AM |
| **M5: Integration Tests** | Write E2E tests with real services | 2 hours | Day 2 AM |
| **M6: Manual Testing** | Manual validation of all flows | 2 hours | Day 2 PM |
| **M7: Documentation** | Update guides and README | 1 hour | Day 2 PM |

**Total Estimated Effort**: 14 hours (1.75 days)

## Success Criteria

### Functional Requirements

- ✅ Registry publishes introspection event on startup
- ✅ Registry registers itself in Consul service discovery
- ✅ Registry registers itself in PostgreSQL database
- ✅ Registry publishes heartbeat events every 30 seconds
- ✅ Registry responds to REGISTRY_REQUEST events (including requests for itself)
- ✅ Registry publishes shutdown introspection on graceful shutdown
- ✅ Registry capabilities include registry-specific information
- ✅ Self-registration continues even if dual registration partially fails

### Non-Functional Requirements

- ✅ Self-registration completes within 5 seconds of startup
- ✅ No breaking changes to existing node registration flow
- ✅ Backward compatible with existing introspection consumers
- ✅ Comprehensive test coverage (>90% for new code)
- ✅ Production-ready logging with structured events
- ✅ Graceful degradation if Kafka/Consul/PostgreSQL unavailable

### Documentation Requirements

- ✅ Update NODE_INTROSPECTION_INTEGRATION.md with self-registration pattern
- ✅ Add self-registration section to BRIDGE_NODES_GUIDE.md
- ✅ Update API_REFERENCE.md with new registry capabilities
- ✅ Document security warnings and limitations in deployment guide
- ✅ Create POST_MVP_PRODUCTION_ENHANCEMENTS.md for production roadmap

## Related Documentation

- **[Node Introspection Integration Guide](../architecture/two-way-registration/NODE_INTROSPECTION_INTEGRATION.md)** - Current introspection implementation
- **[Bridge Nodes Guide](../guides/BRIDGE_NODES_GUIDE.md)** - Bridge node architecture
- **[Post-MVP Production Enhancements](./POST_MVP_PRODUCTION_ENHANCEMENTS.md)** - Production roadmap (next steps)
- **[Database Migrations Guide](../database/DATABASE_MIGRATIONS.md)** - PostgreSQL schema for node registrations

## Next Steps

After completing MVP self-registration:

1. **Phase 1b**: Implement trust and security model (SPIFFE, signed introspection)
2. **Phase 1c**: Add robustness features (lease semantics, tombstones, topology diff stream)
3. **Phase 2**: Implement multi-tenancy and policy engine
4. **Phase 3**: Add federation support for multi-network deployments

See **[POST_MVP_PRODUCTION_ENHANCEMENTS.md](./POST_MVP_PRODUCTION_ENHANCEMENTS.md)** for detailed production roadmap.

---

**Document Version**: 1.0.1
**Last Updated**: October 29, 2025 (Status Review)
**Author**: Planning Documentation Task
**Status**: ⏳ **PLANNED** - Implementation Pending

**Note**: This document describes the planned Phase 1a implementation for registry self-registration. The MVP foundation (Phase 1 & 2) was completed in October 2025, but this specific two-way registration feature is pending implementation as a follow-on enhancement.
