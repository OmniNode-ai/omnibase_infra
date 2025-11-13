# OmniBase Core Mixin Catalog

**Version**: 1.0
**Created**: 2025-11-04
**Purpose**: Comprehensive reference for all 33 omnibase_core mixins
**Audience**: Code generator developers, node implementers

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Usage Guidelines](#usage-guidelines)
- [Mixin Categories](#mixin-categories)
  - [Health & Monitoring](#health--monitoring)
  - [Event-Driven Patterns](#event-driven-patterns)
  - [Service Integration](#service-integration)
  - [Execution Patterns](#execution-patterns)
  - [Data Handling](#data-handling)
  - [Serialization](#serialization)
  - [Contract & Metadata](#contract--metadata)
  - [CLI & Debugging](#cli--debugging)
- [NodeEffect Built-in Features](#nodeeffect-built-in-features)
- [Selection Flowchart](#selection-flowchart)
- [Common Patterns](#common-patterns)

---

## Quick Reference

**Total Mixins**: 33
**Import Path**: `omnibase_core.mixins.*`
**Documentation Level**: Production-Ready

### Quick Lookup Table

| Mixin | Category | Dependencies | NodeEffect Overlap | Use When |
|-------|----------|--------------|-------------------|----------|
| **MixinHealthCheck** | Health & Monitoring | None | NONE | Need component health monitoring |
| **MixinMetrics** | Health & Monitoring | None | Partial (NodeEffect has basic metrics) | Need detailed performance metrics |
| **MixinLogData** | Health & Monitoring | None | NONE | Need structured logging |
| **MixinRequestResponseIntrospection** | Health & Monitoring | None | NONE | Need request/response tracking |
| **MixinEventDrivenNode** | Event-Driven | MixinEventHandler, MixinNodeLifecycle, MixinIntrospectionPublisher | NONE | Building event-driven nodes |
| **MixinEventBus** | Event-Driven | None | NONE | Need event bus integration |
| **MixinEventHandler** | Event-Driven | None | NONE | Handling event requests |
| **MixinEventListener** | Event-Driven | None | NONE | Consuming events |
| **MixinIntrospectionPublisher** | Event-Driven | None | NONE | Publishing introspection events |
| **MixinServiceRegistry** | Service Integration | MixinEventBus (optional) | NONE | Need service discovery |
| **MixinDiscoveryResponder** | Service Integration | None | NONE | Responding to discovery requests |
| **MixinNodeService** | Service Integration | None | NONE | Node-as-service pattern |
| **MixinNodeExecutor** | Execution | MixinEventDrivenNode | NONE | Persistent executor mode |
| **MixinNodeLifecycle** | Execution | None | NONE | Lifecycle management |
| **MixinNodeSetup** | Execution | None | NONE | Setup helpers |
| **MixinHybridExecution** | Execution | None | NONE | Sync/async hybrid patterns |
| **MixinToolExecution** | Execution | None | NONE | Tool execution framework |
| **MixinWorkflowSupport** | Execution | None | NONE | DAG workflow support |
| **MixinHashComputation** | Data Handling | None | NONE | Hash utilities |
| **MixinCaching** | Data Handling | None | PARTIAL (NodeEffect has basic caching) | Need result caching |
| **MixinLazyEvaluation** | Data Handling | None | NONE | Lazy loading patterns |
| **MixinCompletionData** | Data Handling | None | NONE | Completion tracking |
| **MixinCanonicalYAMLSerializer** | Serialization | None | NONE | Canonical YAML serialization |
| **MixinYAMLSerialization** | Serialization | None | NONE | Standard YAML operations |
| **MixinSerializable** | Serialization | None | NONE | Generic serialization |
| **MixinRedaction** | Serialization | None | NONE | Sensitive data redaction |
| **MixinContractMetadata** | Contract & Metadata | None | NONE | Contract metadata handling |
| **MixinContractStateReducer** | Contract & Metadata | None | NONE | Contract state reduction |
| **MixinIntrospectFromContract** | Contract & Metadata | None | NONE | Contract-driven introspection |
| **MixinNodeIdFromContract** | Contract & Metadata | None | NONE | Node ID from contract |
| **MixinNodeIntrospection** | Contract & Metadata | None | NONE | Node introspection support |
| **MixinCLIHandler** | CLI & Debugging | None | NONE | CLI command handling |
| **MixinDebugDiscoveryLogging** | CLI & Debugging | None | NONE | Debug discovery logging |
| **MixinFailFast** | CLI & Debugging | None | NONE | Fail-fast validation |

---

## Usage Guidelines

### When to Use Mixins

**✅ USE MIXINS when:**
- Adding cross-cutting concerns (logging, metrics, health checks)
- Implementing event-driven capabilities
- Need service discovery or registration
- Require serialization or caching
- Building reusable node patterns

**❌ DON'T USE MIXINS when:**
- Feature is built into NodeEffect (circuit breakers, retry, transactions)
- Simple one-off functionality specific to single node
- Performance-critical hot path (prefer inline code)

### Mixin Selection Process

1. **Check NodeEffect Built-ins First** - Many features are already available
2. **Identify Node Type** - Effect, Compute, Orchestrator, Reducer
3. **List Required Capabilities** - Health checks, events, metrics, etc.
4. **Check Dependencies** - Some mixins require others
5. **Review Overlap** - Avoid duplicate implementations
6. **Test Integration** - Ensure mixins work together

---

## Mixin Categories

### Health & Monitoring

#### MixinHealthCheck

**Module**: `omnibase_core.mixins.mixin_health_check`
**Class**: `MixinHealthCheck`

**Purpose**: Provides standardized health check implementation with comprehensive error handling and async support.

**Capabilities**:
- Standard health check endpoint
- Dependency health aggregation
- Custom health check hooks
- Async and sync support
- Component-level health monitoring

**Configuration**: None (configure via method overrides)

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.models.core.model_health_status import ModelHealthStatus
from omnibase_core.enums.enum_node_health_status import EnumNodeHealthStatus

class NodeDatabaseAdapterEffect(NodeEffect, MixinHealthCheck):
    """Database adapter with health monitoring."""

    def get_health_checks(self) -> list[Callable]:
        """Register custom health checks."""
        return [
            self._check_database_health,
            self._check_kafka_health,
        ]

    def _check_database_health(self) -> ModelHealthStatus:
        """Check database connectivity."""
        try:
            # Perform DB health check
            is_healthy = self.postgres_client.check_connection()

            return ModelHealthStatus(
                status=EnumNodeHealthStatus.HEALTHY if is_healthy else EnumNodeHealthStatus.UNHEALTHY,
                message=f"Database {'available' if is_healthy else 'unavailable'}",
                timestamp=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            return ModelHealthStatus(
                status=EnumNodeHealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                timestamp=datetime.now(UTC).isoformat(),
            )

    async def _check_kafka_health(self) -> ModelHealthStatus:
        """Check Kafka connectivity (async)."""
        try:
            is_healthy = await self.kafka_client.check_health()

            return ModelHealthStatus(
                status=EnumNodeHealthStatus.HEALTHY if is_healthy else EnumNodeHealthStatus.DEGRADED,
                message=f"Kafka {'available' if is_healthy else 'degraded'}",
                timestamp=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            return ModelHealthStatus(
                status=EnumNodeHealthStatus.UNHEALTHY,
                message=f"Kafka check failed: {str(e)}",
                timestamp=datetime.now(UTC).isoformat(),
            )
```

**Common Patterns**:
- Register health checks in `__init__` or `initialize()`
- Return `ModelHealthStatus` from each check
- Support both sync and async health checks
- Mark critical vs non-critical components

**When to Use**:
- Node interacts with external services (databases, APIs, message queues)
- Need component-level health monitoring
- Integration with health check endpoints
- Kubernetes liveness/readiness probes

---

#### MixinMetrics

**Module**: `omnibase_core.mixins.mixin_metrics`
**Class**: `MixinMetrics`

**Purpose**: Provides performance metrics collection capabilities (stub implementation - full metrics with Prometheus/StatsD planned).

**Capabilities**:
- Record arbitrary metrics (gauges, counters, histograms)
- Increment counters
- Get metrics snapshot
- Reset metrics

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: PARTIAL - NodeEffect has basic performance metrics; this adds comprehensive collection

**Usage Example**:

```python
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.nodes.node_effect import NodeEffect

class NodeLLMEffect(NodeEffect, MixinMetrics):
    """LLM effect with metrics tracking."""

    async def execute_effect(self, contract):
        """Execute with metrics collection."""
        start_time = time.time()

        try:
            # Track request count
            self.increment_counter("llm_requests_total")

            # Execute LLM call
            response = await self.llm_client.generate(contract.input_data)

            # Track latency
            latency_ms = (time.time() - start_time) * 1000
            self.record_metric("llm_latency_ms", latency_ms, tags={"model": "gpt-4"})

            # Track tokens
            self.record_metric("llm_tokens_used", response.total_tokens, tags={"type": "completion"})

            # Track cost
            cost = response.total_tokens * 0.00003
            self.record_metric("llm_cost_usd", cost)

            self.increment_counter("llm_requests_success")

            return response

        except Exception as e:
            self.increment_counter("llm_requests_failed")
            raise

    def get_metrics_summary(self) -> dict:
        """Get metrics summary for reporting."""
        metrics = self.get_metrics()
        return {
            "total_requests": metrics.get("llm_requests_total", {}).get("value", 0),
            "success_rate": self._calculate_success_rate(metrics),
            "avg_latency_ms": self._calculate_avg_latency(metrics),
            "total_cost_usd": metrics.get("llm_cost_usd", {}).get("value", 0.0),
        }
```

**Common Patterns**:
- Record metrics at operation boundaries
- Use tags for metric dimensions (model, operation type, status)
- Increment counters for events (requests, errors, successes)
- Record gauges for measurements (latency, size, cost)

**When to Use**:
- Need detailed performance metrics
- Tracking operation costs (LLM tokens, API calls)
- Performance monitoring and alerting
- Capacity planning and optimization

---

#### MixinLogData

**Module**: `omnibase_core.mixins.mixin_log_data`
**Class**: `MixinLogData`

**Purpose**: Provides structured logging data model for ONEX nodes.

**Capabilities**:
- Structured log data container
- Integration with `emit_log_event`
- Type-safe log context

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_log_data import MixinLogData
from omnibase_core.logging.structured import emit_log_event
from omnibase_core.enums.enum_log_level import EnumLogLevel

# Create log context
log_data = MixinLogData(
    node_name="database_adapter",
    operation="execute_query",
    correlation_id=str(correlation_id),
)

# Emit structured log
emit_log_event(
    EnumLogLevel.INFO,
    "Executing database query",
    log_data,
)
```

**When to Use**:
- Structured logging with correlation IDs
- Integration with log aggregation systems
- Debug and troubleshooting

---

#### MixinRequestResponseIntrospection

**Module**: `omnibase_core.mixins.mixin_request_response_introspection`
**Class**: `MixinRequestResponseIntrospection`

**Purpose**: Tracks request and response data for real-time service discovery.

**Capabilities**:
- Capture request/response pairs
- Real-time discovery protocol support
- Introspection data collection

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_request_response_introspection import MixinRequestResponseIntrospection

class MyNode(MixinEventDrivenNode, MixinRequestResponseIntrospection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_request_response_introspection()

    def cleanup(self):
        self._teardown_request_response_introspection()
```

**When to Use**:
- Real-time service discovery
- API monitoring and analysis
- Request/response logging

---

### Event-Driven Patterns

#### MixinEventDrivenNode

**Module**: `omnibase_core.mixins.mixin_event_driven_node`
**Class**: `MixinEventDrivenNode`

**Purpose**: Canonical mixin for event-driven ONEX nodes. Composes multiple focused mixins for complete event-driven capabilities.

**Capabilities**:
- Automatic node registration and lifecycle management
- Event handler setup for introspection and discovery
- Auto-publishing of introspection events
- Request-response introspection
- All communication via event bus

**Configuration**:

```python
def __init__(
    self,
    node_id: UUID,
    event_bus: ProtocolEventBus,  # Required
    metadata_loader: ProtocolSchemaLoader | None = None,
    registry: Any = None,
    **kwargs
):
    ...
```

**Dependencies**:
- `MixinEventHandler` (composed)
- `MixinNodeLifecycle` (composed)
- `MixinIntrospectionPublisher` (composed)
- `MixinRequestResponseIntrospection` (composed)

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_event_driven_node import MixinEventDrivenNode
from uuid import uuid4

class NodeMyEventDrivenEffect(MixinEventDrivenNode):
    """Fully event-driven effect node."""

    def __init__(self, event_bus, metadata_loader, **kwargs):
        super().__init__(
            node_id=uuid4(),
            event_bus=event_bus,
            metadata_loader=metadata_loader,
            **kwargs
        )

    def get_capabilities(self) -> list[str]:
        """Define node capabilities."""
        return ["data_processing", "transformation", "validation"]

    def supports_introspection(self) -> bool:
        """Enable introspection."""
        return True
```

**Common Patterns**:
- Use as base for all event-driven nodes
- Override `get_capabilities()` for discovery
- Override `get_introspection_data()` for detailed metadata
- Call `cleanup_event_handlers()` on shutdown

**When to Use**:
- Building event-driven nodes
- Need automatic service discovery
- Integration with event bus systems
- MCP/GraphQL integration

---

#### MixinEventBus

**Module**: `omnibase_core.mixins.mixin_event_bus`
**Class**: `MixinEventBus`

**Purpose**: Unified event bus operations including subscription, listening, and completion publishing.

**Capabilities**:
- Event subscription and listening
- Event completion publishing
- Protocol-based polymorphism
- Async and sync support
- Error handling and structured logging

**Configuration**:

```python
node_name: str  # Node identifier
registry: object | None  # Registry with event bus access
event_bus: object | None  # Direct event bus reference
contract_path: str | None  # Path to contract file
```

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_event_bus import MixinEventBus
from omnibase_core.mixins.mixin_completion_data import MixinCompletionData

class MyNode(MixinEventBus):
    """Node with event bus integration."""

    def get_event_patterns(self) -> list[str]:
        """Define event patterns to listen for."""
        return [
            "generation.tool.start",
            "generation.tool.process",
        ]

    async def process_request(self, input_data):
        """Process and publish completion."""
        try:
            result = await self._do_processing(input_data)

            # Publish success event
            completion_data = MixinCompletionData(
                message="Processing completed successfully",
                success=True,
                tags=["processed", "validated"],
            )
            await self.apublish_completion_event(
                "generation.tool.complete",
                completion_data,
            )

            return result

        except Exception as e:
            # Publish error event
            error_data = MixinCompletionData(
                message=f"Processing failed: {str(e)}",
                success=False,
                tags=["error", "failed"],
            )
            await self.apublish_completion_event(
                "generation.tool.complete",
                error_data,
            )
            raise
```

**Common Patterns**:
- Override `get_event_patterns()` for subscriptions
- Use `publish_completion_event()` for sync contexts
- Use `apublish_completion_event()` for async contexts
- Always publish completion events (success or failure)

**When to Use**:
- Event-driven communication
- Async processing workflows
- Completion tracking
- Event bus integration

---

#### MixinEventHandler

**Module**: `omnibase_core.mixins.mixin_event_handler`
**Class**: `MixinEventHandler`

**Purpose**: Handles incoming event requests with registration and routing.

**Capabilities**:
- Event handler registration
- Event routing
- Handler lifecycle management

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_event_handler import MixinEventHandler

class MyNode(MixinEventHandler):
    def __init__(self, event_bus, **kwargs):
        super().__init__(**kwargs)
        self.event_bus = event_bus
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup custom event handlers."""
        # Register handlers via event bus
        self.event_bus.subscribe("my.event.pattern", self._handle_my_event)

    def _handle_my_event(self, envelope):
        """Handle custom event."""
        event = envelope.payload
        # Process event
        pass
```

**When to Use**:
- Custom event handling logic
- Event routing
- Event processing pipelines

---

#### MixinEventListener

**Module**: `omnibase_core.mixins.mixin_event_listener`
**Class**: `MixinEventListener`

**Purpose**: Event consumption and listening patterns (composed into `MixinEventBus`).

**Note**: This functionality is now part of `MixinEventBus`. Use `MixinEventBus` for new implementations.

---

#### MixinIntrospectionPublisher

**Module**: `omnibase_core.mixins.mixin_introspection_publisher`
**Class**: `MixinIntrospectionPublisher`

**Purpose**: Auto-publishing introspection events for service discovery.

**Capabilities**:
- Gather introspection data
- Publish to event bus
- Periodic refresh

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_introspection_publisher import MixinIntrospectionPublisher

class MyNode(MixinIntrospectionPublisher):
    def __init__(self, event_bus, metadata_loader, **kwargs):
        super().__init__(**kwargs)
        self.event_bus = event_bus
        self.metadata_loader = metadata_loader
        self._publish_introspection_event()

    def _gather_introspection_data(self):
        """Gather node introspection data."""
        return {
            "node_name": self.get_node_name(),
            "version": self.get_node_version(),
            "capabilities": self.get_capabilities(),
            "actions": ["process", "validate", "transform"],
        }
```

**When to Use**:
- Service discovery
- Dynamic service catalogs
- Real-time capability advertisement

---

### Service Integration

#### MixinServiceRegistry

**Module**: `omnibase_core.mixins.mixin_service_registry`
**Class**: `MixinServiceRegistry`

**Purpose**: Event-driven service registry for tool discovery and lifecycle management.

**Capabilities**:
- Automatic tool discovery
- Service registration
- Lifecycle management (online/offline tracking)
- Discovery callbacks
- TTL-based cleanup

**Configuration**:

```python
introspection_timeout: int = 30  # Introspection request timeout (seconds)
service_ttl: int = 300  # Service time-to-live (seconds)
auto_cleanup_interval: int = 60  # Cleanup interval (seconds)
```

**Dependencies**: None (optional: `MixinEventBus` for event-driven discovery)

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_service_registry import MixinServiceRegistry

class NodeServiceHub(MixinServiceRegistry):
    """Service hub with discovery."""

    def __init__(self, event_bus, **kwargs):
        super().__init__(**kwargs)
        self.event_bus = event_bus

        # Add discovery callback
        self.add_discovery_callback(self._on_tool_discovered)

        # Start registry
        self.start_service_registry(domain_filter="generation")

    def _on_tool_discovered(self, event_type: str, entry):
        """Handle tool discovery events."""
        if event_type == "tool_discovered":
            print(f"New tool: {entry.service_name}")
            print(f"Capabilities: {entry.capabilities}")
        elif event_type == "tool_offline":
            print(f"Tool offline: {entry.service_name}")

    def find_tool_for_task(self, capability: str):
        """Find tools with specific capability."""
        tools = self.get_tools_by_capability(capability)
        return tools[0] if tools else None

    def get_status(self):
        """Get registry status."""
        stats = self.get_registry_stats()
        return {
            "total_services": stats["total_services"],
            "online_services": stats["online_services"],
            "offline_services": stats["offline_services"],
        }
```

**Common Patterns**:
- Call `start_service_registry()` to begin discovery
- Use `domain_filter` for scoped discovery
- Register callbacks for discovery events
- Query by capability or service name
- Monitor with `get_registry_stats()`

**When to Use**:
- Building service hubs/coordinators
- Dynamic service discovery
- Multi-service orchestration
- MCP server implementation

---

#### MixinDiscoveryResponder

**Module**: `omnibase_core.mixins.mixin_discovery_responder`
**Class**: `MixinDiscoveryResponder`

**Purpose**: Responds to discovery protocol requests.

**Capabilities**:
- Discovery request handling
- Introspection response generation
- Protocol compliance

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_discovery_responder import MixinDiscoveryResponder

class MyNode(MixinDiscoveryResponder):
    def __init__(self, event_bus, **kwargs):
        super().__init__(**kwargs)
        self.event_bus = event_bus
        self._setup_discovery_response()
```

**When to Use**:
- Nodes need to respond to discovery
- Integration with service registries
- Protocol compliance

---

#### MixinNodeService

**Module**: `omnibase_core.mixins.mixin_node_service`
**Class**: `MixinNodeService`

**Purpose**: Node-as-service capabilities for exposing nodes as network services.

**Capabilities**:
- Service endpoint management
- Request handling
- Service lifecycle

**Configuration**: Varies by service type

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_node_service import MixinNodeService

class MyServiceNode(MixinNodeService):
    """Node exposed as network service."""

    async def handle_service_request(self, request):
        """Handle incoming service request."""
        # Process request
        result = await self.process(request.data)
        return result
```

**When to Use**:
- Exposing nodes as HTTP/gRPC services
- API integration
- External service consumption

---

### Execution Patterns

#### MixinNodeExecutor

**Module**: `omnibase_core.mixins.mixin_node_executor`
**Class**: `MixinNodeExecutor`

**Purpose**: Persistent node executor for TOOL_INVOCATION event handling.

**Capabilities**:
- Respond to TOOL_INVOCATION events
- Convert events to input states
- Execute node.run() with proper context
- Emit TOOL_RESPONSE events
- Health monitoring
- Graceful shutdown

**Configuration**: Inherits from `MixinEventDrivenNode`

**Dependencies**: `MixinEventDrivenNode` (parent)

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_node_executor import MixinNodeExecutor

class NodeMyTool(MixinNodeExecutor):
    """Persistent tool executor."""

    async def run(self, input_state):
        """Execute tool logic."""
        # Process input
        result = await self._process_tool_request(input_state)
        return result

    async def main(self):
        """Main entry point for persistent executor."""
        await self.start_executor_mode()

# Run as persistent service
if __name__ == "__main__":
    tool = NodeMyTool(event_bus=event_bus, metadata_loader=loader)
    asyncio.run(tool.main())
```

**Common Patterns**:
- Use for MCP tool servers
- Handle TOOL_INVOCATION events
- Implement `run()` method for tool logic
- Call `start_executor_mode()` to begin
- Graceful shutdown with `stop_executor_mode()`

**When to Use**:
- MCP tool integration
- GraphQL resolvers
- Persistent tool services
- Long-running executors

---

#### MixinNodeLifecycle

**Module**: `omnibase_core.mixins.mixin_node_lifecycle`
**Class**: `MixinNodeLifecycle`

**Purpose**: Node lifecycle management including registration, shutdown, and event publishing.

**Capabilities**:
- Node registration via NODE_ANNOUNCE
- Shutdown event publishing (NODE_SHUTDOWN_EVENT)
- Lifecycle event emission (NODE_START, NODE_SUCCESS, NODE_FAILURE)
- Cleanup resource management

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_node_lifecycle import MixinNodeLifecycle
from uuid import uuid4

class MyNode(MixinNodeLifecycle):
    """Node with lifecycle management."""

    def __init__(self, event_bus, **kwargs):
        super().__init__(**kwargs)
        self._node_id = uuid4()
        self.event_bus = event_bus

        # Register node
        self._register_node()

        # Register shutdown hook
        self._register_shutdown_hook()

    async def execute_operation(self):
        """Execute with lifecycle events."""
        # Emit start event
        correlation_id = self.emit_node_start()

        try:
            # Do work
            result = await self._do_work()

            # Emit success
            self.emit_node_success(correlation_id=correlation_id)
            return result

        except Exception as e:
            # Emit failure
            self.emit_node_failure(
                metadata={"error": str(e)},
                correlation_id=correlation_id,
            )
            raise

    def shutdown(self):
        """Cleanup on shutdown."""
        self.cleanup_lifecycle_resources()
```

**Common Patterns**:
- Call `_register_node()` during initialization
- Call `_register_shutdown_hook()` to ensure cleanup
- Use `emit_node_start()`, `emit_node_success()`, `emit_node_failure()` for tracking
- Call `cleanup_lifecycle_resources()` on shutdown

**When to Use**:
- Automatic node registration
- Lifecycle event tracking
- Graceful shutdown handling
- Service discovery integration

---

#### MixinNodeSetup

**Module**: `omnibase_core.mixins.mixin_node_setup`
**Class**: `MixinNodeSetup`

**Purpose**: Setup and initialization helpers for nodes.

**Capabilities**:
- Common initialization patterns
- Setup validation
- Configuration helpers

**Configuration**: Varies by use case

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_node_setup import MixinNodeSetup

class MyNode(MixinNodeSetup):
    """Node with setup helpers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_node()
```

**When to Use**:
- Complex initialization logic
- Setup validation
- Configuration management

---

#### MixinHybridExecution

**Module**: `omnibase_core.mixins.mixin_hybrid_execution`
**Class**: `MixinHybridExecution`

**Purpose**: Hybrid sync/async execution patterns for nodes that support both.

**Capabilities**:
- Sync to async conversion
- Async to sync conversion
- Execution mode detection

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_hybrid_execution import MixinHybridExecution

class MyNode(MixinHybridExecution):
    """Node supporting sync and async execution."""

    def process_sync(self, input_data):
        """Synchronous processing."""
        return self._process_data(input_data)

    async def process_async(self, input_data):
        """Asynchronous processing."""
        return await self._process_data_async(input_data)

    def execute(self, input_data, mode="auto"):
        """Auto-detect execution mode."""
        if mode == "sync":
            return self.process_sync(input_data)
        elif mode == "async":
            return self.process_async(input_data)
        else:
            # Auto-detect based on context
            return self._auto_execute(input_data)
```

**When to Use**:
- Supporting both sync/async callers
- Legacy integration with async modernization
- Performance optimization (sync for simple, async for complex)

---

#### MixinToolExecution

**Module**: `omnibase_core.mixins.mixin_tool_execution`
**Class**: `MixinToolExecution`

**Purpose**: Tool execution framework for MCP/GraphQL tool integration.

**Capabilities**:
- Tool invocation handling
- Tool response generation
- Tool metadata management

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_tool_execution import MixinToolExecution

class MyToolNode(MixinToolExecution):
    """Node as MCP tool."""

    def get_tool_definition(self):
        """Define tool metadata."""
        return {
            "name": "my_tool",
            "description": "Does something useful",
            "parameters": {
                "input": {"type": "string", "description": "Input data"},
            },
        }

    async def execute_tool(self, parameters):
        """Execute tool logic."""
        input_data = parameters["input"]
        result = await self._process_tool(input_data)
        return {"output": result}
```

**When to Use**:
- MCP tool implementation
- GraphQL resolver implementation
- Tool-based architectures

---

#### MixinWorkflowSupport

**Module**: `omnibase_core.mixins.mixin_workflow_support`
**Class**: `MixinWorkflowSupport`

**Purpose**: DAG workflow support for complex orchestration.

**Capabilities**:
- DAG construction
- Workflow execution
- Step dependencies
- Parallel execution

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_workflow_support import MixinWorkflowSupport

class MyOrchestrator(MixinWorkflowSupport):
    """Orchestrator with workflow support."""

    def build_workflow(self):
        """Build DAG workflow."""
        workflow = self.create_workflow()

        # Add steps
        step1 = workflow.add_step("validate", self._validate)
        step2 = workflow.add_step("process", self._process, depends_on=[step1])
        step3 = workflow.add_step("save", self._save, depends_on=[step2])

        return workflow

    async def execute_workflow(self, input_data):
        """Execute workflow."""
        workflow = self.build_workflow()
        result = await workflow.execute(input_data)
        return result
```

**When to Use**:
- Complex multi-step workflows
- Parallel step execution
- Workflow orchestration
- Pipeline processing

---

### Data Handling

#### MixinHashComputation

**Module**: `omnibase_core.mixins.mixin_hash_computation`
**Class**: `MixinHashComputation`

**Purpose**: Hash computation utilities for content addressing and verification.

**Capabilities**:
- BLAKE3 hash computation
- SHA256 fallback
- File hashing
- Data hashing

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_hash_computation import MixinHashComputation

class MyNode(MixinHashComputation):
    """Node with hash utilities."""

    def compute_file_hash(self, file_path):
        """Compute file hash."""
        return self.hash_file(file_path)

    def compute_data_hash(self, data):
        """Compute data hash."""
        return self.hash_data(data)
```

**When to Use**:
- Content addressing
- File verification
- Deduplication
- Integrity checks

---

#### MixinCaching

**Module**: `omnibase_core.mixins.mixin_caching`
**Class**: `MixinCaching`

**Purpose**: Result caching for expensive operations (stub - full Redis/Memcached support planned).

**Capabilities**:
- Generate cache keys
- Get/set cached values
- Invalidate cache entries
- Clear cache
- Cache statistics

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: PARTIAL - NodeEffect has basic caching; this adds comprehensive backend support

**Usage Example**:

```python
from omnibase_core.mixins.mixin_caching import MixinCaching

class NodeExpensiveCompute(MixinCaching):
    """Compute node with caching."""

    async def execute_compute(self, contract):
        """Execute with caching."""
        # Generate cache key
        cache_key = self.generate_cache_key(contract.input_data)

        # Check cache
        cached = await self.get_cached(cache_key)
        if cached:
            return cached

        # Compute result
        result = await self._expensive_computation(contract)

        # Cache result (10 min TTL)
        await self.set_cached(cache_key, result, ttl_seconds=600)

        return result

    async def invalidate_for_input(self, input_data):
        """Invalidate cache for specific input."""
        cache_key = self.generate_cache_key(input_data)
        await self.invalidate_cache(cache_key)
```

**Common Patterns**:
- Use `generate_cache_key()` for consistent keys
- Check cache before expensive operations
- Set TTL based on data freshness requirements
- Invalidate on data changes

**When to Use**:
- Expensive computations (>100ms)
- Frequently accessed data
- API rate limiting
- Cost reduction (LLM calls)

---

#### MixinLazyEvaluation

**Module**: `omnibase_core.mixins.mixin_lazy_evaluation`
**Class**: `MixinLazyEvaluation`

**Purpose**: Lazy loading patterns for deferred computation.

**Capabilities**:
- Lazy value computation
- Deferred initialization
- On-demand loading

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_lazy_evaluation import MixinLazyEvaluation

class MyNode(MixinLazyEvaluation):
    """Node with lazy loading."""

    @lazy_property
    def expensive_resource(self):
        """Lazy-loaded resource."""
        print("Loading expensive resource...")
        return self._load_expensive_resource()

    def process(self, data):
        """Process with lazy loading."""
        # Resource only loaded when accessed
        if data.needs_resource:
            return self.expensive_resource.process(data)
        return data
```

**When to Use**:
- Optional expensive resources
- Conditional initialization
- Memory optimization
- Startup performance

---

#### MixinCompletionData

**Module**: `omnibase_core.mixins.mixin_completion_data`
**Class**: `MixinCompletionData`

**Purpose**: Completion tracking data model for event-driven workflows.

**Capabilities**:
- Completion status tracking
- Success/failure metadata
- Correlation ID management
- Event payload construction

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_completion_data import MixinCompletionData

# Success completion
success_data = MixinCompletionData(
    message="Operation completed successfully",
    success=True,
    tags=["processed", "validated"],
    correlation_id=correlation_id,
)

# Failure completion
failure_data = MixinCompletionData(
    message=f"Operation failed: {error}",
    success=False,
    tags=["error", "failed"],
    correlation_id=correlation_id,
)
```

**When to Use**:
- Event-driven completion tracking
- Workflow status reporting
- Error propagation
- Correlation tracking

---

### Serialization

#### MixinCanonicalYAMLSerializer

**Module**: `omnibase_core.mixins.mixin_canonical_serialization`
**Class**: `MixinCanonicalYAMLSerializer`

**Purpose**: Canonical YAML serialization for deterministic hashing and stamping.

**Capabilities**:
- Deterministic YAML output
- Field normalization
- Placeholder injection (for volatile fields)
- Sorted keys
- Consistent formatting

**Configuration**:

```python
volatile_fields: tuple = (EnumNodeMetadataField.HASH, EnumNodeMetadataField.LAST_MODIFIED_AT)
placeholder: str = "<PLACEHOLDER>"
sort_keys: bool = False
explicit_start: bool = True
explicit_end: bool = True
```

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_canonical_serialization import MixinCanonicalYAMLSerializer
from omnibase_core.enums import EnumNodeMetadataField

class MyNode(MixinCanonicalYAMLSerializer):
    """Node with canonical serialization."""

    def compute_hash(self, metadata_block):
        """Compute deterministic hash."""
        # Canonicalize with placeholders for volatile fields
        canonical = self.canonicalize_metadata_block(
            metadata_block,
            volatile_fields=(
                EnumNodeMetadataField.HASH,
                EnumNodeMetadataField.LAST_MODIFIED_AT,
            ),
            placeholder="<PLACEHOLDER>",
        )

        # Hash the canonical representation
        import hashlib
        return hashlib.sha256(canonical.encode()).hexdigest()
```

**Common Patterns**:
- Use for content addressing
- Exclude volatile fields (timestamps, hashes)
- Sort keys for consistency
- Validate round-trip serialization

**When to Use**:
- Content-addressed storage
- Deterministic hashing
- Metadata stamping
- Integrity verification

---

#### MixinYAMLSerialization

**Module**: `omnibase_core.mixins.mixin_yaml_serialization`
**Class**: `MixinYAMLSerialization`

**Purpose**: Standard YAML serialization/deserialization (non-canonical).

**Capabilities**:
- YAML dump
- YAML load
- Safe parsing

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_yaml_serialization import MixinYAMLSerialization

class MyNode(MixinYAMLSerialization):
    """Node with YAML support."""

    def save_config(self, config, path):
        """Save config to YAML."""
        yaml_str = self.to_yaml(config)
        path.write_text(yaml_str)

    def load_config(self, path):
        """Load config from YAML."""
        yaml_str = path.read_text()
        return self.from_yaml(yaml_str)
```

**When to Use**:
- Configuration files
- Data serialization
- Human-readable output

---

#### MixinSerializable

**Module**: `omnibase_core.mixins.mixin_serializable`
**Class**: `MixinSerializable`

**Purpose**: Generic serialization support for multiple formats.

**Capabilities**:
- JSON serialization
- YAML serialization
- Pickle serialization
- Format detection

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_serializable import MixinSerializable

class MyDataModel(MixinSerializable):
    """Serializable data model."""

    def to_json(self):
        """Serialize to JSON."""
        return self.serialize(format="json")

    def to_yaml(self):
        """Serialize to YAML."""
        return self.serialize(format="yaml")

    @classmethod
    def from_json(cls, json_str):
        """Deserialize from JSON."""
        return cls.deserialize(json_str, format="json")
```

**When to Use**:
- Multi-format serialization
- API responses
- Data persistence

---

#### MixinRedaction

**Module**: `omnibase_core.mixins.mixin_redaction`
**Class**: `MixinRedaction`

**Purpose**: Sensitive field redaction for logging and serialization.

**Capabilities**:
- Redact sensitive fields (passwords, API keys, tokens)
- Pattern-based redaction
- Safe logging

**Configuration**:

```python
redact_fields: list[str] = ["password", "api_key", "secret", "token"]
redact_patterns: list[str] = [...]  # Regex patterns
```

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_redaction import MixinRedaction

class MyNode(MixinRedaction):
    """Node with redaction."""

    def log_request(self, request_data):
        """Log request with redaction."""
        redacted = self.redact_sensitive_fields(request_data)
        logger.info(f"Request: {redacted}")

    def save_config(self, config):
        """Save config with redacted secrets."""
        safe_config = self.redact_sensitive_fields(config)
        self._save_to_file(safe_config)
```

**When to Use**:
- Logging with sensitive data
- Config file storage
- Audit trails
- Security compliance

---

### Contract & Metadata

#### MixinContractMetadata

**Module**: `omnibase_core.mixins.mixin_contract_metadata`
**Class**: `MixinContractMetadata`

**Purpose**: Contract metadata handling and extraction.

**Capabilities**:
- Extract metadata from contracts
- Metadata validation
- Schema compliance

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_contract_metadata import MixinContractMetadata

class MyNode(MixinContractMetadata):
    """Node with contract metadata."""

    def load_contract(self, contract_path):
        """Load and extract metadata."""
        metadata = self.extract_contract_metadata(contract_path)
        self.validate_contract_metadata(metadata)
        return metadata
```

**When to Use**:
- Contract processing
- Metadata extraction
- Schema validation

---

#### MixinContractStateReducer

**Module**: `omnibase_core.mixins.mixin_contract_state_reducer`
**Class**: `MixinContractStateReducer`

**Purpose**: Contract state reduction for stateful operations.

**Capabilities**:
- State aggregation
- State transitions
- State validation

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_contract_state_reducer import MixinContractStateReducer

class MyReducerNode(MixinContractStateReducer):
    """Reducer with contract state."""

    def reduce_state(self, current_state, event):
        """Reduce contract state."""
        new_state = self.apply_state_transition(current_state, event)
        self.validate_state(new_state)
        return new_state
```

**When to Use**:
- Stateful reducers
- State machines
- Event sourcing

---

#### MixinIntrospectFromContract

**Module**: `omnibase_core.mixins.mixin_introspect_from_contract`
**Class**: `MixinIntrospectFromContract`

**Purpose**: Contract-driven introspection for automatic metadata extraction.

**Capabilities**:
- Extract introspection from contract
- Auto-generate capabilities
- Schema-driven metadata

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_introspect_from_contract import MixinIntrospectFromContract

class MyNode(MixinIntrospectFromContract):
    """Node with contract-driven introspection."""

    def __init__(self, contract_path, **kwargs):
        super().__init__(**kwargs)
        self.contract_path = contract_path
        self._introspection_data = self.introspect_from_contract(contract_path)

    def get_capabilities(self):
        """Get capabilities from contract."""
        return self._introspection_data.get("capabilities", [])
```

**When to Use**:
- Contract-first development
- Auto-generated metadata
- Schema-driven design

---

#### MixinNodeIdFromContract

**Module**: `omnibase_core.mixins.mixin_node_id_from_contract`
**Class**: `MixinNodeIdFromContract`

**Purpose**: Extract node ID from contract for deterministic identification.

**Capabilities**:
- Deterministic node ID generation
- Contract-based identification
- UUID derivation

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_node_id_from_contract import MixinNodeIdFromContract

class MyNode(MixinNodeIdFromContract):
    """Node with contract-based ID."""

    def __init__(self, contract_path, **kwargs):
        super().__init__(**kwargs)
        self._node_id = self.generate_node_id_from_contract(contract_path)
```

**When to Use**:
- Deterministic node identification
- Contract-based routing
- Reproducible deployments

---

#### MixinNodeIntrospection

**Module**: `omnibase_core.mixins.mixin_introspection`
**Class**: `MixinNodeIntrospection`

**Purpose**: Node introspection support for runtime metadata.

**Capabilities**:
- Runtime introspection
- Capability reporting
- Metadata collection

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_introspection import MixinNodeIntrospection

class MyNode(MixinNodeIntrospection):
    """Node with introspection."""

    def get_introspection_data(self):
        """Get runtime introspection."""
        return {
            "node_id": str(self.node_id),
            "capabilities": self.get_capabilities(),
            "version": self.get_version(),
            "status": self.get_status(),
        }
```

**When to Use**:
- Service discovery
- Runtime metadata
- Monitoring and observability

---

### CLI & Debugging

#### MixinCLIHandler

**Module**: `omnibase_core.mixins.mixin_cli_handler`
**Class**: `MixinCLIHandler`

**Purpose**: CLI command handling for nodes.

**Capabilities**:
- Command parsing
- Argument handling
- Help generation

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_cli_handler import MixinCLIHandler

class MyNode(MixinCLIHandler):
    """Node with CLI support."""

    def register_commands(self):
        """Register CLI commands."""
        self.add_command("process", self.cmd_process, "Process data")
        self.add_command("validate", self.cmd_validate, "Validate input")

    def cmd_process(self, args):
        """Process command."""
        print(f"Processing: {args.input}")

    def cmd_validate(self, args):
        """Validate command."""
        print(f"Validating: {args.input}")

# CLI usage
if __name__ == "__main__":
    node = MyNode()
    node.run_cli()
```

**When to Use**:
- Command-line tools
- Interactive debugging
- Script automation

---

#### MixinDebugDiscoveryLogging

**Module**: `omnibase_core.mixins.mixin_debug_discovery_logging`
**Class**: `MixinDebugDiscoveryLogging`

**Purpose**: Debug logging for discovery protocol.

**Capabilities**:
- Discovery event logging
- Protocol debugging
- Verbose output

**Configuration**:

```python
debug_enabled: bool = True
log_level: str = "DEBUG"
```

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_debug_discovery_logging import MixinDebugDiscoveryLogging

class MyNode(MixinDebugDiscoveryLogging):
    """Node with debug discovery logging."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enable_debug_discovery_logging()

    def handle_discovery_event(self, event):
        """Handle with debug logging."""
        self.log_discovery_event("received", event)
        # Process event
        result = self._process_event(event)
        self.log_discovery_event("processed", result)
        return result
```

**When to Use**:
- Debugging discovery issues
- Protocol development
- Integration troubleshooting

---

#### MixinFailFast

**Module**: `omnibase_core.mixins.mixin_fail_fast`
**Class**: `MixinFailFast`

**Purpose**: Fail-fast validation patterns for early error detection.

**Capabilities**:
- Precondition validation
- Early error detection
- Fast failure

**Configuration**: None

**Dependencies**: None

**NodeEffect Overlap**: NONE

**Usage Example**:

```python
from omnibase_core.mixins.mixin_fail_fast import MixinFailFast

class MyNode(MixinFailFast):
    """Node with fail-fast validation."""

    def process(self, input_data):
        """Process with fail-fast validation."""
        # Validate preconditions
        self.require_not_none(input_data, "input_data")
        self.require_type(input_data, dict, "input_data")
        self.require_key(input_data, "required_field")

        # Process if validation passed
        return self._process_data(input_data)
```

**When to Use**:
- Input validation
- Contract enforcement
- Error prevention
- Development debugging

---

## NodeEffect Built-in Features

**IMPORTANT**: The following features are built into `NodeEffect` base class. **DO NOT reimplement** these using mixins or custom code.

### Circuit Breakers

**Built-in**: `ModelCircuitBreaker`
**Configuration**:

```python
circuit_breaker = ModelCircuitBreaker(
    failure_threshold=5,
    recovery_timeout_seconds=60,
    half_open_max_calls=3,
)
```

**Usage**: Already available in `NodeEffect.execute_with_circuit_breaker()`

---

### Retry Policies

**Built-in**: `ModelRetryPolicy`
**Configuration**:

```python
retry_policy = ModelRetryPolicy(
    max_attempts=3,
    initial_delay_ms=1000,
    max_delay_ms=30000,
    backoff_multiplier=2.0,
)
```

**Usage**: Already available in `NodeEffect.execute_with_retry()`

---

### Transaction Support

**Built-in**: `ModelEffectTransaction`
**Configuration**:

```python
transaction = ModelEffectTransaction(
    isolation_level="READ_COMMITTED",
    timeout_seconds=30,
    rollback_on_error=True,
)
```

**Usage**: Already available in `NodeEffect.execute_in_transaction()`

---

### Timeout Management

**Built-in**: Per-operation timeout configuration
**Usage**: Already available in `NodeEffect.execute_with_timeout()`

---

### Concurrent Execution

**Built-in**: Semaphore-based concurrency control
**Usage**: Already available in `NodeEffect.execute_concurrently()`

---

### Performance Metrics

**Built-in**: Basic metrics tracking
**Usage**: Already available in `NodeEffect.track_metrics()`

**Note**: Use `MixinMetrics` for **comprehensive** metrics with Prometheus/StatsD backends.

---

### Event Emission

**Built-in**: State change event publishing
**Usage**: Already available in `NodeEffect.emit_state_change()`

---

### File Operations

**Built-in**: Atomic file I/O with rollback
**Usage**: Already available in `NodeEffect.atomic_file_write()`

---

## Selection Flowchart

```
┌─────────────────────────────────────────┐
│  What type of node are you building?   │
└───────────┬─────────────────────────────┘
            │
            ├─ Effect Node (I/O, external systems)
            │  └─> Start with NodeEffect
            │      ├─ Need health checks? → MixinHealthCheck
            │      ├─ Need metrics? → MixinMetrics
            │      ├─ Event-driven? → MixinEventDrivenNode
            │      ├─ Service discovery? → MixinServiceRegistry
            │      └─ Caching? → MixinCaching
            │
            ├─ Compute Node (pure transformation)
            │  └─> Start with NodeCompute
            │      ├─ Need caching? → MixinCaching
            │      ├─ Need metrics? → MixinMetrics
            │      └─ Serialization? → MixinCanonicalYAMLSerializer
            │
            ├─ Orchestrator Node (workflow coordination)
            │  └─> Start with NodeOrchestrator
            │      ├─ Event-driven? → MixinEventDrivenNode
            │      ├─ Workflow support? → MixinWorkflowSupport
            │      ├─ Lifecycle tracking? → MixinNodeLifecycle
            │      └─ Metrics? → MixinMetrics
            │
            ├─ Reducer Node (aggregation)
            │  └─> Start with NodeReducer
            │      ├─ Event-driven? → MixinEventDrivenNode
            │      ├─ State reduction? → MixinContractStateReducer
            │      ├─ Caching? → MixinCaching
            │      └─ Metrics? → MixinMetrics
            │
            └─ Service Hub (discovery, registry)
               └─> Start with MixinServiceRegistry
                   ├─ Event-driven? → MixinEventBus
                   ├─ Discovery response? → MixinDiscoveryResponder
                   └─ Health monitoring? → MixinHealthCheck

┌─────────────────────────────────────────┐
│  Common Mixin Combinations              │
└─────────────────────────────────────────┘

1. **Event-Driven Effect Node**:
   - NodeEffect
   - MixinEventDrivenNode
   - MixinHealthCheck
   - MixinMetrics

2. **Service Hub**:
   - MixinServiceRegistry
   - MixinEventBus
   - MixinDiscoveryResponder
   - MixinHealthCheck

3. **Persistent Tool Executor**:
   - MixinNodeExecutor (includes MixinEventDrivenNode)
   - MixinToolExecution
   - MixinMetrics

4. **Workflow Orchestrator**:
   - NodeOrchestrator
   - MixinEventDrivenNode
   - MixinWorkflowSupport
   - MixinNodeLifecycle
   - MixinMetrics

5. **Caching Compute Node**:
   - NodeCompute
   - MixinCaching
   - MixinMetrics
   - MixinHashComputation
```

---

## Common Patterns

### Pattern 1: Event-Driven Service with Health Checks

```python
from omnibase_core.nodes.node_effect import NodeEffect
from omnibase_core.mixins.mixin_event_driven_node import MixinEventDrivenNode
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_metrics import MixinMetrics

class NodeMyService(NodeEffect, MixinEventDrivenNode, MixinHealthCheck, MixinMetrics):
    """Event-driven service with health monitoring and metrics."""

    def __init__(self, event_bus, metadata_loader, **kwargs):
        super().__init__(
            node_id=uuid4(),
            event_bus=event_bus,
            metadata_loader=metadata_loader,
            **kwargs
        )

    def get_health_checks(self):
        return [self._check_database, self._check_api]

    async def execute_effect(self, contract):
        self.increment_counter("requests_total")
        start_time = time.time()

        try:
            result = await self._process_request(contract)
            self.increment_counter("requests_success")
            return result
        except Exception as e:
            self.increment_counter("requests_failed")
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.record_metric("request_latency_ms", latency_ms)
```

### Pattern 2: Service Registry Hub

```python
from omnibase_core.mixins.mixin_service_registry import MixinServiceRegistry
from omnibase_core.mixins.mixin_event_bus import MixinEventBus
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck

class NodeServiceHub(MixinServiceRegistry, MixinEventBus, MixinHealthCheck):
    """Service hub with discovery and health monitoring."""

    def __init__(self, event_bus, **kwargs):
        super().__init__(**kwargs)
        self.event_bus = event_bus

        # Setup registry
        self.add_discovery_callback(self._on_tool_discovered)
        self.start_service_registry(domain_filter="generation")

    def _on_tool_discovered(self, event_type, entry):
        if event_type == "tool_discovered":
            logger.info(f"New tool: {entry.service_name}")

    def get_health_checks(self):
        return [self._check_registry_health]

    def _check_registry_health(self):
        stats = self.get_registry_stats()
        is_healthy = stats["online_services"] > 0

        return ModelHealthStatus(
            status=EnumNodeHealthStatus.HEALTHY if is_healthy else EnumNodeHealthStatus.DEGRADED,
            message=f"{stats['online_services']} services online",
        )
```

### Pattern 3: Cached Compute Node

```python
from omnibase_core.nodes.node_compute import NodeCompute
from omnibase_core.mixins.mixin_caching import MixinCaching
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.mixins.mixin_hash_computation import MixinHashComputation

class NodeExpensiveCompute(NodeCompute, MixinCaching, MixinMetrics, MixinHashComputation):
    """Compute node with caching and metrics."""

    async def execute_compute(self, contract):
        # Generate cache key
        cache_key = self.generate_cache_key(contract.input_data)

        # Check cache
        cached = await self.get_cached(cache_key)
        if cached:
            self.increment_counter("cache_hits")
            return cached

        self.increment_counter("cache_misses")

        # Compute result
        start_time = time.time()
        result = await self._expensive_computation(contract)

        # Track compute time
        compute_ms = (time.time() - start_time) * 1000
        self.record_metric("compute_time_ms", compute_ms)

        # Cache result
        await self.set_cached(cache_key, result, ttl_seconds=600)

        return result
```

### Pattern 4: Workflow Orchestrator

```python
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
from omnibase_core.mixins.mixin_event_driven_node import MixinEventDrivenNode
from omnibase_core.mixins.mixin_workflow_support import MixinWorkflowSupport
from omnibase_core.mixins.mixin_metrics import MixinMetrics

class NodeWorkflowOrchestrator(NodeOrchestrator, MixinEventDrivenNode, MixinWorkflowSupport, MixinMetrics):
    """Workflow orchestrator with event-driven execution."""

    def build_workflow(self):
        workflow = self.create_workflow()

        # Define workflow steps
        validate = workflow.add_step("validate", self._validate_input)
        transform = workflow.add_step("transform", self._transform_data, depends_on=[validate])
        persist = workflow.add_step("persist", self._persist_result, depends_on=[transform])

        return workflow

    async def execute_orchestration(self, contract):
        workflow = self.build_workflow()

        # Track execution
        self.increment_counter("workflows_started")
        start_time = time.time()

        try:
            result = await workflow.execute(contract.input_data)
            self.increment_counter("workflows_completed")
            return result
        except Exception as e:
            self.increment_counter("workflows_failed")
            raise
        finally:
            execution_ms = (time.time() - start_time) * 1000
            self.record_metric("workflow_execution_ms", execution_ms)
```

---

## Appendix A: Import Reference

### Quick Import Examples

```python
# Health & Monitoring
from omnibase_core.mixins.mixin_health_check import MixinHealthCheck
from omnibase_core.mixins.mixin_metrics import MixinMetrics
from omnibase_core.mixins.mixin_log_data import MixinLogData
from omnibase_core.mixins.mixin_request_response_introspection import MixinRequestResponseIntrospection

# Event-Driven
from omnibase_core.mixins.mixin_event_driven_node import MixinEventDrivenNode
from omnibase_core.mixins.mixin_event_bus import MixinEventBus
from omnibase_core.mixins.mixin_event_handler import MixinEventHandler
from omnibase_core.mixins.mixin_introspection_publisher import MixinIntrospectionPublisher

# Service Integration
from omnibase_core.mixins.mixin_service_registry import MixinServiceRegistry
from omnibase_core.mixins.mixin_discovery_responder import MixinDiscoveryResponder
from omnibase_core.mixins.mixin_node_service import MixinNodeService

# Execution
from omnibase_core.mixins.mixin_node_executor import MixinNodeExecutor
from omnibase_core.mixins.mixin_node_lifecycle import MixinNodeLifecycle
from omnibase_core.mixins.mixin_node_setup import MixinNodeSetup
from omnibase_core.mixins.mixin_hybrid_execution import MixinHybridExecution
from omnibase_core.mixins.mixin_tool_execution import MixinToolExecution
from omnibase_core.mixins.mixin_workflow_support import MixinWorkflowSupport

# Data Handling
from omnibase_core.mixins.mixin_hash_computation import MixinHashComputation
from omnibase_core.mixins.mixin_caching import MixinCaching
from omnibase_core.mixins.mixin_lazy_evaluation import MixinLazyEvaluation
from omnibase_core.mixins.mixin_completion_data import MixinCompletionData

# Serialization
from omnibase_core.mixins.mixin_canonical_serialization import MixinCanonicalYAMLSerializer
from omnibase_core.mixins.mixin_yaml_serialization import MixinYAMLSerialization
from omnibase_core.mixins.mixin_serializable import MixinSerializable
from omnibase_core.mixins.mixin_redaction import MixinRedaction

# Contract & Metadata
from omnibase_core.mixins.mixin_contract_metadata import MixinContractMetadata
from omnibase_core.mixins.mixin_contract_state_reducer import MixinContractStateReducer
from omnibase_core.mixins.mixin_introspect_from_contract import MixinIntrospectFromContract
from omnibase_core.mixins.mixin_node_id_from_contract import MixinNodeIdFromContract
from omnibase_core.mixins.mixin_introspection import MixinNodeIntrospection

# CLI & Debugging
from omnibase_core.mixins.mixin_cli_handler import MixinCLIHandler
from omnibase_core.mixins.mixin_debug_discovery_logging import MixinDebugDiscoveryLogging
from omnibase_core.mixins.mixin_fail_fast import MixinFailFast
```

---

## Appendix B: Mixin Dependency Graph

```
MixinEventDrivenNode
├── MixinEventHandler
├── MixinNodeLifecycle
├── MixinIntrospectionPublisher
└── MixinRequestResponseIntrospection

MixinNodeExecutor
└── MixinEventDrivenNode (parent)
    ├── MixinEventHandler
    ├── MixinNodeLifecycle
    ├── MixinIntrospectionPublisher
    └── MixinRequestResponseIntrospection

All other mixins: No dependencies (standalone)
```

---

## Document Metadata

**Version**: 1.0
**Created**: 2025-11-04
**Last Updated**: 2025-11-04
**Status**: Complete
**Total Mixins Documented**: 33
**Target Audience**: Code generator developers, node implementers
**Related Documents**:
- [Code Generator Mixin Enhancement Master Plan](/Volumes/PRO-G40/Code/omninode_bridge/docs/planning/CODEGEN_MIXIN_ENHANCEMENT_MASTER_PLAN.md)
- [Contract Schema Reference](./CONTRACT_SCHEMA.md)
- [Code Generation Guide](../guides/CODE_GENERATION_GUIDE.md)

---

**End of Mixin Catalog**
