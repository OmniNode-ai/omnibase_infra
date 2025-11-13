# Production Node Patterns - Code Generation Best Practices

**Purpose**: Catalog of patterns extracted from 8+ production ONEX nodes in omninode_bridge
**Last Updated**: 2025-11-05
**Analyzed Nodes**: codegen_orchestrator, codegen_metrics_reducer, llm_effect, orchestrator, reducer, distributed_lock_effect, store_effect, test_generator_effect

---

## Table of Contents

1. [Standard Imports](#1-standard-imports)
2. [Class Declaration & Docstring](#2-class-declaration--docstring)
3. [Initialization Pattern](#3-initialization-pattern)
4. [Execute Method Pattern](#4-execute-method-pattern)
5. [Event Publishing](#5-event-publishing)
6. [Error Handling](#6-error-handling)
7. [Lifecycle Methods](#7-lifecycle-methods)
8. [Consul Service Discovery](#8-consul-service-discovery)
9. [Logging Pattern](#9-logging-pattern)
10. [Metrics Tracking](#10-metrics-tracking)
11. [Main Entry Point](#11-main-entry-point)
12. [Type Patterns](#12-type-patterns)

---

## 1. Standard Imports

### Must-Have Imports (All Nodes)

```python
#!/usr/bin/env python3
"""
Node{Name}{Type} - Description.

ONEX v2.0 Compliance:
- Suffix-based naming: Node{Name}{Type}
- Extends Node{Type} from omnibase_core
- Uses ModelOnexError for error handling
- Event-driven architecture with Kafka
"""

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

# ONEX Core Imports (ALWAYS required)
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_{type} import ModelContract{Type}
from omnibase_core.nodes.node_{type} import Node{Type}

# Node-specific imports (models, protocols, services)
from .models import ...  # Local models
from omninode_bridge.protocols import ...  # Protocols
from omninode_bridge.services import ...  # Services

# Aliases for compatibility
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode
```

### Import Patterns by Node Type

**Orchestrator:**
```python
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator
```

**Reducer:**
```python
from omnibase_core.models.contracts.model_contract_reducer import (
    ModelContractReducer,
)
from omnibase_core.nodes.node_reducer import NodeReducer
```

**Effect:**
```python
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.nodes.node_effect import NodeEffect
```

---

## 2. Class Declaration & Docstring

### Pattern

```python
class Node{Name}{Type}(Node{Type}):
    """
    {Name} {Type} for {purpose}.

    {1-2 sentence overview of what this node does}

    Responsibilities:
    - {Responsibility 1}
    - {Responsibility 2}
    - {Responsibility 3}

    ONEX v2.0 Compliance:
    - Suffix-based naming: Node{Name}{Type}
    - Extends Node{Type} from omnibase_core
    - Uses ModelOnexError for error handling
    - Event-driven architecture with Kafka
    - Structured logging with correlation tracking

    Performance Targets:
    - {Metric 1}: {Value}
    - {Metric 2}: {Value}
    - {Metric 3}: {Value}

    Example Usage:
        ```python
        # Initialize node
        container = ModelContainer(...)
        node = Node{Name}{Type}(container)

        # Execute
        result = await node.execute_{type}(contract)
        ```
    """
```

### Real Examples

**Orchestrator:**
```python
class NodeCodegenOrchestrator(NodeOrchestrator):
    """
    Code Generation Orchestrator for ONEX nodes.

    Coordinates the 8-stage generation pipeline:
    1. Prompt parsing (5s)
    2. Intelligence gathering (3s) - optional RAG query
    ...

    Event Publishing:
    - NODE_GENERATION_STARTED: Workflow begins
    - NODE_GENERATION_STAGE_COMPLETED: Each stage completes (8x)
    - NODE_GENERATION_COMPLETED: Successful generation
    - NODE_GENERATION_FAILED: Generation failure
    """
```

**Effect:**
```python
class NodeLLMEffect(NodeEffect):
    """
    LLM Effect Node for multi-tier LLM API calls.

    Tier Configuration:
    - LOCAL: Not implemented yet (future Ollama/vLLM support)
    - CLOUD_FAST: GLM-4.5 via Z.ai (128K context, PRIMARY for Phase 1)
    - CLOUD_PREMIUM: GLM-4.6 via Z.ai (128K context, future)

    Circuit Breaker:
    - Failure threshold: 5 consecutive failures
    - Recovery timeout: 60 seconds
    - Protected: All external LLM API calls
    """
```

---

## 3. Initialization Pattern

### Standard __init__ Pattern

```python
def __init__(self, container: ModelContainer) -> None:
    """
    Initialize {Node} with dependency injection container.

    Args:
        container: ONEX container for dependency injection

    Raises:
        ModelOnexError: If container is invalid or initialization fails
    """
    super().__init__(container)

    # Configuration - defensive pattern for dependency_injector
    try:
        if hasattr(container.config, "get") and callable(container.config.get):
            self.config_value = container.config.get(
                "config_key",
                os.getenv("ENV_VAR", "default_value")
            )
        else:
            # Fallback to defaults
            self.config_value = os.getenv("ENV_VAR", "default_value")
    except Exception:
        # Fallback to defaults if any error
        self.config_value = os.getenv("ENV_VAR", "default_value")

    # Consul configuration for service discovery
    self.consul_host: str = container.config.get(
        "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
    )
    self.consul_port: int = container.config.get(
        "consul_port", int(os.getenv("CONSUL_PORT", "28500"))
    )
    self.consul_enable_registration: bool = container.config.get(
        "consul_enable_registration", True
    )

    # Get or create services from container
    try:
        health_check_mode = (
            container.config.get("health_check_mode", False)
            if hasattr(container.config, "get")
            else False
        )
    except Exception:
        health_check_mode = False

    # Get KafkaClient from container
    self.kafka_client = container.get_service("kafka_client")

    if self.kafka_client is None and not health_check_mode:
        try:
            from omninode_bridge.services.kafka_client import KafkaClient

            self.kafka_client = KafkaClient(
                bootstrap_servers=self.kafka_broker_url,
                enable_dead_letter_queue=True,
                max_retry_attempts=3,
                timeout_seconds=30,
            )
            container.register_service("kafka_client", self.kafka_client)
        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "KafkaClient not available - events will be logged only",
                {"node_id": self.node_id},
            )
            self.kafka_client = None
    elif health_check_mode:
        emit_log_event(
            LogLevel.DEBUG,
            "Health check mode enabled - skipping Kafka initialization",
            {"node_id": self.node_id},
        )
        self.kafka_client = None

    # Initialize metrics tracking
    self._total_operations = 0
    self._total_duration_ms = 0.0
    self._failed_operations = 0

    emit_log_event(
        LogLevel.INFO,
        "Node{Name}{Type} initialized successfully",
        {
            "node_id": self.node_id,
            "kafka_enabled": self.kafka_client is not None,
            "config_key": self.config_value,
        },
    )

    # Register with Consul for service discovery (skip in health check mode)
    if not health_check_mode and self.consul_enable_registration:
        self._register_with_consul_sync()
```

### Health Check Mode Detection

```python
# Always check for health_check_mode before initializing services
try:
    health_check_mode = (
        container.config.get("health_check_mode", False)
        if hasattr(container.config, "get")
        else False
    )
except Exception:
    health_check_mode = False

if health_check_mode:
    # Skip expensive initialization
    emit_log_event(
        LogLevel.DEBUG,
        "Health check mode enabled - skipping {service} initialization",
        {"node_id": self.node_id},
    )
    self.service = None
```

---

## 4. Execute Method Pattern

### Orchestrator Execute Pattern

```python
async def execute_orchestration(
    self, contract: ModelContractOrchestrator
) -> {OutputModel}:
    """
    Execute {workflow} orchestration.

    Args:
        contract: Orchestrator contract with workflow configuration

    Returns:
        {OutputModel} with workflow results

    Raises:
        OnexError: If workflow execution fails
    """
    start_time = time.perf_counter()
    correlation_id = contract.correlation_id
    workflow_id = uuid4()

    emit_log_event(
        LogLevel.INFO,
        "Starting {workflow} orchestration",
        {
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "workflow_id": str(workflow_id),
        },
    )

    try:
        # Extract input data from contract
        if not hasattr(contract, "input_data") or contract.input_data is None:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="Contract missing input_data for orchestration",
                details={
                    "node_id": self.node_id,
                    "correlation_id": str(correlation_id),
                },
            )

        # Execute workflow steps
        result = await self._execute_workflow(contract, workflow_id)

        # Publish completion event
        await self._publish_completed_event(correlation_id, workflow_id, result)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        emit_log_event(
            LogLevel.INFO,
            "{Workflow} orchestration completed successfully",
            {
                "node_id": self.node_id,
                "workflow_id": str(workflow_id),
                "processing_time_ms": processing_time_ms,
            },
        )

        return result

    except Exception as e:
        # Publish failure event
        await self._publish_failed_event(correlation_id, workflow_id, e)

        # Cleanup
        self._cleanup_workflow(workflow_id)

        # Re-raise or wrap exception
        if isinstance(e, OnexError):
            raise

        raise OnexError(
            error_code=EnumCoreErrorCode.OPERATION_FAILED,
            message=f"{Workflow} orchestration failed: {e!s}",
            details={
                "node_id": self.node_id,
                "correlation_id": str(correlation_id),
                "workflow_id": str(workflow_id),
            },
            cause=e,
        )
```

### Reducer Execute Pattern

```python
async def execute_reduction(
    self, contract: ModelContractReducer
) -> {OutputState}:
    """
    Execute pure {aggregation} reduction.

    Args:
        contract: Reducer contract with aggregation configuration

    Returns:
        {OutputState} with aggregated results

    Raises:
        OnexError: If reduction fails or validation errors occur
    """
    start_time = time.perf_counter()

    try:
        # Validate input
        if not hasattr(contract, "input_stream") and not hasattr(
            contract, "input_state"
        ):
            raise ValueError(
                "Contract must have either 'input_stream' or 'input_state' attribute"
            )

        # Extract configuration
        aggregation_type = self._get_aggregation_type(contract)
        batch_size = self._get_batch_size(contract)

        # Initialize aggregation state
        aggregated_data = defaultdict(dict)
        total_items = 0

        # Stream and aggregate data (pure computation)
        async for batch in self._stream_data(contract, batch_size=batch_size):
            for item in batch:
                # Aggregate item
                self._aggregate_item(item, aggregated_data)
                total_items += 1

        # Calculate metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        items_per_second = (
            total_items / (duration_ms / 1000) if duration_ms > 0 else 0.0
        )

        # Build output state
        return {OutputState}(
            aggregation_type=aggregation_type,
            total_items=total_items,
            aggregated_data=aggregated_data,
            duration_ms=duration_ms,
            items_per_second=items_per_second,
        )

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log failure
        emit_log_event(
            LogLevel.ERROR,
            f"{Aggregation} reduction failed: {e}",
            {
                "node_id": self.node_id,
                "error": str(e),
                "duration_ms": duration_ms,
            },
        )

        # Re-raise
        raise
```

### Effect Execute Pattern

```python
async def execute_effect(self, contract: ModelContractEffect) -> {OutputModel}:
    """
    Execute {operation} effect.

    Args:
        contract: Effect contract with input_state containing operation params

    Returns:
        {OutputModel} with operation results

    Raises:
        OnexError: If operation fails
    """
    start_time = time.perf_counter()
    correlation_id = contract.correlation_id

    emit_log_event(
        LogLevel.INFO,
        "Starting {operation} effect",
        {
            "node_id": str(self.node_id),
            "correlation_id": str(correlation_id),
        },
    )

    try:
        # Parse request from contract input_state
        input_state = contract.input_state or {}
        request = {RequestModel}(
            operation=input_state.get("operation"),
            # ... other fields
            correlation_id=correlation_id,
        )

        # Route to appropriate handler
        if request.operation == "operation_type":
            result = await self._handle_operation(request)
        else:
            raise OnexError(
                message=f"Unknown operation: {request.operation}",
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                operation=str(request.operation),
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_log_event(
            LogLevel.INFO,
            f"{Operation} effect completed",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
                "duration_ms": round(duration_ms, 2),
            },
        )

        return result

    except OnexError:
        raise

    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"{Operation} effect failed: {e}",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
                "error": str(e),
            },
        )

        raise OnexError(
            message=f"{Operation} effect failed: {e}",
            error_code=EnumCoreErrorCode.OPERATION_FAILED,
            node_id=str(self.node_id),
            correlation_id=str(correlation_id),
            error=str(e),
        ) from e
```

---

## 5. Event Publishing

### Standard Event Publishing Pattern

```python
async def _publish_event(
    self, event_type: {EventEnum}, data: dict[str, Any]
) -> None:
    """
    Publish event to Kafka using OnexEnvelopeV1 wrapping.

    Args:
        event_type: Event type identifier
        data: Event payload data
    """
    try:
        # Get Kafka topic name
        topic_name = event_type.get_topic_name(namespace=self.default_namespace)

        # Publish to Kafka if client is available
        if self.kafka_client and self.kafka_client.is_connected:
            # Extract correlation ID from data
            correlation_id = data.get("correlation_id") or data.get("workflow_id")

            # Add node metadata to payload
            payload = {
                **data,
                "node_id": self.node_id,
                "published_at": datetime.now(UTC).isoformat(),
            }

            # Publish with OnexEnvelopeV1 wrapping for standardized event format
            # Include Consul service_id for cross-service event correlation
            event_metadata = {
                "event_category": "{category}",
                "node_type": "{type}",
                "namespace": self.default_namespace,
            }

            # Add consul_service_id if available (enables cross-service correlation)
            if hasattr(self, "_consul_service_id"):
                event_metadata["consul_service_id"] = self._consul_service_id

            success = await self.kafka_client.publish_with_envelope(
                event_type=event_type.value,
                source_node_id=str(self.node_id),
                payload=payload,
                topic=topic_name,
                correlation_id=correlation_id,
                metadata=event_metadata,
            )

            if success:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Published Kafka event (OnexEnvelopeV1): {event_type.value}",
                    {
                        "node_id": self.node_id,
                        "event_type": event_type.value,
                        "topic_name": topic_name,
                        "correlation_id": correlation_id,
                        "envelope_wrapped": True,
                    },
                )
            else:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Failed to publish Kafka event: {event_type.value}",
                    {
                        "node_id": self.node_id,
                        "event_type": event_type.value,
                        "topic_name": topic_name,
                    },
                )
        else:
            # Kafka not available - log event only
            emit_log_event(
                LogLevel.DEBUG,
                f"Kafka unavailable, logging event: {event_type.value}",
                {
                    "node_id": self.node_id,
                    "event_type": event_type.value,
                    "topic_name": topic_name,
                    "data": data,
                },
            )

    except Exception as e:
        # Log error but don't fail workflow
        emit_log_event(
            LogLevel.WARNING,
            f"Failed to publish Kafka event: {event_type.value}",
            {
                "node_id": self.node_id,
                "event_type": event_type.value,
                "error": str(e),
            },
        )
```

### Key Points

1. **Always wrap with OnexEnvelopeV1** via `publish_with_envelope()`
2. **Include metadata** with event_category, node_type, namespace
3. **Add consul_service_id** if available for cross-service correlation
4. **Never fail workflow** if event publishing fails (try/except)
5. **Log event locally** if Kafka unavailable
6. **Use DEBUG level** for successful publishes (reduce noise)

---

## 6. Error Handling

### Standard Error Handling Pattern

```python
# Always re-raise OnexError as-is
try:
    result = await some_operation()
except OnexError:
    # Don't wrap OnexError - re-raise to preserve error context
    raise

# Wrap other exceptions in OnexError
except ConnectionError as e:
    # Network errors
    raise OnexError(
        error_code=EnumCoreErrorCode.CONNECTION_ERROR,
        message=f"Network connection failed: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "error_type": "ConnectionError",
        },
        cause=e,
    ) from e

except (TimeoutError, asyncio.TimeoutError) as e:
    # Timeout errors
    raise OnexError(
        error_code=EnumCoreErrorCode.TIMEOUT,
        message=f"Operation timed out: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "timeout_seconds": self.config.timeout,
        },
        cause=e,
    ) from e

except (ValueError, KeyError, AttributeError) as e:
    # Data validation errors
    raise OnexError(
        error_code=EnumCoreErrorCode.VALIDATION_ERROR,
        message=f"Invalid data: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "error_type": type(e).__name__,
        },
        cause=e,
    ) from e

except Exception as e:
    # Unexpected errors - log with exc_info and wrap
    emit_log_event(
        LogLevel.ERROR,
        f"Unexpected error: {type(e).__name__}",
        {
            "node_id": self.node_id,
            "error": str(e),
            "error_type": type(e).__name__,
        },
    )
    logger.error(f"Unexpected error: {type(e).__name__}", exc_info=True)

    raise OnexError(
        error_code=EnumCoreErrorCode.INTERNAL_ERROR,
        message=f"Unexpected error: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "error_type": type(e).__name__,
        },
        cause=e,
    ) from e
```

### Error Code Mapping

```python
# Use these error codes from EnumCoreErrorCode
VALIDATION_ERROR = "validation_error"         # Input validation failures
OPERATION_FAILED = "operation_failed"         # General operation failures
CONNECTION_ERROR = "connection_error"         # Network connection failures
TIMEOUT = "timeout"                           # Operation timeout
INTERNAL_ERROR = "internal_error"             # Unexpected internal errors
DEPENDENCY_ERROR = "dependency_error"         # Service dependency failures
CONFIGURATION_ERROR = "configuration_error"   # Configuration issues
SERVICE_UNAVAILABLE = "service_unavailable"   # External service unavailable
INVALID_INPUT = "invalid_input"               # Invalid input parameters
INVALID_OPERATION = "invalid_operation"       # Unsupported operation
```

---

## 7. Lifecycle Methods

### Standard Startup Pattern

```python
async def startup(self) -> None:
    """
    Node startup lifecycle hook.

    Initializes container services, connects Kafka, registers with Consul,
    and starts background tasks.

    Should be called when node is ready to serve requests.
    """
    emit_log_event(
        LogLevel.INFO,
        "Node{Name}{Type} starting up",
        {"node_id": self.node_id},
    )

    # Initialize container services if available
    if hasattr(self.container, "initialize"):
        try:
            await self.container.initialize()
            emit_log_event(
                LogLevel.INFO,
                "Container services initialized successfully",
                {
                    "node_id": self.node_id,
                    "kafka_connected": (
                        self.kafka_client.is_connected
                        if self.kafka_client
                        else False
                    ),
                },
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Container initialization failed, continuing in degraded mode: {e}",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    # Connect to Kafka if client is available
    if self.kafka_client and not self.kafka_client.is_connected:
        try:
            await self.kafka_client.connect()
            emit_log_event(
                LogLevel.INFO,
                "Kafka client connected",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Kafka connection failed: {e}",
                {"node_id": self.node_id},
            )

    emit_log_event(
        LogLevel.INFO,
        "Node{Name}{Type} startup complete",
        {"node_id": self.node_id},
    )
```

### Standard Shutdown Pattern

```python
async def shutdown(self) -> None:
    """
    Node shutdown lifecycle hook.

    Stops background tasks, disconnects Kafka, deregisters from Consul,
    and cleans up resources.

    Should be called when node is preparing to exit.
    """
    emit_log_event(
        LogLevel.INFO,
        "Node{Name}{Type} shutting down",
        {"node_id": self.node_id},
    )

    # Stop background tasks if any
    if hasattr(self, "_background_task") and self._background_task:
        self._background_task.cancel()
        try:
            await self._background_task
        except asyncio.CancelledError:
            pass

    # Cleanup container services
    if hasattr(self.container, "cleanup"):
        try:
            await self.container.cleanup()
            emit_log_event(
                LogLevel.INFO,
                "Container services cleaned up successfully",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Container cleanup failed: {e}",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    # Deregister from Consul for clean service discovery
    self._deregister_from_consul()

    emit_log_event(
        LogLevel.INFO,
        "Node{Name}{Type} shutdown complete",
        {"node_id": self.node_id},
    )
```

---

## 8. Consul Service Discovery

### Standard Consul Registration Pattern

```python
def _register_with_consul_sync(self) -> None:
    """
    Register {node} with Consul for service discovery (synchronous).

    Registers the {node} as a service with health checks pointing to
    the health endpoint. Includes metadata about node capabilities.

    Note:
        This is a non-blocking registration. Failures are logged but don't
        fail node startup. Service will continue without Consul if registration fails.
    """
    try:
        import consul

        # Initialize Consul client
        consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)

        # Generate unique service ID
        service_id = f"omninode-bridge-{node-name}-{self.node_id}"

        # Get service port from config (default to {port} for {node})
        service_port = int(self.container.config.get("service_port", {port}))

        # Get service host from config (default to localhost)
        service_host = self.container.config.get("service_host", "localhost")

        # Prepare service tags
        service_tags = [
            "onex",
            "bridge",
            "{node-type}",
            f"version:{getattr(self, 'version', '0.1.0')}",
            "omninode_bridge",
        ]

        # Prepare service metadata (encoded in tags for MVP compatibility)
        service_tags.extend(
            [
                "node_type:{node-type}",
                f"kafka_enabled:{self.kafka_client is not None}",
            ]
        )

        # Health check URL (assumes health endpoint is available)
        health_check_url = f"http://{service_host}:{service_port}/health"

        # Register service with Consul
        consul_client.agent.service.register(
            name="omninode-bridge-{node-name}",
            service_id=service_id,
            address=service_host,
            port=service_port,
            tags=service_tags,
            http=health_check_url,
            interval="30s",
            timeout="5s",
        )

        emit_log_event(
            LogLevel.INFO,
            "Registered with Consul successfully",
            {
                "node_id": self.node_id,
                "service_id": service_id,
                "consul_host": self.consul_host,
                "consul_port": self.consul_port,
                "service_host": service_host,
                "service_port": service_port,
            },
        )

        # Store service_id for deregistration
        self._consul_service_id = service_id

    except ImportError:
        emit_log_event(
            LogLevel.WARNING,
            "python-consul not installed - Consul registration skipped",
            {"node_id": self.node_id},
        )
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            "Failed to register with Consul",
            {
                "node_id": self.node_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )


def _deregister_from_consul(self) -> None:
    """
    Deregister {node} from Consul on shutdown (synchronous).

    Removes the service registration from Consul to prevent stale entries
    in the service catalog.

    Note:
        This is called during node shutdown. Failures are logged but don't
        prevent shutdown from completing.
    """
    try:
        if not hasattr(self, "_consul_service_id"):
            # Not registered, nothing to deregister
            return

        import consul

        consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
        consul_client.agent.service.deregister(self._consul_service_id)

        emit_log_event(
            LogLevel.INFO,
            "Deregistered from Consul successfully",
            {
                "node_id": self.node_id,
                "service_id": self._consul_service_id,
            },
        )

    except ImportError:
        # python-consul not installed, silently skip
        pass
    except Exception as e:
        emit_log_event(
            LogLevel.WARNING,
            "Failed to deregister from Consul",
            {
                "node_id": self.node_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
```

### Key Consul Patterns

1. **Always non-blocking**: Failures don't stop node startup
2. **Use service tags**: Metadata via tags (not meta parameter)
3. **Unique service_id**: Include node_id for uniqueness
4. **Health check URL**: Point to `/health` endpoint
5. **Store service_id**: Save for deregistration on shutdown
6. **Graceful failure**: Handle ImportError and connection errors

---

## 9. Logging Pattern

### Standard Logging Pattern

```python
# Import logging at module level
import logging
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

logger = logging.getLogger(__name__)

# Use emit_log_event for structured logging
emit_log_event(
    LogLevel.INFO,
    "Operation completed successfully",
    {
        "node_id": self.node_id,
        "correlation_id": str(correlation_id),
        "duration_ms": round(duration_ms, 2),
        "result_count": len(results),
    },
)

# Use logger for detailed debugging (not emitted to event bus)
logger.debug(
    f"Processing batch {batch_num}/{total_batches}",
    extra={
        "batch_num": batch_num,
        "batch_size": len(batch),
    },
)

# For exceptions, use exc_info=True
logger.error(
    f"Unexpected error: {type(e).__name__}",
    exc_info=True,
    extra={"correlation_id": str(correlation_id)},
)
```

### Log Levels

```python
# DEBUG: Verbose information for debugging
emit_log_event(
    LogLevel.DEBUG,
    "Executing workflow step: {step_type}",
    {"step_type": step_type, "step_id": step_id},
)

# INFO: Normal operations, milestones, completions
emit_log_event(
    LogLevel.INFO,
    "Workflow completed successfully",
    {"duration_ms": duration_ms, "steps_executed": step_count},
)

# WARNING: Degraded mode, fallbacks, retries
emit_log_event(
    LogLevel.WARNING,
    "Service unavailable, using fallback",
    {"service": "onextree", "fallback": "no_intelligence"},
)

# ERROR: Operation failures, exceptions
emit_log_event(
    LogLevel.ERROR,
    f"Operation failed: {e}",
    {"error": str(e), "error_type": type(e).__name__},
)
```

### Context Keys

Always include these keys in log context:
- `node_id`: Node identifier
- `correlation_id`: Request correlation ID
- `duration_ms`: Operation duration
- `error`: Error message (for failures)
- `error_type`: Exception type (for failures)

---

## 10. Metrics Tracking

### Standard Metrics Pattern

```python
def __init__(self, container: ModelContainer) -> None:
    """Initialize with metrics tracking."""
    super().__init__(container)

    # Metrics tracking
    self._total_operations = 0
    self._total_duration_ms = 0.0
    self._failed_operations = 0
    self._successful_operations = 0

async def execute_{type}(self, contract: ModelContract{Type}) -> {OutputModel}:
    """Execute with metrics tracking."""
    start_time = time.perf_counter()

    try:
        # Execute operation
        result = await self._do_operation()

        # Track success metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._total_operations += 1
        self._successful_operations += 1
        self._total_duration_ms += duration_ms

        return result

    except Exception as e:
        # Track failure metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._total_operations += 1
        self._failed_operations += 1
        self._total_duration_ms += duration_ms

        raise

def get_metrics(self) -> dict[str, Any]:
    """
    Get metrics for monitoring and alerting.

    Returns:
        Dictionary with metrics
    """
    avg_duration_ms = (
        self._total_duration_ms / self._total_operations
        if self._total_operations > 0
        else 0
    )

    success_rate = (
        self._successful_operations / self._total_operations
        if self._total_operations > 0
        else 1.0
    )

    return {
        "total_operations": self._total_operations,
        "successful_operations": self._successful_operations,
        "failed_operations": self._failed_operations,
        "success_rate": round(success_rate, 4),
        "avg_duration_ms": round(avg_duration_ms, 2),
        "total_duration_ms": round(self._total_duration_ms, 2),
    }
```

### Time Tracking Pattern

```python
import time

# Always use time.perf_counter() for high-precision timing
start_time = time.perf_counter()

# ... operation ...

# Calculate duration in milliseconds
duration_ms = (time.perf_counter() - start_time) * 1000

# Round to 2 decimal places for logging
emit_log_event(
    LogLevel.INFO,
    "Operation completed",
    {"duration_ms": round(duration_ms, 2)},
)
```

---

## 11. Main Entry Point

### Standard Main Pattern

```python
def main() -> int:
    """
    Entry point for node execution.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from omnibase_core.infrastructure.node_base import NodeBase

        # Contract filename - standard ONEX pattern
        CONTRACT_FILENAME = "contract.yaml"

        node_base = NodeBase(Path(__file__).parent / CONTRACT_FILENAME)
        return 0
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"Node{Name}{Type} execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    exit(main())
```

### Key Points

1. **Always return int** (exit code)
2. **Use Path(__file__).parent** for contract location
3. **Contract filename** is always "contract.yaml"
4. **Return 0** on success, **1** on failure
5. **Log failure** with emit_log_event
6. **Use exit(main())** at bottom

---

## 12. Type Patterns

### Type Hints

```python
from typing import Any, Optional, cast
from uuid import UUID
from datetime import datetime
from pathlib import Path

# Method signatures
async def execute_orchestration(
    self, contract: ModelContractOrchestrator
) -> ModelOutputState:
    """Execute orchestration."""
    pass

# Optional types
self._kafka_client: Optional[KafkaClient] = None

# Dict types
config: dict[str, Any] = {}
metrics: dict[str, float] = {}

# Cast for type narrowing
config_value = cast(str, container.config.get("key"))
```

### Pydantic Models

```python
from pydantic import BaseModel, Field, field_validator

class ModelConfig(BaseModel):
    """Configuration model with validation."""

    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        le=300.0,
        description="Operation timeout in seconds"
    )

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is reasonable."""
        if v < 0.1 or v > 300.0:
            raise ValueError("timeout_seconds must be between 0.1 and 300.0")
        return v
```

---

## Summary: Must-Have Checklist

When generating a new node, ensure it has:

- [ ] Standard imports (omnibase_core, logging, typing, datetime, uuid)
- [ ] Proper class declaration extending Node{Type}
- [ ] Comprehensive docstring with examples
- [ ] Defensive __init__ with health_check_mode detection
- [ ] Consul configuration (host, port, enable_registration)
- [ ] Service resolution from container
- [ ] Metrics tracking (total_operations, duration, failures)
- [ ] execute_{type} method with proper signature
- [ ] Event publishing with OnexEnvelopeV1
- [ ] Error handling (re-raise OnexError, wrap others)
- [ ] startup() and shutdown() lifecycle methods
- [ ] _register_with_consul_sync() and _deregister_from_consul()
- [ ] emit_log_event for structured logging
- [ ] time.perf_counter() for duration tracking
- [ ] get_metrics() method
- [ ] main() entry point
- [ ] Type hints throughout

---

## Comparison: Production vs Templates

### What's MISSING from current templates:

1. **Health check mode detection** - Templates don't skip initialization
2. **Consul service discovery** - Templates don't register with Consul
3. **OnexEnvelopeV1 wrapping** - Templates use old publish() method
4. **Comprehensive error handling** - Templates don't wrap all exception types
5. **Metrics tracking** - Templates have minimal metrics
6. **Service resolution** - Templates don't get services from container
7. **Lifecycle methods** - Templates missing startup/shutdown
8. **Defensive configuration** - Templates don't handle container.config edge cases
9. **Event metadata** - Templates don't include consul_service_id
10. **Time precision** - Templates use time.time() instead of perf_counter()

### What templates DO WELL:

1. Basic structure (class declaration, imports)
2. Method signatures (execute_{type})
3. Type hints
4. Docstrings (need more detail)

---

## Next Steps for Code Generation

1. **Update templates** with all patterns from this document
2. **Add Consul integration** to all node types
3. **Use OnexEnvelopeV1** for all event publishing
4. **Add health_check_mode** detection
5. **Improve error handling** with specific exception types
6. **Add metrics tracking** throughout
7. **Use time.perf_counter()** for all timing
8. **Add lifecycle methods** (startup/shutdown)
9. **Improve configuration** with defensive patterns
10. **Add main() entry point** with proper return codes

---

**Generated**: 2025-11-05
**For**: omninode_bridge code generation service
**Status**: Production patterns catalogued, ready for template updates
