# Runtime Host Architecture Implementation Plan

**Created**: December 3, 2025
**Updated**: December 3, 2025 (Architectural Refinements)
**Status**: Ready for Execution
**Estimated Duration**: 6-8 weeks
**Dependencies**: omnibase_core ^0.3.5, omnibase_spi ^0.2.0

---

## Executive Summary

This plan details the complete implementation of the ONEX Runtime Host architecture, transitioning from a 1-container-per-node model to a unified Runtime Host model. The implementation spans three repositories (omnibase_core, omnibase_spi, omnibase_infra) with strict dependency ordering.

### Key Architectural Invariants

These invariants MUST be maintained throughout the implementation:

1. **Core is transport-agnostic**: `omnibase_core` has NO Kafka/HTTP/DB/Vault imports anywhere
2. **NodeRuntime is a pure in-memory orchestrator**: It does NOT own any event loop or bus consumer
3. **Event bus consumption is an infra concern**: `RuntimeHostProcess` + `ProtocolEventBus` drive the runtime
4. **Handlers use strong typing**: `handler_type` returns `EnumHandlerType`, not `str`
5. **LocalHandler is dev/test-only**: Never used in production contracts
6. **Single source of truth for handler registry**: No duplicate registration logic

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     RuntimeHostProcess (omnibase_infra)          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           KafkaEventBus (ProtocolEventBus impl)            │ │
│  │  Consumes envelopes → calls runtime.route_envelope(...)    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                NodeRuntime (omnibase_core)                  │ │
│  │  Pure in-memory orchestrator - NO event loop, NO I/O       │ │
│  │                                                             │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │ │
│  │  │NodeInstance │ │NodeInstance │ │NodeInstance │ ...      │ │
│  │  │  vault_     │ │  consul_    │ │  postgres_  │          │ │
│  │  │  adapter    │ │  projector  │ │  adapter    │          │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘          │ │
│  │                                                             │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │ Handlers: [local*] [http] [db] [vault] [consul]      │  │ │
│  │  │ *local is dev/test only                               │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ FileRegistry | HealthEndpoint | MetricsCollector           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Handler vs Event Bus Separation

| Abstraction | Purpose | Location | Example |
|-------------|---------|----------|---------|
| `ProtocolHandler` | Per-request operations (db/http/vault/etc.) | SPI protocol, infra impl | `HttpHandler`, `DbHandler` |
| `ProtocolEventBus` | Message transport feeding envelopes into runtime | SPI protocol, infra impl | `KafkaEventBus` |

**Key distinction**:
- `ProtocolHandler` = "do this specific thing for me" (synchronous request/response)
- `ProtocolEventBus` = "deliver messages to me" (async event consumption)

---

## Phase 0: Prerequisites & Validation

**Duration**: 1-2 days
**Repository**: All

### 0.1 Dependency Validation

```bash
# Verify PyPI packages
pip index versions omnibase-core  # Should show ≥0.3.5
pip index versions omnibase-spi   # Should show ≥0.2.0

# Test current imports
python -c "from omnibase_core.nodes import NodeEffect; print('Core OK')"
python -c "from omnibase_spi.protocols import ProtocolEventBus; print('SPI OK')"
```

### 0.2 Repository State Verification

| Repository | Branch | State Required |
|------------|--------|----------------|
| omnibase_core | main | Clean, passing CI |
| omnibase_spi | main | Clean, passing CI |
| omnibase_infra | main | Fresh setup complete |

### 0.3 Success Criteria
- [ ] All dependencies installable from PyPI
- [ ] Import tests pass
- [ ] All repositories have clean main branches
- [ ] Development environments configured

---

## Phase 1: Core Types (omnibase_core)

**Duration**: 1 week
**Repository**: omnibase_core
**Priority**: CRITICAL - Must complete before all other phases

### 1.1 New Enum Values

**File**: `src/omnibase_core/enums/enum_node_kind.py`

```python
from enum import Enum

class EnumNodeKind(str, Enum):
    """Types of ONEX nodes."""
    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"
    RUNTIME_HOST = "runtime_host"  # NEW
```

**File**: `src/omnibase_core/enums/enum_handler_type.py` (NEW)

```python
from enum import Enum

class EnumHandlerType(str, Enum):
    """Types of protocol handlers for per-request operations.

    Note: Kafka is NOT in this enum. Kafka is an event bus (ProtocolEventBus),
    not a per-request handler (ProtocolHandler). Use ProtocolEventBus for
    message consumption and delivery.
    """
    LOCAL = "local"      # Echo/test - dev/test only, no external deps
    HTTP = "http"        # HTTP REST calls
    DB = "db"            # Database operations
    LLM = "llm"          # LLM API calls
    VAULT = "vault"      # Vault secret management
    CONSUL = "consul"    # Consul service discovery
```

### 1.2 OnexEnvelope Model

**File**: `src/omnibase_core/models/runtime/model_onex_envelope.py` (NEW)

```python
"""Unified message envelope for all node communication."""
from __future__ import annotations

from datetime import datetime, UTC
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from omnibase_core.enums.enum_handler_type import EnumHandlerType


class ModelOnexEnvelope(BaseModel):
    """Unified message envelope for Runtime Host communication.

    All communication between nodes, handlers, and external systems
    flows through this envelope format.
    """

    # Identity
    envelope_id: UUID = Field(default_factory=uuid4, description="Unique envelope identifier")
    envelope_version: str = Field(default="1.0.0", description="Envelope schema version")

    # Correlation
    correlation_id: UUID = Field(default_factory=uuid4, description="Request correlation ID")
    causation_id: UUID | None = Field(default=None, description="ID of envelope that caused this one")

    # Routing
    source_node: str = Field(description="Source node slug")
    target_node: str | None = Field(default=None, description="Target node slug (None for broadcast)")
    handler_type: EnumHandlerType = Field(description="Handler type for processing")
    operation: str = Field(description="Operation to perform")

    # Payload
    payload: dict[str, Any] = Field(default_factory=dict, description="Operation payload")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Timing
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Creation timestamp")
    ttl_seconds: int | None = Field(default=None, description="Time-to-live in seconds")

    # Response (for reply envelopes)
    is_response: bool = Field(default=False, description="Whether this is a response envelope")
    success: bool | None = Field(default=None, description="Operation success (for responses)")
    error: str | None = Field(default=None, description="Error message (for failures)")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
        }
```

### 1.3 RuntimeHostContract Model

**File**: `src/omnibase_core/models/contracts/model_runtime_host_contract.py` (NEW)

```python
"""Runtime Host contract definition."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from omnibase_core.enums.enum_handler_type import EnumHandlerType


class ModelHandlerConfig(BaseModel):
    """Configuration for a protocol handler.

    TODO (post-MVP): Introduce per-handler config models (e.g. ModelHttpHandlerConfig,
    ModelDbHandlerConfig) and validate `config` against them, instead of using
    free-form `dict[str, Any]`.
    """
    handler_type: EnumHandlerType
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class ModelEventBusConfig(BaseModel):
    """Configuration for the event bus.

    The event bus is separate from handlers - it's the transport that
    feeds envelopes into NodeRuntime via RuntimeHostProcess.
    """
    enabled: bool = True
    # Transport-agnostic config - infra provides concrete implementation
    config: dict[str, Any] = Field(default_factory=dict)


class ModelNodeRef(BaseModel):
    """Reference to a node to be loaded."""
    slug: str = Field(description="Node slug identifier")
    contract_path: str = Field(description="Path to node contract YAML")
    enabled: bool = True
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ModelRuntimeHostContract(BaseModel):
    """Contract defining a Runtime Host configuration.

    The Runtime Host contract specifies which nodes to load,
    which handlers to enable, and how to configure them.

    Note: Event bus configuration is separate from handlers.
    Handlers are for per-request operations; the event bus is
    for message transport.
    """

    # Identity
    name: str = Field(description="Runtime host name")
    version: str = Field(default="1.0.0", description="Contract version")
    description: str = Field(default="", description="Human-readable description")

    # Node Configuration
    nodes: list[ModelNodeRef] = Field(default_factory=list, description="Nodes to load")
    contracts_directory: str = Field(default="contracts", description="Base directory for contracts")

    # Handler Configuration (for per-request operations)
    handlers: list[ModelHandlerConfig] = Field(default_factory=list, description="Handler configurations")

    # Event Bus Configuration (transport-agnostic)
    event_bus: ModelEventBusConfig = Field(
        default_factory=ModelEventBusConfig,
        description="Event bus config (transport implemented in infra)"
    )

    # Health & Metrics
    health_endpoint: dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "port": 8080, "path": "/health"}
    )
    metrics_endpoint: dict[str, Any] = Field(
        default_factory=lambda: {"enabled": True, "port": 9090, "path": "/metrics"}
    )

    # Runtime Settings
    max_concurrent_operations: int = Field(default=100, description="Max concurrent operations")
    shutdown_timeout_seconds: int = Field(default=30, description="Graceful shutdown timeout")

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelRuntimeHostContract":
        """Load contract from YAML file."""
        from omnibase_core.utils.util_safe_yaml_loader import load_and_validate_yaml_model
        return load_and_validate_yaml_model(path, cls)
```

### 1.4 Handler Protocol Interface (Strongly Typed)

**File**: `src/omnibase_core/protocols/protocol_handler.py` (NEW)

```python
"""Protocol interface for Runtime Host handlers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from omnibase_core.enums.enum_handler_type import EnumHandlerType

if TYPE_CHECKING:
    from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope


class ProtocolHandler(ABC):
    """Abstract protocol for Runtime Host handlers.

    Handlers are responsible for executing per-request operations defined in
    OnexEnvelopes. Each handler type (http, db, vault, etc.) implements
    this protocol to provide its specific functionality.

    Note: This is for synchronous request/response operations.
    For message transport (event consumption/production), use ProtocolEventBus.
    """

    @property
    @abstractmethod
    def handler_type(self) -> EnumHandlerType:
        """Return the handler type identifier.

        Returns:
            EnumHandlerType - strongly typed, not a string
        """
        ...

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize handler with configuration.

        Args:
            config: Handler-specific configuration dictionary
        """
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the handler."""
        ...

    @abstractmethod
    async def execute(self, envelope: "ModelOnexEnvelope") -> "ModelOnexEnvelope":
        """Execute an operation from an envelope.

        Args:
            envelope: Input envelope with operation details

        Returns:
            Response envelope with operation results
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict:
        """Check handler health.

        Returns:
            Dict with 'healthy' bool and optional details
        """
        ...
```

### 1.5 NodeInstance Class

**File**: `src/omnibase_core/runtime/node_instance.py` (NEW)

```python
"""Lightweight node instance wrapper for Runtime Host."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope

if TYPE_CHECKING:
    from omnibase_core.models.contracts.model_node_contract import ModelNodeContract
    from omnibase_core.runtime.node_runtime import NodeRuntime


class NodeInstance:
    """Lightweight wrapper around node business logic.

    NodeInstance represents a single node within the Runtime Host.
    It contains no event loop or I/O code - all operations are
    delegated to handlers through the parent runtime.

    Implementation Note:
        Currently, NodeInstance delegates all operation execution to NodeRuntime,
        which in turn routes to the appropriate ProtocolHandler based on
        `envelope.handler_type`. This keeps NodeInstance free of I/O details
        and maintains the architectural invariant that nodes are pure logic.
    """

    def __init__(
        self,
        contract: "ModelNodeContract",
        runtime: "NodeRuntime",
    ) -> None:
        self._contract = contract
        self._runtime = runtime
        self._logger = logging.getLogger(f"node.{contract.name}")
        self._initialized = False
        self._operation_handlers: dict[str, Callable] = {}

    @property
    def slug(self) -> str:
        """Return node slug identifier."""
        return self._contract.name

    @property
    def node_type(self) -> str:
        """Return node type (effect, compute, reducer, orchestrator)."""
        return self._contract.node_type

    @property
    def contract(self) -> "ModelNodeContract":
        """Return the node contract."""
        return self._contract

    async def initialize(self) -> None:
        """Initialize the node instance."""
        self._logger.info(f"Initializing node instance: {self.slug}")

        # Register operation handlers from contract
        for op in self._contract.io_operations or []:
            self._operation_handlers[op.operation] = self._create_operation_handler(op)

        self._initialized = True
        self._logger.info(f"Node instance initialized: {self.slug}")

    async def shutdown(self) -> None:
        """Shutdown the node instance."""
        self._logger.info(f"Shutting down node instance: {self.slug}")
        self._initialized = False

    async def handle(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Handle an incoming envelope.

        Delegates execution to NodeRuntime which routes to the appropriate
        ProtocolHandler based on envelope.handler_type.

        Args:
            envelope: Input envelope to process

        Returns:
            Response envelope with operation results
        """
        if not self._initialized:
            return self._error_response(envelope, "Node not initialized")

        operation = envelope.operation

        # Validate operation is supported
        if operation not in self._operation_handlers:
            return self._error_response(
                envelope,
                f"Unknown operation: {operation}. Available: {list(self._operation_handlers.keys())}"
            )

        # Delegate to runtime for handler execution
        try:
            return await self._runtime.execute_with_handler(envelope)
        except Exception as e:
            self._logger.exception(f"Error handling operation {operation}")
            return self._error_response(envelope, str(e))

    def _create_operation_handler(self, operation_config: Any) -> Callable:
        """Create a handler function for an operation."""
        async def handler(envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
            return await self._runtime.execute_with_handler(envelope)
        return handler

    def _error_response(self, envelope: ModelOnexEnvelope, error: str) -> ModelOnexEnvelope:
        """Create an error response envelope."""
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node=self.slug,
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            is_response=True,
            success=False,
            error=error,
        )
```

### 1.6 NodeRuntime Class (Transport-Agnostic)

**File**: `src/omnibase_core/runtime/node_runtime.py` (NEW)

```python
"""Core NodeRuntime implementation for Runtime Host."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.enums.enum_handler_type import EnumHandlerType
from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope
from omnibase_core.runtime.node_instance import NodeInstance

if TYPE_CHECKING:
    from omnibase_core.models.contracts.model_runtime_host_contract import ModelRuntimeHostContract
    from omnibase_core.protocols.protocol_handler import ProtocolHandler


class NodeRuntime:
    """Core runtime that hosts multiple node instances.

    NodeRuntime is a pure in-memory orchestrator. It manages node instances,
    routes operations to handlers, and provides shared infrastructure for
    all hosted nodes.

    IMPORTANT ARCHITECTURAL INVARIANT:
        NodeRuntime does NOT own any event loop or message consumer.
        Event bus consumption is implemented in RuntimeHostProcess (omnibase_infra)
        using ProtocolEventBus. RuntimeHostProcess calls route_envelope() for
        each received message.

        This separation ensures:
        - Core remains transport-agnostic (no Kafka/NATS/etc. imports)
        - Testing is simplified (no real message bus needed)
        - Different transports can be swapped in infra without touching core
    """

    def __init__(self, contract: "ModelRuntimeHostContract") -> None:
        self._contract = contract
        self._logger = logging.getLogger(f"runtime.{contract.name}")

        # Node instances
        self._nodes: dict[str, NodeInstance] = {}

        # Handlers (for per-request operations)
        self._handlers: dict[EnumHandlerType, "ProtocolHandler"] = {}

        # State
        self._running = False

    @property
    def name(self) -> str:
        """Return runtime name."""
        return self._contract.name

    @property
    def nodes(self) -> dict[str, NodeInstance]:
        """Return registered node instances."""
        return self._nodes

    @property
    def handlers(self) -> dict[EnumHandlerType, "ProtocolHandler"]:
        """Return registered handlers."""
        return self._handlers

    @property
    def is_running(self) -> bool:
        """Return whether runtime is running."""
        return self._running

    def register_handler(self, handler: "ProtocolHandler") -> None:
        """Register a protocol handler.

        Args:
            handler: Handler instance to register (must return EnumHandlerType)
        """
        # handler.handler_type is now EnumHandlerType, no conversion needed
        handler_type = handler.handler_type
        self._handlers[handler_type] = handler
        self._logger.info(f"Registered handler: {handler_type}")

    def register_node(self, node: NodeInstance) -> None:
        """Register a node instance.

        Args:
            node: Node instance to register
        """
        self._nodes[node.slug] = node
        self._logger.info(f"Registered node: {node.slug}")

    async def load_nodes_from_directory(self, contracts_dir: Path) -> None:
        """Load all node contracts from a directory.

        Args:
            contracts_dir: Directory containing node contract YAML files
        """
        from omnibase_core.runtime.file_registry import FileRegistry

        registry = FileRegistry(contracts_dir)
        contracts = registry.load_all()

        for contract in contracts:
            node = NodeInstance(contract, self)
            self.register_node(node)

    async def initialize(self) -> None:
        """Initialize the runtime and all components."""
        self._logger.info(f"Initializing runtime: {self.name}")

        # Initialize handlers
        for handler_type, handler in self._handlers.items():
            config = self._get_handler_config(handler_type)
            await handler.initialize(config)
            self._logger.info(f"Handler initialized: {handler_type}")

        # Initialize nodes
        for slug, node in self._nodes.items():
            await node.initialize()
            self._logger.info(f"Node initialized: {slug}")

        self._running = True
        self._logger.info(f"Runtime initialized with {len(self._nodes)} nodes and {len(self._handlers)} handlers")

    async def shutdown(self) -> None:
        """Shutdown the runtime gracefully."""
        self._logger.info(f"Shutting down runtime: {self.name}")
        self._running = False

        # Shutdown nodes
        for slug, node in self._nodes.items():
            await node.shutdown()
            self._logger.info(f"Node shutdown: {slug}")

        # Shutdown handlers
        for handler_type, handler in self._handlers.items():
            await handler.shutdown()
            self._logger.info(f"Handler shutdown: {handler_type}")

        self._logger.info(f"Runtime stopped: {self.name}")

    async def execute_with_handler(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Execute an envelope using the appropriate handler.

        Args:
            envelope: Envelope to execute

        Returns:
            Response envelope
        """
        handler_type = envelope.handler_type

        if handler_type not in self._handlers:
            return ModelOnexEnvelope(
                correlation_id=envelope.correlation_id,
                causation_id=envelope.envelope_id,
                source_node="runtime",
                handler_type=handler_type,
                operation=envelope.operation,
                is_response=True,
                success=False,
                error=f"No handler registered for type: {handler_type}",
            )

        handler = self._handlers[handler_type]
        return await handler.execute(envelope)

    async def route_envelope(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Route an envelope to the appropriate node.

        This is the main entry point for envelopes coming from the event bus.
        RuntimeHostProcess calls this method for each message received.

        Args:
            envelope: Envelope to route

        Returns:
            Response envelope
        """
        target = envelope.target_node

        if target is None:
            # Broadcast to all nodes (not typical)
            return await self._broadcast_envelope(envelope)

        if target not in self._nodes:
            return ModelOnexEnvelope(
                correlation_id=envelope.correlation_id,
                causation_id=envelope.envelope_id,
                source_node="runtime",
                handler_type=envelope.handler_type,
                operation=envelope.operation,
                is_response=True,
                success=False,
                error=f"Unknown target node: {target}",
            )

        node = self._nodes[target]
        return await node.handle(envelope)

    async def _broadcast_envelope(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Broadcast envelope to all nodes."""
        results = []
        for node in self._nodes.values():
            result = await node.handle(envelope)
            results.append(result)

        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="runtime",
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            is_response=True,
            success=all(r.success for r in results if r.success is not None),
            payload={"results": [r.model_dump() for r in results]},
        )

    def _get_handler_config(self, handler_type: EnumHandlerType) -> dict:
        """Get configuration for a handler type."""
        for handler_config in self._contract.handlers:
            if handler_config.handler_type == handler_type:
                return handler_config.config
        return {}

    async def health_check(self) -> dict:
        """Check health of runtime and all components."""
        handler_health = {}
        for handler_type, handler in self._handlers.items():
            handler_health[handler_type.value] = await handler.health_check()

        node_health = {}
        for slug, node in self._nodes.items():
            node_health[slug] = {"initialized": node._initialized}

        return {
            "runtime": self.name,
            "running": self._running,
            "handlers": handler_health,
            "nodes": node_health,
            "healthy": self._running and all(
                h.get("healthy", False) for h in handler_health.values()
            ),
        }
```

### 1.7 FileRegistry Class

**File**: `src/omnibase_core/runtime/file_registry.py` (NEW)

```python
"""File-based contract registry for Runtime Host."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.contracts.model_node_contract import ModelNodeContract


class FileRegistry:
    """Registry that loads node contracts from filesystem.

    FileRegistry scans a directory for YAML contract files and
    loads them into ModelNodeContract instances.
    """

    def __init__(self, contracts_dir: Path) -> None:
        self._contracts_dir = contracts_dir
        self._contracts: dict[str, "ModelNodeContract"] = {}
        self._logger = logging.getLogger("file_registry")

    def load_all(self) -> list["ModelNodeContract"]:
        """Load all contracts from the directory.

        Returns:
            List of loaded node contracts
        """
        from omnibase_core.models.contracts.model_node_contract import ModelNodeContract
        from omnibase_core.utils.util_safe_yaml_loader import load_and_validate_yaml_model

        contracts = []

        if not self._contracts_dir.exists():
            self._logger.warning(f"Contracts directory does not exist: {self._contracts_dir}")
            return contracts

        # Find all YAML files
        for yaml_file in self._contracts_dir.glob("**/*.yaml"):
            if yaml_file.name.startswith("_"):
                continue  # Skip private/schema files

            try:
                contract = load_and_validate_yaml_model(yaml_file, ModelNodeContract)
                self._contracts[contract.name] = contract
                contracts.append(contract)
                self._logger.info(f"Loaded contract: {contract.name} from {yaml_file}")
            except Exception as e:
                self._logger.error(f"Failed to load contract from {yaml_file}: {e}")

        return contracts

    def get(self, slug: str) -> "ModelNodeContract | None":
        """Get a contract by slug.

        Args:
            slug: Node slug identifier

        Returns:
            Contract if found, None otherwise
        """
        return self._contracts.get(slug)

    def list_all(self) -> list[str]:
        """List all loaded contract slugs."""
        return list(self._contracts.keys())
```

### 1.8 Local Handler (Dev/Test Only)

**File**: `src/omnibase_core/runtime/handlers/local_handler.py` (NEW)

```python
"""Local echo handler for testing - no external dependencies."""
from __future__ import annotations

import logging
from typing import Any

from omnibase_core.enums.enum_handler_type import EnumHandlerType
from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope
from omnibase_core.protocols.protocol_handler import ProtocolHandler


class LocalHandler(ProtocolHandler):
    """Local echo handler for testing and development only.

    WARNING:
        This handler is not intended for production use. It exists to
        validate Runtime Host wiring without external dependencies.
        Production handlers (http/db/vault/etc.) are all implemented
        in `omnibase_infra`.

    This handler has no external dependencies and simply echoes
    back the input payload. Used for testing the runtime infrastructure.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("handler.local")
        self._initialized = False
        self._config: dict[str, Any] = {}

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return handler type as EnumHandlerType (not str)."""
        return EnumHandlerType.LOCAL

    async def initialize(self, config: dict) -> None:
        """Initialize the local handler."""
        self._config = config
        self._initialized = True
        self._logger.info("Local handler initialized (dev/test only)")

    async def shutdown(self) -> None:
        """Shutdown the local handler."""
        self._initialized = False
        self._logger.info("Local handler shutdown")

    async def execute(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Echo back the envelope payload.

        Supports operations:
        - echo: Return payload as-is
        - transform: Apply simple transformations
        - error: Simulate an error response
        """
        operation = envelope.operation

        if operation == "echo":
            return self._success_response(envelope, envelope.payload)

        elif operation == "transform":
            # Simple transformation: uppercase all string values
            transformed = self._transform_payload(envelope.payload)
            return self._success_response(envelope, transformed)

        elif operation == "error":
            error_msg = envelope.payload.get("error_message", "Simulated error")
            return self._error_response(envelope, error_msg)

        else:
            return self._success_response(envelope, {
                "operation": operation,
                "payload": envelope.payload,
                "message": "Unknown operation, echoing back",
            })

    async def health_check(self) -> dict:
        """Check handler health."""
        return {
            "healthy": self._initialized,
            "handler_type": self.handler_type.value,
            "dev_test_only": True,
        }

    def _success_response(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """Create a success response envelope."""
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="local_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            payload=payload,
            is_response=True,
            success=True,
        )

    def _error_response(self, envelope: ModelOnexEnvelope, error: str) -> ModelOnexEnvelope:
        """Create an error response envelope."""
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="local_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            is_response=True,
            success=False,
            error=error,
        )

    def _transform_payload(self, payload: dict) -> dict:
        """Transform payload values."""
        result = {}
        for key, value in payload.items():
            if isinstance(value, str):
                result[key] = value.upper()
            elif isinstance(value, dict):
                result[key] = self._transform_payload(value)
            else:
                result[key] = value
        return result
```

### 1.9 CLI Entry Point (Dev/Test)

**File**: `src/omnibase_core/cli/runtime_host_cli.py` (NEW)

```python
"""CLI for running the Runtime Host (dev/test mode - local handler only).

For production use, use omnibase_infra.runtime.runtime_host_process which
provides real handlers and event bus integration.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from omnibase_core.models.contracts.model_runtime_host_contract import ModelRuntimeHostContract
from omnibase_core.runtime.node_runtime import NodeRuntime
from omnibase_core.runtime.handlers.local_handler import LocalHandler


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def run_runtime(contract_path: Path) -> None:
    """Run the runtime host with the given contract (dev/test mode).

    This CLI only registers LocalHandler for testing purposes.
    For production, use RuntimeHostProcess from omnibase_infra.
    """
    # Load contract
    contract = ModelRuntimeHostContract.from_yaml(contract_path)

    # Create runtime
    runtime = NodeRuntime(contract)

    # Register local handler only (dev/test)
    runtime.register_handler(LocalHandler())

    # Load nodes
    contracts_dir = contract_path.parent / contract.contracts_directory
    if contracts_dir.exists():
        await runtime.load_nodes_from_directory(contracts_dir)

    # Initialize
    await runtime.initialize()

    logging.info("Runtime initialized (dev/test mode - LocalHandler only)")
    logging.info("For production, use omnibase_infra.runtime.runtime_host_process")

    # Keep running until interrupted
    try:
        while runtime.is_running:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await runtime.shutdown()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ONEX Runtime Host (dev/test mode)",
        epilog="For production use, see omnibase_infra.runtime.runtime_host_process",
    )
    parser.add_argument(
        "contract",
        type=Path,
        help="Path to runtime host contract YAML",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)

    if not args.contract.exists():
        print(f"Contract file not found: {args.contract}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_runtime(args.contract))


if __name__ == "__main__":
    main()
```

### 1.10 Phase 1 Success Criteria

- [ ] `EnumNodeKind.RUNTIME_HOST` added and exported
- [ ] `EnumHandlerType` enum created (WITHOUT Kafka - that's an event bus)
- [ ] `ModelOnexEnvelope` model complete with serialization
- [ ] `ModelRuntimeHostContract` model with separate event_bus config
- [ ] `ProtocolHandler` abstract class with `EnumHandlerType` return type
- [ ] `NodeInstance` class with envelope handling (pass-through documented)
- [ ] `NodeRuntime` class - NO event loop, NO bus consumer
- [ ] `FileRegistry` class for contract loading
- [ ] `LocalHandler` working with dev/test warnings
- [ ] CLI entry point clearly marked as dev/test only
- [ ] Unit tests for all new classes (>90% coverage)
- [ ] mypy --strict passes
- [ ] **VERIFICATION**: No Kafka/HTTP/DB/Vault imports in omnibase_core

---

## Phase 2: SPI Protocol Updates (omnibase_spi)

**Duration**: 3-4 days
**Repository**: omnibase_spi
**Depends On**: Phase 1 complete

### 2.1 Handler Protocol Export

**File**: `src/omnibase_spi/protocols/__init__.py` (UPDATE)

```python
# Add to existing exports
from omnibase_core.protocols.protocol_handler import ProtocolHandler

__all__ = [
    # ... existing exports
    "ProtocolHandler",
]
```

### 2.2 Event Bus Protocol Updates

**File**: `src/omnibase_spi/protocols/protocol_event_bus.py` (UPDATE)

The event bus protocol handles message transport - separate from handlers:

```python
"""Protocol for event bus implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope


class ProtocolEventBus(ABC):
    """Abstract protocol for event bus implementations.

    The event bus is responsible for message transport - delivering envelopes
    to NodeRuntime via RuntimeHostProcess. This is separate from ProtocolHandler
    which handles per-request operations.

    Implementation Note:
        - KafkaEventBus implements this protocol in omnibase_infra
        - RuntimeHostProcess owns the event bus and calls runtime.route_envelope()
        - NodeRuntime is transport-agnostic and only knows about envelopes
    """

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize the event bus connection."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shutdown the event bus."""
        ...

    @abstractmethod
    async def publish_envelope(self, envelope: "ModelOnexEnvelope", topic: str) -> None:
        """Publish an OnexEnvelope to a topic.

        Args:
            envelope: Envelope to publish
            topic: Target topic name
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: Callable[["ModelOnexEnvelope"], Awaitable["ModelOnexEnvelope"]],
    ) -> None:
        """Subscribe to envelopes on a topic.

        Args:
            topic: Topic to subscribe to
            handler: Async callback for each received envelope
        """
        ...

    @abstractmethod
    async def start_consuming(self) -> None:
        """Start the consumer loop.

        This runs until shutdown() is called.
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict:
        """Check event bus health."""
        ...
```

### 2.3 Phase 2 Success Criteria

- [ ] `ProtocolHandler` exported from SPI
- [ ] `ProtocolEventBus` updated with envelope methods
- [ ] Clear separation documented: Handler vs EventBus
- [ ] Backward compatibility maintained
- [ ] Tests passing

---

## Phase 3: Infrastructure Handlers (omnibase_infra)

**Duration**: 2 weeks
**Repository**: omnibase_infra
**Depends On**: Phase 1 and Phase 2 complete

### 3.1 Directory Structure

```
src/omnibase_infra/
├── handlers/                          # Protocol handlers (per-request ops)
│   ├── __init__.py
│   ├── http_handler.py                # HTTP REST handler
│   ├── db_handler.py                  # PostgreSQL handler
│   ├── vault_handler.py               # Vault secrets handler
│   ├── consul_handler.py              # Consul service discovery handler
│   └── llm_handler.py                 # LLM API handler (Phase 4)
├── event_bus/                         # Event bus implementations (transport)
│   ├── __init__.py
│   └── kafka_event_bus.py             # KafkaEventBus (ProtocolEventBus impl)
├── runtime/                           # Runtime host process
│   ├── __init__.py
│   ├── runtime_host_process.py        # Main process wrapper
│   └── wiring.py                      # Single source of truth for handlers
├── contracts/                         # Example contracts
│   ├── runtime/
│   │   └── infra_runtime_host.yaml    # Example runtime contract
│   └── nodes/
│       ├── vault_adapter.yaml
│       ├── consul_adapter.yaml
│       └── postgres_adapter.yaml
└── ... (existing structure)
```

### 3.2 HTTP Handler (Strongly Typed)

**File**: `src/omnibase_infra/handlers/http_handler.py`

```python
"""HTTP REST protocol handler using httpx."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from omnibase_core.enums.enum_handler_type import EnumHandlerType
from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope
from omnibase_core.protocols.protocol_handler import ProtocolHandler


class HttpHandler(ProtocolHandler):
    """HTTP REST protocol handler.

    Executes HTTP requests defined in OnexEnvelopes.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("handler.http")
        self._client: httpx.AsyncClient | None = None
        self._config: dict[str, Any] = {}

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return handler type as EnumHandlerType."""
        return EnumHandlerType.HTTP

    async def initialize(self, config: dict) -> None:
        """Initialize HTTP client."""
        self._config = config
        timeout = config.get("timeout_seconds", 30)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )
        self._logger.info("HTTP handler initialized")

    async def shutdown(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
        self._logger.info("HTTP handler shutdown")

    async def execute(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Execute HTTP request from envelope."""
        if not self._client:
            return self._error_response(envelope, "HTTP client not initialized")

        payload = envelope.payload

        try:
            method = payload.get("method", "GET").upper()
            url = payload.get("url")
            headers = payload.get("headers", {})
            body = payload.get("body")
            params = payload.get("params", {})

            if not url:
                return self._error_response(envelope, "URL is required")

            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if body else None,
                params=params,
            )

            return self._success_response(envelope, {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
            })

        except httpx.TimeoutException:
            return self._error_response(envelope, "Request timed out")
        except Exception as e:
            return self._error_response(envelope, str(e))

    async def health_check(self) -> dict:
        """Check HTTP handler health."""
        return {
            "healthy": self._client is not None,
            "handler_type": self.handler_type.value,
        }

    def _success_response(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="http_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            payload=payload,
            is_response=True,
            success=True,
        )

    def _error_response(self, envelope: ModelOnexEnvelope, error: str) -> ModelOnexEnvelope:
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="http_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            is_response=True,
            success=False,
            error=error,
        )
```

### 3.3 Database Handler (Strongly Typed)

**File**: `src/omnibase_infra/handlers/db_handler.py`

```python
"""PostgreSQL database protocol handler."""
from __future__ import annotations

import logging
from typing import Any

import asyncpg

from omnibase_core.enums.enum_handler_type import EnumHandlerType
from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope
from omnibase_core.protocols.protocol_handler import ProtocolHandler


class DbHandler(ProtocolHandler):
    """PostgreSQL database protocol handler.

    Executes database operations defined in OnexEnvelopes.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("handler.db")
        self._pool: asyncpg.Pool | None = None
        self._config: dict[str, Any] = {}

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return handler type as EnumHandlerType."""
        return EnumHandlerType.DB

    async def initialize(self, config: dict) -> None:
        """Initialize database connection pool."""
        self._config = config

        self._pool = await asyncpg.create_pool(
            host=config.get("host", "localhost"),
            port=config.get("port", 5432),
            database=config.get("database", "omnibase"),
            user=config.get("user", "postgres"),
            password=config.get("password", ""),
            min_size=config.get("pool_min_size", 2),
            max_size=config.get("pool_max_size", 10),
        )
        self._logger.info("Database handler initialized")

    async def shutdown(self) -> None:
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
        self._logger.info("Database handler shutdown")

    async def execute(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Execute database operation from envelope."""
        if not self._pool:
            return self._error_response(envelope, "Database pool not initialized")

        payload = envelope.payload
        operation = envelope.operation

        try:
            if operation == "query":
                return await self._execute_query(envelope, payload)
            elif operation == "execute":
                return await self._execute_command(envelope, payload)
            elif operation == "transaction":
                return await self._execute_transaction(envelope, payload)
            else:
                return self._error_response(envelope, f"Unknown database operation: {operation}")

        except asyncpg.PostgresError as e:
            return self._error_response(envelope, f"Database error: {e}")
        except Exception as e:
            return self._error_response(envelope, str(e))

    async def _execute_query(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """Execute a query and return results."""
        sql = payload.get("sql")
        params = payload.get("params", [])

        if not sql:
            return self._error_response(envelope, "SQL query is required")

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return self._success_response(envelope, {
                "rows": [dict(row) for row in rows],
                "row_count": len(rows),
            })

    async def _execute_command(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """Execute a command (INSERT, UPDATE, DELETE)."""
        sql = payload.get("sql")
        params = payload.get("params", [])

        if not sql:
            return self._error_response(envelope, "SQL command is required")

        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, *params)
            return self._success_response(envelope, {
                "result": result,
            })

    async def _execute_transaction(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """Execute multiple statements in a transaction."""
        statements = payload.get("statements", [])

        if not statements:
            return self._error_response(envelope, "Transaction statements required")

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                results = []
                for stmt in statements:
                    sql = stmt.get("sql")
                    params = stmt.get("params", [])
                    result = await conn.execute(sql, *params)
                    results.append(result)

                return self._success_response(envelope, {
                    "results": results,
                    "statement_count": len(results),
                })

    async def health_check(self) -> dict:
        """Check database handler health."""
        if not self._pool:
            return {"healthy": False, "handler_type": self.handler_type.value, "error": "Pool not initialized"}

        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"healthy": True, "handler_type": self.handler_type.value}
        except Exception as e:
            return {"healthy": False, "handler_type": self.handler_type.value, "error": str(e)}

    def _success_response(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="db_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            payload=payload,
            is_response=True,
            success=True,
        )

    def _error_response(self, envelope: ModelOnexEnvelope, error: str) -> ModelOnexEnvelope:
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="db_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            is_response=True,
            success=False,
            error=error,
        )
```

### 3.4 Kafka Event Bus (ProtocolEventBus, NOT ProtocolHandler)

**File**: `src/omnibase_infra/event_bus/kafka_event_bus.py`

```python
"""Kafka event bus implementation."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope
from omnibase_spi.protocols.protocol_event_bus import ProtocolEventBus


class KafkaEventBus(ProtocolEventBus):
    """Kafka implementation of ProtocolEventBus.

    This is the transport layer that feeds envelopes into NodeRuntime.
    It implements ProtocolEventBus, NOT ProtocolHandler.

    RuntimeHostProcess owns this and calls runtime.route_envelope()
    for each message received.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("event_bus.kafka")
        self._producer: AIOKafkaProducer | None = None
        self._consumer: AIOKafkaConsumer | None = None
        self._config: dict[str, Any] = {}
        self._running = False
        self._handler: Callable[[ModelOnexEnvelope], Awaitable[ModelOnexEnvelope]] | None = None
        self._subscribed_topics: list[str] = []

    async def initialize(self, config: dict) -> None:
        """Initialize Kafka connections."""
        self._config = config
        bootstrap_servers = config.get("bootstrap_servers", "localhost:9092")
        consumer_group = config.get("consumer_group", "runtime-host")

        # Producer for publishing responses
        self._producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self._producer.start()

        # Consumer will be started when subscribe is called
        self._bootstrap_servers = bootstrap_servers
        self._consumer_group = consumer_group

        self._logger.info("Kafka event bus initialized")

    async def shutdown(self) -> None:
        """Close Kafka connections."""
        self._running = False

        if self._producer:
            await self._producer.stop()

        if self._consumer:
            await self._consumer.stop()

        self._logger.info("Kafka event bus shutdown")

    async def publish_envelope(self, envelope: ModelOnexEnvelope, topic: str) -> None:
        """Publish an envelope to a topic."""
        if not self._producer:
            raise RuntimeError("Kafka producer not initialized")

        key = str(envelope.correlation_id).encode("utf-8")
        value = envelope.model_dump(mode="json")

        await self._producer.send_and_wait(topic, value=value, key=key)
        self._logger.debug(f"Published envelope to {topic}: {envelope.envelope_id}")

    async def subscribe(
        self,
        topic: str,
        handler: Callable[[ModelOnexEnvelope], Awaitable[ModelOnexEnvelope]],
    ) -> None:
        """Subscribe to a topic with a handler."""
        self._handler = handler
        self._subscribed_topics.append(topic)
        self._logger.info(f"Subscribed to topic: {topic}")

    async def start_consuming(self) -> None:
        """Start the consumer loop."""
        if not self._subscribed_topics:
            self._logger.warning("No topics subscribed, not starting consumer")
            return

        self._consumer = AIOKafkaConsumer(
            *self._subscribed_topics,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._consumer_group,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await self._consumer.start()

        self._running = True
        self._logger.info(f"Started consuming from: {self._subscribed_topics}")

        try:
            async for msg in self._consumer:
                if not self._running:
                    break

                try:
                    # Parse envelope from message
                    envelope = ModelOnexEnvelope.model_validate(msg.value)

                    # Call the handler (which calls runtime.route_envelope)
                    if self._handler:
                        response = await self._handler(envelope)

                        # Optionally publish response to response topic
                        if response.is_response and response.target_node:
                            response_topic = self._config.get("response_topic")
                            if response_topic:
                                await self.publish_envelope(response, response_topic)

                except Exception as e:
                    self._logger.exception(f"Error processing message: {e}")

        except Exception as e:
            self._logger.exception(f"Consumer loop error: {e}")
        finally:
            self._running = False

    async def health_check(self) -> dict:
        """Check event bus health."""
        return {
            "healthy": self._producer is not None,
            "running": self._running,
            "subscribed_topics": self._subscribed_topics,
        }
```

### 3.5 Vault Handler (Strongly Typed)

**File**: `src/omnibase_infra/handlers/vault_handler.py`

```python
"""Vault secrets management protocol handler."""
from __future__ import annotations

import logging
from typing import Any

import hvac

from omnibase_core.enums.enum_handler_type import EnumHandlerType
from omnibase_core.models.runtime.model_onex_envelope import ModelOnexEnvelope
from omnibase_core.protocols.protocol_handler import ProtocolHandler


class VaultHandler(ProtocolHandler):
    """Vault secrets management protocol handler.

    Handles secret read/write operations via HashiCorp Vault.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("handler.vault")
        self._client: hvac.Client | None = None
        self._config: dict[str, Any] = {}

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return handler type as EnumHandlerType."""
        return EnumHandlerType.VAULT

    async def initialize(self, config: dict) -> None:
        """Initialize Vault client."""
        self._config = config

        self._client = hvac.Client(
            url=config.get("url", "http://localhost:8200"),
            token=config.get("token"),
            namespace=config.get("namespace"),
        )
        self._logger.info("Vault handler initialized")

    async def shutdown(self) -> None:
        """Close Vault client."""
        self._client = None
        self._logger.info("Vault handler shutdown")

    async def execute(self, envelope: ModelOnexEnvelope) -> ModelOnexEnvelope:
        """Execute Vault operation from envelope."""
        if not self._client:
            return self._error_response(envelope, "Vault client not initialized")

        operation = envelope.operation
        payload = envelope.payload

        try:
            if operation == "get_secret":
                return await self._get_secret(envelope, payload)
            elif operation == "set_secret":
                return await self._set_secret(envelope, payload)
            elif operation == "delete_secret":
                return await self._delete_secret(envelope, payload)
            elif operation == "list_secrets":
                return await self._list_secrets(envelope, payload)
            else:
                return self._error_response(envelope, f"Unknown Vault operation: {operation}")

        except hvac.exceptions.VaultError as e:
            return self._error_response(envelope, f"Vault error: {e}")
        except Exception as e:
            return self._error_response(envelope, str(e))

    async def _get_secret(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """Get a secret from Vault."""
        path = payload.get("path")
        mount_point = payload.get("mount_point", "secret")

        if not path:
            return self._error_response(envelope, "Secret path is required")

        response = self._client.secrets.kv.v2.read_secret_version(
            path=path,
            mount_point=mount_point,
        )

        return self._success_response(envelope, {
            "data": response["data"]["data"],
            "metadata": response["data"]["metadata"],
        })

    async def _set_secret(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """Set a secret in Vault."""
        path = payload.get("path")
        data = payload.get("data")
        mount_point = payload.get("mount_point", "secret")

        if not path or not data:
            return self._error_response(envelope, "Secret path and data are required")

        response = self._client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=data,
            mount_point=mount_point,
        )

        return self._success_response(envelope, {
            "version": response["data"]["version"],
            "created_time": response["data"]["created_time"],
        })

    async def _delete_secret(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """Delete a secret from Vault."""
        path = payload.get("path")
        mount_point = payload.get("mount_point", "secret")

        if not path:
            return self._error_response(envelope, "Secret path is required")

        self._client.secrets.kv.v2.delete_metadata_and_all_versions(
            path=path,
            mount_point=mount_point,
        )

        return self._success_response(envelope, {"deleted": True})

    async def _list_secrets(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        """List secrets at a path."""
        path = payload.get("path", "")
        mount_point = payload.get("mount_point", "secret")

        response = self._client.secrets.kv.v2.list_secrets(
            path=path,
            mount_point=mount_point,
        )

        return self._success_response(envelope, {
            "keys": response["data"]["keys"],
        })

    async def health_check(self) -> dict:
        """Check Vault handler health."""
        if not self._client:
            return {"healthy": False, "handler_type": self.handler_type.value, "error": "Client not initialized"}

        try:
            health = self._client.sys.read_health_status(method="GET")
            return {
                "healthy": health.get("initialized", False) and not health.get("sealed", True),
                "handler_type": self.handler_type.value,
                "vault_version": health.get("version"),
            }
        except Exception as e:
            return {"healthy": False, "handler_type": self.handler_type.value, "error": str(e)}

    def _success_response(self, envelope: ModelOnexEnvelope, payload: dict) -> ModelOnexEnvelope:
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="vault_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            payload=payload,
            is_response=True,
            success=True,
        )

    def _error_response(self, envelope: ModelOnexEnvelope, error: str) -> ModelOnexEnvelope:
        return ModelOnexEnvelope(
            correlation_id=envelope.correlation_id,
            causation_id=envelope.envelope_id,
            source_node="vault_handler",
            target_node=envelope.source_node,
            handler_type=envelope.handler_type,
            operation=envelope.operation,
            is_response=True,
            success=False,
            error=error,
        )
```

### 3.6 Handler Wiring (Single Source of Truth)

**File**: `src/omnibase_infra/runtime/wiring.py`

```python
"""Handler registration and wiring utilities.

This is the SINGLE SOURCE OF TRUTH for handler registration.
RuntimeHostProcess uses this - no duplicate logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.enums.enum_handler_type import EnumHandlerType
from omnibase_core.runtime.handlers.local_handler import LocalHandler

from omnibase_infra.handlers.http_handler import HttpHandler
from omnibase_infra.handlers.db_handler import DbHandler
from omnibase_infra.handlers.vault_handler import VaultHandler

if TYPE_CHECKING:
    from omnibase_core.protocols.protocol_handler import ProtocolHandler
    from omnibase_core.runtime.node_runtime import NodeRuntime


# Single source of truth for handler classes
HANDLER_REGISTRY: dict[EnumHandlerType, type["ProtocolHandler"]] = {
    EnumHandlerType.LOCAL: LocalHandler,  # Dev/test only
    EnumHandlerType.HTTP: HttpHandler,
    EnumHandlerType.DB: DbHandler,
    EnumHandlerType.VAULT: VaultHandler,
    # EnumHandlerType.CONSUL: ConsulHandler,  # TODO: implement
    # EnumHandlerType.LLM: LlmHandler,  # TODO: implement
}


def register_handlers_from_config(
    runtime: "NodeRuntime",
    handler_configs: list[dict],
) -> None:
    """Register handlers with the runtime based on config.

    This is the ONLY place where handlers are registered.

    Args:
        runtime: NodeRuntime instance
        handler_configs: List of handler config dicts from contract
    """
    for config in handler_configs:
        handler_type = EnumHandlerType(config["handler_type"])
        enabled = config.get("enabled", True)

        if not enabled:
            continue

        handler_class = HANDLER_REGISTRY.get(handler_type)
        if handler_class:
            handler = handler_class()
            runtime.register_handler(handler)
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")


def get_handler_class(handler_type: EnumHandlerType) -> type["ProtocolHandler"] | None:
    """Get handler class by type."""
    return HANDLER_REGISTRY.get(handler_type)


def list_available_handlers() -> list[EnumHandlerType]:
    """List all available handler types."""
    return list(HANDLER_REGISTRY.keys())
```

### 3.7 RuntimeHostProcess (Uses Wiring, Owns Event Bus)

**File**: `src/omnibase_infra/runtime/runtime_host_process.py`

```python
"""Runtime Host Process - Infrastructure-level wrapper for NodeRuntime."""
from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.models.contracts.model_runtime_host_contract import ModelRuntimeHostContract
from omnibase_core.runtime.node_runtime import NodeRuntime

from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
from omnibase_infra.runtime.wiring import register_handlers_from_config

if TYPE_CHECKING:
    from omnibase_spi.protocols.protocol_event_bus import ProtocolEventBus


class RuntimeHostProcess:
    """Infrastructure-level process wrapper for NodeRuntime.

    RuntimeHostProcess is responsible for:
    - Loading and validating the runtime contract
    - Registering infrastructure handlers (via wiring.py)
    - Owning and managing the event bus (ProtocolEventBus)
    - Driving NodeRuntime with envelopes from the event bus
    - Managing process lifecycle and signals
    - Providing health/metrics endpoints

    IMPORTANT:
        This is where event bus consumption happens.
        NodeRuntime is transport-agnostic - it only knows about envelopes.
        This class bridges the transport (Kafka) to the runtime.
    """

    def __init__(self, contract_path: Path) -> None:
        self._contract_path = contract_path
        self._logger = logging.getLogger("runtime_host_process")
        self._runtime: NodeRuntime | None = None
        self._event_bus: "ProtocolEventBus | None" = None
        self._shutdown_requested = False

    async def start(self) -> None:
        """Start the runtime host process."""
        self._logger.info(f"Starting Runtime Host Process: {self._contract_path}")

        # Load contract
        contract = ModelRuntimeHostContract.from_yaml(self._contract_path)

        # Create runtime
        self._runtime = NodeRuntime(contract)

        # Register handlers (single source of truth: wiring.py)
        handler_configs = [
            {"handler_type": h.handler_type.value, "enabled": h.enabled, **h.config}
            for h in contract.handlers
        ]
        register_handlers_from_config(self._runtime, handler_configs)

        # Load nodes
        contracts_dir = self._contract_path.parent / contract.contracts_directory
        if contracts_dir.exists():
            await self._runtime.load_nodes_from_directory(contracts_dir)

        # Initialize runtime (handlers + nodes)
        await self._runtime.initialize()

        # Initialize event bus if enabled
        if contract.event_bus.enabled:
            await self._setup_event_bus(contract)

        # Setup signal handlers
        self._setup_signals()

        self._logger.info("Runtime Host Process started")

        # Start event bus consumer loop (this is where envelopes flow in)
        if self._event_bus:
            await self._event_bus.start_consuming()
        else:
            # No event bus - just wait for shutdown
            while not self._shutdown_requested:
                await asyncio.sleep(1)

    async def _setup_event_bus(self, contract: ModelRuntimeHostContract) -> None:
        """Setup the event bus."""
        self._event_bus = KafkaEventBus()
        await self._event_bus.initialize(contract.event_bus.config)

        # Subscribe to command topic
        command_topic = contract.event_bus.config.get(
            "command_topic",
            f"onex.cmd.runtime.{contract.name}.v1"
        )

        # The handler bridges event bus to NodeRuntime
        await self._event_bus.subscribe(
            command_topic,
            self._runtime.route_envelope,  # This is where the bridge happens
        )

        self._logger.info(f"Event bus subscribed to: {command_topic}")

    async def stop(self) -> None:
        """Stop the runtime host process."""
        self._shutdown_requested = True

        if self._event_bus:
            await self._event_bus.shutdown()

        if self._runtime:
            await self._runtime.shutdown()

        self._logger.info("Runtime Host Process stopped")

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler)

    def _signal_handler(self) -> None:
        """Handle shutdown signals."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._logger.info("Shutdown signal received")
            asyncio.create_task(self.stop())
```

### 3.8 Example Runtime Contract (Clean Separation)

**File**: `src/omnibase_infra/contracts/runtime/infra_runtime_host.yaml`

```yaml
# Infrastructure Runtime Host Contract
name: "infra_runtime_host"
version: "1.0.0"
description: "ONEX Infrastructure Runtime Host - hosts all infrastructure nodes"

# Node Configuration
contracts_directory: "../nodes"
nodes:
  - slug: "vault_adapter"
    contract_path: "vault_adapter.yaml"
    enabled: true
  - slug: "consul_adapter"
    contract_path: "consul_adapter.yaml"
    enabled: true
  - slug: "postgres_adapter"
    contract_path: "postgres_adapter.yaml"
    enabled: true

# Handler Configuration (per-request operations)
# Note: Kafka is NOT a handler - it's the event bus (see below)
handlers:
  # LocalHandler is dev/test only - DO NOT enable in production
  # - handler_type: "local"
  #   enabled: false

  - handler_type: "http"
    enabled: true
    config:
      timeout_seconds: 30

  - handler_type: "db"
    enabled: true
    config:
      host: "${POSTGRES_HOST}"
      port: "${POSTGRES_PORT}"
      database: "${POSTGRES_DB}"
      user: "${POSTGRES_USER}"
      password: "${POSTGRES_PASSWORD}"
      pool_min_size: 2
      pool_max_size: 10

  - handler_type: "vault"
    enabled: true
    config:
      url: "${VAULT_ADDR}"
      token: "${VAULT_TOKEN}"
      namespace: "${VAULT_NAMESPACE}"

# Event Bus Configuration (transport layer - feeds envelopes to runtime)
# This is SEPARATE from handlers. Kafka is a transport, not a per-request handler.
event_bus:
  enabled: true
  config:
    bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
    consumer_group: "infra-runtime-host"
    command_topic: "onex.cmd.runtime.infra.v1"
    response_topic: "onex.evt.runtime.infra.v1"

# Health & Metrics
health_endpoint:
  enabled: true
  port: 8080
  path: "/health"

metrics_endpoint:
  enabled: true
  port: 9090
  path: "/metrics"

# Runtime Settings
max_concurrent_operations: 100
shutdown_timeout_seconds: 30
```

### 3.9 Phase 3 Success Criteria

- [ ] HttpHandler with `EnumHandlerType.HTTP` return type
- [ ] DbHandler with `EnumHandlerType.DB` return type
- [ ] VaultHandler with `EnumHandlerType.VAULT` return type
- [ ] KafkaEventBus implements `ProtocolEventBus` (NOT ProtocolHandler)
- [ ] `wiring.py` is single source of truth for handler registration
- [ ] RuntimeHostProcess uses `wiring.py` (no duplicate handler map)
- [ ] RuntimeHostProcess owns event bus and calls `runtime.route_envelope()`
- [ ] Example contract shows clean handler vs event_bus separation
- [ ] Integration tests for each handler
- [ ] End-to-end test: event bus → runtime → handler → response

---

## Phase 4-5: Integration, Testing & Deployment

*(Phases 4 and 5 remain largely the same as before, with tests updated to use the new architecture)*

### Key Test Scenarios

1. **LocalHandler echo** (dev/test validation)
2. **HttpHandler external call** (real HTTP request)
3. **DbHandler query** (real database query)
4. **VaultHandler secret** (real Vault operation)
5. **Event bus → NodeRuntime → handler flow** (full integration)
6. **Multi-node routing** (envelope routing to correct node)

---

## Architectural Sanity Checklist

Use this checklist to verify the implementation maintains architectural invariants:

### omnibase_core
- [ ] Has: `EnumHandlerType`, `ModelOnexEnvelope`, `ModelRuntimeHostContract`, `NodeInstance`, `NodeRuntime`, `ProtocolHandler`, `LocalHandler` (dev-only), CLI test runtime
- [ ] Has **NO** Kafka/HTTP/DB/Vault imports anywhere
- [ ] `NodeRuntime` has NO `_event_bus_loop` method
- [ ] `ProtocolHandler.handler_type` returns `EnumHandlerType` (not `str`)
- [ ] `LocalHandler` has dev/test warnings in docstring

### omnibase_spi
- [ ] Exposes: `ProtocolHandler`, `ProtocolEventBus`
- [ ] `ProtocolEventBus` has envelope methods
- [ ] Clear documentation of handler vs event bus distinction

### omnibase_infra
- [ ] Handlers return `EnumHandlerType` (not `str`)
- [ ] `KafkaEventBus` implements `ProtocolEventBus` (NOT `ProtocolHandler`)
- [ ] `wiring.py` is single source of truth
- [ ] `RuntimeHostProcess` uses `wiring.py` for handler registration
- [ ] `RuntimeHostProcess` owns event bus, calls `runtime.route_envelope()`
- [ ] No duplicate handler registration logic

### Contracts
- [ ] `handlers` section contains only per-request handlers (http, db, vault)
- [ ] `event_bus` section is separate from handlers
- [ ] `local` handler is NOT enabled in production contracts

---

## Summary Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Phase 0 + Phase 1 Start | Prerequisites validated, core types begun |
| 2 | Phase 1 Complete | All core classes (transport-agnostic), CLI, local handler |
| 3 | Phase 2 + Phase 3 Start | SPI updates, HTTP handler, KafkaEventBus |
| 4 | Phase 3 Continue | DB, Vault handlers, wiring |
| 5 | Phase 3 Complete + Phase 4 Start | RuntimeHostProcess, integration tests |
| 6 | Phase 4 Complete | All tests passing, benchmarks met |
| 7-8 | Phase 5 | Docker, deployment, migration |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Memory per 10 nodes | <200MB | tracemalloc |
| Envelope throughput | >100/sec | Benchmark suite |
| Handler latency (local) | <1ms | p99 latency |
| Handler latency (http) | <100ms | p99 latency |
| Handler latency (db) | <50ms | p99 latency |
| Test coverage | >90% | pytest-cov |
| Migration downtime | 0 | Shadow deployment |
| Core Kafka imports | 0 | grep verification |

---

*This implementation plan was created on December 3, 2025 and updated with architectural refinements.*
