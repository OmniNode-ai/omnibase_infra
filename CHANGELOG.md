# Changelog

All notable changes to the ONEX Infrastructure (omnibase_infra) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Changes

> **IMPORTANT**: This section documents API changes that may require code modifications when upgrading. Review each item carefully before upgrading.

#### File and Class Naming Standardization (OMN-1305, PR #151)

This refactoring enforces consistent naming conventions across the entire codebase per CLAUDE.md standards. **All import paths and class names have changed.**

##### Summary of Changes

| Category | Count | Pattern Change |
|----------|-------|----------------|
| Event Bus | 2 files, 2 classes | `{name}_event_bus` → `event_bus_{name}` |
| Handlers | 6 files, 6 classes | Suffix → Prefix standardization |
| Protocols | 4 files, 4 classes | Removed `Handler` suffix, domain-specific naming |
| Runtime | 6 files, 6 classes | Added `service_`, `util_`, `registry_` prefixes |
| Validation | 8 files, 8 classes | `{name}_validator` → `validator_{name}` |
| Stores | 2 classes | Suffix → Prefix standardization |

##### Event Bus Renames

| Old File | New File |
|----------|----------|
| `inmemory_event_bus.py` | `event_bus_inmemory.py` |
| `kafka_event_bus.py` | `event_bus_kafka.py` |

| Old Class | New Class |
|-----------|-----------|
| `InMemoryEventBus` | `EventBusInmemory` |
| `KafkaEventBus` | `EventBusKafka` |

**Migration**:
```python
# BEFORE
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus

# AFTER
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
```

##### Handler Renames

| Old File | New File |
|----------|----------|
| `handler_mock_registration_storage.py` | `handler_registration_storage_mock.py` |
| `handler_postgres_registration_storage.py` | `handler_registration_storage_postgres.py` |
| `handler_consul_service_discovery.py` | `handler_service_discovery_consul.py` |
| `handler_mock_service_discovery.py` | `handler_service_discovery_mock.py` |

| Old Class | New Class |
|-----------|-----------|
| `MockRegistrationStorageHandler` | `HandlerRegistrationStorageMock` |
| `PostgresRegistrationStorageHandler` | `HandlerRegistrationStoragePostgres` |
| `ConsulServiceDiscoveryHandler` | `HandlerServiceDiscoveryConsul` |
| `MockServiceDiscoveryHandler` | `HandlerServiceDiscoveryMock` |
| `HttpRestHandler` | `HandlerHttpRest` |

**Migration**:
```python
# BEFORE
from omnibase_infra.handlers.registration_storage.handler_postgres_registration_storage import (
    PostgresRegistrationStorageHandler,
)

# AFTER
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)
```

##### Protocol Renames

| Old File | New File |
|----------|----------|
| `protocol_registration_storage_handler.py` | `protocol_registration_persistence.py` |
| `protocol_service_discovery_handler.py` | `protocol_discovery_operations.py` |

| Old Class | New Class |
|-----------|-----------|
| `ProtocolRegistrationStorageHandler` | `ProtocolRegistrationPersistence` |
| `ProtocolServiceDiscoveryHandler` | `ProtocolDiscoveryOperations` |

**Migration**:
```python
# BEFORE
from omnibase_infra.handlers.registration_storage.protocol_registration_storage_handler import (
    ProtocolRegistrationStorageHandler,
)

# AFTER
from omnibase_infra.handlers.registration_storage.protocol_registration_persistence import (
    ProtocolRegistrationPersistence,
)
```

##### Runtime File Renames

| Old File | New File | Rationale |
|----------|----------|-----------|
| `policy_registry.py` | `registry_policy.py` | Registry prefix pattern |
| `message_dispatch_engine.py` | `service_message_dispatch_engine.py` | Service prefix pattern |
| `runtime_host_process.py` | `service_runtime_host_process.py` | Service prefix pattern |
| `wiring.py` | `util_wiring.py` | Util prefix pattern |
| `container_wiring.py` | `util_container_wiring.py` | Util prefix pattern |
| `validation.py` | `util_validation.py` | Util prefix pattern |

| Old Class | New Class |
|-----------|-----------|
| `PolicyRegistry` | `RegistryPolicy` |
| `ProtocolBindingRegistry` | `RegistryProtocolBinding` |
| `MessageTypeRegistry` | `RegistryMessageType` |
| `EventBusBindingRegistry` | `RegistryEventBusBinding` |

**Migration**:
```python
# BEFORE
from omnibase_infra.runtime.policy_registry import PolicyRegistry
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

# AFTER
from omnibase_infra.runtime.registry_policy import RegistryPolicy
from omnibase_infra.runtime.service_message_dispatch_engine import MessageDispatchEngine
```

##### Validation File Renames

| Old File | New File |
|----------|----------|
| `any_type_validator.py` | `validator_any_type.py` |
| `chain_propagation_validator.py` | `validator_chain_propagation.py` |
| `contract_linter.py` | `linter_contract.py` |
| `registration_security_validator.py` | `validator_registration_security.py` |
| `routing_coverage_validator.py` | `validator_routing_coverage.py` |
| `runtime_shape_validator.py` | `validator_runtime_shape.py` |
| `security_validator.py` | `validator_security.py` |
| `topic_category_validator.py` | `validator_topic_category.py` |
| `validation_aggregator.py` | `service_validation_aggregator.py` |

> **Note**: Class names within validation files remain unchanged (e.g., `AnyTypeDetector`, `ChainPropagationValidator`). Only import paths changed.

**Migration**:
```python
# BEFORE
from omnibase_infra.validation.any_type_validator import AnyTypeDetector
from omnibase_infra.validation.chain_propagation_validator import ChainPropagationValidator

# AFTER
from omnibase_infra.validation.validator_any_type import AnyTypeDetector
from omnibase_infra.validation.validator_chain_propagation import ChainPropagationValidator
```

##### Store Class Renames

| Old Class | New Class |
|-----------|-----------|
| `InMemoryIdempotencyStore` | `StoreIdempotencyInmemory` |
| `PostgresIdempotencyStore` | `StoreIdempotencyPostgres` |

##### Automated Migration

Run these commands to find affected imports in your codebase:

```bash
# Find all affected imports
grep -rE "(InMemoryEventBus|KafkaEventBus|PolicyRegistry|inmemory_event_bus|kafka_event_bus)" \
    --include="*.py" /path/to/your/code

# Specific patterns for each category
grep -r "from omnibase_infra.event_bus.inmemory_event_bus" --include="*.py" .
grep -r "from omnibase_infra.event_bus.kafka_event_bus" --include="*.py" .
grep -r "from omnibase_infra.runtime.policy_registry" --include="*.py" .
grep -r "from omnibase_infra.validation.any_type_validator" --include="*.py" .
```

##### CI Enforcement

A new naming validator (`scripts/validation/validate_naming.py`) enforces these conventions. The CI pipeline will reject PRs that violate naming standards.

#### MixinNodeIntrospection API (OMN-881, PR #54)

##### 1. Cache Invalidation Method Signature Change

**`invalidate_introspection_cache()` is now synchronous (was async)**

This is a **breaking change** for any code that awaits this method.

| Aspect | Details |
|--------|---------|
| **What changed** | Method signature changed from `async def` to `def` (synchronous) |
| **Why it changed** | Cache invalidation is a simple in-memory operation (setting `_introspection_cache = None`) that does not require async I/O. Synchronous semantics simplify usage and avoid unnecessary coroutine overhead. |
| **Error if not migrated** | `TypeError: object NoneType can't be used in 'await' expression` |

**Migration Steps**:

```python
# BEFORE (will cause TypeError after upgrade)
await node.invalidate_introspection_cache()

# AFTER (correct usage)
node.invalidate_introspection_cache()
```

**Search pattern** to find affected code:
```bash
grep -r "await.*invalidate_introspection_cache" --include="*.py"
```

##### 2. Configuration Model API

**`initialize_introspection()` requires `ModelIntrospectionConfig`**

The initialization method uses a typed configuration model for all parameters.

| Aspect | Details |
|--------|---------|
| **What changed** | `initialize_introspection(config: ModelIntrospectionConfig)` is the initialization API |
| **Why** | Typed configuration model provides validation, IDE support, and extensibility |
| **Model location** | `omnibase_infra.models.discovery.ModelIntrospectionConfig` |

**Usage Example**:

```python
from uuid import uuid4
from omnibase_infra.models.discovery import ModelIntrospectionConfig
from omnibase_infra.mixins import MixinNodeIntrospection

class MyNode(MixinNodeIntrospection):
    def __init__(self, event_bus=None):
        config = ModelIntrospectionConfig(
            node_id=uuid4(),
            node_type="EFFECT",
            event_bus=event_bus,
            version="1.0.0",
            cache_ttl=300.0,
        )
        self.initialize_introspection(config)

    async def shutdown(self):
        # Note: invalidate_introspection_cache() is now SYNC (see above)
        self.invalidate_introspection_cache()
```

#### Error Code for Unhandled node_kind (OMN-990, PR #73)
- **Error code changed from `VALIDATION_ERROR` to `INTERNAL_ERROR`**: When `DispatchContextEnforcer.create_context_for_dispatcher()` encounters an unhandled `node_kind` value, it now raises `ModelOnexError` with `INTERNAL_ERROR` instead of `VALIDATION_ERROR`.
  - **Old**: `error_code=EnumCoreErrorCode.VALIDATION_ERROR`
  - **New**: `error_code=EnumCoreErrorCode.INTERNAL_ERROR`
  - **Migration**: If you catch `ModelOnexError` and check for `VALIDATION_ERROR` when calling context creation methods, update to check for `INTERNAL_ERROR`.
  - **Rationale**: Unhandled `node_kind` values represent internal implementation errors (missing switch cases in exhaustive pattern matching) rather than user input validation failures. `INTERNAL_ERROR` more accurately reflects that this indicates a bug in the code rather than invalid configuration.

#### Handler Types (PR #33)
- **HANDLER_TYPE_REDIS renamed to HANDLER_TYPE_VALKEY**: The handler type constant for Redis-compatible cache has been renamed to accurately reflect the service name.
  - **Old**: `HANDLER_TYPE_REDIS = "redis"`
  - **New**: `HANDLER_TYPE_VALKEY = "valkey"`
  - **Migration**: Update any references from `HANDLER_TYPE_REDIS` to `HANDLER_TYPE_VALKEY`
  - **Rationale**: Valkey is the correct service name for the Redis-compatible cache used in the infrastructure. This aligns the codebase with the actual service naming.

### Added

#### Node Introspection (OMN-881, PR #54)
- **ModelIntrospectionConfig**: Configuration model for `MixinNodeIntrospection` that provides typed configuration
  - `node_id` (required): Unique identifier for this node instance (UUID)
  - `node_type` (required): Type of node (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR). Cannot be empty (min_length=1).
  - `event_bus`: Optional event bus for publishing introspection and heartbeat events. Uses duck typing (`object | None`) to accept any object implementing `ProtocolEventBus` protocol.
  - `version`: Node version string (default: `"1.0.0"`)
  - `cache_ttl`: Cache time-to-live in seconds (default: `300.0`, minimum: `0.0`)
  - `operation_keywords`: Optional set of keywords to identify operation methods. If None, uses `MixinNodeIntrospection.DEFAULT_OPERATION_KEYWORDS`.
  - `exclude_prefixes`: Optional set of prefixes to exclude from capability discovery. If None, uses `MixinNodeIntrospection.DEFAULT_EXCLUDE_PREFIXES`.
  - `introspection_topic`: Topic for publishing introspection events (default: `"node.introspection"`). ONEX topics (starting with `onex.`) require version suffix (e.g., `.v1`).
  - `heartbeat_topic`: Topic for publishing heartbeat events (default: `"node.heartbeat"`). ONEX topics require version suffix.
  - `request_introspection_topic`: Topic for receiving introspection requests (default: `"node.request_introspection"`). ONEX topics require version suffix.
  - Model is frozen and forbids extra fields for immutability and strict validation.
- **Performance Metrics Tracking**:
  - Added `IntrospectionPerformanceMetrics` dataclass (internal) and `ModelIntrospectionPerformanceMetrics` Pydantic model (for event payloads)
  - Added `get_performance_metrics()` method for monitoring introspection operation timing and threshold violations
  - Performance thresholds: `get_capabilities` <50ms, `discover_capabilities` <30ms, `total_introspection` <50ms, `cache_hit` <1ms
- **Topic Default Constants**: Exported constants for default topic names:
  - `DEFAULT_INTROSPECTION_TOPIC = "node.introspection"`
  - `DEFAULT_HEARTBEAT_TOPIC = "node.heartbeat"`
  - `DEFAULT_REQUEST_INTROSPECTION_TOPIC = "node.request_introspection"`

#### Handlers
- **HttpHandler** (OMN-237, PR #26): HTTP REST protocol handler for MVP
  - GET and POST operations using httpx async client
  - Fixed 30s timeout (configurable timeout deferred to Beta)
  - Returns `EnumHandlerType.HTTP`
  - Error handling mapping to infrastructure errors (`InfraTimeoutError`, `InfraConnectionError`)
  - Full lifecycle support (initialize, shutdown, health_check, describe)
  - 46 unit tests with 97.93% coverage

#### Event Bus
- **InMemoryEventBus** (OMN-239, PR #25): In-memory event bus for local development and testing
  - Implements `ProtocolEventBus` from omnibase_core
  - Topic-based pub/sub with `asyncio.Queue` per topic
  - Thread-safe subscription management
  - Automatic cleanup on unsubscribe
  - Consumer groups with load balancing
  - Graceful shutdown with message draining
  - Comprehensive error handling
  - 1336+ lines of test coverage

#### Runtime
- **ProtocolBindingRegistry** (OMN-240, PR #24): Handler and event bus registration system
  - Single source of truth for handler registration
  - Thread-safe registration operations
  - Support for handler type constants (HTTP, DATABASE, KAFKA, etc.)
  - Event bus registry (InMemory, Kafka)
  - Protocol resolution utilities

#### Errors
- **Infrastructure Error Taxonomy** (OMN-290, PR #23): Structured error hierarchy
  - `RuntimeHostError`: Base infrastructure error class
  - `ProtocolConfigurationError`: Protocol configuration validation errors
  - `SecretResolutionError`: Secret/credential resolution errors
  - `InfraConnectionError`: Infrastructure connection errors (transport-aware)
  - `InfraTimeoutError`: Infrastructure timeout errors
  - `InfraAuthenticationError`: Infrastructure authentication errors
  - `InfraUnavailableError`: Infrastructure resource unavailable errors
  - `ModelInfraErrorContext`: Structured error context model
  - `EnumInfraTransportType`: Transport type classification

#### Infrastructure
- **Directory Structure** (OMN-236, PR #21): Initial MVP directory structure
  - `handlers/`: Protocol handler implementations
  - `event_bus/`: Event bus implementations
  - `runtime/`: Runtime host components
  - `errors/`: Infrastructure error classes
  - `enums/`: Infrastructure enumerations
  - `validation/`: Contract validation utilities

### Changed

#### Handler to Dispatcher Terminology Migration (OMN-977, PR #63)

The codebase has migrated from "handler" to "dispatcher" terminology for message routing components to better reflect their purpose as message dispatchers rather than generic handlers.

- **Protocol Rename**: `ProtocolHandler` → `ProtocolMessageDispatcher`
- **Class Naming**: Handler implementations renamed to Dispatcher (e.g., `UserEventHandler` → `UserEventDispatcher`)
- **ID Convention**: `dispatcher_id` values now use `-dispatcher` suffix instead of `-handler`
- **Enum Rename**: `EnumDispatchStatus.NO_HANDLER` renamed to `NO_DISPATCHER` with new value `no_dispatcher` for consistency with dispatcher terminology
- **Enum Value**: `EnumDispatchStatus.HANDLER_ERROR` retains its current value `handler_error` (rename deferred; see ADR for rationale)
- **Full Migration Guide**: See `docs/migrations/HANDLER_TO_DISPATCHER_MIGRATION.md` for complete migration details and code examples

#### CI/CD
- **Pre-commit Configuration**: Migrated to fix deprecated stage warnings (PR #25)

## [0.1.0] - Unreleased

### Planned

This version represents the MVP (Minimum Viable Product) for ONEX Runtime Host Infrastructure.

#### Core Components (MVP)
- **BaseRuntimeHostProcess** (OMN-249): Infrastructure wrapper owning event bus and driving NodeRuntime
- **HandlerDb** (OMN-238): PostgreSQL database protocol handler
- **wiring.py** (OMN-240): Handler registration module ✅ (implemented as ProtocolBindingRegistry)

#### Testing (MVP)
- **E2E Flow Test** (OMN-254): InMemoryEventBus -> Runtime -> Handler flow
- **Architecture Verification** (OMN-255): Architectural invariant checks

#### Deployment (MVP)
- **Dockerfile** (OMN-256): Basic container image for runtime host
- **docker-compose** (OMN-264): Local development configuration

#### Documentation (MVP)
- **CLAUDE.md Updates** (OMN-265): Runtime Host architecture documentation

### MVP Philosophy

> **MVP (v0.1.0)**: Prove the architecture works end-to-end with minimal scope
> - InMemoryEventBus only (no Kafka complexity)
> - HTTP + DB handlers only (no Vault, no Consul)
> - Simplified contract format
> - Basic error handling
> - Unit tests with mocks

### Deferred to Beta (v0.2.0)

- KafkaEventBus with backpressure
- HandlerVault and HandlerConsul
- Retry logic and rate limiting
- Full graceful shutdown with drain
- Integration tests with real services
- Observability layer (structured logging, metrics)

---

## Architecture

```
omnibase_infra (YOU ARE HERE)
    ├── handlers/          # Protocol handler implementations
    │   ├── http_handler   # HTTP REST handler (MVP)
    │   └── db_handler     # PostgreSQL handler (MVP)
    ├── event_bus/         # Event bus implementations
    │   ├── inmemory       # InMemory bus (MVP)
    │   └── kafka          # Kafka bus (Beta)
    ├── runtime/           # Runtime host components
    │   ├── handler_registry
    │   └── runtime_host_process
    └── errors/            # Infrastructure errors
        └── infra_errors

DEPENDENCY RULE: infra -> spi -> core (never reverse)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### ONEX Standards
- Zero tolerance for `Any` types
- Contract-driven development
- Protocol-based dependency injection
- Comprehensive test coverage (>80% target)

## License

MIT License - See LICENSE file for details
