# Changelog

All notable changes to the ONEX Infrastructure (omnibase_infra) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-05

### Breaking Changes

#### EventBusSubcontractWiring API Change
- **`EventBusSubcontractWiring.__init__()`** now requires two new parameters: `service` and `version`
  - **Old**: `EventBusSubcontractWiring(event_bus, contract)`
  - **New**: `EventBusSubcontractWiring(event_bus, contract, service="my-service", version="1.0.0")`
  - **Migration**: Add `service` and `version` parameters to all `EventBusSubcontractWiring` instantiations

#### Realm-Agnostic Topics
- **Topics no longer include environment prefix**: The `resolve_topic()` function now returns topic suffixes unchanged
  - **Old**: `resolve_topic("events.v1")` returned `"dev.events.v1"` (with env prefix)
  - **New**: `resolve_topic("events.v1")` returns `"events.v1"` (no prefix)
  - **Impact**: Cross-environment event routing now possible; isolation maintained through envelope identity

#### Subscribe Signature Change (omnibase-core 0.14.0)
- **`ProtocolEventBus.subscribe()`** parameter changed from `group_id: str` to `node_identity: ProtocolNodeIdentity`
  - **Old**: `event_bus.subscribe(topic, group_id="my-group", on_message=handler)`
  - **New**: `event_bus.subscribe(topic, node_identity=ModelEmitterIdentity(...), on_message=handler)`
  - **Migration**: Replace `group_id` with `ModelEmitterIdentity(env, service, node_name, version)`

#### ModelIntrospectionConfig Requires node_name
- **`ModelIntrospectionConfig`** now requires `node_name` as a mandatory field
  - **Old**: Could instantiate with only `node_id` and `node_type`
  - **New**: Must also provide `node_name` parameter
  - **Migration**: Add `node_name=<your_node_name>` to all `ModelIntrospectionConfig` instantiations
  - **Failure**: Omitting `node_name` raises `ValidationError`

#### ModelPostgresIntentPayload.endpoints Validation
- **`ModelPostgresIntentPayload.endpoints`** validator now raises `ValueError` for empty Mapping
  - **Old**: Empty `{}` logged a warning and returned empty tuple
  - **New**: Empty `{}` raises `ValueError("endpoints cannot be an empty Mapping")`
  - **Migration**: Ensure `endpoints` is either `None` or a non-empty Mapping

### Deprecated

#### RegistryPolicy.register_policy()
- **`RegistryPolicy.register_policy()`** method is deprecated
  - **Old**: `policy.register_policy(policy_type, priority, handler)`
  - **New**: `policy.register(ModelPolicyRegistration(policy_type, priority, handler))`
  - **Migration**: Replace `register_policy()` calls with `register(ModelPolicyRegistration(...))`
  - **Warning**: Emits `DeprecationWarning` at call site

### Added

#### Slack Webhook Handler (OMN-1905)
- **HandlerSlackWebhook**: Async handler with Block Kit formatting, retry with exponential backoff, and 429 rate limit handling
- **NodeSlackAlerterEffect**: Pure declarative effect node for Slack alerts
- **EnumAlertSeverity**: Severity levels (critical/error/warning/info)
- **ModelSlackAlert/ModelSlackAlertResult**: Type-safe frozen Pydantic models
- Features: Correlation ID tracking, exponential backoff retry (1s → 2s → 4s), 429 rate limit handling

#### Contract Dependency Resolution (OMN-1903, OMN-1732)
- **ContractDependencyResolver**: Reads protocol dependencies from `contract.yaml` and resolves from container
- **ModelResolvedDependencies**: Pydantic model for resolved protocol instances
- **ProtocolDependencyResolutionError**: Fail-fast error for missing protocols
- **RuntimeHostProcess integration**: Automatic dependency resolution during node discovery
- Zero-code nodes can now receive injected dependencies via constructor

#### Event Ledger Integration Tests (OMN-1649)
- Added comprehensive integration tests for Event Ledger runtime wiring

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.13.1` to `^0.14.0`

## [0.3.2] - 2026-02-02

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.12.0` to `^0.13.1`

#### Database Repository Models Migration
- Moved `ModelDbOperation`, `ModelDbParam`, `ModelDbRepositoryContract`, `ModelDbReturn`, `ModelDbSafetyPolicy` from `omnibase_core.models.contracts` to `omnibase_infra.runtime.db.models`
- These infrastructure-specific models are now owned by omnibase_infra
- Import path changed: `from omnibase_infra.runtime.db import ModelDbRepositoryContract, ...`

## [0.3.1] - 2026-02-02

### Fixed

- **OMN-1842**: Fix ORDER BY injection position when LIMIT clause exists in `PostgresRepositoryRuntime` (#229)
  - ORDER BY is now correctly inserted BEFORE existing LIMIT clause to produce valid SQL
  - Added detection for parameterized LIMIT (`$n`) to prevent duplicate LIMIT injection
  - Before (invalid): `SELECT ... LIMIT $1 ORDER BY id`
  - After (valid): `SELECT ... ORDER BY id LIMIT $1`

## [0.3.0] - 2026-02-01

### Added

- Minor version release

### Changed

- Version bump from 0.2.x to 0.3.0

## [0.2.8] - 2026-01-30

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.9.10` to `^0.9.11`
- Update `omnibase-spi` from `^0.6.3` to `^0.6.4`

## [0.2.7] - 2026-01-30

### Changed

#### Dependencies
- Update `omnibase-core` from `^0.9.9` to `^0.9.10` for OMN-1551 contract-driven topics

## [0.2.6] - 2026-01-30

### Added

#### Contract Registry System
- **OMN-1654**: `KafkaContractSource` for cache-based contract discovery from Kafka topics (#213)
- **OMN-1653**: Contract registry reducer with Postgres projection for persistent contract storage (#212)

#### Event Ledger Persistence
- **OMN-1648**: `NodeLedgerProjectionCompute` for event ledger persistence with compute node pattern (#211)
- **OMN-1647**: PostgreSQL handlers for event ledger persistence operations (#209)
- **OMN-1646**: Event ledger schema and models for tracking event processing state (#208)

#### Declarative Configuration & Routing
- **OMN-1519**: `RuntimeContractConfigLoader` for declarative operation bindings from contract.yaml (#210)
- **OMN-1518**: Declarative topic→operation→handler routing with contract-driven dispatch (#198)
- **OMN-1621**: Contract-driven event bus subscription wiring for automatic topic binding (#200)

#### Emit Daemon
- **OMN-1610**: Emit daemon for persistent Kafka connections with connection pooling (#207)

#### Kafka & Event Bus Improvements
- **OMN-1613**: Event bus topic storage in registry for dynamic topic routing (#199)
- **OMN-1602**: Derived Kafka consumer group IDs with deterministic naming (#197)
- **OMN-1547**: Replace hardcoded topics with validated suffix constants (#206)

#### Handler & Intent Improvements
- **OMN-1509**: Intent storage effect node with integration tests (#195)
- **OMN-1515**: `execute()` dispatcher to `HandlerGraph` for contract discovery (#193)
- **OMN-1614, OMN-1616**: Canonical publish interface ADR and test adapter (#201)

### Changed

#### Dependencies
- **omnibase-core**: Updated from ^0.9.6 to ^0.9.9 (baseline topic constants export)
- **omnibase-spi**: Updated from ^0.6.2 to ^0.6.3

## [0.2.3] - 2026-01-25

### Added

#### OMN-1524: Infrastructure Primitives for Atomic Operations
- `write_atomic_bytes()` / `write_atomic_bytes_async()` for crash-safe file writes with temp file + rename pattern
- `transaction_context()` async context manager with configurable isolation levels, read-only/deferrable options, and per-transaction timeouts
- `retry_on_optimistic_conflict()` decorator/helper with exponential backoff, jitter, and attempt tracking
- Comprehensive test coverage (103 unit tests) for all new utilities

#### OMN-1515: Intent Handler Routing (Demo)
- `HANDLER_TYPE_GRAPH` and `HANDLER_TYPE_INTENT` constants for handler registration
- `HandlerIntent` class wrapping graph operations for intent storage
- Operations: `intent.store`, `intent.query_session`, `intent.query_distribution`
- Auto-routing registration for `HandlerGraph` and `HandlerIntent` in `util_wiring.py`

### Changed

#### Dependencies
- **omnibase-core**: Updated from ^0.9.1 to ^0.9.4 (core release with latest updates)

## [0.2.0] - 2026-01-17

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
            node_name="my_effect_node",
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

#### Dependency Updates (OMN-1361, PR #156)
- **omnibase_core upgraded to 0.7.0**: Breaking changes in core dependency
- **omnibase_spi upgraded to 0.5.0**: Breaking changes in SPI dependency
- **pytest-asyncio 0.25+ compatibility**: Test framework compatibility updates, requires `asyncio_mode = "auto"` in pyproject.toml
- **Infrastructure IP defaults changed to localhost**: Default infrastructure IPs changed from remote server to localhost for local development

#### Error Handling (OMN-1181, PR #158)
- **RuntimeError replaced with structured domain errors**: All generic `RuntimeError` raises have been replaced with specific domain errors from the error taxonomy. If you were catching `RuntimeError`, update to catch the specific error types:
  - `ProtocolConfigurationError` for configuration issues
  - `InfraConnectionError` for connection failures
  - `InfraTimeoutError` for timeout issues
  - `InfraUnavailableError` for unavailable resources

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

#### Documentation
- **Protocol Patterns Documentation** (OMN-1079, PR #166): Added comprehensive documentation for protocol design patterns, cross-mixin composition, and TYPE_CHECKING patterns in `docs/patterns/protocol_patterns.md`

#### Testing
- **Correlation ID Integration Tests** (OMN-1349, PR #160): Added integration tests for correlation ID propagation across service boundaries

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

#### Dependencies
- **omnibase_core**: Updated from 0.6.x to 0.7.0
- **omnibase_spi**: Updated from 0.4.x to 0.5.0
- **pytest-asyncio**: Updated compatibility for 0.25+

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
