# Changelog

All notable changes to the ONEX Infrastructure (omnibase_infra) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking Changes

#### MixinNodeIntrospection API (PR #54)
- **New configuration model**: Added `ModelIntrospectionConfig` as the preferred configuration method.
  - **Migration**: Use `initialize_introspection_from_config(config)` for new code; legacy `initialize_introspection()` method remains supported for backward compatibility.
  - **Rationale**: Typed configuration model provides better validation and extensibility.

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
  - `node_id` (required): Unique identifier for this node instance (UUID, strings auto-converted)
  - `node_type` (required): Type of node (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
  - `event_bus`: Optional event bus for publishing introspection and heartbeat events (must have `publish_envelope()` method if provided)
  - `version`: Node version string (default: `"1.0.0"`)
  - `cache_ttl`: Cache time-to-live in seconds (default: `300.0`)
  - `operation_keywords`: Optional set of keywords to identify operation methods (if None, uses DEFAULT_OPERATION_KEYWORDS)
  - `exclude_prefixes`: Optional set of prefixes to exclude from capability discovery (if None, uses DEFAULT_EXCLUDE_PREFIXES)
- **Performance Metrics Tracking**: Added `IntrospectionPerformanceMetrics` dataclass and `get_performance_metrics()` method for monitoring introspection operation timing and threshold violations

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
- **Enum Stability**: `EnumDispatchStatus.HANDLER_ERROR` value remains **unchanged** for backwards compatibility with existing metrics, logs, and monitoring systems
- **Full Migration Guide**: See `docs/migrations/HANDLER_TO_DISPATCHER_MIGRATION.md` for complete migration details and code examples

#### CI/CD
- **Pre-commit Configuration**: Migrated to fix deprecated stage warnings (PR #25)

## [0.1.0] - Unreleased

### Planned

This version represents the MVP (Minimum Viable Product) for ONEX Runtime Host Infrastructure.

#### Core Components (MVP)
- **BaseRuntimeHostProcess** (OMN-249): Infrastructure wrapper owning event bus and driving NodeRuntime
- **DbHandler** (OMN-238): PostgreSQL database protocol handler
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
- VaultAdapter and ConsulAdapter
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
