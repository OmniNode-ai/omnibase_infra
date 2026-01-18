> **Navigation**: [Home](../index.md) > [Milestones](../MVP_PLAN.md) > Beta Hardening v0.2.0
>
> **Related**: [Previous: MVP Core](./MVP_v0.1.0_CORE.md) | [Next: Production](./PRODUCTION_v0.3.0.md)

# Beta Hardening (v0.2.0) - Milestone Details

**Repository**: omnibase_infra
**Target Version**: v0.2.0
**Timeline**: Sprint 3-4
**Issue Count**: 22
**Prerequisite**: MVP Core (v0.1.0) must be complete

---

## Beta Philosophy

**Beta (v0.2.0)**: Harden for production use
- KafkaEventBus with backpressure
- Vault + Consul handlers
- Full retry/rate-limit policies
- Integration tests with real services
- Observability layer

This milestone builds upon the MVP foundation to add production-readiness features including:
- External service integration (Kafka, Vault, Consul)
- Resilience patterns (retries, circuit breakers, rate limiting)
- Full observability (structured logging, metrics)
- Comprehensive integration testing

---

## Beta Scope Boundaries (Do Not Implement in Beta)

The following are explicitly OUT OF SCOPE until Production (v0.3.0):

- **No multi-region failover** - Single cluster only
- **No automatic topic creation** - Manual infra bootstrap
- **No dynamic handler discovery** - Static wiring only

---

## Beta Contract Additions

Building on MVP simplified contracts, Beta adds full configuration:

```yaml
runtime:
  name: "my_runtime_host"
  version: "1.0.0"

  event_bus:
    kind: "kafka"
    config:
      bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
      consumer_group: "runtime-host-group"
      topics:
        input: "onex.tenant.domain.cmd.v1"
        output: "onex.tenant.domain.evt.v1"

  handlers:
    - type: "http"
      config:
        timeout_ms: 30000
        retry_policy:
          max_retries: 3
          backoff_strategy: "exponential"
    - type: "db"
      config:
        pool_size: 10
    - type: "vault"
      config:
        address: "${VAULT_ADDR}"
    - type: "consul"
      config:
        address: "${CONSUL_ADDR}"

  nodes:
    - slug: "node_a"
    - slug: "node_b"
```

**Cross-Version Compatibility**:
- v0.2 infra MUST accept v0.1 contracts but MUST warn on deprecated fields
- v0.1 contracts MUST NOT use any Beta-only fields
- Contract version mismatch MUST produce clear error message with upgrade instructions

---

## Topic Naming Schema

Topics follow the pattern: `onex.<tenant>.<domain>.<direction>.v<version>`

**Required Directions**:
- `.cmd` - Command/input topic (requests INTO the runtime)
- `.evt` - Event/output topic (responses FROM the runtime)

**MVP Validation Rules** (enforced by contract importer):
- No invalid characters (alphanumeric, dots, hyphens only)
- Version segment must be numeric (e.g., `v1`, `v2`)
- Tenant and domain segments required

**Examples**:
- `onex.acme.orders.cmd.v1` - Input commands
- `onex.acme.orders.evt.v1` - Output events

**Reserved Namespaces** (MUST NOT be used by user contracts):
- `onex.internal.*` - System-internal communication
- `onex.debug.*` - Debug and diagnostics
- `onex.metrics.*` - Metrics and telemetry
- `onex.health.*` - Health check signals
- `onex.admin.*` - Administrative commands

**Allowed Characters in Segments**: `[a-z0-9_-]` (lowercase alphanumeric, underscore, hyphen)

**Topic Naming Examples**:

| Example | Valid? | Reason |
|---------|--------|--------|
| `onex.acme.orders.cmd.v1` | Valid | Correct format |
| `onex.acme.orders.evt.v1` | Valid | Correct format |
| `onex.acme.user-service.cmd.v1` | Valid | Hyphen allowed |
| `onex.acme.user_service.cmd.v1` | Valid | Underscore allowed |
| `onex.ACME.orders.cmd.v1` | Invalid | Uppercase not allowed |
| `onex.acme.orders.command.v1` | Invalid | Must be `cmd` or `evt` |
| `onex.acme.orders.cmd.1` | Invalid | Version must have `v` prefix |
| `onex.acme.orders.cmd` | Invalid | Missing version |
| `acme.orders.cmd.v1` | Invalid | Missing `onex.` prefix |
| `onex..orders.cmd.v1` | Invalid | Empty segment |
| `onex.acme.orders.query.v1` | Invalid | `query` not a valid direction |
| `onex.internal.debug.evt.v1` | Invalid | `internal` is reserved |

**Validation Regex**:
```python
TOPIC_PATTERN = re.compile(
    r"^onex\.[a-z0-9][a-z0-9_-]*\.[a-z0-9][a-z0-9_-]*\.(cmd|evt)\.v[0-9]+$"
)

def validate_topic(topic: str) -> bool:
    if topic.startswith("onex.internal.") or topic.startswith("onex.debug."):
        raise ValueError(f"Reserved namespace: {topic}")
    return bool(TOPIC_PATTERN.match(topic))
```

---

## Phase 0: Cross-Repo Guardrails & Invariants (Beta Issues)

---

### Issue 0.3: Protocol ownership verification [BETA]

**Title**: Verify infra never declares new protocols
**Type**: Infrastructure
**Priority**: High
**Labels**: `architecture`, `ci`, `guardrails`
**Milestone**: v0.2.0 Beta

**Description**:
Ensure `omnibase_infra` only IMPLEMENTS protocols defined in `omnibase_spi`, never declares new abstract protocols.

**Acceptance Criteria**:
- [ ] AST or grep check for `class.*Protocol.*ABC` in infra
- [ ] Only allow concrete implementations of SPI protocols
- [ ] Clear error message on violation
- [ ] Runs in CI

---

### Issue 0.4: Version compatibility matrix check [BETA]

**Title**: Runtime version compatibility verification
**Type**: Infrastructure
**Priority**: High
**Labels**: `ci`, `versioning`
**Milestone**: v0.2.0 Beta

**Description**:
On startup, verify that `omnibase_core` and `omnibase_spi` versions meet minimum requirements.

**Compatibility Matrix**:
- Infra v0.1.0 -> Core >=0.4.0, SPI >=0.3.0

**Acceptance Criteria**:
- [ ] `BaseRuntimeHostProcess` logs resolved versions on startup
- [ ] Fails fast with clear message if incompatible
- [ ] Version matrix documented in README
- [ ] CI test for version check logic

---

## Phase 1: Core Types (omnibase_core) - Beta Issues

---

### Issue 1.12: Create ModelHandlerBindingConfig model [BETA]

**Title**: Implement formalized handler config schema
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `model`, `core`
**Milestone**: v0.2.0 Beta

**Description**:
Create a formal schema for handler binding configuration with validation and defaults.

**File**: `src/omnibase_core/models/runtime/model_handler_binding_config.py` (NEW)

**Fields**:
- `handler_type: EnumHandlerType`
- `name: str` (optional, defaults to handler_type)
- `enabled: bool = True`
- `priority: int = 0`
- `config_ref: str | None` (reference to external config)
- `retry_policy: ModelRetryPolicy | None`
- `timeout_ms: int = 30000`
- `rate_limit_per_second: float | None`

**Sub-model** `ModelRetryPolicy`:
- `max_retries: int = 3`
- `backoff_strategy: Literal["fixed", "exponential"] = "exponential"`
- `base_delay_ms: int = 100`
- `max_delay_ms: int = 5000`

**Acceptance Criteria**:
- [ ] Full validation with Pydantic
- [ ] Sensible defaults for all optional fields
- [ ] Used by `ModelRuntimeHostContract.handlers`
- [ ] Unit tests for validation edge cases
- [ ] mypy --strict passes

---

### Issue 1.13: Extend error taxonomy (core) [BETA]

**Title**: Complete core error hierarchy for runtime
**Type**: Feature
**Priority**: High
**Labels**: `architecture`, `errors`, `core`
**Milestone**: v0.2.0 Beta

**Description**:
Complete the error hierarchy with additional error types.

**File**: `src/omnibase_core/errors/runtime_errors.py` (UPDATE)

**Additional Classes**:
- `HandlerNotFoundError(RuntimeHostError)`
- `NodeNotFoundError(RuntimeHostError)`
- `EnvelopeValidationError(RuntimeHostError)`
- `ProtocolConfigurationError(RuntimeHostError)`
- `SecretResolutionError(RuntimeHostError)`

**Acceptance Criteria**:
- [ ] All inherit from `RuntimeHostError`
- [ ] Structured fields: `handler_type`, `operation`, `correlation_id`
- [ ] NodeRuntime only sees abstract errors, not library exceptions
- [ ] Unit tests for each error class
- [ ] mypy --strict passes

---

## Phase 3: Infrastructure Handlers (omnibase_infra) - Beta Issues

---

### Issue 3.9: Implement HandlerVault [BETA]

**Title**: Create Vault secrets management protocol adapter
**Type**: Feature
**Priority**: High
**Labels**: `handler`, `infrastructure`, `secrets`
**Milestone**: v0.2.0 Beta

**Description**:
Implement `HandlerVault` for HashiCorp Vault operations using hvac.

**File**: `src/omnibase_infra/handlers/handler_vault.py`

**Operations**: `get_secret`, `set_secret`, `delete_secret`, `list_secrets`

**Acceptance Criteria**:
- [ ] Returns `EnumHandlerType.VAULT` (not str)
- [ ] Uses hvac client
- [ ] KV v2 support
- [ ] Maps `hvac.VaultError` -> `HandlerExecutionError`
- [ ] Unit tests with mock client
- [ ] Integration test with dev Vault
- [ ] mypy --strict passes

---

### Issue 3.10: Implement HandlerConsul [BETA]

**Title**: Create Consul service discovery protocol handler
**Type**: Feature
**Priority**: Medium
**Labels**: `handler`, `infrastructure`, `discovery`
**Milestone**: v0.2.0 Beta

**Description**:
Implement `HandlerConsul` for Consul service discovery operations.

**File**: `src/omnibase_infra/handlers/handler_consul.py`

**Operations**: `register_service`, `deregister_service`, `get_service`, `list_services`, `health_check_service`

**Acceptance Criteria**:
- [ ] Returns `EnumHandlerType.CONSUL` (not str)
- [ ] Uses python-consul or httpx
- [ ] Service registration/deregistration
- [ ] Health check integration
- [ ] Unit tests with mock responses
- [ ] Integration test with dev Consul
- [ ] mypy --strict passes

---

### Issue 3.11: Implement KafkaEventBus [BETA]

**Title**: Create Kafka event bus implementation
**Type**: Feature
**Priority**: High
**Labels**: `event-bus`, `infrastructure`, `messaging`
**Milestone**: v0.2.0 Beta

**Description**:
Implement `KafkaEventBus` that implements `ProtocolEventBus` (NOT ProtocolHandler).

**File**: `src/omnibase_infra/event_bus/kafka_event_bus.py`

**Methods**: `initialize`, `shutdown`, `publish_envelope`, `subscribe`, `start_consuming`, `health_check`

**Backpressure Config**:
- `max_inflight_envelopes: int = 100`
- `pause_consumption_threshold: int = 80`
- `circuit_breaker_threshold: int = 5` (consecutive failures)

**Ordering Guarantees**: Runtime MUST NOT assume global ordering. Kafka provides partition-level ordering only.

**Response Pattern**: Event bus handlers must return envelopes; BaseRuntimeHostProcess is responsible for publishing responses.

**Acceptance Criteria**:
- [ ] Implements `ProtocolEventBus` (NOT ProtocolHandler)
- [ ] NO `handler_type` property (event bus, not handler)
- [ ] Uses aiokafka
- [ ] Proper envelope serialization/deserialization
- [ ] Configurable topics and consumer group
- [ ] Backpressure: pauses consumption when queue full
- [ ] Circuit breaker: stops after N consecutive handler failures
- [ ] Unit tests with mock Kafka
- [ ] Integration test with test Kafka
- [ ] mypy --strict passes

---

### Issue 3.12: Create SecretResolver [BETA]

**Title**: Implement centralized secret resolution
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `secrets`
**Milestone**: v0.2.0 Beta

**Description**:
Create a centralized secret resolver so handlers never call `os.getenv` directly.

**File**: `src/omnibase_infra/runtime/secret_resolver.py`

**Interface**:
```python
class SecretResolver:
    async def get_secret(self, logical_name: str) -> str: ...
    async def get_secrets(self, logical_names: list[str]) -> dict[str, str]: ...
```

**Sources** (priority order):
1. Vault (if configured)
2. Environment variables
3. File-based secrets (K8s secrets volume)

**Acceptance Criteria**:
- [ ] Typed interface for secret requests
- [ ] Handlers never call `os.getenv` directly
- [ ] Vault integration optional
- [ ] Unit tests with mocked sources
- [ ] mypy --strict passes

---

### Issue 3.13: Create HandlerConfigResolver [BETA]

**Title**: Implement handler config resolution layer
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `configuration`
**Milestone**: v0.2.0 Beta

**Description**:
Create a resolver that normalizes handler configs from multiple sources.

**File**: `src/omnibase_infra/runtime/handler_config_resolver.py`

**Sources**:
- Contract YAML
- Environment variables (override)
- Vault (secrets only via SecretResolver)
- Files (config refs)

**Acceptance Criteria**:
- [ ] Resolves `config_ref` to actual config dict
- [ ] Merges environment overrides
- [ ] Returns fully validated `ModelHandlerBindingConfig`
- [ ] Clear error messages for missing required fields
- [ ] Unit tests for resolution logic
- [ ] mypy --strict passes

---

### Issue 3.14: Extend infra error hierarchy [BETA]

**Title**: Complete infrastructure error taxonomy
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `errors`
**Milestone**: v0.2.0 Beta

**Description**:
Complete structured error hierarchy for all infra components.

**File**: `src/omnibase_infra/errors/infra_errors.py` (UPDATE)

**Additional Classes**:
- `ProtocolConfigurationError(RuntimeHostError)`
- `SecretResolutionError(RuntimeHostError)`

**Acceptance Criteria**:
- [ ] All inherit from core `RuntimeHostError`
- [ ] Handlers map raw exceptions to these types
- [ ] Structured fields: `handler_type`, `operation`, `correlation_id`
- [ ] Unit tests for each error class
- [ ] mypy --strict passes

---

### Issue 3.15: Implement observability layer [BETA]

**Title**: Create structured logging and metrics infrastructure
**Type**: Feature
**Priority**: High
**Labels**: `infrastructure`, `observability`
**Milestone**: v0.2.0 Beta

**Description**:
Create centralized observability configuration owned by BaseRuntimeHostProcess.

**Files**:
- `src/omnibase_infra/observability/logging_config.py`
- `src/omnibase_infra/observability/metrics.py`

**Logging Requirements**:
- JSON structured logs
- Standard fields: `runtime_id`, `node_id`, `handler_type`, `envelope_id`, `correlation_id`
- Handlers use shared logger (no ad-hoc loggers)

**Metrics Hooks**:
- Envelope count in/out per handler
- Handler latency histogram
- Event bus lag metrics (consumer group offset)

**Acceptance Criteria**:
- [ ] `BaseRuntimeHostProcess` sets up global logging config
- [ ] Handlers get loggers from centralized factory
- [ ] All logs include correlation_id when available
- [ ] Metrics exposed for collection
- [ ] Unit tests for log formatting
- [ ] mypy --strict passes

---

### Issue 3.16: Implement health HTTP endpoint [BETA]

**Title**: Create HTTP health endpoint server
**Type**: Feature
**Priority**: Medium
**Labels**: `infrastructure`, `health`
**Milestone**: v0.2.0 Beta

**Description**:
Optional HTTP server exposing health endpoints for K8s probes.

**File**: `src/omnibase_infra/runtime/health_server.py`

**Endpoints**:
- `/health/live` - Is process alive
- `/health/ready` - Can process envelopes
- `/health/handlers` - Per-handler status snapshot

**Acceptance Criteria**:
- [ ] Optional (can be disabled in config)
- [ ] Configurable port
- [ ] Minimal dependencies (use `aiohttp` or built-in)
- [ ] Integrates with Docker healthchecks
- [ ] Integrates with K8s probes
- [ ] Unit tests
- [ ] mypy --strict passes

---

### Issue 3.17: Implement handler retry wrapper [BETA]

**Title**: Create generic retry/rate-limit wrapper for handlers
**Type**: Feature
**Priority**: Medium
**Labels**: `infrastructure`, `handlers`
**Milestone**: v0.2.0 Beta

**Description**:
Create a decorator/wrapper that adds retry and rate limiting to any handler.

**File**: `src/omnibase_infra/handlers/handler_wrapper.py`

**Features**:
- Retry with configurable backoff (from `ModelRetryPolicy`)
- Token bucket rate limiting
- Circuit breaker pattern

**Acceptance Criteria**:
- [ ] Reads config from `ModelHandlerBindingConfig`
- [ ] Wraps `handler.execute()` transparently
- [ ] Logs retry attempts
- [ ] Rate limit enforced per handler instance
- [ ] Circuit breaker trips after threshold
- [ ] Unit tests for retry logic
- [ ] mypy --strict passes

---

### Issue 3.18: Add contract schema and linting [BETA]

**Title**: Create JSON Schema for runtime host contracts
**Type**: Feature
**Priority**: Medium
**Labels**: `infrastructure`, `validation`
**Milestone**: v0.2.0 Beta

**Description**:
Create validation tooling for runtime host contracts.

**Files**:
- `src/omnibase_infra/contracts/schema/runtime_host_contract.schema.json`
- `src/omnibase_infra/cli/validate_contract.py`

**Lint Rules**:
- No embedded secrets (require `secret_ref`)
- No `LOCAL` handler in production contracts
- Topic names follow naming schema

**CLI**: `omnibase-runtime-validate-contract PATH`

**Acceptance Criteria**:
- [ ] JSON Schema created
- [ ] CLI validates contracts
- [ ] Lint rules enforced
- [ ] CI integration
- [ ] Unit tests for validation
- [ ] mypy --strict passes

---

## Phase 4: Integration & Testing - Beta Issues

---

### Issue 4.5: Unit tests for KafkaEventBus [BETA]

**Title**: Unit tests for KafkaEventBus
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `event-bus`
**Milestone**: v0.2.0 Beta

**Description**:
Unit tests for KafkaEventBus with mocked Kafka.

**File**: `tests/unit/event_bus/test_kafka_event_bus.py`

**Mock Requirements**: MockKafka MUST match aiokafka interface exactly. This prevents Beta teams from mocking the wrong subset of APIs.

**Acceptance Criteria**:
- [ ] Test envelope publishing
- [ ] Test subscription and consumption
- [ ] Test backpressure behavior
- [ ] Test circuit breaker
- [ ] Test error handling
- [ ] Test health check
- [ ] >90% coverage
- [ ] MockKafka implements complete aiokafka interface
- [ ] No aiokafka methods missing from mock

---

### Issue 4.6: Integration tests with Docker [BETA]

**Title**: Integration tests with real services
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `integration`, `docker`
**Milestone**: v0.2.0 Beta

**Description**:
Integration tests using docker-compose with real PostgreSQL, Kafka, Vault.

**File**: `tests/integration/test_runtime_host_integration.py`

**Tests**:
- HandlerDb with real PostgreSQL
- HandlerVault with real Vault
- KafkaEventBus with real Kafka
- Full envelope flow

**Test Performance Requirements**:
- Integration tests must complete within 3 seconds per test case.
- Tests MUST NOT contain sleeps >50ms (use proper async waiting).
- Flaky tests are not acceptable - deterministic behavior required.

**Acceptance Criteria**:
- [ ] docker-compose.test.yaml with all services
- [ ] pytest fixtures for service setup
- [ ] Tests pass in CI
- [ ] Cleanup after tests
- [ ] Each test completes in <3 seconds
- [ ] No sleeps >50ms in test code
- [ ] Tests pass consistently (0 flaky tests)

---

### Issue 4.7: Graceful shutdown tests [BETA]

**Title**: Test graceful shutdown behavior
**Type**: Testing
**Priority**: High
**Labels**: `testing`, `shutdown`
**Milestone**: v0.2.0 Beta

**Description**:
Verify shutdown semantics under load.

**File**: `tests/integration/test_graceful_shutdown.py`

**Scenarios**:
- SIGTERM while 50 envelopes in progress
- Verify no envelopes dropped
- Verify handlers' `shutdown()` called exactly once
- Verify exit within grace period

**Acceptance Criteria**:
- [ ] Simulate SIGTERM under load
- [ ] Assert no silent envelope drops
- [ ] Assert all handlers shutdown
- [ ] Assert exit within configured timeout

---

### Issue 4.8: Backpressure and overload tests [BETA]

**Title**: Test backpressure under overload
**Type**: Testing
**Priority**: Medium
**Labels**: `testing`, `performance`
**Milestone**: v0.2.0 Beta

**Description**:
Verify backpressure behavior when Kafka produces faster than handlers consume.

**File**: `tests/integration/test_backpressure.py`

**Acceptance Criteria**:
- [ ] Simulate high-volume Kafka production
- [ ] Verify event bus pauses consumption
- [ ] Verify no unbounded memory growth
- [ ] Verify recovery when backlog clears

---

### Issue 4.9: Topic naming validation tests [BETA]

**Title**: Validate Kafka topic naming schema
**Type**: Testing
**Priority**: Medium
**Labels**: `testing`, `kafka`
**Milestone**: v0.2.0 Beta

**Description**:
Verify all configured topics follow naming schema.

**Schema**: `onex.<tenant>.<domain>.<signal>.v<version>`
**Signals**: `cmd`, `evt`, `state`, `error`, `log`

**Acceptance Criteria**:
- [ ] Parse topics from all runtime contracts
- [ ] Validate tenant segment exists
- [ ] Validate signal is in allowed set
- [ ] CI integration

---

## Phase 5: Deployment & Migration - Beta Issues

---

### Issue 5.4: Expand docker-compose with full services [BETA]

**Title**: Add Kafka, Vault, Consul to docker-compose
**Type**: DevOps
**Priority**: Medium
**Labels**: `deployment`, `docker`, `development`
**Milestone**: v0.2.0 Beta

**Description**:
Expand docker-compose.yaml with all dependent services.

**Additional Services**:
- kafka (redpanda)
- vault (dev mode)
- consul (dev mode)

**Acceptance Criteria**:
- [ ] All services defined
- [ ] Health checks configured
- [ ] Volumes for persistence
- [ ] Service dependencies correct

---

### Issue 5.5: Document CLIs and environments [BETA]

**Title**: Create CLIs and Environments documentation
**Type**: Documentation
**Priority**: Medium
**Labels**: `documentation`
**Milestone**: v0.2.0 Beta

**Description**:
Document the separation between dev and prod CLIs.

**File**: `docs/CLI_ENVIRONMENTS.md`

**Content**:
- `omninode-runtime-host-dev` (core, dev only, LocalHandler)
- `omnibase-runtime-host` (infra, production, no LocalHandler)
- Environment variables
- When to use each

**Acceptance Criteria**:
- [ ] Clear distinction documented
- [ ] Examples for each scenario
- [ ] Warning about prod CLI restrictions

---

### Issue 5.6: Create version compatibility documentation [BETA]

**Title**: Document version compatibility matrix
**Type**: Documentation
**Priority**: Low
**Labels**: `documentation`, `versioning`
**Milestone**: v0.2.0 Beta

**Description**:
Document which versions of core/spi work with which infra versions.

**File**: `docs/VERSION_COMPATIBILITY.md`

**Acceptance Criteria**:
- [ ] Compatibility matrix table
- [ ] Minimum version requirements
- [ ] How to check versions at runtime
- [ ] Upgrade path documentation

---

## Beta Execution Order

```
Phase 0 (CI Guardrails - Beta)
    |
    +-- 0.3 Protocol ownership verification [BETA]
    +-- 0.4 Version compatibility check [BETA]
    |
    v
Phase 1 (Core Types - Beta)
    |
    +-- 1.12 ModelHandlerBindingConfig [BETA]
    +-- 1.13 Extended error taxonomy [BETA]
    |
    v
Phase 3 (Infra - Beta)
    |
    +-- 3.9 HandlerVault [BETA]
    +-- 3.10 HandlerConsul [BETA]
    +-- 3.11 KafkaEventBus [BETA]
    +-- 3.12 SecretResolver [BETA]
    +-- 3.13 HandlerConfigResolver [BETA]
    +-- 3.14 Extended error hierarchy [BETA]
    +-- 3.15 Observability layer [BETA]
    +-- 3.16 Health HTTP endpoint [BETA]
    +-- 3.17 Handler retry wrapper [BETA]
    +-- 3.18 Contract schema/linting [BETA]
    |
    v
Phase 4 (Testing - Beta)
    |
    +-- 4.5 KafkaEventBus unit tests [BETA]
    +-- 4.6 Integration tests with Docker [BETA]
    +-- 4.7 Graceful shutdown tests [BETA]
    +-- 4.8 Backpressure tests [BETA]
    +-- 4.9 Topic naming validation [BETA]
    |
    v
Phase 5 (Deployment - Beta)
    |
    +-- 5.4 Expand docker-compose [BETA]
    +-- 5.5 CLI environments doc [BETA]
    +-- 5.6 Version compatibility doc [BETA]
    |
    v
v0.2.0 Release
```

---

## Beta Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | >90% | pytest-cov |
| Integration tests pass | Yes | CI |
| Graceful shutdown drain | 100% | No dropped envelopes |
| Circuit breaker trigger | <10s | After threshold failures |
| Architecture violations | 0 | CI checks |

---

## Considerations for Future Work

The following items should be considered during Beta development but may not be critical blockers:

### Consider: Adding integration tests for error classes
- Validate error classes work correctly in real handler scenarios
- Test error propagation through KafkaEventBus
- Verify error serialization/deserialization in envelope flows

### Consider: Creating error recovery examples in documentation
- Document best practices for handling `HandlerExecutionError`
- Provide examples of retry configuration for different error types
- Show how to implement custom error handling in handlers

---

> **Navigation**: [Back to Overview](../MVP_PLAN.md) | [Previous: MVP Core](./MVP_v0.1.0_CORE.md) | [Next: Production](./PRODUCTION_v0.3.0.md)

**Last Updated**: 2025-12-03
