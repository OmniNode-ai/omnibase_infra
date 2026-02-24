# Error Handling Best Practices

> **Status**: Current | **Last Updated**: 2026-02-19

Error handling standards for `omnibase_infra`. All infrastructure errors derive from `RuntimeHostError`, which derives from `ModelOnexError` in `omnibase_core`. This document covers the error hierarchy, mandatory `ModelInfraErrorContext` usage, error class selection, and sanitization rules.

---

## Table of Contents

1. [Error Hierarchy](#error-hierarchy)
2. [ModelInfraErrorContext — Mandatory Usage](#modelinfraerrorcontext-mandatory-usage)
3. [Error Class Selection](#error-class-selection)
4. [Error Sanitization Rules](#error-sanitization-rules)
5. [Correlation ID Rules](#correlation-id-rules)
6. [Chaining with `from e`](#chaining-with-from-e)
7. [Common Mistakes](#common-mistakes)

---

## Error Hierarchy

```text
ModelOnexError (omnibase_core)
└── RuntimeHostError (base infrastructure error)
    ├── ProtocolConfigurationError
    ├── ProtocolDependencyResolutionError
    ├── SecretResolutionError
    ├── InfraConnectionError (transport-aware codes)
    │   └── InfraConsulError
    ├── InfraTimeoutError
    ├── InfraAuthenticationError
    ├── InfraRateLimitedError
    ├── InfraRequestRejectedError
    ├── InfraProtocolError
    ├── InfraUnavailableError
    ├── EnvelopeValidationError
    ├── UnknownHandlerTypeError
    ├── PolicyRegistryError
    ├── ComputeRegistryError
    ├── EventBusRegistryError
    ├── ContainerWiringError
    │   ├── ServiceRegistrationError
    │   ├── ServiceResolutionError
    │   └── ContainerValidationError
    ├── ChainPropagationError
    ├── ArchitectureViolationError
    ├── BindingResolutionError
    ├── RepositoryError
    │   ├── RepositoryContractError
    │   ├── RepositoryValidationError
    │   ├── RepositoryExecutionError
    │   └── RepositoryTimeoutError
    ├── DbOwnershipMismatchError
    ├── DbOwnershipMissingError
    ├── SchemaFingerprintMismatchError
    ├── SchemaFingerprintMissingError
    ├── EventRegistryFingerprintMismatchError
    ├── EventRegistryFingerprintMissingError
    └── ContractPublisherError
```

All error classes are importable from `omnibase_infra.errors`.

---

## ModelInfraErrorContext — Mandatory Usage

**Every infrastructure error must include a `ModelInfraErrorContext`.** This is not optional. The context is used for distributed tracing, DLQ routing, and observability dashboards.

### Creating Context

Two factory patterns:

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

# Pattern 1: New error — auto-generate correlation_id
context = ModelInfraErrorContext.with_correlation(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    target_name="omninode-bridge-postgres",
)

# Pattern 2: Propagate existing correlation_id — preserve trace chain
context = ModelInfraErrorContext.with_correlation(
    correlation_id=request.correlation_id,
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    target_name="omninode-bridge-postgres",
)

raise InfraConnectionError("Failed to connect to database", context=context) from e
```

Always prefer Pattern 2 when a `correlation_id` is available from the incoming request or envelope. Use Pattern 1 only when no upstream correlation exists.

### Context Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `transport_type` | `EnumInfraTransportType \| None` | Yes | Transport category (DATABASE, KAFKA, CONSUL, etc.) |
| `operation` | `str \| None` | Yes | Specific operation being attempted |
| `target_name` | `str \| None` | Recommended | Service instance or endpoint identifier |
| `correlation_id` | `UUID \| None` | Yes (auto-generated if absent) | Request correlation ID |
| `namespace` | `str \| None` | When applicable | Service-specific namespace (e.g., Infisical path scope) |

Always provide `transport_type` and `operation`. `target_name` is strongly recommended.

### Transport Types

| Enum Value | Transport | Example target_name |
|-----------|-----------|---------------------|
| `DATABASE` | PostgreSQL | `"omninode-bridge-postgres"` |
| `KAFKA` | Redpanda / Kafka | `"omninode-bridge-redpanda"` |
| `CONSUL` | HashiCorp Consul | `"omninode-bridge-consul"` |
| `HTTP` | REST/HTTP services | `"archon-intelligence:8053"` |
| `QDRANT` | Qdrant vector DB | `"localhost:6333"` |
| `MCP` | MCP tool server | `"mcp-tool-server"` |
| `FILESYSTEM` | Local filesystem | path or descriptor |
| `INMEMORY` | In-memory bus | `"inmemory"` |
| `RUNTIME` | Host process | `"omninode-runtime"` |

---

## Error Class Selection

Choose the most specific applicable error class:

| Scenario | Error Class | Notes |
|----------|-------------|-------|
| Configuration field invalid / missing | `ProtocolConfigurationError` | Raised at startup or config load |
| Secret not found or resolution failed | `SecretResolutionError` | Infisical secrets unavailable |
| Network / TCP connection failed | `InfraConnectionError` | Also use for DNS failures |
| Consul operation failed | `InfraConsulError` | Subclass of `InfraConnectionError` |
| Operation timeout | `InfraTimeoutError` | Set `transport_type` to identify which transport timed out |
| Auth/credentials rejected | `InfraAuthenticationError` | HTTP 401/403, token expired |
| Rate limited (HTTP 429) | `InfraRateLimitedError` | Include `Retry-After` if known |
| Service down / maintenance | `InfraUnavailableError` | Circuit breaker should raise this when open |
| Provider returned 400/422 | `InfraRequestRejectedError` | Bad request to external service |
| Invalid response format | `InfraProtocolError` | Provider response does not match expected schema |
| Envelope fails pre-dispatch validation | `EnvelopeValidationError` | Use before routing, not inside handlers |
| Unknown handler type prefix | `UnknownHandlerTypeError` | Handler loader cannot resolve handler |
| Container wiring failure (general) | `ContainerWiringError` | Use subclass when available |
| Service registration failed | `ServiceRegistrationError` | Subclass of `ContainerWiringError` |
| Service resolution failed | `ServiceResolutionError` | Subclass of `ContainerWiringError` |
| Container validation failed | `ContainerValidationError` | Subclass of `ContainerWiringError` |
| Correlation/causation chain invalid | `ChainPropagationError` | Chain validation failures |
| Architecture invariant violated | `ArchitectureViolationError` | Blocks startup; reserved for critical violations |
| Declarative binding unresolvable | `BindingResolutionError` | Declarative operation binding failures |
| Repository operation (general) | `RepositoryError` | Use subclass when available |
| Bad operation name / missing params | `RepositoryContractError` | Subclass of `RepositoryError` |
| Type mismatch / constraint violation | `RepositoryValidationError` | Subclass of `RepositoryError` |
| asyncpg error / connection issue | `RepositoryExecutionError` | Subclass of `RepositoryError` |
| Query timeout | `RepositoryTimeoutError` | Subclass of `RepositoryError` |
| DB owned by different service | `DbOwnershipMismatchError` | Schema ownership check |
| DB metadata table missing | `DbOwnershipMissingError` | Schema ownership check |
| Schema fingerprint mismatch | `SchemaFingerprintMismatchError` | Live vs expected schema |
| Schema fingerprint missing | `SchemaFingerprintMissingError` | Expected fingerprint not found |
| Contract publisher failure | `ContractPublisherError` | Publishing node contracts |

**Prefer the most specific subclass.** Only use the base class (`RepositoryError`, `ContainerWiringError`) when none of the subclasses fit.

---

## Error Sanitization Rules

Infrastructure errors appear in logs, DLQ messages, and observability systems. They must never expose credentials or PII.

### Never Include

- Passwords or API keys (even partial values)
- Connection strings containing credentials (`postgresql://user:password@host/db`)
- JWT tokens or session tokens
- PII (user IDs that map to real persons, email addresses, IP addresses of end users)
- Raw Vault secret values
- Infisical secret contents

### Safe to Include

- Service names and endpoint hostnames (e.g., `"omninode-bridge-postgres"`)
- Port numbers
- Operation names (e.g., `"execute_query"`, `"publish_message"`)
- Correlation IDs (UUIDs — no PII)
- Error codes
- Table names and SQL operation type (SELECT/INSERT), but not parameter values
- Kafka topic names
- Consul service names and keys (path structure, not values)

### Sanitization Utilities

Use functions from `omnibase_infra.utils.util_error_sanitization`:

```python
from omnibase_infra.utils.util_error_sanitization import (
    sanitize_error_message,   # Strip credentials from arbitrary error messages
    sanitize_secret_path,     # Strip Vault/Infisical path values
    sanitize_consul_key,      # Strip Consul key values
)

# Before sending to DLQ or logs
safe_message = sanitize_error_message(original_exception_message)
```

Apply sanitization at the boundary where errors are serialized (DLQ publisher, structured logger). Do not sanitize inside error constructors — the raw detail is useful for in-process debugging.

---

## Correlation ID Rules

Follow these rules consistently:

1. **Propagate from incoming requests.** Extract `correlation_id` from the incoming envelope or request and pass it to `ModelInfraErrorContext.with_correlation()`.

2. **Auto-generate when missing.** If no upstream `correlation_id` exists, call `ModelInfraErrorContext.with_correlation()` without a `correlation_id` argument — it auto-generates one via `uuid4()`.

3. **Preserve as UUID objects.** Do not convert to string in internal code paths. Only convert at serialization boundaries (JSON output, log fields).

4. **Include in every error context.** There are no exceptions. Every `raise` of an infrastructure error must go through a `ModelInfraErrorContext`.

```python
# Full pattern
from uuid import UUID, uuid4
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

async def connect_to_postgres(correlation_id: UUID | None = None) -> None:
    try:
        await _establish_connection()
    except OSError as e:
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            target_name="omninode-bridge-postgres",
        )
        raise InfraConnectionError(
            "Failed to establish PostgreSQL connection",
            context=context,
        ) from e
```

---

## Chaining with `from e`

Always chain infrastructure errors back to their original cause:

```python
raise InfraConnectionError("...", context=context) from e        # correct
raise InfraTimeoutError("...", context=context) from original    # correct
raise InfraConnectionError("...", context=context)               # wrong — loses traceback
```

Chaining with `from e` preserves the original exception in `__cause__`, making the full traceback visible in logs and debugging tools.

---

## Common Mistakes

| Wrong | Correct | Rule |
|-------|---------|------|
| `raise ValueError("DB error")` | `raise RepositoryExecutionError(...)` | Always use infra error classes |
| `raise InfraConnectionError("Failed")` | `raise InfraConnectionError("Failed", context=context) from e` | Context is mandatory; chain with `from e` |
| `ModelInfraErrorContext(transport_type=...)` | `ModelInfraErrorContext.with_correlation(transport_type=...)` | Use factory to ensure correlation_id |
| Including password in error message | Use `sanitize_error_message()` | Never expose credentials |
| `raise InfraConnectionError(...)` (no `from e`) | `raise InfraConnectionError(...) from e` | Always chain to preserve traceback |
| Using `RepositoryError` when `RepositoryExecutionError` fits | Use `RepositoryExecutionError` | Prefer most specific subclass |

---

## Related Documentation

- `CLAUDE.md` — Error hierarchy and error context sections
- `docs/patterns/error_handling_patterns.md` — Extended examples with context field table
- `docs/patterns/error_recovery_patterns.md` — Recovery strategies (retry, circuit breaker, graceful degradation)
- `docs/patterns/circuit_breaker_implementation.md` — `MixinAsyncCircuitBreaker` usage
- `docs/patterns/security_patterns.md` — Secret sanitization and access control
- `src/omnibase_infra/errors/__init__.py` — Full export list with correlation ID documentation
- `src/omnibase_infra/utils/util_error_sanitization.py` — Sanitization utilities
