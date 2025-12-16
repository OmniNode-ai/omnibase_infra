# ONEX Infrastructure Patterns

This directory contains detailed implementation guides and best practices for ONEX infrastructure development.

## Pattern Categories

### Error Handling
- **[Error Handling Patterns](./error_handling_patterns.md)** - Error classification, context, sanitization, and hierarchy
- **[Error Recovery Patterns](./error_recovery_patterns.md)** - Exponential backoff, circuit breakers, graceful degradation, credential refresh
- **[Circuit Breaker Implementation](./circuit_breaker_implementation.md)** - Complete production-ready circuit breaker with state machine

### Observability
- **[Correlation ID Tracking](./correlation_id_tracking.md)** - Request tracing, envelope pattern, distributed logging

### Architecture
- **[Container Dependency Injection](./container_dependency_injection.md)** - Service registration, resolution, and testing patterns

### Security
- **[Policy Registry Trust Model](./policy_registry_trust_model.md)** - Trust assumptions, validation boundaries, and security mitigations for policy registration

## Quick Reference

### Error Scenarios

| Scenario | Pattern Document | Key Classes |
|----------|-----------------|-------------|
| Connection failed | [Error Handling](./error_handling_patterns.md#error-class-selection) | `InfraConnectionError` |
| Service unavailable | [Circuit Breaker](./circuit_breaker_implementation.md) | `CircuitBreaker` |
| Timeout | [Error Recovery](./error_recovery_patterns.md#exponential-backoff-pattern) | `InfraTimeoutError` |
| Auth failed | [Error Recovery](./error_recovery_patterns.md#credential-refresh-pattern) | `InfraAuthenticationError` |
| Secret not found | [Error Handling](./error_handling_patterns.md#vault-secret-retrieval-error) | `SecretResolutionError` |

### Common Tasks

| Task | Pattern Document | Section |
|------|-----------------|---------|
| Add retry logic | [Error Recovery](./error_recovery_patterns.md) | Exponential Backoff |
| Prevent cascading failures | [Circuit Breaker](./circuit_breaker_implementation.md) | Complete Implementation |
| Track requests across services | [Correlation ID](./correlation_id_tracking.md) | Correlation ID Flow |
| Inject dependencies | [Container DI](./container_dependency_injection.md) | Constructor Injection |
| Handle cache fallback | [Error Recovery](./error_recovery_patterns.md) | Graceful Degradation |
| Refresh expired tokens | [Error Recovery](./error_recovery_patterns.md) | Credential Refresh |
| Understand policy security | [Policy Registry Trust Model](./policy_registry_trust_model.md) | Trust Assumptions |
| Implement policy allowlist | [Policy Registry Trust Model](./policy_registry_trust_model.md) | Security Mitigations |

### Transport Types

| Transport | Error Code | Pattern Documents |
|-----------|-----------|------------------|
| DATABASE | `DATABASE_CONNECTION_ERROR` | [Error Handling](./error_handling_patterns.md#transport-aware-error-codes) |
| HTTP/GRPC | `NETWORK_ERROR` | [Error Handling](./error_handling_patterns.md#transport-aware-error-codes) |
| KAFKA | `SERVICE_UNAVAILABLE` | [Circuit Breaker](./circuit_breaker_implementation.md#kafka-publisher-protection) |
| CONSUL | `SERVICE_UNAVAILABLE` | [Error Handling](./error_handling_patterns.md#transport-type-mapping) |
| VAULT | `SERVICE_UNAVAILABLE` | [Error Recovery](./error_recovery_patterns.md#credential-refresh-pattern) |
| REDIS | `SERVICE_UNAVAILABLE` | [Error Recovery](./error_recovery_patterns.md#graceful-degradation-pattern) |

## Pattern Relationships

```
Error Handling Patterns
    ├── Defines error classes and context
    ├── Used by: All other patterns
    └── References: Transport types, correlation IDs

Error Recovery Patterns
    ├── Implements resilience strategies
    ├── Depends on: Error Handling Patterns
    └── References: Circuit Breaker, Correlation ID

Circuit Breaker Implementation
    ├── Detailed state machine implementation
    ├── Depends on: Error Handling, Error Recovery
    └── References: Correlation ID, Metrics

Correlation ID Tracking
    ├── Request tracing infrastructure
    ├── Used by: All patterns
    └── References: Error context, logging

Container Dependency Injection
    ├── Service management and resolution
    ├── Used by: All infrastructure services
    └── References: Bootstrap, testing patterns

Policy Registry Trust Model
    ├── Documents security boundaries for policy registration
    ├── Depends on: Container DI (for registry resolution)
    └── References: PolicyRegistry, ProtocolPolicy
```

## Usage Examples

### Complete Error Handling Flow

```python
from omnibase_core.container import ModelONEXContainer
from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType
from uuid import uuid4

# 1. Bootstrap container (Container DI pattern)
container = ModelONEXContainer()
wire_infrastructure_services(container)

# 2. Generate correlation ID (Correlation ID pattern)
correlation_id = uuid4()

# 3. Execute with retry and circuit breaker (Error Recovery patterns)
async def resilient_query(query: str):
    # Circuit breaker protects against cascading failures
    async def _query():
        # Exponential backoff for transient failures
        return await retry_with_exponential_backoff(
            operation=lambda: execute_query(query, correlation_id),
            correlation_id=correlation_id,
        )

    return await db_circuit_breaker.call(_query)

# 4. Handle errors with proper context (Error Handling pattern)
try:
    result = await resilient_query("SELECT * FROM users")
except InfraConnectionError as e:
    # Error includes correlation ID, transport type, operation
    logger.error("Query failed", extra={"correlation_id": e.context.correlation_id})
```

## Contributing Patterns

When adding new patterns:

1. Create focused document for single pattern/concept
2. Include complete code examples
3. Provide usage examples with context
4. Cross-reference related patterns
5. Update this README with pattern summary
6. Add to Quick Reference tables

## Document Standards

Each pattern document should include:

- **Overview** - Purpose and scope
- **Implementation** - Complete, production-ready code
- **Usage Examples** - Real-world scenarios
- **Best Practices** - DO/DON'T guidelines
- **Related Patterns** - Cross-references

## See Also

- [CLAUDE.md](../../CLAUDE.md) - Quick reference rules (references these patterns)
- [Infrastructure Migration Plan](../../CLAUDE.md#infrastructure-migration-plan) - Migration roadmap
- [ONEX Principles](../../CLAUDE.md#core-onex-principles) - Architectural principles
