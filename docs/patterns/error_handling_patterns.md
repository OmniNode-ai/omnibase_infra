# Error Handling Patterns

## Overview

This document provides comprehensive error handling patterns for ONEX infrastructure services. All error handling follows the transport-aware, context-rich pattern based on `ModelOnexError` hierarchy.

## Error Class Selection

Use the appropriate error class based on the failure scenario:

| Scenario | Error Class | When to Use |
|----------|-------------|-------------|
| Config invalid | `ProtocolConfigurationError` | Configuration validation fails, missing required fields, invalid formats |
| Secret not found | `SecretResolutionError` | Vault secrets unavailable, decryption failures, expired credentials |
| Connection failed | `InfraConnectionError` | Network connectivity issues, service unreachable, DNS resolution failures |
| Timeout | `InfraTimeoutError` | Operation exceeds deadline, slow response times, hung connections |
| Auth failed | `InfraAuthenticationError` | Invalid credentials, expired tokens, permission denied |
| Unavailable | `InfraUnavailableError` | Service down, maintenance mode, rate limiting |

## Error Context Usage

All infrastructure errors must include rich context using `ModelInfraErrorContext`:

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="execute_query",
    target_name="postgresql-primary",
    correlation_id=request.correlation_id,
)
raise InfraConnectionError("Failed to connect to database", context=context) from original_error
```

### Context Fields

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `transport_type` | `EnumInfraTransportType` | Service category | `DATABASE`, `KAFKA`, `VAULT` |
| `operation` | `str` | Specific operation attempted | `"execute_query"`, `"publish_message"` |
| `target_name` | `str` | Service instance identifier | `"postgresql-primary"`, `"kafka-broker-1"` |
| `correlation_id` | `UUID` | Request tracking ID | From incoming request envelope |

## Error Hierarchy

```
ModelOnexError (omnibase_core)
└── RuntimeHostError
    ├── ProtocolConfigurationError
    ├── SecretResolutionError
    ├── InfraConnectionError (transport-aware error codes)
    ├── InfraTimeoutError
    ├── InfraAuthenticationError
    └── InfraUnavailableError
```

## Transport-Aware Error Codes

Error codes are automatically selected based on transport type:

| Transport | EnumCoreErrorCode | Use Cases |
|-----------|-------------------|-----------|
| DATABASE | `DATABASE_CONNECTION_ERROR` | PostgreSQL connection failures, query timeouts |
| HTTP/GRPC | `NETWORK_ERROR` | REST API failures, gRPC stream errors |
| KAFKA/CONSUL/VAULT/VALKEY | `SERVICE_UNAVAILABLE` | Message broker down, service registry unavailable |

### Transport Type Mapping

```python
from omnibase_infra.enums import EnumInfraTransportType

# Database operations
EnumInfraTransportType.DATABASE  # PostgreSQL, MySQL, etc.

# HTTP-based services
EnumInfraTransportType.HTTP      # REST APIs
EnumInfraTransportType.GRPC      # gRPC services

# Infrastructure services
EnumInfraTransportType.KAFKA     # Message broker
EnumInfraTransportType.CONSUL    # Service discovery
EnumInfraTransportType.VAULT     # Secret management
EnumInfraTransportType.VALKEY    # Cache layer
```

## Error Sanitization

### Security Requirements

**NEVER include** in error messages or context:
- Passwords or passphrases
- API keys or tokens
- Connection strings with credentials
- Personally Identifiable Information (PII)
- Internal system paths with sensitive data

**SAFE to include**:
- Service names (sanitized)
- Operation names
- Correlation IDs
- Sanitized hostnames (e.g., `postgres-primary` not `postgres://user:pass@host`)
- Port numbers
- Error types and codes

### Sanitization Example

```python
# ❌ UNSAFE - exposes credentials
raise InfraConnectionError(
    f"Failed to connect to postgresql://admin:secret123@db.internal:5432/mydb"
)

# ✅ SAFE - sanitized target
context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.DATABASE,
    operation="connect",
    target_name="postgresql-primary",  # Sanitized identifier
    correlation_id=correlation_id,
)
raise InfraConnectionError("Failed to connect to database", context=context)
```

## Complete Implementation Examples

### Database Connection Error

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType
from uuid import UUID

async def execute_query(
    query: str,
    correlation_id: UUID
) -> list[dict[str, object]]:
    try:
        conn = await asyncpg.connect(
            host=config.db_host,
            port=config.db_port,
            database=config.db_name,
        )
        return await conn.fetch(query)
    except asyncpg.PostgresConnectionError as e:
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="execute_query",
            target_name="postgresql-primary",
            correlation_id=correlation_id,
        )
        raise InfraConnectionError(
            "Failed to establish database connection",
            context=context
        ) from e
    except asyncpg.QueryCanceledError as e:
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="execute_query",
            target_name="postgresql-primary",
            correlation_id=correlation_id,
        )
        raise InfraTimeoutError(
            "Query execution exceeded timeout",
            context=context
        ) from e
```

### Kafka Publish Error

```python
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType
from kafka.errors import KafkaConnectionError

async def publish_event(
    topic: str,
    message: bytes,
    correlation_id: UUID
) -> None:
    try:
        await producer.send(topic, message)
    except KafkaConnectionError as e:
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="publish_message",
            target_name=f"kafka-topic-{topic}",
            correlation_id=correlation_id,
        )
        raise InfraUnavailableError(
            f"Kafka broker unavailable for topic {topic}",
            context=context
        ) from e
```

### Vault Secret Retrieval Error

```python
from omnibase_infra.errors import SecretResolutionError, InfraAuthenticationError
from omnibase_infra.enums import EnumInfraTransportType
from hvac.exceptions import InvalidPath, Unauthorized

async def get_secret(
    path: str,
    correlation_id: UUID
) -> dict[str, str]:
    try:
        response = vault_client.secrets.kv.v2.read_secret_version(path=path)
        return response["data"]["data"]
    except InvalidPath as e:
        # Secret doesn't exist - don't leak path structure
        raise SecretResolutionError(
            "Secret not found at requested path",
            context=ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.VAULT,
                operation="read_secret",
                target_name="vault-kv-v2",
                correlation_id=correlation_id,
            )
        ) from e
    except Unauthorized as e:
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="read_secret",
            target_name="vault-kv-v2",
            correlation_id=correlation_id,
        )
        raise InfraAuthenticationError(
            "Vault authentication failed",
            context=context
        ) from e
```

## Error Chaining

Always chain errors to preserve stack traces:

```python
# ✅ CORRECT - preserves original error
raise InfraConnectionError("Connection failed", context=context) from original_error

# ❌ WRONG - loses original error
raise InfraConnectionError("Connection failed", context=context)
```

## Related Patterns

- [Error Recovery Patterns](./error_recovery_patterns.md) - Retry logic, circuit breakers
- [Correlation ID Tracking](./correlation_id_tracking.md) - Request tracing across services
- [Container Dependency Injection](./container_dependency_injection.md) - Service resolution
