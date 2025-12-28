# Container Wiring Error Handling

This document describes the specific error handling implementation for container wiring operations in `omnibase_infra`.

## Overview

Enhanced error handling has been added to provide clear, actionable feedback when container wiring operations fail. This addresses PR #36 review feedback for better error handling in container wiring.

## New Error Classes

All error classes are located in `src/omnibase_infra/errors/error_container_wiring.py`

### Error Hierarchy

```
ModelOnexError (omnibase_core)
└── RuntimeHostError (omnibase_infra)
    └── ContainerWiringError (BASE)
        ├── ServiceRegistrationError
        ├── ServiceResolutionError
        └── ContainerValidationError
```

### ContainerWiringError

Base error class for all container wiring operations.

**Error Code**: `ONEX_CORE_081_OPERATION_FAILED`

**Usage**:
```python
from omnibase_infra.errors import ContainerWiringError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.RUNTIME,
    operation="wire_services",
    correlation_id=uuid4(),
)

raise ContainerWiringError(
    "Container wiring failed",
    context=context,
)
```

### ServiceRegistrationError

Raised when service registration fails during container wiring.

**Use Cases**:
- Duplicate service registration attempts
- Invalid registration arguments
- Missing container attributes/methods
- Service instantiation failures

**Example**:
```python
from omnibase_infra.errors import ServiceRegistrationError

raise ServiceRegistrationError(
    "Failed to register PolicyRegistry",
    service_name="PolicyRegistry",
    context=context,
    original_error=str(e),
    hint="Ensure container.service_registry has register_instance() method",
) from e
```

### ServiceResolutionError

Raised when service resolution fails from the container.

**Use Cases**:
- Service not registered in container
- Invalid service type requested
- Container API incompatibility
- Missing container attributes

**Example**:
```python
from omnibase_infra.errors import ServiceResolutionError

raise ServiceResolutionError(
    "PolicyRegistry not registered in container. "
    "Call wire_infrastructure_services(container) first.",
    service_name="PolicyRegistry",
    context=context,
    original_error=str(e),
) from e
```

### ContainerValidationError

Raised when container validation fails before wiring operations.

**Use Cases**:
- Container missing `service_registry` attribute
- Service registry missing required methods
- Invalid container instance
- API version incompatibility

**Example**:
```python
from omnibase_infra.errors import ContainerValidationError

raise ContainerValidationError(
    "Container missing required service_registry attribute. "
    "Ensure container is a valid ModelONEXContainer instance.",
    context=context,
    missing_attribute="service_registry",
    container_type=type(container).__name__,
)
```

## Error Context

All container wiring errors support structured context via `ModelInfraErrorContext`:

```python
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType
from uuid import uuid4

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.RUNTIME,  # Always RUNTIME for container ops
    operation="register_policy_registry",           # Operation being performed
    target_name="PolicyRegistry",                   # Service name
    correlation_id=uuid4(),                         # Request tracing ID
)
```

## Error Handling Patterns

### Registration with Error Handling

```python
from omnibase_infra.errors import ServiceRegistrationError

service_name = "PolicyRegistry"
context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.RUNTIME,
    operation="register_policy_registry",
    target_name=service_name,
    correlation_id=correlation_id,
)

try:
    policy_registry = PolicyRegistry()
    await container.service_registry.register_instance(
        interface=PolicyRegistry,
        instance=policy_registry,
        scope="global",
        metadata={
            "description": "ONEX policy plugin registry",
            "version": "1.0.0",
        },
    )
    logger.debug(f"Successfully registered {service_name}")

except AttributeError as e:
    logger.exception(f"Failed to register {service_name}: missing required attribute")
    raise ServiceRegistrationError(
        f"Failed to register {service_name}: container missing required attribute",
        service_name=service_name,
        context=context,
        original_error=str(e),
        hint="Ensure container.service_registry has register_instance() method",
    ) from e

except TypeError as e:
    logger.exception(f"Failed to register {service_name}: invalid arguments")
    raise ServiceRegistrationError(
        f"Failed to register {service_name}: invalid registration arguments",
        service_name=service_name,
        context=context,
        original_error=str(e),
        hint="Check register_instance() signature compatibility",
    ) from e
```

### Resolution with Error Handling

```python
from omnibase_infra.errors import ServiceResolutionError

service_name = "PolicyRegistry"
context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.RUNTIME,
    operation="resolve_policy_registry",
    target_name=service_name,
    correlation_id=correlation_id,
)

try:
    registry = await container.service_registry.resolve_service(PolicyRegistry)
    logger.debug(f"Successfully resolved {service_name}")
    return registry

except AttributeError as e:
    logger.exception(f"Failed to resolve {service_name}: missing required attribute")
    raise ServiceResolutionError(
        f"{service_name} resolution failed: container missing required attribute",
        service_name=service_name,
        context=context,
        original_error=str(e),
        hint="Ensure container.service_registry has resolve_service() method",
    ) from e

except Exception as e:
    logger.exception(f"Failed to resolve {service_name} from container")
    raise ServiceResolutionError(
        f"{service_name} not registered in container. "
        "Call wire_infrastructure_services(container) first.",
        service_name=service_name,
        context=context,
        original_error=str(e),
    ) from e
```

### Validation with Error Handling

```python
from omnibase_infra.errors import ContainerValidationError

context = ModelInfraErrorContext(
    transport_type=EnumInfraTransportType.RUNTIME,
    operation="validate_container",
    correlation_id=correlation_id,
)

if not hasattr(container, "service_registry"):
    logger.error("Container validation failed: missing service_registry attribute")
    raise ContainerValidationError(
        "Container missing required service_registry attribute. "
        "Ensure container is a valid ModelONEXContainer instance.",
        context=context,
        missing_attribute="service_registry",
        container_type=type(container).__name__,
    )

if not hasattr(container.service_registry, "register_instance"):
    logger.error("Container validation failed: missing register_instance method")
    raise ContainerValidationError(
        "Container service_registry missing required register_instance() method. "
        "Ensure container API is compatible with omnibase_core 0.4.x+.",
        context=context,
        missing_method="register_instance",
        registry_type=type(container.service_registry).__name__,
    )
```

## Logging Standards

All container wiring operations include structured logging:

```python
# Success logging
logger.debug(
    f"Successfully registered {service_name}",
    extra={"correlation_id": str(correlation_id)},
)

# Error logging
logger.exception(
    f"Failed to register {service_name}: unexpected error",
    extra={
        "correlation_id": str(correlation_id),
        "error": str(e),
        "error_type": type(e).__name__,
    },
)
```

### Logging Levels

- `DEBUG`: Service creation, registration, resolution (normal operations)
- `INFO`: Wiring start/completion, summary of registered services
- `ERROR`: Validation failures before exceptions
- `EXCEPTION`: All exception paths with stack traces

### Structured Logging Fields

All log entries include:
- `correlation_id`: UUID for request tracing
- `service_name`: Name of service being registered/resolved
- `error`: Error message (on failures)
- `error_type`: Exception type name (on failures)
- `hint`: Actionable guidance (on validation failures)

## Error Recovery Patterns

### Retry with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ServiceRegistrationError),
)
async def register_with_retry(container):
    return await wire_infrastructure_services(container)
```

### Graceful Degradation

```python
try:
    registry = await get_policy_registry_from_container(container)
except ServiceResolutionError:
    logger.warning("PolicyRegistry not found, using fallback")
    registry = PolicyRegistry()  # Fallback to direct instantiation
```

### Auto-Registration Fallback

```python
# Use get_or_create_policy_registry() for auto-registration
registry = await get_or_create_policy_registry(container)
# This will auto-register if not found, but logs a warning
```

## Testing

Comprehensive tests are provided in `tests/unit/errors/test_container_wiring_errors.py`

### Test Coverage

- **16 total tests**
- Error initialization and context
- Error chaining with original exceptions
- Error hierarchy and inheritance
- Realistic usage scenarios
- Correlation ID propagation
- Structured context access

### Running Tests

```bash
# Run container wiring error tests
poetry run pytest tests/unit/errors/test_container_wiring_errors.py -v

# Run all container wiring tests
poetry run pytest tests/unit/runtime/test_container_wiring.py -v

# Run all error tests
poetry run pytest tests/unit/errors/ -v
```

## Migration Guide

### Before (Old Pattern)

```python
try:
    await container.service_registry.register_instance(...)
except Exception as e:
    raise RuntimeError(f"Registration failed: {e}") from e
```

### After (New Pattern)

```python
try:
    await container.service_registry.register_instance(...)
except AttributeError as e:
    raise ServiceRegistrationError(
        "Failed to register: container missing required attribute",
        service_name="MyService",
        context=context,
        original_error=str(e),
        hint="Ensure container.service_registry exists",
    ) from e
```

## Benefits

1. **Clear Error Messages**: Each error type has specific, actionable messages
2. **Structured Context**: All errors include correlation IDs and structured context
3. **Debugging Support**: Original errors are chained, full stack traces preserved
4. **Logging Integration**: Comprehensive structured logging at each step
5. **Type Safety**: Strong typing for all error classes and context
6. **Testing**: Full test coverage for all error scenarios
7. **Recovery Patterns**: Clear patterns for retry, fallback, and graceful degradation

## Related Files

- `src/omnibase_infra/errors/error_container_wiring.py` - Error classes
- `src/omnibase_infra/errors/__init__.py` - Error exports
- `src/omnibase_infra/runtime/container_wiring.py` - Container wiring implementation
- `tests/unit/errors/test_container_wiring_errors.py` - Error tests
- `tests/unit/runtime/test_container_wiring.py` - Container wiring tests

## References

- ONEX Error Handling: `omnibase_core.models.errors.model_onex_error`
- Infrastructure Errors: `omnibase_infra.errors.infra_errors`
- Error Context Model: `omnibase_infra.errors.model_infra_error_context`
