# Mixin Dependencies Guide

This document describes the mixin classes in `omnibase_infra` and their dependencies. Understanding these dependencies is essential when composing classes that use multiple mixins.

## Overview

Mixins in ONEX follow the naming convention: `mixin_<name>.py` -> `Mixin<Name>`. They provide reusable functionality that can be composed into classes through multiple inheritance. Many mixins have dependencies on:

1. **Other mixins** - Must be co-inherited in the class
2. **Host class attributes** - Must be provided by the implementing class
3. **Host class methods** - Must be implemented by the implementing class
4. **Initialization methods** - Must be called during `__init__`

## Quick Reference: Mixin Dependencies

Use this table to quickly identify mixin requirements and common usage patterns. For detailed implementation guidance, see the linked sections.

| Mixin | Dependencies | Init Method | Common Use Case |
|-------|-------------|-------------|-----------------|
| [`MixinAsyncCircuitBreaker`](#mixinasynccircuitbreaker) | None | `_init_circuit_breaker()` | Fault tolerance for external service calls |
| [`MixinKafkaDlq`](#mixinkafkadlq) | Protocol-based | `_init_dlq()` | Dead letter queue for failed Kafka messages |
| [`MixinKafkaBroadcast`](#mixinkafkabroadcast) | Protocol-based | None | Environment/group broadcast messaging |
| [`MixinNodeIntrospection`](#mixinnodeintrospection) | None | `initialize_introspection()` | Node capability discovery and heartbeat |
| [`MixinProjectorSqlOperations`](#mixin-index-by-category) | None | None | SQL operations for projectors |
| [`MixinRetryExecution`](#mixinretryexecution) | `MixinAsyncCircuitBreaker` | None | Retry with exponential backoff |
| [`MixinConsulInitialization`](#mixinconsulinitialization) | `MixinAsyncCircuitBreaker` | None | Consul client setup and connection |
| [`MixinVaultInitialization`](#mixinvaultinitialization) | `MixinAsyncCircuitBreaker` | None | Vault client setup and authentication |
| [`MixinEnvelopeExtraction`](#mixinenvelopeextraction) | None | None | Extract correlation/envelope IDs |
| [`MixinDictLikeAccessors`](#mixindictlikeaccessors) | None | None | Dict-like access for Pydantic models |
| [`MixinPolicyValidation`](#mixinpolicyvalidation) | None | None | Policy registration validation |
| [`MixinSemverCache`](#mixinsemvercache) | None | Class-level config | Cached semantic version parsing |

### Key Dependencies Summary

**Mixins requiring `MixinAsyncCircuitBreaker`** (must inherit first):
- `MixinRetryExecution` - Needs `_circuit_breaker_lock`, `_circuit_breaker_initialized`
- `MixinConsulInitialization` - Uses `_init_circuit_breaker()` for setup
- `MixinVaultInitialization` - Uses `_init_circuit_breaker()` for setup

**Protocol-based mixins** (require host class attributes):
- `MixinKafkaDlq` - Requires `_config`, `_producer`, `_producer_lock`, `_model_headers_to_kafka()`
- `MixinKafkaBroadcast` - Requires `_environment`, `_group`, `publish()`

---

## Core Infrastructure Mixins

### MixinAsyncCircuitBreaker

**Location**: `omnibase_infra/mixins/mixin_async_circuit_breaker.py`

**Dependencies**: None (standalone)

**Purpose**: Provides circuit breaker pattern for fault tolerance in infrastructure components.

**Initialization Required**:
```python
def _init_circuit_breaker(
    self,
    threshold: int = 5,
    reset_timeout: float = 60.0,
    service_name: str = "unknown",
    transport_type: EnumInfraTransportType = EnumInfraTransportType.HTTP,
) -> None
```

**Methods Provided**:
- `_check_circuit_breaker()` - Check if operation allowed
- `_record_circuit_failure()` - Record a failure
- `_reset_circuit_breaker()` - Reset to closed state
- `_get_circuit_breaker_state()` - Get state for introspection

**Concurrency Contract**: All circuit breaker methods require holding `_circuit_breaker_lock`:
```python
async with self._circuit_breaker_lock:
    await self._check_circuit_breaker(operation, correlation_id)
```

**Usage Example**:
```python
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.enums import EnumInfraTransportType

class MyService(MixinAsyncCircuitBreaker):
    def __init__(self):
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="my-service",
            transport_type=EnumInfraTransportType.HTTP,
        )

    async def call_external(self, correlation_id: UUID):
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("call_external", correlation_id)

        try:
            result = await self._do_external_call()
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()
            return result
        except Exception:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("call_external", correlation_id)
            raise
```

---

### MixinEnvelopeExtraction

**Location**: `omnibase_infra/mixins/mixin_envelope_extraction.py`

**Dependencies**: None (standalone)

**Purpose**: Extract `correlation_id` and `envelope_id` from request envelopes for distributed tracing.

**Methods Provided**:
- `_extract_correlation_id(envelope: dict) -> UUID`
- `_extract_envelope_id(envelope: dict) -> UUID`

**Auto-Generation**: If IDs are missing or invalid, new UUIDs are generated automatically.

**Usage Example**:
```python
from omnibase_infra.mixins import MixinEnvelopeExtraction

class MyHandler(MixinEnvelopeExtraction):
    async def handle(self, envelope: dict[str, object]):
        correlation_id = self._extract_correlation_id(envelope)
        envelope_id = self._extract_envelope_id(envelope)
        # Use IDs for tracing and logging
```

---

### MixinDictLikeAccessors

**Location**: `omnibase_infra/mixins/mixin_dict_like_accessors.py`

**Dependencies**: None (standalone, designed for Pydantic models)

**Purpose**: Enable dict-like access patterns (`get()`, `[]`, `in`) for Pydantic BaseModel subclasses.

**Methods Provided**:
- `get(key, default=None)` - Safe access with optional default
- `__getitem__(key)` - Bracket notation (raises `KeyError` if missing)
- `__contains__(key)` - Membership testing (returns `False` if value is `None`)

**Usage Example**:
```python
from pydantic import BaseModel, ConfigDict
from omnibase_infra.mixins import MixinDictLikeAccessors

class FlexibleModel(MixinDictLikeAccessors, BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""

model = FlexibleModel(name="test", custom_field="value")
model.get("name")           # "test"
model["custom_field"]       # "value"
"name" in model             # True
model.get("missing", "N/A") # "N/A"
```

---

### MixinRetryExecution

**Location**: `omnibase_infra/mixins/mixin_retry_execution.py`

**Dependencies**: **REQUIRES `MixinAsyncCircuitBreaker`**

**Purpose**: Provides retry execution with exponential backoff and circuit breaker integration.

**Required from Host Class**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `_circuit_breaker_initialized` | `bool` | Whether circuit breaker is initialized |
| `_executor` | `ThreadPoolExecutor \| None` | Thread pool for sync operations |

**Abstract Methods (must implement)**:
```python
def _classify_error(self, error: Exception, operation: str) -> ModelRetryErrorClassification:
    """Classify an exception for retry handling."""
    ...

def _get_transport_type(self) -> EnumInfraTransportType:
    """Return the transport type for error context."""
    ...

def _get_target_name(self) -> str:
    """Return the target name for error context."""
    ...
```

**Inheritance Order**: `MixinAsyncCircuitBreaker` **MUST** come before `MixinRetryExecution`:
```python
# CORRECT - Circuit breaker provides methods retry execution needs
class HandlerConsul(MixinAsyncCircuitBreaker, MixinRetryExecution):
    pass

# INCORRECT - Will fail because _circuit_breaker_lock not available
class HandlerConsul(MixinRetryExecution, MixinAsyncCircuitBreaker):
    pass
```

**Usage Example**:
```python
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinRetryExecution
from omnibase_infra.enums import EnumInfraTransportType, EnumRetryErrorCategory
from omnibase_infra.models import ModelRetryErrorClassification

class HandlerConsul(MixinAsyncCircuitBreaker, MixinRetryExecution):
    def __init__(self, config):
        self._init_circuit_breaker(
            threshold=5, reset_timeout=60.0,
            service_name="consul", transport_type=EnumInfraTransportType.CONSUL
        )
        self._circuit_breaker_initialized = True
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _classify_error(self, error: Exception, operation: str) -> ModelRetryErrorClassification:
        if isinstance(error, consul.ACLPermissionDenied):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.AUTHENTICATION,
                should_retry=False,
                record_circuit_failure=True,
                error_message="Consul ACL permission denied",
            )
        # ... handle other error types

    def _get_transport_type(self) -> EnumInfraTransportType:
        return EnumInfraTransportType.CONSUL

    def _get_target_name(self) -> str:
        return "consul_handler"
```

---

### MixinNodeIntrospection

**Location**: `omnibase_infra/mixins/mixin_node_introspection.py`

**Dependencies**: None (event bus is optional)

**Purpose**: Automatic capability discovery, endpoint reporting, and heartbeat broadcasting for ONEX nodes.

**Initialization Required**:
```python
config = ModelIntrospectionConfig(
    node_id=node_config.node_id,
    node_type=EnumNodeKind.EFFECT,
    event_bus=event_bus,  # Optional
)
self.initialize_introspection(config)
```

**Methods Provided**:
- `initialize_introspection(config)` - Initialize the mixin
- `get_introspection_data()` - Get node capabilities and metadata
- `publish_introspection(reason)` - Publish introspection event
- `start_introspection_tasks()` - Start background heartbeat/listener
- `stop_introspection_tasks()` - Stop background tasks
- `track_operation()` - Context manager for tracking active operations

**Usage Example**:
```python
from omnibase_infra.mixins import MixinNodeIntrospection
from omnibase_infra.models.discovery import ModelIntrospectionConfig

class MyNode(MixinNodeIntrospection):
    def __init__(self, node_id: UUID, event_bus=None):
        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            event_bus=event_bus,
        )
        self.initialize_introspection(config)

    async def startup(self):
        await self.publish_introspection(reason="startup")
        await self.start_introspection_tasks(
            enable_heartbeat=True,
            heartbeat_interval_seconds=30.0,
        )

    async def shutdown(self):
        await self.publish_introspection(reason="shutdown")
        await self.stop_introspection_tasks()

    async def process_request(self, data):
        async with self.track_operation():
            # Operation is tracked in heartbeat metrics
            return await self._do_work(data)
```

---

## Kafka Event Bus Mixins

### MixinKafkaDlq

**Location**: `omnibase_infra/event_bus/mixin_kafka_dlq.py`

**Dependencies**: Uses `ProtocolKafkaDlqHost` Protocol (duck typing)

**Purpose**: Dead Letter Queue functionality for handling failed Kafka messages.

**Required from Host Class**:
| Attribute/Method | Type | Description |
|-----------------|------|-------------|
| `_config` | `ModelKafkaEventBusConfig` | Kafka configuration with `dead_letter_topic` |
| `_environment` | `str` | Environment context (e.g., "dev", "prod") |
| `_group` | `str` | Consumer group identifier |
| `_producer` | `AIOKafkaProducer \| None` | Kafka producer instance |
| `_producer_lock` | `asyncio.Lock` | Lock for producer access |
| `_timeout_seconds` | `int` | Publish timeout |
| `_model_headers_to_kafka()` | Method | Convert `ModelEventHeaders` to Kafka format |

**Initialization Required**:
```python
def _init_dlq(self) -> None
```

**Methods Provided**:
- `register_dlq_callback()` - Register callback for DLQ events
- `dlq_metrics` - Property returning DLQ metrics
- `_publish_to_dlq()` - Publish failed message to DLQ
- `_publish_raw_to_dlq()` - Publish raw message when deserialization fails

**Usage Example**:
```python
from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq
from omnibase_infra.event_bus.mixin_kafka_broadcast import MixinKafkaBroadcast
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

class EventBusKafka(MixinKafkaBroadcast, MixinKafkaDlq, MixinAsyncCircuitBreaker):
    def __init__(self, config: ModelKafkaEventBusConfig):
        # Initialize host attributes
        self._config = config
        self._environment = config.environment
        self._group = config.consumer_group
        self._producer = None
        self._producer_lock = asyncio.Lock()
        self._timeout_seconds = config.timeout_seconds

        # Initialize mixins
        self._init_dlq()
        self._init_circuit_breaker(...)

    def _model_headers_to_kafka(
        self, headers: ModelEventHeaders
    ) -> list[tuple[str, bytes]]:
        """Convert model headers to Kafka format."""
        return [
            ("source", headers.source.encode("utf-8")),
            ("event_type", headers.event_type.encode("utf-8")),
            # ... other headers
        ]
```

---

### MixinKafkaBroadcast

**Location**: `omnibase_infra/event_bus/mixin_kafka_broadcast.py`

**Dependencies**: Uses `ProtocolKafkaBroadcastHost` Protocol (duck typing)

**Purpose**: Environment-wide and group-specific broadcast messaging.

**Required from Host Class**:
| Attribute/Method | Type | Description |
|-----------------|------|-------------|
| `_environment` | `str` | Environment context |
| `_group` | `str` | Consumer group identifier |
| `publish()` | Async method | Publish message to topic |

**Methods Provided**:
- `broadcast_to_environment()` - Broadcast to all subscribers in environment
- `send_to_group()` - Send to specific consumer group
- `publish_envelope()` - Publish `OnexEnvelope` to topic

**Usage Example**:
```python
from omnibase_infra.event_bus.mixin_kafka_broadcast import MixinKafkaBroadcast
from omnibase_infra.event_bus.mixin_kafka_dlq import MixinKafkaDlq

class EventBusKafka(MixinKafkaBroadcast, MixinKafkaDlq, MixinAsyncCircuitBreaker):
    def __init__(self, config):
        self._environment = config.environment
        self._group = config.consumer_group
        # ... other initialization

    async def publish(self, topic: str, key: bytes | None, value: bytes, headers=None):
        """Base publish method required by MixinKafkaBroadcast."""
        # ... implementation
```

---

## Handler Initialization Mixins

### MixinConsulInitialization

**Location**: `omnibase_infra/handlers/mixins/mixin_consul_initialization.py`

**Dependencies**: **REQUIRES `MixinAsyncCircuitBreaker`**

**Purpose**: Consul client setup, connection verification, and configuration validation.

**Required from Host Class**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `_executor` | `ThreadPoolExecutor \| None` | Thread pool instance |
| `_max_workers` | `int` | Max worker count |
| `_max_queue_size` | `int` | Max queue size |
| `_circuit_breaker_initialized` | `bool` | Circuit breaker init flag |

**Required Methods** (from `MixinAsyncCircuitBreaker`):
```python
def _init_circuit_breaker(
    self, threshold: int, reset_timeout: float,
    service_name: str, transport_type: EnumInfraTransportType
) -> None
```

**Methods Provided**:
- `_validate_consul_config()` - Validate and parse config
- `_setup_consul_client()` - Create Consul client
- `_verify_consul_connection()` - Verify connectivity
- `_setup_thread_pool()` - Set up thread pool
- `_setup_circuit_breaker()` - Initialize circuit breaker
- `_log_initialization_success()` - Log success
- `_raise_auth_error()`, `_raise_connection_error()`, `_raise_runtime_error()` - Error helpers

---

### MixinVaultInitialization

**Location**: `omnibase_infra/handlers/mixins/mixin_vault_initialization.py`

**Dependencies**: **REQUIRES `MixinAsyncCircuitBreaker`**

**Purpose**: Vault client setup, authentication verification, and token TTL tracking.

**Required from Host Class**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `_client` | `hvac.Client \| None` | Vault client instance |
| `_config` | `ModelVaultHandlerConfig \| None` | Vault configuration |
| `_token_expires_at` | `float` | Token expiration timestamp |
| `_executor` | `ThreadPoolExecutor \| None` | Thread pool instance |
| `_max_workers` | `int` | Max worker count |
| `_max_queue_size` | `int` | Max queue size |
| `_circuit_breaker_initialized` | `bool` | Circuit breaker init flag |

**Methods Provided**:
- `_parse_vault_config()` - Parse and validate config
- `_create_hvac_client()` - Create hvac client
- `_verify_vault_auth()` - Verify authentication
- `_initialize_token_ttl()` - Initialize TTL tracking
- `_setup_thread_pool()` - Set up thread pool
- `_setup_circuit_breaker()` - Initialize circuit breaker
- `_log_init_success()` - Log success
- `_handle_init_hvac_error()` - Handle hvac errors

---

## Runtime/Registry Mixins

### MixinPolicyValidation

**Location**: `omnibase_infra/runtime/mixin_policy_validation.py`

**Dependencies**: None (standalone)

**Purpose**: Policy registration validation for `RegistryPolicy`.

**Methods Provided**:
- `_validate_protocol_implementation()` - Validate policy implements `ProtocolPolicy`
- `_validate_sync_enforcement()` - Enforce sync/async policy rules
- `_normalize_policy_type()` - Normalize and validate policy type
- `_normalize_version()` - Normalize version string

---

### MixinSemverCache

**Location**: `omnibase_infra/runtime/mixin_semver_cache.py`

**Dependencies**: None (standalone, class-level)

**Purpose**: LRU-cached semantic version parsing for `RegistryPolicy`.

**Class Attributes**:
- `SEMVER_CACHE_SIZE` - Cache size (default: 128)

**Class Methods**:
- `configure_semver_cache(maxsize)` - Configure cache size before first use
- `_reset_semver_cache()` - Reset cache (for testing)
- `_parse_semver(version)` - Parse version string to `ModelSemVer`
- `_get_semver_cache_info()` - Get cache statistics

---

## Mixin Composition Patterns

### Correct Inheritance Order

When combining mixins with **mixin-based dependencies**, **order matters**. Mixins that provide functionality must come before mixins that depend on them in the MRO. However, **Protocol-based dependencies** (where mixins require attributes from the host class, not from other mixins) do not require specific ordering.

```python
# CORRECT - HandlerConsul: MixinRetryExecution depends on MixinAsyncCircuitBreaker (mixin dependency)
# MixinAsyncCircuitBreaker MUST come first because MixinRetryExecution accesses its attributes via MRO
class HandlerConsul(MixinAsyncCircuitBreaker, MixinRetryExecution):
    pass

# CORRECT - EventBusKafka: All mixins use Protocol-based dependencies (no mixin ordering required)
# MixinAsyncCircuitBreaker can be last because MixinKafkaDlq and MixinKafkaBroadcast
# get their required attributes from the host class, not from other mixins
class EventBusKafka(MixinKafkaBroadcast, MixinKafkaDlq, MixinAsyncCircuitBreaker):
    pass

# CORRECT - RegistryPolicy: Neither mixin depends on the other
class RegistryPolicy(MixinPolicyValidation, MixinSemverCache):
    pass

# INCORRECT - MixinRetryExecution needs MixinAsyncCircuitBreaker first in MRO
class BadHandler(MixinRetryExecution, MixinAsyncCircuitBreaker):  # Will fail!
    pass
```

**Why `EventBusKafka` has `MixinAsyncCircuitBreaker` last:**

The ordering principle applies to **mixin-based dependencies** (where one mixin directly uses methods/attributes from another mixin via MRO). `MixinRetryExecution` has a mixin dependency on `MixinAsyncCircuitBreaker` because it accesses `_circuit_breaker_lock` and `_circuit_breaker_initialized` at runtime.

In contrast, `MixinKafkaDlq` and `MixinKafkaBroadcast` use **Protocol-based dependencies** (duck typing). They require the HOST class to provide certain attributes (`_config`, `_producer`, etc.) but do not depend on any other mixin in the MRO. The host class (e.g., `EventBusKafka`) provides these attributes directly in `__init__`, so the mixin order is flexible.

| Dependency Type | Example | Ordering Required |
|----------------|---------|-------------------|
| **Mixin-based** | `MixinRetryExecution` uses `MixinAsyncCircuitBreaker._circuit_breaker_lock` | Yes - provider mixin must come first |
| **Protocol-based** | `MixinKafkaDlq` requires `ProtocolKafkaDlqHost._config` from host class | No - host provides attributes directly |

### Complete Initialization Example

When using multiple mixins, ensure all initialization methods are called:

```python
class MyCompleteHandler(
    MixinEnvelopeExtraction,
    MixinAsyncCircuitBreaker,
    MixinRetryExecution,
):
    def __init__(self, config):
        # Initialize circuit breaker (required by MixinRetryExecution)
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
            service_name="my-handler",
            transport_type=EnumInfraTransportType.HTTP,
        )

        # Set attributes required by MixinRetryExecution
        self._circuit_breaker_initialized = True
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)

        # MixinEnvelopeExtraction requires no initialization
```

### Protocol-Based Dependencies

Some mixins use Protocol classes to define their requirements. This allows them to work with any class that provides the required attributes/methods without explicit inheritance:

```python
# MixinKafkaDlq defines ProtocolKafkaDlqHost
@runtime_checkable
class ProtocolKafkaDlqHost(Protocol):
    _config: ModelKafkaEventBusConfig
    _environment: str
    _group: str
    _producer: AIOKafkaProducer | None
    _producer_lock: asyncio.Lock
    _timeout_seconds: int

    def _model_headers_to_kafka(
        self, headers: ModelEventHeaders
    ) -> list[tuple[str, bytes]]:
        ...
```

Any class providing these attributes/methods can use `MixinKafkaDlq`.

---

## Mixin Index by Category

### Core Infrastructure
| Mixin | File | Purpose |
|-------|------|---------|
| `MixinAsyncCircuitBreaker` | `mixins/mixin_async_circuit_breaker.py` | Circuit breaker pattern |
| `MixinEnvelopeExtraction` | `mixins/mixin_envelope_extraction.py` | Extract tracing IDs |
| `MixinDictLikeAccessors` | `mixins/mixin_dict_like_accessors.py` | Dict-like Pydantic access |
| `MixinRetryExecution` | `mixins/mixin_retry_execution.py` | Retry with backoff |
| `MixinNodeIntrospection` | `mixins/mixin_node_introspection.py` | Node discovery |

### Kafka Event Bus
| Mixin | File | Purpose |
|-------|------|---------|
| `MixinKafkaDlq` | `event_bus/mixin_kafka_dlq.py` | Dead letter queue |
| `MixinKafkaBroadcast` | `event_bus/mixin_kafka_broadcast.py` | Broadcast messaging |

### Handler Initialization
| Mixin | File | Purpose |
|-------|------|---------|
| `MixinConsulInitialization` | `handlers/mixins/mixin_consul_initialization.py` | Consul client setup |
| `MixinVaultInitialization` | `handlers/mixins/mixin_vault_initialization.py` | Vault client setup |
| `MixinConsulKV` | `handlers/mixins/mixin_consul_kv.py` | Consul KV operations |
| `MixinConsulService` | `handlers/mixins/mixin_consul_service.py` | Consul service ops |
| `MixinVaultSecrets` | `handlers/mixins/mixin_vault_secrets.py` | Vault secret ops |
| `MixinVaultToken` | `handlers/mixins/mixin_vault_token.py` | Vault token ops |
| `MixinVaultRetry` | `handlers/mixins/mixin_vault_retry.py` | Vault retry logic |

### Runtime/Registry
| Mixin | File | Purpose |
|-------|------|---------|
| `MixinPolicyValidation` | `runtime/mixin_policy_validation.py` | Policy validation |
| `MixinSemverCache` | `runtime/mixin_semver_cache.py` | Semver caching |
| `MixinMessageTypeRegistration` | `runtime/registry/mixin_message_type_registration.py` | Message type registration |
| `MixinMessageTypeQuery` | `runtime/registry/mixin_message_type_query.py` | Message type queries |
| `MixinProjectorSqlOperations` | `runtime/mixins/mixin_projector_sql_operations.py` | SQL projector ops |

### Validation
| Mixin | File | Purpose |
|-------|------|---------|
| `MixinAnyTypeExemption` | `validation/mixin_any_type_exemption.py` | Any type exemptions |
| `MixinAnyTypeClassification` | `validation/mixin_any_type_classification.py` | Any type classification |
| `MixinAnyTypeReporting` | `validation/mixin_any_type_reporting.py` | Any type reporting |
| `MixinNodeArchetypeDetection` | `validation/mixin_node_archetype_detection.py` | Node archetype detection |
| `MixinExecutionShapeViolationChecks` | `validation/mixin_execution_shape_violation_checks.py` | Execution shape checks |

### Architecture Validator
| Mixin | File | Purpose |
|-------|------|---------|
| `MixinFilePathRule` | `nodes/architecture_validator/mixins/mixin_file_path_rule.py` | File path rule base |

---

## See Also

- [Circuit Breaker Implementation](./circuit_breaker_implementation.md) - Full circuit breaker pattern details
- [Error Recovery Patterns](./error_recovery_patterns.md) - Retry and backoff patterns
- [Dispatcher Resilience](./dispatcher_resilience.md) - Dispatcher-owned resilience pattern
- [Security Patterns](./security_patterns.md) - Node introspection security considerations
