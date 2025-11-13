# Production vs Template Comparison

**Purpose**: Side-by-side comparison of production code vs typical template output
**Last Updated**: 2025-11-05

---

## 1. Initialization Pattern

### ❌ Template (Basic)

```python
def __init__(self, container: ModelContainer):
    """Initialize node."""
    super().__init__(container)

    # Get services
    self.kafka_client = container.get_service("kafka_client")

    # Initialize metrics
    self._total_operations = 0
```

### ✅ Production (Comprehensive)

```python
def __init__(self, container: ModelContainer):
    """Initialize node with dependency injection container."""
    super().__init__(container)

    # Configuration - defensive pattern for dependency_injector
    try:
        if hasattr(container.config, "get") and callable(container.config.get):
            self.timeout = container.config.get(
                "timeout",
                os.getenv("TIMEOUT_SECONDS", "30")
            )
        else:
            self.timeout = os.getenv("TIMEOUT_SECONDS", "30")
    except Exception:
        self.timeout = os.getenv("TIMEOUT_SECONDS", "30")

    # Consul configuration for service discovery
    self.consul_host: str = container.config.get(
        "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
    )
    self.consul_port: int = container.config.get(
        "consul_port", int(os.getenv("CONSUL_PORT", "28500"))
    )
    self.consul_enable_registration: bool = container.config.get(
        "consul_enable_registration", True
    )

    # Health check mode detection
    try:
        health_check_mode = (
            container.config.get("health_check_mode", False)
            if hasattr(container.config, "get")
            else False
        )
    except Exception:
        health_check_mode = False

    # Get KafkaClient from container
    self.kafka_client = container.get_service("kafka_client")

    if self.kafka_client is None and not health_check_mode:
        try:
            from omninode_bridge.services.kafka_client import KafkaClient
            self.kafka_client = KafkaClient(
                bootstrap_servers=self.kafka_broker_url,
                enable_dead_letter_queue=True,
            )
            container.register_service("kafka_client", self.kafka_client)
        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "KafkaClient not available - events will be logged only",
                {"node_id": self.node_id},
            )
            self.kafka_client = None
    elif health_check_mode:
        emit_log_event(
            LogLevel.DEBUG,
            "Health check mode enabled - skipping Kafka initialization",
            {"node_id": self.node_id},
        )
        self.kafka_client = None

    # Initialize metrics
    self._total_operations = 0
    self._total_duration_ms = 0.0
    self._failed_operations = 0

    emit_log_event(
        LogLevel.INFO,
        "NodeCodegenOrchestrator initialized successfully",
        {
            "node_id": self.node_id,
            "kafka_enabled": self.kafka_client is not None,
            "timeout": self.timeout,
        },
    )

    # Register with Consul for service discovery
    if not health_check_mode and self.consul_enable_registration:
        self._register_with_consul_sync()
```

**Key Differences:**
- ❌ Template: No health_check_mode detection
- ❌ Template: No defensive configuration pattern
- ❌ Template: No Consul configuration
- ❌ Template: No service creation fallback
- ❌ Template: No registration with Consul
- ❌ Template: Missing comprehensive metrics

---

## 2. Event Publishing

### ❌ Template (Old Pattern)

```python
async def _publish_event(self, event_type: str, data: dict):
    """Publish event to Kafka."""
    if self.kafka_client:
        await self.kafka_client.publish(
            topic="events",
            data=data,
        )
```

### ✅ Production (OnexEnvelopeV1)

```python
async def _publish_event(
    self, event_type: EnumCodegenEvent, data: dict[str, Any]
) -> None:
    """
    Publish event to Kafka using OnexEnvelopeV1 wrapping.

    Args:
        event_type: Event type identifier
        data: Event payload data
    """
    try:
        # Get Kafka topic name from event type
        topic_name = event_type.get_topic_name(namespace=self.default_namespace)

        # Publish to Kafka if client is available
        if self.kafka_client and self.kafka_client.is_connected:
            # Extract correlation ID from data
            correlation_id = data.get("correlation_id") or data.get("workflow_id")

            # Add node metadata to payload
            payload = {
                **data,
                "node_id": self.node_id,
                "published_at": datetime.now(UTC).isoformat(),
            }

            # Publish with OnexEnvelopeV1 wrapping for standardized event format
            event_metadata = {
                "event_category": "codegen",
                "node_type": "orchestrator",
                "namespace": self.default_namespace,
            }

            # Add consul_service_id if available
            if hasattr(self, "_consul_service_id"):
                event_metadata["consul_service_id"] = self._consul_service_id

            success = await self.kafka_client.publish_with_envelope(
                event_type=event_type.value,
                source_node_id=str(self.node_id),
                payload=payload,
                topic=topic_name,
                correlation_id=correlation_id,
                metadata=event_metadata,
            )

            if success:
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Published Kafka event (OnexEnvelopeV1): {event_type.value}",
                    {
                        "node_id": self.node_id,
                        "event_type": event_type.value,
                        "topic_name": topic_name,
                        "correlation_id": correlation_id,
                        "envelope_wrapped": True,
                    },
                )
        else:
            # Kafka not available - log event only
            emit_log_event(
                LogLevel.DEBUG,
                f"Kafka unavailable, logging event: {event_type.value}",
                {
                    "node_id": self.node_id,
                    "event_type": event_type.value,
                    "data": data,
                },
            )

    except Exception as e:
        # Log error but don't fail workflow
        emit_log_event(
            LogLevel.WARNING,
            f"Failed to publish Kafka event: {event_type.value}",
            {
                "node_id": self.node_id,
                "event_type": event_type.value,
                "error": str(e),
            },
        )
```

**Key Differences:**
- ❌ Template: Uses old publish() method
- ❌ Template: No OnexEnvelopeV1 wrapping
- ❌ Template: No metadata (event_category, node_type, namespace)
- ❌ Template: No consul_service_id for cross-service correlation
- ❌ Template: No fallback when Kafka unavailable
- ❌ Template: Fails workflow if event publishing fails

---

## 3. Error Handling

### ❌ Template (Basic)

```python
try:
    result = await some_operation()
    return result
except Exception as e:
    raise ModelOnexError(
        message=f"Operation failed: {e}",
        error_code=EnumCoreErrorCode.OPERATION_FAILED,
    )
```

### ✅ Production (Comprehensive)

```python
try:
    result = await some_operation()
    return result

except OnexError:
    # Don't wrap OnexError - re-raise to preserve error context
    raise

except ConnectionError as e:
    # Network errors
    raise OnexError(
        error_code=EnumCoreErrorCode.CONNECTION_ERROR,
        message=f"Network connection failed: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "error_type": "ConnectionError",
        },
        cause=e,
    ) from e

except (TimeoutError, asyncio.TimeoutError) as e:
    # Timeout errors
    raise OnexError(
        error_code=EnumCoreErrorCode.TIMEOUT,
        message=f"Operation timed out: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "timeout_seconds": self.config.timeout,
        },
        cause=e,
    ) from e

except (ValueError, KeyError, AttributeError) as e:
    # Data validation errors
    emit_log_event(
        LogLevel.ERROR,
        f"Invalid data: {e}",
        {
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "error": str(e),
            "error_type": type(e).__name__,
        },
    )

    raise OnexError(
        error_code=EnumCoreErrorCode.VALIDATION_ERROR,
        message=f"Invalid data: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "error_type": type(e).__name__,
        },
        cause=e,
    ) from e

except Exception as e:
    # Unexpected errors - log with exc_info for debugging
    emit_log_event(
        LogLevel.ERROR,
        f"Unexpected error: {type(e).__name__}",
        {
            "node_id": self.node_id,
            "error": str(e),
            "error_type": type(e).__name__,
        },
    )
    logger.error(f"Unexpected error: {type(e).__name__}", exc_info=True)

    raise OnexError(
        error_code=EnumCoreErrorCode.INTERNAL_ERROR,
        message=f"Unexpected error: {e!s}",
        details={
            "node_id": self.node_id,
            "correlation_id": str(correlation_id),
            "error_type": type(e).__name__,
        },
        cause=e,
    ) from e
```

**Key Differences:**
- ❌ Template: No OnexError re-raise
- ❌ Template: No specific error type handling
- ❌ Template: No logging with exc_info
- ❌ Template: Generic error code for all errors
- ❌ Template: No context in details dict
- ❌ Template: No cause chaining

---

## 4. Lifecycle Methods

### ❌ Template (Missing)

```python
# Templates typically don't include startup/shutdown methods
```

### ✅ Production (Complete)

```python
async def startup(self) -> None:
    """
    Node startup lifecycle hook.

    Initializes container services, connects Kafka, registers with Consul,
    and starts background tasks.
    """
    emit_log_event(
        LogLevel.INFO,
        "NodeCodegenOrchestrator starting up",
        {"node_id": self.node_id},
    )

    # Initialize container services if available
    if hasattr(self.container, "initialize"):
        try:
            await self.container.initialize()
            emit_log_event(
                LogLevel.INFO,
                "Container services initialized successfully",
                {
                    "node_id": self.node_id,
                    "kafka_connected": (
                        self.kafka_client.is_connected if self.kafka_client else False
                    ),
                },
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Container initialization failed, continuing in degraded mode: {e}",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    # Connect to Kafka if client is available
    if self.kafka_client and not self.kafka_client.is_connected:
        try:
            await self.kafka_client.connect()
            emit_log_event(
                LogLevel.INFO,
                "Kafka client connected",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Kafka connection failed: {e}",
                {"node_id": self.node_id},
            )

    emit_log_event(
        LogLevel.INFO,
        "NodeCodegenOrchestrator startup complete",
        {"node_id": self.node_id},
    )


async def shutdown(self) -> None:
    """
    Node shutdown lifecycle hook.

    Stops background tasks, disconnects Kafka, deregisters from Consul,
    and cleans up resources.
    """
    emit_log_event(
        LogLevel.INFO,
        "NodeCodegenOrchestrator shutting down",
        {"node_id": self.node_id},
    )

    # Cleanup container services
    if hasattr(self.container, "cleanup"):
        try:
            await self.container.cleanup()
            emit_log_event(
                LogLevel.INFO,
                "Container services cleaned up successfully",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Container cleanup failed: {e}",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    # Deregister from Consul for clean service discovery
    self._deregister_from_consul()

    emit_log_event(
        LogLevel.INFO,
        "NodeCodegenOrchestrator shutdown complete",
        {"node_id": self.node_id},
    )
```

**Key Differences:**
- ❌ Template: No startup() method
- ❌ Template: No shutdown() method
- ❌ Template: No container initialization
- ❌ Template: No Kafka connection management
- ❌ Template: No Consul deregistration
- ❌ Template: No graceful degradation

---

## 5. Consul Registration

### ❌ Template (Missing)

```python
# Templates typically don't include Consul registration
```

### ✅ Production (Complete)

```python
def _register_with_consul_sync(self) -> None:
    """
    Register node with Consul for service discovery (synchronous).

    Registers the node as a service with health checks pointing to
    the health endpoint. Includes metadata about node capabilities.

    Note:
        This is a non-blocking registration. Failures are logged but don't
        fail node startup. Service will continue without Consul if registration fails.
    """
    try:
        import consul

        # Initialize Consul client
        consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)

        # Generate unique service ID
        service_id = f"omninode-bridge-codegen-orchestrator-{self.node_id}"

        # Get service port from config
        service_port = int(self.container.config.get("service_port", 8059))
        service_host = self.container.config.get("service_host", "localhost")

        # Prepare service tags (metadata encoded in tags)
        service_tags = [
            "onex",
            "bridge",
            "codegen_orchestrator",
            "orchestrator",
            f"version:{getattr(self, 'version', '0.1.0')}",
            "omninode_bridge",
            "node_type:codegen_orchestrator",
            f"kafka_enabled:{self.kafka_client is not None}",
        ]

        # Health check URL
        health_check_url = f"http://{service_host}:{service_port}/health"

        # Register service with Consul
        consul_client.agent.service.register(
            name="omninode-bridge-codegen-orchestrator",
            service_id=service_id,
            address=service_host,
            port=service_port,
            tags=service_tags,
            http=health_check_url,
            interval="30s",
            timeout="5s",
        )

        emit_log_event(
            LogLevel.INFO,
            "Registered with Consul successfully",
            {
                "node_id": self.node_id,
                "service_id": service_id,
                "consul_host": self.consul_host,
                "consul_port": self.consul_port,
            },
        )

        # Store service_id for deregistration
        self._consul_service_id = service_id

    except ImportError:
        emit_log_event(
            LogLevel.WARNING,
            "python-consul not installed - Consul registration skipped",
            {"node_id": self.node_id},
        )
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            "Failed to register with Consul",
            {
                "node_id": self.node_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )


def _deregister_from_consul(self) -> None:
    """
    Deregister node from Consul on shutdown (synchronous).

    Removes the service registration from Consul to prevent stale entries
    in the service catalog.
    """
    try:
        if not hasattr(self, "_consul_service_id"):
            return

        import consul

        consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
        consul_client.agent.service.deregister(self._consul_service_id)

        emit_log_event(
            LogLevel.INFO,
            "Deregistered from Consul successfully",
            {
                "node_id": self.node_id,
                "service_id": self._consul_service_id,
            },
        )

    except ImportError:
        pass
    except Exception as e:
        emit_log_event(
            LogLevel.WARNING,
            "Failed to deregister from Consul",
            {
                "node_id": self.node_id,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
```

**Key Differences:**
- ❌ Template: No Consul registration
- ❌ Template: No service discovery
- ❌ Template: No health check URL
- ❌ Template: No deregistration on shutdown
- ❌ Template: No graceful failure handling

---

## 6. Metrics Tracking

### ❌ Template (Minimal)

```python
def __init__(self, container: ModelContainer):
    """Initialize with minimal metrics."""
    super().__init__(container)
    self._total_operations = 0

def get_metrics(self) -> dict:
    """Get metrics."""
    return {"total_operations": self._total_operations}
```

### ✅ Production (Comprehensive)

```python
def __init__(self, container: ModelContainer):
    """Initialize with comprehensive metrics."""
    super().__init__(container)

    # Metrics tracking
    self._total_operations = 0
    self._total_duration_ms = 0.0
    self._failed_operations = 0
    self._successful_operations = 0

async def execute_orchestration(self, contract):
    """Execute with metrics tracking."""
    start_time = time.perf_counter()

    try:
        result = await self._do_operation()

        # Track success metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._total_operations += 1
        self._successful_operations += 1
        self._total_duration_ms += duration_ms

        return result

    except Exception as e:
        # Track failure metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._total_operations += 1
        self._failed_operations += 1
        self._total_duration_ms += duration_ms

        raise

def get_metrics(self) -> dict[str, Any]:
    """
    Get comprehensive metrics for monitoring and alerting.

    Returns:
        Dictionary with metrics
    """
    avg_duration_ms = (
        self._total_duration_ms / self._total_operations
        if self._total_operations > 0
        else 0
    )

    success_rate = (
        self._successful_operations / self._total_operations
        if self._total_operations > 0
        else 1.0
    )

    return {
        "total_operations": self._total_operations,
        "successful_operations": self._successful_operations,
        "failed_operations": self._failed_operations,
        "success_rate": round(success_rate, 4),
        "avg_duration_ms": round(avg_duration_ms, 2),
        "total_duration_ms": round(self._total_duration_ms, 2),
    }
```

**Key Differences:**
- ❌ Template: No duration tracking
- ❌ Template: No success/failure breakdown
- ❌ Template: No calculated metrics (avg, rates)
- ❌ Template: No time.perf_counter() usage
- ❌ Template: Minimal metrics returned

---

## 7. Time Tracking

### ❌ Template (Low Precision)

```python
import time

start = time.time()
# ... operation ...
duration = time.time() - start
```

### ✅ Production (High Precision)

```python
import time

# Use perf_counter for high-precision timing
start_time = time.perf_counter()

# ... operation ...

# Calculate duration in milliseconds
duration_ms = (time.perf_counter() - start_time) * 1000

# Round to 2 decimal places for logging
emit_log_event(
    LogLevel.INFO,
    "Operation completed",
    {
        "node_id": self.node_id,
        "duration_ms": round(duration_ms, 2),
    },
)
```

**Key Differences:**
- ❌ Template: Uses time.time() (low precision)
- ❌ Template: Duration in seconds
- ✅ Production: Uses time.perf_counter() (high precision)
- ✅ Production: Duration in milliseconds
- ✅ Production: Rounded to 2 decimal places

---

## Summary: Top 10 Production Patterns Missing from Templates

1. **Health check mode detection** - Skip expensive initialization in health checks
2. **Consul service discovery** - Registration and deregistration
3. **OnexEnvelopeV1 event wrapping** - Standardized event format
4. **Comprehensive error handling** - Specific exception types with proper wrapping
5. **Lifecycle methods** - startup() and shutdown() hooks
6. **Defensive configuration** - Fallbacks for container.config edge cases
7. **Service resolution** - Get or create services from container
8. **Event metadata** - Include consul_service_id, event_category, namespace
9. **Comprehensive metrics** - Duration, success rate, calculated metrics
10. **High-precision timing** - time.perf_counter() instead of time.time()

---

**Generated**: 2025-11-05
**For**: Code generation template improvements
**Status**: Gap analysis complete
