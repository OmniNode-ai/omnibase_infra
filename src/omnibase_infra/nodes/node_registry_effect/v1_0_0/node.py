# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Effect Node for dual registration to Consul and PostgreSQL.

This effect node bridges the message bus to external infrastructure services,
implementing the 2-way registration pattern for ONEX nodes.

Features:
    - Dual registration (Consul + PostgreSQL) with parallel execution
    - Circuit breaker protection via MixinAsyncCircuitBreaker
    - Graceful degradation (partial success when one backend fails)
    - Correlation ID propagation for distributed tracing
    - UPSERT pattern for idempotent re-registration

Operations:
    - register: Register node with Consul and PostgreSQL
    - deregister: Remove node from both backends
    - discover: Query registered nodes with filters
    - request_introspection: Publish introspection request to event bus

Handler Interface (duck typing):
    consul_handler must implement ProtocolConsulExecutor:
        async def execute(self, envelope: EnvelopeDict) -> ModelConsulHandlerResponse
    db_handler must implement ProtocolDbExecutor:
        async def execute(self, envelope: EnvelopeDict) -> ModelDbQueryResponse

Event Bus Interface (duck typing):
    event_bus must implement:
        async def publish(self, topic: str, key: bytes, value: bytes) -> None
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID

from omnibase_core.models.node_metadata import ModelNodeCapabilitiesInfo

from omnibase_infra.enums import EnumInfraTransportType

# Type alias for registry operation status (must match ModelRegistryResponse.status)
RegistryStatus = Literal["success", "partial", "failed"]

from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models import (
    EnumEnvironment,
    ModelConsulOperationResult,
    ModelNodeIntrospectionPayload,
    ModelNodeRegistration,
    ModelNodeRegistrationMetadata,
    ModelNodeRegistryEffectConfig,
    ModelPostgresOperationResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
    EnvelopeDict,
    JsonValue,
    ProtocolConsulExecutor,
    ProtocolDbExecutor,
    ProtocolEventBus,
)

logger = logging.getLogger(__name__)

# Whitelist of allowed filter keys for SQL query building (SQL injection prevention)
# These correspond to actual columns in the node_registrations table that support
# direct equality filtering. JSONB columns (capabilities, endpoints, metadata) are
# excluded as they require specialized query operators.
#
# SECURITY: This whitelist prevents SQL injection by ensuring only known-safe
# column names can be interpolated into SQL queries. Filter VALUES are always
# parameterized (never interpolated). Invalid filter keys cause the request to
# be rejected with an error, not silently ignored.
ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "node_id",  # Primary key, VARCHAR(255)
        "node_type",  # Node classification (effect, compute, reducer, orchestrator)
        "node_version",  # Semantic version string
        "health_endpoint",  # Health check URL (nullable)
    }
)

# Default performance threshold for slow operation warnings (milliseconds)
# Used as fallback when config not provided
DEFAULT_SLOW_OPERATION_THRESHOLD_MS: float = 1000.0


class NodeRegistryEffect(MixinAsyncCircuitBreaker):
    """Registry Effect Node for dual registration to Consul and PostgreSQL.

    This node implements the EFFECT node type, performing I/O operations
    to external services (Consul, PostgreSQL, Kafka) for node registration.

    Thread Safety:
        Uses MixinAsyncCircuitBreaker for circuit breaker state management.
        All circuit breaker operations require holding _circuit_breaker_lock.

    Error Handling:
        - InfraConnectionError: Backend connection failures
        - InfraUnavailableError: Circuit breaker open
        - RuntimeHostError: Configuration/validation errors

    Duck Typing Interfaces:
        consul_handler: Must have async execute(envelope: dict) -> dict method
        db_handler: Must have async execute(envelope: dict) -> dict method
        event_bus: Must have async publish(topic: str, key: bytes, value: bytes) -> None method

    Connection Pooling:
        This node delegates database operations to db_handler and does not manage
        connections directly. The db_handler implementation (e.g., PostgreSQL adapter)
        is responsible for:
        - Connection pool management to prevent connection exhaustion
        - Connection health checks and automatic reconnection
        - Proper connection release after each operation

        For production deployments, ensure the PostgreSQL adapter uses asyncpg
        with connection pooling (recommended pool size: 10-20 connections).

    Dependency Injection:
        TODO(OMN-901): Migrate to container-based dependency injection.

        Per ONEX patterns (CLAUDE.md), all services should use ModelONEXContainer
        for dependency injection. Current implementation uses direct constructor
        injection because:

        1. The container infrastructure (wire_infrastructure_services) does not
           yet support registration/resolution of:
           - ProtocolConsulExecutor handlers
           - ProtocolDbExecutor handlers
           - ProtocolEventBus implementations

        2. These handlers require runtime configuration (connection strings,
           credentials from Vault) that would need container-level wiring.

        Migration path when container support is ready:
        ```python
        # Future container-based implementation
        @classmethod
        async def create_from_container(
            cls, container: ModelONEXContainer
        ) -> NodeRegistryEffect:
            consul_handler = await container.service_registry.resolve_service(
                ProtocolConsulExecutor, name="consul"
            )
            db_handler = await container.service_registry.resolve_service(
                ProtocolDbExecutor, name="postgres"
            )
            event_bus = await container.service_registry.resolve_service(
                ProtocolEventBus
            )
            return cls(consul_handler, db_handler, event_bus)
        ```

        Required infrastructure work:
        - Add handler registration to wire_infrastructure_services()
        - Add event bus registration to wire_infrastructure_services()
        - Support named service resolution for multiple handlers of same protocol
        - Ensure handlers are wired with proper Vault/Consul credentials
    """

    def __init__(
        self,
        consul_handler: ProtocolConsulExecutor,
        db_handler: ProtocolDbExecutor,
        event_bus: ProtocolEventBus | None = None,
        config: ModelNodeRegistryEffectConfig | None = None,
    ) -> None:
        """Initialize Registry Effect Node.

        Args:
            consul_handler: Handler for Consul operations.
                Must implement: async execute(envelope: EnvelopeDict) -> ResultDict
            db_handler: Handler for PostgreSQL operations.
                Must implement: async execute(envelope: EnvelopeDict) -> ResultDict
                The handler is responsible for connection lifecycle management,
                including connection pooling and automatic reconnection.
            event_bus: Optional event bus for publishing events.
                Must implement: async publish(topic: str, key: bytes, value: bytes) -> None
            config: Configuration for circuit breaker and resilience settings.
                Uses defaults if not provided.

        Note:
            This node does not manage database connections directly. The db_handler
            must handle connection pooling internally to prevent connection exhaustion
            under high load. See class docstring for production recommendations.

            TODO(OMN-901): This constructor uses direct injection. When container
            infrastructure supports handler registration, add a factory method
            `create_from_container()` to resolve dependencies from container.
        """
        self._consul_handler: ProtocolConsulExecutor = consul_handler
        self._db_handler: ProtocolDbExecutor = db_handler
        self._event_bus: ProtocolEventBus | None = event_bus
        self._initialized = False

        # Use defaults if config not provided
        config = config or ModelNodeRegistryEffectConfig()

        # Store slow operation threshold from config (configurable per environment)
        self._slow_operation_threshold_ms: float = config.slow_operation_threshold_ms

        # Initialize circuit breaker
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout,
            service_name="node_registry_effect",
            transport_type=EnumInfraTransportType.RUNTIME,
        )

    def _sanitize_error(self, exception: BaseException) -> str:
        """Sanitize exception message for safe logging and response.

        Removes potential sensitive information and includes exception type.
        Redacts patterns like passwords, tokens, API keys, and connection strings.

        Args:
            exception: The exception to sanitize

        Returns:
            Sanitized error string in format: "{ExceptionType}: {sanitized_message}"
        """
        exception_type = type(exception).__name__
        message = str(exception)

        # Redact potential secrets using case-insensitive patterns
        sensitive_patterns = [
            (r"password[=:]\s*\S+", "password=[REDACTED]"),
            (r"passwd[=:]\s*\S+", "passwd=[REDACTED]"),
            (r"pwd[=:]\s*\S+", "pwd=[REDACTED]"),
            (r"token[=:]\s*\S+", "token=[REDACTED]"),
            (r"api_key[=:]\s*\S+", "api_key=[REDACTED]"),
            (r"apikey[=:]\s*\S+", "apikey=[REDACTED]"),
            (r"key[=:]\s*\S+", "key=[REDACTED]"),
            (r"secret[=:]\s*\S+", "secret=[REDACTED]"),
            (r"credential[s]?[=:]\s*\S+", "credentials=[REDACTED]"),
            (r"auth[=:]\s*\S+", "auth=[REDACTED]"),
            (r"bearer\s+\S+", "bearer [REDACTED]"),
            # Connection string credentials (user:pass@host)
            (r"://[^:]+:[^@]+@", "://[REDACTED]@"),
            # AWS-style keys
            (r"AKIA[A-Z0-9]{16}", "[REDACTED_AWS_KEY]"),
            # Long base64-like tokens
            (r"[A-Za-z0-9+/]{40,}={0,2}", "[REDACTED_TOKEN]"),
        ]

        for pattern, replacement in sensitive_patterns:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)

        # Truncate if too long
        if len(message) > 500:
            message = message[:497] + "..."

        return f"{exception_type}: {message}"

    def _json_default_serializer(
        self, obj: object
    ) -> str | list[str] | dict[str, JsonValue]:
        """Default serializer for non-standard JSON types.

        This method is used as the `default` parameter for json.dumps() to handle
        types that are not natively JSON serializable.

        Supported types:
            - datetime: Converted to ISO 8601 format string
            - UUID: Converted to string representation
            - Enum: Converted to its value
            - bytes: Decoded as UTF-8, with fallback to repr()
            - set/frozenset: Converted to sorted list
            - Pydantic BaseModel: Converted via model_dump()
            - Other: Falls back to str() representation

        Args:
            obj: The object to serialize

        Returns:
            A JSON-serializable representation of the object

        Note:
            This method should handle serialization gracefully without raising
            exceptions. If an object cannot be converted, it returns a string
            representation with a warning indicator.
        """
        from enum import Enum

        from pydantic import BaseModel

        # datetime types
        if isinstance(obj, datetime):
            return obj.isoformat()

        # UUID
        if isinstance(obj, UUID):
            return str(obj)

        # Enum values - convert to string value
        if isinstance(obj, Enum):
            return str(obj.value)

        # bytes
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return repr(obj)

        # set/frozenset - convert to sorted list for consistent output
        if isinstance(obj, (set, frozenset)):
            try:
                return sorted(str(item) for item in obj)
            except TypeError:
                # Items not comparable, just convert to list
                return [str(item) for item in obj]

        # Pydantic models
        if isinstance(obj, BaseModel):
            try:
                return cast(dict[str, JsonValue], obj.model_dump(mode="json"))
            except Exception:
                # If model_dump fails, try dict conversion
                return str(obj)

        # Fallback: string representation with warning indicator
        return f"<non-serializable: {type(obj).__name__}>"

    def _safe_model_dump(
        self,
        model: object,
        correlation_id: UUID | None = None,
        field_name: str = "unknown",
    ) -> dict[str, JsonValue]:
        """Safely dump a Pydantic model to a dictionary.

        This method wraps Pydantic's model_dump() with proper error handling,
        returning an empty dict on failure rather than raising an exception.

        Args:
            model: The Pydantic model to dump (or any object with model_dump method)
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging

        Returns:
            Dictionary representation of the model, or empty dict on failure
        """
        try:
            # Check if object has model_dump method (Pydantic v2)
            if hasattr(model, "model_dump"):
                result = model.model_dump(mode="json")
                if isinstance(result, dict):
                    return cast(dict[str, JsonValue], result)
                logger.warning(
                    f"model_dump() returned non-dict for {field_name}",
                    extra={
                        "correlation_id": str(correlation_id)
                        if correlation_id
                        else None,
                        "field_name": field_name,
                        "result_type": type(result).__name__,
                    },
                )
                return {}

            # Fallback for dict-like objects
            if hasattr(model, "__dict__"):
                return cast(dict[str, JsonValue], dict(model.__dict__))

            logger.warning(
                f"Object has no model_dump or __dict__ for {field_name}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "object_type": type(model).__name__,
                },
            )
            return {}

        except (TypeError, ValueError, AttributeError, RecursionError) as e:
            logger.warning(
                f"Model serialization failed for {field_name}: {type(e).__name__}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
                    "model_type": type(model).__name__,
                },
            )
            return {}

    def _log_operation_performance(
        self,
        operation: str,
        processing_time_ms: float,
        success: bool,
        correlation_id: UUID | None = None,
        node_id: str | None = None,
        record_count: int | None = None,
        status: str | None = None,
    ) -> None:
        """Log structured performance metrics for registry operations.

        Args:
            operation: The operation type (register, deregister, discover, heartbeat)
            processing_time_ms: Time taken to complete the operation in milliseconds
            success: Whether the operation succeeded
            correlation_id: Optional correlation ID for distributed tracing
            node_id: Optional node ID for node-specific operations
            record_count: Optional count of records (for discover operations)
            status: Optional status string (success, partial, failed)
        """
        # Build structured log extra dict
        log_extra: dict[str, str | float | int | bool | None] = {
            "event_type": "registry_operation_complete",
            "operation": operation,
            "processing_time_ms": round(processing_time_ms, 3),
            "correlation_id": str(correlation_id) if correlation_id else None,
            "success": success,
        }

        # Add optional fields only if provided
        if node_id is not None:
            log_extra["node_id"] = node_id
        if record_count is not None:
            log_extra["record_count"] = record_count
        if status is not None:
            log_extra["status"] = status

        # Log the operation completion
        logger.info(
            f"Registry operation completed: {operation}",
            extra=log_extra,
        )

        # Log slow operation warning if threshold exceeded
        if processing_time_ms > self._slow_operation_threshold_ms:
            slow_extra: dict[str, str | float | int | bool | None] = {
                "event_type": "registry_operation_slow",
                "operation": operation,
                "processing_time_ms": round(processing_time_ms, 3),
                "threshold_ms": self._slow_operation_threshold_ms,
                "correlation_id": str(correlation_id) if correlation_id else None,
            }
            if node_id is not None:
                slow_extra["node_id"] = node_id

            logger.warning(
                f"Slow registry operation detected: {operation} took "
                f"{processing_time_ms:.1f}ms (threshold: {self._slow_operation_threshold_ms}ms)",
                extra=slow_extra,
            )

    def _safe_json_dumps(
        self,
        data: JsonValue | dict[str, str],
        correlation_id: UUID | None = None,
        field_name: str = "unknown",
        fallback: str = "{}",
    ) -> str:
        """Safely serialize data to JSON with error handling.

        This method provides robust JSON serialization with:
        - Custom default serializer for non-standard types (datetime, UUID, Enum, etc.)
        - RecursionError handling for deeply nested or circular structures
        - Detailed logging with correlation_id and data type for debugging
        - Fallback value support for graceful degradation

        Args:
            data: The data to serialize (any JSON-serializable value)
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging
            fallback: Value to return on serialization failure (default: "{}")

        Returns:
            JSON string, or fallback value on serialization failure

        See Also:
            _json_default_serializer: Custom serializer for non-standard types
            _safe_json_dumps_strict: Version that reports errors to caller
        """
        try:
            return json.dumps(data, default=self._json_default_serializer)
        except (TypeError, ValueError, RecursionError) as e:
            logger.warning(
                f"JSON serialization failed for {field_name}: {type(e).__name__}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
                    "data_type": type(data).__name__,
                },
            )
            return fallback

    def _safe_json_dumps_strict(
        self,
        data: JsonValue | dict[str, str],
        correlation_id: UUID | None = None,
        field_name: str = "unknown",
    ) -> tuple[str, str | None]:
        """Safely serialize data to JSON with strict error reporting.

        Unlike _safe_json_dumps, this method returns both the result and any error
        that occurred, allowing callers to decide how to handle serialization failures.

        This method is used for critical serialization paths (like event publishing)
        where the caller needs to know if serialization failed and handle it explicitly.

        Features:
        - Custom default serializer for non-standard types (datetime, UUID, Enum, etc.)
        - RecursionError handling for deeply nested or circular structures
        - Returns error message to caller instead of silently falling back
        - Detailed logging with correlation_id and data type for debugging

        Args:
            data: The data to serialize
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging

        Returns:
            Tuple of (json_string, error_message). If successful, error_message is None.
            If failed, json_string is "{}" and error_message describes the failure.

        See Also:
            _json_default_serializer: Custom serializer for non-standard types
            _safe_json_dumps: Version that returns fallback on error
        """
        try:
            return json.dumps(data, default=self._json_default_serializer), None
        except (TypeError, ValueError, RecursionError) as e:
            error_msg = (
                f"JSON serialization failed for {field_name}: {type(e).__name__}"
            )
            logger.warning(
                error_msg,
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
                    "data_type": type(data).__name__,
                },
            )
            return "{}", error_msg

    async def initialize(self) -> None:
        """Initialize the effect node and verify backend connectivity."""
        self._initialized = True
        logger.info("NodeRegistryEffect initialized")

    async def shutdown(self) -> None:
        """Shutdown the effect node and cleanup resources."""
        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()
        self._initialized = False
        logger.info("NodeRegistryEffect shutdown")

    async def execute(self, request: ModelRegistryRequest) -> ModelRegistryResponse:
        """Execute registry operation from request.

        This method routes to the appropriate operation handler based on
        request.operation: register, deregister, discover, or request_introspection.

        Circuit Breaker Behavior:
            This method is protected by MixinAsyncCircuitBreaker. The circuit breaker
            monitors consecutive failures and transitions through three states:

            - CLOSED (normal): Requests proceed normally. Failures increment the
              counter. Success resets the counter.
            - OPEN (blocking): After threshold consecutive failures, the circuit
              opens. All requests fail fast with InfraUnavailableError until
              reset_timeout expires. No backend calls are made.
            - HALF_OPEN (testing): After reset_timeout, the next request is allowed
              through as a test. Success closes the circuit; failure reopens it.

            Configuration (via ModelNodeRegistryEffectConfig):
            - circuit_breaker_threshold: Failures before opening (default: 5)
            - circuit_breaker_reset_timeout: Seconds until auto-reset (default: 60.0)

            Success Criteria:
            - Full success (status="success"): Both backends succeeded, resets circuit
            - Partial success (status="partial"): One backend succeeded, resets circuit
            - Full failure (status="failed"): Both backends failed, does NOT reset circuit

        Retry Guidance for Callers:
            - DO NOT retry when InfraUnavailableError is raised (circuit is open).
              The circuit breaker is protecting degraded backends; retrying wastes
              resources and delays recovery.
            - Check error context for retry_after_seconds which indicates when the
              circuit will attempt auto-reset.
            - For transient failures (status="failed" but circuit not open), implement
              exponential backoff at the caller level (e.g., 1s, 2s, 4s delays).
            - Partial success (status="partial") indicates one backend is healthy;
              the unhealthy backend may recover on subsequent calls.

        Error Recovery Strategies:
            - InfraUnavailableError: Wait for retry_after_seconds, then retry once.
              If still failing, escalate or use fallback data source.
            - RuntimeHostError: Do not retry; fix the request (missing fields,
              invalid operation, not initialized).
            - status="partial": Log warning, consider the operation successful for
              the healthy backend. The failed backend will be retried on next call.
            - status="failed": Implement exponential backoff, consider circuit
              breaker may open after threshold failures.

        Args:
            request: Registry request containing:
                - operation: One of "register", "deregister", "discover",
                  "request_introspection"
                - correlation_id: UUID for distributed tracing
                - introspection_event: Required for "register" operation
                - node_id: Required for "deregister" operation
                - filters: Optional dict for "discover" operation

        Returns:
            ModelRegistryResponse with:
                - operation: Echo of requested operation
                - success: True if at least one backend succeeded
                - status: "success" | "partial" | "failed"
                - consul_result: Result from Consul backend (if applicable)
                - postgres_result: Result from PostgreSQL backend (if applicable)
                - nodes: List of discovered nodes (for "discover" operation)
                - processing_time_ms: Operation duration in milliseconds
                - correlation_id: Echo of request correlation_id
                - error: Sanitized error message (if status="failed")

        Raises:
            RuntimeHostError: If not initialized, invalid request, or missing
                required fields for the operation.
            InfraUnavailableError: If circuit breaker is open. Check
                error.model.context for retry_after_seconds.

        Example:
            ```python
            try:
                response = await registry_effect.execute(request)
                if response.status == "partial":
                    logger.warning("Partial success", extra={...})
            except InfraUnavailableError as e:
                # Circuit is open - do not retry immediately
                retry_after = e.model.context.get("retry_after_seconds", 60)
                logger.error(f"Service unavailable, retry after {retry_after}s")
            ```

        See Also:
            - MixinAsyncCircuitBreaker: Circuit breaker implementation
            - ModelNodeRegistryEffectConfig: Configuration options
            - docs/patterns/circuit_breaker_implementation.md: Detailed patterns
        """
        if not self._initialized:
            raise RuntimeHostError(
                "NodeRegistryEffect not initialized",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="execute",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation=request.operation,
                correlation_id=request.correlation_id,
            )

        start_time = time.perf_counter()

        try:
            if request.operation == "register":
                return await self._register_node(request, start_time)
            elif request.operation == "deregister":
                return await self._deregister_node(request, start_time)
            elif request.operation == "discover":
                return await self._discover_nodes(request, start_time)
            elif request.operation == "request_introspection":
                return await self._request_introspection(request, start_time)
            else:
                raise RuntimeHostError(
                    f"Unknown operation: {request.operation}",
                    context=ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation=request.operation,
                        target_name="node_registry_effect",
                        correlation_id=request.correlation_id,
                    ),
                )
        except (RuntimeHostError, InfraUnavailableError):
            # Re-raise our own errors without circuit breaker recording
            raise
        except Exception:
            # Record failure for circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation=request.operation,
                    correlation_id=request.correlation_id,
                )
            raise

    async def _register_node(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Register node with Consul and PostgreSQL in parallel."""
        if request.introspection_event is None:
            raise RuntimeHostError(
                "introspection_event required for register operation",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="register",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        introspection = request.introspection_event

        # Execute dual registration in parallel
        consul_task = asyncio.create_task(
            self._register_consul(introspection, request.correlation_id)
        )
        postgres_task = asyncio.create_task(
            self._register_postgres(introspection, request.correlation_id)
        )

        consul_result, postgres_result = await asyncio.gather(
            consul_task,
            postgres_task,
            return_exceptions=True,
        )

        # Process results
        consul_op_result = self._process_consul_result(consul_result)
        postgres_op_result = self._process_postgres_result(postgres_result)

        # Determine overall status
        both_success = consul_op_result.success and postgres_op_result.success
        any_success = consul_op_result.success or postgres_op_result.success

        if both_success:
            status: RegistryStatus = "success"
            success = True
        elif any_success:
            status = "partial"
            success = True  # Partial success is still success
        else:
            status = "failed"
            success = False

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Reset circuit breaker on success
        if success:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        # Log structured performance metrics
        self._log_operation_performance(
            operation="register",
            processing_time_ms=processing_time_ms,
            success=success,
            correlation_id=request.correlation_id,
            node_id=introspection.node_id,
            status=status,
        )

        return ModelRegistryResponse(
            operation="register",
            success=success,
            status=status,
            consul_result=consul_op_result,
            postgres_result=postgres_op_result,
            processing_time_ms=processing_time_ms,
            correlation_id=request.correlation_id,
        )

    async def _register_consul(
        self,
        introspection: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> ModelConsulOperationResult:
        """Register node with Consul."""
        try:
            # Build Consul registration payload
            consul_payload: dict[str, JsonValue] = {
                "name": introspection.node_id,
                "service_id": introspection.node_id,
                "tags": [introspection.node_type, f"v{introspection.node_version}"],
            }

            # Add health check if endpoint provided
            if introspection.health_endpoint:
                consul_payload["check"] = {
                    "http": introspection.health_endpoint,
                    "interval": "30s",
                    "timeout": "10s",
                }

            # Execute via consul handler (Protocol-typed)
            result = await self._consul_handler.execute(
                {
                    "operation": "consul.register",
                    "payload": consul_payload,
                    "correlation_id": correlation_id,
                }
            )

            return ModelConsulOperationResult(
                success=result.status == "success",
                service_id=introspection.node_id,
            )
        except Exception as e:
            logger.warning(
                f"Consul registration failed: {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )

    async def _register_postgres(
        self,
        introspection: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> ModelPostgresOperationResult:
        """Register node in PostgreSQL with UPSERT."""
        try:
            # Build UPSERT query for node_registrations table
            upsert_sql = """
                INSERT INTO node_registrations (
                    node_id, node_type, node_version, capabilities,
                    endpoints, metadata, health_endpoint, registered_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
                ON CONFLICT (node_id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    node_version = EXCLUDED.node_version,
                    capabilities = EXCLUDED.capabilities,
                    endpoints = EXCLUDED.endpoints,
                    metadata = EXCLUDED.metadata,
                    health_endpoint = EXCLUDED.health_endpoint,
                    updated_at = NOW()
            """

            # Execute via db handler (Protocol-typed)
            # Use _safe_model_dump to handle Pydantic serialization errors
            # before passing to _safe_json_dumps for JSON encoding.
            # This provides two layers of error handling:
            # 1. _safe_model_dump catches Pydantic model_dump() failures
            # 2. _safe_json_dumps catches json.dumps() failures
            result = await self._db_handler.execute(
                {
                    "operation": "db.execute",
                    "payload": {
                        "sql": upsert_sql,
                        "params": [
                            introspection.node_id,
                            introspection.node_type,
                            introspection.node_version,
                            self._safe_json_dumps(
                                self._safe_model_dump(
                                    introspection.capabilities,
                                    correlation_id,
                                    "capabilities",
                                ),
                                correlation_id,
                                "capabilities",
                            ),
                            self._safe_json_dumps(
                                introspection.endpoints, correlation_id, "endpoints"
                            ),
                            self._safe_json_dumps(
                                self._safe_model_dump(
                                    introspection.runtime_metadata,
                                    correlation_id,
                                    "runtime_metadata",
                                ),
                                correlation_id,
                                "runtime_metadata",
                            ),
                            introspection.health_endpoint,
                        ],
                    },
                    "correlation_id": correlation_id,
                }
            )

            # ModelDbQueryResponse has .payload.row_count for affected rows
            rows_affected = (
                result.payload.row_count if result.payload.row_count > 0 else 1
            )

            return ModelPostgresOperationResult(
                success=result.status == "success",
                rows_affected=rows_affected,
            )
        except Exception as e:
            logger.warning(
                f"PostgreSQL registration failed: {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )

    async def _deregister_node(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Deregister node from Consul and PostgreSQL in parallel."""
        if not request.node_id:
            raise RuntimeHostError(
                "node_id required for deregister operation",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="deregister",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        # Execute dual deregistration in parallel
        consul_task = asyncio.create_task(
            self._deregister_consul(request.node_id, request.correlation_id)
        )
        postgres_task = asyncio.create_task(
            self._deregister_postgres(request.node_id, request.correlation_id)
        )

        consul_result, postgres_result = await asyncio.gather(
            consul_task,
            postgres_task,
            return_exceptions=True,
        )

        consul_op_result = self._process_consul_result(consul_result)
        postgres_op_result = self._process_postgres_result(postgres_result)

        both_success = consul_op_result.success and postgres_op_result.success
        any_success = consul_op_result.success or postgres_op_result.success

        if both_success:
            status: RegistryStatus = "success"
            success = True
        elif any_success:
            status = "partial"
            success = True
        else:
            status = "failed"
            success = False

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        if success:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        # Log structured performance metrics
        self._log_operation_performance(
            operation="deregister",
            processing_time_ms=processing_time_ms,
            success=success,
            correlation_id=request.correlation_id,
            node_id=request.node_id,
            status=status,
        )

        return ModelRegistryResponse(
            operation="deregister",
            success=success,
            status=status,
            consul_result=consul_op_result,
            postgres_result=postgres_op_result,
            processing_time_ms=processing_time_ms,
            correlation_id=request.correlation_id,
        )

    async def _deregister_consul(
        self,
        node_id: str,
        correlation_id: UUID,
    ) -> ModelConsulOperationResult:
        """Deregister node from Consul."""
        try:
            # Execute via consul handler (Protocol-typed)
            result = await self._consul_handler.execute(
                {
                    "operation": "consul.deregister",
                    "payload": {"service_id": node_id},
                    "correlation_id": correlation_id,
                }
            )
            return ModelConsulOperationResult(
                success=result.status == "success",
                service_id=node_id,
            )
        except Exception as e:
            logger.warning(
                f"Consul deregistration failed: {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelConsulOperationResult(
                success=False, error=self._sanitize_error(e)
            )

    async def _deregister_postgres(
        self,
        node_id: str,
        correlation_id: UUID,
    ) -> ModelPostgresOperationResult:
        """Delete node from PostgreSQL."""
        try:
            # Execute via db handler (Protocol-typed)
            result = await self._db_handler.execute(
                {
                    "operation": "db.execute",
                    "payload": {
                        "sql": "DELETE FROM node_registrations WHERE node_id = $1",
                        "params": [node_id],
                    },
                    "correlation_id": correlation_id,
                }
            )
            # ModelDbQueryResponse has .payload.row_count for affected rows
            rows_affected = result.payload.row_count

            return ModelPostgresOperationResult(
                success=result.status == "success",
                rows_affected=rows_affected,
            )
        except Exception as e:
            logger.warning(
                f"PostgreSQL deregistration failed: {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                },
            )
            return ModelPostgresOperationResult(
                success=False, error=self._sanitize_error(e)
            )

    def _validate_filter_keys(
        self,
        filters: dict[str, str],
        correlation_id: UUID,
    ) -> list[str]:
        """Validate filter keys against the whitelist.

        SECURITY: This method prevents SQL injection by ensuring only known-safe
        column names can be used in SQL queries. Invalid keys are rejected, not
        silently ignored, to prevent attackers from probing for vulnerabilities.

        Args:
            filters: Dictionary of filter key-value pairs from the request.
            correlation_id: Correlation ID for security logging.

        Returns:
            List of invalid filter keys (empty if all keys are valid).

        Security Note:
            Invalid filter keys are logged at WARNING level for security monitoring.
            The log includes a sanitized version of the key (truncated, special chars
            removed) to prevent log injection attacks while still enabling detection
            of SQL injection attempts.
        """
        invalid_keys: list[str] = []

        for key in filters:
            if key not in ALLOWED_FILTER_KEYS:
                invalid_keys.append(key)
                # Log security event with sanitized key to prevent log injection
                # Truncate and remove special characters for safe logging
                sanitized_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", key[:50])
                logger.warning(
                    "Invalid filter key rejected in discover operation",
                    extra={
                        "event_type": "security_filter_key_rejected",
                        "correlation_id": str(correlation_id),
                        "sanitized_key": sanitized_key,
                        "key_length": len(key),
                        "allowed_keys": list(ALLOWED_FILTER_KEYS),
                    },
                )

        return invalid_keys

    async def _discover_nodes(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Query registered nodes from PostgreSQL with optional filters.

        Security:
            Filter keys are validated against ALLOWED_FILTER_KEYS whitelist.
            Invalid filter keys cause the request to be rejected with a
            RuntimeHostError, preventing SQL injection attempts. Filter values
            are always parameterized (never interpolated into SQL).
        """
        # Validate filter keys BEFORE building SQL query (SQL injection prevention)
        if request.filters:
            invalid_keys = self._validate_filter_keys(
                request.filters, request.correlation_id
            )
            if invalid_keys:
                # Reject request with invalid filter keys - do not silently ignore
                # Sanitize invalid keys for error message (no SQL structure leaked)
                sanitized_invalid = [
                    re.sub(r"[^a-zA-Z0-9_\-]", "_", k[:30]) for k in invalid_keys[:5]
                ]
                raise RuntimeHostError(
                    f"Invalid filter keys: {sanitized_invalid}. "
                    f"Allowed keys: {sorted(ALLOWED_FILTER_KEYS)}",
                    context=ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.DATABASE,
                        operation="discover",
                        target_name="node_registry_effect",
                        correlation_id=request.correlation_id,
                    ),
                )

        try:
            # Build query with validated filters
            sql = "SELECT * FROM node_registrations"
            params: list[str] = []

            if request.filters:
                conditions = []
                param_idx = 1
                for key, value in request.filters.items():
                    # Keys already validated above - safe to interpolate column names
                    conditions.append(f"{key} = ${param_idx}")
                    params.append(value)
                    param_idx += 1
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

            # Execute via db handler (Protocol-typed)
            result = await self._db_handler.execute(
                {
                    "operation": "db.query",
                    "payload": {"sql": sql, "params": params},
                    "correlation_id": request.correlation_id,
                }
            )

            # Parse results into ModelNodeRegistration
            # ModelDbQueryResponse has .payload.rows as list[dict[str, object]]
            rows = cast(list[dict[str, JsonValue]], result.payload.rows)
            nodes = [
                self._row_to_node_registration(row, request.correlation_id)
                for row in rows
            ]

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Log structured performance metrics for successful discovery
            self._log_operation_performance(
                operation="discover",
                processing_time_ms=processing_time_ms,
                success=True,
                correlation_id=request.correlation_id,
                record_count=len(nodes),
                status="success",
            )

            return ModelRegistryResponse(
                operation="discover",
                success=True,
                status="success",
                nodes=nodes,
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except Exception as e:
            # Record circuit breaker failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("discover", request.correlation_id)

            logger.warning(
                f"Node discovery failed: {type(e).__name__}",
                extra={
                    "filters": request.filters,
                    "correlation_id": str(request.correlation_id),
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for failed discovery
            self._log_operation_performance(
                operation="discover",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                record_count=0,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="discover",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )

    async def _request_introspection(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Publish introspection request to event bus."""
        if self._event_bus is None:
            raise RuntimeHostError(
                "Event bus not configured for request_introspection",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="request_introspection",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        try:
            # Publish REQUEST_INTROSPECTION event (Protocol-typed)
            event_payload = {
                "event_type": "REGISTRY_REQUEST_INTROSPECTION",
                "correlation_id": str(request.correlation_id),
            }

            # Use strict serialization - don't publish malformed events
            json_payload, serialization_error = self._safe_json_dumps_strict(
                event_payload, request.correlation_id, "introspection_event"
            )
            if serialization_error:
                processing_time_ms = (time.perf_counter() - start_time) * 1000

                # Log structured performance metrics for serialization failure
                self._log_operation_performance(
                    operation="request_introspection",
                    processing_time_ms=processing_time_ms,
                    success=False,
                    correlation_id=request.correlation_id,
                    status="failed",
                )

                return ModelRegistryResponse(
                    operation="request_introspection",
                    success=False,
                    status="failed",
                    error=serialization_error,
                    processing_time_ms=processing_time_ms,
                    correlation_id=request.correlation_id,
                )

            await self._event_bus.publish(
                topic="onex.evt.registry-request-introspection.v1",
                key=b"registry",
                value=json_payload.encode("utf-8"),
            )

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Log structured performance metrics for successful introspection request
            self._log_operation_performance(
                operation="request_introspection",
                processing_time_ms=processing_time_ms,
                success=True,
                correlation_id=request.correlation_id,
                status="success",
            )

            return ModelRegistryResponse(
                operation="request_introspection",
                success=True,
                status="success",
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except Exception as e:
            logger.warning(
                f"Introspection request failed: {type(e).__name__}",
                extra={
                    "correlation_id": str(request.correlation_id),
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for failed introspection request
            self._log_operation_performance(
                operation="request_introspection",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="request_introspection",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )

    def _process_consul_result(
        self,
        result: ModelConsulOperationResult | BaseException,
    ) -> ModelConsulOperationResult:
        """Process consul task result, handling exceptions."""
        if isinstance(result, BaseException):
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(result),
            )
        return result

    def _process_postgres_result(
        self,
        result: ModelPostgresOperationResult | BaseException,
    ) -> ModelPostgresOperationResult:
        """Process postgres task result, handling exceptions."""
        if isinstance(result, BaseException):
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(result),
            )
        return result

    def _row_to_node_registration(
        self,
        row: dict[str, JsonValue],
        correlation_id: UUID | None = None,
    ) -> ModelNodeRegistration:
        """Convert database row to ModelNodeRegistration.

        Args:
            row: Database row dictionary with JSON-serializable values
            correlation_id: Optional correlation ID for logging
        """

        def parse_json(
            val: JsonValue, field_name: str = "unknown"
        ) -> dict[str, JsonValue]:
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict):
                        return cast(dict[str, JsonValue], parsed)
                    logger.warning(
                        f"JSON parse result not a dict for {field_name}",
                        extra={
                            "correlation_id": (
                                str(correlation_id) if correlation_id else None
                            ),
                            "field_name": field_name,
                        },
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSON parse failed for {field_name}: {type(e).__name__}",
                        extra={
                            "correlation_id": (
                                str(correlation_id) if correlation_id else None
                            ),
                            "field_name": field_name,
                            "error_type": type(e).__name__,
                        },
                    )
            return {}

        def parse_datetime(val: JsonValue, field_name: str = "datetime") -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            # Fallback to current time when no valid datetime provided
            logger.warning(
                f"Using datetime fallback for {field_name}",
                extra={
                    "correlation_id": (str(correlation_id) if correlation_id else None),
                    "field_name": field_name,
                },
            )
            return datetime.now(UTC)

        # Handle health_endpoint which can be str or None
        health_endpoint_raw = row.get("health_endpoint")
        health_endpoint: str | None = None
        if isinstance(health_endpoint_raw, str) and health_endpoint_raw:
            health_endpoint = health_endpoint_raw

        # Handle last_heartbeat which can be datetime, str, or None
        last_heartbeat_raw = row.get("last_heartbeat")
        last_heartbeat: datetime | None = None
        if last_heartbeat_raw is not None:
            last_heartbeat = parse_datetime(last_heartbeat_raw, "last_heartbeat")

        # Parse timestamps - use current time if missing (shouldn't happen in valid data)
        registered_at_raw = row.get("registered_at")
        registered_at = (
            parse_datetime(registered_at_raw, "registered_at")
            if registered_at_raw is not None
            else datetime.now(UTC)
        )

        updated_at_raw = row.get("updated_at")
        updated_at = (
            parse_datetime(updated_at_raw, "updated_at")
            if updated_at_raw is not None
            else datetime.now(UTC)
        )

        # Convert endpoints dict to proper type (values must be strings)
        raw_endpoints = parse_json(row.get("endpoints", {}), "endpoints")
        endpoints: dict[str, str] = {
            str(k): str(v) for k, v in raw_endpoints.items() if isinstance(v, str)
        }

        # Parse capabilities from database and convert to ModelNodeCapabilitiesInfo
        raw_capabilities = parse_json(row.get("capabilities", {}), "capabilities")
        capabilities = ModelNodeCapabilitiesInfo(
            capabilities=raw_capabilities.get("capabilities", [])
            if isinstance(raw_capabilities.get("capabilities"), list)
            else [],
            supported_operations=raw_capabilities.get("supported_operations", [])
            if isinstance(raw_capabilities.get("supported_operations"), list)
            else [],
        )

        # Parse runtime_metadata from database and convert to ModelNodeRegistrationMetadata
        # Database column is 'metadata', but model field is 'runtime_metadata'
        raw_metadata = parse_json(row.get("metadata", {}), "metadata")
        env_str = raw_metadata.get("environment", "testing")
        try:
            environment = (
                EnumEnvironment(env_str)
                if isinstance(env_str, str)
                else EnumEnvironment.TESTING
            )
        except ValueError:
            environment = EnumEnvironment.TESTING
        runtime_metadata = ModelNodeRegistrationMetadata(
            environment=environment,
            tags=raw_metadata.get("tags", [])
            if isinstance(raw_metadata.get("tags"), list)
            else [],
            labels=raw_metadata.get("labels", {})
            if isinstance(raw_metadata.get("labels"), dict)
            else {},
            release_channel=raw_metadata.get("release_channel")
            if isinstance(raw_metadata.get("release_channel"), str)
            else None,
            region=raw_metadata.get("region")
            if isinstance(raw_metadata.get("region"), str)
            else None,
        )

        return ModelNodeRegistration(
            node_id=str(row.get("node_id", "")),
            node_type=str(row.get("node_type", "")),
            node_version=str(row.get("node_version", "1.0.0")),
            capabilities=capabilities,
            endpoints=endpoints,
            runtime_metadata=runtime_metadata,
            health_endpoint=health_endpoint,
            last_heartbeat=last_heartbeat,
            registered_at=registered_at,
            updated_at=updated_at,
        )


__all__ = ["NodeRegistryEffect"]
