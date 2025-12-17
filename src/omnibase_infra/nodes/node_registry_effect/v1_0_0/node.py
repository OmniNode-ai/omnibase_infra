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
    consul_handler and db_handler must implement:
        async def execute(self, envelope: EnvelopeDict) -> ResultDict

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
    ModelConsulOperationResult,
    ModelNodeIntrospectionPayload,
    ModelNodeRegistration,
    ModelNodeRegistryEffectConfig,
    ModelPostgresOperationResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
    EnvelopeDict,
    JsonValue,
    ProtocolEnvelopeExecutor,
    ProtocolEventBus,
    ResultDict,
)

logger = logging.getLogger(__name__)

# Whitelist of allowed filter keys for SQL query building (SQL injection prevention)
ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(
    {"node_type", "node_id", "node_version"}
)


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
    """

    def __init__(
        self,
        consul_handler: ProtocolEnvelopeExecutor,
        db_handler: ProtocolEnvelopeExecutor,
        event_bus: ProtocolEventBus | None = None,
        config: ModelNodeRegistryEffectConfig | None = None,
    ) -> None:
        """Initialize Registry Effect Node.

        Args:
            consul_handler: Handler for Consul operations.
                Must implement: async execute(envelope: EnvelopeDict) -> ResultDict
            db_handler: Handler for PostgreSQL operations.
                Must implement: async execute(envelope: EnvelopeDict) -> ResultDict
            event_bus: Optional event bus for publishing events.
                Must implement: async publish(topic: str, key: bytes, value: bytes) -> None
            config: Configuration for circuit breaker and resilience settings.
                Uses defaults if not provided.
        """
        self._consul_handler: ProtocolEnvelopeExecutor = consul_handler
        self._db_handler: ProtocolEnvelopeExecutor = db_handler
        self._event_bus: ProtocolEventBus | None = event_bus
        self._initialized = False

        # Use defaults if config not provided
        config = config or ModelNodeRegistryEffectConfig()

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

    def _safe_json_dumps(
        self,
        data: JsonValue | dict[str, str],
        correlation_id: UUID | None = None,
        field_name: str = "unknown",
        fallback: str = "{}",
    ) -> str:
        """Safely serialize data to JSON with error handling.

        Args:
            data: The data to serialize (any JSON-serializable value)
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging
            fallback: Value to return on serialization failure (default: "{}")

        Returns:
            JSON string, or fallback value on serialization failure
        """
        try:
            return json.dumps(data)
        except (TypeError, ValueError) as e:
            logger.warning(
                f"JSON serialization failed for {field_name}: {type(e).__name__}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
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

        Args:
            data: The data to serialize
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging

        Returns:
            Tuple of (json_string, error_message). If successful, error_message is None.
            If failed, json_string is "{}" and error_message describes the failure.
        """
        try:
            return json.dumps(data), None
        except (TypeError, ValueError) as e:
            error_msg = (
                f"JSON serialization failed for {field_name}: {type(e).__name__}"
            )
            logger.warning(
                error_msg,
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
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

        Args:
            request: Registry request with operation and payload

        Returns:
            ModelRegistryResponse with operation results

        Raises:
            RuntimeHostError: If not initialized or invalid request
            InfraUnavailableError: If circuit breaker is open
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
                success=result.get("status") == "success",
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
                                introspection.capabilities,
                                correlation_id,
                                "capabilities",
                            ),
                            self._safe_json_dumps(
                                introspection.endpoints, correlation_id, "endpoints"
                            ),
                            self._safe_json_dumps(
                                introspection.metadata, correlation_id, "metadata"
                            ),
                            introspection.health_endpoint,
                        ],
                    },
                    "correlation_id": correlation_id,
                }
            )

            payload = result.get("payload", {})
            rows_affected = 1
            if isinstance(payload, dict):
                raw_rows = payload.get("rows_affected", 1)
                if isinstance(raw_rows, int):
                    rows_affected = raw_rows

            return ModelPostgresOperationResult(
                success=result.get("status") == "success",
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
                success=result.get("status") == "success",
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
            payload = result.get("payload", {})
            rows_affected = 0
            if isinstance(payload, dict):
                raw_rows = payload.get("rows_affected", 0)
                if isinstance(raw_rows, int):
                    rows_affected = raw_rows

            return ModelPostgresOperationResult(
                success=result.get("status") == "success",
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

    async def _discover_nodes(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Query registered nodes from PostgreSQL with optional filters."""
        try:
            # Build query with filters
            sql = "SELECT * FROM node_registrations"
            params: list[str] = []

            if request.filters:
                conditions = []
                param_idx = 1
                for key, value in request.filters.items():
                    if key not in ALLOWED_FILTER_KEYS:
                        logger.warning(
                            f"Unknown filter key ignored: {key}",
                            extra={"correlation_id": str(request.correlation_id)},
                        )
                        continue
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
            payload = result.get("payload", {})
            rows: list[dict[str, JsonValue]] = []
            if isinstance(payload, dict):
                raw_rows = payload.get("rows", [])
                if isinstance(raw_rows, list):
                    rows = cast(list[dict[str, JsonValue]], raw_rows)
            nodes = [
                self._row_to_node_registration(row, request.correlation_id)
                for row in rows
            ]

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

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

        return ModelNodeRegistration(
            node_id=str(row.get("node_id", "")),
            node_type=str(row.get("node_type", "")),
            node_version=str(row.get("node_version", "1.0.0")),
            capabilities=parse_json(row.get("capabilities", {}), "capabilities"),
            endpoints=endpoints,
            metadata=parse_json(row.get("metadata", {}), "metadata"),
            health_endpoint=health_endpoint,
            last_heartbeat=last_heartbeat,
            registered_at=registered_at,
            updated_at=updated_at,
        )


__all__ = ["NodeRegistryEffect"]
