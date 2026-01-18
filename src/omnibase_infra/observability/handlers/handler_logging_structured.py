# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Structured Logging Handler - EFFECT handler for structured logging with buffer/flush.

This handler implements the ONEX EFFECT handler pattern for structured logging.
It manages the lifecycle of a SinkLoggingStructured instance, providing:
- Contract-driven lifecycle management (initialize, shutdown)
- Periodic background flush with configurable interval
- Buffer threshold flush (when buffer reaches capacity)
- Graceful shutdown with final flush

Architecture Principle: "Handlers own lifecycle, sinks own hot path"
- This handler: Contract-driven lifecycle, buffer/flush management
- SinkLoggingStructured: Fast in-process emission (synchronous, non-blocking)

Supported Operations:
    - logging.emit: Emit a log entry to the buffer (delegates to sink)
    - logging.flush: Force flush all buffered entries immediately
    - logging.configure: Update logging configuration at runtime

Thread Safety:
    The handler uses asyncio locks for configuration changes and flush coordination.
    The underlying sink uses threading locks for buffer operations. This dual-lock
    design allows the sink to be safely used from both async and sync contexts.

Envelope-Based Routing:
    This handler uses envelope-based operation routing. See CLAUDE.md section
    "Intent Model Architecture > Envelope-Based Handler Routing" for the full
    design pattern.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
    EnumResponseStatus,
)
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinEnvelopeExtraction
from omnibase_infra.observability.handlers.model_logging_handler_config import (
    ModelLoggingHandlerConfig,
)
from omnibase_infra.observability.handlers.model_logging_handler_response import (
    ModelLoggingHandlerResponse,
)
from omnibase_infra.observability.sinks import SinkLoggingStructured

if TYPE_CHECKING:
    from omnibase_core.enums import EnumLogLevel

logger = logging.getLogger(__name__)

# Handler ID for responses
HANDLER_ID_LOGGING: str = "logging-handler"

# Supported operations
SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "logging.emit",
        "logging.flush",
        "logging.configure",
    }
)


class HandlerLoggingStructured(MixinEnvelopeExtraction):
    """Structured logging EFFECT handler with buffer/flush management.

    This handler manages the lifecycle of a SinkLoggingStructured instance,
    providing contract-driven initialization, periodic background flush,
    and graceful shutdown with final flush.

    Lifecycle:
        1. initialize(): Creates sink, starts periodic flush task
        2. execute(): Routes operations to sink methods
        3. shutdown(): Cancels flush task, performs final flush

    Buffer Management:
        - The sink maintains a bounded buffer with configurable size
        - Periodic flush: Background task flushes at configured interval
        - Threshold flush: Automatic flush when buffer nears capacity
        - Shutdown flush: Final flush ensures no data loss

    Thread Safety:
        - Configuration changes use asyncio.Lock
        - Sink operations use threading.Lock (internal to sink)
        - Safe for concurrent async callers

    Example:
        ```python
        handler = HandlerLoggingStructured()
        await handler.initialize({
            "buffer_size": 500,
            "flush_interval_seconds": 10.0,
            "output_format": "json",
        })

        # Emit log entries
        await handler.execute({
            "operation": "logging.emit",
            "payload": {
                "level": "INFO",
                "message": "User logged in",
                "context": {"user_id": "u_123"},
            },
        })

        # Force flush
        await handler.execute({
            "operation": "logging.flush",
            "payload": {},
        })

        await handler.shutdown()
        ```
    """

    def __init__(self) -> None:
        """Initialize handler in uninitialized state."""
        self._sink: SinkLoggingStructured | None = None
        self._config: ModelLoggingHandlerConfig | None = None
        self._initialized: bool = False
        self._flush_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._config_lock: asyncio.Lock = asyncio.Lock()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - This handler is an infrastructure
            handler that manages the logging sink lifecycle. It is not a
            NODE_HANDLER (event processing) or COMPUTE_HANDLER (pure computation).

        Note:
            handler_type determines lifecycle, protocol selection, and runtime
            invocation patterns. It answers "what is this handler in the architecture?"

        See Also:
            - handler_category: Behavioral classification (EFFECT/COMPUTE)
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - This handler performs side-effecting
            I/O operations (writing logs to stdout/files). EFFECT handlers are
            not deterministic and interact with external systems.

        Note:
            handler_category determines security rules, determinism guarantees,
            replay safety, and permissions. It answers "how does this handler
            behave at runtime?"

            Categories:
            - COMPUTE: Pure, deterministic transformations (no side effects)
            - EFFECT: Side-effecting I/O (database, HTTP, file writes, logging)
            - NONDETERMINISTIC_COMPUTE: Pure but not deterministic (UUID, random)

        See Also:
            - handler_type: Architectural role (INFRA_HANDLER/NODE_HANDLER/etc.)
            - docs/architecture/HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md
        """
        return EnumHandlerTypeCategory.EFFECT

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the logging handler with configuration.

        Creates the SinkLoggingStructured instance and starts the periodic
        flush background task if configured.

        Args:
            config: Configuration dict containing:
                - buffer_size: Max buffer size (default: 1000)
                - flush_interval_seconds: Flush interval (default: 5.0, 0 to disable)
                - output_format: "json" or "console" (default: "json")
                - output_file: Optional file path for output
                - drop_policy: "drop_oldest" (only supported policy)

        Raises:
            ProtocolConfigurationError: If configuration validation fails.
            RuntimeHostError: If handler is already initialized.
        """
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        if self._initialized:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="initialize",
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Handler already initialized. Call shutdown() first.",
                context=ctx,
            )

        # Validate and parse configuration
        # NOTE: Broad Exception catch is intentional here because Pydantic can raise
        # various exception types (ValidationError, TypeError, ValueError) depending
        # on the validation failure. We wrap all in ProtocolConfigurationError.
        try:
            self._config = ModelLoggingHandlerConfig.model_validate(config)
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=init_correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="initialize",
                target_name="logging_handler",
            )
            raise ProtocolConfigurationError(
                f"Invalid logging handler configuration: {e}",
                context=ctx,
            ) from e

        # Create the sink with configured drop policy
        self._sink = SinkLoggingStructured(
            max_buffer_size=self._config.buffer_size,
            output_format=self._config.output_format,
            drop_policy=self._config.drop_policy,
        )

        # Set up shutdown event and start periodic flush task
        self._shutdown_event = asyncio.Event()

        if self._config.flush_interval_seconds > 0:
            self._flush_task = asyncio.create_task(
                self._periodic_flush_loop(),
                name="logging-handler-flush",
            )

        self._initialized = True

        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "buffer_size": self._config.buffer_size,
                "flush_interval_seconds": self._config.flush_interval_seconds,
                "output_format": self._config.output_format,
                "drop_policy": self._config.drop_policy,
                "correlation_id": str(init_correlation_id),
            },
        )

    async def shutdown(self) -> None:
        """Shutdown the logging handler gracefully.

        Cancels the periodic flush task, performs a final flush of all
        buffered entries, and releases resources.

        This method is idempotent - calling it multiple times is safe.
        """
        shutdown_correlation_id = uuid4()

        logger.info(
            "Shutting down %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(shutdown_correlation_id),
            },
        )

        # Signal shutdown to flush loop
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        # Cancel periodic flush task
        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush - best effort, never fails shutdown
        # NOTE: Broad Exception catch is intentional here to ensure shutdown
        # always completes regardless of I/O errors. Any flush error during
        # shutdown is logged but does not prevent cleanup from completing.
        if self._sink is not None:
            try:
                self._sink.flush()
            except Exception as e:
                logger.warning(
                    "Error during final flush on shutdown: %s",
                    e,
                    extra={
                        "handler": self.__class__.__name__,
                        "correlation_id": str(shutdown_correlation_id),
                        "error_type": type(e).__name__,
                    },
                )

        # Clear state
        self._sink = None
        self._config = None
        self._initialized = False
        self._shutdown_event = None

        logger.info(
            "%s shutdown complete",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(shutdown_correlation_id),
            },
        )

    async def execute(self, envelope: dict[str, object]) -> ModelLoggingHandlerResponse:
        """Execute a logging operation from the envelope.

        Supported operations:
            - logging.emit: Buffer a log entry
            - logging.flush: Force flush all buffered entries
            - logging.configure: Update handler configuration

        Args:
            envelope: Request envelope containing:
                - operation: Operation identifier (logging.emit, etc.)
                - payload: Operation-specific payload
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelLoggingHandlerResponse with operation result

        Raises:
            RuntimeHostError: If handler not initialized or invalid operation.
        """
        correlation_id = self._extract_correlation_id(envelope)

        if not self._initialized or self._sink is None or self._config is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="execute",
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Logging handler not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="execute",
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=ctx,
            )

        if operation not in SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation=operation,
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported. "
                f"Available: {', '.join(sorted(SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation=operation,
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Route to operation handler
        if operation == "logging.emit":
            return self._handle_emit(payload, correlation_id)
        elif operation == "logging.flush":
            return self._handle_flush(correlation_id)
        else:  # logging.configure
            return await self._handle_configure(payload, correlation_id)

    def _handle_emit(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> ModelLoggingHandlerResponse:
        """Handle logging.emit operation.

        Args:
            payload: Payload containing:
                - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, etc.)
                - message: Log message string
                - context: Dict of string key-value context data
            correlation_id: Correlation ID for tracing

        Returns:
            ModelLoggingHandlerResponse with operation result
        """
        if self._sink is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="logging.emit",
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Sink not initialized",
                context=ctx,
            )

        # Extract and validate payload fields
        level_raw = payload.get("level")
        message = payload.get("message")
        context_raw = payload.get("context", {})

        if not isinstance(message, str):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="logging.emit",
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Missing or invalid 'message' in payload - must be string",
                context=ctx,
            )

        if not isinstance(context_raw, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="logging.emit",
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Invalid 'context' in payload - must be dict",
                context=ctx,
            )

        # Convert context values to strings (sink requirement)
        context: dict[str, str] = {str(k): str(v) for k, v in context_raw.items()}

        # Parse log level
        level = self._parse_log_level(level_raw, correlation_id)

        # Emit to sink (synchronous, non-blocking)
        self._sink.emit(level, message, context)

        return ModelLoggingHandlerResponse(
            status=EnumResponseStatus.SUCCESS,
            operation="logging.emit",
            message="Log entry buffered",
            correlation_id=correlation_id,
            buffer_size=self._sink.buffer_size,
            drop_count=self._sink.drop_count,
        )

    def _handle_flush(
        self,
        correlation_id: UUID,
    ) -> ModelLoggingHandlerResponse:
        """Handle logging.flush operation.

        Args:
            correlation_id: Correlation ID for tracing

        Returns:
            ModelLoggingHandlerResponse with operation result
        """
        if self._sink is None:
            ctx = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="logging.flush",
                target_name="logging_handler",
            )
            raise RuntimeHostError(
                "Sink not initialized",
                context=ctx,
            )

        # Capture pre-flush metrics
        pre_buffer_size = self._sink.buffer_size

        # Flush (synchronous I/O)
        self._sink.flush()

        return ModelLoggingHandlerResponse(
            status=EnumResponseStatus.SUCCESS,
            operation="logging.flush",
            message=f"Flushed {pre_buffer_size} entries",
            correlation_id=correlation_id,
            buffer_size=self._sink.buffer_size,
            drop_count=self._sink.drop_count,
        )

    async def _handle_configure(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> ModelLoggingHandlerResponse:
        """Handle logging.configure operation.

        Note: Configuration changes require re-creating the sink, which
        involves flushing the current buffer. This operation acquires
        the config lock to ensure thread safety.

        Args:
            payload: Configuration payload (same fields as ModelLoggingHandlerConfig)
            correlation_id: Correlation ID for tracing

        Returns:
            ModelLoggingHandlerResponse with operation result
        """
        async with self._config_lock:
            if self._sink is None or self._config is None:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="logging.configure",
                    target_name="logging_handler",
                )
                raise RuntimeHostError(
                    "Handler not initialized",
                    context=ctx,
                )

            # Flush current buffer before reconfiguration
            self._sink.flush()

            # Merge existing config with new values
            current_dict = self._config.model_dump()
            current_dict.update(payload)

            # Validate new configuration
            # NOTE: Broad Exception catch is intentional here because Pydantic can raise
            # various exception types (ValidationError, TypeError, ValueError) depending
            # on the validation failure. We wrap all in ProtocolConfigurationError.
            try:
                new_config = ModelLoggingHandlerConfig.model_validate(current_dict)
            except Exception as e:
                ctx = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="logging.configure",
                    target_name="logging_handler",
                )
                raise ProtocolConfigurationError(
                    f"Invalid configuration: {e}",
                    context=ctx,
                ) from e

            # Create new sink with updated config
            new_sink = SinkLoggingStructured(
                max_buffer_size=new_config.buffer_size,
                output_format=new_config.output_format,
                drop_policy=new_config.drop_policy,
            )

            # Swap sink and config
            self._config = new_config
            self._sink = new_sink

            # Handle flush interval changes
            if self._config.flush_interval_seconds > 0:
                # Restart flush task if not running or interval changed
                if self._flush_task is None or self._flush_task.done():
                    self._flush_task = asyncio.create_task(
                        self._periodic_flush_loop(),
                        name="logging-handler-flush",
                    )
            # Cancel flush task if interval is 0
            elif self._flush_task is not None and not self._flush_task.done():
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
                self._flush_task = None

        return ModelLoggingHandlerResponse(
            status=EnumResponseStatus.SUCCESS,
            operation="logging.configure",
            message="Configuration updated",
            correlation_id=correlation_id,
            buffer_size=self._sink.buffer_size if self._sink else 0,
            drop_count=self._sink.drop_count if self._sink else 0,
        )

    def _parse_log_level(
        self,
        level_raw: object,
        correlation_id: UUID,
    ) -> EnumLogLevel:
        """Parse log level from payload.

        Accepts EnumLogLevel enum values or string level names.

        Args:
            level_raw: Raw level value from payload
            correlation_id: Correlation ID for error context

        Returns:
            Parsed EnumLogLevel

        Raises:
            RuntimeHostError: If level is invalid
        """
        # Lazy import to avoid circular dependency
        from omnibase_core.enums import EnumLogLevel

        if isinstance(level_raw, EnumLogLevel):
            return level_raw

        if isinstance(level_raw, str):
            # Try to match by value (case-insensitive)
            level_upper = level_raw.upper()
            for log_level in EnumLogLevel:
                if log_level.value.upper() == level_upper:
                    return log_level
                if log_level.name.upper() == level_upper:
                    return log_level

        # Default to INFO for unknown levels
        if level_raw is None:
            return EnumLogLevel.INFO

        ctx = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="logging.emit",
            target_name="logging_handler",
        )
        raise RuntimeHostError(
            f"Invalid log level: {level_raw!r}. "
            f"Valid levels: {[lvl.value for lvl in EnumLogLevel]}",
            context=ctx,
        )

    async def _periodic_flush_loop(self) -> None:
        """Background task for periodic buffer flush.

        Runs until shutdown_event is set or task is cancelled.
        Handles flush errors gracefully to prevent task crash.
        """
        if self._shutdown_event is None:
            return

        while not self._shutdown_event.is_set():
            try:
                # Wait for flush interval or shutdown signal
                flush_interval = (
                    self._config.flush_interval_seconds
                    if self._config is not None
                    else 5.0
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=flush_interval,
                    )
                    # Shutdown signaled
                    break
                except TimeoutError:
                    # Normal timeout - time to flush
                    pass

                # Perform flush
                if self._sink is not None:
                    self._sink.flush()

            except asyncio.CancelledError:
                # Task cancelled - exit gracefully
                break
            except Exception as e:
                # NOTE: Broad Exception catch is intentional here to keep the
                # background flush loop running despite transient I/O errors.
                # This ensures the handler remains operational even if individual
                # flush operations fail temporarily. The sleep prevents tight loops.
                logger.warning(
                    "Error in periodic flush loop: %s",
                    e,
                    extra={
                        "handler": self.__class__.__name__,
                        "error_type": type(e).__name__,
                    },
                )
                # Brief sleep to prevent tight error loop on persistent failures
                await asyncio.sleep(1.0)

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities for introspection.

        This method exposes the handler's type classification along with its
        operational configuration and capabilities.

        Returns:
            dict containing:
                - handler_type: Architectural role (infra_handler)
                - handler_category: Behavioral classification (effect)
                - supported_operations: List of supported operations
                - initialized: Whether the handler is initialized
                - version: Handler version string
                - config: Current configuration (if initialized)
                - buffer_size: Current buffer size (if initialized)
                - drop_count: Total dropped entries (if initialized)

        See Also:
            - handler_type property: Full documentation of architectural role
            - handler_category property: Full documentation of behavioral classification
        """
        result: dict[str, object] = {
            "handler_type": self.handler_type.value,
            "handler_category": self.handler_category.value,
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "initialized": self._initialized,
            "version": "0.1.0-mvp",
        }

        if self._initialized and self._config is not None:
            result["config"] = {
                "buffer_size": self._config.buffer_size,
                "flush_interval_seconds": self._config.flush_interval_seconds,
                "output_format": self._config.output_format,
                "drop_policy": self._config.drop_policy,
            }

        if self._sink is not None:
            result["buffer_size"] = self._sink.buffer_size
            result["drop_count"] = self._sink.drop_count

        return result


__all__: list[str] = [
    "HandlerLoggingStructured",
    "SUPPORTED_OPERATIONS",
]
