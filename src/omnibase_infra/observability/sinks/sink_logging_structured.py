# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Structured logging sink implementation using structlog.

This module provides a structured logging sink that implements the
ProtocolHotPathLoggingSink protocol. It buffers log entries in memory
and flushes them to structlog with JSON formatting.

Output Format Policy:
    **JSON-Only Recommended for Production**

    The sink supports two output formats controlled by the ``output_format`` parameter:

    - **json** (default): Machine-readable JSON format for production environments.
      Each log line is a complete JSON object suitable for ingestion by log
      aggregators (ELK, Loki, Datadog, etc.). This format guarantees:

      - Valid JSON on every line (parseable by ``json.loads()``)
      - Consistent field ordering for diffing/comparison
      - UTF-8 encoding with proper escaping
      - No control characters or newlines within values

    - **console**: Human-readable colored output for local development only.
      This format is NOT suitable for log aggregation and may produce output
      that is not valid JSON.

    **stdlib Logger Integration**:
    The sink uses structlog's PrintLogger (writes to stdout) by default.
    For integration with Python's stdlib logging:

    1. Configure structlog's stdlib integration at application startup
    2. Use structlog.wrap_logger() with a stdlib LoggerFactory
    3. This sink avoids structlog.configure() to prevent global state conflicts

    Example stdlib integration::

        import logging
        import structlog

        # Configure stdlib logging
        logging.basicConfig(level=logging.INFO)

        # Create sink with JSON output
        sink = SinkLoggingStructured(output_format="json")

Required Context Keys:
    The sink automatically adds the following keys to every log entry (callers
    should NOT include these as they will be overwritten):

    - ``original_timestamp``: ISO-8601 timestamp when emit() was called
    - ``level``: Log level string (added by structlog.stdlib.add_log_level)
    - ``timestamp``: ISO-8601 timestamp when flush() was called (added by TimeStamper)

    **Recommended Context Keys** (callers SHOULD include for observability):

    - ``correlation_id``: UUID for distributed tracing across services
    - ``node_id``: ONEX node identifier (e.g., "node_registration_orchestrator")
    - ``operation``: Current operation name (e.g., "validate_contract")

    Missing recommended keys will trigger a warning log at DEBUG level to help
    identify callers that may benefit from adding tracing context.

Buffer Management:
    The sink maintains a thread-safe buffer of log entries. When the buffer
    reaches capacity, oldest entries are dropped to make room for new ones
    (drop_oldest policy). This ensures the sink never blocks on emit() due
    to buffer fullness.

Thread Safety:
    All buffer operations are protected by a threading.Lock. The emit() method
    acquires the lock briefly to append entries, while flush() acquires the
    lock to copy and clear the buffer before releasing it to perform I/O.
    This design minimizes lock contention in hot paths.

Instance Isolation:
    This implementation uses structlog.wrap_logger() instead of structlog.configure()
    to create instance-specific loggers. This design choice ensures:

    - **No global state modification**: Safe for libraries and multi-tenant apps
    - **Multiple instances can coexist**: Different output formats per instance
    - **No configuration conflicts**: Test environments remain isolated
    - **Thread-safe**: Each instance has its own processor chain

    WARNING: If your application calls structlog.configure() elsewhere, those
    settings may affect the underlying logger behavior. This sink's wrap_logger()
    approach isolates the processor chain but not the base logger configuration.

Fallback Behavior:
    If structlog fails during flush, the sink falls back to writing directly
    to stderr to ensure log entries are never silently lost.
"""

from __future__ import annotations

import json
import sys
import threading
import warnings
from collections import deque
from datetime import UTC, datetime
from typing import Literal

# Dependency validation for structlog with clear error message
_STRUCTLOG_AVAILABLE: bool = False
_STRUCTLOG_IMPORT_ERROR: str | None = None

try:
    import structlog

    _STRUCTLOG_AVAILABLE = True
except ImportError as e:
    _STRUCTLOG_IMPORT_ERROR = str(e)
    # Provide stub for type checking when structlog is not installed
    structlog = None  # type: ignore[assignment]

from omnibase_core.enums import EnumLogLevel
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.observability.models.enum_required_log_context_key import (
    EnumRequiredLogContextKey,
)
from omnibase_infra.observability.models.model_buffered_log_entry import (
    ModelBufferedLogEntry,
)

# Mapping from EnumLogLevel string values to structlog methods
# This is resolved at flush time, not import time, to avoid import issues
_STRUCTLOG_LEVEL_MAP: dict[str, str] = {
    "trace": "debug",  # structlog doesn't have trace, map to debug
    "debug": "debug",
    "info": "info",
    "warning": "warning",
    "error": "error",
    "critical": "critical",
    "fatal": "critical",  # structlog doesn't have fatal, map to critical
    "success": "info",  # success maps to info level
    "unknown": "info",  # unknown defaults to info
}

# Default values for required context keys when not provided
_DEFAULT_CORRELATION_ID = "00000000-0000-0000-0000-000000000000"
_DEFAULT_NODE_ID = "unknown"
_DEFAULT_OPERATION = "unknown"


class SinkLoggingStructured:
    """Structured logging sink implementing ProtocolHotPathLoggingSink.

    This sink buffers log entries in memory and flushes them to structlog
    with JSON formatting. It's designed for hot-path scenarios where
    synchronous logging without blocking is critical.

    Buffer Management:
        - Configurable maximum buffer size (default: 1000 entries)
        - When buffer is full, oldest entries are dropped (drop_oldest policy)
        - Thread-safe: all operations use a lock for synchronization

    Output Formats:
        - json: Machine-readable JSON format (default, recommended for production)
        - console: Human-readable colored console output (development only)

    JSON Output Guarantee:
        When output_format="json", every log line is guaranteed to be valid JSON:
        - Parseable by json.loads()
        - Single-line format (no embedded newlines)
        - UTF-8 encoded with proper escaping
        - Consistent field ordering

    Fallback Behavior:
        If structlog fails during flush, entries are written to stderr
        to prevent silent data loss.

    Attributes:
        max_buffer_size: Maximum number of entries to buffer before dropping.
        output_format: Output format ("json" or "console").
        drop_policy: Buffer overflow policy (only "drop_oldest" supported).
        drop_count: Number of entries dropped due to buffer overflow.

    Example:
        ```python
        from omnibase_core.enums import EnumLogLevel

        # Create sink with custom buffer size (uses drop_oldest policy)
        sink = SinkLoggingStructured(max_buffer_size=500)

        # Hot path - emit without blocking
        for item in large_dataset:
            sink.emit(
                EnumLogLevel.DEBUG,
                f"Processed {item}",
                {
                    "id": str(item.id),
                    "correlation_id": str(correlation_id),  # Recommended
                    "node_id": "data_processor",            # Recommended
                    "operation": "process_batch",           # Recommended
                }
            )

        # Flush when hot path completes
        sink.flush()
        ```

    Thread Safety:
        This implementation is THREAD-SAFE. Multiple threads may call emit()
        and flush() concurrently. The lock is held briefly during emit() and
        released before I/O during flush().

    Global State:
        This implementation intentionally avoids calling structlog.configure(),
        which modifies global state. Instead, each instance uses structlog.wrap_logger()
        to create an instance-specific logger with its own processor chain. This design:

        - Allows multiple instances with different output formats (json vs console)
        - Prevents configuration conflicts in multi-tenant or test environments
        - Follows library best practices for not modifying global logging config

        WARNING: If other code in your application calls structlog.configure(),
        that may affect the underlying logger behavior. This sink isolates the
        processor chain via wrap_logger() but cannot isolate base logger config.
    """

    # Class-level tracking for missing context key warnings
    # Used to warn once per session about missing recommended keys
    _warned_missing_keys: set[str] = set()

    # Instance tracking for global state conflict detection
    # When multiple instances exist, warn users about potential conflicts
    _instance_count: int = 0
    _warned_multiple_instances: bool = False

    def __init__(
        self,
        max_buffer_size: int = 1000,
        output_format: str = "json",
        drop_policy: Literal["drop_oldest"] = "drop_oldest",
        warn_on_missing_recommended_keys: bool = True,
    ) -> None:
        """Initialize the structured logging sink.

        Args:
            max_buffer_size: Maximum number of log entries to buffer.
                When exceeded, oldest entries are dropped. Default: 1000.
            output_format: Output format for log entries.
                - "json": JSON format (default, machine-readable, production-ready)
                - "console": Colored console output (human-readable, dev only)
            drop_policy: Policy for handling buffer overflow. Currently only
                "drop_oldest" is supported, which drops the oldest entries
                when the buffer is full. Default: "drop_oldest".
            warn_on_missing_recommended_keys: If True (default), emit a warning
                to stderr when recommended context keys (correlation_id, node_id,
                operation) are missing. Set to False to suppress these warnings.

        Raises:
            ProtocolConfigurationError: If structlog is not installed (install with:
                pip install structlog), max_buffer_size is less than 1, or
                output_format is not recognized. Includes correlation_id for
                distributed tracing.

        Note:
            The drop_oldest policy is implemented using a collections.deque with
            maxlen set to max_buffer_size. When the buffer is full and a new entry
            is added, the deque automatically discards the oldest entry from the
            left side. This provides O(1) performance for both append and drop
            operations.
        """
        # Validate dependency is available with clear error message
        if not _STRUCTLOG_AVAILABLE:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="initialize_logging_sink",
            )
            raise ProtocolConfigurationError(
                "structlog is required for SinkLoggingStructured but is not installed. "
                f"Install with: pip install structlog. "
                f"Original error: {_STRUCTLOG_IMPORT_ERROR}",
                context=context,
                dependency="structlog",
            )

        if max_buffer_size < 1:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="initialize_logging_sink",
            )
            raise ProtocolConfigurationError(
                f"max_buffer_size must be >= 1, got {max_buffer_size}",
                context=context,
                parameter="max_buffer_size",
                value=max_buffer_size,
            )

        if output_format not in ("json", "console"):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="initialize_logging_sink",
            )
            raise ProtocolConfigurationError(
                f"output_format must be 'json' or 'console', got '{output_format}'",
                context=context,
                parameter="output_format",
                value=output_format,
            )

        self._max_buffer_size = max_buffer_size
        self._output_format = output_format
        self._drop_policy: Literal["drop_oldest"] = drop_policy
        self._warn_on_missing_keys = warn_on_missing_recommended_keys
        # Use deque with maxlen to automatically drop oldest entries when full
        self._buffer: deque[ModelBufferedLogEntry] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._drop_count = 0
        self._logger = self._configure_structlog()

        # Track instances and warn about potential global state conflicts
        self._track_instance_creation()

    def _configure_structlog(self) -> structlog.BoundLogger:
        """Configure and return a structlog logger instance.

        This method creates an instance-specific logger using structlog.wrap_logger()
        instead of structlog.configure(). This is a critical design choice that:

        1. **Avoids global state**: structlog.configure() modifies module-level state,
           which would cause conflicts when multiple SinkLoggingStructured instances
           exist with different output formats (e.g., JSON for production, console
           for local debugging).

        2. **Enables library safety**: Library code should never modify global logging
           configuration, as this can unexpectedly affect the host application.

        3. **Supports testing**: Tests can create multiple instances with different
           configurations without cleanup/teardown concerns.

        Processor Chain:
            The processor chain is configured per-instance:
            - add_log_level: Adds 'level' key to every log entry
            - TimeStamper: Adds ISO-8601 'timestamp' at flush time
            - StackInfoRenderer: Formats stack traces when present
            - UnicodeDecoder: Ensures proper Unicode handling
            - JSONRenderer or ConsoleRenderer: Final output formatting

        JSON Output Enforcement:
            When output_format="json", the JSONRenderer ensures:
            - Valid JSON on every line (can be parsed by json.loads)
            - Proper escaping of special characters
            - UTF-8 encoding
            - No embedded newlines (single-line format)

        Returns:
            Configured structlog BoundLogger instance with instance-specific processors.

        Note:
            Using wrap_logger() is the recommended pattern for library code that
            should not modify global logging configuration. Each instance gets
            its own processor chain based on its output_format setting. This means
            two instances can coexist: one writing JSON to a file, another writing
            colored console output for debugging.

            WARNING: structlog.configure() called elsewhere in your application
            may still affect the underlying logger behavior. This sink's approach
            isolates the processor chain but not the base PrintLogger configuration.
        """
        # Configure processors based on output format
        processors: list[structlog.types.Processor] = [
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        if self._output_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))

        # Use wrap_logger() to create an instance-specific logger without
        # modifying global state. This prevents conflicts when multiple
        # instances are created with different output formats.
        # The PrintLogger writes to stdout by default.
        logger: structlog.BoundLogger = structlog.wrap_logger(
            structlog.PrintLogger(),
            processors=processors,
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
        )
        return logger

    def _track_instance_creation(self) -> None:
        """Track instance creation and warn about potential global state conflicts.

        This method increments the class-level instance counter and emits a
        warning when multiple instances are detected. While this sink uses
        structlog.wrap_logger() to avoid global state modification, the warning
        helps users understand that:

        1. Multiple instances with different output formats can coexist safely
        2. If other code calls structlog.configure(), it may affect all instances
        3. Base logger configuration (PrintLogger) is shared

        The warning is emitted once per process to avoid log spam.

        Note:
            This is a best-effort detection mechanism. In multi-threaded code,
            the instance count may not be perfectly accurate, but this is
            acceptable since the warning is informational only.
        """
        SinkLoggingStructured._instance_count += 1

        if (
            SinkLoggingStructured._instance_count > 1
            and not SinkLoggingStructured._warned_multiple_instances
        ):
            SinkLoggingStructured._warned_multiple_instances = True
            warnings.warn(
                f"Multiple SinkLoggingStructured instances detected "
                f"(count: {SinkLoggingStructured._instance_count}). "
                f"While each instance has isolated processor chains via wrap_logger(), "
                f"be aware that: (1) structlog.configure() called elsewhere may affect "
                f"all instances, (2) all instances share the same base PrintLogger. "
                f"This is typically fine but may cause unexpected behavior if you "
                f"rely on global structlog configuration.",
                stacklevel=3,  # Points to user code that called SinkLoggingStructured()
            )

    @property
    def max_buffer_size(self) -> int:
        """Maximum number of entries the buffer can hold."""
        return self._max_buffer_size

    @property
    def output_format(self) -> str:
        """Current output format ('json' or 'console')."""
        return self._output_format

    @property
    def drop_policy(self) -> Literal["drop_oldest"]:
        """Current drop policy for buffer overflow handling."""
        return self._drop_policy

    @property
    def drop_count(self) -> int:
        """Number of entries dropped due to buffer overflow.

        This counter is incremented each time an entry is dropped
        because the buffer is full. It can be used to monitor
        if the buffer size is adequate for the workload.

        Guarantees:
            - Always non-negative (initialized to 0, only incremented)
            - Thread-safe (protected by lock)

        Returns:
            Non-negative integer count of dropped entries.
        """
        with self._lock:
            return self._drop_count

    @property
    def buffer_size(self) -> int:
        """Current number of entries in the buffer.

        Guarantees:
            - Always non-negative (0 <= buffer_size <= max_buffer_size)
            - Thread-safe (protected by lock)

        Returns:
            Non-negative integer count of buffered entries.
        """
        with self._lock:
            return len(self._buffer)

    def emit(
        self,
        level: EnumLogLevel,
        message: str,
        context: dict[str, JsonType],
    ) -> None:
        """Buffer a log entry for later emission.

        Synchronously buffers a log entry without performing any I/O.
        This method MUST NOT block, perform network calls, or write to disk.
        All I/O is deferred until flush() is called.

        If the buffer is full, the oldest entry is dropped to make room
        for the new entry (drop_oldest policy). The drop_count property
        tracks how many entries have been dropped.

        Args:
            level: Log level from EnumLogLevel (TRACE, DEBUG, INFO, WARNING,
                   ERROR, CRITICAL, FATAL, SUCCESS, UNKNOWN).
            message: Log message content. Should be a complete, self-contained
                     message suitable for structured logging.
            context: Structured context data for the log entry. Values may be
                     any JSON-compatible type (str, int, float, bool, None,
                     list, dict). This enables richer context than string-only:

                     ```python
                     context = {
                         "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
                         "retry_count": 3,           # int is valid
                         "duration_ms": 42.5,        # float is valid
                         "is_retry": True,           # bool is valid
                         "tags": ["hot-path", "db"], # list is valid
                         "metadata": {"version": 1}, # dict is valid
                     }
                     ```

        Automatically Added Keys:
            The sink automatically adds these keys (callers should NOT include them):
            - ``original_timestamp``: ISO-8601 timestamp captured at emit() time
            - ``level``: Log level string (added by structlog processor)
            - ``timestamp``: ISO-8601 timestamp at flush() time (added by structlog)

        Recommended Context Keys:
            For effective observability, callers SHOULD include:
            - ``correlation_id``: UUID for distributed tracing across services
            - ``node_id``: ONEX node identifier (e.g., "node_registration_orchestrator")
            - ``operation``: Current operation name (e.g., "validate_contract")

            Example with recommended keys:
            ```python
            sink.emit(
                level=EnumLogLevel.INFO,
                message="Contract validation completed",
                context={
                    "correlation_id": str(correlation_id),
                    "node_id": "node_contract_validator",
                    "operation": "validate_contract",
                    "contract_path": "/path/to/contract.yaml",
                    "validation_result": "passed",
                }
            )
            ```

        Note:
            This method is synchronous (def, not async def) by design.
            It MUST complete without blocking to maintain hot-path performance.

        Example:
            ```python
            sink.emit(
                level=EnumLogLevel.INFO,
                message="Cache hit for user lookup",
                context={
                    "user_id": "u_123",
                    "cache_key": "user:u_123",
                    "correlation_id": str(correlation_id),
                }
            )
            ```
        """
        # Create immutable Pydantic model for thread-safe buffering
        entry = ModelBufferedLogEntry(
            level=level,
            message=message,
            context=dict(context),  # Defensive copy to prevent mutation
            timestamp=datetime.now(UTC),
        )

        with self._lock:
            # deque with maxlen automatically drops oldest when full
            if len(self._buffer) >= self._max_buffer_size:
                self._drop_count += 1
            self._buffer.append(entry)

    def flush(self) -> None:
        """Flush all buffered log entries to structlog.

        This is the ONLY method in this protocol that may perform I/O.
        All buffered log entries are written to the configured structlog
        backend and the buffer is cleared.

        The flush process:
            1. Acquire lock and copy all entries from buffer
            2. Clear the buffer
            3. Release the lock
            4. Validate required context keys (warn on missing recommended keys)
            5. Write entries to structlog (I/O happens outside the lock)
            6. On error, fall back to stderr

        Required Context Keys Validation:
            At flush time, the sink validates that required keys are present
            and warns about missing recommended keys. This is done at flush
            time (not emit time) to maintain hot-path performance.

            **Always present** (added by sink):
            - original_timestamp
            - level (via structlog)
            - timestamp (via structlog)

            **Recommended** (warning if missing):
            - correlation_id
            - node_id
            - operation

        Thread-Safety:
            This method is safe to call concurrently with emit().
            The lock is held only during the copy/clear phase, not during I/O.

        Error Handling:
            If structlog fails during emission, the sink falls back to
            writing entries directly to stderr to prevent data loss.
            Errors during stderr fallback are silently ignored to prevent
            cascading failures.

        Example:
            ```python
            # Periodic flush in a long-running process
            while processing:
                batch = get_next_batch()
                process_batch(batch, sink)

                # Flush every N iterations
                if iteration % 100 == 0:
                    sink.flush()

            # Final flush on shutdown
            sink.flush()
            ```
        """
        # Copy entries under lock, then release lock before I/O
        with self._lock:
            entries = list(self._buffer)
            self._buffer.clear()

        # Process entries outside the lock to minimize contention
        for entry in entries:
            self._validate_context_keys(entry)
            self._emit_entry(entry)

    def _validate_context_keys(self, entry: ModelBufferedLogEntry) -> None:
        """Validate that required/recommended context keys are present.

        This method checks for recommended context keys and emits warnings
        (once per key per session) when they are missing. Validation is
        performed at flush time to avoid impacting hot-path performance.

        Args:
            entry: The buffered log entry to validate.

        Note:
            The warning is only emitted once per missing key per session
            to avoid log spam. Missing keys are tracked in class-level
            _warned_missing_keys set.
        """
        if not self._warn_on_missing_keys:
            return

        recommended = EnumRequiredLogContextKey.recommended_keys()
        present_keys = set(entry.context.keys())
        missing = recommended - present_keys

        # Warn once per missing key to avoid spam
        for key in missing:
            if key not in SinkLoggingStructured._warned_missing_keys:
                SinkLoggingStructured._warned_missing_keys.add(key)
                warnings.warn(
                    f"Recommended context key '{key}' missing from log entry. "
                    f"Including {', '.join(sorted(recommended))} improves observability.",
                    stacklevel=3,  # Points to user code that called flush()
                )

    def _emit_entry(self, entry: ModelBufferedLogEntry) -> None:
        """Emit a single log entry to structlog.

        This method maps the EnumLogLevel to the appropriate structlog method
        and constructs the final context dict with required keys.

        Args:
            entry: The buffered log entry to emit.

        Required Context Keys (always present in output):
            The following keys are GUARANTEED to be present in every log entry.
            If not provided by the caller, defaults are applied:

            - ``original_timestamp``: ISO-8601 timestamp from emit() time
            - ``correlation_id``: Default "00000000-0000-0000-0000-000000000000" if missing
            - ``node_id``: Default "unknown" if missing
            - ``operation``: Default "unknown" if missing

            The structlog processor chain adds additional keys:
            - ``level``: Log level string (via add_log_level processor)
            - ``timestamp``: ISO-8601 timestamp at flush time (via TimeStamper)

        JSON Output Guarantee:
            When output_format="json", this method ensures valid JSON output:
            - All context values are JSON-serializable (enforced by JsonType)
            - Special characters are properly escaped
            - Output is single-line (no embedded newlines)
            - All required keys are present with valid values
        """
        # Map the log level to structlog method name
        level_str = str(entry.level.value).lower()
        structlog_level = _STRUCTLOG_LEVEL_MAP.get(level_str, "info")

        # Build context dict with REQUIRED keys guaranteed present
        # Apply defaults for recommended keys if not provided by caller
        # All values are JsonType (JSON-compatible) for serialization safety
        log_context: dict[str, JsonType] = {
            # Always-present auto-added key
            EnumRequiredLogContextKey.ORIGINAL_TIMESTAMP: entry.timestamp.isoformat(),
            # Required keys with defaults (applied first, then overwritten by entry.context)
            EnumRequiredLogContextKey.CORRELATION_ID: _DEFAULT_CORRELATION_ID,
            EnumRequiredLogContextKey.NODE_ID: _DEFAULT_NODE_ID,
            EnumRequiredLogContextKey.OPERATION: _DEFAULT_OPERATION,
        }
        # Caller-provided context overwrites defaults
        log_context.update(entry.context)

        try:
            # Get the appropriate structlog method and call it
            log_method = getattr(self._logger, structlog_level, self._logger.info)
            log_method(entry.message, **log_context)
        except (ValueError, TypeError, AttributeError, OSError):
            # Fall back to stderr if structlog fails due to:
            # - ValueError: invalid arguments to log methods
            # - TypeError: type mismatches in context values
            # - AttributeError: missing log method on logger
            # - OSError: I/O errors when writing to stdout
            self._emit_to_stderr(entry, structlog_level)

    def _emit_to_stderr(self, entry: ModelBufferedLogEntry, level: str) -> None:
        """Fall back to stderr when structlog fails.

        This is the last-resort fallback to ensure log entries are not
        silently lost when structlog encounters errors. Output is always
        JSON format to maintain consistency with the primary output.

        This fallback also ensures all required context keys are present
        with defaults, maintaining the same guarantees as the primary path.

        Args:
            entry: The log entry to emit.
            level: The log level string.
        """
        try:
            # Always emit JSON for fallback to ensure consistent format
            # Include all required context keys with defaults (same as primary path)
            fallback_entry: dict[str, JsonType] = {
                EnumRequiredLogContextKey.TIMESTAMP: entry.timestamp.isoformat(),
                EnumRequiredLogContextKey.LEVEL: level.upper(),
                EnumRequiredLogContextKey.ORIGINAL_TIMESTAMP: entry.timestamp.isoformat(),
                # Required keys with defaults
                EnumRequiredLogContextKey.CORRELATION_ID: _DEFAULT_CORRELATION_ID,
                EnumRequiredLogContextKey.NODE_ID: _DEFAULT_NODE_ID,
                EnumRequiredLogContextKey.OPERATION: _DEFAULT_OPERATION,
                "message": entry.message,
            }
            # Caller-provided context overwrites defaults
            fallback_entry.update(entry.context)
            # Use json.dumps for guaranteed valid JSON output
            print(json.dumps(fallback_entry, default=str), file=sys.stderr)
        except (ValueError, TypeError, OSError):
            # Silently ignore errors in the fallback path to prevent cascading
            # failures. Common errors: ValueError (string formatting),
            # TypeError (type issues), OSError (stream write failures)
            pass

    def reset_drop_count(self) -> int:
        """Reset the drop counter and return the previous value.

        This is useful for monitoring and alerting on buffer overflow
        conditions in long-running processes.

        Returns:
            The number of entries that were dropped before the reset.
        """
        with self._lock:
            previous_count = self._drop_count
            self._drop_count = 0
            return previous_count

    @classmethod
    def reset_warning_state(cls) -> None:
        """Reset the class-level warning state.

        This method clears the set of keys that have already triggered
        warnings, allowing warnings to be emitted again. Useful for testing.
        """
        cls._warned_missing_keys.clear()

    @classmethod
    def reset_instance_tracking(cls) -> int:
        """Reset the class-level instance tracking state.

        This method resets the instance counter and warning flag.
        Returns the previous instance count. Useful for testing.

        Returns:
            The number of instances that were tracked before the reset.
        """
        previous_count = cls._instance_count
        cls._instance_count = 0
        cls._warned_multiple_instances = False
        return previous_count

    @classmethod
    def get_instance_count(cls) -> int:
        """Get the current number of tracked instances.

        Returns:
            The number of SinkLoggingStructured instances created since
            the last reset or process start.
        """
        return cls._instance_count


__all__ = ["SinkLoggingStructured"]
