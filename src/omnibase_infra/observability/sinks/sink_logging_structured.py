# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Structured logging sink implementation using structlog.

This module provides a structured logging sink that implements the
ProtocolHotPathLoggingSink protocol. It buffers log entries in memory
and flushes them to structlog with JSON formatting.

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

Fallback Behavior:
    If structlog fails during flush, the sink falls back to writing directly
    to stderr to ensure log entries are never silently lost.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, NamedTuple

import structlog

if TYPE_CHECKING:
    from omnibase_core.enums import EnumLogLevel

# Lazy import to avoid circular dependency at module load time
# EnumLogLevel is only needed at runtime for type checking parameters


class BufferedLogEntry(NamedTuple):
    """Internal representation of a buffered log entry.

    Using NamedTuple for memory efficiency in the hot path.
    All fields are immutable to ensure thread-safety when entries
    are copied during flush operations.
    """

    level: EnumLogLevel
    message: str
    context: dict[str, str]
    timestamp: datetime


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
        - json: Machine-readable JSON format (default)
        - console: Human-readable colored console output

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
            sink.emit(EnumLogLevel.DEBUG, f"Processed {item}", {"id": str(item.id)})

        # Flush when hot path completes
        sink.flush()
        ```

    Thread Safety:
        This implementation is THREAD-SAFE. Multiple threads may call emit()
        and flush() concurrently. The lock is held briefly during emit() and
        released before I/O during flush().
    """

    def __init__(
        self,
        max_buffer_size: int = 1000,
        output_format: str = "json",
        drop_policy: Literal["drop_oldest"] = "drop_oldest",
    ) -> None:
        """Initialize the structured logging sink.

        Args:
            max_buffer_size: Maximum number of log entries to buffer.
                When exceeded, oldest entries are dropped. Default: 1000.
            output_format: Output format for log entries.
                - "json": JSON format (default, machine-readable)
                - "console": Colored console output (human-readable)
            drop_policy: Policy for handling buffer overflow. Currently only
                "drop_oldest" is supported. Default: "drop_oldest".

        Raises:
            ValueError: If max_buffer_size is less than 1 or output_format
                is not recognized.
        """
        if max_buffer_size < 1:
            msg = f"max_buffer_size must be >= 1, got {max_buffer_size}"
            raise ValueError(msg)

        if output_format not in ("json", "console"):
            msg = f"output_format must be 'json' or 'console', got '{output_format}'"
            raise ValueError(msg)

        self._max_buffer_size = max_buffer_size
        self._output_format = output_format
        self._drop_policy: Literal["drop_oldest"] = drop_policy
        # Use deque with maxlen to automatically drop oldest entries when full
        self._buffer: deque[BufferedLogEntry] = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()
        self._drop_count = 0
        self._logger = self._configure_structlog()

    def _configure_structlog(self) -> structlog.BoundLogger:
        """Configure and return a structlog logger instance.

        This method creates an instance-specific logger using structlog.wrap_logger()
        instead of structlog.configure(). This avoids modifying global state,
        preventing conflicts when multiple SinkLoggingStructured instances
        are created with different configurations.

        Returns:
            Configured structlog BoundLogger instance with instance-specific processors.

        Note:
            Using wrap_logger() is the recommended pattern for library code that
            should not modify global logging configuration. Each instance gets
            its own processor chain based on its output_format setting.
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
        """
        with self._lock:
            return self._drop_count

    @property
    def buffer_size(self) -> int:
        """Current number of entries in the buffer."""
        with self._lock:
            return len(self._buffer)

    def emit(
        self,
        level: EnumLogLevel,
        message: str,
        context: dict[str, str],
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
            context: Structured context data for the log entry. All values
                     MUST be strings to ensure serialization safety and
                     prevent type coercion issues in hot paths.

        Note:
            This method is synchronous (def, not async def) by design.
            It MUST complete without blocking to maintain hot-path performance.

        Example:
            ```python
            sink.emit(
                level=EnumLogLevel.INFO,
                message="Cache hit for user lookup",
                context={"user_id": "u_123", "cache_key": "user:u_123"}
            )
            ```
        """
        entry = BufferedLogEntry(
            level=level,
            message=message,
            context=context.copy(),  # Defensive copy to prevent mutation
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
            4. Write entries to structlog (I/O happens outside the lock)
            5. On error, fall back to stderr

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
            self._emit_entry(entry)

    def _emit_entry(self, entry: BufferedLogEntry) -> None:
        """Emit a single log entry to structlog.

        Args:
            entry: The buffered log entry to emit.
        """
        # Map the log level to structlog method name
        level_str = str(entry.level.value).lower()
        structlog_level = _STRUCTLOG_LEVEL_MAP.get(level_str, "info")

        # Build context dict with timestamp
        log_context = {
            "original_timestamp": entry.timestamp.isoformat(),
            **entry.context,
        }

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

    def _emit_to_stderr(self, entry: BufferedLogEntry, level: str) -> None:
        """Fall back to stderr when structlog fails.

        This is the last-resort fallback to ensure log entries are not
        silently lost when structlog encounters errors.

        Args:
            entry: The log entry to emit.
            level: The log level string.
        """
        try:
            timestamp = entry.timestamp.isoformat()
            context_str = " ".join(f"{k}={v}" for k, v in entry.context.items())
            stderr_msg = f"[{timestamp}] [{level.upper()}] {entry.message}"
            if context_str:
                stderr_msg += f" | {context_str}"
            print(stderr_msg, file=sys.stderr)
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


__all__ = ["SinkLoggingStructured"]
