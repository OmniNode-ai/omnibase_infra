"""
Structured Logger Utility for ONEX Infrastructure.

Provides structured logging with correlation IDs, context scoping, and
integration with OnexError. Built on structlog for consistent, machine-parseable
log output across all infrastructure components.

Features:
- Pre-configured logger factory for different components
- Structured logging with correlation IDs and context
- Context managers for log scoping and automatic cleanup
- Integration with OnexError for exception logging
- Environment-aware logging configuration
"""

import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from uuid import UUID

import structlog
from structlog.types import FilteringBoundLogger, Processor

from omnibase_core.core.errors.onex_error import OnexError


class LogLevel:
    """Standard log levels for infrastructure components."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StructuredLoggerConfig:
    """Configuration for structured logging."""

    def __init__(
        self,
        environment: str | None = None,
        log_level: str | None = None,
        json_output: bool | None = None,
        include_timestamps: bool = True,
        include_stack_info: bool = True,
    ):
        """
        Initialize logger configuration.

        Args:
            environment: Deployment environment (production, staging, development)
            log_level: Minimum log level to output
            json_output: Whether to output logs as JSON (auto-detected if None)
            include_timestamps: Whether to include timestamps in log output
            include_stack_info: Whether to include stack traces for errors
        """
        self.environment = environment or self._detect_environment()
        self.log_level = log_level or self._get_log_level()
        self.json_output = (
            json_output
            if json_output is not None
            else self._should_use_json_output()
        )
        self.include_timestamps = include_timestamps
        self.include_stack_info = include_stack_info

    def _detect_environment(self) -> str:
        """Detect current deployment environment."""
        env_vars = ["ENVIRONMENT", "ENV", "DEPLOYMENT_ENV", "NODE_ENV"]
        for var in env_vars:
            value = os.getenv(var)
            if value:
                return value.lower()
        return "development"

    def _get_log_level(self) -> str:
        """Get environment-specific log level."""
        log_levels = {
            "production": LogLevel.INFO,
            "staging": LogLevel.DEBUG,
            "development": LogLevel.DEBUG,
        }
        return os.getenv(
            "LOG_LEVEL", log_levels.get(self.environment, LogLevel.INFO)
        ).lower()

    def _should_use_json_output(self) -> bool:
        """Determine if JSON output should be used based on environment."""
        # Use JSON in production/staging, human-readable in development
        if self.environment in ("production", "staging"):
            return True
        # Override via environment variable
        return os.getenv("LOG_FORMAT", "").lower() == "json"


class StructuredLogger:
    """
    Structured logger wrapper with ONEX infrastructure integration.

    Provides structured logging with correlation IDs, context management,
    and automatic integration with OnexError exceptions.
    """

    def __init__(
        self,
        component_name: str,
        config: StructuredLoggerConfig | None = None,
        base_context: dict[str, Any] | None = None,
    ):
        """
        Initialize structured logger for a component.

        Args:
            component_name: Name of the component using this logger
            config: Logger configuration (auto-detected if None)
            base_context: Base context to include in all log messages
        """
        self.component_name = component_name
        self.config = config or StructuredLoggerConfig()
        self.base_context = base_context or {}

        # Ensure structlog is configured
        if not structlog.is_configured():
            self._configure_structlog()

        # Create bound logger with component context
        self.logger: FilteringBoundLogger = structlog.get_logger(
            component=component_name,
            environment=self.config.environment,
            **self.base_context,
        )

    def _configure_structlog(self) -> None:
        """Configure structlog with appropriate processors."""
        shared_processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso")
            if self.config.include_timestamps
            else structlog.processors.CallsiteParameterAdder(),
            structlog.processors.StackInfoRenderer()
            if self.config.include_stack_info
            else structlog.processors.CallsiteParameterAdder(),
        ]

        # Choose renderer based on configuration
        if self.config.json_output:
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=True)

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                self._get_min_log_level()
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )

    def _get_min_log_level(self) -> int:
        """Convert log level string to structlog level."""
        import logging

        levels = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return levels.get(self.config.log_level, logging.INFO)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **context)

    def info(self, message: str, **context: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **context)

    def error(self, message: str, **context: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **context)

    def critical(self, message: str, **context: Any) -> None:
        """Log critical message with context."""
        self.logger.critical(message, **context)

    def log_exception(
        self,
        exception: Exception,
        message: str | None = None,
        correlation_id: str | UUID | None = None,
        **context: Any,
    ) -> None:
        """
        Log exception with full context.

        Args:
            exception: Exception to log
            message: Optional message (uses exception message if None)
            correlation_id: Optional correlation ID for tracking
            **context: Additional context to include in log
        """
        log_message = message or str(exception)
        log_context = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            **context,
        }

        # Add correlation ID if provided
        if correlation_id:
            log_context["correlation_id"] = str(correlation_id)

        # Add OnexError-specific context if applicable
        if isinstance(exception, OnexError):
            log_context.update(
                {
                    "error_code": exception.code.name,
                    "error_details": exception.details,
                }
            )

        self.logger.error(log_message, exc_info=True, **log_context)

    @contextmanager
    def log_scope(self, **context: Any):
        """
        Context manager for scoped logging with automatic cleanup.

        Args:
            **context: Context to add for the scope

        Yields:
            Bound logger with scoped context

        Example:
            with logger.log_scope(operation="database_query", query_id="123"):
                logger.info("Executing query")
                # All logs in this scope will include operation and query_id
        """
        # Push context
        token = structlog.contextvars.bind_contextvars(**context)
        try:
            yield self.logger.bind(**context)
        finally:
            # Pop context
            structlog.contextvars.unbind_contextvars(*context.keys())

    @contextmanager
    def log_operation(
        self, operation_name: str, correlation_id: str | UUID | None = None, **context: Any
    ):
        """
        Context manager for logging operations with timing.

        Automatically logs start/end of operation with duration.

        Args:
            operation_name: Name of the operation
            correlation_id: Optional correlation ID
            **context: Additional context

        Yields:
            Bound logger for the operation

        Example:
            with logger.log_operation("process_message", correlation_id=msg_id):
                # Process message
                logger.info("Processing step 1")
        """
        import time

        correlation_str = str(correlation_id) if correlation_id else None
        operation_context = {
            "operation": operation_name,
            **({"correlation_id": correlation_str} if correlation_str else {}),
            **context,
        }

        start_time = time.time()
        self.logger.info(
            f"Starting operation: {operation_name}", **operation_context
        )

        try:
            with self.log_scope(**operation_context):
                yield self.logger.bind(**operation_context)

            duration = time.time() - start_time
            self.logger.info(
                f"Completed operation: {operation_name}",
                duration_seconds=duration,
                **operation_context,
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Failed operation: {operation_name}",
                duration_seconds=duration,
                exception_type=type(e).__name__,
                exception_message=str(e),
                **operation_context,
            )
            raise


class LoggerFactory:
    """
    Factory for creating pre-configured structured loggers.

    Provides consistent logger instances for different infrastructure components.
    """

    _config: StructuredLoggerConfig | None = None
    _loggers: dict[str, StructuredLogger] = {}

    @classmethod
    def configure(cls, config: StructuredLoggerConfig) -> None:
        """
        Configure the logger factory globally.

        Args:
            config: Logger configuration to use for all loggers
        """
        cls._config = config

    @classmethod
    def get_logger(
        cls,
        component_name: str,
        base_context: dict[str, Any] | None = None,
    ) -> StructuredLogger:
        """
        Get or create logger for a component.

        Args:
            component_name: Name of the component
            base_context: Optional base context for this logger

        Returns:
            Configured structured logger instance
        """
        # Create cache key including context
        cache_key = f"{component_name}:{hash(frozenset((base_context or {}).items()))}"

        if cache_key not in cls._loggers:
            cls._loggers[cache_key] = StructuredLogger(
                component_name=component_name,
                config=cls._config,
                base_context=base_context,
            )

        return cls._loggers[cache_key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear logger cache (useful for testing)."""
        cls._loggers.clear()
