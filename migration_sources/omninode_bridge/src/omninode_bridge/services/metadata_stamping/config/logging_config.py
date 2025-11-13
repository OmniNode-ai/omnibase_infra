"""
Comprehensive logging configuration for metadata stamping service.

This module provides structured logging with observability features including
correlation IDs, performance metrics, and distributed tracing support.
"""

import json
import logging
import logging.config
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Optional

import structlog


class CorrelationIDProcessor:
    """Add correlation ID to log records for distributed tracing."""

    def __init__(self, correlation_id_key: str = "correlation_id"):
        self.correlation_id_key = correlation_id_key

    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID if not present."""
        if self.correlation_id_key not in event_dict:
            # Try to get from context or generate new one
            correlation_id = getattr(logger, "_correlation_id", None) or str(
                uuid.uuid4()
            )
            event_dict[self.correlation_id_key] = correlation_id
        return event_dict


class PerformanceProcessor:
    """Add performance context to log records."""

    def __call__(self, logger, method_name, event_dict):
        """Add performance metrics if available."""
        # Add execution time if available
        if hasattr(logger, "_execution_start"):
            execution_time = time.perf_counter() - logger._execution_start
            event_dict["execution_time_ms"] = round(execution_time * 1000, 3)

        # Add operation context if available
        if hasattr(logger, "_operation_context"):
            event_dict.update(logger._operation_context)

        return event_dict


class ServiceContextProcessor:
    """Add service context information to log records."""

    def __init__(
        self, service_name: str = "metadata-stamping", service_version: str = "0.1.0"
    ):
        self.service_name = service_name
        self.service_version = service_version

    def __call__(self, logger, method_name, event_dict):
        """Add service context."""
        event_dict.update(
            {
                "service": self.service_name,
                "version": self.service_version,
                "environment": os.getenv("ENVIRONMENT", "development"),
                "log_level": method_name.upper(),
            }
        )
        return event_dict


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            ):
                log_entry[key] = value

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_structured_logging(
    log_level: str = "INFO",
    enable_json: bool = True,
    enable_correlation: bool = True,
    enable_performance: bool = True,
    log_file: Optional[str] = None,
) -> None:
    """
    Setup comprehensive structured logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Enable JSON structured logging
        enable_correlation: Enable correlation ID tracking
        enable_performance: Enable performance metric logging
        log_file: Optional log file path
    """
    # Clear any existing loggers
    logging.getLogger().handlers.clear()

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Add service context
    processors.append(ServiceContextProcessor())

    # Add correlation ID processor if enabled
    if enable_correlation:
        processors.append(CorrelationIDProcessor())

    # Add performance processor if enabled
    if enable_performance:
        processors.append(PerformanceProcessor())

    # Configure output format
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
            },
            "simple": {"format": "%(levelname)s - %(name)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "json" if enable_json else "detailed",
                "stream": sys.stdout,
            }
        },
        "loggers": {
            "omninode_bridge.services.metadata_stamping": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {"level": log_level, "handlers": ["console"]},
    }

    # Add file handler if log file specified
    if log_file:
        logging_config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json" if enable_json else "detailed",
            "filename": log_file,
            "maxBytes": 50 * 1024 * 1024,  # 50MB
            "backupCount": 5,
            "encoding": "utf-8",
        }

        # Add file handler to all loggers
        for logger_config in logging_config["loggers"].values():
            logger_config["handlers"].append("file")
        logging_config["root"]["handlers"].append("file")

    logging.config.dictConfig(logging_config)


def get_logger(
    name: str, correlation_id: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Get a structured logger with optional correlation ID.

    Args:
        name: Logger name
        correlation_id: Optional correlation ID for distributed tracing

    Returns:
        Configured structured logger
    """
    logger = structlog.get_logger(name)

    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)

    return logger


def create_performance_logger(
    name: str, operation: str, correlation_id: Optional[str] = None, **context
) -> structlog.BoundLogger:
    """
    Create a performance-aware logger for operation tracking.

    Args:
        name: Logger name
        operation: Operation being performed
        correlation_id: Optional correlation ID
        **context: Additional context information

    Returns:
        Performance-aware logger
    """
    logger = get_logger(name, correlation_id)

    # Bind operation context
    logger = logger.bind(
        operation=operation, operation_start_time=time.perf_counter(), **context
    )

    return logger


class OperationLogger:
    """Context manager for operation logging with automatic performance tracking."""

    def __init__(
        self,
        logger: structlog.BoundLogger,
        operation: str,
        level: str = "info",
        **context,
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.context = context
        self.start_time = None

    def __enter__(self):
        """Start operation logging."""
        self.start_time = time.perf_counter()
        log_func = getattr(self.logger, self.level)
        log_func(
            f"Starting operation: {self.operation}",
            operation=self.operation,
            operation_status="started",
            **self.context,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete operation logging."""
        execution_time = time.perf_counter() - self.start_time

        if exc_type is None:
            # Success
            log_func = getattr(self.logger, self.level)
            log_func(
                f"Completed operation: {self.operation}",
                operation=self.operation,
                operation_status="completed",
                execution_time_ms=round(execution_time * 1000, 3),
                **self.context,
            )
        else:
            # Error
            self.logger.error(
                f"Failed operation: {self.operation}",
                operation=self.operation,
                operation_status="failed",
                execution_time_ms=round(execution_time * 1000, 3),
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context,
            )


def setup_observability_logging():
    """Setup observability-focused logging for production."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    environment = os.getenv("ENVIRONMENT", "development")

    # Enable JSON logging in production
    enable_json = environment == "production"

    # Setup log file in production
    log_file = None
    if environment == "production":
        log_file = "/app/logs/metadata-stamping.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    setup_structured_logging(
        log_level=log_level,
        enable_json=enable_json,
        enable_correlation=True,
        enable_performance=True,
        log_file=log_file,
    )

    # Log startup information
    logger = get_logger("metadata_stamping.startup")
    logger.info(
        "Observability logging initialized",
        log_level=log_level,
        environment=environment,
        json_logging=enable_json,
        log_file=log_file is not None,
    )


# Export convenience functions
__all__ = [
    "setup_structured_logging",
    "setup_observability_logging",
    "get_logger",
    "create_performance_logger",
    "OperationLogger",
    "CorrelationIDProcessor",
    "PerformanceProcessor",
    "ServiceContextProcessor",
    "JSONFormatter",
]
