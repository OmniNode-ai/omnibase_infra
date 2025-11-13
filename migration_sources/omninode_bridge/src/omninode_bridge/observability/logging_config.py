"""
Structured logging configuration for omninode_bridge.

Provides JSON-formatted logging with correlation tracking, context management,
and integration with OpenTelemetry distributed tracing.
"""

import json
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID

# Context variables for correlation tracking
correlation_id_ctx: ContextVar[Optional[UUID]] = ContextVar(
    "correlation_id", default=None
)
workflow_id_ctx: ContextVar[Optional[UUID]] = ContextVar("workflow_id", default=None)
request_id_ctx: ContextVar[Optional[UUID]] = ContextVar("request_id", default=None)
session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
stage_name_ctx: ContextVar[Optional[str]] = ContextVar("stage_name", default=None)

# Extra context for custom fields (default must be a value, not factory)
extra_context_ctx: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "extra_context", default=None
)


class CorrelationFilter(logging.Filter):
    """Add correlation context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Enrich log record with correlation context."""
        # Add correlation IDs
        record.correlation_id = str(correlation_id_ctx.get() or "none")
        record.workflow_id = str(workflow_id_ctx.get() or "none")
        record.request_id = str(request_id_ctx.get() or "none")
        record.session_id = session_id_ctx.get() or "none"
        record.user_id = user_id_ctx.get() or "none"
        record.stage_name = stage_name_ctx.get() or "none"

        # Add service metadata
        record.service_name = os.getenv("SERVICE_NAME", "omninode-bridge")
        record.environment = os.getenv("ENVIRONMENT", "development")
        record.service_version = os.getenv("SERVICE_VERSION", "1.0.0")

        # Add extra context
        record.extra_data = extra_context_ctx.get() or {}

        return True


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def __init__(self, include_extra: bool = True):
        """Initialize JSON formatter.

        Args:
            include_extra: Whether to include extra context in output
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation context
        if hasattr(record, "correlation_id") and record.correlation_id != "none":
            log_data["correlation_id"] = record.correlation_id
        if hasattr(record, "workflow_id") and record.workflow_id != "none":
            log_data["workflow_id"] = record.workflow_id
        if hasattr(record, "request_id") and record.request_id != "none":
            log_data["request_id"] = record.request_id
        if hasattr(record, "session_id") and record.session_id != "none":
            log_data["session_id"] = record.session_id
        if hasattr(record, "user_id") and record.user_id != "none":
            log_data["user_id"] = record.user_id
        if hasattr(record, "stage_name") and record.stage_name != "none":
            log_data["stage_name"] = record.stage_name

        # Add service metadata
        if hasattr(record, "service_name"):
            log_data["service_name"] = record.service_name
        if hasattr(record, "environment"):
            log_data["environment"] = record.environment
        if hasattr(record, "service_version"):
            log_data["service_version"] = record.service_version

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra context
        if self.include_extra and hasattr(record, "extra_data"):
            extra_data = record.extra_data
            if extra_data:
                log_data["extra"] = extra_data

        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    def __init__(self):
        """Initialize human-readable formatter."""
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] [%(correlation_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def configure_logging(
    level: int = logging.INFO,
    use_json: bool = None,
    include_extra: bool = True,
) -> logging.Logger:
    """Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Use JSON formatter (default: True in production, False in dev)
        include_extra: Include extra context in JSON output

    Returns:
        Configured logger instance
    """
    # Auto-detect JSON format based on environment
    if use_json is None:
        use_json = os.getenv("ENVIRONMENT", "development") != "development"

    # Configure root logger
    logger = logging.getLogger("omninode_bridge")
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler()
    handler.addFilter(CorrelationFilter())

    # Set formatter
    if use_json:
        handler.setFormatter(JsonFormatter(include_extra=include_extra))
    else:
        handler.setFormatter(HumanReadableFormatter())

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    logger.info(
        "Logging configured",
        extra={
            "level": logging.getLevelName(level),
            "format": "json" if use_json else "human",
        },
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


@asynccontextmanager
async def correlation_context(
    correlation_id: Optional[UUID] = None,
    workflow_id: Optional[UUID] = None,
    request_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    stage_name: Optional[str] = None,
):
    """Async context manager for correlation tracking.

    Args:
        correlation_id: Request correlation ID
        workflow_id: Workflow identifier
        request_id: Request identifier
        session_id: Session identifier
        user_id: User identifier
        stage_name: Current stage/step name

    Example:
        async with correlation_context(
            correlation_id=uuid4(),
            workflow_id=uuid4(),
            stage_name="code_generation"
        ):
            logger.info("Processing stage")  # Includes correlation_id
    """
    # Set context variables
    tokens = []
    if correlation_id is not None:
        tokens.append(("correlation_id", correlation_id_ctx.set(correlation_id)))
    if workflow_id is not None:
        tokens.append(("workflow_id", workflow_id_ctx.set(workflow_id)))
    if request_id is not None:
        tokens.append(("request_id", request_id_ctx.set(request_id)))
    if session_id is not None:
        tokens.append(("session_id", session_id_ctx.set(session_id)))
    if user_id is not None:
        tokens.append(("user_id", user_id_ctx.set(user_id)))
    if stage_name is not None:
        tokens.append(("stage_name", stage_name_ctx.set(stage_name)))

    try:
        yield
    finally:
        # Reset context variables
        for name, token in tokens:
            if name == "correlation_id":
                correlation_id_ctx.reset(token)
            elif name == "workflow_id":
                workflow_id_ctx.reset(token)
            elif name == "request_id":
                request_id_ctx.reset(token)
            elif name == "session_id":
                session_id_ctx.reset(token)
            elif name == "user_id":
                user_id_ctx.reset(token)
            elif name == "stage_name":
                stage_name_ctx.reset(token)


@contextmanager
def correlation_context_sync(
    correlation_id: Optional[UUID] = None,
    workflow_id: Optional[UUID] = None,
    request_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    stage_name: Optional[str] = None,
):
    """Sync context manager for correlation tracking.

    Args:
        correlation_id: Request correlation ID
        workflow_id: Workflow identifier
        request_id: Request identifier
        session_id: Session identifier
        user_id: User identifier
        stage_name: Current stage/step name

    Example:
        with correlation_context_sync(correlation_id=uuid4()):
            logger.info("Processing")  # Includes correlation_id
    """
    # Set context variables
    tokens = []
    if correlation_id is not None:
        tokens.append(("correlation_id", correlation_id_ctx.set(correlation_id)))
    if workflow_id is not None:
        tokens.append(("workflow_id", workflow_id_ctx.set(workflow_id)))
    if request_id is not None:
        tokens.append(("request_id", request_id_ctx.set(request_id)))
    if session_id is not None:
        tokens.append(("session_id", session_id_ctx.set(session_id)))
    if user_id is not None:
        tokens.append(("user_id", user_id_ctx.set(user_id)))
    if stage_name is not None:
        tokens.append(("stage_name", stage_name_ctx.set(stage_name)))

    try:
        yield
    finally:
        # Reset context variables
        for name, token in tokens:
            if name == "correlation_id":
                correlation_id_ctx.reset(token)
            elif name == "workflow_id":
                workflow_id_ctx.reset(token)
            elif name == "request_id":
                request_id_ctx.reset(token)
            elif name == "session_id":
                session_id_ctx.reset(token)
            elif name == "user_id":
                user_id_ctx.reset(token)
            elif name == "stage_name":
                stage_name_ctx.reset(token)


def add_extra_context(**kwargs):
    """Add extra fields to log context.

    Args:
        **kwargs: Key-value pairs to add to log context

    Example:
        add_extra_context(batch_size=100, retry_count=3)
        logger.info("Processing batch")  # Includes batch_size and retry_count
    """
    current_extra = extra_context_ctx.get()
    if current_extra is None:
        current_extra = {}
    updated_extra = {**current_extra, **kwargs}
    extra_context_ctx.set(updated_extra)


def clear_extra_context():
    """Clear all extra context fields."""
    extra_context_ctx.set(None)


def get_correlation_context() -> dict[str, Any]:
    """Get current correlation context as dictionary.

    Returns:
        Dictionary with correlation IDs and context
    """
    return {
        "correlation_id": correlation_id_ctx.get(),
        "workflow_id": workflow_id_ctx.get(),
        "request_id": request_id_ctx.get(),
        "session_id": session_id_ctx.get(),
        "user_id": user_id_ctx.get(),
        "stage_name": stage_name_ctx.get(),
    }
