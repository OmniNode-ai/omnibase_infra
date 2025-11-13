"""Tests for structured logging configuration."""

import json
import logging
from uuid import uuid4

import pytest

from omninode_bridge.observability.logging_config import (
    CorrelationFilter,
    JsonFormatter,
    add_extra_context,
    clear_extra_context,
    configure_logging,
    correlation_context,
    correlation_context_sync,
    get_correlation_context,
    get_logger,
)


class TestCorrelationFilter:
    """Tests for CorrelationFilter."""

    def test_filter_adds_correlation_context(self):
        """Test that filter adds correlation context to log record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        correlation_filter = CorrelationFilter()
        correlation_filter.filter(record)

        assert hasattr(record, "correlation_id")
        assert hasattr(record, "workflow_id")
        assert hasattr(record, "request_id")
        assert hasattr(record, "service_name")
        assert hasattr(record, "environment")

    def test_filter_uses_default_values(self):
        """Test that filter uses 'none' for unset context."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        correlation_filter = CorrelationFilter()
        correlation_filter.filter(record)

        assert record.correlation_id == "none"
        assert record.workflow_id == "none"
        assert record.request_id == "none"


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_format_returns_valid_json(self):
        """Test that formatter returns valid JSON."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        # Add correlation context
        correlation_filter = CorrelationFilter()
        correlation_filter.filter(record)

        formatter = JsonFormatter()
        output = formatter.format(record)

        # Should be valid JSON
        log_data = json.loads(output)
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "test message"
        assert "timestamp" in log_data

    def test_format_includes_exception_info(self):
        """Test that formatter includes exception information."""
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="error occurred",
            args=(),
            exc_info=exc_info,
        )

        correlation_filter = CorrelationFilter()
        correlation_filter.filter(record)

        formatter = JsonFormatter()
        output = formatter.format(record)

        log_data = json.loads(output)
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert "Test error" in log_data["exception"]["message"]

    def test_format_excludes_none_correlation_ids(self):
        """Test that formatter excludes 'none' correlation IDs."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        correlation_filter = CorrelationFilter()
        correlation_filter.filter(record)

        formatter = JsonFormatter()
        output = formatter.format(record)

        log_data = json.loads(output)
        # Should not have correlation_id key when value is "none"
        assert "correlation_id" not in log_data or log_data["correlation_id"] != "none"


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_configure_logging_returns_logger(self):
        """Test that configure_logging returns logger instance."""
        logger = configure_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "omninode_bridge"

    def test_configure_logging_sets_level(self):
        """Test that configure_logging sets logging level."""
        logger = configure_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_configure_logging_json_format(self):
        """Test that configure_logging uses JSON format."""
        logger = configure_logging(use_json=True)
        assert len(logger.handlers) > 0

        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_configure_logging_removes_existing_handlers(self):
        """Test that configure_logging removes existing handlers."""
        logger = logging.getLogger("omninode_bridge")
        initial_handler_count = len(logger.handlers)

        configure_logging()

        # Should have exactly one handler after configuration
        assert len(logger.handlers) == 1


class TestGetLogger:
    """Tests for get_logger."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"


@pytest.mark.asyncio
class TestCorrelationContext:
    """Tests for correlation_context."""

    async def test_correlation_context_sets_values(self):
        """Test that correlation_context sets context values."""
        correlation_id = uuid4()
        workflow_id = uuid4()

        async with correlation_context(
            correlation_id=correlation_id, workflow_id=workflow_id
        ):
            context = get_correlation_context()
            assert context["correlation_id"] == correlation_id
            assert context["workflow_id"] == workflow_id

    async def test_correlation_context_resets_after_exit(self):
        """Test that correlation_context resets values after exit."""
        correlation_id = uuid4()

        async with correlation_context(correlation_id=correlation_id):
            pass

        context = get_correlation_context()
        assert context["correlation_id"] is None

    async def test_correlation_context_with_stage_name(self):
        """Test that correlation_context sets stage name."""
        async with correlation_context(stage_name="code_generation"):
            context = get_correlation_context()
            assert context["stage_name"] == "code_generation"

    async def test_correlation_context_with_all_fields(self):
        """Test that correlation_context sets all fields."""
        correlation_id = uuid4()
        workflow_id = uuid4()
        request_id = uuid4()

        async with correlation_context(
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            request_id=request_id,
            session_id="session123",
            user_id="user456",
            stage_name="validation",
        ):
            context = get_correlation_context()
            assert context["correlation_id"] == correlation_id
            assert context["workflow_id"] == workflow_id
            assert context["request_id"] == request_id
            assert context["session_id"] == "session123"
            assert context["user_id"] == "user456"
            assert context["stage_name"] == "validation"


class TestCorrelationContextSync:
    """Tests for correlation_context_sync."""

    def test_correlation_context_sync_sets_values(self):
        """Test that correlation_context_sync sets context values."""
        correlation_id = uuid4()

        with correlation_context_sync(correlation_id=correlation_id):
            context = get_correlation_context()
            assert context["correlation_id"] == correlation_id

    def test_correlation_context_sync_resets_after_exit(self):
        """Test that correlation_context_sync resets values after exit."""
        correlation_id = uuid4()

        with correlation_context_sync(correlation_id=correlation_id):
            pass

        context = get_correlation_context()
        assert context["correlation_id"] is None


class TestExtraContext:
    """Tests for extra context functions."""

    def test_add_extra_context(self):
        """Test adding extra context."""
        clear_extra_context()  # Clear first
        add_extra_context(batch_size=100, retry_count=3)

        # Create log record and check extra data is added
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        correlation_filter = CorrelationFilter()
        correlation_filter.filter(record)

        assert record.extra_data["batch_size"] == 100
        assert record.extra_data["retry_count"] == 3

    def test_clear_extra_context(self):
        """Test clearing extra context."""
        add_extra_context(key="value")
        clear_extra_context()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )

        correlation_filter = CorrelationFilter()
        correlation_filter.filter(record)

        assert record.extra_data == {}
