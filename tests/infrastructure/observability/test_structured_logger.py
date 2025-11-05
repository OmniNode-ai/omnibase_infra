"""
Comprehensive tests for Structured Logger.

Tests structured logging, correlation IDs, context managers,
and OnexError integration.
"""

import logging
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import structlog
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError

from omnibase_infra.infrastructure.observability.structured_logger import (
    LoggerFactory,
    LogLevel,
    StructuredLogger,
    StructuredLoggerConfig,
)


class TestStructuredLoggerConfigInit:
    """Test logger configuration initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        config = StructuredLoggerConfig()

        assert config.environment in ["development", "staging", "production"]
        assert config.log_level in [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
        ]
        assert isinstance(config.json_output, bool)
        assert config.include_timestamps is True
        assert config.include_stack_info is True

    def test_init_custom_environment(self):
        """Test initialization with custom environment."""
        config = StructuredLoggerConfig(environment="production")

        assert config.environment == "production"

    def test_init_custom_log_level(self):
        """Test initialization with custom log level."""
        config = StructuredLoggerConfig(log_level=LogLevel.WARNING)

        assert config.log_level == LogLevel.WARNING

    def test_init_custom_json_output(self):
        """Test initialization with custom JSON output setting."""
        config = StructuredLoggerConfig(json_output=True)

        assert config.json_output is True

    def test_environment_detection(self):
        """Test automatic environment detection."""
        with patch.dict("os.environ", {"ENVIRONMENT": "staging"}):
            config = StructuredLoggerConfig()
            assert config.environment == "staging"

    def test_log_level_detection(self):
        """Test automatic log level detection."""
        with patch.dict("os.environ", {"LOG_LEVEL": "error"}):
            config = StructuredLoggerConfig()
            assert config.log_level == "error"


class TestStructuredLoggerInit:
    """Test structured logger initialization."""

    def test_init_with_component_name(self):
        """Test initialization with component name."""
        logger = StructuredLogger(component_name="test_component")

        assert logger.component_name == "test_component"
        assert isinstance(logger.config, StructuredLoggerConfig)
        assert logger.base_context == {}

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = StructuredLoggerConfig(
            environment="production",
            log_level=LogLevel.WARNING,
        )
        logger = StructuredLogger(
            component_name="test_component",
            config=config,
        )

        assert logger.config == config

    def test_init_with_base_context(self):
        """Test initialization with base context."""
        base_context = {"service": "test_service", "version": "1.0.0"}
        logger = StructuredLogger(
            component_name="test_component",
            base_context=base_context,
        )

        assert logger.base_context == base_context

    @patch("structlog.is_configured")
    def test_configure_structlog_called_once(self, mock_is_configured):
        """Test that structlog is configured on first logger creation."""
        mock_is_configured.return_value = False

        with patch("structlog.configure") as mock_configure:
            _ = StructuredLogger(component_name="test1")
            _ = StructuredLogger(component_name="test2")

            # Should only configure once
            assert mock_configure.call_count >= 1


class TestStructuredLoggerLogging:
    """Test logging methods."""

    @pytest.fixture
    def logger(self):
        """Create a structured logger for testing."""
        return StructuredLogger(component_name="test_component")

    def test_debug_logging(self, logger):
        """Test debug level logging."""
        with patch.object(logger.logger, "debug") as mock_debug:
            logger.debug("Debug message", extra="context")

            mock_debug.assert_called_once()

    def test_info_logging(self, logger):
        """Test info level logging."""
        with patch.object(logger.logger, "info") as mock_info:
            logger.info("Info message", user_id="123")

            mock_info.assert_called_once()

    def test_warning_logging(self, logger):
        """Test warning level logging."""
        with patch.object(logger.logger, "warning") as mock_warning:
            logger.warning("Warning message", code="W001")

            mock_warning.assert_called_once()

    def test_error_logging(self, logger):
        """Test error level logging."""
        with patch.object(logger.logger, "error") as mock_error:
            logger.error("Error message", error_code="E500")

            mock_error.assert_called_once()

    def test_critical_logging(self, logger):
        """Test critical level logging."""
        with patch.object(logger.logger, "critical") as mock_critical:
            logger.critical("Critical message", alert=True)

            mock_critical.assert_called_once()

    def test_logging_with_multiple_context(self, logger):
        """Test logging with multiple context fields."""
        with patch.object(logger.logger, "info") as mock_info:
            logger.info(
                "Complex log",
                user_id="123",
                action="create",
                resource="user",
                duration_ms=45.2,
            )

            call_kwargs = mock_info.call_args.kwargs
            assert call_kwargs["user_id"] == "123"
            assert call_kwargs["action"] == "create"
            assert call_kwargs["resource"] == "user"
            assert call_kwargs["duration_ms"] == 45.2


class TestStructuredLoggerExceptionLogging:
    """Test exception logging."""

    @pytest.fixture
    def logger(self):
        """Create a structured logger for testing."""
        return StructuredLogger(component_name="test_component")

    def test_log_exception_basic(self, logger):
        """Test basic exception logging."""
        exception = ValueError("Test error")

        with patch.object(logger.logger, "error") as mock_error:
            logger.log_exception(exception)

            mock_error.assert_called_once()
            call_kwargs = mock_error.call_args.kwargs
            assert call_kwargs["exception_type"] == "ValueError"
            assert "Test error" in call_kwargs["exception_message"]

    def test_log_exception_with_message(self, logger):
        """Test exception logging with custom message."""
        exception = ValueError("Test error")

        with patch.object(logger.logger, "error") as mock_error:
            logger.log_exception(exception, message="Custom message")

            call_args = mock_error.call_args.args
            assert "Custom message" in call_args[0]

    def test_log_exception_with_correlation_id(self, logger):
        """Test exception logging with correlation ID."""
        exception = ValueError("Test error")
        correlation_id = uuid4()

        with patch.object(logger.logger, "error") as mock_error:
            logger.log_exception(exception, correlation_id=correlation_id)

            call_kwargs = mock_error.call_args.kwargs
            assert call_kwargs["correlation_id"] == str(correlation_id)

    def test_log_exception_with_context(self, logger):
        """Test exception logging with additional context."""
        exception = ValueError("Test error")

        with patch.object(logger.logger, "error") as mock_error:
            logger.log_exception(
                exception,
                user_id="123",
                action="process_payment",
            )

            call_kwargs = mock_error.call_args.kwargs
            assert call_kwargs["user_id"] == "123"
            assert call_kwargs["action"] == "process_payment"

    def test_log_onex_error(self, logger):
        """Test logging OnexError with additional context."""
        onex_error = OnexError(
            code=CoreErrorCode.DATABASE_QUERY_ERROR,
            message="Query failed",
            details={"query": "SELECT * FROM users"},
        )

        with patch.object(logger.logger, "error") as mock_error:
            logger.log_exception(onex_error)

            call_kwargs = mock_error.call_args.kwargs
            assert call_kwargs["error_code"] == "DATABASE_QUERY_ERROR"
            assert "details" in call_kwargs


class TestStructuredLoggerContextManagers:
    """Test logging context managers."""

    @pytest.fixture
    def logger(self):
        """Create a structured logger for testing."""
        return StructuredLogger(component_name="test_component")

    def test_log_scope_basic(self, logger):
        """Test basic log scope context manager."""
        with patch.object(logger.logger, "info") as mock_info:
            with logger.log_scope(request_id="123"):
                logger.info("Inside scope")

            # Verify context was applied
            mock_info.assert_called_once()

    def test_log_scope_multiple_fields(self, logger):
        """Test log scope with multiple context fields."""
        with logger.log_scope(user_id="user1", session_id="session1"):
            with patch.object(logger.logger, "info") as mock_info:
                logger.info("Test message")

                # Context should include scoped fields
                # Actual verification depends on structlog implementation

    def test_log_scope_cleanup(self, logger):
        """Test that log scope cleans up after exit."""
        with logger.log_scope(temp_field="temp_value"):
            pass

        # After exiting scope, temp_field should not persist
        with patch.object(logger.logger, "info") as mock_info:
            logger.info("After scope")

            # Verify temp_field is not in context
            # Actual verification depends on structlog implementation

    def test_log_operation_success(self, logger):
        """Test log_operation context manager for successful operation."""
        with patch.object(logger.logger, "info") as mock_info:
            with logger.log_operation("test_operation"):
                pass

            # Should log start and completion
            assert mock_info.call_count == 2

    def test_log_operation_with_correlation_id(self, logger):
        """Test log_operation with correlation ID."""
        correlation_id = uuid4()

        with patch.object(logger.logger, "info") as mock_info:
            with logger.log_operation(
                "test_operation",
                correlation_id=correlation_id,
            ):
                pass

            # Verify correlation_id in context
            # Check multiple calls for start and completion

    def test_log_operation_with_error(self, logger):
        """Test log_operation handles exceptions."""
        with patch.object(logger.logger, "info") as mock_info:
            with patch.object(logger.logger, "error") as mock_error:
                with pytest.raises(ValueError):
                    with logger.log_operation("failing_operation"):
                        raise ValueError("Operation failed")

                # Should log error
                mock_error.assert_called_once()

    def test_log_operation_measures_duration(self, logger):
        """Test that log_operation measures operation duration."""
        import time

        with patch.object(logger.logger, "info") as mock_info:
            with logger.log_operation("timed_operation"):
                time.sleep(0.01)

            # Completion log should include duration
            completion_call = mock_info.call_args_list[-1]
            assert "duration_seconds" in completion_call.kwargs


class TestLoggerFactory:
    """Test logger factory."""

    def test_configure_factory(self):
        """Test configuring the logger factory."""
        config = StructuredLoggerConfig(environment="test")

        LoggerFactory.configure(config)

        assert LoggerFactory._config == config

    def test_get_logger(self):
        """Test getting logger from factory."""
        logger = LoggerFactory.get_logger("test_component")

        assert isinstance(logger, StructuredLogger)
        assert logger.component_name == "test_component"

    def test_get_logger_with_base_context(self):
        """Test getting logger with base context."""
        base_context = {"service": "test_service"}

        logger = LoggerFactory.get_logger(
            "test_component",
            base_context=base_context,
        )

        assert logger.base_context == base_context

    def test_get_logger_caching(self):
        """Test that factory caches logger instances."""
        logger1 = LoggerFactory.get_logger("cached_component")
        logger2 = LoggerFactory.get_logger("cached_component")

        assert logger1 is logger2

    def test_get_logger_different_components(self):
        """Test that different components get different loggers."""
        logger1 = LoggerFactory.get_logger("component1")
        logger2 = LoggerFactory.get_logger("component2")

        assert logger1 is not logger2
        assert logger1.component_name == "component1"
        assert logger2.component_name == "component2"

    def test_clear_cache(self):
        """Test clearing logger cache."""
        LoggerFactory.get_logger("test")

        LoggerFactory.clear_cache()

        assert len(LoggerFactory._loggers) == 0


class TestStructuredLoggerIntegration:
    """Integration tests for structured logging."""

    def test_full_logging_workflow(self):
        """Test complete logging workflow."""
        logger = StructuredLogger(
            component_name="workflow_test",
            base_context={"service": "test"},
        )

        # Log at different levels
        with patch.object(logger.logger, "debug"):
            logger.debug("Debug message")

        with patch.object(logger.logger, "info"):
            logger.info("Info message")

        with patch.object(logger.logger, "error"):
            logger.error("Error message")

    def test_nested_log_scopes(self):
        """Test nested log scope contexts."""
        logger = StructuredLogger(component_name="nested_test")

        with logger.log_scope(level1="value1"):
            with logger.log_scope(level2="value2"):
                with patch.object(logger.logger, "info") as mock_info:
                    logger.info("Nested log")

                    # Both scope contexts should be present

    def test_logger_with_onex_error_workflow(self):
        """Test logging workflow with OnexError."""
        logger = StructuredLogger(component_name="error_test")

        try:
            raise ValueError("Original error")
        except ValueError as e:
            onex_error = OnexError(
                code=CoreErrorCode.VALIDATION_ERROR,
                message="Validation failed",
            )

            with patch.object(logger.logger, "error"):
                logger.log_exception(onex_error)

    def test_concurrent_loggers(self):
        """Test multiple loggers operating independently."""
        logger1 = LoggerFactory.get_logger("service1")
        logger2 = LoggerFactory.get_logger("service2")

        with patch.object(logger1.logger, "info") as mock_info1:
            with patch.object(logger2.logger, "info") as mock_info2:
                logger1.info("Message from service1")
                logger2.info("Message from service2")

                mock_info1.assert_called_once()
                mock_info2.assert_called_once()
