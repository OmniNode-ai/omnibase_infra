"""
Comprehensive tests for infrastructure error classes.

Tests follow TDD approach:
1. Write tests first (red phase)
2. Implement error classes (green phase)
3. Refactor if needed (refactor phase)

All tests validate:
- Error class instantiation
- Inheritance chain
- Error chaining (raise ... from e)
- Structured context fields via ModelInfraErrorContext
- Error code mapping
- Required fields storage
"""

from uuid import uuid4

import pytest
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors import ModelOnexError
from pydantic import ValidationError

from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.errors.infra_errors import (
    HandlerConfigurationError,
    InfraAuthenticationError,
    InfraConnectionError,
    InfraServiceUnavailableError,
    InfraTimeoutError,
    RuntimeHostError,
    SecretResolutionError,
)


class TestModelInfraErrorContext:
    """Tests for ModelInfraErrorContext configuration model."""

    def test_basic_instantiation(self) -> None:
        """Test basic context model instantiation."""
        context = ModelInfraErrorContext()
        assert context.handler_type is None
        assert context.operation is None
        assert context.service_name is None
        assert context.correlation_id is None

    def test_with_all_fields(self) -> None:
        """Test context model with all fields populated."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            handler_type="http",
            operation="process_request",
            service_name="api-gateway",
            correlation_id=correlation_id,
        )
        assert context.handler_type == "http"
        assert context.operation == "process_request"
        assert context.service_name == "api-gateway"
        assert context.correlation_id == correlation_id

    def test_immutability(self) -> None:
        """Test that context model is immutable (frozen)."""
        context = ModelInfraErrorContext(handler_type="http")
        with pytest.raises(ValidationError):
            context.handler_type = "db"  # type: ignore[misc]


class TestRuntimeHostError:
    """Tests for RuntimeHostError base class."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = RuntimeHostError("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, ModelOnexError)

    def test_with_context_model(self) -> None:
        """Test error with context model."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            handler_type="http",
            operation="process_request",
            service_name="api-service",
            correlation_id=correlation_id,
        )
        error = RuntimeHostError("Test error", context=context)
        assert error.model.correlation_id == correlation_id
        assert error.model.context["handler_type"] == "http"
        assert error.model.context["operation"] == "process_request"
        assert error.model.context["service_name"] == "api-service"

    def test_with_error_code(self) -> None:
        """Test error with explicit error code."""
        error = RuntimeHostError(
            "Test error", error_code=EnumCoreErrorCode.OPERATION_FAILED
        )
        assert error.model.error_code == EnumCoreErrorCode.OPERATION_FAILED

    def test_with_extra_context(self) -> None:
        """Test error with extra context via kwargs."""
        context = ModelInfraErrorContext(
            handler_type="http",
            operation="process_request",
        )
        error = RuntimeHostError(
            "Test error",
            context=context,
            retry_count=3,
            endpoint="/api/v1/users",
        )
        assert error.model.context["handler_type"] == "http"
        assert error.model.context["retry_count"] == 3
        assert error.model.context["endpoint"] == "/api/v1/users"

    def test_error_chaining(self) -> None:
        """Test error chaining with 'raise ... from e' pattern."""
        original_error = ValueError("Original error")
        try:
            raise RuntimeHostError("Wrapped error") from original_error
        except RuntimeHostError as e:
            assert e.__cause__ == original_error
            assert isinstance(e.__cause__, ValueError)

    def test_inheritance_chain(self) -> None:
        """Test that RuntimeHostError properly inherits from ModelOnexError."""
        error = RuntimeHostError("Test error")
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)
        assert isinstance(error, Exception)


class TestHandlerConfigurationError:
    """Tests for HandlerConfigurationError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = HandlerConfigurationError("Invalid handler config")
        assert "Invalid handler config" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model."""
        context = ModelInfraErrorContext(
            handler_type="http",
            operation="validate_config",
        )
        error = HandlerConfigurationError("Invalid config", context=context)
        assert error.model.context["handler_type"] == "http"
        assert error.model.context["operation"] == "validate_config"

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = HandlerConfigurationError("Config error")
        assert error.model.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION

    def test_error_chaining(self) -> None:
        """Test error chaining from original exception."""
        context = ModelInfraErrorContext(handler_type="db")
        config_error = KeyError("missing_key")
        try:
            raise HandlerConfigurationError(
                "Missing required config key", context=context
            ) from config_error
        except HandlerConfigurationError as e:
            assert e.__cause__ == config_error
            assert e.model.context["handler_type"] == "db"


class TestSecretResolutionError:
    """Tests for SecretResolutionError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = SecretResolutionError("Failed to resolve secret")
        assert "Failed to resolve secret" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and extra context."""
        context = ModelInfraErrorContext(
            service_name="vault",
            operation="get_secret",
        )
        error = SecretResolutionError(
            "Secret not found",
            context=context,
            secret_key="db_password",  # noqa: S106
        )
        assert error.model.context["service_name"] == "vault"
        assert error.model.context["operation"] == "get_secret"
        assert error.model.context["secret_key"] == "db_password"

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = SecretResolutionError("Secret error")
        assert error.model.error_code == EnumCoreErrorCode.RESOURCE_NOT_FOUND

    def test_error_chaining(self) -> None:
        """Test error chaining from vault client error."""
        context = ModelInfraErrorContext(service_name="vault")
        vault_error = ConnectionError("Vault unreachable")
        try:
            raise SecretResolutionError(
                "Cannot resolve secret", context=context
            ) from vault_error
        except SecretResolutionError as e:
            assert e.__cause__ == vault_error
            assert e.model.context["service_name"] == "vault"


class TestInfraConnectionError:
    """Tests for InfraConnectionError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = InfraConnectionError("Connection failed")
        assert "Connection failed" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and connection details."""
        context = ModelInfraErrorContext(service_name="postgresql")
        error = InfraConnectionError(
            "Database connection failed",
            context=context,
            host="db.example.com",
            port=5432,
        )
        assert error.model.context["service_name"] == "postgresql"
        assert error.model.context["host"] == "db.example.com"
        assert error.model.context["port"] == 5432

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = InfraConnectionError("Connection error")
        assert error.model.error_code == EnumCoreErrorCode.DATABASE_CONNECTION_ERROR

    def test_error_chaining(self) -> None:
        """Test error chaining from connection exception."""
        context = ModelInfraErrorContext(service_name="redis")
        conn_error = OSError("Connection refused")
        try:
            raise InfraConnectionError(
                "Failed to connect", context=context, host="localhost", port=6379
            ) from conn_error
        except InfraConnectionError as e:
            assert e.__cause__ == conn_error
            assert e.model.context["service_name"] == "redis"
            assert e.model.context["port"] == 6379


class TestInfraTimeoutError:
    """Tests for InfraTimeoutError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = InfraTimeoutError("Operation timed out")
        assert "Operation timed out" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and timeout details."""
        context = ModelInfraErrorContext(
            operation="execute_query",
            service_name="postgresql",
        )
        error = InfraTimeoutError(
            "Query timeout exceeded",
            context=context,
            timeout_seconds=30,
        )
        assert error.model.context["operation"] == "execute_query"
        assert error.model.context["timeout_seconds"] == 30
        assert error.model.context["service_name"] == "postgresql"

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = InfraTimeoutError("Timeout error")
        assert error.model.error_code == EnumCoreErrorCode.TIMEOUT_ERROR

    def test_error_chaining(self) -> None:
        """Test error chaining from timeout exception."""
        context = ModelInfraErrorContext(operation="select")
        timeout = TimeoutError("Operation exceeded deadline")
        try:
            raise InfraTimeoutError(
                "Database query timeout", context=context, timeout_seconds=10
            ) from timeout
        except InfraTimeoutError as e:
            assert e.__cause__ == timeout
            assert e.model.context["operation"] == "select"
            assert e.model.context["timeout_seconds"] == 10


class TestInfraAuthenticationError:
    """Tests for InfraAuthenticationError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = InfraAuthenticationError("Authentication failed")
        assert "Authentication failed" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and auth details."""
        context = ModelInfraErrorContext(
            service_name="consul",
            operation="authenticate",
        )
        error = InfraAuthenticationError(
            "Invalid credentials",
            context=context,
            username="admin",
        )
        assert error.model.context["service_name"] == "consul"
        assert error.model.context["operation"] == "authenticate"
        assert error.model.context["username"] == "admin"

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = InfraAuthenticationError("Auth error")
        assert error.model.error_code == EnumCoreErrorCode.AUTHENTICATION_ERROR

    def test_error_chaining(self) -> None:
        """Test error chaining from auth exception."""
        context = ModelInfraErrorContext(
            service_name="vault",
            operation="login",
        )
        auth_error = PermissionError("Access denied")
        try:
            raise InfraAuthenticationError(
                "Vault authentication failed", context=context
            ) from auth_error
        except InfraAuthenticationError as e:
            assert e.__cause__ == auth_error
            assert e.model.context["service_name"] == "vault"


class TestInfraServiceUnavailableError:
    """Tests for InfraServiceUnavailableError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = InfraServiceUnavailableError("Service unavailable")
        assert "Service unavailable" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and service details."""
        context = ModelInfraErrorContext(service_name="kafka")
        error = InfraServiceUnavailableError(
            "Kafka broker unavailable",
            context=context,
            host="kafka.example.com",
            port=9092,
            retry_count=3,
        )
        assert error.model.context["service_name"] == "kafka"
        assert error.model.context["host"] == "kafka.example.com"
        assert error.model.context["port"] == 9092
        assert error.model.context["retry_count"] == 3

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = InfraServiceUnavailableError("Service error")
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_chaining(self) -> None:
        """Test error chaining from service exception."""
        context = ModelInfraErrorContext(service_name="consul")
        service_error = ConnectionRefusedError("Service not responding")
        try:
            raise InfraServiceUnavailableError(
                "Consul unavailable",
                context=context,
                host="consul.local",
                port=8500,
            ) from service_error
        except InfraServiceUnavailableError as e:
            assert e.__cause__ == service_error
            assert e.model.context["service_name"] == "consul"
            assert e.model.context["port"] == 8500


class TestAllErrorsInheritance:
    """Test that all infrastructure errors properly inherit from RuntimeHostError."""

    def test_all_errors_inherit_from_runtime_host_error(self) -> None:
        """Test inheritance chain for all error classes."""
        errors = [
            HandlerConfigurationError("test"),
            SecretResolutionError("test"),
            InfraConnectionError("test"),
            InfraTimeoutError("test"),
            InfraAuthenticationError("test"),
            InfraServiceUnavailableError("test"),
        ]

        for error in errors:
            assert isinstance(error, RuntimeHostError)
            assert isinstance(error, ModelOnexError)
            assert isinstance(error, Exception)


class TestStructuredFieldsComprehensive:
    """Comprehensive tests for structured field support across all errors."""

    def test_all_errors_support_correlation_id(self) -> None:
        """Test that all errors support correlation_id via context model."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(correlation_id=correlation_id)
        errors = [
            HandlerConfigurationError("test", context=context),
            SecretResolutionError("test", context=context),
            InfraConnectionError("test", context=context),
            InfraTimeoutError("test", context=context),
            InfraAuthenticationError("test", context=context),
            InfraServiceUnavailableError("test", context=context),
        ]

        for error in errors:
            assert error.model.correlation_id == correlation_id

    def test_all_errors_support_handler_type(self) -> None:
        """Test that all errors support handler_type via context model."""
        handler_types = ["http", "vault", "db", "kafka", "consul", "redis"]
        errors = [
            HandlerConfigurationError(
                "test", context=ModelInfraErrorContext(handler_type="http")
            ),
            SecretResolutionError(
                "test", context=ModelInfraErrorContext(handler_type="vault")
            ),
            InfraConnectionError(
                "test", context=ModelInfraErrorContext(handler_type="db")
            ),
            InfraTimeoutError(
                "test", context=ModelInfraErrorContext(handler_type="kafka")
            ),
            InfraAuthenticationError(
                "test", context=ModelInfraErrorContext(handler_type="consul")
            ),
            InfraServiceUnavailableError(
                "test", context=ModelInfraErrorContext(handler_type="redis")
            ),
        ]

        for error, expected_type in zip(errors, handler_types, strict=True):
            assert error.model.context["handler_type"] == expected_type

    def test_all_errors_support_operation(self) -> None:
        """Test that all errors support operation via context model."""
        operations = [
            "validate",
            "resolve",
            "connect",
            "execute",
            "authenticate",
            "check_health",
        ]
        errors = [
            HandlerConfigurationError(
                "test", context=ModelInfraErrorContext(operation="validate")
            ),
            SecretResolutionError(
                "test", context=ModelInfraErrorContext(operation="resolve")
            ),
            InfraConnectionError(
                "test", context=ModelInfraErrorContext(operation="connect")
            ),
            InfraTimeoutError(
                "test", context=ModelInfraErrorContext(operation="execute")
            ),
            InfraAuthenticationError(
                "test", context=ModelInfraErrorContext(operation="authenticate")
            ),
            InfraServiceUnavailableError(
                "test", context=ModelInfraErrorContext(operation="check_health")
            ),
        ]

        for error, operation in zip(errors, operations, strict=True):
            assert error.model.context["operation"] == operation

    def test_all_errors_support_service_name(self) -> None:
        """Test that all errors support service_name via context model."""
        services = ["api", "vault", "postgresql", "kafka", "consul", "redis"]
        errors = [
            HandlerConfigurationError(
                "test", context=ModelInfraErrorContext(service_name="api")
            ),
            SecretResolutionError(
                "test", context=ModelInfraErrorContext(service_name="vault")
            ),
            InfraConnectionError(
                "test", context=ModelInfraErrorContext(service_name="postgresql")
            ),
            InfraTimeoutError(
                "test", context=ModelInfraErrorContext(service_name="kafka")
            ),
            InfraAuthenticationError(
                "test", context=ModelInfraErrorContext(service_name="consul")
            ),
            InfraServiceUnavailableError(
                "test", context=ModelInfraErrorContext(service_name="redis")
            ),
        ]

        for error, service in zip(errors, services, strict=True):
            assert error.model.context["service_name"] == service
