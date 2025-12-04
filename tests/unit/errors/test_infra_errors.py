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

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext
from omnibase_infra.errors.infra_errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraResourceUnavailableError,
    InfraTimeoutError,
    ProtocolConfigurationError,
    RuntimeHostError,
    SecretResolutionError,
)


class TestModelInfraErrorContext:
    """Tests for ModelInfraErrorContext configuration model."""

    def test_basic_instantiation(self) -> None:
        """Test basic context model instantiation."""
        context = ModelInfraErrorContext()
        assert context.transport_type is None
        assert context.operation is None
        assert context.target_name is None
        assert context.correlation_id is None

    def test_with_all_fields(self) -> None:
        """Test context model with all fields populated."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="process_request",
            target_name="api-gateway",
            correlation_id=correlation_id,
        )
        assert context.transport_type == EnumInfraTransportType.HTTP
        assert context.operation == "process_request"
        assert context.target_name == "api-gateway"
        assert context.correlation_id == correlation_id

    def test_immutability(self) -> None:
        """Test that context model is immutable (frozen)."""
        context = ModelInfraErrorContext(transport_type=EnumInfraTransportType.HTTP)
        with pytest.raises(ValidationError):
            context.transport_type = EnumInfraTransportType.DATABASE  # type: ignore[misc]


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
            transport_type=EnumInfraTransportType.HTTP,
            operation="process_request",
            target_name="api-endpoint",
            correlation_id=correlation_id,
        )
        error = RuntimeHostError("Test error", context=context)
        assert error.model.correlation_id == correlation_id
        assert error.model.context["transport_type"] == EnumInfraTransportType.HTTP
        assert error.model.context["operation"] == "process_request"
        assert error.model.context["target_name"] == "api-endpoint"

    def test_with_error_code(self) -> None:
        """Test error with explicit error code."""
        error = RuntimeHostError(
            "Test error", error_code=EnumCoreErrorCode.OPERATION_FAILED
        )
        assert error.model.error_code == EnumCoreErrorCode.OPERATION_FAILED

    def test_with_extra_context(self) -> None:
        """Test error with extra context via kwargs."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="process_request",
        )
        error = RuntimeHostError(
            "Test error",
            context=context,
            retry_count=3,
            endpoint="/api/v1/users",
        )
        assert error.model.context["transport_type"] == EnumInfraTransportType.HTTP
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


class TestProtocolConfigurationError:
    """Tests for ProtocolConfigurationError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = ProtocolConfigurationError("Invalid config")
        assert "Invalid config" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="validate_config",
        )
        error = ProtocolConfigurationError("Invalid config", context=context)
        assert error.model.context["transport_type"] == EnumInfraTransportType.HTTP
        assert error.model.context["operation"] == "validate_config"

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = ProtocolConfigurationError("Config error")
        assert error.model.error_code == EnumCoreErrorCode.INVALID_CONFIGURATION

    def test_error_chaining(self) -> None:
        """Test error chaining from original exception."""
        context = ModelInfraErrorContext(transport_type=EnumInfraTransportType.DATABASE)
        config_error = KeyError("missing_key")
        try:
            raise ProtocolConfigurationError(
                "Missing required config key", context=context
            ) from config_error
        except ProtocolConfigurationError as e:
            assert e.__cause__ == config_error
            assert e.model.context["transport_type"] == EnumInfraTransportType.DATABASE


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
            target_name="vault",
            operation="get_secret",
        )
        error = SecretResolutionError(
            "Secret not found",
            context=context,
            secret_key="db_password",  # noqa: S106
        )
        assert error.model.context["target_name"] == "vault"
        assert error.model.context["operation"] == "get_secret"
        assert error.model.context["secret_key"] == "db_password"

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = SecretResolutionError("Secret error")
        assert error.model.error_code == EnumCoreErrorCode.RESOURCE_NOT_FOUND

    def test_error_chaining(self) -> None:
        """Test error chaining from vault client error."""
        context = ModelInfraErrorContext(target_name="vault")
        vault_error = ConnectionError("Vault unreachable")
        try:
            raise SecretResolutionError(
                "Cannot resolve secret", context=context
            ) from vault_error
        except SecretResolutionError as e:
            assert e.__cause__ == vault_error
            assert e.model.context["target_name"] == "vault"


class TestInfraConnectionError:
    """Tests for InfraConnectionError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = InfraConnectionError("Connection failed")
        assert "Connection failed" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and connection details."""
        context = ModelInfraErrorContext(target_name="postgresql")
        error = InfraConnectionError(
            "Database connection failed",
            context=context,
            host="db.example.com",
            port=5432,
        )
        assert error.model.context["target_name"] == "postgresql"
        assert error.model.context["host"] == "db.example.com"
        assert error.model.context["port"] == 5432

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = InfraConnectionError("Connection error")
        assert error.model.error_code == EnumCoreErrorCode.DATABASE_CONNECTION_ERROR

    def test_error_chaining(self) -> None:
        """Test error chaining from connection exception."""
        context = ModelInfraErrorContext(target_name="redis")
        conn_error = OSError("Connection refused")
        try:
            raise InfraConnectionError(
                "Failed to connect", context=context, host="localhost", port=6379
            ) from conn_error
        except InfraConnectionError as e:
            assert e.__cause__ == conn_error
            assert e.model.context["target_name"] == "redis"
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
            target_name="postgresql",
        )
        error = InfraTimeoutError(
            "Query timeout exceeded",
            context=context,
            timeout_seconds=30,
        )
        assert error.model.context["operation"] == "execute_query"
        assert error.model.context["timeout_seconds"] == 30
        assert error.model.context["target_name"] == "postgresql"

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
            target_name="consul",
            operation="authenticate",
        )
        error = InfraAuthenticationError(
            "Invalid credentials",
            context=context,
            username="admin",
        )
        assert error.model.context["target_name"] == "consul"
        assert error.model.context["operation"] == "authenticate"
        assert error.model.context["username"] == "admin"

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = InfraAuthenticationError("Auth error")
        assert error.model.error_code == EnumCoreErrorCode.AUTHENTICATION_ERROR

    def test_error_chaining(self) -> None:
        """Test error chaining from auth exception."""
        context = ModelInfraErrorContext(
            target_name="vault",
            operation="login",
        )
        auth_error = PermissionError("Access denied")
        try:
            raise InfraAuthenticationError(
                "Vault authentication failed", context=context
            ) from auth_error
        except InfraAuthenticationError as e:
            assert e.__cause__ == auth_error
            assert e.model.context["target_name"] == "vault"


class TestInfraResourceUnavailableError:
    """Tests for InfraResourceUnavailableError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = InfraResourceUnavailableError("Resource unavailable")
        assert "Resource unavailable" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and details."""
        context = ModelInfraErrorContext(target_name="kafka")
        error = InfraResourceUnavailableError(
            "Kafka broker unavailable",
            context=context,
            host="kafka.example.com",
            port=9092,
            retry_count=3,
        )
        assert error.model.context["target_name"] == "kafka"
        assert error.model.context["host"] == "kafka.example.com"
        assert error.model.context["port"] == 9092
        assert error.model.context["retry_count"] == 3

    def test_error_code_mapping(self) -> None:
        """Test that error uses appropriate CoreErrorCode."""
        error = InfraResourceUnavailableError("Resource error")
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_chaining(self) -> None:
        """Test error chaining from exception."""
        context = ModelInfraErrorContext(target_name="consul")
        resource_error = ConnectionRefusedError("Not responding")
        try:
            raise InfraResourceUnavailableError(
                "Consul unavailable",
                context=context,
                host="consul.local",
                port=8500,
            ) from resource_error
        except InfraResourceUnavailableError as e:
            assert e.__cause__ == resource_error
            assert e.model.context["target_name"] == "consul"
            assert e.model.context["port"] == 8500


class TestAllErrorsInheritance:
    """Test that all infrastructure errors properly inherit from RuntimeHostError."""

    def test_all_errors_inherit_from_runtime_host_error(self) -> None:
        """Test inheritance chain for all error classes."""
        errors = [
            ProtocolConfigurationError("test"),
            SecretResolutionError("test"),
            InfraConnectionError("test"),
            InfraTimeoutError("test"),
            InfraAuthenticationError("test"),
            InfraResourceUnavailableError("test"),
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
            ProtocolConfigurationError("test", context=context),
            SecretResolutionError("test", context=context),
            InfraConnectionError("test", context=context),
            InfraTimeoutError("test", context=context),
            InfraAuthenticationError("test", context=context),
            InfraResourceUnavailableError("test", context=context),
        ]

        for error in errors:
            assert error.model.correlation_id == correlation_id

    def test_all_errors_support_transport_type(self) -> None:
        """Test that all errors support transport_type via context model."""
        transport_types = [
            EnumInfraTransportType.HTTP,
            EnumInfraTransportType.VAULT,
            EnumInfraTransportType.DATABASE,
            EnumInfraTransportType.KAFKA,
            EnumInfraTransportType.CONSUL,
            EnumInfraTransportType.REDIS,
        ]
        errors = [
            ProtocolConfigurationError(
                "test",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.HTTP
                ),
            ),
            SecretResolutionError(
                "test",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VAULT
                ),
            ),
            InfraConnectionError(
                "test",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE
                ),
            ),
            InfraTimeoutError(
                "test",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA
                ),
            ),
            InfraAuthenticationError(
                "test",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.CONSUL
                ),
            ),
            InfraResourceUnavailableError(
                "test",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.REDIS
                ),
            ),
        ]

        for error, expected_type in zip(errors, transport_types, strict=True):
            assert error.model.context["transport_type"] == expected_type

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
            ProtocolConfigurationError(
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
            InfraResourceUnavailableError(
                "test", context=ModelInfraErrorContext(operation="check_health")
            ),
        ]

        for error, operation in zip(errors, operations, strict=True):
            assert error.model.context["operation"] == operation

    def test_all_errors_support_target_name(self) -> None:
        """Test that all errors support target_name via context model."""
        targets = ["api", "vault", "postgresql", "kafka", "consul", "redis"]
        errors = [
            ProtocolConfigurationError(
                "test", context=ModelInfraErrorContext(target_name="api")
            ),
            SecretResolutionError(
                "test", context=ModelInfraErrorContext(target_name="vault")
            ),
            InfraConnectionError(
                "test", context=ModelInfraErrorContext(target_name="postgresql")
            ),
            InfraTimeoutError(
                "test", context=ModelInfraErrorContext(target_name="kafka")
            ),
            InfraAuthenticationError(
                "test", context=ModelInfraErrorContext(target_name="consul")
            ),
            InfraResourceUnavailableError(
                "test", context=ModelInfraErrorContext(target_name="redis")
            ),
        ]

        for error, target in zip(errors, targets, strict=True):
            assert error.model.context["target_name"] == target
