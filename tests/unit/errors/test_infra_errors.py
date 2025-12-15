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
    InfraTimeoutError,
    InfraUnavailableError,
    ProtocolConfigurationError,
    RuntimeHostError,
    SecretResolutionError,
)


class TestModelInfraErrorContextWithCorrelation:
    """Tests for ModelInfraErrorContext.with_correlation() factory method."""

    def test_with_correlation_generates_uuid_when_none(self) -> None:
        """Test that with_correlation generates a UUID when none is provided."""
        context = ModelInfraErrorContext.with_correlation()
        assert context.correlation_id is not None

    def test_with_correlation_uses_provided_uuid(self) -> None:
        """Test that with_correlation uses the provided UUID when given."""
        provided_id = uuid4()
        context = ModelInfraErrorContext.with_correlation(correlation_id=provided_id)
        assert context.correlation_id == provided_id

    def test_with_correlation_with_other_fields(self) -> None:
        """Test that with_correlation correctly passes through other kwargs."""
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP,
            operation="process_request",
            target_name="api-gateway",
        )
        assert context.correlation_id is not None
        assert context.transport_type == EnumInfraTransportType.HTTP
        assert context.operation == "process_request"
        assert context.target_name == "api-gateway"

    def test_with_correlation_uuid_is_valid(self) -> None:
        """Test that the generated UUID is a valid UUID4."""
        from uuid import UUID

        context = ModelInfraErrorContext.with_correlation()
        # Verify it's a valid UUID object
        assert isinstance(context.correlation_id, UUID)
        # Verify it's a valid UUID4 (version 4)
        assert context.correlation_id.version == 4


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

    def test_error_code_mapping_without_context(self) -> None:
        """Test that error uses SERVICE_UNAVAILABLE when no context provided."""
        error = InfraConnectionError("Connection error")
        # Without context, defaults to SERVICE_UNAVAILABLE
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_code_mapping_database_transport(self) -> None:
        """Test DATABASE transport uses DATABASE_CONNECTION_ERROR."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            target_name="postgresql",
        )
        error = InfraConnectionError("Database connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.DATABASE_CONNECTION_ERROR

    def test_error_code_mapping_http_transport(self) -> None:
        """Test HTTP transport uses NETWORK_ERROR."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            target_name="api-gateway",
        )
        error = InfraConnectionError("HTTP connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.NETWORK_ERROR

    def test_error_code_mapping_grpc_transport(self) -> None:
        """Test GRPC transport uses NETWORK_ERROR."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.GRPC,
            target_name="grpc-service",
        )
        error = InfraConnectionError("gRPC connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.NETWORK_ERROR

    def test_error_code_mapping_kafka_transport(self) -> None:
        """Test KAFKA transport uses SERVICE_UNAVAILABLE."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            target_name="kafka-broker",
        )
        error = InfraConnectionError("Kafka connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_code_mapping_consul_transport(self) -> None:
        """Test CONSUL transport uses SERVICE_UNAVAILABLE."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            target_name="consul-server",
        )
        error = InfraConnectionError("Consul connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_code_mapping_vault_transport(self) -> None:
        """Test VAULT transport uses SERVICE_UNAVAILABLE."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            target_name="vault-server",
        )
        error = InfraConnectionError("Vault connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_code_mapping_valkey_transport(self) -> None:
        """Test VALKEY transport uses SERVICE_UNAVAILABLE."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VALKEY,
            target_name="valkey-cluster",
        )
        error = InfraConnectionError("Valkey connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_code_mapping_context_without_transport(self) -> None:
        """Test context with no transport_type uses SERVICE_UNAVAILABLE."""
        context = ModelInfraErrorContext(
            operation="connect",
            target_name="unknown-service",
        )
        error = InfraConnectionError("Connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_chaining(self) -> None:
        """Test error chaining from connection exception."""
        context = ModelInfraErrorContext(target_name="valkey")
        conn_error = OSError("Connection refused")
        try:
            raise InfraConnectionError(
                "Failed to connect", context=context, host="localhost", port=6379
            ) from conn_error
        except InfraConnectionError as e:
            assert e.__cause__ == conn_error
            assert e.model.context["target_name"] == "valkey"
            assert e.model.context["port"] == 6379


class TestInfraConnectionErrorTransportMapping:
    """Comprehensive tests for InfraConnectionError transport-aware error code mapping.

    Validates that InfraConnectionError selects the correct EnumCoreErrorCode
    based on the transport_type in ModelInfraErrorContext.
    """

    def test_resolve_connection_error_code_with_none_context(self) -> None:
        """Test _resolve_connection_error_code with None context."""
        error_code = InfraConnectionError._resolve_connection_error_code(None)
        assert error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_resolve_connection_error_code_database(self) -> None:
        """Test _resolve_connection_error_code for DATABASE transport."""
        context = ModelInfraErrorContext(transport_type=EnumInfraTransportType.DATABASE)
        error_code = InfraConnectionError._resolve_connection_error_code(context)
        assert error_code == EnumCoreErrorCode.DATABASE_CONNECTION_ERROR

    def test_resolve_connection_error_code_network_transports(self) -> None:
        """Test _resolve_connection_error_code for network transports (HTTP, GRPC)."""
        for transport in [EnumInfraTransportType.HTTP, EnumInfraTransportType.GRPC]:
            context = ModelInfraErrorContext(transport_type=transport)
            error_code = InfraConnectionError._resolve_connection_error_code(context)
            assert error_code == EnumCoreErrorCode.NETWORK_ERROR, (
                f"Expected NETWORK_ERROR for {transport}, got {error_code}"
            )

    def test_resolve_connection_error_code_service_transports(self) -> None:
        """Test _resolve_connection_error_code for service transports."""
        service_transports = [
            EnumInfraTransportType.KAFKA,
            EnumInfraTransportType.CONSUL,
            EnumInfraTransportType.VAULT,
            EnumInfraTransportType.VALKEY,
        ]
        for transport in service_transports:
            context = ModelInfraErrorContext(transport_type=transport)
            error_code = InfraConnectionError._resolve_connection_error_code(context)
            assert error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE, (
                f"Expected SERVICE_UNAVAILABLE for {transport}, got {error_code}"
            )

    def test_all_transport_types_have_mapping(self) -> None:
        """Test that all EnumInfraTransportType values have error code mappings."""
        for transport in EnumInfraTransportType:
            context = ModelInfraErrorContext(transport_type=transport)
            # Should not raise and should return a valid error code
            error_code = InfraConnectionError._resolve_connection_error_code(context)
            assert isinstance(error_code, EnumCoreErrorCode), (
                f"Transport {transport} returned invalid error code type: {type(error_code)}"
            )

    def test_transport_error_code_map_completeness(self) -> None:
        """Test that the transport error code map includes all transport types."""
        for transport in EnumInfraTransportType:
            assert transport in InfraConnectionError._TRANSPORT_ERROR_CODE_MAP, (
                f"Transport {transport} missing from _TRANSPORT_ERROR_CODE_MAP"
            )
        # Also verify None is in the map
        assert None in InfraConnectionError._TRANSPORT_ERROR_CODE_MAP

    def test_error_code_preserved_in_model(self) -> None:
        """Test that resolved error code is correctly stored in the error model."""
        test_cases = [
            (
                EnumInfraTransportType.DATABASE,
                EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
            ),
            (EnumInfraTransportType.HTTP, EnumCoreErrorCode.NETWORK_ERROR),
            (EnumInfraTransportType.GRPC, EnumCoreErrorCode.NETWORK_ERROR),
            (EnumInfraTransportType.KAFKA, EnumCoreErrorCode.SERVICE_UNAVAILABLE),
            (EnumInfraTransportType.CONSUL, EnumCoreErrorCode.SERVICE_UNAVAILABLE),
            (EnumInfraTransportType.VAULT, EnumCoreErrorCode.SERVICE_UNAVAILABLE),
            (EnumInfraTransportType.VALKEY, EnumCoreErrorCode.SERVICE_UNAVAILABLE),
        ]
        for transport, expected_code in test_cases:
            context = ModelInfraErrorContext(transport_type=transport)
            error = InfraConnectionError("Test error", context=context)
            assert error.model.error_code == expected_code, (
                f"Transport {transport}: expected {expected_code}, got {error.model.error_code}"
            )


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


class TestInfraUnavailableError:
    """Tests for InfraUnavailableError."""

    def test_basic_instantiation(self) -> None:
        """Test basic error instantiation."""
        error = InfraUnavailableError("Resource unavailable")
        assert "Resource unavailable" in str(error)
        assert isinstance(error, RuntimeHostError)

    def test_with_context_model(self) -> None:
        """Test error with context model and details."""
        context = ModelInfraErrorContext(target_name="kafka")
        error = InfraUnavailableError(
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
        error = InfraUnavailableError("Resource error")
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_chaining(self) -> None:
        """Test error chaining from exception."""
        context = ModelInfraErrorContext(target_name="consul")
        resource_error = ConnectionRefusedError("Not responding")
        try:
            raise InfraUnavailableError(
                "Consul unavailable",
                context=context,
                host="consul.local",
                port=8500,
            ) from resource_error
        except InfraUnavailableError as e:
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
            InfraUnavailableError("test"),
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
            InfraUnavailableError("test", context=context),
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
            EnumInfraTransportType.VALKEY,
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
            InfraUnavailableError(
                "test",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.VALKEY
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
            InfraUnavailableError(
                "test", context=ModelInfraErrorContext(operation="check_health")
            ),
        ]

        for error, operation in zip(errors, operations, strict=True):
            assert error.model.context["operation"] == operation

    def test_all_errors_support_target_name(self) -> None:
        """Test that all errors support target_name via context model."""
        targets = ["api", "vault", "postgresql", "kafka", "consul", "valkey"]
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
            InfraUnavailableError(
                "test", context=ModelInfraErrorContext(target_name="valkey")
            ),
        ]

        for error, target in zip(errors, targets, strict=True):
            assert error.model.context["target_name"] == target


class TestErrorChaining:
    """Test error chaining across all infrastructure error classes.

    Validates that the `raise ... from e` pattern properly chains exceptions
    and preserves the original error as __cause__ for all error classes.
    """

    def test_runtime_host_error_chaining_preserves_cause(self) -> None:
        """Test RuntimeHostError properly chains and preserves original exception."""
        original = ValueError("Original value error")
        try:
            try:
                raise original
            except ValueError as e:
                raise RuntimeHostError("Wrapped error") from e
        except RuntimeHostError as wrapped:
            assert wrapped.__cause__ is original
            assert isinstance(wrapped.__cause__, ValueError)
            assert str(wrapped.__cause__) == "Original value error"

    def test_protocol_configuration_error_chaining_preserves_cause(self) -> None:
        """Test ProtocolConfigurationError properly chains and preserves original exception."""
        original = KeyError("missing_config_key")
        try:
            try:
                raise original
            except KeyError as e:
                raise ProtocolConfigurationError("Configuration error") from e
        except ProtocolConfigurationError as wrapped:
            assert wrapped.__cause__ is original
            assert isinstance(wrapped.__cause__, KeyError)
            assert "missing_config_key" in str(wrapped.__cause__)

    def test_secret_resolution_error_chaining_preserves_cause(self) -> None:
        """Test SecretResolutionError properly chains and preserves original exception."""
        original = ConnectionError("Vault connection failed")
        try:
            try:
                raise original
            except ConnectionError as e:
                raise SecretResolutionError("Cannot resolve secret") from e
        except SecretResolutionError as wrapped:
            assert wrapped.__cause__ is original
            assert isinstance(wrapped.__cause__, ConnectionError)
            assert "Vault connection failed" in str(wrapped.__cause__)

    def test_infra_connection_error_chaining_preserves_cause(self) -> None:
        """Test InfraConnectionError properly chains and preserves original exception."""
        original = OSError("Connection refused")
        try:
            try:
                raise original
            except OSError as e:
                raise InfraConnectionError("Database connection failed") from e
        except InfraConnectionError as wrapped:
            assert wrapped.__cause__ is original
            assert isinstance(wrapped.__cause__, OSError)
            assert "Connection refused" in str(wrapped.__cause__)

    def test_infra_timeout_error_chaining_preserves_cause(self) -> None:
        """Test InfraTimeoutError properly chains and preserves original exception."""
        original = TimeoutError("Operation timed out after 30s")
        try:
            try:
                raise original
            except TimeoutError as e:
                raise InfraTimeoutError("Query timeout") from e
        except InfraTimeoutError as wrapped:
            assert wrapped.__cause__ is original
            assert isinstance(wrapped.__cause__, TimeoutError)
            assert "30s" in str(wrapped.__cause__)

    def test_infra_authentication_error_chaining_preserves_cause(self) -> None:
        """Test InfraAuthenticationError properly chains and preserves original exception."""
        original = PermissionError("Access denied")
        try:
            try:
                raise original
            except PermissionError as e:
                raise InfraAuthenticationError("Authentication failed") from e
        except InfraAuthenticationError as wrapped:
            assert wrapped.__cause__ is original
            assert isinstance(wrapped.__cause__, PermissionError)
            assert "Access denied" in str(wrapped.__cause__)

    def test_infra_unavailable_error_chaining_preserves_cause(self) -> None:
        """Test InfraUnavailableError properly chains and preserves original exception."""
        original = ConnectionRefusedError("Service not responding")
        try:
            try:
                raise original
            except ConnectionRefusedError as e:
                raise InfraUnavailableError("Resource unavailable") from e
        except InfraUnavailableError as wrapped:
            assert wrapped.__cause__ is original
            assert isinstance(wrapped.__cause__, ConnectionRefusedError)
            assert "Service not responding" in str(wrapped.__cause__)

    def test_chained_error_with_context_preserved(self) -> None:
        """Test that context is preserved when chaining errors."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="execute_query",
            target_name="postgresql",
            correlation_id=correlation_id,
        )
        original = TimeoutError("Query exceeded deadline")
        try:
            try:
                raise original
            except TimeoutError as e:
                raise InfraTimeoutError(
                    "Database query timeout",
                    context=context,
                    timeout_seconds=30,
                ) from e
        except InfraTimeoutError as wrapped:
            # Verify chaining
            assert wrapped.__cause__ is original
            # Verify context preserved
            assert wrapped.model.correlation_id == correlation_id
            assert (
                wrapped.model.context["transport_type"]
                == EnumInfraTransportType.DATABASE
            )
            assert wrapped.model.context["operation"] == "execute_query"
            assert wrapped.model.context["target_name"] == "postgresql"
            assert wrapped.model.context["timeout_seconds"] == 30

    def test_multi_level_chaining(self) -> None:
        """Test error chaining through multiple levels."""
        root_error = OSError("Network unreachable")
        try:
            try:
                try:
                    raise root_error
                except OSError as e:
                    raise InfraConnectionError("Connection layer error") from e
            except InfraConnectionError as e:
                raise InfraUnavailableError("Service unavailable") from e
        except InfraUnavailableError as final:
            # Verify immediate cause
            assert isinstance(final.__cause__, InfraConnectionError)
            # Verify root cause through chain
            assert isinstance(final.__cause__.__cause__, OSError)
            assert final.__cause__.__cause__ is root_error

    def test_correlation_id_propagates_through_chain(self) -> None:
        """Test correlation_id preserved through multi-level error chaining."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(correlation_id=correlation_id)

        try:
            try:
                raise InfraConnectionError("Connection failed", context=context)
            except InfraConnectionError as e:
                # Correlation ID should propagate
                new_context = ModelInfraErrorContext(
                    correlation_id=e.model.correlation_id
                )
                raise InfraUnavailableError("Service down", context=new_context) from e
        except InfraUnavailableError as final:
            # Same correlation ID throughout the chain
            assert final.model.correlation_id == correlation_id
            assert final.__cause__ is not None
            assert isinstance(final.__cause__, InfraConnectionError)


class TestContextSerialization:
    """Test ModelInfraErrorContext serialization and deserialization.

    Validates that the context model correctly serializes to dict and JSON,
    handles UUID and enum fields properly, and supports roundtrip serialization.
    """

    def test_context_to_dict_empty(self) -> None:
        """Test serialization of empty context to dict."""
        context = ModelInfraErrorContext()
        data = context.model_dump()
        assert data == {
            "transport_type": None,
            "operation": None,
            "target_name": None,
            "correlation_id": None,
        }

    def test_context_to_dict_with_all_fields(self) -> None:
        """Test serialization of fully populated context to dict."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="produce_message",
            target_name="events-topic",
            correlation_id=correlation_id,
        )
        data = context.model_dump()
        assert data["transport_type"] == EnumInfraTransportType.KAFKA
        assert data["operation"] == "produce_message"
        assert data["target_name"] == "events-topic"
        assert data["correlation_id"] == correlation_id

    def test_context_to_dict_mode_json(self) -> None:
        """Test serialization with mode='json' for JSON-compatible output."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation="request",
            target_name="api-endpoint",
            correlation_id=correlation_id,
        )
        data = context.model_dump(mode="json")
        # Enum should be serialized as string value
        assert data["transport_type"] == "http"
        # UUID should be serialized as string
        assert data["correlation_id"] == str(correlation_id)
        assert data["operation"] == "request"
        assert data["target_name"] == "api-endpoint"

    def test_context_to_json_string(self) -> None:
        """Test serialization to JSON string."""
        import json

        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="connect",
            target_name="postgresql",
            correlation_id=correlation_id,
        )
        json_str = context.model_dump_json()
        # Verify valid JSON
        parsed = json.loads(json_str)
        # DATABASE enum value is "db"
        assert parsed["transport_type"] == "db"
        assert parsed["operation"] == "connect"
        assert parsed["target_name"] == "postgresql"
        assert parsed["correlation_id"] == str(correlation_id)

    def test_context_roundtrip_serialization(self) -> None:
        """Test roundtrip serialization: model -> dict -> model."""
        correlation_id = uuid4()
        original = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="get_secret",
            target_name="secrets/database",
            correlation_id=correlation_id,
        )
        # Serialize to dict
        data = original.model_dump()
        # Deserialize back to model
        restored = ModelInfraErrorContext(**data)
        # Verify equality
        assert restored.transport_type == original.transport_type
        assert restored.operation == original.operation
        assert restored.target_name == original.target_name
        assert restored.correlation_id == original.correlation_id

    def test_context_roundtrip_via_json(self) -> None:
        """Test roundtrip serialization via JSON string."""
        import json

        correlation_id = uuid4()
        original = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="register_service",
            target_name="my-service",
            correlation_id=correlation_id,
        )
        # Serialize to JSON string
        json_str = original.model_dump_json()
        # Parse JSON
        data = json.loads(json_str)
        # Deserialize back to model
        restored = ModelInfraErrorContext.model_validate(data)
        # Verify equality
        assert restored.transport_type == original.transport_type
        assert restored.operation == original.operation
        assert restored.target_name == original.target_name
        assert restored.correlation_id == original.correlation_id

    def test_context_uuid_field_serialization(self) -> None:
        """Test that UUID fields serialize and deserialize correctly."""
        from uuid import UUID

        correlation_id = uuid4()
        context = ModelInfraErrorContext(correlation_id=correlation_id)

        # Verify internal type is UUID
        assert isinstance(context.correlation_id, UUID)

        # Serialize with mode='json' converts to string
        json_data = context.model_dump(mode="json")
        assert isinstance(json_data["correlation_id"], str)
        assert json_data["correlation_id"] == str(correlation_id)

        # Standard dump preserves UUID type
        data = context.model_dump()
        assert isinstance(data["correlation_id"], UUID)
        assert data["correlation_id"] == correlation_id

    def test_context_enum_field_serialization(self) -> None:
        """Test that enum fields serialize and deserialize correctly."""
        context = ModelInfraErrorContext(transport_type=EnumInfraTransportType.VALKEY)

        # Verify internal type is enum
        assert isinstance(context.transport_type, EnumInfraTransportType)

        # Serialize with mode='json' converts to string value
        json_data = context.model_dump(mode="json")
        assert isinstance(json_data["transport_type"], str)
        assert json_data["transport_type"] == "valkey"

        # Standard dump preserves enum type
        data = context.model_dump()
        assert isinstance(data["transport_type"], EnumInfraTransportType)
        assert data["transport_type"] == EnumInfraTransportType.VALKEY

    def test_context_none_fields_in_serialization(self) -> None:
        """Test that None fields are properly handled in serialization."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation=None,
            target_name="endpoint",
            correlation_id=None,
        )
        data = context.model_dump()
        assert data["transport_type"] == EnumInfraTransportType.HTTP
        assert data["operation"] is None
        assert data["target_name"] == "endpoint"
        assert data["correlation_id"] is None

        # JSON serialization
        json_data = context.model_dump(mode="json")
        assert json_data["operation"] is None
        assert json_data["correlation_id"] is None

    def test_context_exclude_none_serialization(self) -> None:
        """Test serialization with exclude_none option."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="consume",
        )
        data = context.model_dump(exclude_none=True)
        assert "transport_type" in data
        assert "operation" in data
        assert "target_name" not in data
        assert "correlation_id" not in data

    def test_context_all_transport_types_serialize(self) -> None:
        """Test that all transport types serialize correctly."""
        transport_types = [
            EnumInfraTransportType.HTTP,
            EnumInfraTransportType.VAULT,
            EnumInfraTransportType.DATABASE,
            EnumInfraTransportType.KAFKA,
            EnumInfraTransportType.CONSUL,
            EnumInfraTransportType.VALKEY,
        ]
        for transport in transport_types:
            context = ModelInfraErrorContext(transport_type=transport)
            # Standard serialization
            data = context.model_dump()
            assert data["transport_type"] == transport
            # JSON-mode serialization
            json_data = context.model_dump(mode="json")
            assert json_data["transport_type"] == transport.value
            # Roundtrip
            restored = ModelInfraErrorContext.model_validate(data)
            assert restored.transport_type == transport
