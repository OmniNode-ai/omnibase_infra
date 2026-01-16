# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for InfraVaultError and InfraConsulError.

These tests validate the service-specific infrastructure error classes
that extend InfraConnectionError for Vault and Consul operations.

Tests cover:
    - Basic instantiation with message only
    - Inheritance from InfraConnectionError
    - Context model usage with appropriate transport types
    - Service-specific parameters (secret_path, consul_key, service_name)
    - Error chaining with `raise ... from e` pattern
    - Error code mapping for VAULT and CONSUL transports
    - Extra context passthrough via kwargs
"""

from uuid import uuid4

import pytest
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.errors import ModelOnexError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraConsulError,
    InfraVaultError,
    ModelInfraErrorContext,
)
from omnibase_infra.errors.error_infra import RuntimeHostError


class TestInfraVaultError:
    """Tests for InfraVaultError class."""

    def test_basic_instantiation(self) -> None:
        """Test InfraVaultError can be instantiated with message only."""
        error = InfraVaultError("Vault connection failed")
        assert "Vault connection failed" in str(error)

    def test_inherits_from_infra_connection_error(self) -> None:
        """Test InfraVaultError inherits from InfraConnectionError."""
        error = InfraVaultError("Test error")
        assert isinstance(error, InfraConnectionError)

    def test_inherits_from_runtime_host_error(self) -> None:
        """Test InfraVaultError inherits from RuntimeHostError."""
        error = InfraVaultError("Test error")
        assert isinstance(error, RuntimeHostError)

    def test_inherits_from_model_onex_error(self) -> None:
        """Test InfraVaultError inherits from ModelOnexError."""
        error = InfraVaultError("Test error")
        assert isinstance(error, ModelOnexError)
        assert isinstance(error, Exception)

    def test_with_context_model(self) -> None:
        """Test InfraVaultError with ModelInfraErrorContext."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="read_secret",
            target_name="vault-primary",
            correlation_id=correlation_id,
        )
        error = InfraVaultError(
            "Failed to read secret from Vault",
            context=context,
        )
        assert error.model.correlation_id == correlation_id
        assert error.model.context["transport_type"] == EnumInfraTransportType.VAULT
        assert error.model.context["operation"] == "read_secret"
        assert error.model.context["target_name"] == "vault-primary"

    def test_with_secret_path_parameter(self) -> None:
        """Test InfraVaultError with secret_path parameter.

        Note: Secret paths are sanitized to prevent exposure of infrastructure details.
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="read_secret",
            target_name="vault-primary",
        )
        error = InfraVaultError(
            "Failed to read secret from Vault",
            context=context,
            secret_path="secret/data/database/credentials",  # noqa: S106
        )
        # Path is sanitized to mask sensitive segments
        assert error.model.context["secret_path"] == "secret/***/***"

    def test_secret_path_included_in_extra_context(self) -> None:
        """Test that sanitized secret_path is added to extra_context."""
        error = InfraVaultError(
            "Secret not found",
            secret_path="secret/data/api-key",  # noqa: S106
        )
        # Path is sanitized to mask sensitive segments
        assert error.model.context["secret_path"] == "secret/***/***"

    def test_with_extra_context_kwargs(self) -> None:
        """Test InfraVaultError with additional extra_context kwargs."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="renew_token",
            target_name="vault-primary",
        )
        error = InfraVaultError(
            "Vault token renewal failed",
            context=context,
            retry_count=3,
            host="vault.example.com",
            port=8200,
        )
        assert error.model.context["retry_count"] == 3
        assert error.model.context["host"] == "vault.example.com"
        assert error.model.context["port"] == 8200

    def test_error_chaining(self) -> None:
        """Test error chaining with 'raise ... from e' pattern."""
        original_error = ConnectionError("Connection refused")
        try:
            raise InfraVaultError("Vault connection failed") from original_error
        except InfraVaultError as e:
            assert e.__cause__ == original_error
            assert isinstance(e.__cause__, ConnectionError)

    def test_error_chaining_preserves_context(self) -> None:
        """Test that context is preserved when chaining errors."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="initialize_client",
            target_name="vault-primary",
            correlation_id=correlation_id,
        )
        original_error = OSError("Network unreachable")
        try:
            raise InfraVaultError(
                "Failed to initialize Vault client",
                context=context,
                host="vault.example.com",
                port=8200,
            ) from original_error
        except InfraVaultError as e:
            assert e.__cause__ == original_error
            assert e.model.correlation_id == correlation_id
            assert e.model.context["transport_type"] == EnumInfraTransportType.VAULT
            assert e.model.context["host"] == "vault.example.com"

    def test_error_code_is_service_unavailable_for_vault_transport(self) -> None:
        """Test that VAULT transport uses SERVICE_UNAVAILABLE error code."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            target_name="vault-server",
        )
        error = InfraVaultError("Vault connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_code_without_context(self) -> None:
        """Test error code when no context is provided."""
        error = InfraVaultError("Vault error")
        # Without context, defaults to SERVICE_UNAVAILABLE
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_correlation_id_auto_generated(self) -> None:
        """Test that correlation_id is auto-generated when not provided."""
        from uuid import UUID

        error = InfraVaultError("Vault error")
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)

    def test_secret_path_with_other_extra_context(self) -> None:
        """Test that sanitized secret_path combines correctly with other extra_context."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="get_secret",
        )
        error = InfraVaultError(
            "Secret retrieval failed",
            context=context,
            secret_path="secret/data/db/password",  # noqa: S106
            vault_namespace="production",
            retry_count=2,
        )
        # Path is sanitized to mask sensitive segments
        assert error.model.context["secret_path"] == "secret/***/***"
        assert error.model.context["vault_namespace"] == "production"
        assert error.model.context["retry_count"] == 2

    def test_secret_path_none_not_added_to_context(self) -> None:
        """Test that None secret_path is not added to context."""
        error = InfraVaultError(
            "Vault error",
            secret_path=None,
        )
        assert "secret_path" not in error.model.context


class TestInfraConsulError:
    """Tests for InfraConsulError class."""

    def test_basic_instantiation(self) -> None:
        """Test InfraConsulError can be instantiated with message only."""
        error = InfraConsulError("Consul connection failed")
        assert "Consul connection failed" in str(error)

    def test_inherits_from_infra_connection_error(self) -> None:
        """Test InfraConsulError inherits from InfraConnectionError."""
        error = InfraConsulError("Test error")
        assert isinstance(error, InfraConnectionError)

    def test_inherits_from_runtime_host_error(self) -> None:
        """Test InfraConsulError inherits from RuntimeHostError."""
        error = InfraConsulError("Test error")
        assert isinstance(error, RuntimeHostError)

    def test_inherits_from_model_onex_error(self) -> None:
        """Test InfraConsulError inherits from ModelOnexError."""
        error = InfraConsulError("Test error")
        assert isinstance(error, ModelOnexError)
        assert isinstance(error, Exception)

    def test_with_context_model(self) -> None:
        """Test InfraConsulError with ModelInfraErrorContext."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="kv_get",
            target_name="consul-primary",
            correlation_id=correlation_id,
        )
        error = InfraConsulError(
            "Failed to read key from Consul KV store",
            context=context,
        )
        assert error.model.correlation_id == correlation_id
        assert error.model.context["transport_type"] == EnumInfraTransportType.CONSUL
        assert error.model.context["operation"] == "kv_get"
        assert error.model.context["target_name"] == "consul-primary"

    def test_with_consul_key_parameter(self) -> None:
        """Test InfraConsulError with consul_key parameter.

        Note: Consul keys are sanitized to prevent exposure of infrastructure details.
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="kv_get",
            target_name="consul-primary",
        )
        error = InfraConsulError(
            "Failed to read key from Consul KV store",
            context=context,
            consul_key="config/database/connection",
        )
        # Key is sanitized to mask sensitive segments
        assert error.model.context["consul_key"] == "config/***/***"

    def test_with_service_name_parameter(self) -> None:
        """Test InfraConsulError with service_name parameter."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="register_service",
            target_name="consul-primary",
        )
        error = InfraConsulError(
            "Failed to register service with Consul",
            context=context,
            service_name="api-gateway",
        )
        assert error.model.context["service_name"] == "api-gateway"

    def test_consul_key_included_in_extra_context(self) -> None:
        """Test that sanitized consul_key is added to extra_context."""
        error = InfraConsulError(
            "Key not found",
            consul_key="config/api/endpoint",
        )
        # Key is sanitized to mask sensitive segments
        assert error.model.context["consul_key"] == "config/***/***"

    def test_service_name_included_in_extra_context(self) -> None:
        """Test that service_name is added to extra_context."""
        error = InfraConsulError(
            "Service registration failed",
            service_name="my-service",
        )
        assert error.model.context["service_name"] == "my-service"

    def test_with_both_consul_key_and_service_name(self) -> None:
        """Test InfraConsulError with both consul_key and service_name."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="service_check",
            target_name="consul-primary",
        )
        error = InfraConsulError(
            "Service health check failed",
            context=context,
            consul_key="service/health/my-service",
            service_name="my-service",
        )
        # Key is sanitized to mask sensitive segments
        assert error.model.context["consul_key"] == "service/***/***"
        assert error.model.context["service_name"] == "my-service"

    def test_with_extra_context_kwargs(self) -> None:
        """Test InfraConsulError with additional extra_context kwargs."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="initialize_client",
            target_name="consul-primary",
        )
        error = InfraConsulError(
            "Failed to initialize Consul client",
            context=context,
            host="consul.example.com",
            port=8500,
            retry_count=3,
        )
        assert error.model.context["host"] == "consul.example.com"
        assert error.model.context["port"] == 8500
        assert error.model.context["retry_count"] == 3

    def test_error_chaining(self) -> None:
        """Test error chaining with 'raise ... from e' pattern."""
        original_error = ConnectionError("Connection refused")
        try:
            raise InfraConsulError("Consul connection failed") from original_error
        except InfraConsulError as e:
            assert e.__cause__ == original_error
            assert isinstance(e.__cause__, ConnectionError)

    def test_error_chaining_preserves_context(self) -> None:
        """Test that context is preserved when chaining errors."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="register_service",
            target_name="consul-primary",
            correlation_id=correlation_id,
        )
        original_error = OSError("Network unreachable")
        try:
            raise InfraConsulError(
                "Failed to register service",
                context=context,
                service_name="api-gateway",
                host="consul.example.com",
            ) from original_error
        except InfraConsulError as e:
            assert e.__cause__ == original_error
            assert e.model.correlation_id == correlation_id
            assert e.model.context["transport_type"] == EnumInfraTransportType.CONSUL
            assert e.model.context["service_name"] == "api-gateway"

    def test_error_code_is_service_unavailable_for_consul_transport(self) -> None:
        """Test that CONSUL transport uses SERVICE_UNAVAILABLE error code."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            target_name="consul-server",
        )
        error = InfraConsulError("Consul connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_error_code_without_context(self) -> None:
        """Test error code when no context is provided."""
        error = InfraConsulError("Consul error")
        # Without context, defaults to SERVICE_UNAVAILABLE
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE

    def test_correlation_id_auto_generated(self) -> None:
        """Test that correlation_id is auto-generated when not provided."""
        from uuid import UUID

        error = InfraConsulError("Consul error")
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)

    def test_consul_key_none_not_added_to_context(self) -> None:
        """Test that None consul_key is not added to context."""
        error = InfraConsulError(
            "Consul error",
            consul_key=None,
        )
        assert "consul_key" not in error.model.context

    def test_service_name_none_not_added_to_context(self) -> None:
        """Test that None service_name is not added to context."""
        error = InfraConsulError(
            "Consul error",
            service_name=None,
        )
        assert "service_name" not in error.model.context

    def test_all_parameters_combined(self) -> None:
        """Test InfraConsulError with all parameters combined."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="kv_put",
            target_name="consul-cluster",
            correlation_id=correlation_id,
        )
        error = InfraConsulError(
            "Failed to write key to Consul",
            context=context,
            consul_key="config/app/settings",
            service_name="config-writer",
            host="consul.example.com",
            port=8500,
            datacenter="us-east-1",
        )
        # Verify all context fields
        assert error.model.correlation_id == correlation_id
        assert error.model.context["transport_type"] == EnumInfraTransportType.CONSUL
        assert error.model.context["operation"] == "kv_put"
        assert error.model.context["target_name"] == "consul-cluster"
        # Key is sanitized to mask sensitive segments
        assert error.model.context["consul_key"] == "config/***/***"
        assert error.model.context["service_name"] == "config-writer"
        assert error.model.context["host"] == "consul.example.com"
        assert error.model.context["port"] == 8500
        assert error.model.context["datacenter"] == "us-east-1"


class TestServiceErrorsInheritanceChain:
    """Test inheritance chain for both service-specific error classes."""

    def test_vault_error_full_inheritance_chain(self) -> None:
        """Test InfraVaultError has correct full inheritance chain."""
        error = InfraVaultError("test")
        assert isinstance(error, InfraVaultError)
        assert isinstance(error, InfraConnectionError)
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)
        assert isinstance(error, Exception)

    def test_consul_error_full_inheritance_chain(self) -> None:
        """Test InfraConsulError has correct full inheritance chain."""
        error = InfraConsulError("test")
        assert isinstance(error, InfraConsulError)
        assert isinstance(error, InfraConnectionError)
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize(
        "error_class",
        [InfraVaultError, InfraConsulError],
    )
    def test_service_errors_are_infra_connection_errors(
        self, error_class: type[InfraConnectionError]
    ) -> None:
        """Test that service-specific errors are InfraConnectionError subclasses."""
        error = error_class("Test error message")
        assert isinstance(error, InfraConnectionError)

    @pytest.mark.parametrize(
        ("error_class", "transport_type"),
        [
            (InfraVaultError, EnumInfraTransportType.VAULT),
            (InfraConsulError, EnumInfraTransportType.CONSUL),
        ],
    )
    def test_service_errors_use_service_unavailable_error_code(
        self,
        error_class: type[InfraConnectionError],
        transport_type: EnumInfraTransportType,
    ) -> None:
        """Test that service-specific errors use SERVICE_UNAVAILABLE code."""
        context = ModelInfraErrorContext(
            transport_type=transport_type,
            target_name="test-server",
        )
        error = error_class("Connection failed", context=context)
        assert error.model.error_code == EnumCoreErrorCode.SERVICE_UNAVAILABLE


class TestServiceErrorsMultiLevelChaining:
    """Test multi-level error chaining for service-specific errors."""

    def test_vault_error_multi_level_chaining(self) -> None:
        """Test error chaining through multiple levels with InfraVaultError."""
        root_error = OSError("Network unreachable")
        try:
            try:
                try:
                    raise root_error
                except OSError as e:
                    raise ConnectionError("Connection layer error") from e
            except ConnectionError as e:
                raise InfraVaultError("Vault unavailable") from e
        except InfraVaultError as final:
            # Verify immediate cause
            assert isinstance(final.__cause__, ConnectionError)
            # Verify root cause through chain
            assert isinstance(final.__cause__.__cause__, OSError)
            assert final.__cause__.__cause__ is root_error

    def test_consul_error_multi_level_chaining(self) -> None:
        """Test error chaining through multiple levels with InfraConsulError."""
        root_error = OSError("Network unreachable")
        try:
            try:
                try:
                    raise root_error
                except OSError as e:
                    raise ConnectionError("Connection layer error") from e
            except ConnectionError as e:
                raise InfraConsulError("Consul unavailable") from e
        except InfraConsulError as final:
            # Verify immediate cause
            assert isinstance(final.__cause__, ConnectionError)
            # Verify root cause through chain
            assert isinstance(final.__cause__.__cause__, OSError)
            assert final.__cause__.__cause__ is root_error

    def test_correlation_id_propagates_through_service_error_chain(self) -> None:
        """Test correlation_id preserved through service error chaining."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(correlation_id=correlation_id)

        try:
            try:
                raise InfraVaultError("Vault connection failed", context=context)
            except InfraVaultError as e:
                # Propagate correlation ID to new error
                new_context = ModelInfraErrorContext(
                    correlation_id=e.model.correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                )
                raise InfraConsulError(
                    "Consul fallback failed", context=new_context
                ) from e
        except InfraConsulError as final:
            # Same correlation ID throughout the chain
            assert final.model.correlation_id == correlation_id
            assert final.__cause__ is not None
            assert isinstance(final.__cause__, InfraVaultError)


class TestServiceErrorsRealWorldScenarios:
    """Test real-world usage scenarios for service-specific errors."""

    def test_vault_secret_read_failure(self) -> None:
        """Test realistic Vault secret read failure scenario."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="read_secret",
            target_name="vault-primary",
            correlation_id=correlation_id,
        )
        error = InfraVaultError(
            "Failed to read secret from Vault after 3 retries",
            context=context,
            secret_path="secret/data/database/credentials",  # noqa: S106
            retry_count=3,
            vault_address="https://vault.example.com:8200",
        )
        # Path is sanitized to mask sensitive segments
        assert error.model.context["secret_path"] == "secret/***/***"
        assert error.model.context["retry_count"] == 3
        assert error.model.context["vault_address"] == "https://vault.example.com:8200"

    def test_vault_token_renewal_failure(self) -> None:
        """Test realistic Vault token renewal failure scenario."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="renew_token",
            target_name="vault-primary",
        )
        error = InfraVaultError(
            "Vault token renewal failed",
            context=context,
            token_ttl_remaining=0,
            renewal_attempts=5,
        )
        assert error.model.context["token_ttl_remaining"] == 0
        assert error.model.context["renewal_attempts"] == 5

    def test_consul_service_registration_failure(self) -> None:
        """Test realistic Consul service registration failure scenario."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="register_service",
            target_name="consul-cluster",
            correlation_id=correlation_id,
        )
        error = InfraConsulError(
            "Failed to register service with Consul",
            context=context,
            service_name="api-gateway",
            service_id="api-gateway-1",
            tags=["production", "v2"],
        )
        assert error.model.context["service_name"] == "api-gateway"
        assert error.model.context["service_id"] == "api-gateway-1"
        assert error.model.context["tags"] == ["production", "v2"]

    def test_consul_kv_operation_failure(self) -> None:
        """Test realistic Consul KV operation failure scenario."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="kv_get",
            target_name="consul-primary",
        )
        error = InfraConsulError(
            "Failed to read configuration from Consul KV store",
            context=context,
            consul_key="config/app/database/connection_pool_size",
            datacenter="us-west-2",
            consistency_mode="consistent",
        )
        # Key is sanitized to mask sensitive segments
        assert error.model.context["consul_key"] == "config/***/***"
        assert error.model.context["datacenter"] == "us-west-2"
        assert error.model.context["consistency_mode"] == "consistent"

    def test_consul_health_check_failure(self) -> None:
        """Test realistic Consul health check failure scenario."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="health_check",
            target_name="consul-cluster",
        )
        error = InfraConsulError(
            "Service health check registration failed",
            context=context,
            service_name="database-proxy",
            check_id="database-proxy-tcp-check",
            check_interval="10s",
            check_timeout="5s",
        )
        assert error.model.context["service_name"] == "database-proxy"
        assert error.model.context["check_id"] == "database-proxy-tcp-check"
        assert error.model.context["check_interval"] == "10s"
