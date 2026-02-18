# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for InfraConsulError.

These tests validate the service-specific infrastructure error class
that extends InfraConnectionError for Consul operations.

Tests cover:
    - Basic instantiation with message only
    - Inheritance from InfraConnectionError
    - Context model usage with appropriate transport types
    - Service-specific parameters (consul_key, service_name)
    - Error chaining with `raise ... from e` pattern
    - Error code mapping for CONSUL transport
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
    ModelInfraErrorContext,
)
from omnibase_infra.errors.error_infra import RuntimeHostError


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
        sanitized_key = error.model.context["consul_key"]
        assert sanitized_key == "config/***/***"
        # Negative assertions: prove sensitive data is actually removed
        assert "database" not in sanitized_key
        assert "connection" not in sanitized_key

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
        sanitized_key = error.model.context["consul_key"]
        assert sanitized_key == "config/***/***"
        # Negative assertions: prove sensitive data is actually removed
        assert "api" not in sanitized_key
        assert "endpoint" not in sanitized_key

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
        sanitized_key = error.model.context["consul_key"]
        assert sanitized_key == "service/***/***"
        # Negative assertions: prove sensitive data is actually removed
        assert "health" not in sanitized_key
        assert "my-service" not in sanitized_key
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
        sanitized_key = error.model.context["consul_key"]
        assert sanitized_key == "config/***/***"
        # Negative assertions: prove sensitive data is actually removed
        assert "app" not in sanitized_key
        assert "settings" not in sanitized_key
        assert error.model.context["service_name"] == "config-writer"
        assert error.model.context["host"] == "consul.example.com"
        assert error.model.context["port"] == 8500
        assert error.model.context["datacenter"] == "us-east-1"


class TestServiceErrorsInheritanceChain:
    """Test inheritance chain for service-specific error classes."""

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
        [InfraConsulError],
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


class TestServiceErrorsRealWorldScenarios:
    """Test real-world usage scenarios for service-specific errors."""

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
        sanitized_key = error.model.context["consul_key"]
        assert sanitized_key == "config/***/***"
        # Negative assertions: prove sensitive data is actually removed
        assert "app" not in sanitized_key
        assert "database" not in sanitized_key
        assert "connection_pool_size" not in sanitized_key
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
