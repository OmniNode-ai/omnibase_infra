# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for container wiring error classes."""

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ContainerValidationError,
    ContainerWiringError,
    ModelInfraErrorContext,
    ServiceRegistrationError,
    ServiceResolutionError,
)


class TestContainerWiringError:
    """Test ContainerWiringError base class."""

    def test_basic_initialization(self) -> None:
        """Test basic error initialization."""
        error = ContainerWiringError("Container wiring failed")

        assert "Container wiring failed" in str(error)
        assert error.model.error_code.value == "ONEX_CORE_081_OPERATION_FAILED"

    def test_with_context(self) -> None:
        """Test error with infrastructure context."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="wire_services",
            correlation_id=correlation_id,
        )

        error = ContainerWiringError("Container wiring failed", context=context)

        assert error.model.correlation_id == correlation_id
        assert error.model.context["transport_type"] == EnumInfraTransportType.RUNTIME
        assert error.model.context["operation"] == "wire_services"


class TestServiceRegistrationError:
    """Test ServiceRegistrationError class."""

    def test_basic_initialization(self) -> None:
        """Test basic error initialization."""
        error = ServiceRegistrationError(
            "Failed to register service",
            service_name="PolicyRegistry",
        )

        assert "Failed to register service" in str(error)
        assert error.model.context["service_name"] == "PolicyRegistry"

    def test_with_full_context(self) -> None:
        """Test error with full context."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="register_policy_registry",
            target_name="PolicyRegistry",
            correlation_id=correlation_id,
        )

        error = ServiceRegistrationError(
            "Failed to register PolicyRegistry",
            service_name="PolicyRegistry",
            context=context,
            original_error="Container missing attribute",
            hint="Ensure container.service_registry exists",
        )

        assert error.model.correlation_id == correlation_id
        assert error.model.context["service_name"] == "PolicyRegistry"
        assert error.model.context["target_name"] == "PolicyRegistry"
        assert error.model.context["original_error"] == "Container missing attribute"
        assert error.model.context["hint"] == "Ensure container.service_registry exists"

    def test_error_chaining(self) -> None:
        """Test error chaining with original exception."""
        original = AttributeError(
            "'MockContainer' object has no attribute 'service_registry'"
        )

        try:
            raise ServiceRegistrationError(
                "Failed to register service",
                service_name="PolicyRegistry",
            ) from original
        except ServiceRegistrationError as e:
            assert e.__cause__ is original
            assert isinstance(e.__cause__, AttributeError)


class TestServiceResolutionError:
    """Test ServiceResolutionError class."""

    def test_basic_initialization(self) -> None:
        """Test basic error initialization."""
        error = ServiceResolutionError(
            "Service not found",
            service_name="PolicyRegistry",
        )

        assert "Service not found" in str(error)
        assert error.model.context["service_name"] == "PolicyRegistry"

    def test_with_full_context(self) -> None:
        """Test error with full context."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="resolve_policy_registry",
            target_name="PolicyRegistry",
            correlation_id=correlation_id,
        )

        error = ServiceResolutionError(
            "PolicyRegistry not registered in container",
            service_name="PolicyRegistry",
            context=context,
            original_error="Service not found in registry",
            error_type="ValueError",
        )

        assert error.model.correlation_id == correlation_id
        assert error.model.context["service_name"] == "PolicyRegistry"
        assert error.model.context["target_name"] == "PolicyRegistry"
        assert error.model.context["original_error"] == "Service not found in registry"
        assert error.model.context["error_type"] == "ValueError"

    def test_not_registered_scenario(self) -> None:
        """Test common scenario where service is not registered."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="resolve_service",
        )

        error = ServiceResolutionError(
            "PolicyRegistry not registered. Call wire_infrastructure_services() first.",
            service_name="PolicyRegistry",
            context=context,
        )

        assert "not registered" in str(error).lower()
        assert "wire_infrastructure_services" in str(error)


class TestContainerValidationError:
    """Test ContainerValidationError class."""

    def test_missing_attribute(self) -> None:
        """Test error for missing container attribute."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="validate_container",
        )

        error = ContainerValidationError(
            "Container missing service_registry attribute",
            context=context,
            missing_attribute="service_registry",
            container_type="MockContainer",
        )

        assert "missing service_registry" in str(error).lower()
        assert error.model.context["missing_attribute"] == "service_registry"
        assert error.model.context["container_type"] == "MockContainer"

    def test_missing_method(self) -> None:
        """Test error for missing container method."""
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="validate_container",
        )

        error = ContainerValidationError(
            "Container service_registry missing register_instance() method",
            context=context,
            missing_method="register_instance",
            registry_type="MockRegistry",
        )

        assert "missing register_instance" in str(error).lower()
        assert error.model.context["missing_method"] == "register_instance"
        assert error.model.context["registry_type"] == "MockRegistry"

    def test_with_correlation_id(self) -> None:
        """Test validation error with correlation ID."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="validate_container",
            correlation_id=correlation_id,
        )

        error = ContainerValidationError(
            "Container validation failed",
            context=context,
        )

        assert error.model.correlation_id == correlation_id


class TestErrorHierarchy:
    """Test error class hierarchy and inheritance."""

    def test_inheritance_chain(self) -> None:
        """Test that error classes inherit correctly."""
        from omnibase_core.models.errors.model_onex_error import ModelOnexError

        from omnibase_infra.errors.infra_errors import RuntimeHostError

        # ContainerWiringError extends RuntimeHostError
        error = ContainerWiringError("test")
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)

        # ServiceRegistrationError extends ContainerWiringError
        error = ServiceRegistrationError("test")
        assert isinstance(error, ContainerWiringError)
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)

        # ServiceResolutionError extends ContainerWiringError
        error = ServiceResolutionError("test")
        assert isinstance(error, ContainerWiringError)
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)

        # ContainerValidationError extends ContainerWiringError
        error = ContainerValidationError("test")
        assert isinstance(error, ContainerWiringError)
        assert isinstance(error, RuntimeHostError)
        assert isinstance(error, ModelOnexError)

    def test_error_code_consistency(self) -> None:
        """Test that all container wiring errors use consistent error codes."""
        from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode

        errors = [
            ContainerWiringError("test"),
            ServiceRegistrationError("test"),
            ServiceResolutionError("test"),
            ContainerValidationError("test"),
        ]

        for error in errors:
            # All should use OPERATION_FAILED as base error code
            assert error.model.error_code == EnumCoreErrorCode.OPERATION_FAILED


class TestErrorUsageScenarios:
    """Test realistic error usage scenarios."""

    def test_registration_failure_workflow(self) -> None:
        """Test typical registration failure workflow."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="register_policy_registry",
            target_name="PolicyRegistry",
            correlation_id=correlation_id,
        )

        original_error = TypeError(
            "register_instance() missing required argument: 'scope'"
        )

        try:
            raise ServiceRegistrationError(
                "Failed to register PolicyRegistry: invalid registration arguments",
                service_name="PolicyRegistry",
                context=context,
                original_error=str(original_error),
                hint="Check register_instance() signature compatibility",
            ) from original_error
        except ServiceRegistrationError as e:
            # Verify error structure
            assert e.model.correlation_id == correlation_id
            assert e.model.context["service_name"] == "PolicyRegistry"
            assert "invalid registration arguments" in str(e)
            assert isinstance(e.__cause__, TypeError)

    def test_resolution_failure_workflow(self) -> None:
        """Test typical resolution failure workflow."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="resolve_policy_registry",
            target_name="PolicyRegistry",
            correlation_id=correlation_id,
        )

        original_error = ValueError("Service not found: PolicyRegistry")

        try:
            raise ServiceResolutionError(
                "PolicyRegistry not registered in container. Call wire_infrastructure_services() first.",
                service_name="PolicyRegistry",
                context=context,
                original_error=str(original_error),
            ) from original_error
        except ServiceResolutionError as e:
            # Verify error structure
            assert e.model.correlation_id == correlation_id
            assert e.model.context["service_name"] == "PolicyRegistry"
            assert "not registered" in str(e)
            assert isinstance(e.__cause__, ValueError)

    def test_validation_failure_workflow(self) -> None:
        """Test typical validation failure workflow."""
        correlation_id = uuid4()
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="validate_container",
            correlation_id=correlation_id,
        )

        error = ContainerValidationError(
            "Container missing required service_registry attribute",
            context=context,
            missing_attribute="service_registry",
            container_type="MockContainer",
        )

        # Verify error structure
        assert error.model.correlation_id == correlation_id
        assert error.model.context["missing_attribute"] == "service_registry"
        assert error.model.context["container_type"] == "MockContainer"
        assert "service_registry" in str(error)
