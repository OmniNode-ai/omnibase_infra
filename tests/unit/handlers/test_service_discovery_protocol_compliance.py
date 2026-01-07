# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Service Discovery Handler Protocol Compliance.

This module validates that service discovery handler implementations
correctly implement the ProtocolServiceDiscoveryHandler protocol.

Protocol Compliance Testing
---------------------------
Per ONEX patterns, protocol compliance is verified using duck typing
(hasattr() and callable() checks) rather than isinstance() to support
structural subtyping. This approach allows handlers to implement the
protocol without explicit inheritance.

ProtocolServiceDiscoveryHandler Interface
-----------------------------------------
Required Members:
    - handler_type (property): Returns handler type identifier string
    - register_service(service_info, correlation_id): Async method for registration
    - deregister_service(service_id, correlation_id): Async method for deregistration
    - discover_services(service_name, tags, correlation_id): Async method for discovery
    - health_check(correlation_id): Async method for health verification

Handler Implementations Tested:
    - MockServiceDiscoveryHandler: In-memory mock for testing
    - ConsulServiceDiscoveryHandler: Consul backend implementation

Related:
    - OMN-1131: Capability-oriented node architecture
    - ProtocolServiceDiscoveryHandler: Protocol definition
    - PR #119: Test coverage for protocol compliance
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

from omnibase_infra.handlers.service_discovery.handler_consul_service_discovery import (
    ConsulServiceDiscoveryHandler,
)
from omnibase_infra.handlers.service_discovery.handler_mock_service_discovery import (
    MockServiceDiscoveryHandler,
)
from omnibase_infra.handlers.service_discovery.protocol_service_discovery_handler import (
    ProtocolServiceDiscoveryHandler,
)

if TYPE_CHECKING:
    from omnibase_infra.handlers.service_discovery.models import ModelServiceInfo


# =============================================================================
# Protocol Method Definitions
# =============================================================================

REQUIRED_PROTOCOL_METHODS: tuple[str, ...] = (
    "register_service",
    "deregister_service",
    "discover_services",
    "health_check",
)
"""Required async methods that all handlers must implement."""

REQUIRED_PROTOCOL_PROPERTIES: tuple[str, ...] = ("handler_type",)
"""Required properties that all handlers must implement."""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler() -> MockServiceDiscoveryHandler:
    """Create MockServiceDiscoveryHandler instance for testing."""
    return MockServiceDiscoveryHandler()


@pytest.fixture
def consul_handler() -> ConsulServiceDiscoveryHandler:
    """Create ConsulServiceDiscoveryHandler instance for testing.

    Note: This creates the handler without initializing the connection.
    Protocol compliance tests only verify interface structure, not runtime behavior.
    """
    return ConsulServiceDiscoveryHandler(
        consul_host="localhost",
        consul_port=8500,
        consul_scheme="http",
    )


# =============================================================================
# Protocol Interface Verification Tests
# =============================================================================


class TestProtocolServiceDiscoveryHandlerInterface:
    """Verify ProtocolServiceDiscoveryHandler is a valid runtime-checkable protocol.

    These tests ensure the protocol definition itself is correct and can be
    used for runtime type checking with isinstance().
    """

    def test_protocol_is_runtime_checkable(self) -> None:
        """ProtocolServiceDiscoveryHandler is decorated with @runtime_checkable."""
        # Protocol should be decorated with @runtime_checkable
        assert hasattr(
            ProtocolServiceDiscoveryHandler, "__protocol_attrs__"
        ) or hasattr(ProtocolServiceDiscoveryHandler, "_is_runtime_protocol"), (
            "ProtocolServiceDiscoveryHandler should be @runtime_checkable"
        )

    def test_protocol_defines_handler_type_property(self) -> None:
        """Protocol defines handler_type property."""
        # Check that handler_type is in the protocol's annotations or attrs
        assert "handler_type" in dir(ProtocolServiceDiscoveryHandler), (
            "Protocol must define handler_type property"
        )

    def test_protocol_defines_required_methods(self) -> None:
        """Protocol defines all required async methods."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            assert hasattr(ProtocolServiceDiscoveryHandler, method_name), (
                f"Protocol must define {method_name} method"
            )


# =============================================================================
# MockServiceDiscoveryHandler Protocol Compliance Tests
# =============================================================================


class TestMockServiceDiscoveryHandlerProtocolCompliance:
    """Validate MockServiceDiscoveryHandler implements ProtocolServiceDiscoveryHandler.

    Uses duck typing verification per ONEX patterns to ensure the mock handler
    correctly implements all protocol requirements.
    """

    def test_mock_handler_isinstance_protocol(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """MockServiceDiscoveryHandler passes isinstance check for protocol."""
        assert isinstance(mock_handler, ProtocolServiceDiscoveryHandler), (
            "MockServiceDiscoveryHandler must be an instance of "
            "ProtocolServiceDiscoveryHandler protocol"
        )

    def test_mock_handler_has_handler_type_property(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """MockServiceDiscoveryHandler has handler_type property."""
        assert hasattr(mock_handler, "handler_type"), (
            "MockServiceDiscoveryHandler must have handler_type property"
        )

        # Verify handler_type returns expected value
        handler_type = mock_handler.handler_type
        assert handler_type == "mock", (
            f"MockServiceDiscoveryHandler.handler_type should return 'mock', "
            f"got '{handler_type}'"
        )

    def test_mock_handler_has_all_required_methods(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """MockServiceDiscoveryHandler has all required protocol methods."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            assert hasattr(mock_handler, method_name), (
                f"MockServiceDiscoveryHandler must have {method_name} method"
            )
            assert callable(getattr(mock_handler, method_name)), (
                f"MockServiceDiscoveryHandler.{method_name} must be callable"
            )

    def test_mock_handler_methods_are_async(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """All required methods on MockServiceDiscoveryHandler are async coroutines."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            method = getattr(mock_handler, method_name)
            assert asyncio.iscoroutinefunction(method), (
                f"MockServiceDiscoveryHandler.{method_name} must be an async coroutine"
            )

    def test_mock_handler_register_service_signature(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """register_service method has correct parameter signature."""
        sig = inspect.signature(mock_handler.register_service)
        params = list(sig.parameters.keys())

        assert "service_info" in params, (
            "register_service must accept 'service_info' parameter"
        )
        assert "correlation_id" in params, (
            "register_service must accept 'correlation_id' parameter"
        )

    def test_mock_handler_deregister_service_signature(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """deregister_service method has correct parameter signature."""
        sig = inspect.signature(mock_handler.deregister_service)
        params = list(sig.parameters.keys())

        assert "service_id" in params, (
            "deregister_service must accept 'service_id' parameter"
        )
        assert "correlation_id" in params, (
            "deregister_service must accept 'correlation_id' parameter"
        )

    def test_mock_handler_discover_services_signature(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """discover_services method has correct parameter signature."""
        sig = inspect.signature(mock_handler.discover_services)
        params = list(sig.parameters.keys())

        assert "service_name" in params, (
            "discover_services must accept 'service_name' parameter"
        )
        assert "tags" in params, "discover_services must accept 'tags' parameter"
        assert "correlation_id" in params, (
            "discover_services must accept 'correlation_id' parameter"
        )

    def test_mock_handler_health_check_signature(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """health_check method has correct parameter signature."""
        sig = inspect.signature(mock_handler.health_check)
        params = list(sig.parameters.keys())

        assert "correlation_id" in params, (
            "health_check must accept 'correlation_id' parameter"
        )

    def test_mock_handler_register_service_return_type_annotation(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """register_service method has return type annotation."""
        sig = inspect.signature(mock_handler.register_service)
        assert sig.return_annotation != inspect.Signature.empty, (
            "register_service must have return type annotation"
        )

    def test_mock_handler_discover_services_return_type_annotation(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """discover_services method has return type annotation."""
        sig = inspect.signature(mock_handler.discover_services)
        assert sig.return_annotation != inspect.Signature.empty, (
            "discover_services must have return type annotation"
        )

    def test_mock_handler_health_check_return_type_annotation(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """health_check method has return type annotation."""
        sig = inspect.signature(mock_handler.health_check)
        assert sig.return_annotation != inspect.Signature.empty, (
            "health_check must have return type annotation"
        )


# =============================================================================
# ConsulServiceDiscoveryHandler Protocol Compliance Tests
# =============================================================================


class TestConsulServiceDiscoveryHandlerProtocolCompliance:
    """Validate ConsulServiceDiscoveryHandler implements ProtocolServiceDiscoveryHandler.

    Uses duck typing verification per ONEX patterns to ensure the Consul handler
    correctly implements all protocol requirements.

    Note: These tests verify interface compliance only, not runtime behavior.
    Integration tests with actual Consul are in test_consul_handler_integration.py.
    """

    def test_consul_handler_isinstance_protocol(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """ConsulServiceDiscoveryHandler passes isinstance check for protocol."""
        assert isinstance(consul_handler, ProtocolServiceDiscoveryHandler), (
            "ConsulServiceDiscoveryHandler must be an instance of "
            "ProtocolServiceDiscoveryHandler protocol"
        )

    def test_consul_handler_has_handler_type_property(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """ConsulServiceDiscoveryHandler has handler_type property."""
        assert hasattr(consul_handler, "handler_type"), (
            "ConsulServiceDiscoveryHandler must have handler_type property"
        )

        # Verify handler_type returns expected value
        handler_type = consul_handler.handler_type
        assert handler_type == "consul", (
            f"ConsulServiceDiscoveryHandler.handler_type should return 'consul', "
            f"got '{handler_type}'"
        )

    def test_consul_handler_has_all_required_methods(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """ConsulServiceDiscoveryHandler has all required protocol methods."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            assert hasattr(consul_handler, method_name), (
                f"ConsulServiceDiscoveryHandler must have {method_name} method"
            )
            assert callable(getattr(consul_handler, method_name)), (
                f"ConsulServiceDiscoveryHandler.{method_name} must be callable"
            )

    def test_consul_handler_methods_are_async(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """All required methods on ConsulServiceDiscoveryHandler are async coroutines."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            method = getattr(consul_handler, method_name)
            assert asyncio.iscoroutinefunction(method), (
                f"ConsulServiceDiscoveryHandler.{method_name} must be an async coroutine"
            )

    def test_consul_handler_register_service_signature(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """register_service method has correct parameter signature."""
        sig = inspect.signature(consul_handler.register_service)
        params = list(sig.parameters.keys())

        assert "service_info" in params, (
            "register_service must accept 'service_info' parameter"
        )
        assert "correlation_id" in params, (
            "register_service must accept 'correlation_id' parameter"
        )

    def test_consul_handler_deregister_service_signature(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """deregister_service method has correct parameter signature."""
        sig = inspect.signature(consul_handler.deregister_service)
        params = list(sig.parameters.keys())

        assert "service_id" in params, (
            "deregister_service must accept 'service_id' parameter"
        )
        assert "correlation_id" in params, (
            "deregister_service must accept 'correlation_id' parameter"
        )

    def test_consul_handler_discover_services_signature(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """discover_services method has correct parameter signature."""
        sig = inspect.signature(consul_handler.discover_services)
        params = list(sig.parameters.keys())

        assert "service_name" in params, (
            "discover_services must accept 'service_name' parameter"
        )
        assert "tags" in params, "discover_services must accept 'tags' parameter"
        assert "correlation_id" in params, (
            "discover_services must accept 'correlation_id' parameter"
        )

    def test_consul_handler_health_check_signature(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """health_check method has correct parameter signature."""
        sig = inspect.signature(consul_handler.health_check)
        params = list(sig.parameters.keys())

        assert "correlation_id" in params, (
            "health_check must accept 'correlation_id' parameter"
        )

    def test_consul_handler_has_circuit_breaker_mixin(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """ConsulServiceDiscoveryHandler inherits MixinAsyncCircuitBreaker."""
        from omnibase_infra.mixins import MixinAsyncCircuitBreaker

        assert isinstance(consul_handler, MixinAsyncCircuitBreaker), (
            "ConsulServiceDiscoveryHandler should inherit MixinAsyncCircuitBreaker "
            "for circuit breaker resilience"
        )

    def test_consul_handler_has_circuit_breaker_attributes(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """ConsulServiceDiscoveryHandler has circuit breaker attributes from mixin."""
        # Circuit breaker mixin attributes
        assert hasattr(consul_handler, "_circuit_breaker_lock"), (
            "ConsulServiceDiscoveryHandler must have _circuit_breaker_lock"
        )
        assert hasattr(consul_handler, "_circuit_breaker_open"), (
            "ConsulServiceDiscoveryHandler must have _circuit_breaker_open"
        )

    def test_consul_handler_has_shutdown_method(
        self, consul_handler: ConsulServiceDiscoveryHandler
    ) -> None:
        """ConsulServiceDiscoveryHandler has shutdown method for cleanup."""
        assert hasattr(consul_handler, "shutdown"), (
            "ConsulServiceDiscoveryHandler must have shutdown method for resource cleanup"
        )
        assert callable(consul_handler.shutdown), (
            "ConsulServiceDiscoveryHandler.shutdown must be callable"
        )
        assert asyncio.iscoroutinefunction(consul_handler.shutdown), (
            "ConsulServiceDiscoveryHandler.shutdown must be an async coroutine"
        )


# =============================================================================
# Cross-Handler Protocol Compliance Verification
# =============================================================================


class TestCrossHandlerProtocolCompliance:
    """Cross-validate protocol compliance across all handler implementations.

    These tests ensure uniform protocol implementation across all handler types,
    enabling safe handler swapping at runtime.
    """

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs", "expected_handler_type"),
        [
            (MockServiceDiscoveryHandler, {}, "mock"),
            (
                ConsulServiceDiscoveryHandler,
                {
                    "consul_host": "localhost",
                    "consul_port": 8500,
                    "consul_scheme": "http",
                },
                "consul",
            ),
        ],
    )
    def test_handler_is_protocol_instance(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
        expected_handler_type: str,
    ) -> None:
        """All handlers pass isinstance check for ProtocolServiceDiscoveryHandler."""
        handler = handler_class(**init_kwargs)
        assert isinstance(handler, ProtocolServiceDiscoveryHandler), (
            f"{handler_class.__name__} must be an instance of "
            "ProtocolServiceDiscoveryHandler protocol"
        )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs", "expected_handler_type"),
        [
            (MockServiceDiscoveryHandler, {}, "mock"),
            (
                ConsulServiceDiscoveryHandler,
                {
                    "consul_host": "localhost",
                    "consul_port": 8500,
                    "consul_scheme": "http",
                },
                "consul",
            ),
        ],
    )
    def test_handler_type_returns_correct_value(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
        expected_handler_type: str,
    ) -> None:
        """handler_type property returns correct identifier for each handler."""
        handler = handler_class(**init_kwargs)
        assert handler.handler_type == expected_handler_type, (
            f"{handler_class.__name__}.handler_type should return '{expected_handler_type}', "
            f"got '{handler.handler_type}'"
        )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (MockServiceDiscoveryHandler, {}),
            (
                ConsulServiceDiscoveryHandler,
                {
                    "consul_host": "localhost",
                    "consul_port": 8500,
                    "consul_scheme": "http",
                },
            ),
        ],
    )
    def test_all_handlers_have_same_method_names(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
    ) -> None:
        """All handlers have the same required method names for interoperability."""
        handler = handler_class(**init_kwargs)

        for method_name in REQUIRED_PROTOCOL_METHODS:
            assert hasattr(handler, method_name), (
                f"{handler_class.__name__} must have {method_name} method "
                "for protocol compliance"
            )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (MockServiceDiscoveryHandler, {}),
            (
                ConsulServiceDiscoveryHandler,
                {
                    "consul_host": "localhost",
                    "consul_port": 8500,
                    "consul_scheme": "http",
                },
            ),
        ],
    )
    def test_correlation_id_parameter_is_optional(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
    ) -> None:
        """correlation_id parameter has default value (optional) on all handlers."""
        handler = handler_class(**init_kwargs)

        for method_name in REQUIRED_PROTOCOL_METHODS:
            method = getattr(handler, method_name)
            sig = inspect.signature(method)

            if "correlation_id" in sig.parameters:
                param = sig.parameters["correlation_id"]
                assert param.default is not inspect.Parameter.empty, (
                    f"{handler_class.__name__}.{method_name} correlation_id parameter "
                    "must have a default value (be optional)"
                )


# =============================================================================
# Type Annotation Completeness Tests
# =============================================================================


class TestTypeAnnotationCompleteness:
    """Verify handlers have complete type annotations for ONEX compliance.

    Type annotations are required for:
    - Static type checking with mypy/pyright
    - Runtime introspection for protocol validation
    - Documentation generation
    - IDE support
    """

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (MockServiceDiscoveryHandler, {}),
            (
                ConsulServiceDiscoveryHandler,
                {
                    "consul_host": "localhost",
                    "consul_port": 8500,
                    "consul_scheme": "http",
                },
            ),
        ],
    )
    def test_handler_methods_have_return_annotations(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
    ) -> None:
        """All protocol methods have return type annotations."""
        handler = handler_class(**init_kwargs)

        for method_name in REQUIRED_PROTOCOL_METHODS:
            method = getattr(handler, method_name)
            sig = inspect.signature(method)
            assert sig.return_annotation != inspect.Signature.empty, (
                f"{handler_class.__name__}.{method_name} must have return type annotation"
            )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs"),
        [
            (MockServiceDiscoveryHandler, {}),
            (
                ConsulServiceDiscoveryHandler,
                {
                    "consul_host": "localhost",
                    "consul_port": 8500,
                    "consul_scheme": "http",
                },
            ),
        ],
    )
    def test_handler_methods_have_parameter_annotations(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
    ) -> None:
        """All protocol method parameters (except self) have type annotations."""
        handler = handler_class(**init_kwargs)

        for method_name in REQUIRED_PROTOCOL_METHODS:
            method = getattr(handler, method_name)
            sig = inspect.signature(method)

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                assert param.annotation != inspect.Parameter.empty, (
                    f"{handler_class.__name__}.{method_name} parameter '{param_name}' "
                    "must have type annotation"
                )


__all__: list[str] = [
    "REQUIRED_PROTOCOL_METHODS",
    "REQUIRED_PROTOCOL_PROPERTIES",
    "TestProtocolServiceDiscoveryHandlerInterface",
    "TestMockServiceDiscoveryHandlerProtocolCompliance",
    "TestConsulServiceDiscoveryHandlerProtocolCompliance",
    "TestCrossHandlerProtocolCompliance",
    "TestTypeAnnotationCompleteness",
]
