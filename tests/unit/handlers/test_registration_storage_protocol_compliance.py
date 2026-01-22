# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Registration Storage Handler Protocol Compliance.

This module validates that registration storage handler implementations
correctly implement the ProtocolRegistrationPersistence protocol.

Protocol Compliance Testing
---------------------------
Per ONEX patterns, protocol compliance is verified using duck typing
(hasattr() and callable() checks) rather than isinstance() to support
structural subtyping. This approach allows handlers to implement the
protocol without explicit inheritance.

ProtocolRegistrationPersistence Interface
--------------------------------------------
Required Members:
    - handler_type (property): Returns handler type identifier string
    - store_registration(record, correlation_id): Async method for storing records
    - query_registrations(...): Async method for querying records
    - update_registration(...): Async method for updating records
    - delete_registration(node_id, correlation_id): Async method for deleting records
    - health_check(correlation_id): Async method for health verification

Handler Implementations Tested:
    - HandlerRegistrationStorageMock: In-memory mock for testing
    - HandlerRegistrationStoragePostgres: PostgreSQL backend implementation

Related:
    - OMN-1131: Capability-oriented node architecture
    - ProtocolRegistrationPersistence: Protocol definition
    - PR #119: Test coverage for protocol compliance
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.registration_storage.handler_registration_storage_mock import (
    HandlerRegistrationStorageMock,
)
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)
from omnibase_infra.handlers.registration_storage.protocol_registration_persistence import (
    ProtocolRegistrationPersistence,
)

# =============================================================================
# Protocol Method Definitions
# =============================================================================

REQUIRED_PROTOCOL_METHODS: tuple[str, ...] = (
    "store_registration",
    "query_registrations",
    "update_registration",
    "delete_registration",
    "health_check",
)
"""Required async methods that all handlers must implement."""

REQUIRED_PROTOCOL_PROPERTIES: tuple[str, ...] = ("handler_type",)
"""Required properties that all handlers must implement."""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler() -> HandlerRegistrationStorageMock:
    """Create HandlerRegistrationStorageMock instance for testing."""
    return HandlerRegistrationStorageMock()


@pytest.fixture
def postgres_handler() -> HandlerRegistrationStoragePostgres:
    """Create HandlerRegistrationStoragePostgres instance for testing.

    Note: This creates the handler without initializing the connection pool.
    Protocol compliance tests only verify interface structure, not runtime behavior.
    """
    mock_container = MagicMock(spec=ModelONEXContainer)
    test_password = "test_password"
    return HandlerRegistrationStoragePostgres(
        container=mock_container,
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password=test_password,
    )


# =============================================================================
# Protocol Interface Verification Tests
# =============================================================================


class TestProtocolRegistrationPersistenceInterface:
    """Verify ProtocolRegistrationPersistence is a valid runtime-checkable protocol.

    These tests ensure the protocol definition itself is correct and can be
    used for runtime type checking with isinstance().
    """

    def test_protocol_is_runtime_checkable(self) -> None:
        """ProtocolRegistrationPersistence is decorated with @runtime_checkable."""
        # Protocol should be decorated with @runtime_checkable
        assert hasattr(
            ProtocolRegistrationPersistence, "__protocol_attrs__"
        ) or hasattr(ProtocolRegistrationPersistence, "_is_runtime_protocol"), (
            "ProtocolRegistrationPersistence should be @runtime_checkable"
        )

    def test_protocol_defines_handler_type_property(self) -> None:
        """Protocol defines handler_type property."""
        # Check that handler_type is in the protocol's annotations or attrs
        assert "handler_type" in dir(ProtocolRegistrationPersistence), (
            "Protocol must define handler_type property"
        )

    def test_protocol_defines_required_methods(self) -> None:
        """Protocol defines all required async methods."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            assert hasattr(ProtocolRegistrationPersistence, method_name), (
                f"Protocol must define {method_name} method"
            )


# =============================================================================
# HandlerRegistrationStorageMock Protocol Compliance Tests
# =============================================================================


class TestHandlerRegistrationStorageMockProtocolCompliance:
    """Validate HandlerRegistrationStorageMock implements ProtocolRegistrationPersistence.

    Uses duck typing verification per ONEX patterns to ensure the mock handler
    correctly implements all protocol requirements.
    """

    def test_mock_handler_isinstance_protocol(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """HandlerRegistrationStorageMock passes isinstance check for protocol."""
        assert isinstance(mock_handler, ProtocolRegistrationPersistence), (
            "HandlerRegistrationStorageMock must be an instance of "
            "ProtocolRegistrationPersistence protocol"
        )

    def test_mock_handler_has_handler_type_property(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """HandlerRegistrationStorageMock has handler_type property."""
        assert hasattr(mock_handler, "handler_type"), (
            "HandlerRegistrationStorageMock must have handler_type property"
        )

        # Verify handler_type returns expected value
        handler_type = mock_handler.handler_type
        assert handler_type == "mock", (
            f"HandlerRegistrationStorageMock.handler_type should return 'mock', "
            f"got '{handler_type}'"
        )

    def test_mock_handler_has_all_required_methods(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """HandlerRegistrationStorageMock has all required protocol methods."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            assert hasattr(mock_handler, method_name), (
                f"HandlerRegistrationStorageMock must have {method_name} method"
            )
            assert callable(getattr(mock_handler, method_name)), (
                f"HandlerRegistrationStorageMock.{method_name} must be callable"
            )

    def test_mock_handler_methods_are_async(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """All required methods on HandlerRegistrationStorageMock are async coroutines."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            method = getattr(mock_handler, method_name)
            assert asyncio.iscoroutinefunction(method), (
                f"HandlerRegistrationStorageMock.{method_name} must be an async coroutine"
            )

    def test_mock_handler_store_registration_signature(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """store_registration method has correct parameter signature."""
        sig = inspect.signature(mock_handler.store_registration)
        params = list(sig.parameters.keys())

        assert "record" in params, "store_registration must accept 'record' parameter"
        assert "correlation_id" in params, (
            "store_registration must accept 'correlation_id' parameter"
        )

    def test_mock_handler_query_registrations_signature(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """query_registrations method has correct parameter signature."""
        sig = inspect.signature(mock_handler.query_registrations)
        params = list(sig.parameters.keys())

        # HandlerRegistrationStorageMock uses individual parameters
        assert "node_type" in params or "query" in params, (
            "query_registrations must accept filtering parameters"
        )
        assert "correlation_id" in params, (
            "query_registrations must accept 'correlation_id' parameter"
        )

    def test_mock_handler_delete_registration_signature(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """delete_registration method has correct parameter signature."""
        sig = inspect.signature(mock_handler.delete_registration)
        params = list(sig.parameters.keys())

        # Protocol now uses request model pattern per ONEX standards
        assert "request" in params, (
            "delete_registration must accept 'request' parameter "
            "(ModelDeleteRegistrationRequest with node_id and correlation_id)"
        )

    def test_mock_handler_health_check_signature(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """health_check method has correct parameter signature."""
        sig = inspect.signature(mock_handler.health_check)
        params = list(sig.parameters.keys())

        assert "correlation_id" in params, (
            "health_check must accept 'correlation_id' parameter"
        )

    def test_mock_handler_store_registration_return_type_annotation(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """store_registration method has return type annotation."""
        sig = inspect.signature(mock_handler.store_registration)
        assert sig.return_annotation != inspect.Signature.empty, (
            "store_registration must have return type annotation"
        )

    def test_mock_handler_health_check_return_type_annotation(
        self, mock_handler: HandlerRegistrationStorageMock
    ) -> None:
        """health_check method has return type annotation."""
        sig = inspect.signature(mock_handler.health_check)
        assert sig.return_annotation != inspect.Signature.empty, (
            "health_check must have return type annotation"
        )


# =============================================================================
# HandlerRegistrationStoragePostgres Protocol Compliance Tests
# =============================================================================


class TestHandlerRegistrationStoragePostgresProtocolCompliance:
    """Validate HandlerRegistrationStoragePostgres implements ProtocolRegistrationPersistence.

    Uses duck typing verification per ONEX patterns to ensure the PostgreSQL handler
    correctly implements all protocol requirements.

    Note: These tests verify interface compliance only, not runtime behavior.
    Integration tests with actual PostgreSQL are in test_db_handler_integration.py.
    """

    def test_postgres_handler_isinstance_protocol(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """HandlerRegistrationStoragePostgres passes isinstance check for protocol."""
        assert isinstance(postgres_handler, ProtocolRegistrationPersistence), (
            "HandlerRegistrationStoragePostgres must be an instance of "
            "ProtocolRegistrationPersistence protocol"
        )

    def test_postgres_handler_has_handler_type_property(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """HandlerRegistrationStoragePostgres has handler_type property."""
        assert hasattr(postgres_handler, "handler_type"), (
            "HandlerRegistrationStoragePostgres must have handler_type property"
        )

        # Verify handler_type returns expected value
        handler_type = postgres_handler.handler_type
        assert handler_type == "postgresql", (
            f"HandlerRegistrationStoragePostgres.handler_type should return 'postgresql', "
            f"got '{handler_type}'"
        )

    def test_postgres_handler_has_all_required_methods(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """HandlerRegistrationStoragePostgres has all required protocol methods."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            assert hasattr(postgres_handler, method_name), (
                f"HandlerRegistrationStoragePostgres must have {method_name} method"
            )
            assert callable(getattr(postgres_handler, method_name)), (
                f"HandlerRegistrationStoragePostgres.{method_name} must be callable"
            )

    def test_postgres_handler_methods_are_async(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """All required methods on HandlerRegistrationStoragePostgres are async coroutines."""
        for method_name in REQUIRED_PROTOCOL_METHODS:
            method = getattr(postgres_handler, method_name)
            assert asyncio.iscoroutinefunction(method), (
                f"HandlerRegistrationStoragePostgres.{method_name} must be an async coroutine"
            )

    def test_postgres_handler_store_registration_signature(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """store_registration method has correct parameter signature."""
        sig = inspect.signature(postgres_handler.store_registration)
        params = list(sig.parameters.keys())

        assert "record" in params, "store_registration must accept 'record' parameter"
        assert "correlation_id" in params, (
            "store_registration must accept 'correlation_id' parameter"
        )

    def test_postgres_handler_delete_registration_signature(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """delete_registration method has correct parameter signature."""
        sig = inspect.signature(postgres_handler.delete_registration)
        params = list(sig.parameters.keys())

        # Protocol now uses request model pattern per ONEX standards
        assert "request" in params, (
            "delete_registration must accept 'request' parameter "
            "(ModelDeleteRegistrationRequest with node_id and correlation_id)"
        )

    def test_postgres_handler_health_check_signature(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """health_check method has correct parameter signature."""
        sig = inspect.signature(postgres_handler.health_check)
        params = list(sig.parameters.keys())

        assert "correlation_id" in params, (
            "health_check must accept 'correlation_id' parameter"
        )

    def test_postgres_handler_has_circuit_breaker_mixin(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """HandlerRegistrationStoragePostgres inherits MixinAsyncCircuitBreaker."""
        from omnibase_infra.mixins import MixinAsyncCircuitBreaker

        assert isinstance(postgres_handler, MixinAsyncCircuitBreaker), (
            "HandlerRegistrationStoragePostgres should inherit MixinAsyncCircuitBreaker "
            "for circuit breaker resilience"
        )

    def test_postgres_handler_has_circuit_breaker_attributes(
        self, postgres_handler: HandlerRegistrationStoragePostgres
    ) -> None:
        """HandlerRegistrationStoragePostgres has circuit breaker attributes from mixin."""
        # Circuit breaker mixin attributes
        assert hasattr(postgres_handler, "_circuit_breaker_lock"), (
            "HandlerRegistrationStoragePostgres must have _circuit_breaker_lock"
        )
        assert hasattr(postgres_handler, "_circuit_breaker_open"), (
            "HandlerRegistrationStoragePostgres must have _circuit_breaker_open"
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
            (HandlerRegistrationStorageMock, {}, "mock"),
            (
                HandlerRegistrationStoragePostgres,
                {
                    "container": MagicMock(spec=ModelONEXContainer),
                    "host": "localhost",
                    "port": 5432,
                    "database": "test",
                    "user": "test",
                    "password": "test",
                },
                "postgresql",
            ),
        ],
    )
    def test_handler_is_protocol_instance(
        self,
        handler_class: type,
        init_kwargs: dict[str, object],
        expected_handler_type: str,
    ) -> None:
        """All handlers pass isinstance check for ProtocolRegistrationPersistence."""
        handler = handler_class(**init_kwargs)
        assert isinstance(handler, ProtocolRegistrationPersistence), (
            f"{handler_class.__name__} must be an instance of "
            "ProtocolRegistrationPersistence protocol"
        )

    @pytest.mark.parametrize(
        ("handler_class", "init_kwargs", "expected_handler_type"),
        [
            (HandlerRegistrationStorageMock, {}, "mock"),
            (
                HandlerRegistrationStoragePostgres,
                {
                    "container": MagicMock(spec=ModelONEXContainer),
                    "host": "localhost",
                    "port": 5432,
                    "database": "test",
                    "user": "test",
                    "password": "test",
                },
                "postgresql",
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
            (HandlerRegistrationStorageMock, {}),
            (
                HandlerRegistrationStoragePostgres,
                {
                    "container": MagicMock(spec=ModelONEXContainer),
                    "host": "localhost",
                    "port": 5432,
                    "database": "test",
                    "user": "test",
                    "password": "test",
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
            (HandlerRegistrationStorageMock, {}),
            (
                HandlerRegistrationStoragePostgres,
                {
                    "container": MagicMock(spec=ModelONEXContainer),
                    "host": "localhost",
                    "port": 5432,
                    "database": "test",
                    "user": "test",
                    "password": "test",
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
            (HandlerRegistrationStorageMock, {}),
            (
                HandlerRegistrationStoragePostgres,
                {
                    "container": MagicMock(spec=ModelONEXContainer),
                    "host": "localhost",
                    "port": 5432,
                    "database": "test",
                    "user": "test",
                    "password": "test",
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
            (HandlerRegistrationStorageMock, {}),
            (
                HandlerRegistrationStoragePostgres,
                {
                    "container": MagicMock(spec=ModelONEXContainer),
                    "host": "localhost",
                    "port": 5432,
                    "database": "test",
                    "user": "test",
                    "password": "test",
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
    "TestProtocolRegistrationPersistenceInterface",
    "TestHandlerRegistrationStorageMockProtocolCompliance",
    "TestHandlerRegistrationStoragePostgresProtocolCompliance",
    "TestCrossHandlerProtocolCompliance",
    "TestTypeAnnotationCompleteness",
]
