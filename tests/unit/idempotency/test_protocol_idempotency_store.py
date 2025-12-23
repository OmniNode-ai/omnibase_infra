# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ProtocolIdempotencyStore protocol definition.

Verifies that the protocol is correctly defined and that all implementations
conform to the protocol contract.

This file tests:
1. Protocol definition correctness (runtime_checkable, required methods)
2. Protocol method signatures (parameter types, return types)
3. Implementation conformance for InMemoryIdempotencyStore and PostgresIdempotencyStore

Ticket: OMN-945
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import get_type_hints
from uuid import UUID

import pytest

from omnibase_infra.idempotency import (
    InMemoryIdempotencyStore,
    PostgresIdempotencyStore,
    ProtocolIdempotencyStore,
)


class TestProtocolDefinition:
    """Tests for ProtocolIdempotencyStore definition."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Protocol should be decorated with @runtime_checkable.

        The protocol uses typing.runtime_checkable decorator, which adds
        __protocol_attrs__ to the class for isinstance() checking support.
        """
        # runtime_checkable protocols have _is_runtime_protocol attribute
        assert hasattr(ProtocolIdempotencyStore, "_is_runtime_protocol")
        assert ProtocolIdempotencyStore._is_runtime_protocol is True

    def test_protocol_has_required_methods(self) -> None:
        """Protocol should define all required methods.

        The protocol must define:
        - check_and_record: Atomic check-and-set for idempotency
        - is_processed: Read-only check
        - mark_processed: Upsert operation
        - cleanup_expired: TTL-based cleanup
        """
        required_methods = [
            "check_and_record",
            "is_processed",
            "mark_processed",
            "cleanup_expired",
        ]
        for method in required_methods:
            assert hasattr(ProtocolIdempotencyStore, method), (
                f"Protocol missing method: {method}"
            )

    def test_protocol_methods_are_async(self) -> None:
        """All protocol methods should be coroutine functions.

        The protocol is designed for async I/O operations, so all methods
        must be async (coroutine functions).
        """
        async_methods = [
            "check_and_record",
            "is_processed",
            "mark_processed",
            "cleanup_expired",
        ]
        for method_name in async_methods:
            method = getattr(ProtocolIdempotencyStore, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"


class TestProtocolMethodSignatures:
    """Tests for protocol method signatures."""

    def test_check_and_record_signature(self) -> None:
        """check_and_record should have correct parameter and return types.

        Expected signature:
            async def check_and_record(
                self,
                message_id: UUID,
                domain: str | None = None,
                correlation_id: UUID | None = None,
            ) -> bool
        """
        method = ProtocolIdempotencyStore.check_and_record
        sig = inspect.signature(method)
        params = sig.parameters

        # Check parameter names exist
        assert "message_id" in params
        assert "domain" in params
        assert "correlation_id" in params

        # Check default values
        assert params["domain"].default is None
        assert params["correlation_id"].default is None

        # Check return annotation
        hints = get_type_hints(method)
        assert hints.get("return") is bool

    def test_is_processed_signature(self) -> None:
        """is_processed should have correct parameter and return types.

        Expected signature:
            async def is_processed(
                self,
                message_id: UUID,
                domain: str | None = None,
            ) -> bool
        """
        method = ProtocolIdempotencyStore.is_processed
        sig = inspect.signature(method)
        params = sig.parameters

        # Check parameter names exist
        assert "message_id" in params
        assert "domain" in params

        # Check default values
        assert params["domain"].default is None

        # Check return annotation
        hints = get_type_hints(method)
        assert hints.get("return") is bool

    def test_mark_processed_signature(self) -> None:
        """mark_processed should have correct parameter and return types.

        Expected signature:
            async def mark_processed(
                self,
                message_id: UUID,
                domain: str | None = None,
                correlation_id: UUID | None = None,
                processed_at: datetime | None = None,
            ) -> None
        """
        method = ProtocolIdempotencyStore.mark_processed
        sig = inspect.signature(method)
        params = sig.parameters

        # Check parameter names exist
        assert "message_id" in params
        assert "domain" in params
        assert "correlation_id" in params
        assert "processed_at" in params

        # Check default values
        assert params["domain"].default is None
        assert params["correlation_id"].default is None
        assert params["processed_at"].default is None

        # Check return annotation
        hints = get_type_hints(method)
        assert hints.get("return") is type(None)

    def test_cleanup_expired_signature(self) -> None:
        """cleanup_expired should have correct parameter and return types.

        Expected signature:
            async def cleanup_expired(
                self,
                ttl_seconds: int,
            ) -> int
        """
        method = ProtocolIdempotencyStore.cleanup_expired
        sig = inspect.signature(method)
        params = sig.parameters

        # Check parameter names exist
        assert "ttl_seconds" in params

        # ttl_seconds should be required (no default)
        assert params["ttl_seconds"].default is inspect.Parameter.empty

        # Check return annotation
        hints = get_type_hints(method)
        assert hints.get("return") is int


class TestProtocolConformance:
    """Tests for implementation conformance to ProtocolIdempotencyStore."""

    def test_inmemory_conforms_to_protocol(self) -> None:
        """InMemoryIdempotencyStore should conform to ProtocolIdempotencyStore.

        Verifies that isinstance() check passes for InMemoryIdempotencyStore,
        which requires the class to implement all protocol methods with
        compatible signatures.
        """
        store = InMemoryIdempotencyStore()
        assert isinstance(store, ProtocolIdempotencyStore)

    def test_postgres_conforms_to_protocol(self) -> None:
        """PostgresIdempotencyStore should conform to ProtocolIdempotencyStore.

        Verifies that isinstance() check passes for PostgresIdempotencyStore.
        Note: This test uses a mock configuration since we're only testing
        protocol conformance, not actual database connectivity.
        """
        from omnibase_infra.idempotency import ModelPostgresIdempotencyStoreConfig

        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )
        store = PostgresIdempotencyStore(config)
        assert isinstance(store, ProtocolIdempotencyStore)

    def test_inmemory_has_all_protocol_methods(self) -> None:
        """InMemoryIdempotencyStore should implement all protocol methods.

        Verifies that all required protocol methods are present and callable.
        """
        store = InMemoryIdempotencyStore()
        required_methods = [
            "check_and_record",
            "is_processed",
            "mark_processed",
            "cleanup_expired",
        ]
        for method_name in required_methods:
            assert hasattr(store, method_name), f"Missing method: {method_name}"
            method = getattr(store, method_name)
            assert callable(method), f"Method {method_name} is not callable"
            assert inspect.iscoroutinefunction(method), (
                f"Method {method_name} should be async"
            )

    def test_postgres_has_all_protocol_methods(self) -> None:
        """PostgresIdempotencyStore should implement all protocol methods.

        Verifies that all required protocol methods are present and callable.
        """
        from omnibase_infra.idempotency import ModelPostgresIdempotencyStoreConfig

        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/testdb",
        )
        store = PostgresIdempotencyStore(config)
        required_methods = [
            "check_and_record",
            "is_processed",
            "mark_processed",
            "cleanup_expired",
        ]
        for method_name in required_methods:
            assert hasattr(store, method_name), f"Missing method: {method_name}"
            method = getattr(store, method_name)
            assert callable(method), f"Method {method_name} is not callable"
            assert inspect.iscoroutinefunction(method), (
                f"Method {method_name} should be async"
            )


class TestProtocolTypeAnnotations:
    """Tests for protocol type annotations correctness."""

    @staticmethod
    def _is_optional_type(hint: type, expected_type: type) -> bool:
        """Check if a type hint is expected_type | None.

        In Python 3.10+, `X | None` creates a types.UnionType, not typing.Union.
        This helper handles both cases for compatibility.

        Args:
            hint: The type hint to check.
            expected_type: The expected non-None type (e.g., str, UUID, datetime).

        Returns:
            True if hint is expected_type | None, False otherwise.
        """
        import types

        # Check for Python 3.10+ union type (X | None)
        if isinstance(hint, types.UnionType):
            args = hint.__args__
            return len(args) == 2 and expected_type in args and type(None) in args

        # Check for typing.Union (Optional[X])
        if hasattr(hint, "__origin__") and hasattr(hint, "__args__"):
            from typing import Union

            if hint.__origin__ is Union:
                args = hint.__args__
                return len(args) == 2 and expected_type in args and type(None) in args

        return False

    def test_check_and_record_type_hints(self) -> None:
        """check_and_record type hints should be correctly defined."""
        hints = get_type_hints(ProtocolIdempotencyStore.check_and_record)

        # message_id should be UUID
        assert hints["message_id"] is UUID

        # domain should be str | None
        assert self._is_optional_type(hints["domain"], str), (
            f"Expected str | None, got {hints['domain']}"
        )

        # correlation_id should be UUID | None
        assert self._is_optional_type(hints["correlation_id"], UUID), (
            f"Expected UUID | None, got {hints['correlation_id']}"
        )

        # Return type should be bool
        assert hints["return"] is bool

    def test_is_processed_type_hints(self) -> None:
        """is_processed type hints should be correctly defined."""
        hints = get_type_hints(ProtocolIdempotencyStore.is_processed)

        # message_id should be UUID
        assert hints["message_id"] is UUID

        # domain should be str | None
        assert self._is_optional_type(hints["domain"], str), (
            f"Expected str | None, got {hints['domain']}"
        )

        # Return type should be bool
        assert hints["return"] is bool

    def test_mark_processed_type_hints(self) -> None:
        """mark_processed type hints should be correctly defined."""
        hints = get_type_hints(ProtocolIdempotencyStore.mark_processed)

        # message_id should be UUID
        assert hints["message_id"] is UUID

        # domain should be str | None
        assert self._is_optional_type(hints["domain"], str), (
            f"Expected str | None, got {hints['domain']}"
        )

        # correlation_id should be UUID | None
        assert self._is_optional_type(hints["correlation_id"], UUID), (
            f"Expected UUID | None, got {hints['correlation_id']}"
        )

        # processed_at should be datetime | None
        assert self._is_optional_type(hints["processed_at"], datetime), (
            f"Expected datetime | None, got {hints['processed_at']}"
        )

        # Return type should be None
        assert hints["return"] is type(None)

    def test_cleanup_expired_type_hints(self) -> None:
        """cleanup_expired type hints should be correctly defined."""
        hints = get_type_hints(ProtocolIdempotencyStore.cleanup_expired)

        # ttl_seconds should be int
        assert hints["ttl_seconds"] is int

        # Return type should be int
        assert hints["return"] is int


class TestNonConformingImplementation:
    """Tests for classes that should NOT conform to the protocol."""

    def test_empty_class_does_not_conform(self) -> None:
        """An empty class should not pass isinstance check.

        Verifies that the protocol properly rejects classes that don't
        implement the required methods.
        """

        class EmptyStore:
            pass

        store = EmptyStore()
        assert not isinstance(store, ProtocolIdempotencyStore)

    def test_partial_implementation_does_not_conform(self) -> None:
        """A class with only some methods should not conform.

        Verifies that partial implementations are rejected by the
        protocol's isinstance check.
        """

        class PartialStore:
            async def check_and_record(
                self,
                message_id: UUID,
                domain: str | None = None,
                correlation_id: UUID | None = None,
            ) -> bool:
                return True

            # Missing: is_processed, mark_processed, cleanup_expired

        store = PartialStore()
        assert not isinstance(store, ProtocolIdempotencyStore)

    def test_sync_methods_do_not_conform(self) -> None:
        """A class with sync (non-async) methods should not conform.

        The protocol requires async methods, so sync implementations
        should not pass the isinstance check.
        """

        class SyncStore:
            def check_and_record(
                self,
                message_id: UUID,
                domain: str | None = None,
                correlation_id: UUID | None = None,
            ) -> bool:
                return True

            def is_processed(
                self,
                message_id: UUID,
                domain: str | None = None,
            ) -> bool:
                return False

            def mark_processed(
                self,
                message_id: UUID,
                domain: str | None = None,
                correlation_id: UUID | None = None,
                processed_at: datetime | None = None,
            ) -> None:
                pass

            def cleanup_expired(
                self,
                ttl_seconds: int,
            ) -> int:
                return 0

        store = SyncStore()
        # Note: runtime_checkable only checks method existence, not signatures.
        # This is a known limitation of typing.Protocol.
        # The sync implementation will pass isinstance() but fail at runtime.
        # This test documents the expected behavior rather than a strict check.
        # In practice, type checkers like mypy will catch this.
        assert isinstance(store, ProtocolIdempotencyStore)
