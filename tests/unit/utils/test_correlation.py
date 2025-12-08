# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for correlation ID utilities.

Tests the correlation ID generation, context management, and propagation
functionality used for distributed tracing across infrastructure components.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.utils.correlation import (
    CorrelationContext,
    clear_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
)


class TestGenerateCorrelationId:
    """Tests for generate_correlation_id function."""

    def test_returns_uuid(self) -> None:
        """Should return a UUID instance."""
        result = generate_correlation_id()
        assert isinstance(result, UUID)

    def test_returns_uuid4(self) -> None:
        """Should return a UUID version 4."""
        result = generate_correlation_id()
        # UUID4 has version 4 in the version field
        assert result.version == 4

    def test_generates_unique_ids(self) -> None:
        """Should generate unique IDs on each call."""
        ids = {generate_correlation_id() for _ in range(100)}
        # All IDs should be unique
        assert len(ids) == 100

    def test_returns_different_id_each_call(self) -> None:
        """Should return different ID on each call."""
        id1 = generate_correlation_id()
        id2 = generate_correlation_id()
        assert id1 != id2


class TestGetCorrelationId:
    """Tests for get_correlation_id function."""

    def setup_method(self) -> None:
        """Clear correlation ID before each test."""
        clear_correlation_id()

    def test_generates_new_id_when_not_set(self) -> None:
        """Should generate a new ID when none is set in context."""
        result = get_correlation_id()
        assert isinstance(result, UUID)
        assert result.version == 4

    def test_returns_same_id_on_subsequent_calls(self) -> None:
        """Should return the same ID on subsequent calls."""
        id1 = get_correlation_id()
        id2 = get_correlation_id()
        assert id1 == id2

    def test_returns_set_id(self) -> None:
        """Should return ID that was previously set."""
        expected = generate_correlation_id()
        set_correlation_id(expected)
        result = get_correlation_id()
        assert result == expected


class TestSetCorrelationId:
    """Tests for set_correlation_id function."""

    def setup_method(self) -> None:
        """Clear correlation ID before each test."""
        clear_correlation_id()

    def test_sets_correlation_id(self) -> None:
        """Should set the correlation ID in context."""
        expected = generate_correlation_id()
        set_correlation_id(expected)
        assert get_correlation_id() == expected

    def test_overwrites_existing_id(self) -> None:
        """Should overwrite an existing correlation ID."""
        first = generate_correlation_id()
        second = generate_correlation_id()

        set_correlation_id(first)
        assert get_correlation_id() == first

        set_correlation_id(second)
        assert get_correlation_id() == second


class TestClearCorrelationId:
    """Tests for clear_correlation_id function."""

    def test_clears_correlation_id(self) -> None:
        """Should clear the correlation ID from context."""
        set_correlation_id(generate_correlation_id())
        clear_correlation_id()

        # Getting correlation ID should generate a new one
        id1 = get_correlation_id()
        clear_correlation_id()
        id2 = get_correlation_id()

        # They should be different since we cleared between calls
        assert id1 != id2

    def test_clear_when_not_set(self) -> None:
        """Should not raise when clearing unset ID."""
        clear_correlation_id()  # Should not raise
        clear_correlation_id()  # Should not raise


class TestCorrelationContext:
    """Tests for CorrelationContext context manager."""

    def setup_method(self) -> None:
        """Clear correlation ID before each test."""
        clear_correlation_id()

    def test_generates_new_id_when_not_provided(self) -> None:
        """Should generate a new correlation ID when none is provided."""
        with CorrelationContext() as correlation_id:
            assert isinstance(correlation_id, UUID)
            assert correlation_id.version == 4

    def test_uses_provided_id(self) -> None:
        """Should use the provided correlation ID."""
        expected = generate_correlation_id()
        with CorrelationContext(correlation_id=expected) as correlation_id:
            assert correlation_id == expected

    def test_sets_correlation_id_in_context(self) -> None:
        """Should set correlation ID in context during execution."""
        with CorrelationContext() as correlation_id:
            assert get_correlation_id() == correlation_id

    def test_restores_previous_id_on_exit(self) -> None:
        """Should restore previous correlation ID on context exit."""
        outer_id = generate_correlation_id()
        set_correlation_id(outer_id)

        with CorrelationContext() as inner_id:
            assert get_correlation_id() == inner_id
            assert inner_id != outer_id

        # After exiting, outer ID should be restored
        assert get_correlation_id() == outer_id

    def test_nested_contexts(self) -> None:
        """Should properly handle nested correlation contexts."""
        with CorrelationContext() as outer_id:
            assert get_correlation_id() == outer_id

            with CorrelationContext() as inner_id:
                assert get_correlation_id() == inner_id
                assert inner_id != outer_id

            # After inner context exits, outer should be restored
            assert get_correlation_id() == outer_id

    def test_handles_exception_in_context(self) -> None:
        """Should restore correlation ID even when exception is raised."""
        outer_id = generate_correlation_id()
        set_correlation_id(outer_id)

        with pytest.raises(ValueError, match="test error"):
            with CorrelationContext():
                raise ValueError("test error")

        # Outer ID should still be restored
        assert get_correlation_id() == outer_id

    def test_correlation_id_property(self) -> None:
        """Should expose correlation_id via property."""
        expected = generate_correlation_id()
        context = CorrelationContext(correlation_id=expected)
        assert context.correlation_id == expected

    def test_property_returns_generated_id(self) -> None:
        """Should return the generated ID via property when not provided."""
        context = CorrelationContext()
        assert isinstance(context.correlation_id, UUID)

        # The same ID should be returned from property and __enter__
        with context as entered_id:
            assert entered_id == context.correlation_id


class TestCorrelationContextIntegration:
    """Integration tests for correlation context with other functions."""

    def setup_method(self) -> None:
        """Clear correlation ID before each test."""
        clear_correlation_id()

    def test_get_correlation_id_within_context(self) -> None:
        """get_correlation_id should return context's ID when in context."""
        with CorrelationContext() as context_id:
            retrieved_id = get_correlation_id()
            assert retrieved_id == context_id

    def test_set_correlation_id_within_context(self) -> None:
        """set_correlation_id should work within context."""
        with CorrelationContext() as original_id:
            new_id = generate_correlation_id()
            set_correlation_id(new_id)
            assert get_correlation_id() == new_id
            assert new_id != original_id

    def test_clear_within_context(self) -> None:
        """clear_correlation_id should work within context."""
        with CorrelationContext():
            clear_correlation_id()
            # Next get should generate a new ID
            new_id = get_correlation_id()
            assert isinstance(new_id, UUID)


class TestCorrelationIdStringConversion:
    """Tests for string representation of correlation IDs."""

    def test_str_conversion(self) -> None:
        """Correlation ID should convert to standard UUID string format."""
        correlation_id = generate_correlation_id()
        str_id = str(correlation_id)

        # UUID string format: 8-4-4-4-12 hex digits
        assert len(str_id) == 36
        assert str_id.count("-") == 4

    def test_roundtrip_string_conversion(self) -> None:
        """Should be able to convert to string and back."""
        original = generate_correlation_id()
        str_id = str(original)
        reconstructed = UUID(str_id)
        assert original == reconstructed
