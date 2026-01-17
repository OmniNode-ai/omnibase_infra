# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Registration Storage Effect Node Models.

This module validates the models used by NodeRegistrationStorageEffect:
- ModelUpsertResult: Insert/update operation results
- ModelStorageQuery: Query parameters with filtering and pagination
- ModelStorageResult: Query result container
- ModelStorageHealthCheckResult: Health check status

Test Coverage:
    - Model construction and validation
    - Immutability (frozen=True)
    - Default values
    - Helper methods (was_inserted, was_updated, is_single_record_query)
    - Validation constraints (ge, le, etc.)

Related:
    - OMN-1131: Capability-oriented node architecture
    - NodeRegistrationStorageEffect: Effect node using these models
    - PR #119: Test coverage for capability-oriented nodes
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelStorageQuery,
    ModelUpsertResult,
)

# =============================================================================
# ModelUpsertResult Tests
# =============================================================================


class TestModelUpsertResultConstruction:
    """Tests for ModelUpsertResult construction and validation."""

    def test_create_success_insert(self) -> None:
        """Create successful insert result."""
        node_id = uuid4()
        result = ModelUpsertResult(
            success=True,
            node_id=node_id,
            operation="insert",
        )

        assert result.success is True
        assert result.node_id == node_id
        assert result.operation == "insert"
        assert result.error is None
        assert result.duration_ms == 0.0
        assert result.backend_type == "unknown"

    def test_create_success_update(self) -> None:
        """Create successful update result."""
        node_id = uuid4()
        correlation_id = uuid4()
        result = ModelUpsertResult(
            success=True,
            node_id=node_id,
            operation="update",
            duration_ms=45.5,
            backend_type="postgresql",
            correlation_id=correlation_id,
        )

        assert result.success is True
        assert result.operation == "update"
        assert result.duration_ms == 45.5
        assert result.backend_type == "postgresql"
        assert result.correlation_id == correlation_id

    def test_create_failure_result(self) -> None:
        """Create failure result with error message."""
        node_id = uuid4()
        result = ModelUpsertResult(
            success=False,
            node_id=node_id,
            operation="insert",
            error="Connection timeout",
            duration_ms=5000.0,
        )

        assert result.success is False
        assert result.error == "Connection timeout"

    def test_invalid_operation_raises_validation_error(self) -> None:
        """Invalid operation value raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelUpsertResult(
                success=True,
                node_id=uuid4(),
                operation="delete",  # type: ignore[arg-type] # Invalid
            )

        assert "operation" in str(exc_info.value)

    def test_negative_duration_raises_validation_error(self) -> None:
        """Negative duration_ms raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelUpsertResult(
                success=True,
                node_id=uuid4(),
                operation="insert",
                duration_ms=-1.0,
            )

        assert "duration_ms" in str(exc_info.value)


class TestModelUpsertResultImmutability:
    """Tests for ModelUpsertResult immutability."""

    def test_model_is_frozen(self) -> None:
        """ModelUpsertResult is immutable (frozen)."""
        result = ModelUpsertResult(
            success=True,
            node_id=uuid4(),
            operation="insert",
        )

        with pytest.raises((TypeError, ValidationError)):
            result.success = False  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed."""
        with pytest.raises(ValidationError) as exc_info:
            ModelUpsertResult(
                success=True,
                node_id=uuid4(),
                operation="insert",
                extra_field="not allowed",  # type: ignore[call-arg]
            )

        assert "extra_field" in str(exc_info.value) or "Extra inputs" in str(
            exc_info.value
        )


class TestModelUpsertResultHelperMethods:
    """Tests for ModelUpsertResult helper methods."""

    def test_was_inserted_returns_true_for_successful_insert(self) -> None:
        """was_inserted() returns True for successful insert."""
        result = ModelUpsertResult(
            success=True,
            node_id=uuid4(),
            operation="insert",
        )

        assert result.was_inserted() is True
        assert result.was_updated() is False

    def test_was_inserted_returns_false_for_failed_insert(self) -> None:
        """was_inserted() returns False for failed insert."""
        result = ModelUpsertResult(
            success=False,
            node_id=uuid4(),
            operation="insert",
            error="Failed",
        )

        assert result.was_inserted() is False

    def test_was_updated_returns_true_for_successful_update(self) -> None:
        """was_updated() returns True for successful update."""
        result = ModelUpsertResult(
            success=True,
            node_id=uuid4(),
            operation="update",
        )

        assert result.was_updated() is True
        assert result.was_inserted() is False

    def test_was_updated_returns_false_for_failed_update(self) -> None:
        """was_updated() returns False for failed update."""
        result = ModelUpsertResult(
            success=False,
            node_id=uuid4(),
            operation="update",
            error="Failed",
        )

        assert result.was_updated() is False

    def test_was_inserted_and_was_updated_mutually_exclusive(self) -> None:
        """was_inserted() and was_updated() are mutually exclusive."""
        insert_result = ModelUpsertResult(
            success=True,
            node_id=uuid4(),
            operation="insert",
        )
        update_result = ModelUpsertResult(
            success=True,
            node_id=uuid4(),
            operation="update",
        )

        # For insert: was_inserted True, was_updated False
        assert insert_result.was_inserted() is True
        assert insert_result.was_updated() is False

        # For update: was_inserted False, was_updated True
        assert update_result.was_inserted() is False
        assert update_result.was_updated() is True


# =============================================================================
# ModelStorageQuery Tests
# =============================================================================


class TestModelStorageQueryConstruction:
    """Tests for ModelStorageQuery construction and validation."""

    def test_create_empty_query(self) -> None:
        """Create query with all defaults (match all)."""
        query = ModelStorageQuery()

        assert query.node_id is None
        assert query.node_type is None
        assert query.capability_filter is None
        assert query.limit == 100
        assert query.offset == 0

    def test_create_query_by_node_id(self) -> None:
        """Create query filtering by specific node_id."""
        node_id = uuid4()
        query = ModelStorageQuery(node_id=node_id)

        assert query.node_id == node_id
        assert query.is_single_record_query() is True

    def test_create_query_by_node_type(self) -> None:
        """Create query filtering by node type."""
        query = ModelStorageQuery(node_type=EnumNodeKind.EFFECT)

        assert query.node_type == EnumNodeKind.EFFECT
        assert query.is_single_record_query() is False

    def test_create_query_by_capability(self) -> None:
        """Create query filtering by capability."""
        query = ModelStorageQuery(capability_filter="registration.storage")

        assert query.capability_filter == "registration.storage"

    def test_create_query_with_pagination(self) -> None:
        """Create query with pagination parameters."""
        query = ModelStorageQuery(limit=50, offset=100)

        assert query.limit == 50
        assert query.offset == 100

    def test_create_combined_query(self) -> None:
        """Create query with multiple filters."""
        query = ModelStorageQuery(
            node_type=EnumNodeKind.EFFECT,
            capability_filter="storage",
            limit=25,
        )

        assert query.node_type == EnumNodeKind.EFFECT
        assert query.capability_filter == "storage"
        assert query.limit == 25


class TestModelStorageQueryValidation:
    """Tests for ModelStorageQuery validation constraints."""

    def test_limit_minimum_is_one(self) -> None:
        """limit must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            ModelStorageQuery(limit=0)

        assert "limit" in str(exc_info.value)

    def test_limit_maximum_is_1000(self) -> None:
        """limit must be at most 1000."""
        with pytest.raises(ValidationError) as exc_info:
            ModelStorageQuery(limit=1001)

        assert "limit" in str(exc_info.value)

    def test_offset_minimum_is_zero(self) -> None:
        """offset must be at least 0."""
        with pytest.raises(ValidationError) as exc_info:
            ModelStorageQuery(offset=-1)

        assert "offset" in str(exc_info.value)

    def test_limit_at_boundaries(self) -> None:
        """limit accepts boundary values."""
        query_min = ModelStorageQuery(limit=1)
        query_max = ModelStorageQuery(limit=1000)

        assert query_min.limit == 1
        assert query_max.limit == 1000

    def test_offset_at_boundary(self) -> None:
        """offset accepts zero."""
        query = ModelStorageQuery(offset=0)
        assert query.offset == 0


class TestModelStorageQueryImmutability:
    """Tests for ModelStorageQuery immutability."""

    def test_model_is_frozen(self) -> None:
        """ModelStorageQuery is immutable (frozen)."""
        query = ModelStorageQuery()

        with pytest.raises((TypeError, ValidationError)):
            query.limit = 50  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields are not allowed."""
        with pytest.raises(ValidationError):
            ModelStorageQuery(
                unknown_field="value",  # type: ignore[call-arg]
            )


class TestModelStorageQueryHelperMethods:
    """Tests for ModelStorageQuery helper methods."""

    def test_is_single_record_query_with_node_id(self) -> None:
        """is_single_record_query() returns True when node_id is set."""
        query = ModelStorageQuery(node_id=uuid4())
        assert query.is_single_record_query() is True

    def test_is_single_record_query_without_node_id(self) -> None:
        """is_single_record_query() returns False when node_id is None."""
        query = ModelStorageQuery(node_type=EnumNodeKind.EFFECT)
        assert query.is_single_record_query() is False

    def test_is_single_record_query_empty_query(self) -> None:
        """is_single_record_query() returns False for empty query."""
        query = ModelStorageQuery()
        assert query.is_single_record_query() is False


# =============================================================================
# Model Serialization Tests
# =============================================================================


class TestModelSerialization:
    """Tests for model serialization/deserialization."""

    def test_upsert_result_to_dict(self) -> None:
        """ModelUpsertResult serializes to dict correctly."""
        node_id = uuid4()
        correlation_id = uuid4()
        result = ModelUpsertResult(
            success=True,
            node_id=node_id,
            operation="insert",
            duration_ms=10.5,
            backend_type="postgresql",
            correlation_id=correlation_id,
        )

        data = result.model_dump()

        assert data["success"] is True
        assert data["node_id"] == node_id
        assert data["operation"] == "insert"
        assert data["duration_ms"] == 10.5
        assert data["backend_type"] == "postgresql"
        assert data["correlation_id"] == correlation_id

    def test_storage_query_to_dict(self) -> None:
        """ModelStorageQuery serializes to dict correctly."""
        node_id = uuid4()
        query = ModelStorageQuery(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            limit=50,
        )

        data = query.model_dump()

        assert data["node_id"] == node_id
        assert data["node_type"] == EnumNodeKind.EFFECT
        assert data["limit"] == 50

    def test_upsert_result_json_round_trip(self) -> None:
        """ModelUpsertResult survives JSON round-trip."""
        result = ModelUpsertResult(
            success=True,
            node_id=uuid4(),
            operation="update",
            duration_ms=25.0,
        )

        json_str = result.model_dump_json()
        restored = ModelUpsertResult.model_validate_json(json_str)

        assert restored.success == result.success
        assert restored.node_id == result.node_id
        assert restored.operation == result.operation
        assert restored.duration_ms == result.duration_ms

    def test_storage_query_json_round_trip(self) -> None:
        """ModelStorageQuery survives JSON round-trip."""
        query = ModelStorageQuery(
            node_type=EnumNodeKind.COMPUTE,
            capability_filter="compute.transform",
            limit=75,
            offset=25,
        )

        json_str = query.model_dump_json()
        restored = ModelStorageQuery.model_validate_json(json_str)

        assert restored.node_type == query.node_type
        assert restored.capability_filter == query.capability_filter
        assert restored.limit == query.limit
        assert restored.offset == query.offset


__all__: list[str] = [
    "TestModelUpsertResultConstruction",
    "TestModelUpsertResultImmutability",
    "TestModelUpsertResultHelperMethods",
    "TestModelStorageQueryConstruction",
    "TestModelStorageQueryValidation",
    "TestModelStorageQueryImmutability",
    "TestModelStorageQueryHelperMethods",
    "TestModelSerialization",
]
