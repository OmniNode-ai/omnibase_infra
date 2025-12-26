# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive tests for ModelTracingContext sentinel behavior.

Tests cover:
- Sentinel UUID behavior (default construction uses nil UUID)
- None-to-sentinel conversion via field validators
- has_* properties (correlation_id, trace_id, span_id)
- is_empty property
- Factory methods (empty(), from_uuids())
- to_dict() behavior (only includes set fields, string values)
- String UUID conversion
- Pydantic model behaviors (immutability, serialization)

OMN-1004: Union reduction phase - ModelTracingContext uses sentinel values
instead of nullable unions to minimize union count.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.dispatch.model_tracing_context import ModelTracingContext

# Sentinel UUID for comparison (nil UUID: 00000000-0000-0000-0000-000000000000)
_SENTINEL_UUID = UUID(int=0)


# ============================================================================
# Sentinel UUID Behavior Tests
# ============================================================================


@pytest.mark.unit
class TestModelTracingContextSentinel:
    """Tests for sentinel UUID behavior in ModelTracingContext.

    The model uses nil UUID (00000000-0000-0000-0000-000000000000) as a sentinel
    value meaning "not set", instead of using Optional[UUID] = None.
    """

    def test_default_construction_uses_sentinel_for_correlation_id(self) -> None:
        """Test that default construction sets correlation_id to sentinel UUID."""
        ctx = ModelTracingContext()
        assert ctx.correlation_id == _SENTINEL_UUID

    def test_default_construction_uses_sentinel_for_trace_id(self) -> None:
        """Test that default construction sets trace_id to sentinel UUID."""
        ctx = ModelTracingContext()
        assert ctx.trace_id == _SENTINEL_UUID

    def test_default_construction_uses_sentinel_for_span_id(self) -> None:
        """Test that default construction sets span_id to sentinel UUID."""
        ctx = ModelTracingContext()
        assert ctx.span_id == _SENTINEL_UUID

    def test_default_construction_all_fields_are_sentinel(self) -> None:
        """Test that default construction sets all fields to sentinel UUID."""
        ctx = ModelTracingContext()

        assert ctx.correlation_id == _SENTINEL_UUID
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == _SENTINEL_UUID

    def test_sentinel_uuid_is_nil_uuid(self) -> None:
        """Test that sentinel UUID is the nil UUID (all zeros)."""
        ctx = ModelTracingContext()

        expected_str = "00000000-0000-0000-0000-000000000000"
        assert str(ctx.correlation_id) == expected_str
        assert str(ctx.trace_id) == expected_str
        assert str(ctx.span_id) == expected_str


# ============================================================================
# None-to-Sentinel Conversion Tests
# ============================================================================


@pytest.mark.unit
class TestNoneToSentinelConversion:
    """Tests for None-to-sentinel conversion in ModelTracingContext.

    The model's field validators convert None to the sentinel UUID for
    backwards compatibility with code that expects to pass None.
    """

    def test_none_correlation_id_converted_to_sentinel(self) -> None:
        """Test that passing correlation_id=None converts to sentinel UUID."""
        ctx = ModelTracingContext(correlation_id=None)
        assert ctx.correlation_id == _SENTINEL_UUID

    def test_none_trace_id_converted_to_sentinel(self) -> None:
        """Test that passing trace_id=None converts to sentinel UUID."""
        ctx = ModelTracingContext(trace_id=None)
        assert ctx.trace_id == _SENTINEL_UUID

    def test_none_span_id_converted_to_sentinel(self) -> None:
        """Test that passing span_id=None converts to sentinel UUID."""
        ctx = ModelTracingContext(span_id=None)
        assert ctx.span_id == _SENTINEL_UUID

    def test_all_none_converted_to_sentinel(self) -> None:
        """Test that passing all None values converts to sentinel UUIDs."""
        ctx = ModelTracingContext(
            correlation_id=None,
            trace_id=None,
            span_id=None,
        )

        assert ctx.correlation_id == _SENTINEL_UUID
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == _SENTINEL_UUID

    def test_mixed_none_and_real_uuids(self) -> None:
        """Test that None is converted while real UUIDs are preserved."""
        real_uuid = uuid4()
        ctx = ModelTracingContext(
            correlation_id=real_uuid,
            trace_id=None,
            span_id=None,
        )

        assert ctx.correlation_id == real_uuid
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == _SENTINEL_UUID


# ============================================================================
# has_* Properties Tests
# ============================================================================


@pytest.mark.unit
class TestHasProperties:
    """Tests for has_correlation_id, has_trace_id, has_span_id properties.

    These properties return True if the field is set (not sentinel),
    and False if the field is sentinel (not set).
    """

    def test_has_correlation_id_false_for_default(self) -> None:
        """Test that has_correlation_id is False for default construction."""
        ctx = ModelTracingContext()
        assert ctx.has_correlation_id is False

    def test_has_trace_id_false_for_default(self) -> None:
        """Test that has_trace_id is False for default construction."""
        ctx = ModelTracingContext()
        assert ctx.has_trace_id is False

    def test_has_span_id_false_for_default(self) -> None:
        """Test that has_span_id is False for default construction."""
        ctx = ModelTracingContext()
        assert ctx.has_span_id is False

    def test_has_correlation_id_false_for_none(self) -> None:
        """Test that has_correlation_id is False when None is passed."""
        ctx = ModelTracingContext(correlation_id=None)
        assert ctx.has_correlation_id is False

    def test_has_trace_id_false_for_none(self) -> None:
        """Test that has_trace_id is False when None is passed."""
        ctx = ModelTracingContext(trace_id=None)
        assert ctx.has_trace_id is False

    def test_has_span_id_false_for_none(self) -> None:
        """Test that has_span_id is False when None is passed."""
        ctx = ModelTracingContext(span_id=None)
        assert ctx.has_span_id is False

    def test_has_correlation_id_true_for_real_uuid(self) -> None:
        """Test that has_correlation_id is True for a real UUID."""
        ctx = ModelTracingContext(correlation_id=uuid4())
        assert ctx.has_correlation_id is True

    def test_has_trace_id_true_for_real_uuid(self) -> None:
        """Test that has_trace_id is True for a real UUID."""
        ctx = ModelTracingContext(trace_id=uuid4())
        assert ctx.has_trace_id is True

    def test_has_span_id_true_for_real_uuid(self) -> None:
        """Test that has_span_id is True for a real UUID."""
        ctx = ModelTracingContext(span_id=uuid4())
        assert ctx.has_span_id is True

    def test_mixed_has_properties(self) -> None:
        """Test that has_* properties work correctly with mixed values."""
        ctx = ModelTracingContext(
            correlation_id=uuid4(),
            trace_id=None,
            span_id=uuid4(),
        )

        assert ctx.has_correlation_id is True
        assert ctx.has_trace_id is False
        assert ctx.has_span_id is True

    def test_only_correlation_id_set(self) -> None:
        """Test has_* properties when only correlation_id is set."""
        ctx = ModelTracingContext(correlation_id=uuid4())

        assert ctx.has_correlation_id is True
        assert ctx.has_trace_id is False
        assert ctx.has_span_id is False

    def test_only_trace_id_set(self) -> None:
        """Test has_* properties when only trace_id is set."""
        ctx = ModelTracingContext(trace_id=uuid4())

        assert ctx.has_correlation_id is False
        assert ctx.has_trace_id is True
        assert ctx.has_span_id is False

    def test_only_span_id_set(self) -> None:
        """Test has_* properties when only span_id is set."""
        ctx = ModelTracingContext(span_id=uuid4())

        assert ctx.has_correlation_id is False
        assert ctx.has_trace_id is False
        assert ctx.has_span_id is True


# ============================================================================
# is_empty Property Tests
# ============================================================================


@pytest.mark.unit
class TestIsEmptyProperty:
    """Tests for is_empty property.

    is_empty returns True when all tracing fields are unset (all sentinel UUIDs).
    """

    def test_is_empty_true_for_default(self) -> None:
        """Test that is_empty is True for default construction."""
        ctx = ModelTracingContext()
        assert ctx.is_empty is True

    def test_is_empty_true_for_all_none(self) -> None:
        """Test that is_empty is True when all values are None."""
        ctx = ModelTracingContext(
            correlation_id=None,
            trace_id=None,
            span_id=None,
        )
        assert ctx.is_empty is True

    def test_is_empty_false_when_correlation_id_set(self) -> None:
        """Test that is_empty is False when correlation_id is set."""
        ctx = ModelTracingContext(correlation_id=uuid4())
        assert ctx.is_empty is False

    def test_is_empty_false_when_trace_id_set(self) -> None:
        """Test that is_empty is False when trace_id is set."""
        ctx = ModelTracingContext(trace_id=uuid4())
        assert ctx.is_empty is False

    def test_is_empty_false_when_span_id_set(self) -> None:
        """Test that is_empty is False when span_id is set."""
        ctx = ModelTracingContext(span_id=uuid4())
        assert ctx.is_empty is False

    def test_is_empty_false_when_all_fields_set(self) -> None:
        """Test that is_empty is False when all fields are set."""
        ctx = ModelTracingContext(
            correlation_id=uuid4(),
            trace_id=uuid4(),
            span_id=uuid4(),
        )
        assert ctx.is_empty is False

    def test_is_empty_false_when_one_field_set(self) -> None:
        """Test that is_empty is False when at least one field is set."""
        # Only correlation_id set
        ctx1 = ModelTracingContext(correlation_id=uuid4())
        assert ctx1.is_empty is False

        # Only trace_id set
        ctx2 = ModelTracingContext(trace_id=uuid4())
        assert ctx2.is_empty is False

        # Only span_id set
        ctx3 = ModelTracingContext(span_id=uuid4())
        assert ctx3.is_empty is False


# ============================================================================
# Factory Methods Tests
# ============================================================================


@pytest.mark.unit
class TestFactoryMethods:
    """Tests for ModelTracingContext factory methods.

    Factory methods include:
    - empty(): Create context with all sentinel UUIDs
    - from_uuids(): Create context from optional UUID values
    """

    # -------------------------------------------------------------------------
    # empty() Factory Method
    # -------------------------------------------------------------------------

    def test_empty_returns_context_with_all_sentinel(self) -> None:
        """Test that empty() returns context with all sentinel UUIDs."""
        ctx = ModelTracingContext.empty()

        assert ctx.correlation_id == _SENTINEL_UUID
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == _SENTINEL_UUID

    def test_empty_returns_empty_context(self) -> None:
        """Test that empty() returns a context where is_empty is True."""
        ctx = ModelTracingContext.empty()
        assert ctx.is_empty is True

    def test_empty_has_properties_all_false(self) -> None:
        """Test that empty() context has all has_* properties False."""
        ctx = ModelTracingContext.empty()

        assert ctx.has_correlation_id is False
        assert ctx.has_trace_id is False
        assert ctx.has_span_id is False

    def test_empty_equivalent_to_default_constructor(self) -> None:
        """Test that empty() is equivalent to default constructor."""
        ctx_empty = ModelTracingContext.empty()
        ctx_default = ModelTracingContext()

        assert ctx_empty.correlation_id == ctx_default.correlation_id
        assert ctx_empty.trace_id == ctx_default.trace_id
        assert ctx_empty.span_id == ctx_default.span_id

    # -------------------------------------------------------------------------
    # from_uuids() Factory Method
    # -------------------------------------------------------------------------

    def test_from_uuids_with_correlation_id_only(self) -> None:
        """Test from_uuids() with only correlation_id set."""
        cid = uuid4()
        ctx = ModelTracingContext.from_uuids(correlation_id=cid)

        assert ctx.correlation_id == cid
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == _SENTINEL_UUID
        assert ctx.has_correlation_id is True
        assert ctx.has_trace_id is False
        assert ctx.has_span_id is False

    def test_from_uuids_with_trace_id_only(self) -> None:
        """Test from_uuids() with only trace_id set."""
        tid = uuid4()
        ctx = ModelTracingContext.from_uuids(trace_id=tid)

        assert ctx.correlation_id == _SENTINEL_UUID
        assert ctx.trace_id == tid
        assert ctx.span_id == _SENTINEL_UUID
        assert ctx.has_correlation_id is False
        assert ctx.has_trace_id is True
        assert ctx.has_span_id is False

    def test_from_uuids_with_span_id_only(self) -> None:
        """Test from_uuids() with only span_id set."""
        sid = uuid4()
        ctx = ModelTracingContext.from_uuids(span_id=sid)

        assert ctx.correlation_id == _SENTINEL_UUID
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == sid
        assert ctx.has_correlation_id is False
        assert ctx.has_trace_id is False
        assert ctx.has_span_id is True

    def test_from_uuids_with_all_ids_set(self) -> None:
        """Test from_uuids() with all IDs set."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()
        ctx = ModelTracingContext.from_uuids(
            correlation_id=cid,
            trace_id=tid,
            span_id=sid,
        )

        assert ctx.correlation_id == cid
        assert ctx.trace_id == tid
        assert ctx.span_id == sid
        assert ctx.has_correlation_id is True
        assert ctx.has_trace_id is True
        assert ctx.has_span_id is True
        assert ctx.is_empty is False

    def test_from_uuids_with_all_none(self) -> None:
        """Test from_uuids() with all None values - same as empty()."""
        ctx = ModelTracingContext.from_uuids(None, None, None)

        assert ctx.correlation_id == _SENTINEL_UUID
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == _SENTINEL_UUID
        assert ctx.is_empty is True

    def test_from_uuids_with_no_arguments(self) -> None:
        """Test from_uuids() with no arguments - same as empty()."""
        ctx = ModelTracingContext.from_uuids()

        assert ctx.is_empty is True
        assert ctx.correlation_id == _SENTINEL_UUID

    def test_from_uuids_positional_arguments(self) -> None:
        """Test from_uuids() with positional arguments."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()
        ctx = ModelTracingContext.from_uuids(cid, tid, sid)

        assert ctx.correlation_id == cid
        assert ctx.trace_id == tid
        assert ctx.span_id == sid

    def test_from_uuids_mixed_none_and_real(self) -> None:
        """Test from_uuids() with mixed None and real UUIDs."""
        cid = uuid4()
        ctx = ModelTracingContext.from_uuids(
            correlation_id=cid,
            trace_id=None,
            span_id=None,
        )

        assert ctx.correlation_id == cid
        assert ctx.trace_id == _SENTINEL_UUID
        assert ctx.span_id == _SENTINEL_UUID
        assert ctx.is_empty is False


# ============================================================================
# to_dict() Method Tests
# ============================================================================


@pytest.mark.unit
class TestToDictMethod:
    """Tests for to_dict() method.

    to_dict() returns a dictionary containing only fields that are set,
    with UUID values converted to strings.
    """

    def test_to_dict_empty_context_returns_empty_dict(self) -> None:
        """Test that to_dict() returns empty dict for empty context."""
        ctx = ModelTracingContext()
        result = ctx.to_dict()

        assert result == {}
        assert isinstance(result, dict)

    def test_to_dict_only_correlation_id_set(self) -> None:
        """Test to_dict() when only correlation_id is set."""
        cid = uuid4()
        ctx = ModelTracingContext(correlation_id=cid)
        result = ctx.to_dict()

        assert "correlation_id" in result
        assert "trace_id" not in result
        assert "span_id" not in result
        assert len(result) == 1

    def test_to_dict_only_trace_id_set(self) -> None:
        """Test to_dict() when only trace_id is set."""
        tid = uuid4()
        ctx = ModelTracingContext(trace_id=tid)
        result = ctx.to_dict()

        assert "correlation_id" not in result
        assert "trace_id" in result
        assert "span_id" not in result
        assert len(result) == 1

    def test_to_dict_only_span_id_set(self) -> None:
        """Test to_dict() when only span_id is set."""
        sid = uuid4()
        ctx = ModelTracingContext(span_id=sid)
        result = ctx.to_dict()

        assert "correlation_id" not in result
        assert "trace_id" not in result
        assert "span_id" in result
        assert len(result) == 1

    def test_to_dict_all_fields_set(self) -> None:
        """Test to_dict() when all fields are set."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()
        ctx = ModelTracingContext(
            correlation_id=cid,
            trace_id=tid,
            span_id=sid,
        )
        result = ctx.to_dict()

        assert "correlation_id" in result
        assert "trace_id" in result
        assert "span_id" in result
        assert len(result) == 3

    def test_to_dict_values_are_strings(self) -> None:
        """Test that to_dict() values are strings, not UUIDs."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()
        ctx = ModelTracingContext(
            correlation_id=cid,
            trace_id=tid,
            span_id=sid,
        )
        result = ctx.to_dict()

        assert isinstance(result["correlation_id"], str)
        assert isinstance(result["trace_id"], str)
        assert isinstance(result["span_id"], str)

    def test_to_dict_values_match_uuid_string(self) -> None:
        """Test that to_dict() string values match UUID string representation."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()
        ctx = ModelTracingContext(
            correlation_id=cid,
            trace_id=tid,
            span_id=sid,
        )
        result = ctx.to_dict()

        assert result["correlation_id"] == str(cid)
        assert result["trace_id"] == str(tid)
        assert result["span_id"] == str(sid)

    def test_to_dict_mixed_fields(self) -> None:
        """Test to_dict() with mixed set/unset fields."""
        cid, sid = uuid4(), uuid4()
        ctx = ModelTracingContext(
            correlation_id=cid,
            trace_id=None,
            span_id=sid,
        )
        result = ctx.to_dict()

        assert "correlation_id" in result
        assert "trace_id" not in result
        assert "span_id" in result
        assert len(result) == 2

    def test_to_dict_from_empty_factory(self) -> None:
        """Test to_dict() on context created via empty() factory."""
        ctx = ModelTracingContext.empty()
        result = ctx.to_dict()

        assert result == {}


# ============================================================================
# String UUID Conversion Tests
# ============================================================================


@pytest.mark.unit
class TestStringUUIDConversion:
    """Tests for string UUID conversion via field validators.

    The model's field validators accept string UUIDs and convert them to UUID type.
    """

    def test_string_uuid_converted_for_correlation_id(self) -> None:
        """Test that string UUID is converted for correlation_id."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        ctx = ModelTracingContext(correlation_id=uuid_str)  # type: ignore[arg-type]

        assert isinstance(ctx.correlation_id, UUID)
        assert str(ctx.correlation_id) == uuid_str

    def test_string_uuid_converted_for_trace_id(self) -> None:
        """Test that string UUID is converted for trace_id."""
        uuid_str = "abcdef01-2345-6789-abcd-ef0123456789"
        ctx = ModelTracingContext(trace_id=uuid_str)  # type: ignore[arg-type]

        assert isinstance(ctx.trace_id, UUID)
        assert str(ctx.trace_id) == uuid_str

    def test_string_uuid_converted_for_span_id(self) -> None:
        """Test that string UUID is converted for span_id."""
        uuid_str = "fedcba98-7654-3210-fedc-ba9876543210"
        ctx = ModelTracingContext(span_id=uuid_str)  # type: ignore[arg-type]

        assert isinstance(ctx.span_id, UUID)
        assert str(ctx.span_id) == uuid_str

    def test_string_uuid_has_property_true(self) -> None:
        """Test that string UUID results in has_* property being True."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        ctx = ModelTracingContext(correlation_id=uuid_str)  # type: ignore[arg-type]

        assert ctx.has_correlation_id is True

    def test_uppercase_string_uuid(self) -> None:
        """Test that uppercase string UUID is accepted."""
        uuid_str = "12345678-1234-5678-1234-567812345678"
        ctx = ModelTracingContext(correlation_id=uuid_str.upper())  # type: ignore[arg-type]

        assert isinstance(ctx.correlation_id, UUID)
        # UUID normalizes to lowercase
        assert str(ctx.correlation_id) == uuid_str.lower()

    def test_invalid_string_uuid_raises_error(self) -> None:
        """Test that invalid string UUID raises validation error."""
        with pytest.raises((ValueError, ValidationError)):
            ModelTracingContext(correlation_id="not-a-valid-uuid")  # type: ignore[arg-type]

    def test_empty_string_uuid_raises_error(self) -> None:
        """Test that empty string raises validation error."""
        with pytest.raises((ValueError, ValidationError)):
            ModelTracingContext(correlation_id="")  # type: ignore[arg-type]


# ============================================================================
# Pydantic Model Behaviors Tests
# ============================================================================


@pytest.mark.unit
class TestPydanticModelBehaviors:
    """Tests for canonical Pydantic model behaviors of ModelTracingContext.

    These tests verify that the frozen Pydantic model correctly implements:
    - Immutability (frozen=True)
    - Serialization (model_dump)
    - Deserialization (model_validate)
    - Copying (model_copy)
    - Extra field rejection (extra='forbid')
    - Equality comparison
    - Hashability
    """

    def test_immutability_frozen_model(self) -> None:
        """Test that ModelTracingContext is immutable (frozen=True)."""
        ctx = ModelTracingContext(correlation_id=uuid4())

        with pytest.raises(ValidationError):
            ctx.correlation_id = uuid4()  # type: ignore[misc]

        with pytest.raises(ValidationError):
            ctx.trace_id = uuid4()  # type: ignore[misc]

        with pytest.raises(ValidationError):
            ctx.span_id = uuid4()  # type: ignore[misc]

    def test_model_dump_all_fields(self) -> None:
        """Test model_dump() returns all fields correctly."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()
        ctx = ModelTracingContext(
            correlation_id=cid,
            trace_id=tid,
            span_id=sid,
        )

        data = ctx.model_dump()

        assert data["correlation_id"] == cid
        assert data["trace_id"] == tid
        assert data["span_id"] == sid

    def test_model_dump_default_values(self) -> None:
        """Test model_dump() includes default sentinel values."""
        ctx = ModelTracingContext()
        data = ctx.model_dump()

        assert data["correlation_id"] == _SENTINEL_UUID
        assert data["trace_id"] == _SENTINEL_UUID
        assert data["span_id"] == _SENTINEL_UUID

    def test_model_validate_from_dict(self) -> None:
        """Test model_validate() creates model from dictionary."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()
        data = {
            "correlation_id": cid,
            "trace_id": tid,
            "span_id": sid,
        }

        ctx = ModelTracingContext.model_validate(data)

        assert ctx.correlation_id == cid
        assert ctx.trace_id == tid
        assert ctx.span_id == sid

    def test_model_validate_roundtrip(self) -> None:
        """Test model_dump() -> model_validate() roundtrip preserves data."""
        original = ModelTracingContext(
            correlation_id=uuid4(),
            trace_id=uuid4(),
            span_id=uuid4(),
        )

        data = original.model_dump()
        restored = ModelTracingContext.model_validate(data)

        assert restored.correlation_id == original.correlation_id
        assert restored.trace_id == original.trace_id
        assert restored.span_id == original.span_id

    def test_model_copy_creates_new_instance(self) -> None:
        """Test model_copy() creates a new independent instance."""
        original = ModelTracingContext(correlation_id=uuid4())
        new_cid = uuid4()

        copied = original.model_copy(update={"correlation_id": new_cid})

        assert copied.correlation_id == new_cid
        assert original.correlation_id != new_cid
        assert copied is not original

    def test_model_copy_deep(self) -> None:
        """Test model_copy(deep=True) creates deep copy."""
        original = ModelTracingContext(correlation_id=uuid4())
        copied = original.model_copy(deep=True)

        assert copied == original
        assert copied is not original

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelTracingContext(
                correlation_id=uuid4(),
                unexpected_field="should_fail",  # type: ignore[call-arg]
            )

        assert "unexpected_field" in str(exc_info.value)

    def test_equality_comparison_equal(self) -> None:
        """Test equality comparison between equal ModelTracingContext instances."""
        cid, tid, sid = uuid4(), uuid4(), uuid4()

        ctx1 = ModelTracingContext(
            correlation_id=cid,
            trace_id=tid,
            span_id=sid,
        )
        ctx2 = ModelTracingContext(
            correlation_id=cid,
            trace_id=tid,
            span_id=sid,
        )

        assert ctx1 == ctx2

    def test_equality_comparison_not_equal(self) -> None:
        """Test equality comparison between different ModelTracingContext instances."""
        ctx1 = ModelTracingContext(correlation_id=uuid4())
        ctx2 = ModelTracingContext(correlation_id=uuid4())

        assert ctx1 != ctx2

    def test_equality_comparison_different_types(self) -> None:
        """Test equality comparison with different types returns False."""
        ctx = ModelTracingContext(correlation_id=uuid4())

        assert ctx != "not a context"
        assert ctx != {"correlation_id": str(ctx.correlation_id)}

    def test_hashability(self) -> None:
        """Test that frozen model is hashable and can be used in sets/dicts."""
        cid = uuid4()

        ctx1 = ModelTracingContext(correlation_id=cid)
        ctx2 = ModelTracingContext(correlation_id=cid)
        ctx3 = ModelTracingContext(correlation_id=uuid4())

        # Should be hashable
        hash1 = hash(ctx1)
        hash2 = hash(ctx2)
        hash3 = hash(ctx3)

        # Equal objects should have same hash
        assert hash1 == hash2

        # Different objects likely have different hashes
        assert hash1 != hash3

        # Can be used in set
        context_set = {ctx1, ctx2, ctx3}
        assert len(context_set) == 2  # ctx1 and ctx2 are equal

        # Can be used as dict key
        context_dict = {ctx1: "first", ctx3: "second"}
        assert context_dict[ctx2] == "first"  # ctx2 equals ctx1

    def test_from_attributes_config(self) -> None:
        """Test from_attributes=True allows creation from objects with attributes."""

        class TracingData:
            """Simple class with matching attributes."""

            def __init__(self) -> None:
                self.correlation_id = uuid4()
                self.trace_id = uuid4()
                self.span_id = uuid4()

        source = TracingData()
        ctx = ModelTracingContext.model_validate(source)

        assert ctx.correlation_id == source.correlation_id
        assert ctx.trace_id == source.trace_id
        assert ctx.span_id == source.span_id


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================


@pytest.mark.unit
class TestEdgeCasesAndBoundaries:
    """Tests for edge cases and boundary conditions."""

    def test_sentinel_uuid_explicitly_passed(self) -> None:
        """Test that explicitly passing sentinel UUID behaves as not set."""
        ctx = ModelTracingContext(correlation_id=_SENTINEL_UUID)

        assert ctx.has_correlation_id is False
        assert ctx.is_empty is True

    def test_same_uuid_for_all_fields(self) -> None:
        """Test that same UUID can be used for all fields."""
        shared_uuid = uuid4()
        ctx = ModelTracingContext(
            correlation_id=shared_uuid,
            trace_id=shared_uuid,
            span_id=shared_uuid,
        )

        assert ctx.correlation_id == shared_uuid
        assert ctx.trace_id == shared_uuid
        assert ctx.span_id == shared_uuid
        assert ctx.has_correlation_id is True
        assert ctx.has_trace_id is True
        assert ctx.has_span_id is True

    def test_uuid_v1_accepted(self) -> None:
        """Test that UUID version 1 is accepted."""
        from uuid import uuid1

        cid = uuid1()
        ctx = ModelTracingContext(correlation_id=cid)

        assert ctx.correlation_id == cid
        assert ctx.has_correlation_id is True

    def test_to_dict_usable_for_logging(self) -> None:
        """Test that to_dict() output is usable for logging extra context."""
        cid = uuid4()
        ctx = ModelTracingContext(correlation_id=cid)

        # Simulate logging extra context usage
        log_extra = {"request_path": "/api/test", **ctx.to_dict()}

        assert "correlation_id" in log_extra
        assert "request_path" in log_extra
        assert log_extra["correlation_id"] == str(cid)

    def test_multiple_instances_independent(self) -> None:
        """Test that multiple instances are independent."""
        ctx1 = ModelTracingContext(correlation_id=uuid4())
        ctx2 = ModelTracingContext(correlation_id=uuid4())
        ctx3 = ModelTracingContext()

        assert ctx1.correlation_id != ctx2.correlation_id
        assert ctx3.is_empty is True
        assert ctx1.is_empty is False

    def test_repr_contains_uuids(self) -> None:
        """Test that repr() contains UUID information."""
        cid = uuid4()
        ctx = ModelTracingContext(correlation_id=cid)

        repr_str = repr(ctx)

        # Should contain the correlation_id string
        assert str(cid) in repr_str or "correlation_id" in repr_str
