# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelMaterializedDispatch.

This module tests the materialized dispatch message model that defines
the canonical runtime contract for all dispatched messages.

Test categories:
- Schema validation (required/optional fields)
- Aliasing behavior (double-underscore keys)
- Serialization/deserialization
- Extra fields rejection
- JSON safety (all values must be JSON-serializable)

.. versionadded:: 0.2.7
    Added as part of OMN-1518 - Architectural hardening of dispatch contract.

.. versionchanged:: 0.2.8
    Updated for strict JSON-safe contract:
    - debug_original_envelope → debug_trace (serialized snapshot)
    - payload must be JsonType (no arbitrary Python objects)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.dispatch.model_materialized_dispatch import (
    ModelMaterializedDispatch,
)


class TestModelMaterializedDispatchSchema:
    """Tests for schema validation."""

    def test_minimal_valid_envelope(self) -> None:
        """Envelope with only required fields is valid."""
        envelope = ModelMaterializedDispatch(payload={"key": "value"})
        assert envelope.payload == {"key": "value"}
        assert envelope.bindings == {}
        assert envelope.debug_trace is None

    def test_full_valid_envelope(self) -> None:
        """Envelope with all fields is valid."""
        debug_snapshot = {
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "trace_id": "660e8400-e29b-41d4-a716-446655440001",
            "topic": "dev.user.events.v1",
        }
        envelope = ModelMaterializedDispatch(
            payload={"user_id": "123"},
            bindings={"user_id": "123", "limit": 100},
            debug_trace=debug_snapshot,
        )
        assert envelope.payload == {"user_id": "123"}
        assert envelope.bindings == {"user_id": "123", "limit": 100}
        assert envelope.debug_trace == debug_snapshot

    def test_missing_payload_raises(self) -> None:
        """Missing payload field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelMaterializedDispatch()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("payload",) for e in errors)

    def test_extra_fields_rejected(self) -> None:
        """Extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelMaterializedDispatch(
                payload={"key": "value"},
                unexpected_field="should_fail",  # type: ignore[call-arg]
            )

        errors = exc_info.value.errors()
        assert any("extra" in str(e) for e in errors)

    def test_payload_accepts_nested_json(self) -> None:
        """Payload field accepts deeply nested JSON structures."""
        nested_payload = {
            "users": [
                {"id": 1, "profile": {"name": "Alice", "tags": ["admin"]}},
                {"id": 2, "profile": {"name": "Bob", "tags": ["user", "beta"]}},
            ],
            "metadata": {"version": "1.0", "count": 2},
        }
        envelope = ModelMaterializedDispatch(payload=nested_payload)
        assert envelope.payload == nested_payload

    def test_payload_accepts_json_primitives(self) -> None:
        """Payload field accepts JSON primitive types."""
        # String payload (wrapped in dict by serialization)
        envelope = ModelMaterializedDispatch(payload={"_raw": "string_value"})
        assert envelope.payload == {"_raw": "string_value"}

        # Number payload
        envelope = ModelMaterializedDispatch(payload={"count": 42})
        assert envelope.payload == {"count": 42}

        # Boolean payload
        envelope = ModelMaterializedDispatch(payload={"active": True})
        assert envelope.payload == {"active": True}

        # List payload
        envelope = ModelMaterializedDispatch(payload=[1, 2, 3])
        assert envelope.payload == [1, 2, 3]


class TestModelMaterializedDispatchAliasing:
    """Tests for double-underscore alias behavior.

    These tests verify that the model correctly handles aliasing between:
    - Python attributes: bindings, debug_trace
    - Dict keys: __bindings, __debug_trace
    """

    def test_model_dump_uses_aliases(self) -> None:
        """model_dump(by_alias=True) produces double-underscore keys."""
        envelope = ModelMaterializedDispatch(
            payload={"key": "value"},
            bindings={"param": "resolved"},
        )
        dumped = envelope.model_dump(by_alias=True)

        assert "__bindings" in dumped, "Should have __bindings key"
        assert "__debug_trace" in dumped, "Should have __debug_trace key"
        assert "bindings" not in dumped, "Should NOT have Python attribute name"
        assert dumped["__bindings"] == {"param": "resolved"}

    def test_model_validate_from_aliased_dict(self) -> None:
        """model_validate accepts dict with double-underscore keys."""
        raw_dict = {
            "payload": {"user_id": "123"},
            "__bindings": {"user_id": "123"},
            "__debug_trace": {"correlation_id": "test-id", "topic": "test.topic"},
        }
        envelope = ModelMaterializedDispatch.model_validate(raw_dict)

        assert envelope.payload == {"user_id": "123"}
        assert envelope.bindings == {"user_id": "123"}
        assert envelope.debug_trace == {
            "correlation_id": "test-id",
            "topic": "test.topic",
        }

    def test_model_validate_from_python_names(self) -> None:
        """model_validate accepts dict with Python attribute names."""
        raw_dict = {
            "payload": {"key": "value"},
            "bindings": {"param": "resolved"},
            "debug_trace": None,
        }
        envelope = ModelMaterializedDispatch.model_validate(raw_dict)

        assert envelope.bindings == {"param": "resolved"}

    def test_model_dump_without_alias(self) -> None:
        """model_dump() without by_alias uses Python attribute names."""
        envelope = ModelMaterializedDispatch(
            payload={"key": "value"},
            bindings={"param": "resolved"},
        )
        dumped = envelope.model_dump()

        # Without by_alias, uses Python attribute names
        assert "bindings" in dumped
        assert "debug_trace" in dumped


class TestModelMaterializedDispatchRepr:
    """Tests for repr/string representation."""

    def test_debug_trace_excluded_from_repr(self) -> None:
        """__debug_trace is excluded from repr (repr=False)."""
        large_trace = {
            "correlation_id": "test-id",
            "topic": "test.topic",
            "event_type": "LargeEvent" * 100,  # Large value
        }
        envelope = ModelMaterializedDispatch(
            payload={"small": "data"},
            debug_trace=large_trace,
        )

        repr_str = repr(envelope)

        # The large trace data should not appear in repr
        assert "LargeEvent" * 50 not in repr_str
        # payload should appear
        assert "small" in repr_str

    def test_str_representation(self) -> None:
        """String representation is readable."""
        envelope = ModelMaterializedDispatch(
            payload={"user": "test"},
            bindings={"user": "test"},
        )
        str_repr = str(envelope)

        assert "payload" in str_repr
        assert "bindings" in str_repr


class TestModelMaterializedDispatchImmutability:
    """Tests for frozen model behavior."""

    def test_model_is_frozen(self) -> None:
        """Model instances are immutable (frozen=True)."""
        envelope = ModelMaterializedDispatch(payload={"key": "value"})

        with pytest.raises(ValidationError):
            envelope.payload = {"new": "value"}  # type: ignore[misc]

    def test_bindings_cannot_be_modified(self) -> None:
        """Bindings field cannot be reassigned."""
        envelope = ModelMaterializedDispatch(
            payload={"key": "value"},
            bindings={"param": "value"},
        )

        with pytest.raises(ValidationError):
            envelope.bindings = {"new": "bindings"}  # type: ignore[misc]


class TestModelMaterializedDispatchRoundTrip:
    """Tests for serialization round-trip behavior."""

    def test_json_round_trip(self) -> None:
        """Model survives JSON serialization round-trip."""
        envelope = ModelMaterializedDispatch(
            payload={"user_id": "123", "count": 42, "active": True},
            bindings={"user_id": "123", "limit": 100},
        )

        # Serialize to JSON
        json_str = envelope.model_dump_json(by_alias=True)

        # Deserialize back
        restored = ModelMaterializedDispatch.model_validate_json(json_str)

        assert restored.payload == envelope.payload
        assert restored.bindings == envelope.bindings

    def test_dict_round_trip_with_alias(self) -> None:
        """Model survives dict round-trip with aliasing."""
        original = ModelMaterializedDispatch(
            payload={"key": "value"},
            bindings={"param": "resolved"},
        )

        # Convert to dict with aliases
        as_dict = original.model_dump(by_alias=True)

        # Restore from dict
        restored = ModelMaterializedDispatch.model_validate(as_dict)

        assert restored.payload == original.payload
        assert restored.bindings == original.bindings

    def test_debug_trace_survives_round_trip(self) -> None:
        """Debug trace snapshot survives serialization round-trip."""
        debug_snapshot = {
            "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
            "trace_id": "660e8400-e29b-41d4-a716-446655440001",
            "topic": "dev.user.events.v1",
            "timestamp": "2025-01-27T12:00:00Z",
        }
        original = ModelMaterializedDispatch(
            payload={"event": "data"},
            debug_trace=debug_snapshot,
        )

        # JSON round-trip
        json_str = original.model_dump_json(by_alias=True)
        restored = ModelMaterializedDispatch.model_validate_json(json_str)

        assert restored.debug_trace == debug_snapshot


class TestModelMaterializedDispatchJsonSafety:
    """Tests verifying JSON-safe contract (no arbitrary Python objects)."""

    def test_payload_is_json_serializable(self) -> None:
        """Payload must be JSON-serializable."""
        envelope = ModelMaterializedDispatch(
            payload={"nested": {"deeply": {"value": 42}}},
            bindings={"key": "value"},
        )

        # Should serialize without error
        json_str = envelope.model_dump_json()
        assert '"nested"' in json_str
        assert '"deeply"' in json_str
        assert "42" in json_str

    def test_bindings_are_json_serializable(self) -> None:
        """Bindings must be JSON-serializable."""
        envelope = ModelMaterializedDispatch(
            payload={"data": "test"},
            bindings={
                "string": "value",
                "number": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
                "nested": {"key": "value"},
            },
        )

        # Should serialize without error
        json_str = envelope.model_dump_json()
        assert '"string"' in json_str
        assert "42" in json_str
        assert "3.14" in json_str

    def test_debug_trace_is_string_dict(self) -> None:
        """Debug trace must be a dict of strings (or None)."""
        trace = {
            "correlation_id": "uuid-string",
            "trace_id": "another-uuid",
            "topic": "topic.name",
            "timestamp": None,  # None is allowed
        }
        envelope = ModelMaterializedDispatch(
            payload={"data": "test"},
            debug_trace=trace,
        )

        assert envelope.debug_trace == trace
        # All values should be strings or None
        assert all(
            isinstance(v, str) or v is None
            for v in envelope.debug_trace.values()  # type: ignore[union-attr]
        )
