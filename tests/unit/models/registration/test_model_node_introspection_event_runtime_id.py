# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelNodeIntrospectionEvent.runtime_id (OMN-24529).

Tests validate:
- Old-shape JSON (no runtime_id key) parses with runtime_id None and
  round-trips through model_dump(mode="json") to an equal event (golden)
- A runtime_id distinct from node_id parses and round-trips through JSON
- runtime_id equal to node_id is refused, with the guard's message in the
  pydantic error chain
- Unknown extra fields are still refused (extra="forbid")
- Non-UUID runtime_id strings are refused
- Control: with the validator overridden to a no-op, a self-parented event
  is accepted (the control the self-parent refusal is checked against)
- The field name is pinned via model_fields
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError, model_validator

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent

pytestmark = [pytest.mark.unit]

# Fixed test timestamp for deterministic testing (time injection pattern)
TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

# Wire payload as an older producer wrote it: every key it knew, and no
# runtime_id key (the field did not exist before OMN-24529).
OLD_SHAPE_PAYLOAD: dict[str, object] = {
    "node_id": "11111111-1111-4111-8111-111111111111",
    "node_type": "effect",
    "node_version": {"major": 1, "minor": 4, "patch": 2},
    "endpoints": {"health": "http://localhost:8080/health"},
    "reason": "startup",
    "correlation_id": "22222222-2222-4222-8222-222222222222",
    "timestamp": "2025-01-15T12:00:00+00:00",
    "network_id": "network-001",
    "epoch": 3,
}


class TestModelNodeIntrospectionEventRuntimeId:
    """Tests for the runtime_id typed parent (OMN-24529)."""

    def test_old_shape_payload_parses_without_runtime_id(self) -> None:
        """Old-shape JSON with no runtime_id key parses; runtime_id is None."""
        event = ModelNodeIntrospectionEvent.model_validate(OLD_SHAPE_PAYLOAD)
        assert event.runtime_id is None
        assert event.node_id == UUID("11111111-1111-4111-8111-111111111111")
        assert str(event.node_version) == "1.4.2"
        assert event.epoch == 3

    def test_old_shape_payload_json_roundtrip_golden(self) -> None:
        """model_dump(mode='json') of a parsed old event round-trips (golden)."""
        event = ModelNodeIntrospectionEvent.model_validate(OLD_SHAPE_PAYLOAD)
        dumped = event.model_dump(mode="json")
        assert dumped["runtime_id"] is None
        restored = ModelNodeIntrospectionEvent.model_validate(dumped)
        assert restored == event
        assert restored.runtime_id is None

    def test_runtime_id_distinct_from_node_id_parses_and_roundtrips(self) -> None:
        """A runtime_id different from node_id parses and survives JSON."""
        node_id = uuid4()
        runtime_id = uuid4()
        assert runtime_id != node_id
        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
            runtime_id=runtime_id,
        )
        assert event.runtime_id == runtime_id
        restored = ModelNodeIntrospectionEvent.model_validate_json(
            event.model_dump_json()
        )
        assert restored == event
        assert restored.runtime_id == runtime_id

    def test_runtime_id_equal_to_node_id_raises(self) -> None:
        """runtime_id == node_id is refused via the pydantic error chain."""
        node_id = uuid4()
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=node_id,
                node_type=EnumNodeKind.EFFECT,
                correlation_id=uuid4(),
                timestamp=TEST_TIMESTAMP,
                runtime_id=node_id,
            )
        messages = [error["msg"] for error in exc_info.value.errors()]
        assert any(
            "runtime_id must differ from node_id" in message for message in messages
        )

    def test_unknown_extra_field_is_refused(self) -> None:
        """An unknown extra field (runtime_ident) is still refused."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent.model_validate(
                {
                    **OLD_SHAPE_PAYLOAD,
                    "runtime_ident": "11111111-1111-4111-8111-111111111111",
                }
            )
        assert any(
            error["loc"] == ("runtime_ident",) for error in exc_info.value.errors()
        )

    def test_runtime_id_non_uuid_string_is_refused(self) -> None:
        """A runtime_id that is not a UUID string is refused."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent.model_validate(
                {
                    **OLD_SHAPE_PAYLOAD,
                    "runtime_id": "not-a-uuid",
                }
            )
        assert any(error["loc"] == ("runtime_id",) for error in exc_info.value.errors())

    def test_control_without_the_validator_a_self_parent_is_accepted(self) -> None:
        """Control: with the guard overridden to a no-op, self-parent parses.

        This documents what test_runtime_id_equal_to_node_id_raises fails
        against: acceptance here is only possible because the OMN-24529
        model_validator is the sole guard.
        """

        class _UnguardedEvent(ModelNodeIntrospectionEvent):
            """Control subclass with the self-parent guard disabled."""

            @model_validator(mode="after")
            def validate_runtime_id_is_not_self(self) -> _UnguardedEvent:
                return self

        node_id = uuid4()
        event = _UnguardedEvent(
            node_id=node_id,
            node_type=EnumNodeKind.EFFECT,
            correlation_id=uuid4(),
            timestamp=TEST_TIMESTAMP,
            runtime_id=node_id,
        )
        assert event.runtime_id == node_id

    def test_runtime_id_field_name_is_pinned(self) -> None:
        """The field is pinned: present, default None, annotation UUID | None."""
        field = ModelNodeIntrospectionEvent.model_fields["runtime_id"]
        assert field.default is None
        assert field.annotation == UUID | None
