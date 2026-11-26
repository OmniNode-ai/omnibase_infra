# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for the introspection wire shape with runtime_id (OMN-24529).

Verifies, through the byte-level JSON path a bus consumer uses, that an event
written by an older producer (no runtime_id key) is still accepted, that a
new-shape event round-trips, and that a self-parented event is refused.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.enums import EnumNodeKind
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent

pytestmark = pytest.mark.integration


def _wire_payload(node_id: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "node_type": "effect",
        "node_version": {"major": 1, "minor": 0, "patch": 0},
        "endpoints": {"health": "http://localhost:8080/health"},
        "reason": "startup",
        "correlation_id": str(uuid4()),
        "timestamp": "2025-01-15T12:00:00+00:00",
    }


class TestIntrospectionRuntimeIdWire:
    def test_old_shape_bytes_are_accepted(self) -> None:
        raw = json.dumps(_wire_payload(str(uuid4()))).encode()
        event = ModelNodeIntrospectionEvent.model_validate_json(raw)
        assert event.runtime_id is None
        assert event.node_type == EnumNodeKind.EFFECT

    def test_new_shape_bytes_round_trip(self) -> None:
        payload = _wire_payload(str(uuid4()))
        runtime_id = uuid4()
        payload["runtime_id"] = str(runtime_id)
        event = ModelNodeIntrospectionEvent.model_validate_json(
            json.dumps(payload).encode()
        )
        assert event.runtime_id == runtime_id
        again = ModelNodeIntrospectionEvent.model_validate_json(event.model_dump_json())
        assert again == event

    def test_self_parent_bytes_are_refused(self) -> None:
        node_id = str(uuid4())
        payload = _wire_payload(node_id)
        payload["runtime_id"] = node_id
        with pytest.raises(ValidationError, match="runtime_id must differ"):
            ModelNodeIntrospectionEvent.model_validate_json(
                json.dumps(payload).encode()
            )
