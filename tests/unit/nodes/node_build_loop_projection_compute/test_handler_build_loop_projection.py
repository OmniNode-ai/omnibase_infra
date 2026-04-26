# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerBuildLoopProjection (OMN-9774).

Pure-logic tests for the COMPUTE handler that projects build_loop terminal
events into ModelPayloadBuildLoopAppend intents. No Kafka or Postgres needed.

Test coverage:
    - Round-trips run_id, workflow_name, event_type, terminal_event_at, payload
    - Falls back to sensible defaults when fields are missing (audit semantics)
    - Unwraps ModelEventEnvelope-style wrappers when payload is nested
    - Raises RuntimeHostError on unparseable JSON and non-object root
    - Emits ModelIntent with intent_type='build_loop.append' and a postgres:// target
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_build_loop_projection_compute import (
    HandlerBuildLoopProjection,
    ModelPayloadBuildLoopAppend,
)

pytestmark = [pytest.mark.unit]


def _make_message(value: bytes) -> ModelEventMessage:
    return ModelEventMessage(
        topic="onex.evt.omnimarket.build-loop-orchestrator-completed.v1",
        partition=0,
        offset="0",
        key=None,
        value=value,
        headers=ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="test-build-loop",
            event_type="build-loop-orchestrator-completed",
        ),
    )


def _from_dict(body: Mapping[str, Any]) -> ModelEventMessage:
    return _make_message(json.dumps(dict(body)).encode("utf-8"))


@pytest.fixture
def handler() -> HandlerBuildLoopProjection:
    return HandlerBuildLoopProjection(container=MagicMock())


def test_project_extracts_canonical_fields(
    handler: HandlerBuildLoopProjection,
) -> None:
    run_id = "build-loop-123"
    ts = datetime.now(UTC).isoformat()
    cid = str(uuid4())
    body = {
        "run_id": run_id,
        "workflow_name": "build_loop",
        "event_type": "build-loop-orchestrator-completed",
        "terminal_event_at": ts,
        "correlation_id": cid,
        "outcome": "success",
    }

    intent = handler.project(_from_dict(body))

    assert intent.intent_type == "build_loop.append"
    assert intent.target.startswith("postgres://build_loop_runs/")
    payload = intent.payload
    assert isinstance(payload, ModelPayloadBuildLoopAppend)
    assert payload.run_id == run_id
    assert payload.workflow_name == "build_loop"
    assert payload.event_type == "build-loop-orchestrator-completed"
    assert payload.terminal_event_at == datetime.fromisoformat(ts)
    assert payload.payload["outcome"] == "success"
    assert payload.correlation_id == UUID(cid)


def test_project_unwraps_envelope_payload(
    handler: HandlerBuildLoopProjection,
) -> None:
    inner = {"run_id": "r-1", "workflow_name": "build_loop"}
    envelope_shaped = {"envelope_id": str(uuid4()), "payload": inner}

    intent = handler.project(_from_dict(envelope_shaped))

    assert isinstance(intent.payload, ModelPayloadBuildLoopAppend)
    assert intent.payload.run_id == "r-1"
    # The full payload preserved is the unwrapped inner body, not the envelope.
    assert intent.payload.payload == inner


def test_project_falls_back_when_fields_missing(
    handler: HandlerBuildLoopProjection,
) -> None:
    intent = handler.project(_from_dict({"unrelated": True}))

    payload = intent.payload
    assert isinstance(payload, ModelPayloadBuildLoopAppend)
    assert payload.run_id == "unknown"
    assert payload.workflow_name == "build_loop"
    assert payload.event_type == "build-loop-orchestrator-completed"
    # terminal_event_at falls back to "now"; we don't pin an exact value.
    assert payload.terminal_event_at.tzinfo is not None
    assert payload.payload == {"unrelated": True}


def test_project_raises_when_value_is_not_json(
    handler: HandlerBuildLoopProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b"not-json"))


def test_project_raises_when_value_is_not_object(
    handler: HandlerBuildLoopProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b'["array", "not", "object"]'))


def test_project_raises_when_value_is_empty_bytes(
    handler: HandlerBuildLoopProjection,
) -> None:
    """Empty bytes raise via JSON-decode failure (no None branch needed since
    ModelEventMessage.value is non-nullable per schema)."""
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b""))
