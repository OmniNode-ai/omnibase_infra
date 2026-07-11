# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerPrStateProjection (OMN-14375).

Pure-logic tests for the COMPUTE handler that projects GitHub PR status
events into ModelPayloadPrStateUpsert intents. No Kafka or Postgres needed.

Test coverage:
    - Round-trips repo, pr_number, triage_state, title, correlation_id
    - Raises RuntimeHostError when repo/pr_number is missing (identity is
      required -- unlike build_loop's audit-semantics fallback)
    - Unwraps ModelEventEnvelope-style wrappers when payload is nested
    - Raises RuntimeHostError on unparseable JSON and non-object root
    - Emits ModelIntent with intent_type='pr_state.upsert' and a postgres:// target
    - handle() drives the real dispatch-shaped entry point (ModelHandlerOutput)
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
from omnibase_infra.nodes.node_pr_state_projection_compute import (
    HandlerPrStateProjection,
    ModelPayloadPrStateUpsert,
)

pytestmark = [pytest.mark.unit]


def _make_message(value: bytes) -> ModelEventMessage:
    return ModelEventMessage(
        topic="onex.evt.github.pr-status.v1",
        partition=0,
        offset="0",
        key=None,
        value=value,
        headers=ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="test-pr-poller",
            event_type="github.pr-status",
        ),
    )


def _from_dict(body: Mapping[str, Any]) -> ModelEventMessage:
    return _make_message(json.dumps(dict(body)).encode("utf-8"))


@pytest.fixture
def handler() -> HandlerPrStateProjection:
    return HandlerPrStateProjection(container=MagicMock())


def test_project_extracts_canonical_fields(
    handler: HandlerPrStateProjection,
) -> None:
    ts = datetime.now(UTC).isoformat()
    cid = str(uuid4())
    body = {
        "repo": "OmniNode-ai/omnibase_infra",
        "pr_number": 2260,
        "triage_state": "ready_to_merge",
        "title": "fix(OMN-14375): pr_state projection",
        "as_of": ts,
        "correlation_id": cid,
    }

    intent = handler.project(_from_dict(body))

    assert intent.intent_type == "pr_state.upsert"
    assert intent.target == "postgres://pr_state/OmniNode-ai/omnibase_infra:2260"
    payload = intent.payload
    assert isinstance(payload, ModelPayloadPrStateUpsert)
    assert payload.repo == "OmniNode-ai/omnibase_infra"
    assert payload.pr_number == 2260
    assert payload.triage_state == "ready_to_merge"
    assert payload.title == "fix(OMN-14375): pr_state projection"
    assert payload.as_of == datetime.fromisoformat(ts)
    assert payload.correlation_id == UUID(cid)
    # Reserved columns are None until a richer producer lands.
    assert payload.ci_status is None
    assert payload.review_decision is None


def test_project_unwraps_envelope_payload(
    handler: HandlerPrStateProjection,
) -> None:
    inner = {
        "repo": "OmniNode-ai/omniclaude",
        "pr_number": 42,
        "triage_state": "needs_review",
    }
    envelope_shaped = {"envelope_id": str(uuid4()), "payload": inner}

    intent = handler.project(_from_dict(envelope_shaped))

    assert isinstance(intent.payload, ModelPayloadPrStateUpsert)
    assert intent.payload.repo == "OmniNode-ai/omniclaude"
    assert intent.payload.pr_number == 42


def test_project_falls_back_when_triage_state_missing(
    handler: HandlerPrStateProjection,
) -> None:
    intent = handler.project(
        _from_dict({"repo": "OmniNode-ai/omnimarket", "pr_number": 1})
    )

    payload = intent.payload
    assert isinstance(payload, ModelPayloadPrStateUpsert)
    assert payload.triage_state == "needs_review"
    assert payload.title == ""
    # as_of falls back to "now"; we don't pin an exact value.
    assert payload.as_of.tzinfo is not None


def test_project_raises_when_repo_missing(
    handler: HandlerPrStateProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_from_dict({"pr_number": 1, "triage_state": "draft"}))


def test_project_raises_when_pr_number_missing(
    handler: HandlerPrStateProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(
            _from_dict({"repo": "OmniNode-ai/omnimarket", "triage_state": "draft"})
        )


def test_project_raises_when_value_is_not_json(
    handler: HandlerPrStateProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b"not-json"))


def test_project_raises_when_value_is_not_object(
    handler: HandlerPrStateProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b'["array", "not", "object"]'))


def test_project_raises_when_value_is_empty_bytes(
    handler: HandlerPrStateProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b""))


@pytest.mark.asyncio
async def test_handle_drives_dispatch_shaped_entry_point(
    handler: HandlerPrStateProjection,
) -> None:
    """handle() is the auto-wiring entry point the real dispatch path calls."""
    body = {
        "repo": "OmniNode-ai/omnibase_infra",
        "pr_number": 99,
        "triage_state": "approved_pending_ci",
    }
    message = _from_dict(body)

    output = await handler.handle(message)

    assert output.result is not None
    assert output.result.intent_type == "pr_state.upsert"
    assert output.result.payload.repo == "OmniNode-ai/omnibase_infra"
    assert output.result.payload.pr_number == 99
