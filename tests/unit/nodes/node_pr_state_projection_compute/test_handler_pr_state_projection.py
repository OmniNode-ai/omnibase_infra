# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerPrStateProjection (OMN-14375, OMN-14394).

Pure-logic tests for the COMPUTE handler that projects GitHub PR status
events into ModelPayloadPrStateUpsert intents. No Kafka or Postgres needed.

Test coverage:
    - Round-trips repo, pr_number, triage_state, title, is_draft, correlation_id
    - Raises RuntimeHostError when repo/pr_number is missing (identity is
      required -- unlike build_loop's audit-semantics fallback)
    - Unwraps ModelEventEnvelope-style wrappers when payload is nested
    - Raises RuntimeHostError on unparseable JSON and non-object root
    - Emits ModelIntent with intent_type='pr_state.upsert' and a postgres:// target
    - handle() drives the real dispatch-shaped entry point (ModelHandlerOutput)
    - handle() also accepts a raw DICT-shaped envelope (not just a typed
      ModelEventMessage) -- the actual shape MessageDispatchEngine delivers
      for a no-event_model/operation_match dispatch path (OMN-14139 lesson;
      OMN-14394 closes the test gap for this handler's handle()).
    - is_draft always resolves to a concrete bool (never None), matching
      omnimarket's ModelOpenPrSummary.is_draft field-for-field (OMN-14394).
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
    # is_draft was absent from `body`; it still resolves to a concrete bool.
    assert payload.is_draft is False
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


def test_project_extracts_is_draft_true(
    handler: HandlerPrStateProjection,
) -> None:
    """OMN-14394: is_draft round-trips from the poller event body."""
    intent = handler.project(
        _from_dict(
            {
                "repo": "OmniNode-ai/omnimarket",
                "pr_number": 5,
                "triage_state": "draft",
                "is_draft": True,
            }
        )
    )

    payload = intent.payload
    assert isinstance(payload, ModelPayloadPrStateUpsert)
    assert payload.is_draft is True


def test_project_defaults_is_draft_false_when_missing(
    handler: HandlerPrStateProjection,
) -> None:
    """OMN-14394: unlike ci_status/review_decision (None = "not yet
    populated"), is_draft always resolves to a concrete bool -- it mirrors
    ModelOpenPrSummary.is_draft (omnimarket reader), a non-nullable field."""
    intent = handler.project(
        _from_dict({"repo": "OmniNode-ai/omnimarket", "pr_number": 6})
    )

    payload = intent.payload
    assert isinstance(payload, ModelPayloadPrStateUpsert)
    assert payload.is_draft is False


def test_is_draft_field_type_matches_reader_contract() -> None:
    """OMN-14208 seam discipline: ModelPayloadPrStateUpsert.is_draft must stay
    a non-nullable bool defaulting to False, matching omnimarket's
    ModelOpenPrSummary.is_draft (node_github_repo_gateway_effect)
    field-for-field. The reader lives in a separate repo (omnimarket), so a
    live cross-repo runtime round-trip isn't possible from here -- this pins
    the producer's half of the contract so a type change on this side (e.g.
    widening to bool | None, which the reader's required bool would reject)
    fails loudly in THIS repo's suite rather than silently at the seam."""
    field = ModelPayloadPrStateUpsert.model_fields["is_draft"]
    assert field.annotation is bool
    assert field.default is False


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


@pytest.mark.asyncio
async def test_handle_accepts_dict_shaped_envelope(
    handler: HandlerPrStateProjection,
) -> None:
    """OMN-14394: MessageDispatchEngine delivers a DICT envelope (not an
    already-typed ModelEventMessage) for a no-event_model/operation_match
    dispatch path -- the same input shape that caused the OMN-14139
    production bug on the ledger pipeline. `_coerce_event_message` already
    handles this (both a `getattr(raw, "payload", raw)` branch and an
    `isinstance(raw, dict)` branch), but until this test, no test exercised
    the dict branch for THIS handler -- only the typed-message path was
    covered (test_handle_drives_dispatch_shaped_entry_point above).

    This also exercises the full producer-side seam for is_draft: dict
    envelope -> _coerce_event_message -> project() -> _extract_payload ->
    ModelPayloadPrStateUpsert.is_draft.
    """
    body = {
        "repo": "OmniNode-ai/omnibase_infra",
        "pr_number": 2262,
        "triage_state": "draft",
        "title": "feat(OMN-14375): pr_state projection",
        "is_draft": True,
    }
    raw_event_message: dict[str, Any] = {
        "topic": "onex.evt.github.pr-status.v1",
        "partition": 0,
        "offset": "0",
        "key": None,
        "value": json.dumps(body).encode("utf-8"),
        "headers": {
            "timestamp": datetime.now(UTC),
            "source": "test-pr-poller",
            "event_type": "github.pr-status",
        },
    }
    dict_envelope: dict[str, Any] = {"payload": raw_event_message}

    output = await handler.handle(dict_envelope)

    assert output.result is not None
    assert output.result.intent_type == "pr_state.upsert"
    payload = output.result.payload
    assert isinstance(payload, ModelPayloadPrStateUpsert)
    assert payload.repo == "OmniNode-ai/omnibase_infra"
    assert payload.pr_number == 2262
    assert payload.is_draft is True
