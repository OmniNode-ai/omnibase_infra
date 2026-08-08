# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerGatewayLinkHealthProjection (OMN-15570, G3).

Pure-logic tests for the COMPUTE handler that projects gateway heartbeat
events into ModelPayloadGatewayLinkHealthUpsert intents. No Kafka or
Postgres needed. Mirrors test_handler_pr_state_projection.py.

Test coverage:
    - Round-trips tenant_id, principal_id, local_transport_flavor,
      last_seen_at from a ModelGatewayHeartbeat-shaped body
    - lag_messages/lag_seconds default to None when absent (today's real
      producer shape) and round-trip when present (forward-compat)
    - Raises RuntimeHostError when tenant_id/principal_id/
      local_transport_flavor is missing
    - Unwraps ModelEventEnvelope-style wrappers when payload is nested
    - Raises RuntimeHostError on unparseable JSON and non-object root
    - Emits ModelIntent with intent_type='gateway_link_health.upsert' and a
      postgres:// target
    - handle() drives the real dispatch-shaped entry point (ModelHandlerOutput)
    - handle() also accepts a raw DICT-shaped envelope (OMN-14139 lesson)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.node_gateway_link_health_projection_compute import (
    HandlerGatewayLinkHealthProjection,
    ModelPayloadGatewayLinkHealthUpsert,
)

pytestmark = [pytest.mark.unit]


def _make_message(value: bytes) -> ModelEventMessage:
    return ModelEventMessage(
        topic="onex.evt.omnibase-infra.gateway-heartbeat.v1",
        partition=0,
        offset="0",
        key=None,
        value=value,
        headers=ModelEventHeaders(
            timestamp=datetime.now(UTC),
            source="test-gateway-forwarder",
            event_type="omnibase-infra.gateway-heartbeat",
        ),
    )


def _from_dict(body: Mapping[str, Any]) -> ModelEventMessage:
    return _make_message(json.dumps(dict(body)).encode("utf-8"))


@pytest.fixture
def handler() -> HandlerGatewayLinkHealthProjection:
    return HandlerGatewayLinkHealthProjection(container=MagicMock())


def test_project_extracts_canonical_fields(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    ts = datetime.now(UTC).isoformat()
    body = {
        "tenant_id": "beta-gateway-canary-79afa7263852",
        "principal_id": "t-abc123",
        "status": "active",
        "emitted_at": ts,
        "local_transport_flavor": "containerized",
    }

    intent = handler.project(_from_dict(body))

    assert intent.intent_type == "gateway_link_health.upsert"
    assert (
        intent.target
        == "postgres://gateway_link_health/beta-gateway-canary-79afa7263852"
    )
    payload = intent.payload
    assert isinstance(payload, ModelPayloadGatewayLinkHealthUpsert)
    assert payload.tenant_id == "beta-gateway-canary-79afa7263852"
    assert payload.principal_id == "t-abc123"
    assert payload.local_transport_flavor == "containerized"
    assert payload.last_seen_at == datetime.fromisoformat(ts)
    # Today's real ModelGatewayHeartbeat producer never supplies lag data.
    assert payload.lag_messages is None
    assert payload.lag_seconds is None


def test_project_round_trips_lag_fields_when_present(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    """Forward-compat: once G1/G2 land a lag-carrying producer, this handler
    already round-trips the fields without a code change."""
    body = {
        "tenant_id": "beta-gateway-canary-79afa7263852",
        "principal_id": "t-abc123",
        "local_transport_flavor": "lightweight",
        "lag_messages": 12,
        "lag_seconds": 3.5,
    }

    intent = handler.project(_from_dict(body))

    payload = intent.payload
    assert isinstance(payload, ModelPayloadGatewayLinkHealthUpsert)
    assert payload.lag_messages == 12
    assert payload.lag_seconds == 3.5


def test_project_unwraps_envelope_payload(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    inner = {
        "tenant_id": "tenant-a",
        "principal_id": "t-a",
        "local_transport_flavor": "containerized",
    }
    envelope_shaped = {"envelope_id": str(uuid4()), "payload": inner}

    intent = handler.project(_from_dict(envelope_shaped))

    assert isinstance(intent.payload, ModelPayloadGatewayLinkHealthUpsert)
    assert intent.payload.tenant_id == "tenant-a"


def test_project_falls_back_to_now_when_emitted_at_missing(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    intent = handler.project(
        _from_dict(
            {
                "tenant_id": "tenant-b",
                "principal_id": "t-b",
                "local_transport_flavor": "containerized",
            }
        )
    )

    payload = intent.payload
    assert isinstance(payload, ModelPayloadGatewayLinkHealthUpsert)
    assert payload.last_seen_at.tzinfo is not None


def test_project_raises_when_tenant_id_missing(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(
            _from_dict(
                {"principal_id": "t-a", "local_transport_flavor": "containerized"}
            )
        )


def test_project_raises_when_principal_id_missing(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(
            _from_dict(
                {"tenant_id": "tenant-a", "local_transport_flavor": "containerized"}
            )
        )


def test_project_raises_when_local_transport_flavor_missing(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_from_dict({"tenant_id": "tenant-a", "principal_id": "t-a"}))


def test_project_raises_when_value_is_not_json(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b"not-json"))


def test_project_raises_when_value_is_not_object(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b'["array", "not", "object"]'))


def test_project_raises_when_value_is_empty_bytes(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    with pytest.raises(RuntimeHostError):
        handler.project(_make_message(b""))


@pytest.mark.asyncio
async def test_handle_drives_dispatch_shaped_entry_point(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    """handle() is the auto-wiring entry point the real dispatch path calls."""
    body = {
        "tenant_id": "beta-gateway-canary-79afa7263852",
        "principal_id": "t-abc123",
        "local_transport_flavor": "containerized",
    }
    message = _from_dict(body)

    output = await handler.handle(message)

    assert output.result is not None
    assert output.result.intent_type == "gateway_link_health.upsert"
    assert output.result.payload.tenant_id == "beta-gateway-canary-79afa7263852"


@pytest.mark.asyncio
async def test_handle_accepts_dict_shaped_envelope(
    handler: HandlerGatewayLinkHealthProjection,
) -> None:
    """OMN-14139 lesson: MessageDispatchEngine delivers a DICT envelope (not
    an already-typed ModelEventMessage) for a no-event_model/operation_match
    dispatch path."""
    body = {
        "tenant_id": "beta-gateway-canary-79afa7263852",
        "principal_id": "t-abc123",
        "local_transport_flavor": "lightweight",
    }
    raw_event_message: dict[str, Any] = {
        "topic": "onex.evt.omnibase-infra.gateway-heartbeat.v1",
        "partition": 0,
        "offset": "0",
        "key": None,
        "value": json.dumps(body).encode("utf-8"),
        "headers": {
            "timestamp": datetime.now(UTC),
            "source": "test-gateway-forwarder",
            "event_type": "omnibase-infra.gateway-heartbeat",
        },
    }
    dict_envelope: dict[str, Any] = {"payload": raw_event_message}

    output = await handler.handle(dict_envelope)

    assert output.result is not None
    payload = output.result.payload
    assert isinstance(payload, ModelPayloadGatewayLinkHealthUpsert)
    assert payload.tenant_id == "beta-gateway-canary-79afa7263852"
    assert payload.local_transport_flavor == "lightweight"
