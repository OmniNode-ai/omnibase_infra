# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16459: the HTTPS route deduplicates on the shared wire identity."""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_core.models.contracts.model_ingest_contract import compute_event_id
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from tests.unit.nodes.node_bus_forwarder_effect.test_https_ingest_omn16459 import (
    recording_https_ingest,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_TOPIC = "onex.evt.omniclaude.tool-executed.v1"


async def test_event_id_is_content_addressed_and_distinct_from_envelope_id() -> None:
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=UUID("00000000-0000-0000-0000-000000000001"),
        event_type="omnibase.hook.tool-executed",
        payload={"tool": "read", "redaction_state": "redacted"},
    )
    transport, records = recording_https_ingest()
    value = envelope.model_dump_json().encode("utf-8")
    headers = {"message-id": b"record-a"}

    await transport.publish(_TOPIC, b"key", value, headers)

    event_id = records[0]["event_id"]
    assert event_id == compute_event_id(_TOPIC, b"key", value, headers)
    assert isinstance(event_id, str)
    assert event_id != str(envelope.envelope_id)
    assert len(event_id) == 64


async def test_event_id_ignores_transport_headers_and_detects_content_tampering() -> (
    None
):
    envelope = ModelEventEnvelope[dict[str, object]](
        event_type="omnibase.hook.tool-executed",
        payload={"tool": "read", "redaction_state": "redacted"},
    )
    transport, records = recording_https_ingest()
    value = envelope.model_dump_json().encode("utf-8")
    tampered = envelope.model_copy(update={"payload": {"tool": "write"}})
    tampered_value = tampered.model_dump_json().encode("utf-8")

    await transport.publish(_TOPIC, None, value)
    headers = {"gateway-direction": b"local-to-cloud"}
    await transport.publish(_TOPIC, None, value, headers)
    await transport.publish(_TOPIC, None, tampered_value, headers)

    assert records[0]["event_id"] == compute_event_id(_TOPIC, None, value, {})
    assert records[1]["event_id"] == records[0]["event_id"]
    assert records[2]["event_id"] == compute_event_id(
        _TOPIC, None, tampered_value, headers
    )
    assert records[2]["event_id"] != records[0]["event_id"]
