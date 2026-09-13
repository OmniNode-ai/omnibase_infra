# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16459: the HTTPS route deduplicates a content-addressed event identity."""

from __future__ import annotations

import hashlib
from uuid import UUID

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    content_addressed_event_id,
    envelope_with_event_id,
)

pytestmark = pytest.mark.unit


def test_event_id_is_content_addressed_and_distinct_from_envelope_id() -> None:
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=UUID("00000000-0000-0000-0000-000000000001"),
        event_type="omnibase.hook.tool-executed",
        payload={"tool": "read", "redaction_state": "redacted"},
    )
    canonical_topic = "onex.evt.omniclaude.tool-executed.v1"
    identified = envelope_with_event_id(envelope, canonical_topic)
    event_id = identified.metadata.tags["event_id"]

    assert event_id == content_addressed_event_id(envelope, canonical_topic)
    assert (
        event_id
        == hashlib.sha256(
            b'onex.evt.omniclaude.tool-executed.v1\n{"redaction_state":"redacted","tool":"read"}'
        ).hexdigest()
    )
    assert event_id != str(envelope.envelope_id)
    assert len(event_id) == 64


def test_event_id_survives_transport_metadata_and_rejects_content_tampering() -> None:
    envelope = ModelEventEnvelope[dict[str, object]](
        event_type="omnibase.hook.tool-executed",
        payload={"tool": "read", "redaction_state": "redacted"},
    )
    canonical_topic = "onex.evt.omniclaude.tool-executed.v1"
    identified = envelope_with_event_id(envelope, canonical_topic)
    forwarded = identified.model_copy(
        update={
            "metadata": identified.metadata.model_copy(
                update={
                    "tags": {
                        **identified.metadata.tags,
                        "gateway_direction": "local-to-cloud",
                    }
                }
            )
        }
    )

    assert (
        content_addressed_event_id(forwarded, canonical_topic)
        == identified.metadata.tags["event_id"]
    )
    tampered = forwarded.model_copy(update={"payload": {"tool": "write"}})
    assert (
        content_addressed_event_id(tampered, canonical_topic)
        != identified.metadata.tags["event_id"]
    )
