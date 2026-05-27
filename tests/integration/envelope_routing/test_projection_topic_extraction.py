# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for projection route topic extraction."""

from __future__ import annotations

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import _extract_projection_topic


def test_projection_topic_uses_onex_event_type_when_envelope_topic_is_absent() -> None:
    envelope = ModelEventEnvelope[dict[str, str]](
        event_type="onex.evt.omniclaude.task-delegated.v1",
        payload={"correlation_id": "runtime-proof"},
    )

    assert _extract_projection_topic(envelope) == envelope.event_type
