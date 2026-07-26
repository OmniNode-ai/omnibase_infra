# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Default replay-envelope deserializer for node_kafka_replay_compute.

Isolates the concrete ``ModelEventEnvelope`` deserialization boundary in its own
non-handler support module so the canonical def-B handler core
(``handler_replay.py``) references only the narrow ``ProtocolReplayEnvelope`` view
-- satisfying the C-core requirement of the canonical handler-shape ratchet
(OMN-14355). The handler injects this as the default ``envelope_deserializer`` and
swaps it out under test.
"""

from __future__ import annotations

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_kafka_replay_compute.protocols.protocol_replay_envelope import (
    ProtocolReplayEnvelope,
)


def default_envelope_deserializer(value: bytes) -> ProtocolReplayEnvelope:
    """Deserialize canonical event-bus envelope bytes into a replay envelope view."""
    return ModelEventEnvelope[object].model_validate_json(value)


__all__ = ["default_envelope_deserializer"]
