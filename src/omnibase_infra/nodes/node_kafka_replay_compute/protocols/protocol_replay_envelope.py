# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay-envelope protocol for Kafka replay compute.

The replay handler only needs the ``correlation_id`` of a deserialized event-bus
envelope to build its correlation chain. Depending on a narrow structural protocol
(instead of the concrete ``ModelEventEnvelope`` type) keeps the canonical def-B
handler module free of the envelope type -- the C-core requirement of the
canonical handler-shape ratchet (OMN-14355). The default deserializer still yields
a ``ModelEventEnvelope``, which structurally satisfies this protocol.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class ProtocolReplayEnvelope(Protocol):
    """Structural view of a deserialized replay envelope used by the handler core."""

    correlation_id: UUID | None


__all__ = ["ProtocolReplayEnvelope"]
