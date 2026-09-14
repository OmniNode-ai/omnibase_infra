# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event-bus operations consumed by :class:`MixinNodeIntrospection`."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)


@runtime_checkable
class ProtocolIntrospectionEventBus(ProtocolPatternBBrokerTransport, Protocol):
    """Publish and subscribe surface required by node introspection.

    ``MixinNodeIntrospection`` publishes introspection, heartbeat, and
    registration-ack envelopes.  The envelope path preserves the message and
    correlation identities that ``EventBusKafka`` projects into wire headers.
    Raw publish and subscribe semantics are inherited from the canonical
    Pattern B broker transport contract.
    """

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
        *,
        key: bytes | None = None,
    ) -> None:
        """Publish an envelope while preserving its transport identity."""
        ...


__all__ = ["ProtocolIntrospectionEventBus"]
