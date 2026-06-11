# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executable bus-to-bus gateway forwarder service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from omnibase_infra.nodes.node_bus_forwarder_effect.handlers import (
    HandlerConsumeInbound,
    HandlerForwardOutbound,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayEnvelope,
    ModelGatewayForwarderConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
)


class ProtocolGatewayBus(Protocol):
    """Structural subset shared by EventBusKafka, EventBusInmemory, and tests."""

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Any | None = None,
    ) -> None:
        """Publish bytes to a topic."""

    async def subscribe(
        self,
        topic: str,
        node_identity: Any | None = None,
        on_message: Callable[[Any], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        **kwargs: Any,
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe to a topic and return an async unsubscribe callback."""


class ServiceGatewayForwarder:
    """Subscribe to mirrored topics on both legs and republish transformed envelopes."""

    def __init__(
        self,
        *,
        config: ModelGatewayForwarderConfig,
        local_bus: ProtocolGatewayBus,
        cloud_bus: ProtocolGatewayBus,
    ) -> None:
        self._config = config
        self._local_bus = local_bus
        self._cloud_bus = cloud_bus
        self._outbound_handler = HandlerForwardOutbound(config)
        self._inbound_handler = HandlerConsumeInbound(config)
        self._unsubscribe_callbacks: list[Callable[[], Awaitable[None]]] = []
        self._started = False

    async def start(self) -> None:
        """Start subscriptions on both bus legs."""
        if self._started:
            return

        for topic in self._config.mirror_topics.outbound:
            unsubscribe = await self._local_bus.subscribe(
                topic=topic,
                group_id=self._group_id("outbound"),
                on_message=self._forward_outbound_message,
            )
            self._unsubscribe_callbacks.append(unsubscribe)

        tenant_slug = self._config.tenant_identity.tenant_slug
        for topic in self._config.mirror_topics.inbound:
            unsubscribe = await self._cloud_bus.subscribe(
                topic=prefix_topic(tenant_slug, topic),
                group_id=self._group_id("inbound"),
                on_message=self._consume_inbound_message,
            )
            self._unsubscribe_callbacks.append(unsubscribe)

        self._started = True

    async def stop(self) -> None:
        """Stop all active subscriptions."""
        callbacks = list(reversed(self._unsubscribe_callbacks))
        self._unsubscribe_callbacks.clear()
        self._started = False
        for unsubscribe in callbacks:
            await unsubscribe()

    async def _forward_outbound_message(self, message: Any) -> None:
        envelope = self._decode_message(message)
        transformed = self._outbound_handler.handle(envelope)
        await self._cloud_bus.publish(
            topic=transformed.wire_topic,
            key=getattr(message, "key", None),
            value=self._encode_envelope(transformed),
            headers=getattr(message, "headers", None),
        )

    async def _consume_inbound_message(self, message: Any) -> None:
        envelope = self._decode_message(message)
        transformed = self._inbound_handler.handle(envelope)
        await self._local_bus.publish(
            topic=transformed.canonical_topic,
            key=getattr(message, "key", None),
            value=self._encode_envelope(transformed),
            headers=getattr(message, "headers", None),
        )

    def _group_id(self, direction: str) -> str:
        identity = self._config.tenant_identity
        return f"tenant-{identity.tenant_slug}-gateway-forwarder-{direction}"

    @staticmethod
    def _decode_message(message: Any) -> ModelGatewayEnvelope:
        value = getattr(message, "value", message)
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes):
            raise TypeError("gateway bus message value must be bytes or string")
        return ModelGatewayEnvelope.model_validate_json(value)

    @staticmethod
    def _encode_envelope(envelope: ModelGatewayEnvelope) -> bytes:
        return envelope.model_dump_json().encode("utf-8")
