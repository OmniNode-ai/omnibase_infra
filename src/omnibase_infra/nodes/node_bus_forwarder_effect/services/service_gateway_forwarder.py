# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Executable bus-to-bus gateway forwarder service.

The wire on both broker legs is the platform canonical
``ModelEventEnvelope``.  ``ModelGatewayEnvelope`` is a transform-boundary
model used by the node handlers; it is not a second event-bus envelope and
must never be required from ordinary runtime producers or consumers.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayForwarderConfig,
    ModelGatewayHeartbeat,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
    strip_topic_prefix,
)
from omnibase_infra.shared.tenant_stamp import stamp_verified_tenant_slug

logger = logging.getLogger(__name__)


class ProtocolGatewayPublisher(Protocol):
    """Destination publish boundary shared by push and pull transports."""

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        """Publish bytes to a topic."""


class ServiceGatewayForwarder:
    """Validate, transform, and republish explicitly polled gateway envelopes."""

    def __init__(
        self,
        *,
        config: ModelGatewayForwarderConfig,
        local_bus: ProtocolGatewayPublisher,
        cloud_bus: ProtocolGatewayPublisher,
        retry_sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._local_bus = local_bus
        self._cloud_bus = cloud_bus
        if retry_sleep is None:
            import asyncio

            retry_sleep = asyncio.sleep
        self._retry_sleep = retry_sleep

    async def _forward_outbound_message(self, message: object) -> None:
        source_topic = self._message_topic(message)
        envelope = self._decode_message(message)
        if envelope.metadata.tags.get("gateway_direction") == "cloud-to-local":
            logger.debug(
                "Skipping gateway loopback on local topic %s",
                source_topic,
            )
            return
        transformed = self._prepare_outbound(envelope, source_topic)
        wire_topic = prefix_topic(
            self._config.tenant_identity.tenant_slug,
            source_topic,
        )
        await self._publish_with_delivery_retry(
            bus=self._cloud_bus,
            topic=wire_topic,
            key=getattr(message, "key", None),
            value=self._encode_envelope(transformed),
            headers=getattr(message, "headers", None),
        )

    async def forward_outbound_message(self, message: object) -> None:
        """Validate, transform, and broker-acknowledge one outbound message."""
        await self._forward_outbound_message(message)

    def validate_outbound_message(self, message: object) -> None:
        """Validate an outbound trust-boundary message without publishing it."""
        source_topic = self._message_topic(message)
        envelope = self._decode_message(message)
        if envelope.metadata.tags.get("gateway_direction") != "cloud-to-local":
            self._prepare_outbound(envelope, source_topic)

    async def _consume_inbound_message(self, message: object) -> None:
        wire_topic = self._message_topic(message)
        envelope = self._decode_message(message)
        if envelope.metadata.tags.get("gateway_direction") == "local-to-cloud":
            logger.debug(
                "Skipping gateway loopback on cloud topic %s",
                wire_topic,
            )
            return
        transformed, canonical_topic = self._prepare_inbound(envelope, wire_topic)
        await self._publish_with_delivery_retry(
            bus=self._local_bus,
            topic=canonical_topic,
            key=getattr(message, "key", None),
            value=self._encode_envelope(transformed),
            headers=getattr(message, "headers", None),
        )

    async def consume_inbound_message(self, message: object) -> None:
        """Validate, transform, and broker-acknowledge one inbound message."""
        await self._consume_inbound_message(message)

    def validate_inbound_message(self, message: object) -> None:
        """Validate an inbound trust-boundary message without publishing it."""
        wire_topic = self._message_topic(message)
        envelope = self._decode_message(message)
        if envelope.metadata.tags.get("gateway_direction") != "local-to-cloud":
            self._prepare_inbound(envelope, wire_topic)

    @classmethod
    def decode_message(
        cls,
        message: object,
    ) -> ModelEventEnvelope[dict[str, object]]:
        """Decode the canonical envelope used as the durable dedupe key source."""
        return cls._decode_message(message)

    async def publish_heartbeat(self) -> None:
        """Publish one tenant-scoped liveness event onto the cloud wire topic."""
        identity = self._config.tenant_identity
        now = datetime.now(UTC)
        envelope_id = uuid4()
        heartbeat = ModelGatewayHeartbeat(
            tenant_id=identity.tenant_slug,
            principal_id=identity.principal_id,
            emitted_at=now,
            local_transport_flavor=self._config.local_transport_flavor,
        )
        envelope = ModelEventEnvelope[dict[str, object]](
            envelope_id=envelope_id,
            envelope_timestamp=now,
            correlation_id=envelope_id,
            source_tool="gateway-forwarder",
            event_type="omnibase-infra.gateway-heartbeat",
            payload=heartbeat.model_dump(mode="json"),
            metadata=ModelEnvelopeMetadata(
                tags={
                    "source_tenant_id": str(identity.tenant_id),
                    "source_tenant_principal_id": identity.principal_id,
                }
            ),
        )
        canonical_topic = next(
            topic
            for topic in self._config.mirror_topics.outbound
            if topic.endswith(".gateway-heartbeat.v1")
        )
        transformed = self._prepare_outbound(envelope, canonical_topic)
        await self._publish_with_delivery_retry(
            bus=self._cloud_bus,
            topic=prefix_topic(identity.tenant_slug, canonical_topic),
            key=str(identity.tenant_id).encode("utf-8"),
            value=self._encode_envelope(transformed),
        )

    async def _publish_with_delivery_retry(
        self,
        *,
        bus: ProtocolGatewayPublisher,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object | None = None,
    ) -> None:
        """Block source acknowledgement until a transient destination recovers.

        ``KafkaTransport.send`` awaits the destination broker acknowledgement.
        The gateway adds a process-lifetime retry around that boundary so the
        delivery node cannot record its durable marker or commit the source
        offset while the message exists on only one broker. Cancellation during
        shutdown is deliberately not caught.
        """
        delay = self._config.forward_retry_initial_seconds
        attempt = 0
        while True:
            try:
                await bus.publish(
                    topic=topic,
                    key=key,
                    value=value,
                    headers=headers,
                )
                return
            except RuntimeHostError as exc:
                attempt += 1
                logger.warning(
                    "Gateway destination unavailable; retaining source message "
                    "and retrying topic=%s attempt=%d delay_seconds=%.1f "
                    "error_type=%s",
                    topic,
                    attempt,
                    delay,
                    type(exc).__name__,
                )
                await self._retry_sleep(delay)
                delay = min(delay * 2, self._config.forward_retry_max_seconds)

    @staticmethod
    def _decode_message(message: object) -> ModelEventEnvelope[dict[str, object]]:
        value = getattr(message, "value", message)
        if isinstance(value, ModelEventEnvelope):
            return ModelEventEnvelope[dict[str, object]].model_validate(value)
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes):
            raise TypeError("gateway bus message value must be bytes or string")
        return ModelEventEnvelope[dict[str, object]].model_validate_json(value)

    @staticmethod
    def _encode_envelope(
        envelope: ModelEventEnvelope[dict[str, object]],
    ) -> bytes:
        return envelope.model_dump_json(exclude_none=True).encode("utf-8")

    @staticmethod
    def _message_topic(message: object) -> str:
        topic = getattr(message, "topic", None)
        if not isinstance(topic, str) or not topic:
            raise TypeError("gateway bus message must carry its source topic")
        return topic

    def _prepare_inbound(
        self,
        envelope: ModelEventEnvelope[dict[str, object]],
        wire_topic: str,
    ) -> tuple[ModelEventEnvelope[dict[str, object]], str]:
        """Validate a cloud command and stamp its config-bound local tenant."""
        identity = self._config.tenant_identity
        canonical_topic = strip_topic_prefix(identity.tenant_slug, wire_topic)
        if canonical_topic not in self._config.mirror_topics.inbound:
            raise ValueError("canonical_topic is not declared for inbound mirroring")

        tags = envelope.metadata.tags
        if tags.get("source_tenant_id") != str(identity.tenant_id):
            raise ValueError("envelope tenant_id does not match attached tenant")
        if tags.get("source_tenant_principal_id") != str(identity.principal_id):
            raise ValueError("envelope principal_id does not match attached tenant")

        payload = stamp_verified_tenant_slug(
            envelope.payload,
            identity.tenant_slug,
        )
        metadata = envelope.metadata.model_copy(
            update={
                "tags": {
                    **tags,
                    "gateway_tenant_id": str(identity.tenant_id),
                    "gateway_tenant_slug": identity.tenant_slug,
                    "gateway_principal_id": str(identity.principal_id),
                    "gateway_wire_topic": wire_topic,
                    "gateway_canonical_topic": canonical_topic,
                    "gateway_direction": "cloud-to-local",
                }
            }
        )
        return envelope.model_copy(
            update={"payload": payload, "metadata": metadata}
        ), canonical_topic

    def _prepare_outbound(
        self,
        envelope: ModelEventEnvelope[dict[str, object]],
        canonical_topic: str,
    ) -> ModelEventEnvelope[dict[str, object]]:
        """Validate a local event and bind it to the attached tenant."""
        identity = self._config.tenant_identity
        if canonical_topic not in self._config.mirror_topics.outbound:
            raise ValueError("canonical_topic is not declared for outbound mirroring")

        payload_tenant = envelope.payload.get("tenant_id")
        if payload_tenant is not None and payload_tenant != identity.tenant_slug:
            raise ValueError(
                "outbound payload tenant_id does not match attached tenant"
            )

        tags = envelope.metadata.tags
        existing_tenant_id = tags.get("source_tenant_id")
        if existing_tenant_id is not None and existing_tenant_id != str(
            identity.tenant_id
        ):
            raise ValueError(
                "outbound envelope tenant_id does not match attached tenant"
            )
        existing_principal_id = tags.get("source_tenant_principal_id")
        if existing_principal_id is not None and existing_principal_id != str(
            identity.principal_id
        ):
            raise ValueError(
                "outbound envelope principal_id does not match attached tenant"
            )

        wire_topic = prefix_topic(identity.tenant_slug, canonical_topic)
        metadata = envelope.metadata.model_copy(
            update={
                "tags": {
                    **tags,
                    "source_tenant_id": str(identity.tenant_id),
                    "source_tenant_principal_id": str(identity.principal_id),
                    "gateway_tenant_slug": identity.tenant_slug,
                    "gateway_wire_topic": wire_topic,
                    "gateway_canonical_topic": canonical_topic,
                    "gateway_direction": "local-to-cloud",
                }
            }
        )
        return envelope.model_copy(update={"metadata": metadata})
