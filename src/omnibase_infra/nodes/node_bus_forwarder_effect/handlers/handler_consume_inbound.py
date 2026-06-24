# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Inbound gateway transform handler."""

from __future__ import annotations

from uuid import UUID

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayEnvelope,
    ModelGatewayForwarderConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
    strip_topic_prefix,
)


class HandlerConsumeInbound:
    """Validate cloud wire topics and strip to a bare local contract topic."""

    def __init__(self, config: ModelGatewayForwarderConfig | None = None) -> None:
        self._config = config

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self,
        envelope: ModelEventEnvelope[object],
    ) -> ModelHandlerOutput[ModelGatewayEnvelope]:
        """Dispatch entrypoint: extract gateway envelope, apply inbound transform, return compute output."""
        gateway_envelope, envelope_id, correlation_id = _coerce_dispatch_input(envelope)
        transformed = self.consume_inbound(gateway_envelope)
        return ModelHandlerOutput.for_compute(
            input_envelope_id=envelope_id,
            correlation_id=correlation_id,
            handler_id=type(self).__name__,
            result=transformed,
        )

    def consume_inbound(self, envelope: ModelGatewayEnvelope) -> ModelGatewayEnvelope:
        """Return an inbound envelope with a validated bare canonical topic."""
        config = self._require_config()
        identity = config.tenant_identity
        if envelope.tenant_id != identity.tenant_id:
            raise ValueError("envelope tenant_id does not match attached tenant")
        if envelope.tenant_slug != identity.tenant_slug:
            raise ValueError("envelope tenant_slug does not match attached tenant")

        canonical_topic = strip_topic_prefix(identity.tenant_slug, envelope.wire_topic)
        if canonical_topic != envelope.canonical_topic:
            raise ValueError("wire_topic and canonical_topic do not match")
        if canonical_topic not in config.mirror_topics.inbound:
            raise ValueError("canonical_topic is not declared for inbound mirroring")
        expected_wire_topic = prefix_topic(identity.tenant_slug, canonical_topic)
        if envelope.wire_topic != expected_wire_topic:
            raise ValueError("wire_topic does not match canonical tenant prefix")

        return envelope.model_copy(
            update={
                "source_topic": envelope.wire_topic,
                "canonical_topic": canonical_topic,
            }
        )

    def _require_config(self) -> ModelGatewayForwarderConfig:
        if self._config is None:
            raise ProtocolConfigurationError(
                "HandlerConsumeInbound requires ModelGatewayForwarderConfig"
            )
        return self._config


def _coerce_dispatch_input(
    envelope: ModelEventEnvelope[object],
) -> tuple[ModelGatewayEnvelope, UUID, UUID]:
    payload = envelope.payload
    gateway_envelope = (
        payload
        if isinstance(payload, ModelGatewayEnvelope)
        else ModelGatewayEnvelope.model_validate(payload)
    )
    return (
        gateway_envelope,
        envelope.envelope_id,
        envelope.correlation_id or gateway_envelope.correlation_id,
    )
