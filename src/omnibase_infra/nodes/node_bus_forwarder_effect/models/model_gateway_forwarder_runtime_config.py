# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolved process configuration for the gateway bus forwarder."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_config import (
    ModelGatewayForwarderConfig,
)


class ModelGatewayForwarderRuntimeConfig(BaseModel):
    """Contract plus the two resolved broker legs used by one edge process.

    ``forwarder.cloud_bus`` remains the provider-neutral declaration of the
    required capabilities and secret references.  ``cloud_bus`` and
    ``local_bus`` are the resolved, typed materialization consumed by
    ``KafkaTransport``.  Keeping the layers separate prevents a process-wide
    ``KAFKA_*`` environment from silently making both legs point at one broker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forwarder: ModelGatewayForwarderConfig
    local_bus: ModelKafkaEventBusConfig
    cloud_bus: ModelKafkaEventBusConfig

    @model_validator(mode="after")
    def _validate_resolved_legs(self) -> ModelGatewayForwarderRuntimeConfig:
        if self.forwarder.local_transport_flavor != "containerized":
            raise ValueError(
                "the production gateway process currently requires the "
                "containerized local transport flavor"
            )
        if self.local_bus.bootstrap_servers == self.cloud_bus.bootstrap_servers:
            raise ValueError("gateway local_bus and cloud_bus must be distinct")
        if self.local_bus.enable_auto_commit or self.cloud_bus.enable_auto_commit:
            raise ValueError(
                "gateway transport legs require enable_auto_commit=false; source "
                "offsets are committed only after durable destination delivery"
            )
        if not any(
            topic.endswith(".gateway-heartbeat.v1")
            for topic in self.forwarder.mirror_topics.outbound
        ):
            raise ValueError(
                "production gateway forwarder requires an outbound heartbeat topic"
            )

        declared_cloud = self.forwarder.cloud_bus
        if self.cloud_bus.security_protocol != declared_cloud.security_protocol:
            raise ValueError(
                "resolved cloud security_protocol does not match the gateway contract"
            )
        if self.cloud_bus.sasl_mechanism != declared_cloud.sasl_mechanism:
            raise ValueError(
                "resolved cloud sasl_mechanism does not match the gateway contract"
            )
        if (
            self.cloud_bus.sasl_mechanism == "AWS_MSK_IAM"
            and not self.cloud_bus.msk_region
        ):
            raise ValueError("resolved AWS_MSK_IAM cloud bus requires msk_region")
        return self


__all__ = ["ModelGatewayForwarderRuntimeConfig"]
