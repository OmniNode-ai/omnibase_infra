# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed config for the tenant gateway bus forwarder."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_cloud_bus_config import (
    ModelGatewayCloudBusConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_mirror_topics import (
    ModelGatewayMirrorTopics,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_tenant_identity import (
    ModelGatewayTenantIdentity,
)


class ModelGatewayForwarderConfig(BaseModel):
    """Complete forwarder config for one attached tenant edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_identity: ModelGatewayTenantIdentity
    cloud_bus: ModelGatewayCloudBusConfig
    local_transport_flavor: Literal["containerized", "lightweight"]
    mirror_topics: ModelGatewayMirrorTopics
    heartbeat_interval_seconds: int = Field(default=15, ge=1)
    max_silence_window_seconds: int = Field(default=60, ge=1)
    lag_threshold_messages: int = Field(default=500, ge=1)
    lag_threshold_seconds: int = Field(default=120, ge=1)
    drain_deadline_seconds: int = Field(default=30, ge=1)
    dedupe_retention_hours: int = Field(default=24, ge=1)

    @model_validator(mode="after")
    def _validate_liveness_windows(self) -> ModelGatewayForwarderConfig:
        if self.max_silence_window_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "max_silence_window_seconds must exceed heartbeat_interval_seconds"
            )
        return self
