# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed config for the tenant gateway bus forwarder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_canary_config import (
    ModelGatewayCanaryConfig,
)
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
    canary: ModelGatewayCanaryConfig
    heartbeat_interval_seconds: int = Field(default=15, ge=1)
    max_silence_window_seconds: int = Field(default=60, ge=1)
    lag_threshold_messages: int = Field(default=500, ge=1)
    lag_threshold_seconds: int = Field(default=120, ge=1)
    drain_deadline_seconds: int = Field(default=30, ge=1)
    dedupe_store_path: Path
    dedupe_retention_hours: int = Field(default=24, ge=24)
    forward_retry_initial_seconds: float = Field(default=1.0, gt=0)
    forward_retry_max_seconds: float = Field(default=30.0, gt=0)
    reconnect_backoff_initial_seconds: float = Field(default=1.0, gt=0)
    reconnect_backoff_max_seconds: float = Field(default=30.0, gt=0)
    reconnect_backoff_jitter_seconds: float = Field(default=0.5, ge=0)
    degraded_after_seconds: int = Field(default=60, ge=1)

    @field_validator("dedupe_store_path")
    @classmethod
    def _validate_dedupe_store_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(
                "dedupe_store_path must be absolute so deployment persistence "
                "cannot depend on the container working directory"
            )
        return value

    @model_validator(mode="after")
    def _validate_liveness_windows(self) -> ModelGatewayForwarderConfig:
        if self.max_silence_window_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "max_silence_window_seconds must exceed heartbeat_interval_seconds"
            )
        if self.forward_retry_max_seconds < self.forward_retry_initial_seconds:
            raise ValueError(
                "forward_retry_max_seconds must be greater than or equal to "
                "forward_retry_initial_seconds"
            )
        if self.canary.topic in self.mirror_topics.inbound or (
            self.canary.topic in self.mirror_topics.outbound
        ):
            raise ValueError(
                "canary.topic must be dedicated and must not appear in "
                "mirror_topics.inbound or mirror_topics.outbound"
            )
        if self.reconnect_backoff_max_seconds < self.reconnect_backoff_initial_seconds:
            raise ValueError(
                "reconnect_backoff_max_seconds must be greater than or equal to "
                "reconnect_backoff_initial_seconds"
            )
        return self
