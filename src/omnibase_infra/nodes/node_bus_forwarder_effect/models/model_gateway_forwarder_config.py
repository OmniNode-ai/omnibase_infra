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
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_egress_redaction import (
    ModelGatewayEgressRedaction,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_https_ingest_config import (
    ModelGatewayHttpsIngestConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_lane_mirror_config import (
    ModelGatewayLaneMirrorConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_mirror_topics import (
    ModelGatewayMirrorTopics,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_tenant_identity import (
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    CONTENT_BEARING_HOOK_TOPICS,
)


class ModelGatewayForwarderConfig(BaseModel):
    """Complete forwarder config for one attached tenant edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_identity: ModelGatewayTenantIdentity
    cloud_bus: ModelGatewayCloudBusConfig
    local_transport_flavor: Literal["containerized", "lightweight"]
    mirror_topics: ModelGatewayMirrorTopics
    canary: ModelGatewayCanaryConfig
    # OMN-17034. Optional so a deployment that predates the lane-mirror leg
    # (or one where there is only one lane to begin with) keeps booting with
    # the two trust-boundary legs alone; the runtime config below is what
    # refuses a declaration whose broker legs were not resolved.
    lane_mirror: ModelGatewayLaneMirrorConfig | None = None
    # OMN-16459: opt-in HTTPS ingest leg for the OUTBOUND publish boundary.
    # ``None`` (the default) keeps the direct-MSK Kafka outbound leg, so every
    # deployment that has not opted in is byte-unchanged. The INBOUND leg is
    # a Kafka pull from the cloud broker either way -- see
    # ModelGatewayHttpsIngestConfig's module docstring for why that means this
    # block alone does not retire the OMN-16449 bastion.
    https_ingest: ModelGatewayHttpsIngestConfig | None = None
    # OMN-16979: fail-closed admission gate for the content-bearing hook
    # classes this ticket adds to ``mirror_topics.outbound``. Optional so every
    # deployment predating the widening keeps its exact behaviour; the
    # cross-field validator below is what refuses an inconsistent pairing.
    egress_redaction: ModelGatewayEgressRedaction | None = None
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
        self._validate_egress_redaction_pairing()
        return self

    def _validate_egress_redaction_pairing(self) -> None:
        """OMN-16979: the widening and the gate must agree, in both directions.

        A gate that names a topic nobody mirrors is dead policy that reads like
        live policy. A content-bearing hook class in the outbound set that the
        gate does NOT name is the credential pipeline OMN-17209 exists to
        prevent -- so it is refused here rather than merely discouraged.
        """
        policy = self.egress_redaction
        outbound = set(self.mirror_topics.outbound)
        if policy is not None:
            ungoverned_declarations = sorted(set(policy.governed_topics) - outbound)
            if ungoverned_declarations:
                raise ValueError(
                    "egress_redaction.governed_topics must all appear in "
                    f"mirror_topics.outbound; missing: {ungoverned_declarations}"
                )
        governed = set(policy.governed_topics) if policy is not None else set()
        unguarded = sorted(
            topic
            for topic in outbound
            if topic in CONTENT_BEARING_HOOK_TOPICS and topic not in governed
        )
        if unguarded:
            raise ValueError(
                "content-bearing hook topics may not be mirrored outbound "
                "unless egress_redaction declares them governed; unguarded: "
                f"{unguarded}"
            )
