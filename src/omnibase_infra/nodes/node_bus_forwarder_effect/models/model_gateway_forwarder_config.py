# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed config for the tenant gateway bus forwarder."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    RESERVED_TENANT_SLUGS,
    validate_canonical_topic,
)

_TENANT_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,61}[a-z0-9]$")


class ModelGatewayTenantIdentity(BaseModel):
    """Immutable tenant identity used to bind the gateway session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: UUID
    tenant_slug: str
    principal_id: str

    @field_validator("tenant_slug")
    @classmethod
    def _validate_tenant_slug(cls, value: str) -> str:
        slug = value.strip()
        if slug in RESERVED_TENANT_SLUGS:
            raise ValueError(f"tenant_slug is reserved: {slug}")
        if not _TENANT_SLUG_RE.match(slug) or "--" in slug:
            raise ValueError("tenant_slug must be DNS-compatible lowercase slug")
        return slug

    @field_validator("principal_id")
    @classmethod
    def _validate_principal_id(cls, value: str) -> str:
        principal_id = value.strip()
        if not principal_id:
            raise ValueError("principal_id must not be empty")
        return principal_id


class ModelGatewayCloudBusConfig(BaseModel):
    """Provider-neutral cloud Kafka leg config resolved from contract refs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_provider_id: str = Field(..., min_length=1)
    cloud_broker_ref: str = Field(..., min_length=1)
    cloud_auth_ref: str = Field(..., min_length=1)
    acl_provisioner_ref: str = Field(..., min_length=1)
    client_id_ref: str = Field(..., min_length=1)
    client_secret_api_key_ref: str = Field(..., min_length=1)
    security_protocol: Literal["SASL_SSL"] = "SASL_SSL"
    sasl_mechanism: Literal["OAUTHBEARER"] = "OAUTHBEARER"

    @field_validator(
        "broker_provider_id",
        "cloud_broker_ref",
        "cloud_auth_ref",
        "acl_provisioner_ref",
        "client_id_ref",
        "client_secret_api_key_ref",
    )
    @classmethod
    def _validate_contract_ref(cls, value: str) -> str:
        ref = value.strip()
        if not ref:
            raise ValueError("gateway cloud bus refs must not be empty")
        if ref.startswith("KAFKA_"):
            raise ValueError(
                "gateway cloud bus config must use contract refs, not KAFKA_* env"
            )
        return ref


class ModelGatewayMirrorTopics(BaseModel):
    """Bare contract-declared topics mirrored across the gateway."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inbound: tuple[str, ...] = Field(..., min_length=1)
    outbound: tuple[str, ...] = Field(..., min_length=1)

    @field_validator("inbound", "outbound")
    @classmethod
    def _validate_topics(cls, topics: tuple[str, ...]) -> tuple[str, ...]:
        for topic in topics:
            validate_canonical_topic(topic)
        return topics


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
