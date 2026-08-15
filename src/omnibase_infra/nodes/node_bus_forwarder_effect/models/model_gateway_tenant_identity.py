# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tenant identity model for the gateway forwarder."""

from __future__ import annotations

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    RESERVED_TENANT_SLUGS,
)

_TENANT_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,61}[a-z0-9]$")
_PRINCIPAL_ID_RE = re.compile(r"^t-[0-9a-f]{32}$")


class ModelGatewayTenantIdentity(BaseModel):
    """Immutable tenant identity used to bind the gateway session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: UUID
    tenant_slug: str
    # Canonical MSK principal is t-<tenant UUID hex>, not a UUID field.
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
        if not _PRINCIPAL_ID_RE.fullmatch(principal_id):
            raise ValueError(
                "principal_id must use the canonical t-<32 lowercase hex> form"
            )
        return principal_id
