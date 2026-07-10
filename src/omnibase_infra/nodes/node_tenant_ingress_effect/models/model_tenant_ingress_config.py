# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Config model for the tenant-ingress effect node."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Reserved slugs mirror node_bus_forwarder_effect's convention (the same
# tenant-<slug>. wire-prefix scheme, OMN-12908/12911) -- kept local rather than
# cross-imported so this node stays independently deployable.
_RESERVED_TENANT_SLUGS = frozenset({"", "system"})
_TENANT_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,61}[a-z0-9]$")


def validate_tenant_slug(tenant_slug: str) -> str:
    """Validate a non-reserved tenant slug for tenant-ingress wire prefixes."""
    slug = tenant_slug.strip() if tenant_slug else tenant_slug
    if slug in _RESERVED_TENANT_SLUGS:
        raise ValueError(f"tenant_slug is reserved: {slug!r}")
    if not slug or not _TENANT_SLUG_RE.match(slug) or "--" in slug:
        raise ValueError("tenant_slug must be DNS-compatible lowercase slug")
    return slug


class ModelTenantIngressConfig(BaseModel):
    """Frozen config: the static list of provisioned tenants and the target topic.

    OMN-14349 (OMN-14208 Path A): a static per-tenant list is correct at
    current dogfood/early-beta scale. Dynamic per-tenant provisioning at
    scale is Daniyal's forward AWS/MSK work (OMN-14110), not this node's job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenants: tuple[str, ...] = Field(
        ...,
        description=(
            "Provisioned tenant slugs. Each gets its own subscription to "
            "tenant-<slug>.<canonical_topic>."
        ),
    )
    canonical_topic: str = Field(
        ...,
        description=(
            "Bare, contract-declared topic to republish stamped messages onto. "
            "Never hardcoded in code -- resolved from contract.yaml."
        ),
    )

    @field_validator("tenants")
    @classmethod
    def _validate_tenants(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(validate_tenant_slug(slug) for slug in value)
        if len(set(validated)) != len(validated):
            raise ValueError("tenants must not contain duplicate slugs")
        return validated

    @field_validator("canonical_topic")
    @classmethod
    def _validate_canonical_topic(cls, value: str) -> str:
        topic = value.strip() if value else value
        if not topic:
            raise ValueError("canonical_topic must not be empty")
        if topic.startswith("tenant-"):
            raise ValueError("canonical_topic must be bare, not tenant-prefixed")
        return topic


__all__ = ["ModelTenantIngressConfig", "validate_tenant_slug"]
