# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tenant-prefix transform helpers for the gateway trust boundary."""

from __future__ import annotations

import re

from omnibase_core.validation import validate_topic_suffix

RESERVED_TENANT_SLUGS = frozenset({"", "system"})
_TENANT_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,61}[a-z0-9]$")


def validate_tenant_slug(tenant_slug: str) -> str:
    """Validate a non-reserved tenant slug for gateway wire prefixes."""
    slug = tenant_slug.strip() if tenant_slug else tenant_slug
    if slug in RESERVED_TENANT_SLUGS:
        raise ValueError(f"tenant_slug is reserved: {slug!r}")
    if not slug or not _TENANT_SLUG_RE.match(slug) or "--" in slug:
        raise ValueError("tenant_slug must be DNS-compatible lowercase slug")
    return slug


def validate_canonical_topic(canonical_topic: str) -> str:
    """Validate a bare ONEX contract topic, never a tenant-prefixed wire topic."""
    topic = canonical_topic.strip() if canonical_topic else canonical_topic
    if not topic:
        raise ValueError("canonical_topic must not be empty")
    if topic.startswith("tenant-"):
        raise ValueError(
            "canonical_topic must be bare and must not carry tenant prefix"
        )
    result = validate_topic_suffix(topic)
    if not result.is_valid:
        raise ValueError(f"invalid canonical gateway topic: {result.error}")
    return topic


def prefix_topic(tenant_slug: str, canonical_topic: str) -> str:
    """Return the tenant-prefixed cloud wire topic."""
    slug = validate_tenant_slug(tenant_slug)
    topic = validate_canonical_topic(canonical_topic)
    return f"tenant-{slug}.{topic}"


def strip_topic_prefix(tenant_slug: str, wire_topic: str) -> str:
    """Strip and validate the tenant prefix from a cloud wire topic."""
    slug = validate_tenant_slug(tenant_slug)
    prefix = f"tenant-{slug}."
    if not wire_topic.startswith(prefix):
        raise ValueError("wire_topic does not match attached tenant prefix")
    canonical_topic = wire_topic[len(prefix) :]
    return validate_canonical_topic(canonical_topic)
