# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tenant-prefix transform helpers for the gateway trust boundary.

OMN-15792 (2026-08-09 operator addressing ruling): ``resolve_physical_topic``
and its inverse ``resolve_tenant_from_wire_topic`` are THE single runtime
topic resolver -- physical topic addressing resolved from a contract-declared
canonical topic plus optional tenant execution context. Every publish and
subscribe call site that needs tenant-aware physical topic addressing MUST
route through these two functions rather than re-deriving the transform.
This module does not redesign the wire format (still ``tenant-{slug}.``);
it makes ``prefix_topic``/``strip_topic_prefix`` the sole path instead of one
of several independent implementations (OMN-15757/OMN-15778's structural root
cause).
"""

from __future__ import annotations

import re

from omnibase_core.validation import validate_topic_suffix

RESERVED_TENANT_SLUGS = frozenset({"", "system"})
_TENANT_SLUG_PATTERN = r"[a-z][a-z0-9-]{1,61}[a-z0-9]"
_TENANT_SLUG_RE = re.compile(rf"^{_TENANT_SLUG_PATTERN}$")
# Inline-cites _TENANT_SLUG_PATTERN rather than hand-duplicating the character
# class (OMN-15759 is the tracked ticket for a cross-repo shared constant;
# until it lands, this is the single in-repo copy the wire-prefix matcher and
# the slug validator both derive from).
_TENANT_WIRE_PREFIX_RE = re.compile(rf"^tenant-({_TENANT_SLUG_PATTERN})\.")


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


def resolve_physical_topic(canonical_topic: str, *, tenant_slug: str | None) -> str:
    """Resolve a contract-declared canonical topic to the physical wire topic.

    THE single resolver (OMN-15792) consulted by both the publish path and
    the subscribe/dispatch path: contract-declared canonical topic + optional
    tenant execution context -> physical topic.

    * ``tenant_slug=None`` -> bare canonical topic (unchanged pass-through
      behavior for non-gateway/local-only paths).
    * ``tenant_slug`` present -> ``tenant-{slug}.{canonical_topic}``, via
      ``prefix_topic`` -- the wire format is not redesigned here.
    """
    if tenant_slug is None:
        return validate_canonical_topic(canonical_topic)
    return prefix_topic(tenant_slug, canonical_topic)


def resolve_tenant_from_wire_topic(wire_topic: str) -> tuple[str | None, str]:
    """Inverse of ``resolve_physical_topic``: derive ``(tenant_slug, canonical_topic)``.

    THE single resolver's subscribe-side direction (OMN-15792) -- the runtime
    dispatch layer calls this instead of re-deriving a tenant prefix with a
    private regex.

    Returns ``(None, wire_topic)`` unchanged when ``wire_topic`` does not
    start with the ``tenant-`` prefix at all -- never a defaulted or guessed
    tenant (Stage-1 warn semantics, matching the OMN-14349 stamp's existing
    contract).

    ``tenant-`` is a reserved wire-format prefix: ``validate_canonical_topic``
    already rejects any bare contract-declared topic that starts with it, so
    ANY string starting with ``tenant-`` is by construction an attempted
    tenant-wire topic, never a coincidentally-named bare canonical topic.
    Once that prefix is present, the full ``tenant-<slug>.`` shape and the
    embedded slug (``validate_tenant_slug`` -- reserved or malformed slugs
    included) are both enforced; a shape or slug failure raises rather than
    silently falling back to "no tenant". This closes the divergence class
    where a malformed-looking ``tenant-`` prefix (wrong case, too short) was
    published-side REJECTED but subscribe-side silently passed through
    untenanted -- the two directions must agree (both-accept or
    both-reject), not just on well-formed-but-reserved slugs.
    """
    if not wire_topic.startswith("tenant-"):
        return None, wire_topic
    match = _TENANT_WIRE_PREFIX_RE.match(wire_topic)
    if match is None:
        raise ValueError(
            "wire_topic starts with the reserved 'tenant-' prefix but does "
            f"not match the tenant wire-topic shape: {wire_topic!r}"
        )
    slug = validate_tenant_slug(match.group(1))
    canonical_topic = validate_canonical_topic(wire_topic[len(match.group(0)) :])
    return slug, canonical_topic
