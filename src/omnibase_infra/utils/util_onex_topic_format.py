# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ONEX topic format validator.

Validates that Kafka topic names conform to the canonical ONEX 5-segment format:

    onex.<kind>.<producer>.<event-name>.v<N>

where kind is one of: evt, cmd, intent, dlq.

Tenant gateway wire topics matching ``tenant-<slug>.<canonical-onex-topic>``
and legacy DLQ topics matching ``<prefix>.dlq.<name>.v<N>`` are accepted with
distinct result codes so callers can distinguish them from bare canonical names.

Kafka-internal topics (prefixed with ``__``) are silently skipped.

OMN-15792: the ``tenant-<slug>.`` wire-prefix branch delegates to
``resolve_tenant_from_wire_topic`` -- THE single runtime topic resolver
(``service_gateway_topic_transform``) also consulted by the subscribe/dispatch
path (``handler_wiring.py``). This module previously hand-rolled its own
``tenant-<slug>.`` regex with no ``RESERVED_TENANT_SLUGS`` awareness -- a
THIRD independent resolver on the live Kafka publish path
(``event_bus_kafka.py``'s ``_enforce_onex_topic_format``) that could
publish-ALLOW a topic the subscribe side rejects (or vice versa). Delegating
here closes that divergence: this module and the subscribe path now share the
exact same slug-validation primitive and cannot disagree.
"""

from __future__ import annotations

import re
from enum import StrEnum

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    resolve_tenant_from_wire_topic,
)

_RE_ONEX_TOPIC = re.compile(
    r"^onex\.(evt|cmd|intent|dlq)\.[a-z0-9-]+\.[a-z0-9._-]+\.v[1-9]\d*$"
)

_RE_LEGACY_DLQ = re.compile(r"^[a-z][a-z0-9-]*\.dlq\.[a-z0-9-]+\.v[1-9]\d*$")

_KAFKA_INTERNAL_PREFIX = "__"


class TopicValidationResult(StrEnum):
    """Outcome of validating a topic name against the ONEX format."""

    VALID = "valid"
    VALID_TENANT_WIRE = "valid_tenant_wire"
    VALID_LEGACY_DLQ = "valid_legacy_dlq"
    INVALID = "invalid"
    SKIPPED_INTERNAL = "skipped_internal"


def validate_onex_topic_format(topic: str) -> tuple[TopicValidationResult, str]:
    """Validate *topic* against the canonical ONEX topic format.

    Returns a ``(result, reason)`` tuple.  *reason* is an empty string when
    the topic is valid or skipped, and a human-readable explanation otherwise.
    """
    if topic.startswith(_KAFKA_INTERNAL_PREFIX):
        return (TopicValidationResult.SKIPPED_INTERNAL, "")
    if _RE_ONEX_TOPIC.match(topic):
        return (TopicValidationResult.VALID, "")
    try:
        tenant_slug, _canonical_topic = resolve_tenant_from_wire_topic(topic)
    except ValueError as exc:
        # A structurally tenant-prefixed topic (``tenant-<slug>.``) whose slug
        # the shared resolver rejects (reserved, e.g. ``tenant-system.``, or
        # malformed) -- publish-enforce and subscribe-resolve must agree, so
        # this raises INVALID here exactly as the subscribe path raises.
        return (
            TopicValidationResult.INVALID,
            f"Topic '{topic}' has an invalid tenant wire prefix: {exc}",
        )
    if tenant_slug is not None:
        return (TopicValidationResult.VALID_TENANT_WIRE, "")
    if _RE_LEGACY_DLQ.match(topic):
        return (TopicValidationResult.VALID_LEGACY_DLQ, "legacy DLQ format")
    return (
        TopicValidationResult.INVALID,
        f"Topic '{topic}' does not match ONEX format: "
        "onex.(evt|cmd|intent|dlq).<producer>.<event-name>.v<N> or "
        "tenant-<slug>.<canonical-onex-topic>",
    )
