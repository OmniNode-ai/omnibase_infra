# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic naming constants and utilities for ONEX event bus.

This module defines DLQ (Dead Letter Queue) topic naming conventions and
wiring health monitoring topic constants for the ONEX event-driven architecture.

IMPORTANT: All ONEX topics (including DLQ) are realm-agnostic -- environment
prefixes (dev., prod., etc.) must NOT appear on the wire. Environment isolation
is enforced at the bus level (separate Redpanda instances for local vs cloud).
See ``omnibase_infra.topics.TopicResolver`` for the canonical resolution path.

DLQ Topic Naming:
    - **Format**: ``onex.dlq.<category>.<version>``
    - Example: ``onex.dlq.intents.v1``, ``onex.dlq.events.v1``

    This convention ensures:
    - DLQ topics are clearly identifiable by the 'dlq' domain
    - Category (intents, events, commands) is preserved for routing analysis
    - Version control for DLQ message schema evolution

Usage:
    >>> from omnibase_infra.event_bus.topic_constants import (
    ...     build_dlq_topic,
    ...     DLQ_INTENT_TOPIC_SUFFIX,
    ... )
    >>>
    >>> # Build realm-agnostic DLQ topic
    >>> topic = build_dlq_topic("intents")
    >>> print(topic)
    onex.dlq.intents.v1

See Also:
    - ModelKafkaEventBusConfig.dead_letter_topic: DLQ configuration
    - EventBusKafka._publish_to_dlq(): DLQ publishing implementation
    - topic_category_validator.py: Topic naming validation

.. versionchanged:: 0.21.0
    OMN-5189: DLQ topics are now realm-agnostic (fixed ``onex`` prefix).
    ``build_dlq_topic()`` no longer takes an ``environment`` parameter.
"""

from __future__ import annotations

import re
from typing import Final

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

# ==============================================================================
# DLQ Topic Version
# ==============================================================================
# Version suffix for DLQ topics. Increment when DLQ message schema changes.
# Current schema includes: original_topic, original_message, failure_reason,
# failure_timestamp, correlation_id, retry_count, error_type

DLQ_TOPIC_VERSION: Final[str] = "v1"

# ==============================================================================
# DLQ Topic Domain
# ==============================================================================
# The 'dlq' domain identifies Dead Letter Queue topics

DLQ_DOMAIN: Final[str] = "dlq"

# ==============================================================================
# DLQ Topic Suffixes (without environment prefix)
# ==============================================================================
# These suffixes can be combined with environment prefix to form full topic names.
# Format: dlq.<category>.<version>

DLQ_INTENT_TOPIC_SUFFIX: Final[str] = f"{DLQ_DOMAIN}.intents.{DLQ_TOPIC_VERSION}"
"""DLQ topic suffix for permanently failed intents: 'dlq.intents.v1'"""

DLQ_EVENT_TOPIC_SUFFIX: Final[str] = f"{DLQ_DOMAIN}.events.{DLQ_TOPIC_VERSION}"
"""DLQ topic suffix for permanently failed events: 'dlq.events.v1'"""

DLQ_COMMAND_TOPIC_SUFFIX: Final[str] = f"{DLQ_DOMAIN}.commands.{DLQ_TOPIC_VERSION}"
"""DLQ topic suffix for permanently failed commands: 'dlq.commands.v1'"""

# ==============================================================================
# Category-to-Suffix Mapping
# ==============================================================================

DLQ_CATEGORY_SUFFIXES: Final[dict[str, str]] = {
    "intent": DLQ_INTENT_TOPIC_SUFFIX,
    "intents": DLQ_INTENT_TOPIC_SUFFIX,
    "event": DLQ_EVENT_TOPIC_SUFFIX,
    "events": DLQ_EVENT_TOPIC_SUFFIX,
    "command": DLQ_COMMAND_TOPIC_SUFFIX,
    "commands": DLQ_COMMAND_TOPIC_SUFFIX,
}
"""Mapping from message category to DLQ topic suffix (singular and plural forms)."""

# ==============================================================================
# DLQ Topic Validation Pattern
# ==============================================================================
# Validates DLQ topics in realm-agnostic format: onex.dlq.<category>.<version>
# - prefix: must be 'onex' (fixed, realm-agnostic)
# - domain: must be 'dlq'
# - category: lowercase identifier (intents, events, commands, intelligence, platform, etc.)
# - version: v followed by digits (e.g., v1, v2)

DLQ_TOPIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<prefix>[\w-]+)\.dlq\.(?P<category>[a-z][a-z0-9_-]*)\.(?P<version>v\d+)$",
    re.IGNORECASE,
)
"""
Regex pattern for validating DLQ topic names.

Groups:
    - prefix: Topic prefix (canonical: 'onex'; legacy env prefixes also matched
      for backward-compatible parsing)
    - category: DLQ category (intents, events, commands, intelligence, platform, etc.)
    - version: Topic version (e.g., 'v1')

Example matches:
    - onex.dlq.intents.v1
    - onex.dlq.events.v1
    - onex.dlq.commands.v2
    - onex.dlq.intelligence.v1
    - onex.dlq.platform.v1

.. versionchanged:: 0.7.0
    Expanded category pattern from ``intents|events|commands`` to any
    lowercase identifier to support domain-based DLQ routing (OMN-2040).

.. versionchanged:: 0.21.0
    OMN-5189: DLQ topics now use fixed ``onex`` prefix. Pattern still accepts
    any alphanumeric prefix for backward-compatible parsing of legacy topics.
"""

# ==============================================================================
# DLQ Category Validation Pattern
# ==============================================================================
# Validates DLQ category identifiers: starts with letter, followed by lowercase
# letters, digits, hyphens, or underscores. This pattern is used by build_dlq_topic()
# to accept both standard categories (intents, events, commands) and domain-based
# categories (intelligence, platform, etc.).

_DLQ_CATEGORY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_-]*$")
"""
Regex pattern for validating DLQ category identifiers.

Valid examples: 'intents', 'events', 'intelligence', 'platform', 'my-domain'
Invalid examples: '123abc', '-starts-with-dash', '', 'UPPER'

.. versionadded:: 0.7.0
    Added for domain-based DLQ routing (OMN-2040).
"""


_DLQ_PREFIX: Final[str] = "onex"
"""Fixed prefix for all DLQ topics. DLQ topics are realm-agnostic."""


def build_dlq_topic(
    category: str,
    *,
    version: str | None = None,
) -> str:
    # INTENTIONAL ENV PREFIX: DLQ topics are infrastructure-scoped, not event routing.
    # They use the fixed "onex" prefix rather than environment-based prefixes because
    # DLQ routing is an infrastructure concern, not a domain routing concern.
    """Build a realm-agnostic DLQ topic name from components.

    Constructs a Dead Letter Queue topic name following ONEX conventions
    in realm-agnostic format: ``onex.dlq.<category>.<version>``.

    Args:
        category: DLQ category identifier. Accepts standard message categories
            in singular or plural form ('intent'/'intents', 'event'/'events',
            'command'/'commands') which are normalized to plural form, as well
            as domain-based categories ('intelligence', 'platform', 'agent',
            etc.) which pass through as-is.
        version: Optional topic version (e.g., 'v1', 'v2'). If not provided,
            defaults to DLQ_TOPIC_VERSION ('v1').

    Returns:
        Realm-agnostic DLQ topic name.

    Raises:
        ProtocolConfigurationError: If category is invalid.

    Example:
        >>> build_dlq_topic("intents")
        'onex.dlq.intents.v1'
        >>> build_dlq_topic("intent")  # Singular form accepted
        'onex.dlq.intents.v1'
        >>> build_dlq_topic("events", version="v2")
        'onex.dlq.events.v2'
        >>> build_dlq_topic("commands")
        'onex.dlq.commands.v1'
        >>> build_dlq_topic("intelligence")
        'onex.dlq.intelligence.v1'

    .. versionchanged:: 0.21.0
        OMN-5189: Removed ``environment`` parameter. DLQ topics now use
        fixed ``onex`` prefix for realm-agnostic naming.
    """
    # Normalize category to lowercase and validate format
    cat_lower = category.lower().strip()
    if not cat_lower:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="build_dlq_topic",
        )
        raise ProtocolConfigurationError(
            "category cannot be empty",
            context=context,
            parameter="category",
        )

    # Validate category format: must start with letter, then alphanumeric/hyphens/underscores
    if not _DLQ_CATEGORY_PATTERN.match(cat_lower):
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="build_dlq_topic",
        )
        raise ProtocolConfigurationError(
            f"Invalid category '{category}'. "
            "Must start with a letter and contain only lowercase letters, digits, "
            "hyphens, or underscores.",
            context=context,
            parameter="category",
            value=category,
        )

    # Determine version to use
    topic_version = version if version else DLQ_TOPIC_VERSION

    # Normalize standard categories to plural form for consistency;
    # domain-based categories (e.g., "intelligence", "platform") pass through as-is.
    normalized_category = _normalize_category(cat_lower)

    return f"{_DLQ_PREFIX}.{DLQ_DOMAIN}.{normalized_category}.{topic_version}"


def _normalize_category(category: str) -> str:
    """Normalize category to plural form.

    Args:
        category: Category in singular or plural form.

    Returns:
        Category in plural form (intents, events, commands).
    """
    category_map = {
        "intent": "intents",
        "intents": "intents",
        "event": "events",
        "events": "events",
        "command": "commands",
        "commands": "commands",
    }
    return category_map.get(category, category)


def parse_dlq_topic(topic: str) -> dict[str, str] | None:
    """Parse a DLQ topic name into its components.

    Extracts prefix, category, and version from a DLQ topic name
    that follows the ONEX naming convention.

    Args:
        topic: The DLQ topic name to parse.

    Returns:
        A dictionary with keys 'prefix', 'category', and 'version'
        if the topic matches the DLQ pattern, or None if it doesn't match.

    Example:
        >>> parse_dlq_topic("onex.dlq.intents.v1")
        {'prefix': 'onex', 'category': 'intents', 'version': 'v1'}
        >>> parse_dlq_topic("onex.dlq.events.v2")
        {'prefix': 'onex', 'category': 'events', 'version': 'v2'}
        >>> parse_dlq_topic("not.a.dlq.topic")
        None
    """
    match = DLQ_TOPIC_PATTERN.match(topic)
    if not match:
        return None

    return {
        "prefix": match.group("prefix"),
        "category": match.group("category"),
        "version": match.group("version"),
    }


def is_dlq_topic(topic: str) -> bool:
    """Check if a topic name is a DLQ topic.

    Args:
        topic: The topic name to check.

    Returns:
        True if the topic matches the DLQ naming pattern, False otherwise.

    Example:
        >>> is_dlq_topic("onex.dlq.intents.v1")
        True
        >>> is_dlq_topic("onex.evt.platform.node-registered.v1")
        False
    """
    return DLQ_TOPIC_PATTERN.match(topic) is not None


def get_dlq_topic_for_original(
    original_topic: str,
) -> str | None:
    """Get the DLQ topic for an original message topic.

    Infers the appropriate DLQ topic based on the category of the original
    topic. If it follows ONEX naming conventions, the category is extracted
    automatically. DLQ topics are realm-agnostic (always ``onex.dlq.*``).

    Args:
        original_topic: The original topic where the message was consumed from.

    Returns:
        The DLQ topic name, or None if the category cannot be determined.

    Example:
        >>> get_dlq_topic_for_original("onex.evt.platform.node-registered.v1")
        'onex.dlq.events.v1'
        >>> get_dlq_topic_for_original("onex.cmd.intent-classified.v1")
        'onex.dlq.commands.v1'

    .. versionchanged:: 0.21.0
        OMN-5189: Removed ``environment`` parameter. DLQ topics are
        realm-agnostic.
    """
    # Import here to avoid circular imports
    from omnibase_infra.enums import EnumMessageCategory

    # Try to infer category from topic
    category = EnumMessageCategory.from_topic(original_topic)
    if category is None:
        return None

    return build_dlq_topic(category.topic_suffix)


def derive_dlq_topic_for_event_type(
    event_type: str | None,
    original_topic: str,
) -> str | None:
    """Derive the DLQ topic for an unroutable message based on its event_type.

    When ``MessageDispatchEngine`` finds no registered dispatcher for an envelope,
    this function determines which DLQ topic the message should be routed to.
    All DLQ topics are realm-agnostic (``onex.dlq.*``).

    The DLQ category is derived from the event_type domain prefix:

    - ``intelligence.*`` -> ``onex.dlq.intelligence.v1``
    - ``platform.*`` -> ``onex.dlq.platform.v1``
    - ``agent.*`` -> ``onex.dlq.agent.v1``

    For messages with no event_type (Phase 1 legacy), the function falls back
    to the existing topic-based DLQ routing via ``get_dlq_topic_for_original()``,
    which uses the message category (events/commands/intents) from the topic name.

    Args:
        event_type: The event_type from the envelope. May be None or empty for
            legacy messages that don't use event_type-based routing.
        original_topic: The Kafka topic the message was consumed from. Used as
            fallback for legacy DLQ routing when event_type is absent.

    Returns:
        The DLQ topic name (e.g., ``onex.dlq.intelligence.v1``), or None if
        neither event_type nor topic-based DLQ routing can determine a target.

    Example:
        >>> derive_dlq_topic_for_event_type(
        ...     "intelligence.code-analysis-completed.v1",
        ...     "onex.evt.intelligence.code-analysis.v1",
        ... )
        'onex.dlq.intelligence.v1'
        >>> derive_dlq_topic_for_event_type(
        ...     "platform.node-registered.v1",
        ...     "onex.evt.platform.node-registration.v1",
        ... )
        'onex.dlq.platform.v1'
        >>> derive_dlq_topic_for_event_type(
        ...     None,
        ...     "onex.evt.platform.node-registration.v1",
        ... )
        'onex.dlq.events.v1'

    .. versionadded:: 0.7.0
        Added for DLQ routing of unknown event_type (OMN-2040).

    .. versionchanged:: 0.21.0
        OMN-5189: Removed ``environment`` parameter. DLQ topics are
        realm-agnostic.
    """
    # Normalize event_type
    normalized = str(event_type).strip() if event_type is not None else ""

    if normalized:
        # Extract domain prefix: first segment before the first '.'
        dot_index = normalized.find(".")
        if dot_index > 0:
            domain = normalized[:dot_index].lower()
        else:
            # Single-segment event_type (no dots) - use the whole string as domain
            domain = normalized.lower()

        # Validate domain is a valid category identifier
        if _DLQ_CATEGORY_PATTERN.match(domain):
            return build_dlq_topic(domain)

        # Domain prefix is invalid (e.g., starts with digit) — cannot
        # determine DLQ topic from event_type.  Return None rather than
        # falling back to topic-based routing, because the presence of an
        # event_type indicates the new routing model where the domain prefix
        # is authoritative.
        return None

    # Legacy path: no event_type, use topic-based DLQ routing.
    return get_dlq_topic_for_original(original_topic)


# ---------------------------------------------------------------------------
# Session Coordination Topics (OMN-6854)
# ---------------------------------------------------------------------------

TOPIC_SESSION_COORDINATION_SIGNAL: Final[str] = (
    "onex.evt.omniclaude.session-coordination-signal.v1"
)
"""Topic for session coordination signals between concurrent sessions."""

TOPIC_SESSION_STATUS_CHANGED: Final[str] = (
    "onex.evt.omniclaude.session-status-changed.v1"
)
"""Topic for session status change notifications."""

# ---------------------------------------------------------------------------
# Eval Pipeline Topics (OMN-6798)
# ---------------------------------------------------------------------------

TOPIC_EVAL_COMPLETED: Final[str] = "onex.evt.omnibase-infra.eval-completed.v1"
"""Evaluation pipeline completed event.

Published by ServiceAutoEvalRunner after each eval task completes.
Carries eval_id, model_id, pass/fail counts, and overall verdict.

Producer: ServiceAutoEvalRunner (OMN-6796)
Consumer: omnidash eval dashboard, observability
Ticket: OMN-6798
"""


# ---------------------------------------------------------------------------
# Delegation Pipeline Topics (OMN-7040)
# ---------------------------------------------------------------------------

TOPIC_DELEGATION_REQUEST: Final[str] = "onex.cmd.omnibase-infra.delegation-request.v1"
"""Command topic for delegation requests from /delegate skill."""

TOPIC_DELEGATION_ROUTING_DECISION: Final[str] = (
    "onex.evt.omnibase-infra.routing-decision.v1"
)
"""Event topic for routing decisions from the delegation routing reducer."""

TOPIC_DELEGATION_COMPLETED: Final[str] = (
    "onex.evt.omnibase-infra.delegation-completed.v1"
)
"""Event topic for successful delegation completions."""

TOPIC_DELEGATION_FAILED: Final[str] = "onex.evt.omnibase-infra.delegation-failed.v1"
"""Event topic for failed delegation attempts."""

TOPIC_DELEGATION_QUALITY_GATE_RESULT: Final[str] = (
    "onex.evt.omnibase-infra.quality-gate-result.v1"
)
"""Event topic for quality gate evaluation results."""

TOPIC_DELEGATION_ROUTING_REQUEST: Final[str] = (
    "onex.cmd.omnibase-infra.delegation-routing-request.v1"
)
"""Command topic for routing reducer invocation from the delegation orchestrator."""

TOPIC_DELEGATION_QUALITY_GATE_REQUEST: Final[str] = (
    "onex.cmd.omnibase-infra.delegation-quality-gate-request.v1"
)
"""Command topic for quality gate reducer invocation from the delegation orchestrator."""

TOPIC_DELEGATION_INFERENCE_REQUEST: Final[str] = (
    "onex.cmd.omnibase-infra.delegation-inference-request.v1"
)
"""Command topic for LLM inference invocation from the delegation orchestrator."""

TOPIC_DELEGATION_INFERENCE_RESPONSE: Final[str] = (
    "onex.evt.omnibase-infra.inference-response.v1"
)
"""Event topic for LLM inference responses in the delegation pipeline."""

TOPIC_DELEGATION_TASK_DELEGATED: Final[str] = "onex.evt.omniclaude.task-delegated.v1"
"""Backward-compatible event topic for omnidash delegation projection."""

TOPIC_DELEGATION_BASELINE_COMPARISON: Final[str] = (
    "onex.cmd.omnibase-infra.baseline-comparison-request.v1"
)
"""Command topic for baseline comparison compute from the delegation orchestrator."""

__all__ = [
    "TOPIC_DELEGATION_COMPLETED",
    "TOPIC_DELEGATION_FAILED",
    "TOPIC_DELEGATION_INFERENCE_REQUEST",
    "TOPIC_DELEGATION_INFERENCE_RESPONSE",
    "TOPIC_DELEGATION_QUALITY_GATE_REQUEST",
    "TOPIC_DELEGATION_QUALITY_GATE_RESULT",
    "TOPIC_DELEGATION_REQUEST",
    "TOPIC_DELEGATION_ROUTING_DECISION",
    "TOPIC_EVAL_COMPLETED",
    "DLQ_CATEGORY_SUFFIXES",
    "DLQ_COMMAND_TOPIC_SUFFIX",
    "DLQ_DOMAIN",
    "DLQ_EVENT_TOPIC_SUFFIX",
    "DLQ_INTENT_TOPIC_SUFFIX",
    "DLQ_TOPIC_PATTERN",
    "DLQ_TOPIC_VERSION",
    "_DLQ_PREFIX",
    "build_dlq_topic",
    "derive_dlq_topic_for_event_type",
    "get_dlq_topic_for_original",
    "is_dlq_topic",
    "parse_dlq_topic",
]
