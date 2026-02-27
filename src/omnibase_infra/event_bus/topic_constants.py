# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic naming constants and utilities for ONEX event bus.

This module defines DLQ (Dead Letter Queue) topic naming conventions and
wiring health monitoring topic constants for the ONEX event-driven architecture.

IMPORTANT: ONEX event topics are realm-agnostic -- environment prefixes (dev.,
prod., etc.) must NOT appear on the wire for event routing topics. Environment
isolation is enforced via envelope identity and consumer group naming.
See ``omnibase_infra.topics.TopicResolver`` for the canonical resolution path.

DLQ topics are the sole exception: they use ``<env>.dlq.<category>.<version>``
because dead letter storage is per-environment infrastructure, not event routing.

DLQ Topic Naming:
    - **Format**: ``<env>.dlq.<category>.<version>``
    - Example: ``prod.dlq.intents.v1``, ``staging.dlq.events.v1``

    This convention ensures:
    - DLQ topics are clearly identifiable by the 'dlq' domain
    - Category (intents, events, commands) is preserved for routing analysis
    - Environment separation for multi-environment DLQ storage
    - Version control for DLQ message schema evolution

Usage:
    >>> from omnibase_infra.event_bus.topic_constants import (
    ...     build_dlq_topic,
    ...     DLQ_INTENT_TOPIC_SUFFIX,
    ... )
    >>>
    >>> # Build environment-specific DLQ topic
    >>> topic = build_dlq_topic("prod", "intents")
    >>> print(topic)
    prod.dlq.intents.v1

See Also:
    - ModelKafkaEventBusConfig.dead_letter_topic: DLQ configuration
    - EventBusKafka._publish_to_dlq(): DLQ publishing implementation
    - topic_category_validator.py: Topic naming validation
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
# Validates DLQ topics in Environment-Aware format: <env>.dlq.<category>.<version>
# - env: alphanumeric with underscores/hyphens (e.g., dev, staging, prod, test-1)
# - domain: must be 'dlq'
# - category: lowercase identifier (intents, events, commands, intelligence, platform, etc.)
# - version: v followed by digits (e.g., v1, v2)

DLQ_TOPIC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<env>[\w-]+)\.dlq\.(?P<category>[a-z][a-z0-9_-]*)\.(?P<version>v\d+)$",
    re.IGNORECASE,
)
"""
Regex pattern for validating DLQ topic names.

Groups:
    - env: Environment identifier (e.g., 'dev', 'prod', 'onex')
    - category: DLQ category (intents, events, commands, intelligence, platform, etc.)
    - version: Topic version (e.g., 'v1')

Example matches:
    - dev.dlq.intents.v1
    - prod.dlq.events.v1
    - staging.dlq.commands.v2
    - onex.dlq.intelligence.v1
    - onex.dlq.platform.v1

.. versionchanged:: 0.7.0
    Expanded category pattern from ``intents|events|commands`` to any
    lowercase identifier to support domain-based DLQ routing (OMN-2040).
"""

# ==============================================================================
# Environment Validation Pattern
# ==============================================================================
# Validates environment identifier: alphanumeric with underscores/hyphens only.
# This is the same pattern used in DLQ_TOPIC_PATTERN for the env group.

ENV_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[\w-]+$")
"""
Regex pattern for validating environment identifiers.

Valid examples: 'dev', 'prod', 'staging', 'test-1', 'my_env'
Invalid examples: 'env.name', 'env name', 'env@name', ''
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


def build_dlq_topic(
    environment: str,
    category: str,
    *,
    version: str | None = None,
) -> str:
    """Build a DLQ topic name from components.

    Constructs a Dead Letter Queue topic name following ONEX conventions
    in the Environment-Aware format: `<env>.dlq.<category>.<version>`.

    Args:
        environment: Environment identifier (e.g., 'dev', 'prod', 'staging').
            Must be alphanumeric with optional underscores or hyphens.
        category: DLQ category identifier. Accepts standard message categories
            in singular or plural form ('intent'/'intents', 'event'/'events',
            'command'/'commands') which are normalized to plural form, as well
            as domain-based categories ('intelligence', 'platform', 'agent',
            etc.) which pass through as-is.
        version: Optional topic version (e.g., 'v1', 'v2'). If not provided,
            defaults to DLQ_TOPIC_VERSION ('v1').

    Returns:
        Fully-qualified DLQ topic name.

    Raises:
        ProtocolConfigurationError: If environment is empty/whitespace, has
            invalid format, or category is invalid.

    Example:
        >>> build_dlq_topic("dev", "intents")
        'dev.dlq.intents.v1'
        >>> build_dlq_topic("prod", "intent")  # Singular form accepted
        'prod.dlq.intents.v1'
        >>> build_dlq_topic("staging", "events", version="v2")
        'staging.dlq.events.v2'
        >>> build_dlq_topic("test-env", "commands")
        'test-env.dlq.commands.v1'
        >>> build_dlq_topic("my_env", "intents")  # Underscores allowed
        'my_env.dlq.intents.v1'

    Test cases for environment validation:
        - Valid: 'dev', 'prod', 'staging', 'test-1', 'my_env', 'env123'
        - Invalid: 'env.name' (dots), 'env name' (spaces), 'env@name' (special chars)
        - Invalid: '' (empty), '   ' (whitespace only)
    """
    # Validate environment
    env = environment.strip()
    if not env:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="build_dlq_topic",
        )
        raise ProtocolConfigurationError(
            "environment cannot be empty",
            context=context,
            parameter="environment",
        )

    if not ENV_PATTERN.match(env):
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="build_dlq_topic",
        )
        raise ProtocolConfigurationError(
            f"Invalid environment '{environment}'. "
            "Must be alphanumeric with optional underscores or hyphens (pattern: [\\w-]+).",
            context=context,
            parameter="environment",
            value=environment,
        )

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

    return f"{env}.{DLQ_DOMAIN}.{normalized_category}.{topic_version}"


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

    Extracts environment, category, and version from a DLQ topic name
    that follows the ONEX naming convention.

    Args:
        topic: The DLQ topic name to parse.

    Returns:
        A dictionary with keys 'environment', 'category', and 'version'
        if the topic matches the DLQ pattern, or None if it doesn't match.

    Example:
        >>> parse_dlq_topic("dev.dlq.intents.v1")
        {'environment': 'dev', 'category': 'intents', 'version': 'v1'}
        >>> parse_dlq_topic("prod.dlq.events.v2")
        {'environment': 'prod', 'category': 'events', 'version': 'v2'}
        >>> parse_dlq_topic("not.a.dlq.topic")
        None
    """
    match = DLQ_TOPIC_PATTERN.match(topic)
    if not match:
        return None

    return {
        "environment": match.group("env"),
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
        >>> is_dlq_topic("dev.dlq.intents.v1")
        True
        >>> is_dlq_topic("dev.user.events.v1")
        False
    """
    return DLQ_TOPIC_PATTERN.match(topic) is not None


def get_dlq_topic_for_original(
    original_topic: str,
    environment: str | None = None,
) -> str | None:
    """Get the DLQ topic for an original message topic.

    Infers the appropriate DLQ topic based on the category of the original
    topic. If it follows ONEX naming conventions, the category is extracted
    automatically.

    Args:
        original_topic: The original topic where the message was consumed from.
        environment: Optional environment override. If not provided, attempts
            to extract from the original topic (Environment-Aware format only).

    Returns:
        The DLQ topic name, or None if the category cannot be determined.

    Example:
        >>> get_dlq_topic_for_original("dev.checkout.intents.v1")
        'dev.dlq.intents.v1'
        >>> get_dlq_topic_for_original("prod.order.events.v1")
        'prod.dlq.events.v1'
        >>> get_dlq_topic_for_original("onex.registration.commands")
        None  # ONEX format has no environment, must provide explicitly
        >>> get_dlq_topic_for_original("onex.registration.commands", environment="prod")
        'prod.dlq.commands.v1'
    """
    # Import here to avoid circular imports
    from omnibase_infra.enums import EnumMessageCategory

    # Try to infer category from topic
    category = EnumMessageCategory.from_topic(original_topic)
    if category is None:
        return None

    # Try to extract environment from topic if not provided
    if environment is None:
        # Environment-Aware format: <env>.<domain>.<category>.<version>
        parts = original_topic.split(".")
        if len(parts) >= 2 and parts[0].lower() not in ("onex",):
            environment = parts[0]
        else:
            # Cannot determine environment from ONEX format topics
            return None

    return build_dlq_topic(environment, category.topic_suffix)


def derive_dlq_topic_for_event_type(
    event_type: str | None,
    original_topic: str,
    *,
    environment: str = "onex",
) -> str | None:
    """Derive the DLQ topic for an unroutable message based on its event_type.

    When ``MessageDispatchEngine`` finds no registered dispatcher for an envelope,
    this function determines which DLQ topic the message should be routed to.

    The DLQ category is derived from the event_type domain prefix:

    - ``intelligence.*`` -> ``{env}.dlq.intelligence.v1``
    - ``platform.*`` -> ``{env}.dlq.platform.v1``
    - ``agent.*`` -> ``{env}.dlq.agent.v1``

    For messages with no event_type (Phase 1 legacy), the function falls back
    to the existing topic-based DLQ routing via ``get_dlq_topic_for_original()``,
    which uses the message category (events/commands/intents) from the topic name.

    Args:
        event_type: The event_type from the envelope. May be None or empty for
            legacy messages that don't use event_type-based routing.
        original_topic: The Kafka topic the message was consumed from. Used as
            fallback for legacy DLQ routing when event_type is absent.
        environment: Environment prefix for the DLQ topic when using event_type-based
            routing. Defaults to "onex". This parameter is NOT used for the legacy
            topic-based fallback path, which extracts the environment from the
            original topic instead.

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
        ...     "dev.user.events.v1",
        ... )
        'dev.dlq.events.v1'
        >>> derive_dlq_topic_for_event_type(
        ...     "",
        ...     "onex.registration.commands",
        ... )  # Returns None (ONEX-format topic, environment not extractable)

    .. versionadded:: 0.7.0
        Added for DLQ routing of unknown event_type (OMN-2040).
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
            return build_dlq_topic(environment, domain)

        # Domain prefix is invalid (e.g., starts with digit) — cannot
        # determine DLQ topic from event_type.  Return None rather than
        # falling back to topic-based routing, because the presence of an
        # event_type indicates the new routing model where the domain prefix
        # is authoritative.
        return None

    # Legacy path: no event_type, use topic-based DLQ routing.
    # Do NOT pass the environment parameter here so that
    # get_dlq_topic_for_original can extract it from the original topic.
    # This ensures "dev.user.events.v1" produces "dev.dlq.events.v1"
    # instead of using the default "onex" environment.
    return get_dlq_topic_for_original(original_topic)


# ==============================================================================
# Wiring Health Monitoring Topics
# ==============================================================================
# Topics monitored for emission/consumption health checks.
#
# NOTE: session-outcome uses dual-publish by design:
#   - cmd topic: consumed by omniintelligence for pattern feedback (triggers processing)
#   - evt topic: consumed by observability/dashboards (informational fact)
# The 'cmd' prefix is intentional because the intelligence consumer treats the
# message as a command to evaluate patterns, not merely an observable event.
#
# See: OMN-1895 - Wiring health monitor implementation

TOPIC_SESSION_OUTCOME_CURRENT: Final[str] = (
    "onex.cmd.omniintelligence.session-outcome.v1"
)
"""Session-outcome command topic for intelligence processing.

Uses 'cmd' prefix intentionally: omniintelligence treats session outcomes as
commands triggering pattern feedback evaluation. This is the dual-publish
design -- cmd for intelligence, evt for observability.

Producer: omniclaude (SessionEnd hook)
Consumer: omniintelligence/node_pattern_feedback_effect
"""

TOPIC_SESSION_OUTCOME_CANONICAL: Final[str] = (
    "onex.evt.omniintelligence.session-outcome.v1"
)
"""Session-outcome event topic for observability.

The evt counterpart of the dual-publish pair. Observability consumers
(dashboards, metrics) subscribe to this topic for session outcome facts.
"""

# Injection effectiveness topics (already correctly named with 'evt')
TOPIC_INJECTION_CONTEXT_UTILIZATION: Final[str] = (
    "onex.evt.omniclaude.context-utilization.v1"
)
"""Context utilization metrics from omniclaude injection hooks."""

TOPIC_INJECTION_AGENT_MATCH: Final[str] = "onex.evt.omniclaude.agent-match.v1"
"""Agent match metrics from omniclaude injection hooks."""

TOPIC_INJECTION_LATENCY_BREAKDOWN: Final[str] = (
    "onex.evt.omniclaude.latency-breakdown.v1"
)
"""Latency breakdown metrics from omniclaude injection hooks."""

# LLM call metrics events
TOPIC_LLM_CALL_COMPLETED: Final[str] = "onex.evt.omniintelligence.llm-call-completed.v1"
"""LLM call completed metrics event.

Producer: HandlerLlmOpenaiCompatible, HandlerLlmOllama
Consumer: omniintelligence cost aggregation pipeline
Payload: ContractLlmCallMetrics (per-call token counts, cost, latency)
"""

# Effectiveness data invalidation events (OMN-2303)
TOPIC_EFFECTIVENESS_INVALIDATION: Final[str] = (
    "onex.evt.omnibase-infra.effectiveness-data-changed.v1"
)
"""Effectiveness data invalidation events for dashboard refresh.

Producer: WriterInjectionEffectivenessPostgres, ServiceBatchComputeEffectivenessMetrics
Consumer: Dashboard WebSocket servers, API caches
Payload: ModelEffectivenessInvalidationEvent (tables_affected, rows_written, source)
"""

# Agent status events (OMN-2846: aligned with omniclaude producer)
TOPIC_AGENT_STATUS: Final[str] = "onex.evt.omniclaude.agent-status.v1"
"""Agent status events for real-time agent visibility.

Producer: omniclaude (agent status hooks)
Consumer: agent_actions consumer for persistence
Renamed: onex.evt.agent.status.v1 -> onex.evt.omniclaude.agent-status.v1 (OMN-2846)
"""

# Reward architecture topics (OMN-2552)
TOPIC_RUN_EVALUATED: Final[str] = "onex.evt.omnimemory.run-evaluated.v1"
"""Reward run evaluation summary with objective fingerprint.

Producer: NodeRewardBinderEffect
Consumer: Policy update consumers, analytics
Ticket: OMN-2552
"""

TOPIC_REWARD_ASSIGNED: Final[str] = "onex.evt.omnimemory.reward-assigned.v1"
"""Per-target reward assignment with traceable evidence refs.

Producer: NodeRewardBinderEffect
Consumer: Tool/model/pattern/agent reward consumers
Ticket: OMN-2552
"""

TOPIC_POLICY_STATE_UPDATED: Final[str] = "onex.evt.omnimemory.policy-state-updated.v1"
"""Policy state transition with old/new snapshots.

Producer: NodeRewardBinderEffect
Consumer: Policy state consumers, audit log
Ticket: OMN-2552
"""

# Resolution event ledger (OMN-2895 / Phase 6)
TOPIC_RESOLUTION_DECIDED: Final[str] = "onex.evt.platform.resolution-decided.v1"
"""Resolution decision audit events.

Published after every tiered dependency resolution decision. Records the
full tier progression, proofs attempted, and final outcome for audit,
replay, and intelligence.

Producer: ServiceResolutionEventPublisher
Consumer: Audit log, intelligence pipeline, replay infrastructure
Ticket: OMN-2895 (Phase 6 of OMN-2897 epic)
"""

# Grouped constants for wiring health monitoring
WIRING_HEALTH_MONITORED_TOPICS: Final[tuple[str, ...]] = (
    TOPIC_SESSION_OUTCOME_CURRENT,
    TOPIC_INJECTION_CONTEXT_UTILIZATION,
    TOPIC_INJECTION_AGENT_MATCH,
    TOPIC_INJECTION_LATENCY_BREAKDOWN,
)
"""Topics monitored by wiring health for emission/consumption comparison."""


__all__ = [
    "DLQ_CATEGORY_SUFFIXES",
    "DLQ_COMMAND_TOPIC_SUFFIX",
    "DLQ_DOMAIN",
    "DLQ_EVENT_TOPIC_SUFFIX",
    "DLQ_INTENT_TOPIC_SUFFIX",
    "DLQ_TOPIC_PATTERN",
    # Constants
    "DLQ_TOPIC_VERSION",
    "ENV_PATTERN",
    # Agent Status Topics
    "TOPIC_AGENT_STATUS",
    # Effectiveness Invalidation Topics
    "TOPIC_EFFECTIVENESS_INVALIDATION",
    # Reward Architecture Topics (OMN-2552)
    "TOPIC_POLICY_STATE_UPDATED",
    "TOPIC_REWARD_ASSIGNED",
    # Resolution Event Ledger (OMN-2895)
    "TOPIC_RESOLUTION_DECIDED",
    "TOPIC_RUN_EVALUATED",
    # LLM Call Metrics Topics
    "TOPIC_LLM_CALL_COMPLETED",
    # Wiring Health Topics
    "TOPIC_INJECTION_AGENT_MATCH",
    "TOPIC_INJECTION_CONTEXT_UTILIZATION",
    "TOPIC_INJECTION_LATENCY_BREAKDOWN",
    "TOPIC_SESSION_OUTCOME_CANONICAL",
    "TOPIC_SESSION_OUTCOME_CURRENT",
    "WIRING_HEALTH_MONITORED_TOPICS",
    # Functions
    "build_dlq_topic",
    "derive_dlq_topic_for_event_type",
    "get_dlq_topic_for_original",
    "is_dlq_topic",
    "parse_dlq_topic",
]
