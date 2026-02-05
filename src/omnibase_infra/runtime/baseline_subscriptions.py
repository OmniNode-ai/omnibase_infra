# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Baseline Subscriptions Assembly Module.

Aggregates platform baseline topic constants from omnibase_core for use
by the ONEX runtime when wiring Kafka subscriptions.

Part of OMN-1696: Wire baseline topic constants from omnibase_core into
omnibase_infra runtime subscription assembly.

Platform Baseline Topics:
    These topics are automatically subscribed to by the runtime during
    initialization, providing core platform functionality:

    - **Contract Registered**: Notifies when a new handler contract is registered.
      Used by KafkaContractSource for dynamic contract discovery.

    - **Contract Deregistered**: Notifies when a handler contract is removed.
      Used by KafkaContractSource for cache invalidation.

    - **Node Heartbeat**: Periodic health signals from running nodes.
      Used for liveness monitoring and service discovery.

Usage:
    >>> from omnibase_infra.runtime.baseline_subscriptions import (
    ...     BASELINE_CONTRACT_TOPICS,
    ...     BASELINE_PLATFORM_TOPICS,
    ...     get_baseline_topics,
    ... )
    >>>
    >>> # Get contract-only topics (registration/deregistration)
    >>> contract_topics = get_baseline_topics(include_heartbeat=False)
    >>>
    >>> # Get all platform baseline topics (including heartbeat)
    >>> all_topics = get_baseline_topics(include_heartbeat=True)

Related:
    - KafkaContractSource: Consumes contract registration events
    - RuntimeHostProcess: Wires baseline subscriptions on startup
    - omnibase_core.constants: Source of truth for topic suffix constants

Related Tickets:
    - OMN-1696: Wire baseline topic constants from omnibase_core
    - OMN-1652: Define platform baseline topic suffixes in omnibase_core
    - OMN-1654: KafkaContractSource (cache + discovery)

.. versionadded:: 0.2.9
    Created as part of OMN-1696 baseline subscription wiring.
"""

from __future__ import annotations

from omnibase_core.constants import (
    PLATFORM_BASELINE_TOPIC_SUFFIXES,
    TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
    TOPIC_SUFFIX_CONTRACT_REGISTERED,
    TOPIC_SUFFIX_NODE_HEARTBEAT,
)

# Contract registration/deregistration topics only.
# Use when you need to track contract lifecycle events but not node health.
BASELINE_CONTRACT_TOPICS: frozenset[str] = frozenset(
    {
        TOPIC_SUFFIX_CONTRACT_REGISTERED,
        TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
    }
)
"""Contract lifecycle topics (registration and deregistration).

This subset excludes heartbeat topics and is appropriate when:
    - You only need to track contract changes
    - Heartbeat processing is handled separately
    - You want to minimize subscription overhead

Note:
    Topics are realm-agnostic in ONEX. The environment/realm is enforced via
    envelope identity, not topic naming. Subscribe directly to the topic suffix.

Example:
    >>> for topic_suffix in BASELINE_CONTRACT_TOPICS:
    ...     subscribe(topic_suffix)  # No environment prefix needed
"""

# All platform baseline topics including heartbeat.
# Use when you need complete platform observability.
BASELINE_PLATFORM_TOPICS: frozenset[str] = frozenset(PLATFORM_BASELINE_TOPIC_SUFFIXES)
"""All platform baseline topics including heartbeat.

This is the full set of platform-reserved topics that the runtime
subscribes to automatically during initialization.

Includes:
    - Contract registered events
    - Contract deregistered events
    - Node heartbeat events

Note:
    Topics are realm-agnostic in ONEX. The environment/realm is enforced via
    envelope identity, not topic naming. Subscribe directly to the topic suffix.

Example:
    >>> for topic_suffix in BASELINE_PLATFORM_TOPICS:
    ...     subscribe(topic_suffix)  # No environment prefix needed
"""


def get_baseline_topics(*, include_heartbeat: bool = True) -> frozenset[str]:
    """Return the appropriate set of baseline topic suffixes.

    This helper function provides a convenient way to select between
    the full platform topic set or contract-only topics based on
    runtime requirements.

    Args:
        include_heartbeat: If True (default), returns all platform baseline
            topics including node heartbeat. If False, returns only contract
            registration/deregistration topics.

    Returns:
        A frozenset of topic suffix strings. Topics are realm-agnostic in ONEX;
        subscribe directly to these suffixes without environment prefix.
        The environment/realm is enforced via envelope identity, not topic naming.

    Example:
        >>> # For full platform observability
        >>> topics = get_baseline_topics(include_heartbeat=True)
        >>> assert TOPIC_SUFFIX_NODE_HEARTBEAT in topics
        >>>
        >>> # For contract-only tracking
        >>> topics = get_baseline_topics(include_heartbeat=False)
        >>> assert TOPIC_SUFFIX_NODE_HEARTBEAT not in topics
        >>> assert TOPIC_SUFFIX_CONTRACT_REGISTERED in topics

    Note:
        The returned frozenset is immutable and can be safely shared
        across threads without synchronization.

    .. versionadded:: 0.2.9
        Created as part of OMN-1696.
    """
    if include_heartbeat:
        return BASELINE_PLATFORM_TOPICS
    return BASELINE_CONTRACT_TOPICS


__all__: list[str] = [
    # Topic sets
    "BASELINE_CONTRACT_TOPICS",
    "BASELINE_PLATFORM_TOPICS",
    # Helper function
    "get_baseline_topics",
    # Re-exported constants from omnibase_core for convenience
    "PLATFORM_BASELINE_TOPIC_SUFFIXES",
    "TOPIC_SUFFIX_CONTRACT_DEREGISTERED",
    "TOPIC_SUFFIX_CONTRACT_REGISTERED",
    "TOPIC_SUFFIX_NODE_HEARTBEAT",
]
