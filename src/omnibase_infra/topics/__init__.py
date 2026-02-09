"""ONEX Infrastructure Topic Constants.

This module provides platform-reserved topic suffix constants for ONEX infrastructure
components. Domain services should NOT import from this module - domain topics should
be defined in domain contracts.

IMPORTANT: ONEX topics are realm-agnostic. Environment prefixes (dev., prod., etc.)
must NOT appear on the wire. Environment isolation is enforced via envelope identity
and consumer group naming, not topic prefixing. The canonical resolution path is
TopicResolver, which validates suffixes and returns them unchanged.

Exports:
    Platform topic suffix constants (e.g., SUFFIX_NODE_REGISTRATION)
    ALL_PLATFORM_SUFFIXES: Complete tuple of all platform-reserved suffixes
    TopicResolver: Canonical resolver for topic suffix -> concrete Kafka topic
    TopicResolutionError: Error raised when topic resolution fails
"""

from omnibase_infra.topics.platform_topic_suffixes import (
    ALL_PLATFORM_SUFFIXES,
    SUFFIX_FSM_STATE_TRANSITIONS,
    SUFFIX_NODE_HEARTBEAT,
    SUFFIX_NODE_INTROSPECTION,
    SUFFIX_NODE_REGISTRATION,
    SUFFIX_REGISTRATION_SNAPSHOTS,
    SUFFIX_REQUEST_INTROSPECTION,
    SUFFIX_RUNTIME_TICK,
)
from omnibase_infra.topics.topic_resolver import TopicResolutionError, TopicResolver

__all__: list[str] = [
    # Individual suffix constants
    "SUFFIX_NODE_REGISTRATION",
    "SUFFIX_NODE_INTROSPECTION",
    "SUFFIX_NODE_HEARTBEAT",
    "SUFFIX_REQUEST_INTROSPECTION",
    "SUFFIX_FSM_STATE_TRANSITIONS",
    "SUFFIX_RUNTIME_TICK",
    "SUFFIX_REGISTRATION_SNAPSHOTS",
    # Aggregate tuple
    "ALL_PLATFORM_SUFFIXES",
    # Topic resolution
    "TopicResolver",
    "TopicResolutionError",
]
