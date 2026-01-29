"""ONEX Infrastructure Topic Constants.

This module provides platform-reserved topic suffix constants for ONEX infrastructure
components. Domain services should NOT import from this module - domain topics should
be defined in domain contracts.

Exports:
    Platform topic suffix constants (e.g., SUFFIX_NODE_REGISTRATION)
    ALL_PLATFORM_SUFFIXES: Complete tuple of all platform-reserved suffixes
    build_full_topic: Compose full topic from env, namespace, and suffix
    TopicCompositionError: Error raised when topic composition fails
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
from omnibase_infra.topics.util_topic_composition import (
    MAX_NAMESPACE_LENGTH,
    TopicCompositionError,
    build_full_topic,
)

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
    # Topic composition utilities
    "build_full_topic",
    "TopicCompositionError",
    "MAX_NAMESPACE_LENGTH",
]
