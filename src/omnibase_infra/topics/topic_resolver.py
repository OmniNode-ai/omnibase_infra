# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Canonical topic resolver for ONEX infrastructure.

Formal Invariant:
    Contracts declare realm-agnostic topic suffixes. The TopicResolver is the
    single canonical function that maps topic suffix -> concrete Kafka topic.

    All scattered ``resolve_topic()`` methods across the codebase (event bus
    wiring, adapters, dispatchers, etc.) MUST delegate to this class. Direct
    pass-through logic in individual components is prohibited.

Current Behavior:
    Pass-through. Topic suffixes are returned unchanged because ONEX topics are
    realm-agnostic. The environment/realm is enforced via envelope identity and
    consumer group naming, NOT via topic name prefixing.

Future Phases:
    This class is the single extension point for realm-based routing, topic
    aliasing, or tenant-scoped topic mapping. When those features are needed,
    they are added HERE and all callers automatically benefit.

Topic Suffix Format:
    onex.<kind>.<producer>.<event-name>.v<version>

    Examples:
        onex.evt.platform.node-registration.v1
        onex.cmd.platform.request-introspection.v1

See Also:
    omnibase_core.validation.validate_topic_suffix - Suffix format validation
    omnibase_infra.topics.util_topic_composition - Full topic composition
    omnibase_infra.topics.platform_topic_suffixes - Platform-reserved suffixes
"""

from omnibase_core.errors import OnexError
from omnibase_core.validation import validate_topic_suffix


class TopicResolutionError(OnexError):
    """Raised when a topic suffix cannot be resolved to a concrete topic.

    This error indicates that the provided topic suffix does not conform to the
    ONEX topic naming convention and therefore cannot be mapped to a Kafka topic.

    Extends OnexError to follow ONEX error handling conventions.
    """


class TopicResolver:
    """Canonical resolver that maps ONEX topic suffixes to concrete Kafka topics.

    This is the single source of truth for topic name resolution in ONEX. All
    components that need to resolve a topic suffix to a concrete Kafka topic
    MUST use this class rather than implementing their own resolution logic.

    The resolver validates that the provided suffix conforms to the ONEX topic
    naming convention before returning it. Invalid suffixes are rejected with
    a ``TopicResolutionError``.

    Current behavior is pass-through (realm-agnostic topics, no environment
    prefix). The environment is enforced via consumer group naming, not topic
    names.

    Example:
        >>> resolver = TopicResolver()
        >>> resolver.resolve("onex.evt.platform.node-registration.v1")
        'onex.evt.platform.node-registration.v1'

        >>> resolver.resolve("bad-topic")
        Traceback (most recent call last):
            ...
        TopicResolutionError: Invalid topic suffix 'bad-topic': ...
    """

    def resolve(self, topic_suffix: str) -> str:
        """Resolve a topic suffix to a concrete Kafka topic name.

        Validates the suffix against the ONEX topic naming convention and
        returns the resolved topic name. Currently this is a pass-through
        (the suffix IS the topic name) because ONEX topics are realm-agnostic.

        Args:
            topic_suffix: ONEX format topic suffix
                (e.g., ``'onex.evt.platform.node-registration.v1'``)

        Returns:
            Concrete Kafka topic name. Currently identical to the input suffix.

        Raises:
            TopicResolutionError: If the suffix does not match the required
                ONEX topic format ``onex.<kind>.<producer>.<event-name>.v<n>``.
        """
        result = validate_topic_suffix(topic_suffix)
        if not result.is_valid:
            raise TopicResolutionError(
                f"Invalid topic suffix '{topic_suffix}': {result.error}"
            )
        return topic_suffix
