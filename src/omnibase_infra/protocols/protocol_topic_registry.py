# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol definition for the Topic Registry.

Defines the interface contract for topic registry implementations that map
logical topic keys to concrete Kafka topic strings. This enables DI-based
topic resolution instead of scattered global constant imports.

Design Principles:
    - Protocol-based interface for flexibility and testability
    - Runtime-checkable for isinstance() validation
    - Simple key->string mapping (not a heavyweight routing engine)
    - Separate from TopicResolver (suffix->concrete mapping for multi-bus)
    - Separate from EventRegistry (metadata injection, fingerprinting)

Ownership Boundary:
    omnibase_infra defines the protocol and provides ServiceTopicRegistry
    as the default implementation. Alternate registries (e.g., test doubles,
    externalized config) can implement this protocol.

Related:
    - OMN-5839: Topic registry consolidation epic
    - ServiceTopicRegistry: Primary implementation of this protocol
    - topic_keys: Logical key constants for resolve() calls

.. versionadded:: 0.24.0
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolTopicRegistry(Protocol):
    """Protocol for topic registry implementations.

    Defines the interface contract for registries that map logical topic
    keys to concrete Kafka topic strings. Consumers use ``resolve(key)``
    instead of importing global ``TOPIC_*`` constants.

    Example Implementation:
        .. code-block:: python

            class MyTopicRegistry:
                def resolve(self, topic_key: str) -> str:
                    return self._topics[topic_key]

                def monitored_topics(self) -> frozenset[str]:
                    return self._monitored

                def all_keys(self) -> frozenset[str]:
                    return frozenset(self._topics)

            # Verify protocol compliance
            registry: ProtocolTopicRegistry = MyTopicRegistry()

    See Also:
        - :class:`ServiceTopicRegistry`: Primary implementation
        - :mod:`topic_keys`: Logical key constants

    .. versionadded:: 0.24.0
    """

    def resolve(self, topic_key: str) -> str:
        """Resolve a logical topic key to its full Kafka topic string.

        Args:
            topic_key: A logical key from ``topic_keys`` module
                (e.g., ``topic_keys.RESOLUTION_DECIDED``).

        Returns:
            The concrete Kafka topic string
            (e.g., ``"onex.evt.platform.resolution-decided.v1"``).

        Raises:
            KeyError: If ``topic_key`` is not registered. The error message
                should include the list of available keys for debugging.

        .. versionadded:: 0.24.0
        """
        ...

    def monitored_topics(self) -> frozenset[str]:
        """Return the set of topic strings monitored for wiring health.

        These are the concrete Kafka topic strings (not keys) that the
        wiring health system tracks for emission/consumption comparison.

        Returns:
            Frozen set of concrete topic strings.

        .. versionadded:: 0.24.0
        """
        ...

    def all_keys(self) -> frozenset[str]:
        """Return all registered topic keys.

        Useful for debugging, validation, and completeness checks.

        Returns:
            Frozen set of all logical topic keys in this registry.

        .. versionadded:: 0.24.0
        """
        ...


__all__ = ["ProtocolTopicRegistry"]
