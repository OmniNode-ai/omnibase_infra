# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Concrete implementation of ProtocolTopicRegistry.

Maps logical topic keys to concrete Kafka topic strings. The
``from_defaults()`` factory creates a registry pre-populated with
all canonical topic strings from the current platform.

Usage:
    >>> from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry
    >>> from omnibase_infra.topics import topic_keys
    >>>
    >>> registry = ServiceTopicRegistry.from_defaults()
    >>> topic = registry.resolve(topic_keys.RESOLUTION_DECIDED)
    >>> print(topic)
    onex.evt.platform.resolution-decided.v1

Related:
    - OMN-5839: Topic registry consolidation epic
    - ProtocolTopicRegistry: Protocol this class satisfies
    - topic_keys: Logical key constants

.. versionadded:: 0.24.0
"""

from __future__ import annotations

from omnibase_infra.topics import topic_keys


class ServiceTopicRegistry:
    """Concrete topic registry mapping logical keys to Kafka topic strings.

    Satisfies ``ProtocolTopicRegistry`` via structural typing.

    Args:
        topics: Mapping of logical key -> concrete Kafka topic string.
        monitored: Set of concrete topic strings monitored for wiring health.

    .. versionadded:: 0.24.0
    """

    def __init__(
        self,
        topics: dict[str, str],
        monitored: frozenset[str],
    ) -> None:
        self._topics = dict(topics)  # defensive copy
        self._monitored = monitored

    @classmethod
    def from_defaults(cls) -> ServiceTopicRegistry:
        """Build registry with all canonical topic strings.

        Values match the current ``topic_constants.py`` TOPIC_* constants.

        Returns:
            A fully populated ServiceTopicRegistry.

        .. versionadded:: 0.24.0
        """
        topics = {
            # Session
            topic_keys.SESSION_OUTCOME_CURRENT: (
                "onex.cmd.omniintelligence.session-outcome.v1"
            ),
            topic_keys.SESSION_OUTCOME_CANONICAL: (
                "onex.evt.omniclaude.session-outcome.v1"
            ),
            # Injection effectiveness
            topic_keys.INJECTION_CONTEXT_UTILIZATION: (
                "onex.evt.omniclaude.context-utilization.v1"
            ),
            topic_keys.INJECTION_AGENT_MATCH: ("onex.evt.omniclaude.agent-match.v1"),
            topic_keys.INJECTION_LATENCY_BREAKDOWN: (
                "onex.evt.omniclaude.latency-breakdown.v1"
            ),
            # LLM
            topic_keys.LLM_CALL_COMPLETED: (
                "onex.evt.omniintelligence.llm-call-completed.v1"
            ),
            topic_keys.LLM_CALL_COMPLETED_INFRA: (
                "onex.evt.omnibase-infra.llm-call-completed.v1"
            ),
            topic_keys.LLM_ENDPOINT_HEALTH: (
                "onex.evt.omnibase-infra.llm-endpoint-health.v1"
            ),
            topic_keys.LLM_INFERENCE_REQUEST: (
                "onex.cmd.omnibase-infra.llm-inference-request.v1"
            ),
            topic_keys.LLM_EMBEDDING_REQUEST: (
                "onex.cmd.omnibase-infra.llm-embedding-request.v1"
            ),
            # Effectiveness
            topic_keys.EFFECTIVENESS_INVALIDATION: (
                "onex.evt.omnibase-infra.effectiveness-data-changed.v1"
            ),
            # Agent
            topic_keys.AGENT_STATUS: "onex.evt.omniclaude.agent-status.v1",
            # Reward
            topic_keys.REWARD_ASSIGNED: ("onex.evt.omnimemory.reward-assigned.v1"),
            # Resolution
            topic_keys.RESOLUTION_DECIDED: ("onex.evt.platform.resolution-decided.v1"),
            # Circuit breaker
            topic_keys.CIRCUIT_BREAKER_STATE: (
                "onex.evt.omnibase-infra.circuit-breaker-state.v1"
            ),
            # Wiring health
            topic_keys.WIRING_HEALTH_SNAPSHOT: (
                "onex.evt.omnibase-infra.wiring-health-snapshot.v1"
            ),
            # Savings estimation
            topic_keys.SAVINGS_ESTIMATED: (
                "onex.evt.omnibase-infra.savings-estimated.v1"
            ),
            topic_keys.VALIDATOR_CATCH: ("onex.evt.omniclaude.validator-catch.v1"),
            topic_keys.HOOK_CONTEXT_INJECTED: (
                "onex.evt.omniclaude.hook-context-injected.v1"
            ),
            # Consumer health
            topic_keys.CONSUMER_HEALTH: ("onex.evt.omnibase-infra.consumer-health.v1"),
            topic_keys.CONSUMER_RESTART_CMD: (
                "onex.cmd.omnibase-infra.consumer-restart.v1"
            ),
            # Runtime error
            topic_keys.RUNTIME_ERROR: ("onex.evt.omnibase-infra.runtime-error.v1"),
            topic_keys.ERROR_TRIAGED: ("onex.evt.omnibase-infra.error-triaged.v1"),
            # Baselines
            topic_keys.BASELINES_COMPUTED: (
                "onex.evt.omnibase-infra.baselines-computed.v1"
            ),
        }

        # Wiring health monitored topics (matches WIRING_HEALTH_MONITORED_TOPICS)
        monitored = frozenset(
            {
                topics[topic_keys.SESSION_OUTCOME_CURRENT],
                topics[topic_keys.INJECTION_CONTEXT_UTILIZATION],
                topics[topic_keys.INJECTION_AGENT_MATCH],
                topics[topic_keys.INJECTION_LATENCY_BREAKDOWN],
            }
        )

        return cls(topics=topics, monitored=monitored)

    def resolve(self, topic_key: str) -> str:
        """Resolve a logical topic key to its Kafka topic string.

        Args:
            topic_key: A logical key from ``topic_keys`` module.

        Returns:
            The concrete Kafka topic string.

        Raises:
            KeyError: If ``topic_key`` is not registered.

        .. versionadded:: 0.24.0
        """
        try:
            return self._topics[topic_key]
        except KeyError:
            available = ", ".join(sorted(self._topics))
            raise KeyError(
                f"Unknown topic key '{topic_key}'. Available: {available}"
            ) from None

    def monitored_topics(self) -> frozenset[str]:
        """Return topic strings monitored for wiring health.

        Returns:
            Frozen set of concrete topic strings.

        .. versionadded:: 0.24.0
        """
        return self._monitored

    def all_keys(self) -> frozenset[str]:
        """Return all registered topic keys.

        Returns:
            Frozen set of all logical topic keys.

        .. versionadded:: 0.24.0
        """
        return frozenset(self._topics)


__all__ = ["ServiceTopicRegistry"]
