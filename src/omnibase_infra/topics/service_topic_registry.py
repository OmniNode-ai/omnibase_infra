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

from omnibase_infra.topics import platform_topic_suffixes as topic_suffixes
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

        Values are derived from the provisioned topic suffix catalog.

        Returns:
            A fully populated ServiceTopicRegistry.

        .. versionadded:: 0.24.0
        """
        topics = {
            # Session
            topic_keys.SESSION_OUTCOME_CURRENT: (
                topic_suffixes.SUFFIX_INTELLIGENCE_SESSION_OUTCOME
            ),
            topic_keys.SESSION_OUTCOME_CANONICAL: (
                topic_suffixes.SUFFIX_OMNICLAUDE_SESSION_OUTCOME
            ),
            # Injection effectiveness
            topic_keys.INJECTION_CONTEXT_UTILIZATION: (
                topic_suffixes.SUFFIX_OMNICLAUDE_CONTEXT_UTILIZATION
            ),
            topic_keys.INJECTION_AGENT_MATCH: (
                topic_suffixes.SUFFIX_OMNICLAUDE_AGENT_MATCH
            ),
            topic_keys.INJECTION_LATENCY_BREAKDOWN: (
                topic_suffixes.SUFFIX_OMNICLAUDE_LATENCY_BREAKDOWN
            ),
            topic_keys.INJECTION_CONTEXT_ENRICHMENT: (
                topic_suffixes.SUFFIX_OMNICLAUDE_CONTEXT_ENRICHMENT
            ),
            topic_keys.INJECTION_RECORDED: (
                topic_suffixes.SUFFIX_OMNICLAUDE_INJECTION_RECORDED
            ),
            # Manifest injection lifecycle (OMN-1888)
            topic_keys.MANIFEST_INJECTION_STARTED: (
                topic_suffixes.SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_STARTED
            ),
            topic_keys.MANIFEST_INJECTED: (
                topic_suffixes.SUFFIX_OMNICLAUDE_MANIFEST_INJECTED
            ),
            topic_keys.MANIFEST_INJECTION_FAILED: (
                topic_suffixes.SUFFIX_OMNICLAUDE_MANIFEST_INJECTION_FAILED
            ),
            # LLM
            topic_keys.LLM_CALL_COMPLETED: (
                topic_suffixes.SUFFIX_INTELLIGENCE_LLM_CALL_COMPLETED
            ),
            topic_keys.DISPATCH_OUTCOME_EVALUATED: (
                topic_suffixes.SUFFIX_INTELLIGENCE_DISPATCH_OUTCOME_EVALUATED
            ),
            topic_keys.LLM_CALL_COMPLETED_INFRA: (
                topic_suffixes.SUFFIX_LLM_CALL_COMPLETED_INFRA
            ),
            topic_keys.LLM_ENDPOINT_HEALTH: (topic_suffixes.SUFFIX_LLM_ENDPOINT_HEALTH),
            topic_keys.LLM_INFERENCE_REQUEST: (
                topic_suffixes.SUFFIX_LLM_INFERENCE_REQUEST
            ),
            topic_keys.LLM_EMBEDDING_REQUEST: (
                topic_suffixes.SUFFIX_LLM_EMBEDDING_REQUEST
            ),
            topic_keys.EVAL_COMPLETED: topic_suffixes.SUFFIX_EVAL_COMPLETED,
            # Effectiveness
            topic_keys.EFFECTIVENESS_INVALIDATION: (
                topic_suffixes.SUFFIX_EFFECTIVENESS_INVALIDATION
            ),
            # Agent
            topic_keys.AGENT_STATUS: topic_suffixes.SUFFIX_OMNICLAUDE_AGENT_STATUS,
            # Reward
            topic_keys.REWARD_ASSIGNED: topic_suffixes.SUFFIX_OMNIMEMORY_REWARD_ASSIGNED,
            # Resolution
            topic_keys.RESOLUTION_DECIDED: topic_suffixes.SUFFIX_RESOLUTION_DECIDED,
            # Circuit breaker
            topic_keys.CIRCUIT_BREAKER_STATE: (
                topic_suffixes.SUFFIX_CIRCUIT_BREAKER_STATE
            ),
            # Wiring health
            topic_keys.WIRING_HEALTH_SNAPSHOT: (
                topic_suffixes.SUFFIX_WIRING_HEALTH_SNAPSHOT
            ),
            # Savings estimation
            topic_keys.SAVINGS_ESTIMATED: topic_suffixes.SUFFIX_SAVINGS_ESTIMATED,
            topic_keys.RUNNER_USAGE_RECORDED: (
                topic_suffixes.SUFFIX_RUNNER_USAGE_RECORDED
            ),
            topic_keys.VALIDATOR_CATCH: (
                topic_suffixes.SUFFIX_OMNICLAUDE_VALIDATOR_CATCH
            ),
            topic_keys.PATTERN_ENFORCEMENT: (
                topic_suffixes.SUFFIX_OMNICLAUDE_PATTERN_ENFORCEMENT
            ),
            topic_keys.HOOK_CONTEXT_INJECTED: (
                topic_suffixes.SUFFIX_OMNICLAUDE_HOOK_CONTEXT_INJECTED
            ),
            # Consumer health
            topic_keys.CONSUMER_HEALTH: topic_suffixes.SUFFIX_CONSUMER_HEALTH,
            topic_keys.CONSUMER_RESTART_CMD: (
                topic_suffixes.SUFFIX_CONSUMER_RESTART_CMD
            ),
            # Runtime health check
            topic_keys.RUNTIME_HEALTH_CHECK: (
                topic_suffixes.SUFFIX_RUNTIME_HEALTH_CHECK
            ),
            # Projection freshness SLA monitoring
            topic_keys.PROJECTION_FRESHNESS_DEGRADED: (
                topic_suffixes.SUFFIX_PROJECTION_FRESHNESS_DEGRADED
            ),
            topic_keys.PROJECTION_FRESHNESS_RECOVERED: (
                topic_suffixes.SUFFIX_PROJECTION_FRESHNESS_RECOVERED
            ),
            # Runtime error
            topic_keys.RUNTIME_ERROR: topic_suffixes.SUFFIX_RUNTIME_ERROR,
            topic_keys.ERROR_TRIAGED: topic_suffixes.TOPIC_ERROR_TRIAGED_V1,
            # Routing
            topic_keys.ROUTING_DECIDED: topic_suffixes.SUFFIX_ROUTING_DECIDED,
            # Baselines
            topic_keys.BASELINES_COMPUTED: topic_suffixes.SUFFIX_BASELINES_COMPUTED,
            # Waitlist
            topic_keys.WAITLIST_SIGNUP: topic_suffixes.SUFFIX_WAITLIST_SIGNUP,
            # Build Loop commands (OMN-5113)
            topic_keys.BUILD_LOOP_START: topic_suffixes.SUFFIX_BUILD_LOOP_START,
            topic_keys.BUILD_LOOP_CLOSEOUT: (topic_suffixes.SUFFIX_BUILD_LOOP_CLOSEOUT),
            topic_keys.BUILD_LOOP_VERIFY: topic_suffixes.SUFFIX_BUILD_LOOP_VERIFY,
            topic_keys.BUILD_LOOP_FILL: topic_suffixes.SUFFIX_BUILD_LOOP_FILL,
            topic_keys.BUILD_LOOP_CLASSIFY: (topic_suffixes.SUFFIX_BUILD_LOOP_CLASSIFY),
            topic_keys.BUILD_LOOP_BUILD: topic_suffixes.SUFFIX_BUILD_LOOP_BUILD,
            # Build Loop events (OMN-5113)
            topic_keys.BUILD_LOOP_STARTED: topic_suffixes.SUFFIX_BUILD_LOOP_STARTED,
            topic_keys.BUILD_LOOP_CLOSEOUT_COMPLETED: (
                topic_suffixes.SUFFIX_BUILD_LOOP_CLOSEOUT_COMPLETED
            ),
            topic_keys.BUILD_LOOP_VERIFY_COMPLETED: (
                topic_suffixes.SUFFIX_BUILD_LOOP_VERIFY_COMPLETED
            ),
            topic_keys.BUILD_LOOP_FILL_COMPLETED: (
                topic_suffixes.SUFFIX_BUILD_LOOP_FILL_COMPLETED
            ),
            topic_keys.BUILD_LOOP_CLASSIFY_COMPLETED: (
                topic_suffixes.SUFFIX_BUILD_LOOP_CLASSIFY_COMPLETED
            ),
            topic_keys.BUILD_LOOP_BUILD_COMPLETED: (
                topic_suffixes.SUFFIX_BUILD_LOOP_BUILD_COMPLETED
            ),
            topic_keys.BUILD_LOOP_CYCLE_COMPLETED: (
                topic_suffixes.SUFFIX_BUILD_LOOP_CYCLE_COMPLETED
            ),
            topic_keys.BUILD_LOOP_FAILED: topic_suffixes.SUFFIX_BUILD_LOOP_FAILED,
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
