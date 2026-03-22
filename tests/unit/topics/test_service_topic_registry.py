# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for ServiceTopicRegistry.

TDD tests written before the implementation per OMN-5842.
Validates protocol compliance, resolution, error handling, and
monitored topics.

.. versionadded:: 0.24.0
"""

from __future__ import annotations

import pytest

from omnibase_infra.protocols import ProtocolTopicRegistry
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry


@pytest.mark.unit
class TestServiceTopicRegistryProtocol:
    """Verify ServiceTopicRegistry satisfies ProtocolTopicRegistry."""

    def test_satisfies_protocol(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert isinstance(registry, ProtocolTopicRegistry)

    def test_is_not_protocol_itself(self) -> None:
        """ServiceTopicRegistry is a concrete class, not the protocol."""
        assert ServiceTopicRegistry is not ProtocolTopicRegistry


@pytest.mark.unit
class TestServiceTopicRegistryResolve:
    """Test resolve() for known and unknown keys."""

    def test_resolve_resolution_decided(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.RESOLUTION_DECIDED)
            == "onex.evt.platform.resolution-decided.v1"
        )

    def test_resolve_session_outcome_current(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.SESSION_OUTCOME_CURRENT)
            == "onex.cmd.omniintelligence.session-outcome.v1"
        )

    def test_resolve_session_outcome_canonical(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.SESSION_OUTCOME_CANONICAL)
            == "onex.evt.omniclaude.session-outcome.v1"
        )

    def test_resolve_effectiveness_invalidation(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.EFFECTIVENESS_INVALIDATION)
            == "onex.evt.omnibase-infra.effectiveness-data-changed.v1"
        )

    def test_resolve_llm_call_completed(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.LLM_CALL_COMPLETED)
            == "onex.evt.omniintelligence.llm-call-completed.v1"
        )

    def test_resolve_llm_endpoint_health(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.LLM_ENDPOINT_HEALTH)
            == "onex.evt.omnibase-infra.llm-endpoint-health.v1"
        )

    def test_resolve_agent_status(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.AGENT_STATUS)
            == "onex.evt.omniclaude.agent-status.v1"
        )

    def test_resolve_circuit_breaker_state(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.CIRCUIT_BREAKER_STATE)
            == "onex.evt.omnibase-infra.circuit-breaker-state.v1"
        )

    def test_resolve_consumer_health(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.CONSUMER_HEALTH)
            == "onex.evt.omnibase-infra.consumer-health.v1"
        )

    def test_resolve_consumer_restart_cmd(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.CONSUMER_RESTART_CMD)
            == "onex.cmd.omnibase-infra.consumer-restart.v1"
        )

    def test_resolve_runtime_error(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        assert (
            registry.resolve(topic_keys.RUNTIME_ERROR)
            == "onex.evt.omnibase-infra.runtime-error.v1"
        )

    def test_resolve_unknown_key_raises(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        with pytest.raises(KeyError, match="NONEXISTENT"):
            registry.resolve("NONEXISTENT")

    def test_resolve_unknown_key_lists_available(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        with pytest.raises(KeyError, match="Available:"):
            registry.resolve("NONEXISTENT")


@pytest.mark.unit
class TestServiceTopicRegistryMonitored:
    """Test monitored_topics() returns correct set."""

    def test_monitored_topics_returns_frozenset(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        topics = registry.monitored_topics()
        assert isinstance(topics, frozenset)

    def test_monitored_topics_count(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        topics = registry.monitored_topics()
        assert len(topics) == 4  # WIRING_HEALTH_MONITORED_TOPICS has 4 entries

    def test_monitored_topics_contains_session_outcome(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        topics = registry.monitored_topics()
        assert "onex.cmd.omniintelligence.session-outcome.v1" in topics

    def test_monitored_topics_all_start_with_onex(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        for topic in registry.monitored_topics():
            assert topic.startswith("onex."), (
                f"Monitored topic missing onex prefix: {topic}"
            )


@pytest.mark.unit
class TestServiceTopicRegistryAllKeys:
    """Test all_keys() returns complete set."""

    def test_all_keys_returns_frozenset(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        keys = registry.all_keys()
        assert isinstance(keys, frozenset)

    def test_all_keys_count(self) -> None:
        registry = ServiceTopicRegistry.from_defaults()
        keys = registry.all_keys()
        assert len(keys) == 20  # actual count from topic_constants.py

    def test_all_keys_match_topic_keys_module(self) -> None:
        """Every key in topic_keys.__all__ must be in the registry."""
        registry = ServiceTopicRegistry.from_defaults()
        registered_keys = registry.all_keys()
        for key_name in topic_keys.__all__:
            key_value = getattr(topic_keys, key_name)
            assert key_value in registered_keys, (
                f"topic_keys.{key_name} (={key_value!r}) not in registry"
            )

    def test_all_keys_resolvable(self) -> None:
        """Every registered key must resolve to a valid onex.* topic."""
        registry = ServiceTopicRegistry.from_defaults()
        for key in registry.all_keys():
            topic = registry.resolve(key)
            assert topic.startswith("onex."), (
                f"Topic for key {key!r} doesn't match naming: {topic}"
            )


@pytest.mark.unit
class TestServiceTopicRegistryCustom:
    """Test construction with custom topic mappings."""

    def test_custom_registry(self) -> None:
        registry = ServiceTopicRegistry(
            topics={"MY_KEY": "onex.evt.test.my-topic.v1"},
            monitored=frozenset(),
        )
        assert registry.resolve("MY_KEY") == "onex.evt.test.my-topic.v1"
        assert registry.all_keys() == frozenset({"MY_KEY"})
        assert registry.monitored_topics() == frozenset()
