# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for baseline_subscriptions module.

Tests the baseline topic constants and helper functions that aggregate
platform-reserved topics from omnibase_core. These constants serve as
the single source of truth for runtime subscription wiring.

Related:
    - OMN-1696: Wire baseline topic constants from omnibase_core into
      omnibase_infra runtime layer
    - src/omnibase_infra/runtime/baseline_subscriptions.py

Expected Behavior:
    - BASELINE_CONTRACT_TOPICS: frozenset containing contract registration and
      deregistration topic suffixes (immutable, 2 items)
    - BASELINE_PLATFORM_TOPICS: frozenset containing all platform-reserved topic
      suffixes including heartbeat (3 items)
    - get_baseline_topics(include_heartbeat=True/False): Helper to select between
      full platform topics or contract-only topics
"""

from __future__ import annotations

import pytest

from omnibase_core.constants import (
    TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
    TOPIC_SUFFIX_CONTRACT_REGISTERED,
    TOPIC_SUFFIX_NODE_HEARTBEAT,
)
from omnibase_infra.runtime.baseline_subscriptions import (
    BASELINE_CONTRACT_TOPICS,
    BASELINE_PLATFORM_TOPICS,
    get_baseline_topics,
)

# =============================================================================
# BASELINE_CONTRACT_TOPICS Tests
# =============================================================================


class TestBaselineContractTopics:
    """Tests for BASELINE_CONTRACT_TOPICS constant."""

    def test_contains_exactly_two_items(self) -> None:
        """BASELINE_CONTRACT_TOPICS should contain exactly 2 topic suffixes.

        The contract topics collection includes only the registration and
        deregistration topic suffixes used for handler contract discovery.
        """
        assert len(BASELINE_CONTRACT_TOPICS) == 2

    def test_contains_contract_registered_topic(self) -> None:
        """BASELINE_CONTRACT_TOPICS should include contract-registered suffix.

        The contract-registered topic is used by KafkaContractSource to discover
        new handler contracts when they are registered in the system.
        """
        assert TOPIC_SUFFIX_CONTRACT_REGISTERED in BASELINE_CONTRACT_TOPICS

    def test_contains_contract_deregistered_topic(self) -> None:
        """BASELINE_CONTRACT_TOPICS should include contract-deregistered suffix.

        The contract-deregistered topic is used by KafkaContractSource to remove
        handler contracts from the cache when they are deregistered.
        """
        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED in BASELINE_CONTRACT_TOPICS

    def test_is_frozenset_type(self) -> None:
        """BASELINE_CONTRACT_TOPICS should be a frozenset for immutability.

        Frozensets are used to ensure the topic collection cannot be modified
        at runtime while allowing efficient membership testing.
        """
        assert isinstance(BASELINE_CONTRACT_TOPICS, frozenset)

    def test_is_immutable(self) -> None:
        """BASELINE_CONTRACT_TOPICS should be immutable.

        Frozensets do not support add/remove operations, ensuring the constant
        cannot be modified at runtime.
        """
        with pytest.raises(AttributeError):
            BASELINE_CONTRACT_TOPICS.add("modified")  # type: ignore[attr-defined]

    def test_does_not_contain_heartbeat(self) -> None:
        """BASELINE_CONTRACT_TOPICS should NOT include heartbeat topic.

        Contract topics are a subset - heartbeat is only in platform topics.
        """
        assert TOPIC_SUFFIX_NODE_HEARTBEAT not in BASELINE_CONTRACT_TOPICS

    def test_values_match_omnibase_core_constants(self) -> None:
        """Topic suffixes should match canonical values from omnibase_core.

        The baseline_subscriptions module re-exports constants from omnibase_core
        to ensure single source of truth for topic naming.
        """
        expected = frozenset(
            {TOPIC_SUFFIX_CONTRACT_REGISTERED, TOPIC_SUFFIX_CONTRACT_DEREGISTERED}
        )
        assert expected == BASELINE_CONTRACT_TOPICS

    def test_topic_format_contains_expected_substrings(self) -> None:
        """Topic suffixes should contain expected substrings for clarity.

        Topic names should be self-documenting with clear domain indicators.
        """
        topics_str = " ".join(BASELINE_CONTRACT_TOPICS)
        assert "contract" in topics_str.lower()

    def test_all_items_are_strings(self) -> None:
        """All items in BASELINE_CONTRACT_TOPICS should be strings."""
        for topic in BASELINE_CONTRACT_TOPICS:
            assert isinstance(topic, str)
            assert len(topic) > 0


# =============================================================================
# BASELINE_PLATFORM_TOPICS Tests
# =============================================================================


class TestBaselinePlatformTopics:
    """Tests for BASELINE_PLATFORM_TOPICS constant."""

    def test_contains_exactly_three_items(self) -> None:
        """BASELINE_PLATFORM_TOPICS should contain exactly 3 topic suffixes.

        Platform topics include: contract-registered, contract-deregistered,
        and node-heartbeat.
        """
        assert len(BASELINE_PLATFORM_TOPICS) == 3

    def test_is_superset_of_contract_topics(self) -> None:
        """BASELINE_PLATFORM_TOPICS should include all contract topics.

        Platform topics is a superset that includes contract topics plus
        the node heartbeat topic.
        """
        for topic in BASELINE_CONTRACT_TOPICS:
            assert topic in BASELINE_PLATFORM_TOPICS

    def test_contains_contract_registered_topic(self) -> None:
        """BASELINE_PLATFORM_TOPICS should include contract-registered suffix."""
        assert TOPIC_SUFFIX_CONTRACT_REGISTERED in BASELINE_PLATFORM_TOPICS

    def test_contains_contract_deregistered_topic(self) -> None:
        """BASELINE_PLATFORM_TOPICS should include contract-deregistered suffix."""
        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED in BASELINE_PLATFORM_TOPICS

    def test_contains_node_heartbeat_topic(self) -> None:
        """BASELINE_PLATFORM_TOPICS should include node-heartbeat suffix.

        The heartbeat topic is what distinguishes platform topics from
        contract-only topics.
        """
        assert TOPIC_SUFFIX_NODE_HEARTBEAT in BASELINE_PLATFORM_TOPICS

    def test_is_frozenset_type(self) -> None:
        """BASELINE_PLATFORM_TOPICS should be a frozenset for immutability."""
        assert isinstance(BASELINE_PLATFORM_TOPICS, frozenset)

    def test_is_immutable(self) -> None:
        """BASELINE_PLATFORM_TOPICS should be immutable.

        Frozensets do not support add/remove operations, ensuring the constant
        cannot be modified at runtime.
        """
        with pytest.raises(AttributeError):
            BASELINE_PLATFORM_TOPICS.add("modified")  # type: ignore[attr-defined]

    def test_all_items_are_strings(self) -> None:
        """All items in BASELINE_PLATFORM_TOPICS should be strings."""
        for topic in BASELINE_PLATFORM_TOPICS:
            assert isinstance(topic, str)
            assert len(topic) > 0

    def test_exactly_one_more_than_contract_topics(self) -> None:
        """Platform topics should have exactly one more topic than contract topics.

        The additional topic is the node heartbeat.
        """
        assert len(BASELINE_PLATFORM_TOPICS) == len(BASELINE_CONTRACT_TOPICS) + 1


# =============================================================================
# get_baseline_topics() Tests
# =============================================================================


class TestGetBaselineTopics:
    """Tests for get_baseline_topics() helper function."""

    def test_returns_frozenset(self) -> None:
        """get_baseline_topics should return a frozenset of topic names."""
        result = get_baseline_topics()
        assert isinstance(result, frozenset)

    def test_include_heartbeat_true_returns_platform_topics(self) -> None:
        """get_baseline_topics(include_heartbeat=True) should return platform topics.

        When heartbeat is included, the full set of platform topics is returned.
        """
        result = get_baseline_topics(include_heartbeat=True)
        assert result == BASELINE_PLATFORM_TOPICS
        assert result is BASELINE_PLATFORM_TOPICS

    def test_include_heartbeat_false_returns_contract_topics(self) -> None:
        """get_baseline_topics(include_heartbeat=False) should return contract topics.

        When heartbeat is excluded, only contract lifecycle topics are returned.
        """
        result = get_baseline_topics(include_heartbeat=False)
        assert result == BASELINE_CONTRACT_TOPICS
        assert result is BASELINE_CONTRACT_TOPICS

    def test_default_includes_heartbeat(self) -> None:
        """get_baseline_topics() with no args should include heartbeat by default.

        The default behavior is to return all platform topics including heartbeat.
        """
        result = get_baseline_topics()
        assert TOPIC_SUFFIX_NODE_HEARTBEAT in result
        assert len(result) == 3

    def test_include_heartbeat_true_has_three_items(self) -> None:
        """With include_heartbeat=True, result should have 3 items."""
        result = get_baseline_topics(include_heartbeat=True)
        assert len(result) == 3

    def test_include_heartbeat_false_has_two_items(self) -> None:
        """With include_heartbeat=False, result should have 2 items."""
        result = get_baseline_topics(include_heartbeat=False)
        assert len(result) == 2

    def test_include_heartbeat_true_contains_heartbeat(self) -> None:
        """With include_heartbeat=True, result should contain heartbeat topic."""
        result = get_baseline_topics(include_heartbeat=True)
        assert TOPIC_SUFFIX_NODE_HEARTBEAT in result

    def test_include_heartbeat_false_excludes_heartbeat(self) -> None:
        """With include_heartbeat=False, result should NOT contain heartbeat topic."""
        result = get_baseline_topics(include_heartbeat=False)
        assert TOPIC_SUFFIX_NODE_HEARTBEAT not in result

    def test_both_options_contain_contract_registered(self) -> None:
        """Both options should contain contract-registered topic."""
        with_heartbeat = get_baseline_topics(include_heartbeat=True)
        without_heartbeat = get_baseline_topics(include_heartbeat=False)

        assert TOPIC_SUFFIX_CONTRACT_REGISTERED in with_heartbeat
        assert TOPIC_SUFFIX_CONTRACT_REGISTERED in without_heartbeat

    def test_both_options_contain_contract_deregistered(self) -> None:
        """Both options should contain contract-deregistered topic."""
        with_heartbeat = get_baseline_topics(include_heartbeat=True)
        without_heartbeat = get_baseline_topics(include_heartbeat=False)

        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED in with_heartbeat
        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED in without_heartbeat

    def test_keyword_only_argument(self) -> None:
        """include_heartbeat should be a keyword-only argument.

        This enforces explicit usage and prevents positional argument confusion.
        """
        # This should work - keyword argument
        result = get_baseline_topics(include_heartbeat=True)
        assert len(result) == 3

        # This should fail - positional argument
        with pytest.raises(TypeError):
            get_baseline_topics(True)  # type: ignore[misc]

    def test_result_is_immutable(self) -> None:
        """Returned frozenset should be immutable."""
        result = get_baseline_topics()
        with pytest.raises(AttributeError):
            result.add("test")  # type: ignore[attr-defined]


# =============================================================================
# Module Export Tests
# =============================================================================


class TestBaselineSubscriptionsExports:
    """Tests for module exports from runtime."""

    def test_baseline_contract_topics_exported_from_runtime(self) -> None:
        """BASELINE_CONTRACT_TOPICS should be exported from runtime module."""
        from omnibase_infra.runtime import (
            BASELINE_CONTRACT_TOPICS as EXPORTED_CONTRACT_TOPICS,
        )

        assert EXPORTED_CONTRACT_TOPICS is BASELINE_CONTRACT_TOPICS

    def test_baseline_platform_topics_exported_from_runtime(self) -> None:
        """BASELINE_PLATFORM_TOPICS should be exported from runtime module."""
        from omnibase_infra.runtime import (
            BASELINE_PLATFORM_TOPICS as EXPORTED_PLATFORM_TOPICS,
        )

        assert EXPORTED_PLATFORM_TOPICS is BASELINE_PLATFORM_TOPICS

    def test_get_baseline_topics_exported_from_runtime(self) -> None:
        """get_baseline_topics should be exported from runtime module."""
        from omnibase_infra.runtime import get_baseline_topics as exported_func

        assert exported_func is get_baseline_topics

    def test_exports_listed_in_module_all(self) -> None:
        """Module exports should be listed in __all__."""
        from omnibase_infra.runtime import baseline_subscriptions

        assert "BASELINE_CONTRACT_TOPICS" in baseline_subscriptions.__all__
        assert "BASELINE_PLATFORM_TOPICS" in baseline_subscriptions.__all__
        assert "get_baseline_topics" in baseline_subscriptions.__all__

    def test_exports_available_in_runtime_all(self) -> None:
        """Baseline exports should be available in runtime __all__."""
        from omnibase_infra import runtime

        assert "BASELINE_CONTRACT_TOPICS" in runtime.__all__
        assert "BASELINE_PLATFORM_TOPICS" in runtime.__all__
        assert "get_baseline_topics" in runtime.__all__

    def test_reexported_constants_available(self) -> None:
        """Re-exported omnibase_core constants should be available."""
        from omnibase_infra.runtime import baseline_subscriptions

        # These are re-exported for convenience
        assert "TOPIC_SUFFIX_CONTRACT_REGISTERED" in baseline_subscriptions.__all__
        assert "TOPIC_SUFFIX_CONTRACT_DEREGISTERED" in baseline_subscriptions.__all__
        assert "TOPIC_SUFFIX_NODE_HEARTBEAT" in baseline_subscriptions.__all__

    def test_platform_baseline_topic_suffixes_reexported(self) -> None:
        """PLATFORM_BASELINE_TOPIC_SUFFIXES should be re-exported from omnibase_core.

        This constant is the canonical source of all platform baseline topic
        suffixes and is re-exported for convenience. It should contain all
        three baseline topic suffixes.
        """
        from omnibase_infra.runtime import baseline_subscriptions
        from omnibase_infra.runtime.baseline_subscriptions import (
            PLATFORM_BASELINE_TOPIC_SUFFIXES,
        )

        # Verify it's listed in __all__
        assert "PLATFORM_BASELINE_TOPIC_SUFFIXES" in baseline_subscriptions.__all__

        # Verify it's a collection type (tuple, frozenset, or list)
        assert isinstance(PLATFORM_BASELINE_TOPIC_SUFFIXES, (tuple, frozenset, list)), (
            f"Expected collection type, got {type(PLATFORM_BASELINE_TOPIC_SUFFIXES)}"
        )

        # Verify it contains exactly 3 topic suffixes
        assert len(PLATFORM_BASELINE_TOPIC_SUFFIXES) == 3

        # Verify it contains the expected topic suffixes
        assert TOPIC_SUFFIX_CONTRACT_REGISTERED in PLATFORM_BASELINE_TOPIC_SUFFIXES
        assert TOPIC_SUFFIX_CONTRACT_DEREGISTERED in PLATFORM_BASELINE_TOPIC_SUFFIXES
        assert TOPIC_SUFFIX_NODE_HEARTBEAT in PLATFORM_BASELINE_TOPIC_SUFFIXES


# =============================================================================
# Integration with omnibase_core Constants Tests
# =============================================================================


class TestOmnibaseCoreIntegration:
    """Tests verifying integration with omnibase_core constants."""

    def test_contract_registered_suffix_format(self) -> None:
        """TOPIC_SUFFIX_CONTRACT_REGISTERED should follow expected format.

        Topic suffix should contain version indicator and descriptive name.
        """
        assert isinstance(TOPIC_SUFFIX_CONTRACT_REGISTERED, str)
        assert len(TOPIC_SUFFIX_CONTRACT_REGISTERED) > 0
        # Should contain "contract" or "registered" keyword
        suffix_lower = TOPIC_SUFFIX_CONTRACT_REGISTERED.lower()
        assert "contract" in suffix_lower or "registered" in suffix_lower

    def test_contract_deregistered_suffix_format(self) -> None:
        """TOPIC_SUFFIX_CONTRACT_DEREGISTERED should follow expected format.

        Topic suffix should contain version indicator and descriptive name.
        """
        assert isinstance(TOPIC_SUFFIX_CONTRACT_DEREGISTERED, str)
        assert len(TOPIC_SUFFIX_CONTRACT_DEREGISTERED) > 0
        # Should contain "contract" or "deregistered" keyword
        suffix_lower = TOPIC_SUFFIX_CONTRACT_DEREGISTERED.lower()
        assert "contract" in suffix_lower or "deregistered" in suffix_lower

    def test_node_heartbeat_suffix_format(self) -> None:
        """TOPIC_SUFFIX_NODE_HEARTBEAT should follow expected format.

        Topic suffix should contain version indicator and descriptive name.
        """
        assert isinstance(TOPIC_SUFFIX_NODE_HEARTBEAT, str)
        assert len(TOPIC_SUFFIX_NODE_HEARTBEAT) > 0
        # Should contain "heartbeat" or "node" keyword
        suffix_lower = TOPIC_SUFFIX_NODE_HEARTBEAT.lower()
        assert "heartbeat" in suffix_lower or "node" in suffix_lower

    def test_suffixes_are_distinct(self) -> None:
        """All topic suffixes should be distinct from each other."""
        suffixes = {
            TOPIC_SUFFIX_CONTRACT_REGISTERED,
            TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
            TOPIC_SUFFIX_NODE_HEARTBEAT,
        }
        assert len(suffixes) == 3  # No duplicates

    def test_no_environment_prefix_in_suffixes(self) -> None:
        """Topic suffixes should not contain environment prefixes.

        Environment prefix is added by runtime wiring, not in suffix.
        """
        for suffix in BASELINE_PLATFORM_TOPICS:
            assert not suffix.startswith("dev.")
            assert not suffix.startswith("staging.")
            assert not suffix.startswith("prod.")
            assert not suffix.startswith("test.")

    def test_suffixes_contain_version_indicator(self) -> None:
        """Topic suffixes should contain a version indicator (e.g., 'v1').

        ONEX topic naming convention includes version in the suffix.
        """
        for suffix in BASELINE_PLATFORM_TOPICS:
            # Should contain version pattern like "v1", "v2", etc.
            assert ".v" in suffix.lower() or "v1" in suffix.lower()
