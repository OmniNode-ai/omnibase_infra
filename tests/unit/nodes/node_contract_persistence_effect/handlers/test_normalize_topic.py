# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for normalize_topic_for_storage() function.

Tests the topic normalization utility that strips environment prefixes
from topic names before storage. This enables environment-agnostic
topic routing queries and consistent topic identity across environments.

Tested Prefixes:
    - {env}. - Placeholder prefix (most common in contracts)
    - dev. - Development environment
    - prod. - Production environment
    - staging. - Staging environment
    - local. - Local development
    - test. - Test environment

Related:
    - HandlerPostgresTopicUpdate: Uses this function for topic normalization
    - OMN-1845: Implementation ticket
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_contract_persistence_effect.handlers.handler_postgres_topic_update import (
    normalize_topic_for_storage,
)


class TestNormalizeTopicForStorage:
    """Test suite for normalize_topic_for_storage() function."""

    @pytest.mark.parametrize(
        ("input_topic", "expected"),
        [
            # Placeholder prefix
            ("{env}.topic.name", "topic.name"),
            ("{env}.onex.evt.platform.v1", "onex.evt.platform.v1"),
            # Environment prefixes
            ("dev.topic.name", "topic.name"),
            ("prod.topic.name", "topic.name"),
            ("staging.topic.name", "topic.name"),
            ("local.topic.name", "topic.name"),
            ("test.topic.name", "topic.name"),
            # No prefix - unchanged
            ("onex.evt.platform.v1", "onex.evt.platform.v1"),
            ("topic.without.env.prefix", "topic.without.env.prefix"),
            # Empty string
            ("", ""),
            # Real-world examples from contracts
            (
                "{env}.onex.evt.platform.contract-registered.v1",
                "onex.evt.platform.contract-registered.v1",
            ),
            (
                "onex.evt.platform.contract-registered.v1",
                "onex.evt.platform.contract-registered.v1",
            ),
            (
                "prod.archon-intelligence.intelligence.code-analysis-requested.v1",
                "archon-intelligence.intelligence.code-analysis-requested.v1",
            ),
        ],
    )
    def test_normalize_topic_strips_prefix(
        self, input_topic: str, expected: str
    ) -> None:
        """Test that normalize_topic_for_storage strips known prefixes correctly."""
        assert normalize_topic_for_storage(input_topic) == expected

    def test_normalize_topic_env_placeholder(self) -> None:
        """Test {env}. placeholder prefix is stripped."""
        result = normalize_topic_for_storage("{env}.topic.name")
        assert result == "topic.name"

    def test_normalize_topic_dev_prefix(self) -> None:
        """Test dev. prefix is stripped."""
        result = normalize_topic_for_storage("dev.topic.name")
        assert result == "topic.name"

    def test_normalize_topic_prod_prefix(self) -> None:
        """Test prod. prefix is stripped."""
        result = normalize_topic_for_storage("prod.topic.name")
        assert result == "topic.name"

    def test_normalize_topic_staging_prefix(self) -> None:
        """Test staging. prefix is stripped."""
        result = normalize_topic_for_storage("staging.topic.name")
        assert result == "topic.name"

    def test_normalize_topic_local_prefix(self) -> None:
        """Test local. prefix is stripped."""
        result = normalize_topic_for_storage("local.topic.name")
        assert result == "topic.name"

    def test_normalize_topic_test_prefix(self) -> None:
        """Test test. prefix is stripped."""
        result = normalize_topic_for_storage("test.topic.name")
        assert result == "topic.name"

    def test_normalize_topic_no_prefix(self) -> None:
        """Test topic without known prefix is returned unchanged."""
        topic = "onex.evt.platform.v1"
        result = normalize_topic_for_storage(topic)
        assert result == topic

    def test_normalize_topic_empty_string(self) -> None:
        """Test empty string returns empty string."""
        result = normalize_topic_for_storage("")
        assert result == ""

    def test_normalize_topic_only_strips_first(self) -> None:
        """Test that only the first matching prefix is stripped.

        If a topic has multiple segments that look like prefixes,
        only the first one at the start should be stripped.
        """
        # "dev.test.something" -> strips "dev." first, result is "test.something"
        # NOT "something" (i.e., it doesn't strip "test." after stripping "dev.")
        result = normalize_topic_for_storage("dev.test.something")
        assert result == "test.something"

        # Verify "test." alone at start would be stripped
        result2 = normalize_topic_for_storage("test.something")
        assert result2 == "something"

    def test_normalize_topic_real_topic_example(self) -> None:
        """Test real-world topic example from contract definitions."""
        input_topic = "{env}.onex.evt.platform.contract-registered.v1"
        expected = "onex.evt.platform.contract-registered.v1"
        result = normalize_topic_for_storage(input_topic)
        assert result == expected

    def test_normalize_topic_prefix_not_at_start(self) -> None:
        """Test that prefix-like strings not at start are preserved."""
        # "something.dev.other" should remain unchanged
        topic = "something.dev.other"
        result = normalize_topic_for_storage(topic)
        assert result == topic

    def test_normalize_topic_partial_prefix_not_stripped(self) -> None:
        """Test that partial matches (without dot) are not stripped.

        'developer' should NOT have 'dev' stripped (no trailing dot).
        """
        result = normalize_topic_for_storage("developer.topic")
        assert result == "developer.topic"

        result2 = normalize_topic_for_storage("production.topic")
        assert result2 == "production.topic"

        result3 = normalize_topic_for_storage("testing.topic")
        assert result3 == "testing.topic"

    def test_normalize_topic_case_sensitive(self) -> None:
        """Test that prefix matching is case-sensitive.

        'DEV.topic' should NOT be stripped (uppercase).
        """
        result = normalize_topic_for_storage("DEV.topic")
        assert result == "DEV.topic"

        result2 = normalize_topic_for_storage("Dev.topic")
        assert result2 == "Dev.topic"

        result3 = normalize_topic_for_storage("{ENV}.topic")
        assert result3 == "{ENV}.topic"

    def test_normalize_topic_prefix_only(self) -> None:
        """Test topic that is exactly a prefix results in empty string."""
        # "{env}." alone -> ""
        result = normalize_topic_for_storage("{env}.")
        assert result == ""

        result2 = normalize_topic_for_storage("dev.")
        assert result2 == ""

    def test_normalize_topic_double_prefix(self) -> None:
        """Test topic with repeated prefix is stripped only once."""
        # "dev.dev.topic" -> "dev.topic"
        result = normalize_topic_for_storage("dev.dev.topic")
        assert result == "dev.topic"

        # "{env}.{env}.topic" -> "{env}.topic"
        result2 = normalize_topic_for_storage("{env}.{env}.topic")
        assert result2 == "{env}.topic"


class TestNormalizeTopicEdgeCases:
    """Edge case tests for normalize_topic_for_storage()."""

    @pytest.mark.parametrize(
        ("input_topic", "expected"),
        [
            # Empty string
            ("", ""),
            # Only prefix (results in empty string after stripping)
            ("dev.", ""),
            ("{env}.", ""),
            ("prod.", ""),
            ("staging.", ""),
            ("local.", ""),
            ("test.", ""),
            # Mixed double prefix - strips only the first ({env}.)
            ("{env}.dev.topic.v1", "dev.topic.v1"),
            ("{env}.prod.topic.v1", "prod.topic.v1"),
            ("{env}.staging.topic.v1", "staging.topic.v1"),
            ("dev.{env}.topic.v1", "{env}.topic.v1"),
            # Same-type double prefix - strips only first occurrence
            ("dev.dev.topic.v1", "dev.topic.v1"),
            ("{env}.{env}.topic.v1", "{env}.topic.v1"),
            # Prefix in middle (should NOT strip - not at start)
            ("topic.dev.name", "topic.dev.name"),
            ("topic.{env}.name", "topic.{env}.name"),
            ("onex.prod.events", "onex.prod.events"),
            # Multiple dots but no matching prefix
            ("topic.name.v1", "topic.name.v1"),
            ("onex.evt.platform.v1", "onex.evt.platform.v1"),
            # Single dot topics without prefix
            ("topic.v1", "topic.v1"),
        ],
    )
    def test_normalize_topic_edge_cases(self, input_topic: str, expected: str) -> None:
        """Test edge cases for topic normalization.

        This parametrized test covers:
        - Empty string handling
        - Prefix-only topics that result in empty string
        - Mixed double prefixes (e.g., {env}.dev.) - only first is stripped
        - Same-type double prefixes (e.g., dev.dev.) - only first is stripped
        - Prefix-like strings in middle of topic - should NOT be stripped
        - Topics with multiple dots but no matching prefix
        """
        assert normalize_topic_for_storage(input_topic) == expected

    def test_normalize_topic_with_numbers(self) -> None:
        """Test topics with version numbers are handled correctly."""
        result = normalize_topic_for_storage("onex.evt.v1.0.0")
        assert result == "onex.evt.v1.0.0"

    def test_normalize_topic_with_dashes(self) -> None:
        """Test topics with dashes are handled correctly."""
        result = normalize_topic_for_storage("prod.code-analysis-requested.v1")
        assert result == "code-analysis-requested.v1"

    def test_normalize_topic_with_underscores(self) -> None:
        """Test topics with underscores are handled correctly."""
        result = normalize_topic_for_storage("staging.node_registration_event.v2")
        assert result == "node_registration_event.v2"

    def test_normalize_topic_unicode(self) -> None:
        """Test that unicode characters are preserved."""
        result = normalize_topic_for_storage("dev.topic.with.unicode.name")
        assert result == "topic.with.unicode.name"

    def test_normalize_topic_whitespace_not_stripped(self) -> None:
        """Test that whitespace in topic is preserved (not recommended but valid)."""
        # Leading whitespace prevents prefix match
        result = normalize_topic_for_storage(" dev.topic")
        assert result == " dev.topic"

        # Trailing whitespace preserved
        result2 = normalize_topic_for_storage("dev.topic ")
        assert result2 == "topic "
