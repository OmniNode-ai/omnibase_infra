# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the ONEX topic format validator."""

from __future__ import annotations

import pytest

from omnibase_infra.utils.util_onex_topic_format import (
    TopicValidationResult,
    validate_onex_topic_format,
)


class TestValidateOnexTopicFormat:
    """Test suite for validate_onex_topic_format()."""

    # ------------------------------------------------------------------
    # Valid canonical ONEX topics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "onex.evt.omniclaude.session-started.v1",
            "onex.cmd.omniintelligence.code-analysis.v1",
            "onex.intent.omnimemory.crawl-requested.v1",
            "onex.dlq.omniclaude.agent-actions.v1",
            "onex.evt.platform.node-heartbeat.v1",
            "onex.evt.omniclaude.transformation.completed.v1",
            "onex.evt.platform.node-registration.v12",
        ],
    )
    def test_valid_onex_topics(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.VALID
        assert reason == ""

    # ------------------------------------------------------------------
    # Valid legacy DLQ topics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "onex.dlq.intelligence.v1",
            "local.dlq.intents.v1",
        ],
    )
    def test_valid_legacy_dlq_topics(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.VALID_LEGACY_DLQ
        assert reason == "legacy DLQ format"

    # ------------------------------------------------------------------
    # Invalid topics
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "dev.onex.evt.omniclaude.session-started.v1",  # env prefix
            "onex.evt.omniclaude.session-started",  # missing version
            "agent-actions",  # legacy flat name
            "staging.onex.cmd.omniintelligence.code-analysis.v1",  # env prefix
            "",  # empty
            "onex.evt",  # too few segments
            "onex.unknown.omniclaude.session-started.v1",  # invalid kind
            "onex.evt.omniclaude.session-started.v0",  # v0 not allowed
        ],
    )
    def test_invalid_topics(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.INVALID
        assert reason != ""

    # ------------------------------------------------------------------
    # Kafka internal topics (skipped)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "topic",
        [
            "__consumer_offsets",
            "__transaction_state",
        ],
    )
    def test_kafka_internal_topics_skipped(self, topic: str) -> None:
        result, reason = validate_onex_topic_format(topic)
        assert result == TopicValidationResult.SKIPPED_INTERNAL
        assert reason == ""
