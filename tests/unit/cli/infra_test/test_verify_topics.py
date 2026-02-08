# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ONEX topic naming validation regex.

Verifies that the ONEX_TOPIC_PATTERN correctly identifies compliant
and non-compliant topic names against the 5-segment convention:
``onex.<kind>.<producer>.<event-name>.v<version>``
"""

from __future__ import annotations

import pytest

from omnibase_infra.cli.infra_test.verify import ONEX_TOPIC_PATTERN


@pytest.mark.unit
class TestOnexTopicPattern:
    """Test ONEX 5-segment topic naming regex."""

    @pytest.mark.parametrize(
        "topic",
        [
            "onex.evt.platform.node-introspection.v1",
            "onex.evt.platform.node-registration-result.v1",
            "onex.evt.platform.node-registration-initiated.v1",
            "onex.evt.platform.node-became-active.v1",
            "onex.cmd.platform.node-registration-acked.v1",
            "onex.intent.platform.runtime-tick.v1",
            "onex.snapshot.platform.registration-snapshots.v1",
            "onex.dlq.platform.registration.v1",
            "onex.evt.platform.node-heartbeat.v1",
            "onex.evt.platform.registry-request-introspection.v1",
            "onex.evt.myteam.custom-event.v2",
            "onex.evt.platform.node-liveness-expired.v1",
            "onex.evt.platform.node-registration-ack-timed-out.v1",
            "onex.evt.platform.node-registration-ack-received.v1",
            "onex.evt.platform.node-registration-accepted.v1",
            "onex.evt.platform.node-registration-rejected.v1",
        ],
    )
    def test_valid_onex_topics(self, topic: str) -> None:
        """Valid ONEX topics match the pattern."""
        assert ONEX_TOPIC_PATTERN.match(topic), f"Expected match for: {topic}"

    @pytest.mark.parametrize(
        "topic",
        [
            # Missing prefix
            "evt.platform.node-introspection.v1",
            # Wrong prefix
            "kafka.evt.platform.node-introspection.v1",
            # Invalid kind
            "onex.event.platform.node-introspection.v1",
            "onex.command.platform.node-introspection.v1",
            # Missing segments
            "onex.evt.platform.v1",
            "onex.evt.v1",
            # Version without number
            "onex.evt.platform.node-introspection.v",
            # Capital letters in producer/name
            "onex.evt.Platform.node-introspection.v1",
            "onex.evt.platform.NodeIntrospection.v1",
            # Legacy format (non-ONEX)
            "dev.registration.events.v1",
            "agent-actions",
            "router-performance-metrics",
            # Underscores instead of hyphens
            "onex.evt.platform.node_introspection.v1",
            # Empty segments
            "onex.evt..node-introspection.v1",
            # Spaces
            "onex.evt.platform.node introspection.v1",
        ],
    )
    def test_invalid_topics(self, topic: str) -> None:
        """Non-compliant topics do not match."""
        assert not ONEX_TOPIC_PATTERN.match(topic), f"Expected no match for: {topic}"

    def test_internal_topics_skipped(self) -> None:
        """Internal Kafka/Redpanda topics (starting with _) are not ONEX topics."""
        assert not ONEX_TOPIC_PATTERN.match("_schemas")
        assert not ONEX_TOPIC_PATTERN.match("__consumer_offsets")
