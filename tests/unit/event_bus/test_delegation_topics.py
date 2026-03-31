# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for delegation pipeline topic constants.

Verifies that all delegation topics follow ONEX naming conventions:
    onex.<kind>.<producer>.<event-name>.v<N>

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

import re

import pytest

from omnibase_infra.event_bus.topic_constants import (
    TOPIC_DELEGATION_COMPLETED,
    TOPIC_DELEGATION_FAILED,
    TOPIC_DELEGATION_QUALITY_GATE_RESULT,
    TOPIC_DELEGATION_REQUEST,
    TOPIC_DELEGATION_ROUTING_DECISION,
)

pytestmark = [pytest.mark.unit]

# ONEX topic naming pattern: onex.<kind>.<producer>.<event-name>.v<N>
_TOPIC_PATTERN = re.compile(
    r"^onex\.(cmd|evt|int)\.[a-z][a-z0-9-]*\.[a-z][a-z0-9-]*\.v\d+$"
)

_ALL_DELEGATION_TOPICS = [
    TOPIC_DELEGATION_REQUEST,
    TOPIC_DELEGATION_ROUTING_DECISION,
    TOPIC_DELEGATION_COMPLETED,
    TOPIC_DELEGATION_FAILED,
    TOPIC_DELEGATION_QUALITY_GATE_RESULT,
]


class TestDelegationTopicNaming:
    """Verify ONEX naming convention compliance."""

    @pytest.mark.parametrize("topic", _ALL_DELEGATION_TOPICS)
    def test_topic_matches_onex_pattern(self, topic: str) -> None:
        assert _TOPIC_PATTERN.match(topic), (
            f"Topic '{topic}' does not match ONEX naming: "
            "onex.<kind>.<producer>.<event-name>.v<N>"
        )

    @pytest.mark.parametrize("topic", _ALL_DELEGATION_TOPICS)
    def test_producer_is_omnibase_infra(self, topic: str) -> None:
        parts = topic.split(".")
        assert parts[2] == "omnibase-infra", (
            f"Topic '{topic}' producer should be 'omnibase-infra', got '{parts[2]}'"
        )


class TestDelegationTopicValues:
    """Verify exact topic string values."""

    def test_delegation_request_topic(self) -> None:
        assert (
            TOPIC_DELEGATION_REQUEST == "onex.cmd.omnibase-infra.delegation-request.v1"
        )

    def test_routing_decision_topic(self) -> None:
        assert (
            TOPIC_DELEGATION_ROUTING_DECISION
            == "onex.evt.omnibase-infra.routing-decision.v1"
        )

    def test_delegation_completed_topic(self) -> None:
        assert (
            TOPIC_DELEGATION_COMPLETED
            == "onex.evt.omnibase-infra.delegation-completed.v1"
        )

    def test_delegation_failed_topic(self) -> None:
        assert TOPIC_DELEGATION_FAILED == "onex.evt.omnibase-infra.delegation-failed.v1"

    def test_quality_gate_result_topic(self) -> None:
        assert (
            TOPIC_DELEGATION_QUALITY_GATE_RESULT
            == "onex.evt.omnibase-infra.quality-gate-result.v1"
        )


class TestDelegationTopicKinds:
    """Verify command vs event topic kinds."""

    def test_request_is_command(self) -> None:
        assert TOPIC_DELEGATION_REQUEST.startswith("onex.cmd.")

    def test_events_are_evt(self) -> None:
        event_topics = [
            TOPIC_DELEGATION_ROUTING_DECISION,
            TOPIC_DELEGATION_COMPLETED,
            TOPIC_DELEGATION_FAILED,
            TOPIC_DELEGATION_QUALITY_GATE_RESULT,
        ]
        for topic in event_topics:
            assert topic.startswith("onex.evt."), (
                f"Topic '{topic}' should be an event (onex.evt.*)"
            )
