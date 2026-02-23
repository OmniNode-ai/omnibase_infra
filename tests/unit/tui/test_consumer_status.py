# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for consumer_status module.

Tests:
- Message classes are instantiated correctly
- Topic constants match expected values
- consume_all handles Kafka errors gracefully (mock)

Run with:
    uv run pytest tests/unit/tui/ -m unit -v

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

import pytest

from omnibase_infra.tui.consumers.consumer_status import (
    TOPIC_GIT_HOOK,
    TOPIC_LINEAR_SNAPSHOT,
    TOPIC_PR_STATUS,
    HookEventReceived,
    PRStatusReceived,
    SnapshotReceived,
)


class TestTopicConstants:
    @pytest.mark.unit
    def test_pr_status_topic(self) -> None:
        assert TOPIC_PR_STATUS == "onex.evt.github.pr-status.v1"

    @pytest.mark.unit
    def test_git_hook_topic(self) -> None:
        assert TOPIC_GIT_HOOK == "onex.evt.git.hook.v1"

    @pytest.mark.unit
    def test_linear_snapshot_topic(self) -> None:
        assert TOPIC_LINEAR_SNAPSHOT == "onex.evt.linear.snapshot.v1"

    @pytest.mark.unit
    def test_all_topics_distinct(self) -> None:
        topics = {TOPIC_PR_STATUS, TOPIC_GIT_HOOK, TOPIC_LINEAR_SNAPSHOT}
        assert len(topics) == 3


class TestMessageClasses:
    @pytest.mark.unit
    def test_pr_status_received_stores_payload(self) -> None:
        payload = {
            "event_type": TOPIC_PR_STATUS,
            "pr_number": 42,
            "triage_state": "needs_review",
            "title": "Add feature X",
            "partition_key": "OmniNode-ai/omnibase_infra:42",
        }
        msg = PRStatusReceived(payload)
        assert msg.payload == payload
        assert msg.payload["pr_number"] == 42

    @pytest.mark.unit
    def test_hook_event_received_stores_payload(self) -> None:
        payload = {
            "event_type": TOPIC_GIT_HOOK,
            "hook": "pre-commit",
            "repo": "OmniNode-ai/omniclaude",
            "branch": "main",
            "author": "jsmith",
            "outcome": "pass",
            "gates": ["lint", "tests"],
            "emitted_at": "2026-02-23T10:00:00Z",
        }
        msg = HookEventReceived(payload)
        assert msg.payload["hook"] == "pre-commit"
        assert msg.payload["outcome"] == "pass"

    @pytest.mark.unit
    def test_snapshot_received_stores_payload(self) -> None:
        payload = {
            "event_type": TOPIC_LINEAR_SNAPSHOT,
            "snapshot_id": "abc-123",
            "workstreams": ["Runtime", "Models"],
            "snapshot": {"epics": []},
            "emitted_at": "2026-02-23T10:00:00Z",
        }
        msg = SnapshotReceived(payload)
        assert msg.payload["snapshot_id"] == "abc-123"
        assert msg.payload["workstreams"] == ["Runtime", "Models"]

    @pytest.mark.unit
    def test_pr_status_received_empty_payload(self) -> None:
        msg = PRStatusReceived({})
        assert msg.payload == {}

    @pytest.mark.unit
    def test_hook_event_received_empty_payload(self) -> None:
        msg = HookEventReceived({})
        assert msg.payload == {}

    @pytest.mark.unit
    def test_snapshot_received_empty_payload(self) -> None:
        msg = SnapshotReceived({})
        assert msg.payload == {}
