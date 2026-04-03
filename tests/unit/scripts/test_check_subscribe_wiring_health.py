# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the subscribe-topic wiring health check.

Verifies that:
1. Dead-letter subscriptions are detected (subscribe with no publisher)
2. Orphan publishers are flagged as warnings
3. Allowlisted topics are skipped
4. Infrastructure topics (DLQ, broadcast) are skipped
5. The current contract set passes with baseline allowlist

[OMN-7385]
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from scripts.check_subscribe_wiring_health import (
    _BASELINE_DEAD_LETTER_ALLOWLIST,
    _EXTERNAL_PUBLISHER_ALLOWLIST,
    _is_infrastructure_topic,
    check_wiring_health,
)

pytestmark = pytest.mark.unit

NODES_DIR = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"


class TestInfrastructureTopicFilter:
    """Verify infrastructure topic detection."""

    def test_dlq_topic_is_infrastructure(self) -> None:
        assert _is_infrastructure_topic("onex.evt.platform.foo.dlq.v1")

    def test_broadcast_topic_is_infrastructure(self) -> None:
        assert _is_infrastructure_topic("onex.evt.platform.foo.broadcast.v1")

    def test_normal_topic_is_not_infrastructure(self) -> None:
        assert not _is_infrastructure_topic("onex.evt.platform.node-registration.v1")


class TestWiringHealthWithSyntheticContracts:
    """Test wiring health check logic with temporary contract files."""

    def _write_contract(
        self,
        tmp_path: Path,
        node_name: str,
        subscribe: list[str] | None = None,
        publish: list[str] | None = None,
    ) -> Path:
        """Write a minimal contract.yaml for testing."""
        node_dir = tmp_path / f"node_{node_name}"
        node_dir.mkdir(parents=True)
        contract = {
            "name": f"node_{node_name}",
            "node_type": "EFFECT_GENERIC",
            "event_bus": {
                "subscribe_topics": subscribe or [],
                "publish_topics": publish or [],
            },
        }
        contract_path = node_dir / "contract.yaml"
        contract_path.write_text(yaml.dump(contract))
        return tmp_path

    def test_no_violations_when_wired(self, tmp_path: Path) -> None:
        """Subscribe topics with matching publishers should pass."""
        self._write_contract(
            tmp_path,
            "producer",
            publish=["onex.evt.test.event.v1"],
        )
        self._write_contract(
            tmp_path,
            "consumer",
            subscribe=["onex.evt.test.event.v1"],
        )
        errors, _warnings = check_wiring_health([tmp_path])
        assert errors == []

    def test_detects_dead_letter_subscription(self, tmp_path: Path) -> None:
        """Subscribe topic with no publisher should be flagged."""
        self._write_contract(
            tmp_path,
            "consumer",
            subscribe=["onex.evt.test.orphan.v1"],
        )
        errors, _warnings = check_wiring_health([tmp_path])
        assert len(errors) == 1
        assert "DEAD_LETTER" in errors[0]
        assert "onex.evt.test.orphan.v1" in errors[0]

    def test_detects_orphan_publisher(self, tmp_path: Path) -> None:
        """Publish topic with no subscriber should produce a warning."""
        self._write_contract(
            tmp_path,
            "producer",
            publish=["onex.evt.test.lonely.v1"],
        )
        errors, warnings = check_wiring_health([tmp_path])
        assert errors == []
        assert len(warnings) == 1
        assert "NO_SUBSCRIBER" in warnings[0]

    def test_self_loop_is_valid(self, tmp_path: Path) -> None:
        """A node that publishes and subscribes to the same topic is valid."""
        self._write_contract(
            tmp_path,
            "loopback",
            subscribe=["onex.evt.test.self.v1"],
            publish=["onex.evt.test.self.v1"],
        )
        errors, warnings = check_wiring_health([tmp_path])
        assert errors == []
        assert warnings == []

    def test_skips_missing_directory(self, tmp_path: Path) -> None:
        """Non-existent directories should be skipped gracefully."""
        errors, _warnings = check_wiring_health([tmp_path / "nonexistent"])
        assert errors == []

    def test_multiple_directories(self, tmp_path: Path) -> None:
        """Topics across multiple directories should be cross-referenced."""
        dir_a = tmp_path / "repo_a"
        dir_b = tmp_path / "repo_b"
        self._write_contract(
            dir_a,
            "producer",
            publish=["onex.evt.cross.event.v1"],
        )
        self._write_contract(
            dir_b,
            "consumer",
            subscribe=["onex.evt.cross.event.v1"],
        )
        errors, _warnings = check_wiring_health([dir_a, dir_b])
        assert errors == []


class TestAllowlists:
    """Verify allowlist structure and content."""

    def test_external_allowlist_entries_are_valid_topic_strings(self) -> None:
        for topic in _EXTERNAL_PUBLISHER_ALLOWLIST:
            assert topic.startswith("onex."), f"Bad topic: {topic}"

    def test_baseline_allowlist_entries_are_valid_topic_strings(self) -> None:
        for topic in _BASELINE_DEAD_LETTER_ALLOWLIST:
            assert topic.startswith("onex."), f"Bad topic: {topic}"

    def test_baseline_entries_have_owner_and_expiry(self) -> None:
        for topic, reason in _BASELINE_DEAD_LETTER_ALLOWLIST.items():
            assert "owner:" in reason, f"Missing owner for {topic}"
            assert "expiry:" in reason, f"Missing expiry for {topic}"

    def test_no_overlap_between_allowlists(self) -> None:
        overlap = set(_EXTERNAL_PUBLISHER_ALLOWLIST) & set(
            _BASELINE_DEAD_LETTER_ALLOWLIST
        )
        assert not overlap, f"Topics in both allowlists: {overlap}"


class TestCurrentContractWiring:
    """Verify the current contract set passes with baseline allowlist."""

    def test_current_contracts_pass(self) -> None:
        """The real node contracts should pass with baseline allowlist."""
        if not NODES_DIR.exists():
            pytest.skip("Nodes directory not found")

        errors, _warnings = check_wiring_health([NODES_DIR])
        assert errors == [], (
            "Dead-letter subscriptions found (not in baseline allowlist):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
