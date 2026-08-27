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

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import yaml

from scripts.check_subscribe_wiring_health import (
    _BASELINE_DEAD_LETTER_ALLOWLIST,
    _EXTERNAL_PUBLISHER_ALLOWLIST,
    _is_infrastructure_topic,
    _parse_allowlist_expiry,
    check_allowlist_hygiene,
    check_wiring_health,
    collect_subscribed_topics,
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


# ---------------------------------------------------------------------------
# OMN-16795: allowlist hygiene enforcement.
#
# Before this, an allowlist entry's `expiry:` was decoration — the checker never
# read it, so a 45-entry baseline drifted with dates already lapsing and nothing
# went red. An amnesty list nobody can be forced off is not a baseline, it is a
# permanent exemption with a date-shaped comment on it.
#
# Three failure modes are enforced here, each of which used to be silent:
#   1. EXPIRED  — the owner's own deadline passed.
#   2. MALFORMED— no parseable `expiry:`, so the entry can never expire.
#   3. STALE    — nothing subscribes to the topic any more, so the entry is
#                 dead weight that hides the list's real size.
# ---------------------------------------------------------------------------


class TestAllowlistExpiryParsing:
    """`expiry:` must be machine-readable, not prose."""

    def test_parses_a_well_formed_expiry(self) -> None:
        assert _parse_allowlist_expiry(
            "Some reason | owner: jonah | expiry: 2026-09-01"
        ) == date(2026, 9, 1)

    def test_returns_none_when_expiry_absent(self) -> None:
        assert _parse_allowlist_expiry("Some reason | owner: jonah") is None

    def test_returns_none_on_unparseable_date(self) -> None:
        assert (
            _parse_allowlist_expiry("reason | owner: jonah | expiry: someday") is None
        )

    def test_tolerates_surrounding_whitespace(self) -> None:
        assert _parse_allowlist_expiry(
            "reason | owner: jonah | expiry:   2026-12-01  "
        ) == date(2026, 12, 1)


class TestAllowlistHygieneEnforcement:
    """An entry past its own expiry must FAIL, not silently persist."""

    def test_expired_entry_is_an_error(self) -> None:
        errors = check_allowlist_hygiene(
            allowlists={"onex.cmd.x.y.v1": "r | owner: jonah | expiry: 2026-01-01"},
            subscribed_topics={"onex.cmd.x.y.v1"},
            today=date(2026, 8, 27),
        )
        assert any("EXPIRED" in e and "onex.cmd.x.y.v1" in e for e in errors), errors

    def test_entry_expiring_today_is_an_error(self) -> None:
        """Expiry is inclusive: on the stated day the exemption is over."""
        errors = check_allowlist_hygiene(
            allowlists={"onex.cmd.x.y.v1": "r | owner: jonah | expiry: 2026-08-27"},
            subscribed_topics={"onex.cmd.x.y.v1"},
            today=date(2026, 8, 27),
        )
        assert any("EXPIRED" in e for e in errors), errors

    def test_unexpired_entry_is_not_an_error(self) -> None:
        errors = check_allowlist_hygiene(
            allowlists={"onex.cmd.x.y.v1": "r | owner: jonah | expiry: 2026-12-01"},
            subscribed_topics={"onex.cmd.x.y.v1"},
            today=date(2026, 8, 27),
        )
        assert errors == [], errors

    def test_missing_expiry_is_an_error(self) -> None:
        """No parseable expiry means the entry could never expire — reject it."""
        errors = check_allowlist_hygiene(
            allowlists={"onex.cmd.x.y.v1": "reason with no expiry | owner: jonah"},
            subscribed_topics={"onex.cmd.x.y.v1"},
            today=date(2026, 8, 27),
        )
        assert any("MALFORMED" in e for e in errors), errors

    def test_stale_entry_with_no_subscriber_is_an_error(self) -> None:
        """Nothing subscribes to it, so the exemption is dead weight."""
        errors = check_allowlist_hygiene(
            allowlists={"onex.cmd.x.y.v1": "r | owner: jonah | expiry: 2026-12-01"},
            subscribed_topics=set(),
            today=date(2026, 8, 27),
        )
        assert any("STALE" in e and "onex.cmd.x.y.v1" in e for e in errors), errors

    def test_error_names_the_owner_so_it_is_actionable(self) -> None:
        errors = check_allowlist_hygiene(
            allowlists={"onex.cmd.x.y.v1": "r | owner: jonah | expiry: 2026-01-01"},
            subscribed_topics={"onex.cmd.x.y.v1"},
            today=date(2026, 8, 27),
        )
        assert any("jonah" in e for e in errors), errors


class TestLiveAllowlistHygiene:
    """The REAL allowlists must be clean TODAY — this is the ratchet.

    This is the row that makes the enforcement bite: it runs against the
    checked-in allowlists with the real clock, so an entry that lapses turns
    this suite (and the CI gate wired to it) RED on the day it lapses.
    """

    def test_live_allowlists_have_no_expired_malformed_or_stale_entries(self) -> None:
        if not NODES_DIR.exists():
            pytest.skip("Nodes directory not found")

        subscribed = collect_subscribed_topics([NODES_DIR])
        merged = {**_EXTERNAL_PUBLISHER_ALLOWLIST, **_BASELINE_DEAD_LETTER_ALLOWLIST}
        errors = check_allowlist_hygiene(
            allowlists=merged,
            subscribed_topics=subscribed,
            today=datetime.now(UTC).date(),
        )
        assert errors == [], (
            "Allowlist hygiene failures — renew with a fresh reason+expiry if the "
            "exemption is still deserved, or delete the entry:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def test_every_live_entry_carries_a_parseable_expiry(self) -> None:
        merged = {**_EXTERNAL_PUBLISHER_ALLOWLIST, **_BASELINE_DEAD_LETTER_ALLOWLIST}
        missing = [t for t, r in merged.items() if _parse_allowlist_expiry(r) is None]
        assert missing == [], f"Entries with no parseable expiry: {missing}"


class TestOmn16795IncidentReplay:
    """Replay the REAL dead-subscribe contract through the REAL checker (OMN-15547).

    Incident OMN-16755. ``node_pr_state_write_effect``'s contract declares
    ``subscribe_topics: [onex.cmd.omnibase-infra.pr-state-upsert.v1]`` and NO
    contract anywhere publishes to it. That is not cosmetic: the runtime gates
    on ``subcontract.subscribe_topics``, provisions the topic, and mints one
    Kafka consumer group per topic, so the declaration manufactures a
    permanently ``Stable``-and-empty consumer group. Chain-liveness tooling read
    that empty group as a live-but-idle consumer and reported a dead chain — a
    false alarm that cost a triage lane.

    The checker that detects exactly this shipped in OMN-7385 and was wired to
    NOTHING, so the condition sat in the tree green for months. Direction is
    therefore ``false_green``: the enforcement said OK on a real bad input,
    because the enforcement was never running.

    The fixture is the contract's real bytes, read out of the git object at
    ``e4932fe4`` — not a synthetic minimal contract. A hand-written stand-in
    would be the defect OMN-15547 exists to stop: it can only exhibit the
    failure modes whoever wrote it already thought of.
    """

    FIXTURE = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "fixtures"
        / "omn16795"
        / "node_pr_state_write_effect-contract.yaml.captured"
    )

    def test_real_dead_subscribe_contract_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must say DEAD_LETTER on the real bytes, absent an exemption.

        The allowlists are emptied for this assertion on purpose. The live tree
        exempts this topic (it is genuinely routed by intent, not Kafka), and
        that exemption is what the rest of this suite covers. What is under test
        HERE is the detector itself: given the actual contract that produced the
        OMN-16755 false alarm, does the checker identify it at all? A guard that
        only ever sees allowlisted input is never exercised.
        """
        assert self.FIXTURE.is_file(), f"captured fixture missing: {self.FIXTURE}"

        node_dir = tmp_path / "node_pr_state_write_effect"
        node_dir.mkdir(parents=True)
        # Byte-for-byte, no reserialization — a reformatted artifact is no
        # longer the artifact that failed.
        (node_dir / "contract.yaml").write_bytes(self.FIXTURE.read_bytes())

        monkeypatch.setattr(
            "scripts.check_subscribe_wiring_health._EXTERNAL_PUBLISHER_ALLOWLIST", {}
        )
        monkeypatch.setattr(
            "scripts.check_subscribe_wiring_health._BASELINE_DEAD_LETTER_ALLOWLIST", {}
        )

        errors, _warnings = check_wiring_health([tmp_path])

        assert any(
            "DEAD_LETTER" in e and "onex.cmd.omnibase-infra.pr-state-upsert.v1" in e
            for e in errors
        ), (
            "the checker did not flag the REAL contract whose dead subscribe "
            f"declaration produced the OMN-16755 false alarm. errors={errors}"
        )

    def test_captured_fixture_still_declares_the_dead_subscribe(self) -> None:
        """Non-vacuity: if the capture were replaced by bytes lacking the
        declaration, the replay above would pass for the wrong reason."""
        text = self.FIXTURE.read_text()
        assert "subscribe_topics:" in text
        assert "onex.cmd.omnibase-infra.pr-state-upsert.v1" in text
