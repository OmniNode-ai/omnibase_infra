# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for cross-contract chain verification [OMN-7040].

Pure YAML parsing tests -- no Docker or runtime needed.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.cross_contract import verify_contract_chain


def _write_contract(tmp_path: Path, name: str, content: str) -> Path:
    """Write a contract.yaml to a temp directory and return its path."""
    contract_dir = tmp_path / name
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_file = contract_dir / "contract.yaml"
    contract_file.write_text(dedent(content))
    return contract_file


@pytest.mark.unit
class TestVerifyContractChain:
    """Tests for verify_contract_chain cross-contract verification."""

    def test_connected_chain_passes(self, tmp_path: Path) -> None:
        """Orchestrator -> reducer -> effect with matching topics passes."""
        orchestrator = _write_contract(
            tmp_path,
            "orchestrator",
            """\
            name: node_registration_orchestrator
            node_type: ORCHESTRATOR_GENERIC
            event_bus:
              subscribe_topics:
                - onex.cmd.registration.requested.v1
              publish_topics:
                - onex.evt.registration.state_changed.v1
            """,
        )
        reducer = _write_contract(
            tmp_path,
            "reducer",
            """\
            name: node_registration_reducer
            node_type: REDUCER
            event_bus:
              subscribe_topics:
                - onex.evt.registration.state_changed.v1
              publish_topics:
                - onex.evt.registration.reduced.v1
            """,
        )
        effect = _write_contract(
            tmp_path,
            "effect",
            """\
            name: node_registration_storage_effect
            node_type: EFFECT
            event_bus:
              subscribe_topics:
                - onex.evt.registration.reduced.v1
              publish_topics:
                - onex.evt.registration.stored.v1
            """,
        )

        report = verify_contract_chain([orchestrator, reducer, effect])

        assert report.overall_verdict == EnumValidationVerdict.PASS
        assert len(report.checks) == 3
        assert report.node_type == "CHAIN_VERIFICATION"
        assert report.probe_mode == "static"

    def test_disconnected_chain_fails(self, tmp_path: Path) -> None:
        """Orchestrator publishes topic X but reducer subscribes to topic Y."""
        orchestrator = _write_contract(
            tmp_path,
            "orchestrator",
            """\
            name: node_registration_orchestrator
            node_type: ORCHESTRATOR_GENERIC
            event_bus:
              subscribe_topics:
                - onex.cmd.registration.requested.v1
              publish_topics:
                - onex.evt.registration.state_changed.v1
            """,
        )
        reducer = _write_contract(
            tmp_path,
            "reducer",
            """\
            name: node_registration_reducer
            node_type: REDUCER
            event_bus:
              subscribe_topics:
                - onex.evt.WRONG_TOPIC.v1
              publish_topics:
                - onex.evt.registration.reduced.v1
            """,
        )

        report = verify_contract_chain([orchestrator, reducer])

        assert report.overall_verdict == EnumValidationVerdict.FAIL
        # Topic connectivity check should fail
        topic_checks = [
            c
            for c in report.checks
            if "connectivity" in c.message.lower() or "gap" in c.message.lower()
        ]
        assert any(c.verdict == EnumValidationVerdict.FAIL for c in topic_checks)

    def test_single_contract_passes_trivially(self, tmp_path: Path) -> None:
        """A chain with a single contract has nothing to verify."""
        solo = _write_contract(
            tmp_path,
            "solo",
            """\
            name: node_solo
            node_type: COMPUTE
            event_bus:
              subscribe_topics:
                - onex.cmd.solo.v1
              publish_topics:
                - onex.evt.solo.done.v1
            """,
        )

        report = verify_contract_chain([solo])
        assert report.overall_verdict == EnumValidationVerdict.PASS

    def test_event_type_routing_detected(self, tmp_path: Path) -> None:
        """Event type matching between published_events and consumed_events."""
        upstream = _write_contract(
            tmp_path,
            "upstream",
            """\
            name: upstream_node
            node_type: ORCHESTRATOR_GENERIC
            event_bus:
              subscribe_topics:
                - onex.cmd.up.v1
              publish_topics:
                - onex.evt.up.done.v1
            published_events:
              - event_type: ModelRegistrationStateChanged
            consumed_events:
              - event_type: ModelRegistrationRequested
            """,
        )
        downstream = _write_contract(
            tmp_path,
            "downstream",
            """\
            name: downstream_node
            node_type: REDUCER
            event_bus:
              subscribe_topics:
                - onex.evt.up.done.v1
              publish_topics:
                - onex.evt.down.done.v1
            published_events:
              - event_type: ModelRegistrationReduced
            consumed_events:
              - event_type: ModelRegistrationStateChanged
            """,
        )

        report = verify_contract_chain([upstream, downstream])

        assert report.overall_verdict == EnumValidationVerdict.PASS
        # Event routing check should show connected pair
        event_checks = [
            c
            for c in report.checks
            if "event" in c.evidence.lower() or "routing" in c.message.lower()
        ]
        assert any(c.verdict == EnumValidationVerdict.PASS for c in event_checks)

    def test_event_type_gap_detected(self, tmp_path: Path) -> None:
        """Upstream publishes event A but downstream consumes event B."""
        upstream = _write_contract(
            tmp_path,
            "upstream",
            """\
            name: upstream_node
            node_type: ORCHESTRATOR_GENERIC
            event_bus:
              subscribe_topics:
                - onex.cmd.up.v1
              publish_topics:
                - onex.evt.up.done.v1
            published_events:
              - event_type: ModelEventA
            """,
        )
        downstream = _write_contract(
            tmp_path,
            "downstream",
            """\
            name: downstream_node
            node_type: REDUCER
            event_bus:
              subscribe_topics:
                - onex.evt.up.done.v1
              publish_topics:
                - onex.evt.down.done.v1
            consumed_events:
              - event_type: ModelEventB
            """,
        )

        report = verify_contract_chain([upstream, downstream])

        # Event routing should fail (RECOMMENDED severity, so overall may still pass)
        event_checks = [c for c in report.checks if "routing" in c.message.lower()]
        assert any(c.verdict == EnumValidationVerdict.FAIL for c in event_checks)

    def test_no_event_types_is_informational_pass(self, tmp_path: Path) -> None:
        """Contracts with no event type declarations pass projection check."""
        a = _write_contract(
            tmp_path,
            "a",
            """\
            name: node_a
            node_type: COMPUTE
            event_bus:
              subscribe_topics:
                - onex.cmd.a.v1
              publish_topics:
                - onex.evt.a.done.v1
            """,
        )
        b = _write_contract(
            tmp_path,
            "b",
            """\
            name: node_b
            node_type: COMPUTE
            event_bus:
              subscribe_topics:
                - onex.evt.a.done.v1
              publish_topics:
                - onex.evt.b.done.v1
            """,
        )

        report = verify_contract_chain([a, b])
        assert report.overall_verdict == EnumValidationVerdict.PASS

    def test_report_has_fingerprint(self, tmp_path: Path) -> None:
        """Report includes a non-empty fingerprint."""
        a = _write_contract(
            tmp_path,
            "a",
            """\
            name: node_a
            node_type: COMPUTE
            event_bus:
              subscribe_topics:
                - onex.cmd.a.v1
              publish_topics:
                - onex.evt.a.done.v1
            """,
        )

        report = verify_contract_chain([a])
        assert len(report.report_fingerprint) == 64  # SHA-256 hex

    def test_chain_name_in_report(self, tmp_path: Path) -> None:
        """Report contract_name reflects the full chain."""
        a = _write_contract(
            tmp_path,
            "a",
            """\
            name: alpha
            node_type: COMPUTE
            event_bus:
              subscribe_topics: []
              publish_topics:
                - t.v1
            """,
        )
        b = _write_contract(
            tmp_path,
            "b",
            """\
            name: beta
            node_type: COMPUTE
            event_bus:
              subscribe_topics:
                - t.v1
              publish_topics: []
            """,
        )

        report = verify_contract_chain([a, b])
        assert report.contract_name == "alpha -> beta"

    def test_duration_is_non_negative(self, tmp_path: Path) -> None:
        """Duration is tracked and non-negative."""
        a = _write_contract(
            tmp_path,
            "a",
            """\
            name: node_a
            node_type: COMPUTE
            """,
        )
        report = verify_contract_chain([a])
        assert report.duration_ms >= 0
