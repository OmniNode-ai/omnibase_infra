# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI gate tests: dispatcher route coverage (OMN-12858, OMN-12879, OMN-12880).

Verifies the check_dispatcher_route_coverage gate logic using synthetic
in-memory contracts via the gate's check_route_coverage() API.

No Kafka, no DB, no real handler imports — pure static contract analysis.

Test cases:
    RED  — contract subscribes a command topic but declares no handler_routing
           or runtime_dispatch => gate FAILS with the unrouted topic in failures
    GREEN — contract subscribes a command topic and declares handler_routing
           => gate PASSES (0 failures)
    COMPAT — compatibility_publish_topics are sender-side only; the gate MUST
           NOT flag them as unrouted (OMN-12880)
    CHANGED — when --changed-contracts mode is active, only the specified
           contracts are checked; unchanged contracts with gaps are skipped (OMN-12879)
    ALLOWLIST — known pre-existing violations in _ALLOWLISTED_CMD_TOPICS pass
           without failing the gate

These tests prove the gate would have caught:
  1. The June 9 DLQ regression: node_generation_consumer subscribed
     onex.cmd.omnimarket.node-generation-requested.v1 but auto-wiring produced
     zero dispatchers after a sole-handler revert — messages went to DLQ silently.
  2. The June 12 DEL-01 live finding: delegate-skill consumed-but-undispatched
     on dev (onex.cmd.omnimarket.delegate-skill.v1 with no matching route).

[OMN-12858, OMN-12879, OMN-12880]
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Locate the gate script so tests can import its public check_route_coverage()
# API without needing the package installed.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from check_dispatcher_route_coverage import (  # type: ignore[import-not-found]
    RouteCoverageReport,
    check_route_coverage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_contract(tmp_path: Path, node_name: str, contract_yaml: str) -> Path:
    """Write a synthetic contract.yaml for a node and return its directory."""
    node_dir = tmp_path / node_name
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "contract.yaml").write_text(textwrap.dedent(contract_yaml))
    return node_dir


# ---------------------------------------------------------------------------
# RED: unrouted command topic — gate FAILS
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_unrouted_command_topic_detected(tmp_path: Path) -> None:
    """Gate FAILS when a subscribed command topic has no handler_routing.

    Shape of the June 9 DLQ regression and the June 12 DEL-01 live finding:
    the contract subscribes a command topic but has no handler_routing or
    runtime_dispatch, so messages go to DLQ silently at runtime.
    """
    _write_contract(
        tmp_path,
        "node_synthetic_unrouted",
        """
        name: node_synthetic_unrouted
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.delegate-skill.v1"
        """,
    )

    report: RouteCoverageReport = check_route_coverage([tmp_path])

    assert not report.passed, "Gate should FAIL for unrouted command topic"
    assert len(report.failures) == 1
    assert report.failures[0].topic == "onex.cmd.omnimarket.delegate-skill.v1"
    assert report.failures[0].contract_name == "node_synthetic_unrouted"


@pytest.mark.unit
def test_june9_regression_shape_detected(tmp_path: Path) -> None:
    """Gate catches the exact June 9 DLQ regression shape.

    node_generation_consumer subscribed .node-generation-requested.v1 but had
    no dispatcher route after a handler revert.
    """
    _write_contract(
        tmp_path,
        "node_generation_consumer",
        """
        name: node_generation_consumer
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.node-generation-requested.v1"
        """,
    )

    report: RouteCoverageReport = check_route_coverage([tmp_path])

    assert not report.passed
    assert any(
        f.topic == "onex.cmd.omnimarket.node-generation-requested.v1"
        for f in report.failures
    ), "June 9 regression topic must appear in failures"


# ---------------------------------------------------------------------------
# GREEN: routed command topic — gate PASSES
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_routed_via_handler_routing_passes(tmp_path: Path) -> None:
    """Gate PASSES when handler_routing is declared on the contract."""
    _write_contract(
        tmp_path,
        "node_delegate_skill",
        """
        name: node_delegate_skill
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.delegate-skill.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - event_type: omnimarket.delegate-skill
              handler:
                name: HandlerDelegateSkill
                module: omnimarket.nodes.node_delegate_skill.handlers.handler_delegate_skill
        """,
    )

    report: RouteCoverageReport = check_route_coverage([tmp_path])

    assert report.passed, f"Gate should PASS; failures={report.failures}"
    assert len(report.failures) == 0


@pytest.mark.unit
def test_routed_via_runtime_dispatch_passes(tmp_path: Path) -> None:
    """Gate PASSES when runtime_dispatch is declared on the contract."""
    _write_contract(
        tmp_path,
        "node_pattern_b",
        """
        name: node_pattern_b
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnibase-infra.invocation.v1"
        runtime_dispatch:
          strategy: pattern_b
          broker_class: RuntimePatternBBroker
        """,
    )

    report: RouteCoverageReport = check_route_coverage([tmp_path])

    assert report.passed, (
        f"Gate should PASS with runtime_dispatch; failures={report.failures}"
    )


# ---------------------------------------------------------------------------
# COMPAT: compatibility_publish_topics are sender-side only (OMN-12880)
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_compat_publish_topic_not_flagged(tmp_path: Path) -> None:
    """OMN-12880: compatibility_publish_topics MUST NOT be treated as gaps.

    A contract that only declares a compatibility_publish_topic (sender-side)
    must not cause the gate to report a missing dispatcher route.
    """
    _write_contract(
        tmp_path,
        "node_compat_publisher",
        """
        name: node_compat_publisher
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        compatibility_publish_topics:
          - "onex.cmd.omniclaude.task-delegated.v1"
        """,
    )

    report: RouteCoverageReport = check_route_coverage([tmp_path])

    assert report.passed, (
        "Gate must NOT flag compatibility_publish_topics as gaps; "
        f"failures={report.failures}"
    )


@pytest.mark.unit
def test_event_topic_not_flagged(tmp_path: Path) -> None:
    """Event (evt) topics are not command topics and must be ignored by the gate."""
    _write_contract(
        tmp_path,
        "node_event_only",
        """
        name: node_event_only
        node_type: REDUCER_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.evt.omnimarket.node-generation-completed.v1"
        """,
    )

    report: RouteCoverageReport = check_route_coverage([tmp_path])

    assert report.passed, "Gate must not flag evt topics; failures={report.failures}"


# ---------------------------------------------------------------------------
# CHANGED-CONTRACT MODE (OMN-12879): only changed contracts are checked
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_changed_contract_mode_only_checks_specified(tmp_path: Path) -> None:
    """OMN-12879: in changed-contract mode, only specified paths are checked.

    Two unrouted contracts exist; only the one listed in changed_contract_paths
    should appear as a failure.
    """
    dir_a = _write_contract(
        tmp_path,
        "node_a",
        """
        name: node_a
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.aislop-sweep-start.v1"
        """,
    )
    _write_contract(
        tmp_path,
        "node_b",
        """
        name: node_b
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.bus-audit-start.v1"
        """,
    )

    # Only node_a is "changed" in this PR
    changed = frozenset([dir_a / "contract.yaml"])
    report: RouteCoverageReport = check_route_coverage(
        [tmp_path], changed_contract_paths=changed
    )

    # Only node_a should be in failures; node_b is unchanged and skipped
    assert not report.passed
    failure_topics = {f.topic for f in report.failures}
    assert "onex.cmd.omnimarket.aislop-sweep-start.v1" in failure_topics
    assert "onex.cmd.omnimarket.bus-audit-start.v1" not in failure_topics, (
        "Unchanged contract node_b must not appear in failures"
    )


@pytest.mark.unit
def test_changed_contract_mode_passes_when_routed(tmp_path: Path) -> None:
    """OMN-12879: gate PASSES in changed-contract mode when changed contract is routed."""
    node_dir = _write_contract(
        tmp_path,
        "node_routed_changed",
        """
        name: node_routed_changed
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.compliance-sweep-start.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - event_type: omnimarket.compliance-sweep-start
              handler:
                name: HandlerComplianceSweep
                module: omnimarket.nodes.node_compliance_sweep.handlers.handler_compliance_sweep
        """,
    )

    changed = frozenset([node_dir / "contract.yaml"])
    report: RouteCoverageReport = check_route_coverage(
        [tmp_path], changed_contract_paths=changed
    )

    assert report.passed, f"Gate should PASS; failures={report.failures}"


# ---------------------------------------------------------------------------
# ALLOWLIST: pre-existing violations are not re-flagged
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_allowlisted_topic_not_in_failures(tmp_path: Path) -> None:
    """Known pre-existing violations in the allowlist must not appear in failures."""
    _write_contract(
        tmp_path,
        "node_pattern_b_broker",
        """
        name: node_pattern_b_broker
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnibase-infra.pattern-b-dispatch.v1"
        """,
    )

    report: RouteCoverageReport = check_route_coverage([tmp_path])

    assert report.passed, (
        "Allowlisted topic onex.cmd.omnibase-infra.pattern-b-dispatch.v1 "
        f"must not cause gate failure; failures={report.failures}"
    )
    assert len(report.allowlisted) == 1, (
        f"Expected 1 allowlisted entry; got {report.allowlisted}"
    )


# ---------------------------------------------------------------------------
# MULTI-DIR: both infra and omnimarket contracts scanned together
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_multiple_dirs_all_routed_passes(tmp_path: Path) -> None:
    """Gate PASSES when all contracts across multiple directories are routed."""
    infra_dir = tmp_path / "infra_nodes"
    market_dir = tmp_path / "market_nodes"

    _write_contract(
        infra_dir,
        "node_infra_effect",
        """
        name: node_infra_effect
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnibase-infra.delegation-request.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - event_type: omnibase-infra.delegation-request
              handler:
                name: HandlerDelegationRequestConsumer
                module: omnibase_infra.nodes.node_delegation_effect.handlers.handler_delegation_request_consumer
        """,
    )
    _write_contract(
        market_dir,
        "node_market_effect",
        """
        name: node_market_effect
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.node-generation-requested.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - event_type: omnimarket.node-generation-requested
              handler:
                name: HandlerNodeGenerationConsumer
                module: omnimarket.nodes.node_generation_consumer.handlers.handler_node_generation_consumer
        """,
    )

    report: RouteCoverageReport = check_route_coverage([infra_dir, market_dir])

    assert report.passed, f"Gate should PASS; failures={report.failures}"
    assert len(report.scanned) == 2


@pytest.mark.unit
def test_multiple_dirs_one_unrouted_fails(tmp_path: Path) -> None:
    """Gate FAILS when one contract across multiple directories is unrouted."""
    infra_dir = tmp_path / "infra_nodes"
    market_dir = tmp_path / "market_nodes"

    _write_contract(
        infra_dir,
        "node_infra_routed",
        """
        name: node_infra_routed
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnibase-infra.delegation-request.v1"
        handler_routing:
          routing_strategy: operation_match
          handlers:
            - event_type: omnibase-infra.delegation-request
              handler:
                name: HandlerDelegationRequestConsumer
                module: omnibase_infra.nodes.node_delegation_effect.handlers.handler_delegation_request_consumer
        """,
    )
    _write_contract(
        market_dir,
        "node_market_unrouted",
        """
        name: node_market_unrouted
        node_type: EFFECT_GENERIC
        contract_version: {major: 1, minor: 0, patch: 0}
        event_bus:
          subscribe_topics:
            - "onex.cmd.omnimarket.delegate-skill.v1"
        """,
    )

    report: RouteCoverageReport = check_route_coverage([infra_dir, market_dir])

    assert not report.passed
    failure_topics = {f.topic for f in report.failures}
    assert "onex.cmd.omnimarket.delegate-skill.v1" in failure_topics
    assert "onex.cmd.omnibase-infra.delegation-request.v1" not in failure_topics


# ---------------------------------------------------------------------------
# LIVE REGRESSION PROOF: run gate against real contracts in the repo tree
# Passes only when the current checked-out contracts are fully wired.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_live_infra_contracts_pass(tmp_path: Path) -> None:
    """Gate PASSES against the actual omnibase_infra contract tree.

    This test is the proof that would have blocked the June 9 DLQ regression
    and the June 12 DEL-01 finding had it been wired as a pre-merge check.
    """
    infra_nodes_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "omnibase_infra"
        / "nodes"
    )

    if not infra_nodes_dir.is_dir():
        pytest.skip(f"omnibase_infra nodes dir not found: {infra_nodes_dir}")

    report: RouteCoverageReport = check_route_coverage([infra_nodes_dir])

    assert report.passed, (
        "Live omnibase_infra contracts have unrouted command topics:\n"
        + "\n".join(f"  {f.topic} ({f.contract_name})" for f in report.failures)
    )
