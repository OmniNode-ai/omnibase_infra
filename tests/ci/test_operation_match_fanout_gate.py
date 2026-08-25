# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression for the generic operation_match fan-out gate (OMN-16088).

Split from OMN-15978's AC4: that ticket fixed the fan-through defect contract-side for
``node_gateway_attach_effect`` alone (each of its 3 operation_match entries pinned to
its own ``topic:``). This proves the GENERIC gate — the one that fails the same shape
on ANY contract, not just that one node — with three things, all against the REAL
detector + REAL production helper:

1. RED: an operation_match entry with no topic/event_type scoping, on a contract with
   more than one subscribe topic, is flagged (the exact node_gateway_attach_effect
   PRE-fix shape — 3 operation-scoped handlers, 0 topic pinning, 3 subscribe topics).
2. GREEN: the FIXED shape (each entry pinned to its own topic) is NOT flagged.
3. GREEN: the real, currently-discovered ``node_gateway_attach_effect`` contract
   passes the gate unchanged — no regression on the node OMN-15978 fixed.

Plus a live-repo assertion that the seeded baseline is green day-1 (the
WARN-on-baseline / hard-fail-on-growth ratchet).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.validators.operation_match_fanout import (
    main,
    operation_match_fanout_findings,
)

pytestmark = pytest.mark.unit

_ATTACH_TOPIC = "onex.cmd.omnibase-infra.gateway-attach-request.v1"
_HEARTBEAT_TOPIC = "onex.cmd.omnibase-infra.gateway-heartbeat-request.v1"
_DETACH_TOPIC = "onex.cmd.omnibase-infra.gateway-detach-request.v1"


def _entry(operation: str, *, topic: str | None = None) -> ModelHandlerRoutingEntry:
    return ModelHandlerRoutingEntry(
        operation=operation,
        topic=topic,
        handler=ModelHandlerRef(name=f"Handler{operation.title()}", module="fake"),
    )


def _contract(
    *,
    name: str,
    subscribe_topics: tuple[str, ...],
    handlers: tuple[ModelHandlerRoutingEntry, ...],
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/tmp/omn-16088/contract.yaml"),  # noqa: S108
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe_topics, publish_topics=()
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match", handlers=handlers
        ),
    )


def test_gate_flags_unscoped_operation_match_fanout() -> None:
    """RED: the pre-OMN-15978 node_gateway_attach_effect shape — 3 operation_match
    entries, none pinned to a topic, 3 subscribe topics — is flagged for each entry."""
    contract = _contract(
        name="node_x_effect",
        subscribe_topics=(_ATTACH_TOPIC, _HEARTBEAT_TOPIC, _DETACH_TOPIC),
        handlers=(
            _entry("gateway.attach"),
            _entry("gateway.heartbeat"),
            _entry("gateway.detach"),
        ),
    )
    findings = operation_match_fanout_findings([contract])
    assert {f.operation for f in findings} == {
        "gateway.attach",
        "gateway.heartbeat",
        "gateway.detach",
    }
    assert all(f.contract == "node_x_effect" for f in findings)
    assert all(f.topic_count == 3 for f in findings)


def test_gate_passes_topic_pinned_entries() -> None:
    """GREEN: the FIXED shape — each entry pinned to its own topic — is NOT flagged."""
    contract = _contract(
        name="node_x_effect_fixed",
        subscribe_topics=(_ATTACH_TOPIC, _HEARTBEAT_TOPIC, _DETACH_TOPIC),
        handlers=(
            _entry("gateway.attach", topic=_ATTACH_TOPIC),
            _entry("gateway.heartbeat", topic=_HEARTBEAT_TOPIC),
            _entry("gateway.detach", topic=_DETACH_TOPIC),
        ),
    )
    assert operation_match_fanout_findings([contract]) == []


def test_gate_passes_single_subscribe_topic() -> None:
    """GREEN: an unscoped operation_match entry on a contract with only ONE subscribe
    topic is not a fan-out risk — there is nowhere else for it to fan to."""
    contract = _contract(
        name="node_single_topic",
        subscribe_topics=(_ATTACH_TOPIC,),
        handlers=(_entry("gateway.attach"),),
    )
    assert operation_match_fanout_findings([contract]) == []


def test_real_node_gateway_attach_effect_passes_unchanged() -> None:
    """No regression on the node OMN-15978 fixed: the real discovered contract must
    pass this generic gate exactly as it passes its own node-scoped regression test."""
    manifest = discover_contracts()
    matches = [c for c in manifest.contracts if c.name == "node_gateway_attach_effect"]
    assert len(matches) == 1, (
        f"expected one discovered node_gateway_attach_effect contract, found "
        f"{len(matches)}"
    )
    assert operation_match_fanout_findings(matches) == []


def test_seeded_baseline_is_green_day_one() -> None:
    """The live repo scan against the seeded baseline exits 0 (WARN-on-baseline).

    A non-zero exit here means either a new offender slipped in (growth) or the
    baseline went stale (a fixed entry still listed) — both are ratchet failures
    this gate must surface, and both mean the seed drifted from reality."""
    repo_root = Path(__file__).resolve().parents[2]
    scan_root = repo_root / "src" / "omnibase_infra"
    baseline = (
        repo_root / "config" / "validation" / "operation_match_fanout_baseline.yaml"
    )
    assert main([str(scan_root), "--baseline", str(baseline)]) == 0
