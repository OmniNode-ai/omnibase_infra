# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-13852 — six guard-tripped orchestrators no longer silently drop events.

Deterministic proof for the fix independent of the parity oracle:

* Routing (defect 1): every handler entry on the six formerly-guard-tripped
  orchestrators now resolves to a non-empty subscribe-topic set, so a
  ``ModelDispatchRoute`` registers per handler. Before OMN-13852 the multi-handler
  ambiguity guard (``_topics_for_handler_entry`` -> ``()``) registered ZERO routes.
* End-to-end dispatch: the exact live-proven P0-1 scenario — a real
  ``ModelRsdScoreResult`` payload on ``onex.evt.rsd.scores-calculated.v1`` — now
  DISPATCHES (status != NO_DISPATCHER) instead of falling through to a DLQ topic
  that never existed.
* DLQ integrity (defect 2): each contract declares ``event_bus.dlq_topics`` and the
  declared topic matches the DLQ topic the engine derives for that contract's
  event_type domain on a NO_DISPATCHER fall-through — so the dead-letter escape
  hatch targets a provisioned topic.
"""

from __future__ import annotations

import asyncio

import pytest

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.topic_constants import derive_dlq_topic_for_event_type
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _read_dlq_topics,
    _topics_for_handler_entry,
)
from tests.fixtures.dispatch_parity import harness

pytestmark = pytest.mark.integration

# The six multi-handler orchestrators OMN-13852 disambiguated. Value = the
# event_type domain whose derived DLQ topic each contract must declare.
_SIX_ORCHESTRATORS: dict[str, str] = {
    "node_rsd_orchestrator": "rsd",
    "node_routing_orchestrator": "router",
    "node_merge_sweep_workflow_orchestrator": "skill",
    "node_chain_orchestrator": "omnibase-infra",
    "node_registration_orchestrator": "platform",
    "node_scope_workflow_orchestrator": "skill",
}


@pytest.fixture(scope="module")
def contracts_by_name() -> dict[str, object]:
    manifest = discover_contracts()
    return {c.name: c for c in manifest.contracts}


@pytest.mark.parametrize("node_name", sorted(_SIX_ORCHESTRATORS))
def test_every_handler_registers_a_route(
    contracts_by_name: dict[str, object], node_name: str
) -> None:
    """Every handler entry resolves to a non-empty topic set (route registers)."""
    contract = contracts_by_name[node_name]
    handlers = contract.handler_routing.handlers  # type: ignore[attr-defined]
    unrouted = [
        entry.handler.name
        for entry in handlers
        if not _topics_for_handler_entry(contract, entry)  # type: ignore[arg-type]
    ]
    assert not unrouted, (
        f"{node_name} has handler entries that register ZERO dispatch routes "
        f"(the OMN-13852 silent-drop signature): {unrouted}. Each needs a "
        "topic-derived event_type alias in contract.yaml."
    )


@pytest.mark.parametrize("node_name", sorted(_SIX_ORCHESTRATORS))
def test_declares_matching_dlq_topic(
    contracts_by_name: dict[str, object], node_name: str
) -> None:
    """The declared DLQ topic matches the engine-derived NO_DISPATCHER target."""
    contract = contracts_by_name[node_name]
    declared = _read_dlq_topics(contract.contract_path)  # type: ignore[attr-defined]
    domain = _SIX_ORCHESTRATORS[node_name]
    # Any inbound event_type in this contract's domain derives the same DLQ topic.
    expected = derive_dlq_topic_for_event_type(
        event_type=f"{domain}.example-event", original_topic=""
    )
    assert expected is not None
    assert expected in declared, (
        f"{node_name} must declare its NO_DISPATCHER DLQ topic {expected!r} in "
        f"event_bus.dlq_topics so the escape hatch is provisioned; got {declared}."
    )


def test_live_p0_1_rsd_score_result_now_dispatches(
    contracts_by_name: dict[str, object],
) -> None:
    """The exact live-proven drop now dispatches.

    Reproduces the 2026-07-02 stability-lane P0-1 probe deterministically in-process:
    a real ``ModelRsdScoreResult`` on ``onex.evt.rsd.scores-calculated.v1`` was
    logged as 'No dispatcher found ... routing to DLQ topic
    onex.dlq.omnibase-infra.rsd.v1'. After the fix it must select a dispatcher.
    """
    rsd = contracts_by_name["node_rsd_orchestrator"]
    engine, _dispatcher_meta, _routes, _topics = harness._build_engine([rsd])

    topic = "onex.evt.rsd.scores-calculated.v1"
    # Build the real event_model instance HandlerRsdScoreComplete consumes.
    instance, err = harness._model_construct_instance(
        "omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_score_result",
        "ModelRsdScoreResult",
    )
    assert err is None and instance is not None, f"could not construct payload: {err}"

    async def _run() -> ModelDispatchResult:
        envelope = harness._envelope(
            event_type="rsd.scores-calculated", payload=instance
        )
        return await engine.dispatch(topic=topic, envelope=envelope)

    result = asyncio.run(_run())
    assert result.status != EnumDispatchStatus.NO_DISPATCHER, (
        "node_rsd_orchestrator STILL drops the live P0-1 event to DLQ "
        f"(dlq_topic={getattr(result, 'dlq_topic', None)}); the routing fix regressed."
    )
    assert result.status == EnumDispatchStatus.SUCCESS, (
        f"expected SUCCESS after routing fix, got {result.status}"
    )
