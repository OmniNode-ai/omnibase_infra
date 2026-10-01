# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-13852 — six guard-tripped orchestrators no longer silently drop events.

Deterministic proof for the fix independent of the parity oracle:

* Routing (defect 1): every handler entry on the six formerly-guard-tripped
  orchestrators now resolves to a non-empty subscribe-topic set, so a
  ``ModelDispatchRoute`` registers per handler. Before OMN-13852 the multi-handler
  ambiguity guard (``_topics_for_handler_entry`` -> ``()``) registered ZERO routes.
* End-to-end dispatch: a real event payload on a multi-handler orchestrator's
  own EVENT topic DISPATCHES (status != NO_DISPATCHER) instead of falling through
  to a DLQ topic. The live-proven P0-1 instance of the drop was
  ``ModelRsdScoreResult`` on ``onex.evt.rsd.scores-calculated.v1``; OMN-17427
  deleted that subscription because nothing in the fleet publishes the topic, so
  the same property is now pinned on node_chain_orchestrator's
  ``onex.evt.omnibase-infra.chain-retrieval-result.v1``, which
  node_chain_retrieval_effect does publish.
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


def test_multi_handler_orchestrator_event_payload_dispatches(
    contracts_by_name: dict[str, object],
) -> None:
    """A real event payload on a multi-handler orchestrator's own topic dispatches.

    The 2026-07-02 stability-lane P0-1 probe logged a real ``ModelRsdScoreResult``
    on ``onex.evt.rsd.scores-calculated.v1`` as 'No dispatcher found ... routing to
    DLQ topic onex.dlq.omnibase-infra.rsd.v1'. OMN-17427 deleted that subscription
    (no contract and no runtime path in the fleet publishes the topic), so the same
    regression is pinned on node_chain_orchestrator: three handler entries, one of
    them an EVENT handler whose topic has a declared publisher
    (node_chain_retrieval_effect). Before OMN-13852 the multi-handler ambiguity
    guard registered zero routes here too.
    """
    chain = contracts_by_name["node_chain_orchestrator"]
    assert len(chain.handler_routing.handlers) > 1  # type: ignore[attr-defined]
    engine, _dispatcher_meta, _routes, _topics = harness._build_engine([chain])

    topic = "onex.evt.omnibase-infra.chain-retrieval-result.v1"
    # Build the real event_model instance HandlerChainRetrievalComplete consumes.
    instance, err = harness._model_construct_instance(
        "omnibase_infra.nodes.node_chain_orchestrator.models.model_chain_retrieval_result",
        "ModelChainRetrievalResult",
    )
    assert err is None and instance is not None, f"could not construct payload: {err}"

    async def _run() -> ModelDispatchResult:
        envelope = harness._envelope(
            event_type="omnibase-infra.chain-retrieval-result", payload=instance
        )
        return await engine.dispatch(topic=topic, envelope=envelope)

    result = asyncio.run(_run())
    assert result.status != EnumDispatchStatus.NO_DISPATCHER, (
        "node_chain_orchestrator drops a real event payload to DLQ "
        f"(dlq_topic={getattr(result, 'dlq_topic', None)}); the OMN-13852 routing "
        "fix regressed."
    )
    assert result.status == EnumDispatchStatus.SUCCESS, (
        f"expected SUCCESS after routing fix, got {result.status}"
    )
