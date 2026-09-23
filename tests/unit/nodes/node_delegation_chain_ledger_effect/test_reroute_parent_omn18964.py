# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A re-routed delegation's chain must replay green (OMN-18964).

What was measured
-----------------
Over the last 30 chains on the .201 compose dev lane, every 5-hop chain
replayed green (18 of 18) and every 6-hop chain replayed red (12 of 12). The
sixth hop is a SECOND ``delegation-routing-request``: the sanctioned OMN-14234
free-tier best-of-N re-route, emitted by the delegation orchestrator while it
consumes a failing ``quality-gate-result`` (score 0.720 against a 0.800 bar,
``decision=climb reason=acceptance_criteria_failed``). The delegation
COMPLETED on the second draft. Only its chain grade was wrong.

The re-route's recorded parent is that quality-gate-result envelope, and it
is correct: ``uuid5(correlation, "ModelQualityGateResult:0")`` reproduces it
exactly on two independent correlations. The hop replayed red for two
separate reasons, and fixing either alone leaves it red:

1. ``quality-gate-result.v1`` was never projected into ``event_ledger``, so
   the chain writer could not observe the parent at all.
2. ``chain_topology`` gave the routing request exactly one parent, so even an
   observed quality-gate-result envelope was never a candidate for it.

The fixture below is the recorded chain of chain-canary run 35762660734,
correlation ``f8d87788-468c-4b3c-a189-e6e86cdb393d``, read out of
``public.ledger_chain`` on the dev lane, plus the quality-gate-result envelope
the fix makes observable. It is graded against the REAL contract topology, not
a test-local one, so the declaration under test is the one the writer loads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.chain_replay import (
    assemble_replay_and_verify,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    HandlerDelegationChainLedger,
    _parse_declared_topology,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.enum_tier_two_verdict import (
    EnumTierTwoVerdict,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_observed_hop import (
    ModelObservedHop,
)

pytestmark = pytest.mark.unit

_NODES = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"
_CHAIN_LEDGER = _NODES / "node_delegation_chain_ledger_effect" / "contract.yaml"
_LEDGER_PROJECTION = _NODES / "node_ledger_projection_compute" / "contract.yaml"

SKILL = "onex.cmd.omnimarket.delegate-skill.v1"
REQUEST = "onex.cmd.omnibase-infra.delegation-request.v1"
ROUTING_REQUEST = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
ROUTING_DECISION = "onex.evt.omnibase-infra.routing-decision.v1"
COMPLETED = "onex.evt.omnimarket.delegate-skill-completed.v1"
QUALITY_GATE_RESULT = "onex.evt.omnibase-infra.quality-gate-result.v1"

# Recorded, chain-canary run 35762660734.
CORRELATION = UUID("f8d87788-468c-4b3c-a189-e6e86cdb393d")
E_SKILL = UUID("071b9142-e2c0-4fc3-8dd6-6037ed6ceb94")
E_REQUEST = UUID("c4552b9d-2ce1-4d1a-ba35-71d35d715d20")
E_ROUTING_REQUEST = UUID("4b673501-e367-57e6-b2c7-da783c45bee8")
E_ROUTING_DECISION = UUID("42b03274-02b8-5f3e-9293-1b6d87f1e3e9")
E_REROUTE = UUID("c15242e1-cecd-5936-8418-9d34790ea90b")
E_COMPLETED = UUID("421c9026-828f-53ed-ba50-563c82351c0f")
# uuid5(CORRELATION, "ModelQualityGateResult:0") -- the re-route's recorded
# parent, never present in event_ledger before this fix.
E_QUALITY_GATE_RESULT = UUID("99fb6966-9fd9-57e3-b9fe-dadd1de7fcb7")
UNRELATED = UUID("bbbbbbbb-0000-0000-0000-00000000ffff")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _declared() -> tuple[ModelDeclaredChainHop, ...]:
    return _parse_declared_topology(_load(_CHAIN_LEDGER)["chain_topology"])


def _hop(topic: str, envelope_id: UUID, parent: UUID | None) -> ModelObservedHop:
    return ModelObservedHop(
        topic=topic,
        envelope_id=envelope_id,
        parent_envelope_id=parent,
        correlation_id=CORRELATION,
    )


def _five_hop_chain() -> list[ModelObservedHop]:
    return [
        _hop(SKILL, E_SKILL, None),
        _hop(REQUEST, E_REQUEST, E_SKILL),
        _hop(ROUTING_REQUEST, E_ROUTING_REQUEST, E_REQUEST),
        _hop(ROUTING_DECISION, E_ROUTING_DECISION, E_ROUTING_REQUEST),
        _hop(COMPLETED, E_COMPLETED, E_SKILL),
    ]


def _rerouted_chain(*, with_quality_gate_result: bool) -> list[ModelObservedHop]:
    """The recorded 6-hop chain, in observed order.

    ``with_quality_gate_result`` is what the projection fix changes: before it,
    ``event_ledger`` held no quality-gate-result row, so the writer's read
    returned exactly the six hops ``ledger_chain`` recorded.
    """
    chain = _five_hop_chain()[:4]
    if with_quality_gate_result:
        # Its own parent (the quality-gate REQUEST) is not projected and is not
        # graded: this envelope is read as the re-route's parent evidence only.
        chain.append(_hop(QUALITY_GATE_RESULT, E_QUALITY_GATE_RESULT, None))
    chain.append(_hop(ROUTING_REQUEST, E_REROUTE, E_QUALITY_GATE_RESULT))
    chain.append(_hop(COMPLETED, E_COMPLETED, E_SKILL))
    return chain


def test_the_recorded_rerouted_chain_replays_green_against_the_contract() -> None:
    """AC3, both halves together: the re-route edge closes, every row green."""
    rows = assemble_replay_and_verify(
        CORRELATION, _rerouted_chain(with_quality_gate_result=True), _declared()
    )

    red = [
        (r.hop_index, r.observed_topic, r.replay_detail)
        for r in rows
        if not r.replay_green
    ]
    assert not red, f"the recorded re-routed chain still replays red: {red}"
    assert all(r.verifier_verdict is EnumTierTwoVerdict.PASS for r in rows)
    # The parent evidence is not a graded hop: the chain is still SIX rows,
    # so nothing that counts hops or reads the last row changes meaning.
    assert [r.observed_topic for r in rows] == [
        SKILL,
        REQUEST,
        ROUTING_REQUEST,
        ROUTING_DECISION,
        ROUTING_REQUEST,
        COMPLETED,
    ]
    assert rows[4].parent_envelope_id == E_QUALITY_GATE_RESULT


def test_the_five_hop_chain_still_replays_green() -> None:
    """Control: the no-re-route shape is unchanged by the fix."""
    rows = assemble_replay_and_verify(CORRELATION, _five_hop_chain(), _declared())
    assert len(rows) == 5
    assert all(r.replay_green for r in rows)
    assert all(r.verifier_verdict is EnumTierTwoVerdict.PASS for r in rows)


def test_the_rerouted_chain_is_red_when_its_parent_was_never_projected() -> None:
    """Negative control: the topology half ALONE does not green the chain.

    Without the quality-gate-result row there is no observed envelope for the
    re-route edge to close against, so the hop stays red with a stated reason.
    """
    rows = assemble_replay_and_verify(
        CORRELATION, _rerouted_chain(with_quality_gate_result=False), _declared()
    )
    assert not rows[4].replay_green
    assert QUALITY_GATE_RESULT in rows[4].replay_detail


def test_a_reroute_citing_an_unrelated_envelope_is_still_red() -> None:
    """Negative control: accepting a re-route is not accepting any parent."""
    chain = _rerouted_chain(with_quality_gate_result=True)
    chain[5] = _hop(ROUTING_REQUEST, E_REROUTE, UNRELATED)
    rows = assemble_replay_and_verify(CORRELATION, chain, _declared())
    assert not rows[4].replay_green
    assert "does not close" in rows[4].replay_detail


def test_the_first_routing_request_may_not_cite_a_reroute_parent() -> None:
    """A re-route is a REPEAT. The first routing request has one cause.

    The first occurrence of the hop closes against its declared parent only;
    a quality-gate verdict cannot have caused the routing it was a verdict on.
    """
    chain = [
        _hop(SKILL, E_SKILL, None),
        _hop(REQUEST, E_REQUEST, E_SKILL),
        _hop(QUALITY_GATE_RESULT, E_QUALITY_GATE_RESULT, None),
        _hop(ROUTING_REQUEST, E_ROUTING_REQUEST, E_QUALITY_GATE_RESULT),
        _hop(ROUTING_DECISION, E_ROUTING_DECISION, E_ROUTING_REQUEST),
        _hop(COMPLETED, E_COMPLETED, E_SKILL),
    ]
    rows = assemble_replay_and_verify(CORRELATION, chain, _declared())
    assert not rows[2].replay_green
    assert rows[2].observed_topic == ROUTING_REQUEST


def test_the_contract_declares_the_reroute_parent_on_the_routing_request() -> None:
    """The declaration is the single source; the writer must READ it."""
    declared = {hop.topic: hop for hop in _declared()}
    assert QUALITY_GATE_RESULT in declared[ROUTING_REQUEST].reroute_parents
    # Five declared hops, unchanged: EXPECTED_LEDGER_HOPS and the canary's
    # completeness check read the declared hops, and a re-route parent is not
    # a hop every chain must traverse.
    assert len(_declared()) == 5
    assert QUALITY_GATE_RESULT not in declared


def test_the_writer_reads_the_reroute_parent_but_does_not_wait_for_it() -> None:
    """Read set includes the parent evidence; settle does not require it.

    A 5-hop chain has no quality-gate-result on the re-route path, so a settle
    rule that waited for one would burn its whole budget on every dispatch.
    """
    handler = HandlerDelegationChainLedger.__new__(HandlerDelegationChainLedger)
    handler._declared_chain = _declared()
    assert QUALITY_GATE_RESULT in handler._declared_topics()
    assert handler._observation_is_settled(
        {SKILL, REQUEST, ROUTING_REQUEST, ROUTING_DECISION, COMPLETED}
    )


def test_every_topic_the_chain_writer_reads_is_projected_into_event_ledger() -> None:
    """Producer half: a topic the writer reads must reach ``event_ledger``.

    ``node_ledger_projection_compute`` is the only writer of that relation. A
    topic the chain writer reads but the projection does not subscribe to AND
    route (OMN-14594 pairing) is empty by construction -- the class this
    defect is the third instance of, after OMN-18419 and OMN-18937.
    """
    read_set = {topic for hop in _declared() for topic in hop.topics}
    read_set.update(parent for hop in _declared() for parent in hop.reroute_parents)
    # Positive control: the derivation sees the re-route parent, so an empty
    # difference below is evidence and not a vacuous pass.
    assert QUALITY_GATE_RESULT in read_set

    projection = _load(_LEDGER_PROJECTION)
    subscribed = set(projection["event_bus"]["subscribe_topics"])
    routed = {entry["topic"] for entry in projection["handler_routing"]["handlers"]}
    assert sorted(read_set - subscribed) == []
    assert sorted(read_set - routed) == []


@pytest.mark.parametrize(
    ("reroute_parents", "reason"),
    [
        (("",), "empty re-route parent"),
        ((ROUTING_REQUEST,), "names itself as a re-route parent"),
        ((QUALITY_GATE_RESULT, QUALITY_GATE_RESULT), "repeats a re-route parent"),
    ],
)
def test_a_malformed_reroute_declaration_is_refused(
    reroute_parents: tuple[str, ...], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        ModelDeclaredChainHop(
            topic=ROUTING_REQUEST, parent=REQUEST, reroute_parents=reroute_parents
        )


# Recorded on the .201 dev lane while this fix was hotpatched, 2026-09-22,
# correlation 4eec9ee4-6740-4123-80d5-038d3f9e71e7: a delegation that climbed
# every rung and FAILED. It carries both re-route kinds -- one caused by a
# failing quality-gate-result, one by an inference-response
# (uuid5(correlation, "ModelInferenceResponseData:0") reproduces 38272eda) --
# plus the redeliveries event_ledger holds as repeated envelope ids.
ESC_CORRELATION = UUID("4eec9ee4-6740-4123-80d5-038d3f9e71e7")
INFERENCE_RESPONSE = "onex.evt.omnibase-infra.inference-response.v1"
FAILED = "onex.evt.omnimarket.delegate-skill-failed.v1"


def _escalated_chain() -> list[ModelObservedHop]:
    def hop(topic: str, envelope: str, parent: str | None) -> ModelObservedHop:
        return ModelObservedHop(
            topic=topic,
            envelope_id=UUID(envelope),
            parent_envelope_id=UUID(parent) if parent else None,
            correlation_id=ESC_CORRELATION,
        )

    skill = "2e876d50-dfd1-408e-8032-1ad8cfad0a73"
    request = "18edbba0-e409-4d03-92fc-184cbb6c8491"
    first_routing = "f9607e0d-f65f-54f1-9f04-491d2c28af44"
    decision = "4f555f01-a917-5c0c-951a-eb8dff19f937"
    gate = "8d2e6ebe-0e90-50e7-b739-e8b506f675a6"
    gate_reroute = "e5d8b64b-2309-58bd-89dc-12bca8355692"
    inference = "38272eda-fbea-5b36-9243-751a4229d84b"
    inference_reroute = "7342e282-353b-5166-b306-3b350fb142fa"
    return [
        hop(SKILL, skill, None),
        hop(REQUEST, request, skill),
        hop(ROUTING_REQUEST, first_routing, request),
        hop(ROUTING_DECISION, decision, first_routing),
        hop(QUALITY_GATE_RESULT, gate, "151919f3-d00f-5187-9663-61ef1a93d226"),
        hop(ROUTING_REQUEST, gate_reroute, gate),
        hop(ROUTING_DECISION, decision, gate_reroute),
        hop(QUALITY_GATE_RESULT, gate, "151919f3-d00f-5187-9663-61ef1a93d226"),
        hop(ROUTING_REQUEST, gate_reroute, gate),
        hop(INFERENCE_RESPONSE, inference, None),
        hop(ROUTING_REQUEST, inference_reroute, inference),
        hop(ROUTING_DECISION, decision, inference_reroute),
        hop(FAILED, "4cb765ea-7666-5476-a605-2780a952ba33", skill),
    ]


def test_both_reroute_kinds_replay_green_on_the_recorded_escalated_chain() -> None:
    rows = assemble_replay_and_verify(ESC_CORRELATION, _escalated_chain(), _declared())
    red = [(r.hop_index, r.replay_detail) for r in rows if not r.replay_green]
    assert not red, red
    assert [r.observed_topic for r in rows] == [
        SKILL,
        REQUEST,
        ROUTING_REQUEST,
        ROUTING_DECISION,
        ROUTING_REQUEST,
        ROUTING_REQUEST,
        FAILED,
    ]


def test_the_inference_reroute_is_red_without_its_projected_parent() -> None:
    """Control: inference-response is a declared re-route parent, not a pass."""
    chain = [h for h in _escalated_chain() if h.topic != INFERENCE_RESPONSE]
    rows = assemble_replay_and_verify(ESC_CORRELATION, chain, _declared())
    assert [r.replay_green for r in rows] == [True, True, True, True, True, False, True]
    assert INFERENCE_RESPONSE in rows[5].replay_detail
