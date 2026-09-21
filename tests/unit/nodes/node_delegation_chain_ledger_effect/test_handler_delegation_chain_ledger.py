# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hermetic writer tests for OMN-16964 chain-canary link 5."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.models.model_db_query_payload import ModelDbQueryPayload
from omnibase_infra.handlers.models.model_db_query_response import ModelDbQueryResponse
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    HandlerDelegationChainLedger,
    _parse_declared_topology,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    EnumTierTwoVerdict,
    ModelDeclaredChainHop,
    ModelDelegationTerminalPayload,
    ModelLedgerChainRow,
    ModelObservedHop,
)

# OMN-18419: the declaration is hops with a parent relation, not bare topics.
# This fixture is a LINE -- every hop caused by the one before it -- which is
# what `_complete_observation` below builds. The TREE cases are pinned in
# test_chain_replay.py, where the grading rule itself lives.
_CHAIN_TOPICS = ("command", "route-request", "route-decision", "completed")
_CHAIN = tuple(
    ModelDeclaredChainHop(
        topic=topic,
        parent=None if index == 0 else _CHAIN_TOPICS[index - 1],
    )
    for index, topic in enumerate(_CHAIN_TOPICS)
)


def _get_repo_root() -> Path:
    """Resolve the repository root without depending on test-file depth."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repository root from pyproject.toml")


_REPO_ROOT = _get_repo_root()
_CONTRACT_PATH = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_delegation_chain_ledger_effect/contract.yaml"
)
_LEDGER_PROJECTION_CONTRACT_PATH = (
    _REPO_ROOT / "src/omnibase_infra/nodes/node_ledger_projection_compute/contract.yaml"
)


def _assert_no_bus_triggered_undeliverable_output_model(
    contract: dict[str, object],
) -> None:
    """Refuse the OMN-18390 wiring defect on any contract, not just this node.

    A bus-triggered EFFECT contract (``event_bus.subscribe_topics`` non-empty)
    that declares an ``output_model`` with no ``event_bus.publish_topics`` has
    no result applier to deliver the output its handler will produce (auto_wiring
    builds the applier from ``publish_topics`` only). Every successful dispatch
    then dead-letters as ``UndeliverableDispatchOutputError`` even though the
    handler did its job correctly.

    Scoped deliberately to bus-triggered contracts: an ``output_model`` on a
    contract with no ``subscribe_topics`` is never dispatched off the bus (it is
    invoked directly, e.g. by CLI or another node's synchronous call), so it
    never reaches the auto-wiring boundary this defect lives in. A blanket
    "output_model implies publish_topics" rule is false in this codebase: at
    least 36 EFFECT_GENERIC contracts and 21 COMPUTE_GENERIC contracts declare
    an output_model with no publish_topics and are never bus-dispatched, so
    they are not defective.
    """
    event_bus = contract.get("event_bus") or {}
    subscribe_topics = event_bus.get("subscribe_topics") or []
    if not subscribe_topics:
        return
    publish_topics = event_bus.get("publish_topics") or []
    if contract.get("output_model") and not publish_topics:
        raise AssertionError(
            "bus-triggered contract declares output_model "
            f"{contract['output_model']!r} with subscribe_topics "
            f"{subscribe_topics!r} but no publish_topics: every successful "
            "dispatch will produce an output event with no result applier "
            "wired to deliver it (OMN-18390)"
        )


def test_bus_triggered_output_model_without_publish_topics_is_a_wiring_defect() -> None:
    """Class-general fixture pin (AC2): the defect shape is refused generically."""
    defective_contract = {
        "name": "node_fixture_effect",
        "node_type": "EFFECT_GENERIC",
        "output_model": {"name": "ModelFixtureResult", "module": "fixture.models"},
        "event_bus": {
            "subscribe_topics": ["onex.evt.fixture.something-happened.v1"],
            # no publish_topics key at all -- the exact OMN-18390 shape
        },
    }
    with pytest.raises(AssertionError, match="OMN-18390"):
        _assert_no_bus_triggered_undeliverable_output_model(defective_contract)


def test_output_model_without_publish_topics_is_fine_when_not_bus_triggered() -> None:
    """Scope control: no subscribe_topics means never dispatched off the bus."""
    directly_invoked_contract = {
        "name": "node_fixture_compute",
        "node_type": "COMPUTE_GENERIC",
        "output_model": {"name": "ModelFixtureResult", "module": "fixture.models"},
    }
    _assert_no_bus_triggered_undeliverable_output_model(directly_invoked_contract)


def test_output_model_with_declared_publish_topic_is_fine() -> None:
    """Scope control: a real publish_topics entry means an applier can be wired."""
    correctly_wired_contract = {
        "name": "node_fixture_effect",
        "node_type": "EFFECT_GENERIC",
        "output_model": {"name": "ModelFixtureResult", "module": "fixture.models"},
        "event_bus": {
            "subscribe_topics": ["onex.evt.fixture.something-happened.v1"],
            "publish_topics": ["onex.evt.fixture.something-recorded.v1"],
        },
    }
    _assert_no_bus_triggered_undeliverable_output_model(correctly_wired_contract)


def test_this_node_contract_declares_no_undeliverable_output_model() -> None:
    """Node-pinned regression guard (AC1): this exact contract stays fixed."""
    contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert "output_model" not in contract, (
        "node_delegation_chain_ledger_effect must not declare output_model: "
        "nothing consumes ModelLedgerChainWriteResult (OMN-16964 comment, "
        "2026-09-15T11:52:22Z) and the node is bus-triggered with no "
        "publish_topics, so any declared output_model dead-letters every "
        "successful dispatch (OMN-18390)"
    )
    _assert_no_bus_triggered_undeliverable_output_model(contract)


def _handler(
    declared_chain: tuple[ModelDeclaredChainHop, ...] = _CHAIN,
) -> HandlerDelegationChainLedger:
    handler = HandlerDelegationChainLedger(
        cast("ModelONEXContainer", object()),
        db_dsn="postgresql://test.invalid/test",
        declared_chain=declared_chain,
        settle_attempts=1,
        settle_delay_seconds=0,
    )
    handler._ensure_db_ready = AsyncMock()  # type: ignore[method-assign]
    return handler


def _request(correlation_id: UUID) -> ModelDelegationTerminalPayload:
    return ModelDelegationTerminalPayload(
        correlation_id=correlation_id,
        status="completed",
    )


def _complete_observation(
    correlation_id: UUID, *, broken_at: int | None = None
) -> tuple[ModelObservedHop, ...]:
    envelope_ids = tuple(uuid4() for _ in _CHAIN_TOPICS)
    return tuple(
        ModelObservedHop(
            topic=topic,
            envelope_id=envelope_ids[index],
            parent_envelope_id=(
                None
                if index == 0
                else uuid4()
                if index == broken_at
                else envelope_ids[index - 1]
            ),
            correlation_id=correlation_id,
        )
        for index, topic in enumerate(_CHAIN_TOPICS)
    )


@pytest.mark.asyncio
async def test_handle_never_returns_a_bus_publishable_result() -> None:
    """Node-pinned regression guard (AC1/AC2 fallback): no undeliverable output.

    ``ModelHandlerOutput.for_compute(result=...)`` puts a BaseModel on
    ``.result``, and the auto-wiring boundary appends any BaseModel ``.result``
    to ``output_events`` regardless of contract wiring (OMN-18390 mechanism).
    Pin the handler to the effect/no-events shape so it can never regress into
    producing an output event this contract has no applier for.
    """
    correlation_id = uuid4()
    handler = _handler()
    handler._read_observed = AsyncMock(  # type: ignore[method-assign]
        return_value=_complete_observation(correlation_id)
    )
    handler._persist_rows = AsyncMock()  # type: ignore[method-assign]

    output = await handler.handle(_request(correlation_id))

    assert output.node_kind is EnumNodeKind.EFFECT
    assert output.result is None
    assert output.events == ()
    assert output.intents == ()
    assert output.projections == ()


@pytest.mark.asyncio
async def test_complete_chain_writes_green_rows() -> None:
    correlation_id = uuid4()
    handler = _handler()
    handler._read_observed = AsyncMock(  # type: ignore[method-assign]
        return_value=_complete_observation(correlation_id)
    )
    persisted = AsyncMock()
    handler._persist_rows = persisted  # type: ignore[method-assign]

    output = await handler.handle(_request(correlation_id))

    assert output.result is None
    rows = persisted.await_args.args[0]
    assert len(rows) == 4
    assert all(row.replay_green for row in rows)
    assert all(row.verifier_verdict is EnumTierTwoVerdict.PASS for row in rows)
    assert {row.hop for row in rows} == set(_CHAIN_TOPICS)


@pytest.mark.asyncio
async def test_broken_edge_cannot_be_written_as_replay_green() -> None:
    correlation_id = uuid4()
    handler = _handler()
    handler._read_observed = AsyncMock(  # type: ignore[method-assign]
        return_value=_complete_observation(correlation_id, broken_at=2)
    )
    persisted = AsyncMock()
    handler._persist_rows = persisted  # type: ignore[method-assign]

    output = await handler.handle(_request(correlation_id))

    assert output.result is None
    rows = persisted.await_args.args[0]
    assert {row.hop for row in rows} == set(_CHAIN_TOPICS)
    assert rows[2].replay_green is False
    assert "causal link does not close" in rows[2].replay_detail


@pytest.mark.asyncio
async def test_empty_declaration_writes_skip_and_never_passes() -> None:
    correlation_id = uuid4()
    handler = _handler(())
    handler._read_observed = AsyncMock(  # type: ignore[method-assign]
        return_value=_complete_observation(correlation_id)
    )
    persisted = AsyncMock()
    handler._persist_rows = persisted  # type: ignore[method-assign]

    output = await handler.handle(_request(correlation_id))

    assert output.result is None
    rows = persisted.await_args.args[0]
    assert all(row.verifier_verdict is EnumTierTwoVerdict.SKIP for row in rows)
    assert not any(row.verifier_verdict is EnumTierTwoVerdict.PASS for row in rows)


@pytest.mark.asyncio
async def test_unavailable_event_ledger_fails_closed() -> None:
    correlation_id = uuid4()
    handler = _handler()
    handler._read_observed = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("event ledger unavailable")
    )
    handler._persist_rows = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="event ledger unavailable"):
        await handler.handle(_request(correlation_id))

    handler._persist_rows.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_missing_hop_is_persisted_as_incomplete_not_filled() -> None:
    correlation_id = uuid4()
    observed = _complete_observation(correlation_id)
    handler = _handler()
    handler._read_observed = AsyncMock(  # type: ignore[method-assign]
        return_value=observed[:2] + observed[3:]
    )
    persisted = AsyncMock()
    handler._persist_rows = persisted  # type: ignore[method-assign]

    output = await handler.handle(_request(correlation_id))

    assert output.result is None
    rows = persisted.await_args.args[0]
    assert len(rows) == 3
    assert tuple(row.hop for row in rows) == (
        "command",
        "route-request",
        "completed",
    )


# --- OMN-18398: a write that cannot happen is never reported as success ---
#
# `_persist_rows` is a `for row in rows:` loop, so over an empty row set it
# executes no statement at all, and `handle()` returned
# `ModelHandlerOutput.for_effect(...)` unconditionally afterwards. Zero rows
# written was therefore indistinguishable from four rows written, and on the
# .201 compose dev lane that is the state every dispatch was in: the evidence
# read is structurally empty because `node_ledger_projection_compute` (the only
# writer of `public.event_ledger`) subscribed to none of this contract's four
# `chain_topology` topics, so `public.ledger_chain` held zero rows while the
# handler reported success on every dispatch.


@pytest.mark.asyncio
async def test_absent_evidence_is_a_typed_refusal_not_a_silent_success() -> None:
    """AC2: zero observed hops must refuse, naming the topics it looked for."""
    correlation_id = uuid4()
    handler = _handler()
    handler._read_observed = AsyncMock(return_value=())  # type: ignore[method-assign]
    persisted = AsyncMock()
    handler._persist_rows = persisted  # type: ignore[method-assign]

    with pytest.raises(RuntimeHostError) as excinfo:
        await handler.handle(_request(correlation_id))

    message = str(excinfo.value)
    assert str(correlation_id) in message
    for topic in _CHAIN_TOPICS:
        assert topic in message
    persisted.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_evidence_still_persists_rather_than_refusing() -> None:
    """Scope control: absence of a HOP is honest evidence, absence of a CHAIN is not.

    A partial observation is written as-is -- `chain_replay` deliberately fills
    no gap, and the canary reports the shortfall as CHAIN_INCOMPLETE. Only an
    entirely empty write set is refused, because that is the case where a
    success verdict rests on nothing at all.
    """
    correlation_id = uuid4()
    observed = _complete_observation(correlation_id)
    handler = _handler()
    handler._read_observed = AsyncMock(return_value=observed[:1])  # type: ignore[method-assign]
    persisted = AsyncMock()
    handler._persist_rows = persisted  # type: ignore[method-assign]

    output = await handler.handle(_request(correlation_id))

    assert output.result is None
    assert len(persisted.await_args.args[0]) == 1


@pytest.mark.asyncio
async def test_upsert_that_changes_no_row_is_a_typed_refusal() -> None:
    """AC2: a statement that ran but wrote nothing is not a successful write."""
    correlation_id = uuid4()
    handler = _handler()
    row = ModelLedgerChainRow(
        correlation_id=correlation_id,
        hop_index=0,
        hop=_CHAIN_TOPICS[0],
        replay_green=True,
        verifier_verdict=EnumTierTwoVerdict.PASS,
        observed_topic=_CHAIN_TOPICS[0],
        envelope_id=uuid4(),
        parent_envelope_id=None,
        replay_detail="",
        verifier_detail="",
    )
    handler._db_handler.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id="fixture-db",
            result=ModelDbQueryResponse(
                status=EnumResponseStatus.SUCCESS,
                payload=ModelDbQueryPayload(rows=[], row_count=0),
                correlation_id=correlation_id,
            ),
        )
    )

    with pytest.raises(RuntimeHostError, match="affected no row"):
        await handler._persist_rows((row,))


@pytest.mark.asyncio
async def test_upsert_that_changes_a_row_is_accepted() -> None:
    """Positive control for the guard above: row_count 1 passes."""
    correlation_id = uuid4()
    handler = _handler()
    row = ModelLedgerChainRow(
        correlation_id=correlation_id,
        hop_index=0,
        hop=_CHAIN_TOPICS[0],
        replay_green=True,
        verifier_verdict=EnumTierTwoVerdict.PASS,
        observed_topic=_CHAIN_TOPICS[0],
        envelope_id=uuid4(),
        parent_envelope_id=None,
        replay_detail="",
        verifier_detail="",
    )
    handler._db_handler.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation_id,
            handler_id="fixture-db",
            result=ModelDbQueryResponse(
                status=EnumResponseStatus.SUCCESS,
                payload=ModelDbQueryPayload(rows=[], row_count=1),
                correlation_id=correlation_id,
            ),
        )
    )

    await handler._persist_rows((row,))


def test_ledger_projection_records_every_declared_chain_topic() -> None:
    """AC3: the evidence surface this writer reads must carry its own topology.

    `_read_observed` selects from `public.event_ledger`, which only
    `node_ledger_projection_compute` writes. A `chain_topology` topic absent
    from that node's subscription can never appear in the relation, so the
    writer's read is empty by construction and no ledger_chain row is reachable
    however the writer behaves.
    """
    chain_contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    projection_contract = yaml.safe_load(
        _LEDGER_PROJECTION_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    recorded = set(projection_contract["event_bus"]["subscribe_topics"])
    dispatched = {
        entry["topic"] for entry in projection_contract["handler_routing"]["handlers"]
    }

    # OMN-18419: `chain_topology` entries are declared hops, not bare topics.
    # OMN-18937: and a hop may answer to more than one topic. An ALTERNATIVE
    # is a topic this writer really does read out of event_ledger, so leaving
    # it out of this gate would let the delegation failure terminal be
    # declared here and unrecorded there -- the exact by-construction empty
    # read this test exists to refuse, one alias further along.
    declared_topics = [
        topic
        for hop in chain_contract["chain_topology"]
        for topic in (hop["topic"], *(hop.get("alternatives") or ()))
    ]

    missing = [t for t in declared_topics if t not in recorded]
    assert not missing, (
        f"node_ledger_projection_compute does not record {missing!r}, so "
        "public.event_ledger can never carry the evidence "
        "node_delegation_chain_ledger_effect reads (OMN-18398)"
    )
    undispatched = [t for t in declared_topics if t not in dispatched]
    assert not undispatched, (
        f"{undispatched!r} are subscribed but have no handler_routing entry, "
        "which is subscribed-but-never-dispatched (OMN-14594 pairing rule)"
    )


# ---------------------------------------------------------------------------
# OMN-18937: parsing and settling a topology whose terminal has alternatives
#
# `_parse_declared_topology` had no direct coverage at all before this block.
# Two of its four refusals had to widen over aliases; the other two had to
# stay counting entries. Both halves are asserted here, because a widening
# that also widened the head count would silently accept a two-chain
# declaration, and a narrowing left behind would let an alias collide.
# ---------------------------------------------------------------------------

_FAILED_TERMINAL = "onex.evt.omnimarket.delegate-skill-failed.v1"


def _topology_entry(
    topic: str, parent: str | None, alternatives: tuple[str, ...] = ()
) -> dict[str, object]:
    entry: dict[str, object] = {"topic": topic, "parent": parent}
    if alternatives:
        entry["alternatives"] = list(alternatives)
    return entry


def test_a_topology_with_an_alternative_terminal_parses_and_keeps_one_head() -> None:
    """The live shape: five entries, one head, the terminal answering to two."""
    hops = _parse_declared_topology(
        [
            _topology_entry("command", None),
            _topology_entry("terminal", "command", (_FAILED_TERMINAL,)),
        ]
    )

    assert len(hops) == 2, "an alternative is another NAME for a hop, not a hop"
    assert [hop.parent for hop in hops].count(None) == 1
    assert hops[1].topics == ("terminal", _FAILED_TERMINAL)


def test_an_alternative_colliding_with_another_hops_topic_is_refused() -> None:
    """The duplicate refusal must see aliases, or it resolves first-match-wins.

    Left counting canonical topics only, this declaration parses and
    `_declared_hop_for` silently grades the collided topic against whichever
    hop it reaches first -- the exact ambiguity the refusal exists to stop,
    spelled one alias further along.
    """
    with pytest.raises(RuntimeError, match="more than once"):
        _parse_declared_topology(
            [
                _topology_entry("command", None),
                _topology_entry(_FAILED_TERMINAL, "command"),
                _topology_entry("terminal", "command", (_FAILED_TERMINAL,)),
            ]
        )


def test_a_parent_may_cite_an_alternative() -> None:
    """The dangling-parent refusal must see aliases for the same reason."""
    hops = _parse_declared_topology(
        [
            _topology_entry("command", None),
            _topology_entry("terminal", "command", (_FAILED_TERMINAL,)),
            _topology_entry("postmortem", _FAILED_TERMINAL),
        ]
    )
    assert hops[2].parent == _FAILED_TERMINAL


def test_alternatives_do_not_create_a_second_head() -> None:
    """Head counting stays over ENTRIES: an alias is a name, not a hop."""
    with pytest.raises(RuntimeError, match="2 heads"):
        _parse_declared_topology(
            [
                _topology_entry("command", None),
                _topology_entry("orphan", None, (_FAILED_TERMINAL,)),
            ]
        )


def test_the_settle_loop_does_not_wait_for_a_terminal_that_cannot_arrive() -> None:
    """A hop with alternatives settles on ONE of them (OMN-18937).

    The settle predicate shares its declaration with the event_ledger read
    filter, which must carry EVERY alias or an alternative is never selected
    out of the relation. Requiring every alias to be OBSERVED instead would
    mean a delegation -- which terminates as completed or failed, never both
    -- never settles, and every single dispatch burns its full attempt budget
    before writing a chain that was already complete on the first read.
    """
    declared = (
        ModelDeclaredChainHop(topic="command", parent=None),
        ModelDeclaredChainHop(
            topic="terminal", parent="command", alternatives=(_FAILED_TERMINAL,)
        ),
    )
    handler = _handler(declared_chain=declared)

    # The read filter carries both names...
    assert set(handler._declared_topics()) == {"command", "terminal", _FAILED_TERMINAL}
    # ...and either terminal alone settles the observation.
    assert handler._observation_is_settled({"command", "terminal"})
    assert handler._observation_is_settled({"command", _FAILED_TERMINAL})
    # A hop observed on NO name is still unsettled -- widening the accepted
    # set is not the same as dropping the requirement.
    assert not handler._observation_is_settled({"command"})
