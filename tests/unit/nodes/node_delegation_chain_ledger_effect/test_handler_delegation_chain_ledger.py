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
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    HandlerDelegationChainLedger,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    EnumTierTwoVerdict,
    ModelDelegationTerminalPayload,
    ModelObservedHop,
)

_CHAIN = ("command", "route-request", "route-decision", "completed")
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_delegation_chain_ledger_effect/contract.yaml"
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


def _handler(declared_chain: tuple[str, ...] = _CHAIN) -> HandlerDelegationChainLedger:
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
    envelope_ids = tuple(uuid4() for _ in _CHAIN)
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
        for index, topic in enumerate(_CHAIN)
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
    assert {row.hop for row in rows} == set(_CHAIN)


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
    assert {row.hop for row in rows} == set(_CHAIN)
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
