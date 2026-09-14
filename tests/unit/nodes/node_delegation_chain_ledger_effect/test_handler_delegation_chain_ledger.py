# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hermetic writer tests for OMN-16964 chain-canary link 5."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    HandlerDelegationChainLedger,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    EnumTierTwoVerdict,
    ModelDelegationTerminalPayload,
    ModelObservedHop,
)

_CHAIN = ("command", "route-request", "route-decision", "completed")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CHAIN_CONTRACT = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_delegation_chain_ledger_effect/contract.yaml"
)
_LEDGER_PROJECTION_CONTRACT = (
    _REPO_ROOT / "src/omnibase_infra/nodes/node_ledger_projection_compute/contract.yaml"
)


def test_declared_chain_is_audited_before_writer_reads_event_ledger() -> None:
    """Every writer hop must reach event_ledger through one typed route.

    The chain writer cannot repair a missing audit subscription: it selects
    only rows already persisted in ``event_ledger``. Keep its topology coupled
    to the generic audit projection so a newly declared hop cannot make every
    live replay honestly-but-permanently incomplete.
    """
    chain_raw = yaml.safe_load(_CHAIN_CONTRACT.read_text(encoding="utf-8")) or {}
    projection_raw = (
        yaml.safe_load(_LEDGER_PROJECTION_CONTRACT.read_text(encoding="utf-8")) or {}
    )

    declared_chain = tuple(chain_raw.get("chain_topology") or ())
    subscribed = set(projection_raw.get("event_bus", {}).get("subscribe_topics") or ())
    route_counts = Counter(
        entry.get("topic")
        for entry in projection_raw.get("handler_routing", {}).get("handlers") or ()
        if entry.get("topic")
    )

    assert declared_chain, "chain writer must declare a non-empty topology"
    assert set(declared_chain) <= subscribed, (
        "chain writer reads event_ledger, but these declared hops are not "
        f"audited into it: {sorted(set(declared_chain) - subscribed)}"
    )
    assert {topic: route_counts[topic] for topic in declared_chain} == dict.fromkeys(
        declared_chain, 1
    )


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
async def test_complete_chain_writes_green_rows() -> None:
    correlation_id = uuid4()
    handler = _handler()
    handler._read_observed = AsyncMock(  # type: ignore[method-assign]
        return_value=_complete_observation(correlation_id)
    )
    persisted = AsyncMock()
    handler._persist_rows = persisted  # type: ignore[method-assign]

    output = await handler.handle(_request(correlation_id))

    assert output.result is not None
    assert output.result.rows_written == 4
    assert output.result.chain_complete is True
    assert output.result.replay_green is True
    assert output.result.verifier_verdict is EnumTierTwoVerdict.PASS
    rows = persisted.await_args.args[0]
    assert all(row.replay_green for row in rows)


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

    assert output.result is not None
    assert output.result.chain_complete is True
    assert output.result.replay_green is False
    rows = persisted.await_args.args[0]
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

    assert output.result is not None
    assert output.result.verifier_verdict is EnumTierTwoVerdict.SKIP
    assert output.result.verifier_verdict is not EnumTierTwoVerdict.PASS
    rows = persisted.await_args.args[0]
    assert all(row.verifier_verdict is EnumTierTwoVerdict.SKIP for row in rows)


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

    assert output.result is not None
    assert output.result.rows_written == 3
    assert output.result.chain_complete is False
    rows = persisted.await_args.args[0]
    assert tuple(row.hop for row in rows) == (
        "command",
        "route-request",
        "completed",
    )
