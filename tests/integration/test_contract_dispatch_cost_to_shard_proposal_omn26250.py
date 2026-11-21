# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration: measured dispatch cost flows into the shard proposal (OMN-26250).

Wraps real async dispatchers with the per-contract counter, dispatches through
them on a running event loop, closes a window and feeds the readings to the
shard proposal handler. Nothing is faked on the path from the wrapped
dispatcher to the ranked proposal.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from omnibase_infra.handlers.contract_dispatch_cost import (
    ContractDispatchCostCounter,
    measure_contract_dispatch,
)
from omnibase_infra.handlers.handler_contract_cost_shard_proposal import (
    HandlerContractCostShardProposal,
)
from omnibase_infra.models.health.model_contract_cost_shard_request import (
    ModelContractCostShardRequest,
)

pytestmark = pytest.mark.asyncio


async def test_measured_dispatch_cost_ranks_contracts_in_the_proposal() -> None:
    async def slow(envelope: object) -> str:
        await asyncio.sleep(0.05)
        return "slow"

    async def fast(envelope: object) -> str:
        return "fast"

    counter = ContractDispatchCostCounter()
    counter.register("idle_contract")
    start = datetime.now(UTC)
    for name, fn in (("slow_contract", slow), ("fast_contract", fast)):
        wrapped = measure_contract_dispatch(name, fn, counter)
        await wrapped(object())
    readings = counter.close_window(start, datetime.now(UTC))
    assert [r.contract_label for r in readings] == [
        "fast_contract",
        "idle_contract",
        "slow_contract",
    ]

    proposal = HandlerContractCostShardProposal(container=MagicMock()).handle(
        ModelContractCostShardRequest(
            readings=readings,
            served_contracts=("fast_contract", "idle_contract", "slow_contract"),
        )
    )
    assert proposal.contracts[0].contract_label == "slow_contract"
    assert proposal.contracts[0].on_loop_wall_ms >= 40.0
    assert proposal.unread_contracts == ()
