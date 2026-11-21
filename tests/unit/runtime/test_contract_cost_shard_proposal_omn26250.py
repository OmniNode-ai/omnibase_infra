# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract cost shard proposal (OMN-26250).

Validates:
- Contracts are ranked by summed loop time, heaviest first, and each carries its
  share of the total loop time
- The top-10 share is the summed share of the ten heaviest
- The proposal names the window the readings cover
- A served contract with no reading is listed as unread, not dropped
- Re-running over the same readings gives the same proposal
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.handler_contract_cost_shard_proposal import (
    HandlerContractCostShardProposal,
)
from omnibase_infra.models.health.model_contract_cost_reading import (
    ModelContractCostReading,
)
from omnibase_infra.models.health.model_contract_cost_shard_proposal import (
    ModelContractCostShardProposal,
)
from omnibase_infra.models.health.model_contract_cost_shard_request import (
    ModelContractCostShardRequest,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 11, 21, 0, 0, 0, tzinfo=UTC)


def _reading(name: str, loop_ms: float, window: int) -> ModelContractCostReading:
    start = _T0 + timedelta(minutes=window)
    return ModelContractCostReading(
        contract_name=name,
        window_start=start,
        window_end=start + timedelta(minutes=1),
        dispatch_count=2,
        on_loop_wall_ms=loop_ms,
        off_loop_wall_ms=1.0,
        payload_bytes=10,
    )


def _handler() -> HandlerContractCostShardProposal:
    return HandlerContractCostShardProposal(container=MagicMock())


def test_contracts_are_ranked_by_summed_loop_time_with_shares() -> None:
    readings = (
        _reading("light", 10.0, 0),
        _reading("heavy", 300.0, 0),
        _reading("heavy", 100.0, 1),
        _reading("middle", 90.0, 1),
    )
    proposal = _handler().handle(
        ModelContractCostShardRequest(
            readings=readings, served_contracts=("heavy", "light", "middle")
        )
    )
    assert [c.contract_name for c in proposal.contracts] == ["heavy", "middle", "light"]
    assert proposal.contracts[0].on_loop_wall_ms == pytest.approx(400.0)
    assert proposal.contracts[0].dispatch_count == 4
    assert proposal.contracts[0].share_of_loop_time == pytest.approx(0.8)
    assert proposal.total_on_loop_wall_ms == pytest.approx(500.0)
    assert proposal.reading_count == 4
    assert proposal.window_start == _T0
    assert proposal.window_end == _T0 + timedelta(minutes=2)
    assert proposal.unread_contracts == ()


def test_top_10_share_is_the_share_of_the_ten_heaviest() -> None:
    # Twelve contracts: 1 ms .. 12 ms of loop time. The two lightest are 3 of 78.
    readings = tuple(_reading(f"c{n:02d}", float(n), 0) for n in range(1, 13))
    proposal = _handler().handle(
        ModelContractCostShardRequest(
            readings=readings, served_contracts=tuple(r.contract_name for r in readings)
        )
    )
    assert proposal.top_10_share_of_loop_time == pytest.approx(75.0 / 78.0)
    assert len(proposal.contracts) == 12


def test_a_served_contract_with_no_reading_is_listed_unread() -> None:
    proposal = _handler().handle(
        ModelContractCostShardRequest(
            readings=(_reading("seen", 5.0, 0),),
            served_contracts=("seen", "z_unread", "a_unread"),
        )
    )
    assert proposal.unread_contracts == ("a_unread", "z_unread")


def test_rerunning_over_the_same_readings_gives_the_same_proposal() -> None:
    request = ModelContractCostShardRequest(
        readings=(_reading("a", 5.0, 0), _reading("b", 5.0, 0)),
        served_contracts=("a", "b"),
    )
    assert _handler().handle(request) == _handler().handle(request)
    # Equal loop time ties break on name, so the order is stable.
    assert [c.contract_name for c in _handler().handle(request).contracts] == ["a", "b"]


def test_all_zero_loop_time_reads_as_zero_shares() -> None:
    proposal = _handler().handle(
        ModelContractCostShardRequest(
            readings=(_reading("idle", 0.0, 0),), served_contracts=("idle",)
        )
    )
    assert proposal.top_10_share_of_loop_time == 0.0


def test_request_needs_at_least_one_reading() -> None:
    with pytest.raises(ValidationError):
        ModelContractCostShardRequest(readings=(), served_contracts=())


def test_proposal_refuses_an_unsorted_table() -> None:
    proposal = _handler().handle(
        ModelContractCostShardRequest(
            readings=(_reading("a", 1.0, 0), _reading("b", 9.0, 0)),
            served_contracts=("a", "b"),
        )
    )
    data = proposal.model_dump()
    data["contracts"] = tuple(reversed(data["contracts"]))
    with pytest.raises(ValidationError):
        ModelContractCostShardProposal(**data)
