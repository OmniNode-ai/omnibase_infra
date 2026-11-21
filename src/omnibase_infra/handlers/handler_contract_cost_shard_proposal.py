# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rank stored contract cost readings by loop time (OMN-26250).

The proposal is the output of the readings and nothing else: sum each
contract's readings, sort by loop time, state the share of the ten heaviest,
and list served contracts that have no reading. Re-running it over the same
stored readings gives the same table.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.health.model_contract_cost_shard_proposal import (
    ModelContractCostShardProposal,
    ModelContractCostShare,
)
from omnibase_infra.models.health.model_contract_cost_shard_request import (
    ModelContractCostShardRequest,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

TOP_CONTRACTS = 10


class HandlerContractCostShardProposal:
    """Sums stored readings per contract and ranks them by loop time."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler for the effect node."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral category: effect, as the handlers package declares."""
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelContractCostShardRequest
    ) -> ModelContractCostShardProposal:
        """Return the contracts of ``request.readings`` heaviest loop time first."""
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for reading in request.readings:
            row = sums[reading.contract_name]
            row[0] += reading.on_loop_wall_ms
            row[1] += reading.off_loop_wall_ms
            row[2] += reading.dispatch_count
            row[3] += reading.payload_bytes
        total = sum(row[0] for row in sums.values())

        def _share(loop_ms: float) -> float:
            return loop_ms / total if total > 0 else 0.0

        ranked = sorted(sums.items(), key=lambda item: (-item[1][0], item[0]))
        contracts = tuple(
            ModelContractCostShare(
                contract_name=name,
                on_loop_wall_ms=row[0],
                off_loop_wall_ms=row[1],
                dispatch_count=int(row[2]),
                payload_bytes=int(row[3]),
                share_of_loop_time=_share(row[0]),
            )
            for name, row in ranked
        )
        return ModelContractCostShardProposal(
            window_start=min(r.window_start for r in request.readings),
            window_end=max(r.window_end for r in request.readings),
            reading_count=len(request.readings),
            total_on_loop_wall_ms=total,
            top_10_share_of_loop_time=min(
                sum(c.share_of_loop_time for c in contracts[:TOP_CONTRACTS]), 1.0
            ),
            contracts=contracts,
            unread_contracts=tuple(sorted(set(request.served_contracts) - set(sums))),
        )


__all__: list[str] = ["HandlerContractCostShardProposal", "TOP_CONTRACTS"]
