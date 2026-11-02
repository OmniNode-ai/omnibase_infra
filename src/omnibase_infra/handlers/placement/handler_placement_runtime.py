# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime placement: membership, election per node, self-fence, rebuild then consume (OMN-21421).

``any`` nodes start at once on every runtime. A ``primary`` node starts only on
the runtime that wins its election, and only after that runtime has rebuilt
the node's projection. A runtime that cannot reach etcd never starts a
``primary`` node. Three independent tasks run while the runtime is up:

* keepalive: renews the session every ``keepalive_interval_s`` and re-reads each
  held leader key (``StepDown`` when it is not ours);
* election: campaigns for every ``primary`` node this runtime does not hold;
* fence: every ``fence_tick_s``, stops any node whose election is no longer
  acting. It reads only the local clock, so a hung etcd call cannot delay it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    InfraConnectionError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.handlers.placement.model_placement_command import (
    ModelPlacementCommand,
)
from omnibase_infra.handlers.placement.model_placement_policy import (
    ModelPlacementPolicy,
)
from omnibase_infra.handlers.placement.model_placement_serving_report import (
    ModelPlacementServingReport,
)
from omnibase_infra.models.model_placement_serving_request import (
    ModelPlacementServingRequest,
)
from omnibase_infra.protocols.protocol_placement_election import (
    ProtocolPlacementElection,
)
from omnibase_infra.protocols.protocol_placement_serving_reader import (
    ProtocolPlacementServingReader,
)
from omnibase_infra.protocols.protocol_placement_session import (
    ProtocolPlacementSession,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ProtocolPlacedNode(Protocol):
    """The runtime's hooks into one loaded node's consumer."""

    async def rebuild(self, placement_epoch: int) -> None:
        """Fold the projection from the stored table and offsets."""
        ...

    async def start_consuming(self, placement_epoch: int) -> None:
        """Join the consumer group; ``placement_epoch`` is 0 for ``any`` nodes."""
        ...

    async def stop_consuming(self) -> None:
        """Leave the consumer group and stop publishing."""
        ...


class HandlerPlacementRuntime:
    """Places one runtime's loaded nodes according to their placement policy.

    Every collaborator arrives through the constructor: the registry in
    ``node_placement_runtime_effect`` builds the session, the elections and the
    serving reader, and the contract profile supplies every timing.
    """

    def __init__(
        self,
        *,
        runtime_id: str,
        session: ProtocolPlacementSession,
        elections: Mapping[str, ProtocolPlacementElection],
        nodes: Mapping[str, tuple[ModelPlacementPolicy, ProtocolPlacedNode]],
        serving_reader: ProtocolPlacementServingReader,
        lock: AbstractAsyncContextManager[None],
        min_serving: int,
        keepalive_interval_s: float,
        campaign_interval_s: float,
        fence_tick_s: float,
    ) -> None:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="construct_placement_runtime",
        )
        primary = {n for n, (policy, _) in nodes.items() if policy.mode == "primary"}
        if set(elections) != primary:
            raise ProtocolConfigurationError(
                "elections must cover exactly the primary-mode nodes: "
                f"primary={sorted(primary)} elections={sorted(elections)}",
                context=context,
            )
        if session.runtime_id != runtime_id or set(session.node_names) != set(nodes):
            raise ProtocolConfigurationError(
                "session and handler disagree on runtime_id or node set",
                context=context,
            )
        self._runtime_id = runtime_id
        self._session = session
        self._elections = dict(elections)
        self._nodes = dict(nodes)
        self._serving_reader = serving_reader
        self._min_serving = min_serving
        self._keepalive_interval_s = keepalive_interval_s
        self._campaign_interval_s = campaign_interval_s
        self._fence_tick_s = fence_tick_s
        self._started: dict[str, int] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._lock = lock

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure runtime for node placement."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Side-effecting node consumer lifecycle management."""
        return EnumHandlerTypeCategory.EFFECT

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    def started_nodes(self) -> dict[str, int]:
        """Nodes this runtime is consuming for, with the epoch each started at."""
        return dict(self._started)

    def may_publish(self, node_name: str) -> bool:
        """True only for an ``any`` node, or a ``primary`` node whose election is acting."""
        election = self._elections.get(node_name)
        if election is None:
            return node_name in self._started
        return node_name in self._started and election.acting()

    def stamp(self, node_name: str, body: str) -> ModelPlacementCommand:
        """Stamp a command with the epoch this runtime holds; refuses when fenced."""
        election = self._elections.get(node_name)
        if election is None or not election.acting():
            raise PermissionError(f"{node_name}: not the acting primary")
        return ModelPlacementCommand(
            node_key=node_name,
            placement_epoch=election.epoch,
            runtime_key=self._runtime_id,
            body=body,
        )

    async def serving_report(
        self, *, min_serving: int | None = None
    ) -> ModelPlacementServingReport:
        effective = self._min_serving if min_serving is None else min_serving
        if effective <= 0:
            raise ProtocolConfigurationError(
                "min_serving must be positive",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="serving_report",
                ),
            )
        return await self._serving_reader(
            expected_nodes=sorted(self._nodes), min_serving=effective
        )

    async def handle(
        self, request: ModelPlacementServingRequest
    ) -> ModelPlacementServingReport:
        """Answer one placement decision: the per-node serving report."""
        return await self.serving_report(min_serving=request.min_serving)

    async def start(self) -> None:
        for name, (policy, node) in self._nodes.items():
            if policy.mode == "any":
                await node.start_consuming(0)
                self._started[name] = 0
        await self._try_open_session()
        self._tasks = [
            asyncio.create_task(self._keepalive_loop()),
            asyncio.create_task(self._election_loop()),
            asyncio.create_task(self._fence_loop()),
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        for name in list(self._started):
            await self._stop_node(name)
        for election in self._elections.values():
            election.step_down()
        with contextlib.suppress(InfraConnectionError):
            await self._session.close()

    async def _try_open_session(self) -> bool:
        if self._session.lease_id:
            return True
        try:
            await self._session.open()
        except InfraConnectionError as exc:
            self._session.drop()
            logger.warning("placement: etcd unreachable, primary nodes held: %s", exc)
            return False
        return True

    async def _stop_node(self, name: str) -> None:
        async with self._lock:
            if self._started.pop(name, None) is None:
                return
            await self._nodes[name][1].stop_consuming()

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(self._keepalive_interval_s)
            try:
                if not self._session.lease_id:
                    await self._try_open_session()
                    continue
                if not await self._session.keepalive():
                    for election in self._elections.values():
                        election.step_down()
                    continue
                for election in self._elections.values():
                    await election.verify_holder()
            except InfraConnectionError as exc:
                logger.warning("placement: keepalive failed: %s", exc)

    async def _election_loop(self) -> None:
        while True:
            for name, election in self._elections.items():
                if name in self._started:
                    continue
                try:
                    won = await election.campaign_once()
                except InfraConnectionError:
                    continue
                if won:
                    await self._acquire(name, election)
            await asyncio.sleep(self._campaign_interval_s)

    async def _acquire(self, name: str, election: ProtocolPlacementElection) -> None:
        node = self._nodes[name][1]
        epoch = election.epoch
        await node.rebuild(epoch)
        async with self._lock:
            if not election.acting() or election.epoch != epoch:
                return
            await node.start_consuming(epoch)
            self._started[name] = epoch

    async def _fence_loop(self) -> None:
        while True:
            await asyncio.sleep(self._fence_tick_s)
            for name, election in self._elections.items():
                if name in self._started and not election.acting():
                    election.step_down()
                    await self._stop_node(name)


__all__ = ["ProtocolPlacedNode", "HandlerPlacementRuntime"]
