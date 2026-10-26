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
from typing import Protocol, runtime_checkable

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.handlers.placement.handler_placement_election import (
    HandlerPlacementElection,
)
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.handlers.placement.handler_placement_serving import (
    read_serving_report,
)
from omnibase_infra.handlers.placement.handler_placement_session import (
    HandlerPlacementSession,
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
    """Places one runtime's loaded nodes according to their placement policy."""

    def __init__(
        self,
        *,
        runtime_id: str,
        adapter: HandlerPlacementEtcd,
        nodes: Mapping[str, tuple[ModelPlacementPolicy, ProtocolPlacedNode]],
        ttl_s: int = 4,
        margin_s: float = 1.5,
        keepalive_interval_s: float = 1.0,
        campaign_interval_s: float = 0.25,
        fence_tick_s: float = 0.1,
    ) -> None:
        self._runtime_id = runtime_id
        self._adapter = adapter
        self._nodes = dict(nodes)
        self._keepalive_interval_s = keepalive_interval_s
        self._campaign_interval_s = campaign_interval_s
        self._fence_tick_s = fence_tick_s
        self._session = HandlerPlacementSession(
            adapter=adapter,
            runtime_id=runtime_id,
            node_names=sorted(self._nodes),
            ttl_s=ttl_s,
            margin_s=margin_s,
        )
        self._elections = {
            name: HandlerPlacementElection(
                adapter=adapter, session=self._session, node_name=name
            )
            for name, (policy, _) in self._nodes.items()
            if policy.mode == "primary"
        }
        self._started: dict[str, int] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._lock = asyncio.Lock()

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
        self, *, min_serving: int = 2
    ) -> ModelPlacementServingReport:
        return await read_serving_report(
            self._adapter, expected_nodes=sorted(self._nodes), min_serving=min_serving
        )

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

    async def _acquire(self, name: str, election: HandlerPlacementElection) -> None:
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
