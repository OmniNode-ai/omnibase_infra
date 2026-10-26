# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Primary election for one ``primary`` node (OMN-21421).

``Campaign``: one atomic create-if-absent of the node's leader key, bound to the
session lease; the store revision of that grant is the epoch. ``StepDown``:
the runtime stops acting when its session deadline passes or when a read finds
the key held by someone else or gone.
"""

from __future__ import annotations

from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.handlers.placement.handler_placement_session import (
    HandlerPlacementSession,
)

PRIMARY_PREFIX = "/onex/placement/primary/"


class HandlerPlacementElection:
    """One runtime's candidacy for one node's primary role."""

    def __init__(
        self,
        *,
        adapter: HandlerPlacementEtcd,
        session: HandlerPlacementSession,
        node_name: str,
    ) -> None:
        self._adapter = adapter
        self._session = session
        self._node_name = node_name
        self._key = f"{PRIMARY_PREFIX}{node_name}"
        self._epoch = 0

    @property
    def node_name(self) -> str:
        return self._node_name

    @property
    def epoch(self) -> int:
        """The epoch this runtime won with, ``0`` when it is not primary."""
        return self._epoch

    def acting(self) -> bool:
        """The model's ``Acting``: primary and before the local self-fence deadline."""
        return self._epoch != 0 and self._session.within_deadline()

    async def campaign_once(self) -> bool:
        """Try to take the free leader key; True when this runtime is primary."""
        if self._epoch != 0:
            return self.acting()
        if not self._session.within_deadline():
            return False
        created, revision = await self._adapter.create_if_absent(
            self._key, self._session.runtime_id, lease_id=self._session.lease_id
        )
        if created and revision > 0:
            self._epoch = revision
        return self.acting()

    async def verify_holder(self) -> bool:
        """Step down when the leader key is not this runtime's at this epoch."""
        if self._epoch == 0:
            return False
        held = await self._adapter.get(self._key)
        if held is None or held != (self._session.runtime_id, self._epoch):
            self.step_down()
            return False
        return self.acting()

    def step_down(self) -> None:
        self._epoch = 0


__all__ = ["PRIMARY_PREFIX", "HandlerPlacementElection"]
