# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime's membership session and its self-fence deadline (OMN-21421).

The session is one etcd lease. The runtime's membership key and every leader
key it wins are bound to it, so the store drops them together when the lease
expires. The local deadline follows the model's ``Renew``: it is set to
``sent + TTL - margin``, where ``sent`` is the monotonic time the successful
grant or keepalive request was sent, never the time its reply arrived. The
runtime acts only before that deadline, so it has stopped before the store can
expire the lease and grant the key to another runtime.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)

MEMBERS_PREFIX = "/onex/placement/members/"


class HandlerPlacementSession:
    """Membership session: lease, membership key and local self-fence deadline."""

    def __init__(
        self,
        *,
        adapter: HandlerPlacementEtcd,
        runtime_id: str,
        node_names: Sequence[str],
        ttl_s: int = 4,
        margin_s: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if margin_s <= 0 or margin_s >= ttl_s:
            raise ValueError("margin_s must be positive and below ttl_s")
        self._adapter = adapter
        self._runtime_id = runtime_id
        self._node_names = tuple(node_names)
        self._ttl_s = ttl_s
        self._margin_s = margin_s
        self._clock = clock
        self._lease_id = 0
        self._deadline = 0.0

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure session for placement membership."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Side-effecting etcd lease management."""
        return EnumHandlerTypeCategory.EFFECT

    @property
    def lease_id(self) -> int:
        return self._lease_id

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    def within_deadline(self) -> bool:
        """True while the session is open and its local deadline has not passed."""
        return self._lease_id != 0 and self._clock() < self._deadline

    def _extend(self, sent: float, granted_ttl_s: int) -> None:
        self._deadline = sent + granted_ttl_s - self._margin_s

    async def open(self) -> None:
        """Grant the lease and register membership; raises when etcd is unreachable."""
        sent = self._clock()
        lease_id, granted = await self._adapter.lease_grant(self._ttl_s)
        if granted <= self._margin_s:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="placement_session_open",
            )
            raise InfraConnectionError(
                f"etcd granted TTL {granted}s, not above the {self._margin_s}s margin",
                context=context,
            )
        self._lease_id = lease_id
        self._extend(sent, granted)
        await self._adapter.put(
            f"{MEMBERS_PREFIX}{self._runtime_id}",
            json.dumps(
                {"runtime_id": self._runtime_id, "nodes": list(self._node_names)}
            ),
            lease_id=lease_id,
        )

    async def keepalive(self) -> bool:
        """Renew once. False when the lease is gone; the session is then closed."""
        if self._lease_id == 0:
            return False
        if not self.within_deadline():
            # The model's Renew needs Acting: a lease past its local deadline is
            # never renewed. A new session is opened and the old lease expires.
            self.drop()
            return False
        sent = self._clock()
        remaining = await self._adapter.lease_keepalive(self._lease_id)
        if remaining <= 0:
            self._lease_id = 0
            self._deadline = 0.0
            return False
        self._extend(sent, remaining)
        return True

    def drop(self) -> None:
        """Forget the session locally (the store expires it on its own TTL)."""
        self._lease_id = 0
        self._deadline = 0.0

    async def close(self) -> None:
        """Revoke the lease so membership and leader keys go at once."""
        lease_id = self._lease_id
        self.drop()
        if lease_id:
            await self._adapter.lease_revoke(lease_id)


__all__ = ["MEMBERS_PREFIX", "HandlerPlacementSession"]
