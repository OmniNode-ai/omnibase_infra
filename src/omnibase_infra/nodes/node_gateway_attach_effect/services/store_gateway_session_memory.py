# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-process session store -- default backend for the first slice."""

from __future__ import annotations

import asyncio
from uuid import UUID

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)


class StoreGatewaySessionMemory:
    """Async-safe, single-process session store keyed by session_id."""

    def __init__(self) -> None:
        self._sessions: dict[UUID, ModelGatewaySession] = {}
        self._lock = asyncio.Lock()

    async def put(self, session: ModelGatewaySession) -> None:
        async with self._lock:
            self._sessions[session.session_id] = session

    async def get(self, session_id: UUID) -> ModelGatewaySession | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: UUID) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)


__all__ = ["StoreGatewaySessionMemory"]
