# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session store boundary -- DI seam so the store backend is swappable.

First slice ships ``StoreGatewaySessionMemory`` (single-process,
adequate for one control-plane pod attaching one tenant). A multi-pod
deployment needs a shared backend (Valkey, matching the rest of the ONEX
runtime session state) behind this same Protocol -- tracked as follow-on,
not built in this slice (see contract.yaml metadata).
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)


class ProtocolGatewaySessionStore(Protocol):
    """Async CRUD boundary for attach sessions."""

    async def put(self, session: ModelGatewaySession) -> None: ...

    async def get(self, session_id: UUID) -> ModelGatewaySession | None: ...

    async def delete(self, session_id: UUID) -> None: ...

    async def put_if_present(self, session: ModelGatewaySession) -> bool:
        """Atomically overwrite a session iff it is still present.

        OMN-15918 R2: closes the heartbeat resurrection race. A handler that
        reads a session, awaits network I/O (introspection), and then writes
        the refreshed session back must not blindly overwrite -- if a
        concurrent detach removed the session during that I/O gap, an
        unconditional ``put`` would silently resurrect it (an observable
        half-state: the caller detached, but the session reappears ACTIVE on
        the next read). ``put_if_present`` performs the presence check and
        the write as one atomic step and returns ``False`` (no-op) instead of
        resurrecting the row when the session is no longer present.
        """
        ...


__all__ = ["ProtocolGatewaySessionStore"]
