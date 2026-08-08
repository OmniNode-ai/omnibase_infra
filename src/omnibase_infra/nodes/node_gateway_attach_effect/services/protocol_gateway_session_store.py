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


__all__ = ["ProtocolGatewaySessionStore"]
