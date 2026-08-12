# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for gateway.detach -- explicit, edge-initiated teardown."""

from __future__ import annotations

from datetime import UTC, datetime

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_event_type import (
    EnumGatewaySessionEventType,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_detach_request import (
    ModelGatewayDetachRequest,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_detach_response import (
    ModelGatewayDetachResponse,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.protocol_gateway_session_store import (
    ProtocolGatewaySessionStore,
)

__all__ = ["HandlerGatewayDetach"]


class SessionNotFoundError(Exception):
    """Raised when detach targets an unknown or already-torn-down session."""


class HandlerGatewayDetach:
    """Tear down a session on explicit edge-initiated detach."""

    def __init__(self, session_store: ProtocolGatewaySessionStore) -> None:
        self._session_store = session_store

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelGatewayDetachRequest
    ) -> ModelGatewayDetachResponse:
        session = await self._session_store.get(request.session_id)
        if session is None:
            raise SessionNotFoundError(f"no session {request.session_id}")

        await self._session_store.delete(request.session_id)
        now = datetime.now(UTC)
        event = ModelGatewaySessionEvent(
            event_type=EnumGatewaySessionEventType.DETACHED,
            session_id=session.session_id,
            tenant_id=session.tenant_id,
            tenant_slug=session.tenant_slug,
            principal_id=session.principal_id,
            edge_instance_id=session.edge_instance_id,
            emitted_at=now,
        )
        return ModelGatewayDetachResponse(
            session_id=request.session_id,
            status=EnumGatewaySessionStatus.DETACHED,
            session_event=event,
        )
