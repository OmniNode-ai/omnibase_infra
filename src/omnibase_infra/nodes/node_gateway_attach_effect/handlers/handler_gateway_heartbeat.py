# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for gateway.heartbeat -- re-validate via Keycloak introspection.

This is the revocation-detection path: disabling the tenant's Keycloak
client makes the introspection call in ``service_keycloak_token_validator``
return ``active: false`` on the next heartbeat, which flips the session to
REVOKED and deletes it from the store -- independent of the presented
token's own unexpired ``exp`` claim.
"""

from __future__ import annotations

from datetime import UTC, datetime

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_event_type import (
    EnumGatewaySessionEventType,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_heartbeat_request import (
    ModelGatewayHeartbeatRequest,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_heartbeat_response import (
    ModelGatewayHeartbeatResponse,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_keycloak_token_validator as token_validator,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.protocol_gateway_session_store import (
    ProtocolGatewaySessionStore,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

__all__ = ["HandlerGatewayHeartbeat"]


class SessionNotFoundError(Exception):
    """Raised when a heartbeat targets an unknown or already-torn-down session."""


class HandlerGatewayHeartbeat:
    """Re-validate a session's token via Keycloak introspection."""

    def __init__(
        self,
        config: ModelGatewayAttachConfig,
        session_store: ProtocolGatewaySessionStore,
        secret_resolver: SecretResolver,
    ) -> None:
        self._config = config
        self._session_store = session_store
        self._secret_resolver = secret_resolver

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelGatewayHeartbeatRequest
    ) -> ModelGatewayHeartbeatResponse:
        session = await self._session_store.get(request.session_id)
        if session is None:
            raise SessionNotFoundError(f"no session {request.session_id}")

        now = datetime.now(UTC)
        is_active = await token_validator.introspect(
            access_token=request.access_token,
            client_id=session.keycloak_client_id,
            config=self._config,
            secret_resolver=self._secret_resolver,
            correlation_id=session.session_id,
        )

        if not is_active:
            await self._session_store.delete(session.session_id)
            revoked_session = session.model_copy(
                update={"status": EnumGatewaySessionStatus.REVOKED}
            )
            event = ModelGatewaySessionEvent(
                event_type=EnumGatewaySessionEventType.REVOKED,
                session_id=session.session_id,
                tenant_id=session.tenant_id,
                tenant_slug=session.tenant_slug,
                principal_id=session.principal_id,
                edge_instance_id=session.edge_instance_id,
                emitted_at=now,
            )
            return ModelGatewayHeartbeatResponse(
                session=revoked_session, revoked=True, session_event=event
            )

        elapsed = (now - session.last_heartbeat_at).total_seconds()
        status = (
            EnumGatewaySessionStatus.DEGRADED
            if elapsed > self._config.session_degraded_after_seconds
            else EnumGatewaySessionStatus.ACTIVE
        )
        updated_session = session.model_copy(
            update={"status": status, "last_heartbeat_at": now}
        )
        await self._session_store.put(updated_session)

        event_type = (
            EnumGatewaySessionEventType.HEARTBEAT_DEGRADED
            if status is EnumGatewaySessionStatus.DEGRADED
            else EnumGatewaySessionEventType.HEARTBEAT_OK
        )
        event = ModelGatewaySessionEvent(
            event_type=event_type,
            session_id=updated_session.session_id,
            tenant_id=updated_session.tenant_id,
            tenant_slug=updated_session.tenant_slug,
            principal_id=updated_session.principal_id,
            edge_instance_id=updated_session.edge_instance_id,
            emitted_at=now,
        )
        return ModelGatewayHeartbeatResponse(
            session=updated_session, revoked=False, session_event=event
        )
