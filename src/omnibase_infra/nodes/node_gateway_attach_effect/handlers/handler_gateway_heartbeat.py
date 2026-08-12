# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for gateway.heartbeat -- re-validate via Keycloak introspection.

This is the revocation-detection path: disabling the tenant's Keycloak
client makes ``_introspect`` (RFC 7662) return ``active: false`` on the next
heartbeat, which flips the session to REVOKED and deletes it from the store
-- independent of the presented token's own unexpired ``exp`` claim.

The introspection HTTP call is inline in this module (not in a freestanding
``services/`` helper): it is the only I/O this node performs, and the
imperative-contract-guard requires raw-transport calls to live under
``handlers/``, never in a freestanding module the guard cannot attribute to a
contract-declared handler.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import httpx

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
from omnibase_infra.nodes.node_gateway_attach_effect.services.protocol_gateway_session_store import (
    ProtocolGatewaySessionStore,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.service_keycloak_token_validator import (
    TokenValidationError,
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
        is_active = await self._introspect(
            access_token=request.access_token,
            client_id=session.keycloak_client_id,
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

    async def _introspect(
        self,
        *,
        access_token: str,
        client_id: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """RFC 7662 token introspection. Returns True iff Keycloak reports ``active``.

        Fail-closed: any transport error, non-200 response, or malformed body
        is treated as NOT active -- an unreachable or misbehaving Keycloak
        must never read as "still valid."
        """
        introspection_url_secret = await self._secret_resolver.get_secret_async(
            self._config.keycloak_introspection_ref,
            required=True,
            correlation_id=correlation_id,
        )
        admin_client_id_secret = await self._secret_resolver.get_secret_async(
            f"{self._config.keycloak_admin_client_ref}.client_id",
            required=True,
            correlation_id=correlation_id,
        )
        admin_client_secret_secret = await self._secret_resolver.get_secret_async(
            f"{self._config.keycloak_admin_client_ref}.client_secret",
            required=True,
            correlation_id=correlation_id,
        )
        # required=True guarantees non-None (SecretResolver raises otherwise);
        # the return type stays Optional to serve required=False callers
        # elsewhere.
        if (
            introspection_url_secret is None
            or admin_client_id_secret is None
            or admin_client_secret_secret is None
        ):
            raise TokenValidationError(
                "Keycloak introspection secret refs resolved to None despite required=True"
            )
        introspection_url = introspection_url_secret.get_secret_value()
        admin_client_id = admin_client_id_secret.get_secret_value()
        admin_client_secret = admin_client_secret_secret.get_secret_value()

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    introspection_url,
                    data={
                        "token": access_token,
                        "token_type_hint": "access_token",
                        "client_id": admin_client_id,
                        "client_secret": admin_client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        except httpx.HTTPError:
            return False

        if response.status_code != 200:
            return False
        try:
            body = response.json()
        except ValueError:
            return False
        active = body.get("active")
        if active is not True:
            return False
        # Defense in depth: introspection must confirm the same client_id the
        # session was attached with. A token re-issued for a *different*
        # tenant client must never validate a stale session's heartbeat.
        return str(body.get("client_id", "")) == client_id
