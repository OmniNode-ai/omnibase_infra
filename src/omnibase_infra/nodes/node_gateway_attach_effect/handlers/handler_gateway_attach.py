# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for gateway.attach -- validate token, register session.

Canonical definition-B EFFECT handler: ``handle(request) -> response``. I/O
(the issuer secret-ref resolution and the session store write are local to
this process for the first slice) lives entirely inside this handler and the
services it calls, never in the node's dispatch wiring. Claim decode itself
has no network call; the expected-issuer *value* it compares against is
resolved here via ``SecretResolver``, mirroring how
``HandlerGatewayHeartbeat._introspect`` resolves the introspection endpoint
ref -- the same ref-resolution pattern, applied to ``keycloak_issuer_ref``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

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
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_request import (
    ModelGatewayAttachRequest,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_response import (
    ModelGatewayAttachResponse,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
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

__all__ = ["HandlerGatewayAttach"]


class HandlerGatewayAttach:
    """Validate a client-credentials token and register a tenant-bound session."""

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
        self, request: ModelGatewayAttachRequest
    ) -> ModelGatewayAttachResponse:
        issuer_secret = await self._secret_resolver.get_secret_async(
            self._config.keycloak_issuer_ref,
            required=True,
        )
        if issuer_secret is None:
            raise token_validator.TokenValidationError(
                "Keycloak issuer secret ref resolved to None despite required=True"
            )
        expected_issuer = issuer_secret.get_secret_value()

        claims = token_validator.decode_claims(
            request.access_token, self._config, expected_issuer=expected_issuer
        )

        now = datetime.now(UTC)
        token_ttl_seconds = claims.expires_at_epoch - int(now.timestamp())
        if token_ttl_seconds <= 0:
            raise token_validator.TokenValidationError("access_token has expired")
        session_ttl_seconds = min(
            token_ttl_seconds, self._config.max_session_ttl_seconds
        )
        expires_at = now + timedelta(seconds=session_ttl_seconds)

        session = ModelGatewaySession(
            session_id=uuid4(),
            tenant_id=claims.tenant_id,
            tenant_slug=claims.tenant_slug,
            principal_id=claims.principal_id,
            keycloak_client_id=claims.client_id,
            edge_instance_id=request.edge_instance_id,
            status=EnumGatewaySessionStatus.ACTIVE,
            attached_at=now,
            last_heartbeat_at=now,
            expires_at=expires_at,
        )
        await self._session_store.put(session)

        event = ModelGatewaySessionEvent(
            event_type=EnumGatewaySessionEventType.ATTACHED,
            session_id=session.session_id,
            tenant_id=session.tenant_id,
            tenant_slug=session.tenant_slug,
            principal_id=session.principal_id,
            edge_instance_id=session.edge_instance_id,
            emitted_at=now,
        )
        return ModelGatewayAttachResponse(
            session=session,
            heartbeat_interval_seconds=self._config.heartbeat_interval_seconds,
            session_event=event,
        )
