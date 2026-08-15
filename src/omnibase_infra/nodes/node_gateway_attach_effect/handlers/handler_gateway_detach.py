# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for gateway.detach -- explicit, edge-initiated teardown.

OMN-15918 R1 + R2: detach previously took only a session identifier and a
free-text reason -- zero credential, zero identity check, so any caller
holding a session identifier could detach any tenant's session. This
handler now requires the same signature-verified access token as attach and
heartbeat, and binds it to the STORED session's tenant/principal/client
identity before deleting. The JWKS fetch is the only I/O this handler
performs and stays inline here (imperative-contract-guard's handlers/-only
I/O boundary), mirroring ``HandlerGatewayAttach._fetch_jwks`` /
``HandlerGatewayHeartbeat._fetch_jwks``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_event_type import (
    EnumGatewaySessionEventType,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_status import (
    EnumGatewaySessionStatus,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
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
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_keycloak_token_validator as token_validator,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.protocol_gateway_session_store import (
    ProtocolGatewaySessionStore,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

__all__ = ["HandlerGatewayDetach"]


class SessionNotFoundError(Exception):
    """Raised when detach targets an unknown or already-torn-down session."""


class HandlerGatewayDetach(MixinAsyncCircuitBreaker):
    """Tear down a session on explicit edge-initiated detach."""

    def __init__(
        self,
        config: ModelGatewayAttachConfig,
        session_store: ProtocolGatewaySessionStore,
        secret_resolver: SecretResolver,
    ) -> None:
        self._config = config
        self._session_store = session_store
        self._secret_resolver = secret_resolver
        self._init_circuit_breaker(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout_seconds,
            service_name="gateway-detach.keycloak-jwks",
            transport_type=EnumInfraTransportType.HTTP,
        )

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

        issuer_secret = await self._secret_resolver.get_secret_async(
            self._config.keycloak_issuer_ref, required=True
        )
        if issuer_secret is None:
            raise token_validator.TokenValidationError(
                "Keycloak issuer secret ref resolved to None despite required=True"
            )
        jwks_keys = await self._fetch_jwks()
        claims = token_validator.verify_and_decode_claims(
            request.access_token,
            jwks_keys,
            self._config,
            expected_issuer=issuer_secret.get_secret_value(),
        )
        if (
            claims.tenant_id != session.tenant_id
            or claims.principal_id != session.principal_id
            or claims.client_id != session.keycloak_client_id
        ):
            raise token_validator.TokenValidationError(
                "access_token identity does not match the stored session "
                "(tenant/principal/client binding mismatch)"
            )

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

    async def _fetch_jwks(self) -> list[dict[str, object]]:
        """Fetch the JWKS keyset (RFC 7517). Circuit-breaker guarded.

        Mirrors ``HandlerGatewayAttach._fetch_jwks`` -- see that docstring
        for the fail-closed-but-distinguishable rationale.
        """
        jwks_url_secret = await self._secret_resolver.get_secret_async(
            self._config.keycloak_jwks_ref, required=True
        )
        if jwks_url_secret is None:
            raise token_validator.TokenValidationError(
                "Keycloak JWKS secret ref resolved to None despite required=True"
            )
        jwks_url = jwks_url_secret.get_secret_value()

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation="fetch_jwks")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(jwks_url)
        except httpx.HTTPError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation="fetch_jwks")
            raise InfraUnavailableError(
                "Keycloak JWKS endpoint unreachable",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="fetch_jwks",
                ),
            ) from exc

        if response.status_code != 200:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation="fetch_jwks")
            raise InfraUnavailableError(
                f"Keycloak JWKS endpoint returned HTTP {response.status_code}",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="fetch_jwks",
                ),
            )

        try:
            body = response.json()
        except ValueError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation="fetch_jwks")
            raise InfraUnavailableError(
                "Keycloak JWKS response was not valid JSON",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="fetch_jwks",
                ),
            ) from exc

        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

        keys = body.get("keys") if isinstance(body, dict) else None
        if not isinstance(keys, list) or not keys:
            raise token_validator.TokenValidationError(
                "Keycloak JWKS response contained no keys"
            )
        return keys
