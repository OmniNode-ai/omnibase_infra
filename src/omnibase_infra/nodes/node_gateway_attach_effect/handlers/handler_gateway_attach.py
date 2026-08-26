# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for gateway.attach -- validate token, register session.

Canonical definition-B EFFECT handler: ``handle(request) -> response``. I/O
(the issuer/JWKS secret-ref resolution, the JWKS fetch, and the session
store write) lives entirely inside this handler and the services it calls,
never in the node's dispatch wiring.

JWKS fetch (OMN-15918 R1): the token's signature is verified against
Keycloak's real signing keys before any claim is trusted. The fetch itself
is network I/O and stays inline here (imperative-contract-guard's
handlers/-only I/O boundary) wrapped in a circuit-breaker guard so a
Keycloak/JWKS outage fails a single attach attempt (``InfraUnavailableError``,
retry-able) instead of masquerading as a token-validation rejection.
Verification itself (CPU-only, no I/O) lives in
``service_keycloak_token_validator.verify_and_decode_claims``.

Introspection at the attach boundary (OMN-16032): local verification proves
a token was *minted* by this realm and has not yet expired. It cannot prove
the credential is still *live*, because disabling a Keycloak client -- this
platform's designated revocation mechanism -- does not invalidate tokens the
client already holds. Attach previously trusted the local proof alone, so a
client an operator had just disabled in response to a suspected compromise
could still open a tenant-bound session for the remainder of its token's
lifetime (up to ``max_attach_token_lifetime_seconds``, 900s), and was caught
only on the first heartbeat. Attach now makes the same RFC 7662 call the
heartbeat path makes, via the shared ``_keycloak_introspection`` seam, and
refuses before any session is registered.

The two Keycloak surfaces get two independent circuit breakers for the same
reason the heartbeat handler holds two: the public JWKS endpoint and the
admin-credentialed introspection endpoint fail independently, and merging
their failure counts would let one degraded endpoint trip the other. An
outage on either raises ``InfraUnavailableError`` and is never reported as a
rejected credential.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.nodes.node_gateway_attach_effect.handlers._keycloak_introspection import (
    GatewayCircuitBreakerGuard,
    introspect_token,
)
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
    service_gateway_renewal_policy as renewal_policy,
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
        self._jwks_circuit = GatewayCircuitBreakerGuard(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout_seconds,
            service_name="gateway-attach.keycloak-jwks",
        )
        self._introspection_circuit = GatewayCircuitBreakerGuard(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout_seconds,
            service_name="gateway-attach.keycloak-introspection",
        )

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

        jwks_keys = await self._fetch_jwks()
        claims = token_validator.verify_and_decode_claims(
            request.access_token,
            jwks_keys,
            self._config,
            expected_issuer=expected_issuer,
        )

        now = datetime.now(UTC)
        token_ttl_seconds = claims.expires_at_epoch - int(now.timestamp())
        if token_ttl_seconds <= 0:
            raise token_validator.TokenValidationError("access_token has expired")

        # OMN-16032: ask Keycloak whether this credential is still live,
        # before any session exists. Ordered after local verification (a
        # forged or malformed token is rejected without spending a network
        # round-trip on it) and before the store write (a refusal must leave
        # nothing behind). The client_id compared here is the token's own
        # verified ``azp``, so introspection has to agree with the claim the
        # session would otherwise be bound to.
        is_active = await introspect_token(
            config=self._config,
            secret_resolver=self._secret_resolver,
            circuit=self._introspection_circuit,
            access_token=request.access_token,
            client_id=claims.client_id,
        )
        if not is_active:
            raise token_validator.TokenValidationError(
                "access_token is not active per Keycloak introspection "
                "(client disabled or token revoked)"
            )
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
        # OMN-15952: hand the runtime its renewal cycle in the same response
        # that gives it the session. The alternative -- publishing the terms
        # in documentation and hoping each client implements them -- is what
        # left an unattended runtime with no defined behaviour at expiry in
        # the first place. Computed from the session just stamped, so the
        # directive can never disagree with the ceiling it is racing.
        renewal = renewal_policy.build_renewal_directive(session, config=self._config)

        return ModelGatewayAttachResponse(
            session=session,
            heartbeat_interval_seconds=self._config.heartbeat_interval_seconds,
            renewal=renewal,
            session_event=event,
        )

    async def _fetch_jwks(self) -> list[dict[str, object]]:
        """Fetch the JWKS keyset (RFC 7517). Circuit-breaker guarded.

        Fail-closed but distinguishable: a Keycloak/JWKS outage raises
        ``InfraUnavailableError`` (retry-able, never treated as a rejected
        token) rather than silently falling back to unsigned trust.
        """
        jwks_url_secret = await self._secret_resolver.get_secret_async(
            self._config.keycloak_jwks_ref,
            required=True,
        )
        if jwks_url_secret is None:
            raise token_validator.TokenValidationError(
                "Keycloak JWKS secret ref resolved to None despite required=True"
            )
        jwks_url = jwks_url_secret.get_secret_value()

        async with self._jwks_circuit._circuit_breaker_lock:
            await self._jwks_circuit._check_circuit_breaker(operation="fetch_jwks")

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(jwks_url)
        except httpx.HTTPError as exc:
            async with self._jwks_circuit._circuit_breaker_lock:
                await self._jwks_circuit._record_circuit_failure(operation="fetch_jwks")
            raise InfraUnavailableError(
                "Keycloak JWKS endpoint unreachable",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="fetch_jwks",
                ),
            ) from exc

        if response.status_code != 200:
            async with self._jwks_circuit._circuit_breaker_lock:
                await self._jwks_circuit._record_circuit_failure(operation="fetch_jwks")
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
            async with self._jwks_circuit._circuit_breaker_lock:
                await self._jwks_circuit._record_circuit_failure(operation="fetch_jwks")
            raise InfraUnavailableError(
                "Keycloak JWKS response was not valid JSON",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="fetch_jwks",
                ),
            ) from exc

        async with self._jwks_circuit._circuit_breaker_lock:
            await self._jwks_circuit._reset_circuit_breaker()

        keys = body.get("keys") if isinstance(body, dict) else None
        if not isinstance(keys, list) or not keys:
            raise token_validator.TokenValidationError(
                "Keycloak JWKS response contained no keys"
            )
        return keys
