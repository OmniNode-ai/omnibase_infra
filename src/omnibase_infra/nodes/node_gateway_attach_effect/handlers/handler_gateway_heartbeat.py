# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for gateway.heartbeat -- re-validate via Keycloak introspection.

This is the revocation-detection path: disabling the tenant's Keycloak
client makes ``_introspect`` (RFC 7662) return ``active: false`` on the next
heartbeat, which flips the session to REVOKED and deletes it from the store
-- independent of the presented token's own unexpired ``exp`` claim.

The introspection HTTP call (and the JWKS fetch, OMN-15918 R1) are inline in
this module (not in a freestanding ``services/`` helper): they are the only
I/O this node performs, and the imperative-contract-guard requires
raw-transport calls to live under ``handlers/``, never in a freestanding
module the guard cannot attribute to a contract-declared handler.

OMN-15918 hardening in this handler:
  - R1: the presented heartbeat token's signature is verified against the
    JWKS keyset (same as attach) before its claims are trusted for identity
    binding below.
  - R2: bind heartbeat identity to the STORED session (tenant_id,
    principal_id, keycloak_client_id) from attach time, not the caller's
    claims alone -- a token that decodes clean but names a *different*
    tenant/principal/client than the session it is heartbeating for is
    rejected before introspection ever runs.
  - R3 (atomicity): the refreshed session is written with
    ``put_if_present`` rather than an unconditional ``put`` -- a concurrent
    detach that removed the session during the introspection await window
    must not be resurrected by this handler's write.
  - R4: introspection failures split into two classes. A genuine Keycloak
    ``active: false`` (or client_id mismatch) is real revocation and tears
    the session down as before. A transport error, non-200, malformed body,
    or open circuit breaker is an *outage* -- ``InfraUnavailableError`` is
    raised and the session is left untouched (retry-able), never treated as
    revoked.

OMN-16022 adds the two time bounds that R4 left unbounded, without
weakening R4 itself:

  - ``expires_at``, written at attach and previously read by nothing, is
    now enforced here. A session past it is torn down and the runtime
    re-attaches.
  - Degraded mode is bounded by ``max_unverified_session_seconds``. R4's
    fail-open is still the default and a Keycloak blip still revokes
    nothing -- but a session that has gone a full attach-token lifetime
    with no successful revalidation is quarantined rather than trusted
    indefinitely, because otherwise anyone able to partition the gateway
    from Keycloak could hold revocation open for as long as they liked,
    including against a credential the operator had just rotated. Only the
    CEILING terminates; the outage never does.

Both bounds are evaluated before any network I/O, so enforcement does not
depend on the reachability of the service whose unreachability is the
threat. Entry into degraded mode is alarmed when it starts (once, not per
tick), not only when the ceiling finally fires.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

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
from omnibase_infra.nodes.node_gateway_attach_effect.models.enum_gateway_session_termination_reason import (
    EnumGatewaySessionTerminationReason,
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
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session import (
    ModelGatewaySession,
)
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_session_event import (
    ModelGatewaySessionEvent,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_gateway_session_policy as session_policy,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_keycloak_token_validator as token_validator,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services.protocol_gateway_session_store import (
    ProtocolGatewaySessionStore,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

__all__ = ["HandlerGatewayHeartbeat"]

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when a heartbeat targets an unknown or already-torn-down session."""


class GatewayHeartbeatCircuitBreakerGuard(MixinAsyncCircuitBreaker):
    """Composition wrapper: lets one handler hold >1 independent circuit breaker.

    ``MixinAsyncCircuitBreaker`` keeps its state on ``self`` (one breaker per
    instance). This handler talks to two independently-failing Keycloak
    surfaces (the public JWKS endpoint and the admin-credentialed
    introspection endpoint), so each gets its own guard instance rather than
    sharing one breaker's failure count across unrelated calls.
    """

    def __init__(
        self, *, threshold: int, reset_timeout: float, service_name: str
    ) -> None:
        self._init_circuit_breaker(
            threshold=threshold,
            reset_timeout=reset_timeout,
            service_name=service_name,
            transport_type=EnumInfraTransportType.HTTP,
        )


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
        self._jwks_circuit = GatewayHeartbeatCircuitBreakerGuard(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout_seconds,
            service_name="gateway-heartbeat.keycloak-jwks",
        )
        self._introspection_circuit = GatewayHeartbeatCircuitBreakerGuard(
            threshold=config.circuit_breaker_threshold,
            reset_timeout=config.circuit_breaker_reset_timeout_seconds,
            service_name="gateway-heartbeat.keycloak-introspection",
        )

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

        # OMN-16022: both lifetime bounds are evaluated here, before token
        # verification and before any network call. A session that has
        # breached one is over no matter what the presented token says or
        # what Keycloak would say about it, and checking after the JWKS
        # fetch would make enforcement depend on the reachability of the
        # very service whose unreachability the ceiling exists to bound.
        if session_policy.is_expired(session, now=now):
            logger.warning(
                "gateway session rejected: past expires_at",
                extra={
                    "alarm": "gateway.session.expired",
                    "session_id": str(session.session_id),
                    "tenant_id": str(session.tenant_id),
                    "tenant_slug": session.tenant_slug,
                    "edge_instance_id": session.edge_instance_id,
                    "expires_at": session.expires_at.isoformat(),
                },
            )
            return await self._terminate(
                session,
                now=now,
                reason=EnumGatewaySessionTerminationReason.EXPIRED,
                status=EnumGatewaySessionStatus.EXPIRED,
                event_type=EnumGatewaySessionEventType.EXPIRED,
            )

        if session_policy.exceeds_unverified_ceiling(
            session, now=now, config=self._config
        ):
            logger.error(
                "gateway session quarantined: degraded-mode ceiling breached",
                extra={
                    "alarm": "gateway.session.unverified_ceiling_breached",
                    "session_id": str(session.session_id),
                    "tenant_id": str(session.tenant_id),
                    "tenant_slug": session.tenant_slug,
                    "edge_instance_id": session.edge_instance_id,
                    "unverified_seconds": session_policy.unverified_seconds(
                        session, now=now
                    ),
                    "ceiling_seconds": self._config.max_unverified_session_seconds,
                },
            )
            return await self._terminate(
                session,
                now=now,
                reason=EnumGatewaySessionTerminationReason.UNVERIFIED_CEILING,
                status=EnumGatewaySessionStatus.QUARANTINED,
                event_type=EnumGatewaySessionEventType.QUARANTINED,
            )

        # R1 + R2: verify the presented token's signature, then bind it to
        # the STORED session identity -- never trust caller-supplied claims
        # in isolation. A token that verifies clean but names a different
        # tenant/principal/client than the session it targets is rejected
        # before introspection (or any store mutation) runs.
        issuer_secret = await self._secret_resolver.get_secret_async(
            self._config.keycloak_issuer_ref, required=True
        )
        if issuer_secret is None:
            raise token_validator.TokenValidationError(
                "Keycloak issuer secret ref resolved to None despite required=True"
            )
        try:
            jwks_keys = await self._fetch_jwks()
        except InfraUnavailableError:
            # Same outage class as an unreachable introspection endpoint --
            # this heartbeat could not re-verify the credential, so the
            # revocation-blind window opens here too. TokenValidationError
            # is deliberately not caught: a token that fails verification
            # against a JWKS we DID fetch is a rejection, not an outage.
            await self._enter_degraded(session, reason="jwks_unavailable")
            raise
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

        try:
            is_active = await self._introspect(
                access_token=request.access_token,
                client_id=session.keycloak_client_id,
                correlation_id=session.session_id,
            )
        except InfraUnavailableError:
            # OMN-15918 R4 is unchanged: an unreachable Keycloak is not
            # revocation, the session is left alive, and the caller retries.
            # OMN-16022 adds only that entering this revocation-blind window
            # is recorded on the session and alarmed once, so the operator
            # sees it start rather than inferring it from a later teardown.
            # The ceiling checked at the top of this method is what
            # eventually ends it.
            await self._enter_degraded(session, reason="introspection_unavailable")
            raise

        if not is_active:
            return await self._terminate(
                session,
                now=now,
                reason=EnumGatewaySessionTerminationReason.REVOKED,
                status=EnumGatewaySessionStatus.REVOKED,
                event_type=EnumGatewaySessionEventType.REVOKED,
            )

        elapsed = session_policy.unverified_seconds(session, now=now)
        status = (
            EnumGatewaySessionStatus.DEGRADED
            if elapsed > self._config.session_degraded_after_seconds
            else EnumGatewaySessionStatus.ACTIVE
        )
        if (
            status is EnumGatewaySessionStatus.DEGRADED
            and session.status is not EnumGatewaySessionStatus.DEGRADED
        ):
            self._log_degraded_entry(
                session, reason="heartbeat_gap", unverified_for=elapsed
            )
        updated_session = session.model_copy(
            update={"status": status, "last_heartbeat_at": now}
        )
        # R3: put_if_present, not put -- if a concurrent detach removed this
        # session during the introspection await above, this write must not
        # resurrect it.
        still_present = await self._session_store.put_if_present(updated_session)
        if not still_present:
            raise SessionNotFoundError(
                f"session {session.session_id} was detached during heartbeat "
                "revalidation"
            )

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
            session=updated_session, termination_reason=None, session_event=event
        )

    async def _terminate(
        self,
        session: ModelGatewaySession,
        *,
        now: datetime,
        reason: EnumGatewaySessionTerminationReason,
        status: EnumGatewaySessionStatus,
        event_type: EnumGatewaySessionEventType,
    ) -> ModelGatewayHeartbeatResponse:
        """Remove the session and return the matching terminal response.

        One teardown path for all three terminal outcomes (revocation,
        enforced expiry, ceiling breach) so none of them can drift into
        deleting the row without reporting it, or reporting it without
        deleting the row. The delete happens before the response is built,
        so a caller that sees a ``termination_reason`` is guaranteed the
        session is already gone.
        """
        await self._session_store.delete(session.session_id)
        terminated_session = session.model_copy(update={"status": status})
        event = ModelGatewaySessionEvent(
            event_type=event_type,
            session_id=session.session_id,
            tenant_id=session.tenant_id,
            tenant_slug=session.tenant_slug,
            principal_id=session.principal_id,
            edge_instance_id=session.edge_instance_id,
            emitted_at=now,
        )
        return ModelGatewayHeartbeatResponse(
            session=terminated_session,
            termination_reason=reason,
            session_event=event,
        )

    async def _enter_degraded(
        self, session: ModelGatewaySession, *, reason: str
    ) -> None:
        """Mark a session DEGRADED and alarm, once, on the transition into it.

        Called only from the introspection-outage path, where the session
        deliberately survives. Re-entry is a no-op: a fleet riding out a
        long Keycloak outage would otherwise emit one alarm per session per
        heartbeat interval, which is how an alarm stops being read.

        The write uses ``put_if_present`` rather than ``put`` for the same
        reason the success path does (OMN-15918 R3): a detach that landed
        during the introspection await must not be undone by this handler
        marking the row degraded.
        """
        if session.status is EnumGatewaySessionStatus.DEGRADED:
            return
        degraded_session = session.model_copy(
            update={"status": EnumGatewaySessionStatus.DEGRADED}
        )
        await self._session_store.put_if_present(degraded_session)
        self._log_degraded_entry(
            session,
            reason=reason,
            unverified_for=session_policy.unverified_seconds(
                session, now=datetime.now(UTC)
            ),
        )

    def _log_degraded_entry(
        self, session: ModelGatewaySession, *, reason: str, unverified_for: float
    ) -> None:
        """Structured alarm on ENTRY to degraded mode.

        Entry, not breach: the window in which this gateway cannot observe
        a revocation opens here, and the operator needs it while there is
        still time to act, not once the ceiling has already torn sessions
        down.
        """
        logger.warning(
            "gateway session entered degraded mode",
            extra={
                "alarm": "gateway.session.degraded_entered",
                "session_id": str(session.session_id),
                "tenant_id": str(session.tenant_id),
                "tenant_slug": session.tenant_slug,
                "edge_instance_id": session.edge_instance_id,
                "degraded_reason": reason,
                "unverified_seconds": unverified_for,
                "ceiling_seconds": self._config.max_unverified_session_seconds,
            },
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

    async def _introspect(
        self,
        *,
        access_token: str,
        client_id: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """RFC 7662 token introspection. Returns True iff Keycloak reports ``active``.

        Two failure classes, deliberately not conflated:
          - Genuine revocation (a clean HTTP 200 whose body says
            ``active: false``, or whose ``client_id`` does not match the
            session's) returns ``False`` -- the caller treats this as real
            revocation and tears the session down.
          - An outage (circuit open, transport error, non-200, or malformed
            body) raises ``InfraUnavailableError`` -- the caller must NOT
            treat this as revocation. A Keycloak blip must never mass-revoke
            every active session on its next heartbeat.
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
            raise token_validator.TokenValidationError(
                "Keycloak introspection secret refs resolved to None despite required=True"
            )
        introspection_url = introspection_url_secret.get_secret_value()
        admin_client_id = admin_client_id_secret.get_secret_value()
        admin_client_secret = admin_client_secret_secret.get_secret_value()

        error_context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP,
            operation="introspect",
            correlation_id=correlation_id,
        )

        async with self._introspection_circuit._circuit_breaker_lock:
            await self._introspection_circuit._check_circuit_breaker(
                operation="introspect", correlation_id=correlation_id
            )

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
        except httpx.HTTPError as exc:
            async with self._introspection_circuit._circuit_breaker_lock:
                await self._introspection_circuit._record_circuit_failure(
                    operation="introspect", correlation_id=correlation_id
                )
            raise InfraUnavailableError(
                "Keycloak introspection endpoint unreachable",
                context=error_context,
            ) from exc

        if response.status_code != 200:
            async with self._introspection_circuit._circuit_breaker_lock:
                await self._introspection_circuit._record_circuit_failure(
                    operation="introspect", correlation_id=correlation_id
                )
            raise InfraUnavailableError(
                f"Keycloak introspection endpoint returned HTTP {response.status_code}",
                context=error_context,
            )

        try:
            body = response.json()
        except ValueError as exc:
            async with self._introspection_circuit._circuit_breaker_lock:
                await self._introspection_circuit._record_circuit_failure(
                    operation="introspect", correlation_id=correlation_id
                )
            raise InfraUnavailableError(
                "Keycloak introspection response was not valid JSON",
                context=error_context,
            ) from exc

        async with self._introspection_circuit._circuit_breaker_lock:
            await self._introspection_circuit._reset_circuit_breaker()

        active = body.get("active")
        if active is not True:
            return False
        # Defense in depth: introspection must confirm the same client_id the
        # session was attached with. A token re-issued for a *different*
        # tenant client must never validate a stale session's heartbeat.
        return str(body.get("client_id", "")) == client_id
