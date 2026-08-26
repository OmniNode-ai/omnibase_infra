# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared RFC 7662 introspection seam for the gateway handlers (OMN-16032).

Why this module exists, and why it is where it is
-------------------------------------------------
Introspection used to be a private method on ``HandlerGatewayHeartbeat``,
which made revocation observable only on the *next heartbeat*: a Keycloak
client the operator had disabled could still complete a fresh attach for as
long as one of its already-minted tokens stayed within ``exp`` (OMN-16032).
Closing that window means the attach handler has to make the same call, so
the call has to live somewhere both handlers can reach.

It lives under ``handlers/`` because that is where this repo's
imperative-contract guard requires raw-transport calls to sit -- a
freestanding ``services/`` module performing ``httpx`` I/O is exactly what
``scan_freestanding_imperative_io`` exists to reject (see
``onex_change_control.validators.arch_handler_contract_compliance
._is_node_governed_module``, which exempts ``node_*/handlers/**`` and audits
everything else).

The leading underscore is load-bearing, not stylistic: the handler-contract
compliance scanner treats every non-underscore ``*.py`` under ``node_*/
handlers/`` as a routed handler and raises ``MISSING_HANDLER_ROUTING`` for
any file absent from the contract's ``handler_routing`` block
(``handler_contract_compliance.scan_node_handlers``). This is a shared
internal seam, not a fourth operation, so it must not be declared there --
and the underscore is how the scanner is told which it is.

Circuit-breaker ownership stays with the CALLER. ``MixinAsyncCircuitBreaker``
keeps its state on ``self``, one breaker per instance, and the JWKS endpoint
and the admin-credentialed introspection endpoint fail independently. Each
handler therefore passes in its own guard instance rather than this module
owning a shared one -- a module-level breaker would merge two handlers' and
two endpoints' failure counts into a single trip.
"""

from __future__ import annotations

from uuid import UUID

import httpx

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_keycloak_token_validator as token_validator,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

__all__ = ["GatewayCircuitBreakerGuard", "introspect_token"]


class GatewayCircuitBreakerGuard(MixinAsyncCircuitBreaker):
    """Composition wrapper: lets one handler hold >1 independent circuit breaker.

    ``MixinAsyncCircuitBreaker`` keeps its state on ``self`` (one breaker per
    instance). The gateway handlers talk to two independently-failing
    Keycloak surfaces (the public JWKS endpoint and the admin-credentialed
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


async def introspect_token(
    *,
    config: ModelGatewayAttachConfig,
    secret_resolver: SecretResolver,
    circuit: GatewayCircuitBreakerGuard,
    access_token: str,
    client_id: str,
    correlation_id: UUID | None = None,
) -> bool:
    """RFC 7662 token introspection. Returns True iff Keycloak reports ``active``.

    Two failure classes, deliberately not conflated:
      - Genuine revocation (a clean HTTP 200 whose body says ``active:
        false``, or whose ``client_id`` does not match ``client_id``) returns
        ``False`` -- the caller treats this as real revocation: the heartbeat
        path tears the session down, the attach path refuses to open one.
      - An outage (circuit open, transport error, non-200, or malformed body)
        raises ``InfraUnavailableError`` -- the caller must NOT treat this as
        revocation. A Keycloak blip must never mass-revoke every active
        session on its next heartbeat, nor present to every attaching runtime
        as a rejected credential.

    Args:
        config: resolved node config carrying the introspection/admin refs.
        secret_resolver: resolves those refs to live values.
        circuit: the CALLER's introspection breaker -- see the module
            docstring for why ownership is not moved in here.
        access_token: the token to introspect.
        client_id: the client the caller expects this token to belong to
            (the stored session's client on heartbeat; the verified token's
            ``azp`` on attach).
        correlation_id: propagated into secret resolution and error context.
    """
    introspection_url_secret = await secret_resolver.get_secret_async(
        config.keycloak_introspection_ref,
        required=True,
        correlation_id=correlation_id,
    )
    admin_client_id_secret = await secret_resolver.get_secret_async(
        f"{config.keycloak_admin_client_ref}.client_id",
        required=True,
        correlation_id=correlation_id,
    )
    admin_client_secret_secret = await secret_resolver.get_secret_async(
        f"{config.keycloak_admin_client_ref}.client_secret",
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

    async with circuit._circuit_breaker_lock:
        await circuit._check_circuit_breaker(
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
        async with circuit._circuit_breaker_lock:
            await circuit._record_circuit_failure(
                operation="introspect", correlation_id=correlation_id
            )
        raise InfraUnavailableError(
            "Keycloak introspection endpoint unreachable",
            context=error_context,
        ) from exc

    if response.status_code != 200:
        async with circuit._circuit_breaker_lock:
            await circuit._record_circuit_failure(
                operation="introspect", correlation_id=correlation_id
            )
        raise InfraUnavailableError(
            f"Keycloak introspection endpoint returned HTTP {response.status_code}",
            context=error_context,
        )

    try:
        body = response.json()
    except ValueError as exc:
        async with circuit._circuit_breaker_lock:
            await circuit._record_circuit_failure(
                operation="introspect", correlation_id=correlation_id
            )
        raise InfraUnavailableError(
            "Keycloak introspection response was not valid JSON",
            context=error_context,
        ) from exc

    async with circuit._circuit_breaker_lock:
        await circuit._reset_circuit_breaker()

    active = body.get("active")
    if active is not True:
        return False
    # Defense in depth: introspection must confirm the same client_id the
    # caller expects. A token re-issued for a *different* tenant client must
    # never validate a stale session's heartbeat, nor open a session bound to
    # the client its own claims named.
    return str(body.get("client_id", "")) == client_id
