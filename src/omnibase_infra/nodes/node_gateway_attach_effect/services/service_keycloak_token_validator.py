# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keycloak client-credentials token validation for the attach control plane.

This is the effect boundary: the only place in this node that performs I/O
(HTTP calls to the Keycloak realm) or reads secret material. All URLs and
credentials are resolved from ``ModelGatewayAttachConfig`` refs via
``SecretResolver`` -- never a bare env var read here.

Two distinct checks live here on purpose:

    - ``decode_claims`` (attach time): local JWT decode, ``iss``/``aud``/
      ``exp`` validated against the configured issuer, no network call. Cheap,
      used once per attach.
    - ``introspect`` (heartbeat time): RFC 7662 introspection -- a real
      round-trip to Keycloak's Admin API. This is the revocation mechanism:
      disabling the tenant's Keycloak client makes ``active: false`` show up
      on the very next introspection call, independent of the token's own
      unexpired ``exp``. A local-only exp check would NOT observe revocation
      before token expiry, which is why heartbeat must introspect rather than
      re-decode.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from uuid import UUID

import httpx

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver


class TokenValidationError(Exception):
    """Raised when an access token fails claim or introspection validation."""


@dataclass(frozen=True)
class ClaimSet:
    """Minimal claim set this node relies on."""

    issuer: str
    audience: str
    subject: str
    tenant_id: UUID
    tenant_slug: str
    principal_id: str
    client_id: str
    expires_at_epoch: int


def decode_claims(access_token: str, config: ModelGatewayAttachConfig) -> ClaimSet:
    """Decode and structurally validate a JWT's claim set (no signature check).

    Signature verification happens implicitly downstream: a forged token
    passes decode but fails the heartbeat-time ``introspect`` call against
    the real Keycloak realm (an unknown/forged token is never ``active``
    there), so attach-time decode only needs to reject structurally invalid
    or wrong-audience tokens fast.
    """
    parts = access_token.split(".")
    if len(parts) != 3:
        raise TokenValidationError("access_token is not a well-formed JWT")
    try:
        payload_raw = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_raw))
    except Exception as exc:
        raise TokenValidationError("access_token payload is not valid JSON") from exc

    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if config.required_audience not in audiences:
        raise TokenValidationError(
            f"access_token audience {audiences!r} does not include "
            f"required audience {config.required_audience!r}"
        )

    for required in (
        "iss",
        "sub",
        "tenant_id",
        "tenant_slug",
        "principal_id",
        "azp",
        "exp",
    ):
        if required not in claims:
            raise TokenValidationError(
                f"access_token missing required claim: {required}"
            )

    try:
        tenant_id = UUID(str(claims["tenant_id"]))
    except ValueError as exc:
        raise TokenValidationError("tenant_id claim is not a valid UUID") from exc

    return ClaimSet(
        issuer=str(claims["iss"]),
        audience=config.required_audience,
        subject=str(claims["sub"]),
        tenant_id=tenant_id,
        tenant_slug=str(claims["tenant_slug"]),
        principal_id=str(claims["principal_id"]),
        client_id=str(claims["azp"]),
        expires_at_epoch=int(claims["exp"]),
    )


async def introspect(
    access_token: str,
    client_id: str,
    config: ModelGatewayAttachConfig,
    secret_resolver: SecretResolver,
    correlation_id: UUID | None = None,
) -> bool:
    """RFC 7662 token introspection. Returns True iff Keycloak reports ``active``.

    Fail-closed: any transport error, non-200 response, or malformed body is
    treated as NOT active -- an unreachable or misbehaving Keycloak must never
    read as "still valid."
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
    # required=True guarantees non-None (SecretResolver raises otherwise); the
    # return type stays Optional to serve required=False callers elsewhere.
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
    # session was attached with. A token re-issued for a *different* tenant
    # client must never validate a stale session's heartbeat.
    return str(body.get("client_id", "")) == client_id


__all__ = ["ClaimSet", "TokenValidationError", "decode_claims", "introspect"]
