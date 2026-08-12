# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keycloak client-credentials claim decode for the attach control plane.

``decode_claims`` (attach time): local JWT decode, ``iss``/``aud``/``exp``
validated against the configured issuer, no network call. Cheap, used once
per attach.

The heartbeat-time RFC 7662 introspection round-trip (the revocation
mechanism -- disabling the tenant's Keycloak client makes ``active: false``
show up on the very next introspection call, independent of the token's own
unexpired ``exp``) is NOT here: that is the only I/O this node performs, and
it lives inline in ``HandlerGatewayHeartbeat`` (the sole caller) so the actual
HTTP call stays under ``handlers/`` per the imperative-contract-guard's
freestanding-IO boundary. This module stays I/O-free by design -- do not
re-add an httpx call here.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from uuid import UUID

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)


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
    passes decode but fails the heartbeat-time introspection call (see
    ``HandlerGatewayHeartbeat._introspect``) against the real Keycloak realm
    (an unknown/forged token is never ``active`` there), so attach-time
    decode only needs to reject structurally invalid or wrong-audience
    tokens fast.
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


__all__ = ["ClaimSet", "TokenValidationError", "decode_claims"]
