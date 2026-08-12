# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keycloak client-credentials claim verification for the attach control plane.

``verify_and_decode_claims`` (attach time, and heartbeat/detach identity
re-check): verifies the JWT's signature against a resolved JWKS keyset, then
validates ``iss``/``aud``/``exp`` and decodes the tenant claim set. This
closes OMN-15918 R1: the previous ``decode_claims`` only base64-decoded the
payload segment and never referenced the token's signature at all, so any
structurally-valid token from any signer (forged or otherwise) that happened
to satisfy the other claims would attach and hold an ACTIVE session for up
to one heartbeat interval before introspection caught it.

The JWKS *fetch* (RFC 7517, network I/O) is NOT here -- it lives inline in
the calling handler (``HandlerGatewayAttach._fetch_jwks`` /
``HandlerGatewayHeartbeat._fetch_jwks``), matching the pattern
``HandlerGatewayHeartbeat._introspect`` already established: the
imperative-contract-guard requires raw-transport calls to live under
``handlers/``, never in a freestanding module the guard cannot attribute to
a contract-declared handler. This module stays I/O-free by design -- it
receives an already-fetched JWKS keyset and does signature verification
(CPU-only) against it. Do not re-add an httpx call here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

import jwt
from jwt import PyJWK
from jwt.exceptions import InvalidTokenError

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)

_REQUIRED_CLAIMS = ("exp", "iss", "aud", "sub")
_REQUIRED_TENANT_CLAIMS = ("tenant_id", "tenant_slug", "principal_id", "azp")


class TokenValidationError(Exception):
    """Raised when an access token fails signature, claim, or introspection validation."""


@dataclass(frozen=True)
class ClaimSet:
    """Minimal, signature-verified claim set this node relies on."""

    issuer: str
    audience: str
    subject: str
    tenant_id: UUID
    tenant_slug: str
    principal_id: str
    client_id: str
    expires_at_epoch: int


def _resolve_signing_key(
    jwks_keys: Sequence[Mapping[str, object]], *, kid: str
) -> PyJWK:
    matching_jwk = next((key for key in jwks_keys if key.get("kid") == kid), None)
    if matching_jwk is None:
        raise TokenValidationError(
            f"access_token kid {kid!r} not present in the resolved JWKS keyset"
        )
    try:
        return PyJWK.from_dict(dict(matching_jwk))
    except Exception as exc:
        raise TokenValidationError(
            "JWKS key material for the token's kid is malformed"
        ) from exc


def verify_and_decode_claims(
    access_token: str,
    jwks_keys: Sequence[Mapping[str, object]],
    config: ModelGatewayAttachConfig,
    *,
    expected_issuer: str,
) -> ClaimSet:
    """Verify a JWT's signature against the resolved JWKS keyset, then decode its claims.

    Fails closed on every dimension: an unparsable header, a missing/unknown
    ``kid``, a bad or absent signature, a wrong ``iss``/``aud``, an expired
    ``exp``, or a missing tenant claim all raise ``TokenValidationError``
    before any claim is trusted. ``alg: none`` tokens are rejected implicitly
    -- PyJWT never resolves a signing key for ``none`` and the explicit
    ``algorithms=`` allowlist below never includes it.
    """
    try:
        unverified_header = jwt.get_unverified_header(access_token)
    except Exception as exc:
        raise TokenValidationError("access_token header is not decodable") from exc

    kid = unverified_header.get("kid")
    if not kid or not isinstance(kid, str):
        raise TokenValidationError("access_token header missing kid")

    alg = unverified_header.get("alg")
    if not alg or not isinstance(alg, str) or alg.lower() == "none":
        raise TokenValidationError(
            f"access_token alg header {alg!r} is not an accepted signing algorithm"
        )

    signing_key = _resolve_signing_key(jwks_keys, kid=kid)

    try:
        claims = jwt.decode(
            access_token,
            key=signing_key,
            algorithms=[alg],
            audience=config.required_audience,
            issuer=expected_issuer,
            options={"require": list(_REQUIRED_CLAIMS)},
        )
    except InvalidTokenError as exc:
        raise TokenValidationError(
            f"access_token signature/claims verification failed: {exc}"
        ) from exc

    for required in _REQUIRED_TENANT_CLAIMS:
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


__all__ = ["ClaimSet", "TokenValidationError", "verify_and_decode_claims"]
