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

# ``iat`` is required (OMN-16023) so the token's own lifetime is always
# measurable. Without it the ``exp - iat`` bound below would silently not
# apply to any token that simply omitted the claim.
_REQUIRED_CLAIMS = ("exp", "iat", "iss", "aud", "sub")
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


def _assert_exact_audience(raw_audience: object, *, expected: str) -> None:
    """Require ``aud`` to be exactly one audience: the expected one.

    Set equality, not membership. Keycloak may serialize a single audience
    as a bare string or a one-element array -- both are the same claim, so
    both are normalized to a set before comparison; what is rejected is a
    token carrying any audience beyond the expected one, whether that extra
    audience is the known-bad broker audience or something nobody has
    thought of yet.
    """
    if isinstance(raw_audience, str):
        presented = {raw_audience}
    elif isinstance(raw_audience, (list, tuple, set)):
        presented = {str(entry) for entry in raw_audience}
    else:
        raise TokenValidationError(
            f"access_token aud claim has unusable type {type(raw_audience).__name__}"
        )

    if presented != {expected}:
        raise TokenValidationError(
            "access_token audience set must be exactly "
            f"{{{expected!r}}}, got {sorted(presented)!r} -- a token carrying "
            "any additional audience is rejected"
        )


def _integer_timestamp_claim(claims: Mapping[str, object], name: str) -> int:
    raw_claim = claims[name]
    if isinstance(raw_claim, bool) or not isinstance(raw_claim, int):
        raise TokenValidationError(
            f"access_token {name} claim is not an integer timestamp"
        )
    return raw_claim


def _assert_bounded_lifetime(
    claims: Mapping[str, object], *, max_lifetime_seconds: int
) -> None:
    """Require ``exp - iat`` to be within the accepted attach-token lifetime.

    ``_REQUIRED_CLAIMS`` guarantees both claims are present; this guards
    their values. A non-positive lifetime is rejected as malformed rather
    than merely useless -- ``exp <= iat`` describes no valid window, and
    letting it through would leave the bound satisfied by arithmetic that
    means nothing.
    """
    issued_at = _integer_timestamp_claim(claims, "iat")
    expires_at = _integer_timestamp_claim(claims, "exp")

    lifetime_seconds = expires_at - issued_at
    if lifetime_seconds <= 0:
        raise TokenValidationError(
            f"access_token lifetime is non-positive (exp - iat = {lifetime_seconds}s)"
        )
    if lifetime_seconds > max_lifetime_seconds:
        raise TokenValidationError(
            f"access_token lifetime {lifetime_seconds}s exceeds the maximum "
            f"accepted attach-token lifetime of {max_lifetime_seconds}s"
        )


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

    OMN-16023 added the two assertions PyJWT cannot make for us:

      - ``aud`` is exact set equality against the single expected audience,
        not membership. PyJWT's ``audience=`` argument is satisfied as soon
        as the expected value appears anywhere in the claim, so it accepts
        a dual-audience token; asserting the whole set means any widening
        of the ga-* client's audience mappers fails closed here rather than
        being discovered later.
      - ``exp - iat`` is bounded by ``max_attach_token_lifetime_seconds``.
        PyJWT validates that ``exp`` has not passed; nothing validates how
        far out it was set in the first place.

    Both are relying-party assertions on a presented token, deliberately
    independent of how the IdP is configured -- configuration is not an
    invariant, because a realm-level or mapper-level change silently
    rewrites it for every token minted thereafter.
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

    _assert_exact_audience(claims.get("aud"), expected=config.required_audience)
    _assert_bounded_lifetime(
        claims, max_lifetime_seconds=config.max_attach_token_lifetime_seconds
    )

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
        expires_at_epoch=_integer_timestamp_claim(claims, "exp"),
    )


__all__ = ["ClaimSet", "TokenValidationError", "verify_and_decode_claims"]
