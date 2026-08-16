# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared RS256 signing/JWKS test support for node_gateway_attach_effect.

Not a test module itself (no ``test_`` prefix, not collected by pytest) --
OMN-15918 replaced the previous ``alg: none`` fake-JWT fixtures with real
RS256-signed tokens once ``verify_and_decode_claims`` started verifying
signatures against a JWKS keyset. Centralized here so
``test_handlers.py`` and ``test_service_keycloak_token_validator.py`` share
one keypair-generation/signing/JWKS-serialization path rather than
duplicating it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt.algorithms import RSAAlgorithm

TENANT_KID = "test-tenant-key-1"
OTHER_KID = "attacker-key-1"


@dataclass(frozen=True)
class KeyMaterial:
    """A generated RSA keypair plus its serialized JWKS entry."""

    kid: str
    private_key: RSAPrivateKey
    jwk: dict[str, Any]


def generate_key_material(kid: str) -> KeyMaterial:
    """Generate a fresh RSA keypair and its public JWKS representation."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = kid
    jwk["alg"] = "RS256"
    jwk["use"] = "sig"
    return KeyMaterial(kid=kid, private_key=private_key, jwk=jwk)


def sign_claims(key_material: KeyMaterial, claims: dict[str, Any]) -> str:
    """Sign ``claims`` as a real RS256 JWT using ``key_material``'s private key."""
    return jwt.encode(
        claims,
        key_material.private_key,
        algorithm="RS256",
        headers={"kid": key_material.kid},
    )


def jwks_response_body(*key_materials: KeyMaterial) -> dict[str, Any]:
    """Build a JWKS response body (``{"keys": [...]}``) from public JWK entries."""
    return {"keys": [km.jwk for km in key_materials]}


__all__ = [
    "OTHER_KID",
    "TENANT_KID",
    "KeyMaterial",
    "generate_key_material",
    "jwks_response_body",
    "sign_claims",
]
