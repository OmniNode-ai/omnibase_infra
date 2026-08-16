# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the Keycloak claim-verification service.

Covers ``verify_and_decode_claims``: signature verification against a JWKS
keyset plus claim validation (attach time, and the heartbeat/detach
identity-binding re-check). OMN-15918 R1: previously ``decode_claims`` only
base64-decoded the payload segment and never verified the signature at all
-- these tests are the RED-before/GREEN-after proof that a forged,
wrong-key-signed, or ``alg: none`` token is rejected before its claims are
ever trusted.
"""

from __future__ import annotations

import base64
import json
from uuid import UUID

import jwt
import pytest

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_keycloak_token_validator as validator,
)
from tests.unit.nodes.node_gateway_attach_effect._jwt_test_support import (
    OTHER_KID,
    TENANT_KID,
    generate_key_material,
    jwks_response_body,
    sign_claims,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
EXPECTED_ISSUER = "https://keycloak.example/realms/omninode"


def _valid_claims(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "iss": EXPECTED_ISSUER,
        "sub": "svc-acct-abc",
        "aud": "gateway-attach",
        "tenant_id": str(TENANT_ID),
        "tenant_slug": "acme",
        "principal_id": "t-11111111111111111111111111111111",
        "azp": "gw-tenant-acme",
        "exp": 9999999999,
    }
    base.update(overrides)
    return base


@pytest.fixture
def config() -> ModelGatewayAttachConfig:
    return ModelGatewayAttachConfig()


@pytest.fixture
def tenant_key():
    return generate_key_material(TENANT_KID)


@pytest.fixture
def attacker_key():
    return generate_key_material(OTHER_KID)


@pytest.fixture
def jwks(tenant_key):
    return jwks_response_body(tenant_key)["keys"]


class TestVerifyAndDecodeClaims:
    def test_valid_signed_token_decodes(self, config, tenant_key, jwks) -> None:
        token = sign_claims(tenant_key, _valid_claims())
        claims = validator.verify_and_decode_claims(
            token, jwks, config, expected_issuer=EXPECTED_ISSUER
        )
        assert claims.tenant_id == TENANT_ID
        assert claims.tenant_slug == "acme"
        assert claims.client_id == "gw-tenant-acme"
        assert claims.issuer == EXPECTED_ISSUER

    def test_malformed_token_raises(self, config, jwks) -> None:
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                "not-a-jwt", jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_alg_none_token_rejected(self, config, jwks) -> None:
        """R1: a structurally-valid alg:none token must never verify."""
        header = base64.urlsafe_b64encode(b'{"alg":"none","kid":"whatever"}')
        header = header.rstrip(b"=")
        payload = base64.urlsafe_b64encode(json.dumps(_valid_claims()).encode()).rstrip(
            b"="
        )
        forged_token = f"{header.decode()}.{payload.decode()}."
        with pytest.raises(validator.TokenValidationError, match="alg"):
            validator.verify_and_decode_claims(
                forged_token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_token_signed_by_wrong_key_rejected(
        self, config, attacker_key, jwks
    ) -> None:
        """R1: a token signed by a key NOT in the resolved JWKS must be rejected.

        This is the forged-token proof CodeRabbit flagged: the token is
        structurally perfect (right claims, right header shape) but signed
        by a private key that never matches anything in the tenant's real
        JWKS keyset.
        """
        forged_with_unknown_kid = sign_claims(attacker_key, _valid_claims())
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                forged_with_unknown_kid, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_token_with_known_kid_but_wrong_signature_rejected(
        self, config, tenant_key, attacker_key, jwks
    ) -> None:
        """R1: an attacker who signs with the WRONG key but claims the RIGHT kid
        (to pass the kid lookup) must still fail signature verification."""
        forgery = jwt.encode(
            _valid_claims(),
            attacker_key.private_key,
            algorithm="RS256",
            headers={"kid": tenant_key.kid},
        )
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                forgery, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_wrong_audience_raises(self, config, tenant_key, jwks) -> None:
        token = sign_claims(tenant_key, _valid_claims(aud="some-other-audience"))
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_missing_tenant_claim_raises(self, config, tenant_key, jwks) -> None:
        claims = _valid_claims()
        del claims["tenant_id"]
        token = sign_claims(tenant_key, claims)
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_non_uuid_tenant_id_raises(self, config, tenant_key, jwks) -> None:
        token = sign_claims(tenant_key, _valid_claims(tenant_id="not-a-uuid"))
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_mismatched_issuer_raises(self, config, tenant_key, jwks) -> None:
        token = sign_claims(
            tenant_key,
            _valid_claims(iss="https://attacker.example/realms/evil"),
        )
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_matching_issuer_happy_path(self, config, tenant_key, jwks) -> None:
        token = sign_claims(tenant_key, _valid_claims(iss=EXPECTED_ISSUER))
        claims = validator.verify_and_decode_claims(
            token, jwks, config, expected_issuer=EXPECTED_ISSUER
        )
        assert claims.issuer == EXPECTED_ISSUER

    def test_expired_token_raises(self, config, tenant_key, jwks) -> None:
        token = sign_claims(tenant_key, _valid_claims(exp=1))
        with pytest.raises(validator.TokenValidationError):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_unknown_kid_raises(self, config, tenant_key, jwks) -> None:
        """kid present in header but absent from the resolved JWKS keyset."""
        token = sign_claims(tenant_key, _valid_claims())
        with pytest.raises(validator.TokenValidationError, match="kid"):
            validator.verify_and_decode_claims(
                token, [], config, expected_issuer=EXPECTED_ISSUER
            )
