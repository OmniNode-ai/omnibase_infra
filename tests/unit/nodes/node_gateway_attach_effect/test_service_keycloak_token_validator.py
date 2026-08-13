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
import time
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
    """A token shaped the way the real ga-* Keycloak client mints them.

    OMN-16023 replaced the previous ``exp: 9999999999`` (year 2286, no
    ``iat`` at all) with a real 900s window. That old fixture was not a
    token any IdP would issue, and its absence of ``iat`` meant the
    lifetime bound this module now asserts had nothing to measure.
    """
    issued_at = int(time.time())
    base: dict[str, object] = {
        "iss": EXPECTED_ISSUER,
        "sub": "svc-acct-abc",
        "aud": "gateway-attach",
        "tenant_id": str(TENANT_ID),
        "tenant_slug": "acme",
        "principal_id": "t-11111111111111111111111111111111",
        "azp": "gw-tenant-acme",
        "iat": issued_at,
        "exp": issued_at + 900,
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


class TestOmn16023ValidatorAssertions:
    """OMN-16023: assert the token's shape, do not trust the IdP's config.

    Both invariants below were previously properties of Keycloak client
    configuration rather than checks the relying party makes. Configuration
    is not an invariant: a realm-level accessTokenLifespan bump or one
    added audience mapper silently widens every token minted thereafter,
    and nothing on this side notices. Once the OMN-15952 re-grant loop
    exists a single silent widening is re-minted ~96x/day/runtime, so a
    one-time IdP misconfiguration becomes continuous exposure.

    RED-first: both tests fail against the pre-OMN-16023 validator on a
    wrong outcome (the bad token is accepted), not on a missing symbol.
    """

    def test_rejects_dual_audience_token(self, config, tenant_key, jwks) -> None:
        """The audience check must be exact set equality, not membership.

        PyJWT's ``audience=`` argument passes as soon as the expected value
        is IN the token's aud list, so a token carrying BOTH
        ``gateway-attach`` and a second audience sails through today. The
        known-bad broker audience is only one instance of that class; a
        mapper that later adds ``onex-api`` produces exactly the
        dual-audience token invariant 4 of the renewal design forbids, and
        a broker-only denylist would never see it.
        """
        token = sign_claims(
            tenant_key, _valid_claims(aud=["gateway-attach", "onex-api"])
        )
        with pytest.raises(validator.TokenValidationError, match="audience"):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_accepts_single_audience_token_presented_as_a_list(
        self, config, tenant_key, jwks
    ) -> None:
        """Exact-set-equality, not exact-type-equality.

        Keycloak may serialize a single audience as either a bare string
        or a one-element array; both are the same claim and both must pass.
        Without this the strict check would reject legitimate tokens on a
        serialization detail.
        """
        token = sign_claims(tenant_key, _valid_claims(aud=["gateway-attach"]))
        claims = validator.verify_and_decode_claims(
            token, jwks, config, expected_issuer=EXPECTED_ISSUER
        )
        assert claims.audience == "gateway-attach"

    def test_rejects_token_whose_lifetime_exceeds_the_bound(
        self, config, tenant_key, jwks
    ) -> None:
        """``exp - iat`` over the bound fails closed regardless of signature.

        The token is validly signed, unexpired, correctly audienced and
        issued by the right realm -- it is wrong only in that Keycloak was
        configured to hand out a longer-lived credential than the gateway
        accepts. That is precisely the case the assertion exists for.
        """
        issued_at = int(time.time())
        # Literal 901 rather than the config constant: this is the ticket's
        # headline assertion, and it should fail on the pre-fix validator
        # by ACCEPTING a 901s token, not by tripping over a config field
        # that does not exist yet. The boundary case below pins the
        # constant itself.
        token = sign_claims(
            tenant_key,
            _valid_claims(iat=issued_at, exp=issued_at + 901),
        )
        with pytest.raises(validator.TokenValidationError, match="lifetime"):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_accepts_token_exactly_at_the_lifetime_bound(
        self, config, tenant_key, jwks
    ) -> None:
        """The bound is a maximum allowed, not a value forbidden.

        A correctly-configured 900s Keycloak client mints tokens sitting
        exactly on this boundary, so an off-by-one here would reject every
        real token.
        """
        issued_at = int(time.time())
        token = sign_claims(
            tenant_key,
            _valid_claims(
                iat=issued_at,
                exp=issued_at + config.max_attach_token_lifetime_seconds,
            ),
        )
        claims = validator.verify_and_decode_claims(
            token, jwks, config, expected_issuer=EXPECTED_ISSUER
        )
        assert claims.tenant_id == TENANT_ID

    def test_rejects_token_without_iat(self, config, tenant_key, jwks) -> None:
        """No ``iat`` means no measurable lifetime, so it fails closed.

        Without this the TTL assertion is trivially bypassable: an IdP that
        omits ``iat`` (or an attacker who strips it, if anything downstream
        ever accepted an unsigned rebuild) would leave nothing to measure
        ``exp`` against, and the bound would silently not apply.
        """
        claims = _valid_claims()
        del claims["iat"]
        token = sign_claims(tenant_key, claims)
        with pytest.raises(validator.TokenValidationError, match="iat"):
            validator.verify_and_decode_claims(
                token, jwks, config, expected_issuer=EXPECTED_ISSUER
            )

    def test_lifetime_bound_is_not_the_session_ceiling(self, config) -> None:
        """The bound has its own constant, per the ticket's explicit requirement.

        ``max_session_ttl_seconds`` (3600s) bounds how long a SESSION may
        live and is applied by the attach handler.
        ``max_attach_token_lifetime_seconds`` (900s) bounds the shape of a
        presented TOKEN and is applied here, at a different layer. Reusing
        the session ceiling for this check would silently accept 3600s
        tokens.
        """
        assert config.max_attach_token_lifetime_seconds == 900
        assert (
            config.max_attach_token_lifetime_seconds != config.max_session_ttl_seconds
        )
