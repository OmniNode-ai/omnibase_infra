# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the Keycloak claim-decode service.

Covers ``decode_claims``: local structural/claim validation (attach time),
the only function this module owns. The RFC 7662 introspection round-trip
(heartbeat time, the revocation-detection mechanism) moved to
``HandlerGatewayHeartbeat._introspect`` -- see
``test_handlers.py::TestHeartbeatIntrospection`` for that coverage.
"""

from __future__ import annotations

import base64
import json
from uuid import UUID

import pytest

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_keycloak_token_validator as validator,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
EXPECTED_ISSUER = "https://keycloak.example/realms/omninode"


def _fake_jwt(claims: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    )
    return f"{header}.{payload}.sig"


def _valid_claims(**overrides: object) -> dict[str, object]:
    base = {
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


class TestDecodeClaims:
    def test_valid_token_decodes(self, config: ModelGatewayAttachConfig) -> None:
        token = _fake_jwt(_valid_claims())
        claims = validator.decode_claims(token, config, expected_issuer=EXPECTED_ISSUER)
        assert claims.tenant_id == TENANT_ID
        assert claims.tenant_slug == "acme"
        assert claims.client_id == "gw-tenant-acme"
        assert claims.issuer == EXPECTED_ISSUER

    def test_malformed_token_raises(self, config: ModelGatewayAttachConfig) -> None:
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims(
                "not-a-jwt", config, expected_issuer=EXPECTED_ISSUER
            )

    def test_wrong_audience_raises(self, config: ModelGatewayAttachConfig) -> None:
        token = _fake_jwt(_valid_claims(aud="some-other-audience"))
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims(token, config, expected_issuer=EXPECTED_ISSUER)

    def test_missing_tenant_claim_raises(
        self, config: ModelGatewayAttachConfig
    ) -> None:
        claims = _valid_claims()
        del claims["tenant_id"]
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims(
                _fake_jwt(claims), config, expected_issuer=EXPECTED_ISSUER
            )

    def test_non_uuid_tenant_id_raises(self, config: ModelGatewayAttachConfig) -> None:
        token = _fake_jwt(_valid_claims(tenant_id="not-a-uuid"))
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims(token, config, expected_issuer=EXPECTED_ISSUER)

    def test_mismatched_issuer_raises(self, config: ModelGatewayAttachConfig) -> None:
        """R3: iss must be validated against the configured issuer, not just present."""
        token = _fake_jwt(_valid_claims(iss="https://attacker.example/realms/evil"))
        with pytest.raises(validator.TokenValidationError, match="issuer"):
            validator.decode_claims(token, config, expected_issuer=EXPECTED_ISSUER)

    def test_matching_issuer_happy_path(self, config: ModelGatewayAttachConfig) -> None:
        """R3: a token whose iss matches the resolved expected issuer decodes clean."""
        token = _fake_jwt(_valid_claims(iss=EXPECTED_ISSUER))
        claims = validator.decode_claims(token, config, expected_issuer=EXPECTED_ISSUER)
        assert claims.issuer == EXPECTED_ISSUER
