# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the Keycloak token-validation service.

Covers the two independent checks this module owns:
    - decode_claims: local structural/claim validation (attach time).
    - introspect: RFC 7662 round-trip (heartbeat time) -- this is the
      revocation-detection mechanism under adversarial test here.
"""

from __future__ import annotations

import base64
import json
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr

from omnibase_infra.nodes.node_gateway_attach_effect.models.model_gateway_attach_config import (
    ModelGatewayAttachConfig,
)
from omnibase_infra.nodes.node_gateway_attach_effect.services import (
    service_keycloak_token_validator as validator,
)

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")


def _fake_jwt(claims: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    )
    return f"{header}.{payload}.sig"


def _valid_claims(**overrides: object) -> dict[str, object]:
    base = {
        "iss": "https://keycloak.example/realms/omninode",
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
        claims = validator.decode_claims(token, config)
        assert claims.tenant_id == TENANT_ID
        assert claims.tenant_slug == "acme"
        assert claims.client_id == "gw-tenant-acme"

    def test_malformed_token_raises(self, config: ModelGatewayAttachConfig) -> None:
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims("not-a-jwt", config)

    def test_wrong_audience_raises(self, config: ModelGatewayAttachConfig) -> None:
        token = _fake_jwt(_valid_claims(aud="some-other-audience"))
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims(token, config)

    def test_missing_tenant_claim_raises(
        self, config: ModelGatewayAttachConfig
    ) -> None:
        claims = _valid_claims()
        del claims["tenant_id"]
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims(_fake_jwt(claims), config)

    def test_non_uuid_tenant_id_raises(self, config: ModelGatewayAttachConfig) -> None:
        token = _fake_jwt(_valid_claims(tenant_id="not-a-uuid"))
        with pytest.raises(validator.TokenValidationError):
            validator.decode_claims(token, config)


class _FakeSecretResolver:
    """Duck-typed stand-in for SecretResolver.get_secret_async."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    async def get_secret_async(
        self, logical_name: str, required: bool = True, correlation_id: object = None
    ) -> SecretStr:
        return SecretStr(self._values[logical_name])


def _resolver(config: ModelGatewayAttachConfig) -> _FakeSecretResolver:
    return _FakeSecretResolver(
        {
            config.keycloak_introspection_ref: "https://keycloak.example/realms/omninode/protocol/openid-connect/token/introspect",
            f"{config.keycloak_admin_client_ref}.client_id": "admin-cli",
            f"{config.keycloak_admin_client_ref}.client_secret": "admin-secret",
        }
    )


class _FakeResponse:
    def __init__(self, status_code: int, body: dict[str, object]) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict[str, object]:
        return self._body


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse | Exception) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestIntrospect:
    @pytest.mark.asyncio
    async def test_active_matching_client_returns_true(
        self, monkeypatch: pytest.MonkeyPatch, config: ModelGatewayAttachConfig
    ) -> None:
        response = _FakeResponse(200, {"active": True, "client_id": "gw-tenant-acme"})
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **_: _FakeAsyncClient(response)
        )
        result = await validator.introspect(
            access_token="tok",
            client_id="gw-tenant-acme",
            config=config,
            secret_resolver=_resolver(config),  # type: ignore[arg-type]
            correlation_id=uuid4(),
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_revoked_client_returns_false(
        self, monkeypatch: pytest.MonkeyPatch, config: ModelGatewayAttachConfig
    ) -> None:
        # This is the revocation case: Keycloak reports active:false once the
        # tenant's confidential client has been disabled.
        response = _FakeResponse(200, {"active": False})
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **_: _FakeAsyncClient(response)
        )
        result = await validator.introspect(
            access_token="tok",
            client_id="gw-tenant-acme",
            config=config,
            secret_resolver=_resolver(config),  # type: ignore[arg-type]
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_client_id_mismatch_returns_false(
        self, monkeypatch: pytest.MonkeyPatch, config: ModelGatewayAttachConfig
    ) -> None:
        response = _FakeResponse(
            200, {"active": True, "client_id": "gw-tenant-someone-else"}
        )
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **_: _FakeAsyncClient(response)
        )
        result = await validator.introspect(
            access_token="tok",
            client_id="gw-tenant-acme",
            config=config,
            secret_resolver=_resolver(config),  # type: ignore[arg-type]
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_transport_error_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, config: ModelGatewayAttachConfig
    ) -> None:
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda **_: _FakeAsyncClient(httpx.ConnectError("unreachable")),
        )
        result = await validator.introspect(
            access_token="tok",
            client_id="gw-tenant-acme",
            config=config,
            secret_resolver=_resolver(config),  # type: ignore[arg-type]
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_non_200_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, config: ModelGatewayAttachConfig
    ) -> None:
        response = _FakeResponse(500, {})
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **_: _FakeAsyncClient(response)
        )
        result = await validator.introspect(
            access_token="tok",
            client_id="gw-tenant-acme",
            config=config,
            secret_resolver=_resolver(config),  # type: ignore[arg-type]
        )
        assert result is False
