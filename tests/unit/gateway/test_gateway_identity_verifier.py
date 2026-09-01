# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex auth status`` asks the gateway, not the config file (OMN-17028).

WHAT THESE TESTS PIN
    1. The identity reported is the SERVER'S answer. A test that let the local
       ``tenant_slug`` satisfy the assertion would pass against a verifier that
       never made the call, which is the whole failure mode.
    2. Every non-happy outcome is a NON-ZERO exit with a named remediation:
       refused, unreachable, non-JSON, and 200-without-an-identity are four
       different messages because they need four different operator actions.
       None of them is "print the config and exit 0".
    3. The key is presented in ``x-api-key`` and never appears in the URL, the
       output, or an error message.
    4. A stored label that disagrees with the gateway's own answer FAILS.
       The label names the ref the key is filed under and appears in every
       later message about this credential; leaving the two disagreeing means
       every one of those messages names the wrong tenant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.cli_auth import auth_group
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.gateway.client.gateway_identity_verifier import (
    GATEWAY_WHOAMI_PATH,
    GatewayIdentityVerifier,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)

from .conftest import FakeHttpResponse

pytestmark = pytest.mark.unit

_BASE = "https://dev.api.invalid"
_KEY = "onxk_test-not-a-real-key"  # pragma: allowlist secret
_SERVER_SLUG = "acme"
_TENANT_ID = "22222222-2222-4222-8222-222222222222"


class WhoamiTransport:
    """Near-side fake of onex-api's ``GET /v1/whoami``.

    Answers only for a request that actually presents the key, so a verifier
    that stopped sending the header would fail here rather than silently keep
    reporting an identity.
    """

    def __init__(self, *, status: int = 200, body: str | None = None) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []
        self._status = status
        self._body = body

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHttpResponse:
        sent = dict(headers or {})
        self.requests.append((url, sent))
        if self._body is not None or self._status != 200:
            return FakeHttpResponse(self._status, self._body or '{"detail":"no"}')
        if sent.get("x-api-key") != _KEY:
            return FakeHttpResponse(401, '{"detail":"Unauthorized"}')
        return FakeHttpResponse(
            200, json.dumps({"tenant_id": _TENANT_ID, "tenant_slug": _SERVER_SLUG})
        )


class UnreachableTransport:
    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHttpResponse:
        raise InfraUnavailableError(
            "no route to host",
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="whoami",
            ),
        )


def _credential(*, tenant_slug: str = _SERVER_SLUG) -> ModelGatewayApiKeyCredential:
    return ModelGatewayApiKeyCredential(
        tenant_slug=tenant_slug,
        api_key=SecretStr(_KEY),
        api_key_ref=f"{tenant_slug}-api-key",
        base_url=_BASE,
    )


class TestVerifier:
    @pytest.mark.asyncio
    async def test_reports_the_tenant_the_gateway_resolved(self) -> None:
        transport = WhoamiTransport()

        identity = await GatewayIdentityVerifier(
            transport=transport, credential=_credential()
        ).verify()

        assert identity.tenant_slug == _SERVER_SLUG
        assert str(identity.tenant_id) == _TENANT_ID
        url, headers = transport.requests[0]
        assert url == f"{_BASE}{GATEWAY_WHOAMI_PATH}"
        assert headers["x-api-key"] == _KEY

    @pytest.mark.asyncio
    async def test_the_key_is_never_in_the_url(self) -> None:
        transport = WhoamiTransport()

        await GatewayIdentityVerifier(
            transport=transport, credential=_credential()
        ).verify()

        assert _KEY not in transport.requests[0][0]

    @pytest.mark.asyncio
    async def test_a_refusal_names_the_ref_and_never_echoes_the_key(self) -> None:
        transport = WhoamiTransport(status=401)

        with pytest.raises(ModelOnexError) as caught:
            await GatewayIdentityVerifier(
                transport=transport, credential=_credential()
            ).verify()

        assert "acme-api-key" in str(caught.value)
        assert _KEY not in str(caught.value)

    @pytest.mark.asyncio
    async def test_unreachable_is_distinguished_from_refused(self) -> None:
        """'Your gateway is not there' is a different action from 'refused'."""
        with pytest.raises(ModelOnexError) as caught:
            await GatewayIdentityVerifier(
                transport=UnreachableTransport(), credential=_credential()
            ).verify()

        message = str(caught.value)
        assert "could not reach" in message
        assert "not judged" in message
        assert "--no-verify" in message

    @pytest.mark.asyncio
    async def test_a_200_whose_tenant_id_is_not_a_uuid_is_refused(self) -> None:
        """An unreadable identity is refused, not reported as verified."""
        transport = WhoamiTransport(
            status=200, body=json.dumps({"tenant_id": "acme", "tenant_slug": "acme"})
        )

        with pytest.raises(ModelOnexError, match="cannot read"):
            await GatewayIdentityVerifier(
                transport=transport, credential=_credential()
            ).verify()

    @pytest.mark.asyncio
    async def test_a_200_carrying_no_identity_is_not_authenticated(self) -> None:
        """A 200 is not the assertion; a resolved tenant is."""
        transport = WhoamiTransport(status=200, body='{"ok": true}')

        with pytest.raises(ModelOnexError, match="no tenant identity"):
            await GatewayIdentityVerifier(
                transport=transport, credential=_credential()
            ).verify()

    @pytest.mark.asyncio
    async def test_a_200_that_is_not_json_is_refused(self) -> None:
        transport = WhoamiTransport(status=200, body="<html>gateway timeout</html>")

        with pytest.raises(ModelOnexError, match="not a JSON object"):
            await GatewayIdentityVerifier(
                transport=transport, credential=_credential()
            ).verify()


class TestAuthStatusCommand:
    """The command surface, over a store the test actually wrote."""

    @pytest.fixture
    def onex_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        store = StoreGatewayCredential(onex_home=home / ".onex")
        store.save_api_key(tenant_slug=_SERVER_SLUG, api_key=_KEY, base_url=_BASE)
        return home / ".onex"

    def test_no_verify_keeps_the_command_purely_local(self, onex_home: Path) -> None:
        result = CliRunner().invoke(auth_group, ["status", "--no-verify"])

        assert result.exit_code == 0, result.output
        assert "tenant API key" in result.output
        assert "not attempted" in result.output
        assert _KEY not in result.output

    def test_status_demands_no_attach_plane_field_of_a_key(
        self, onex_home: Path
    ) -> None:
        """The regression itself: status used to require four absent fields."""
        result = CliRunner().invoke(auth_group, ["status", "--no-verify"])

        assert result.exit_code == 0, result.output
        for absent in ("edge_instance_id", "token_endpoint", "principal_id"):
            assert absent not in result.output

    def test_verification_failure_exits_non_zero(
        self, onex_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unreachable gateway must not read as 'authenticated'."""
        monkeypatch.setattr(
            "omnibase_infra.cli.cli_auth.GatewayTransportHttpx",
            UnreachableTransport,
        )

        result = CliRunner().invoke(auth_group, ["status"])

        assert result.exit_code == 1
        assert "could not reach" in result.output

    def test_a_label_that_disagrees_with_the_gateway_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        StoreGatewayCredential(onex_home=home / ".onex").save_api_key(
            tenant_slug="typo-slug", api_key=_KEY, base_url=_BASE
        )
        monkeypatch.setattr(
            "omnibase_infra.cli.cli_auth.GatewayTransportHttpx",
            WhoamiTransport,
        )

        result = CliRunner().invoke(auth_group, ["status"])

        assert result.exit_code == 1
        assert "authenticates as tenant 'acme'" in result.output
        assert "typo-slug" in result.output

    def test_a_verified_key_prints_the_servers_answer(
        self, onex_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "omnibase_infra.cli.cli_auth.GatewayTransportHttpx",
            WhoamiTransport,
        )

        result = CliRunner().invoke(auth_group, ["status"])

        assert result.exit_code == 0, result.output
        assert "resolved this key" in result.output
        assert _KEY not in result.output
