# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit-only full probe flow; fake HTTP/verifier are not live OIDC evidence."""

import base64
import hashlib
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from urllib.parse import urlencode
from uuid import UUID

import pytest

from scripts.runtime_build.prove_sim_preflight_oidc import (
    ISSUER,
    REDIRECT,
    prove,
    verify_browser_token,
)

TENANT = "00000000-0000-4000-8000-000000000001"
SUBJECT = "00000000-0000-4000-8000-000000000002"
TOKEN = "unit-header.unit-payload.unit-signature"


class FakeOIDCAuthError(Exception):
    """Fixture verifier refusal, never a live verification result."""


@dataclass
class UnitFlow:
    credentials: Path
    realm: Path
    output: Path
    claims: dict[str, str] = field(
        default_factory=lambda: {
            "sub": SUBJECT,
            "azp": "omnidash",
            "principal_id": f"t-{UUID(TENANT).hex}",
        }
    )
    tenant: str = TENANT
    subject: str = SUBJECT
    slug: str = "sim-preflight"
    accept_wrong_audience: bool = False
    accept_tamper: bool = False
    verifier_calls: list[tuple[str, str]] = field(default_factory=list)
    authorization_params: dict[str, str] = field(default_factory=dict)
    exchanged: bool = False


@pytest.fixture
def flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[UnitFlow]:
    tmp_path.chmod(0o700)
    credentials = tmp_path / "credentials.env"
    realm = tmp_path / "realm.json"
    identity = tmp_path / "identity.json"
    credentials.write_text(
        "SIM_PREFLIGHT_DEMO_USERNAME=unit-demo\n"
        "SIM_PREFLIGHT_DEMO_PASSWORD=unit-fixture-password\n"
        "SIM_PREFLIGHT_OMNIDASH_CLIENT_SECRET=unit-fixture-client-secret\n"
    )
    realm.write_text(json.dumps({"users": [{"attributes": {"tenant_id": [TENANT]}}]}))
    identity.write_text(
        json.dumps(
            {
                "tenant_id": TENANT,
                "subject": SUBJECT,
                "principal_id": f"t-{UUID(TENANT).hex}",
            }
        )
    )
    for path in (credentials, realm, identity):
        path.chmod(0o600)
    fixture = UnitFlow(credentials, realm, tmp_path / "token.jwt")

    class FakeSession:
        trust_env = True

        def get(self, url: str, **kwargs: object) -> SimpleNamespace:
            assert not self.trust_env
            assert url == ISSUER + "/protocol/openid-connect/auth"
            assert kwargs["allow_redirects"] is False
            assert kwargs["timeout"] == 15
            fixture.authorization_params = cast("dict[str, str]", kwargs["params"])
            return SimpleNamespace(
                status_code=200,
                text=f'<form id="kc-form-login" action="{ISSUER}/login-actions/authenticate?execution=unit"></form>',
            )

        def post(self, url: str, **kwargs: object) -> SimpleNamespace:
            assert not self.trust_env
            assert kwargs["allow_redirects"] is False
            assert kwargs["timeout"] == 15
            data = cast("dict[str, str]", kwargs["data"])
            if "/login-actions/" in url:
                assert url.startswith(ISSUER + "/login-actions/")
                assert data["username"] == "unit-demo"
                assert data["password"] == "unit-fixture-password"
                query = urlencode(
                    {
                        "state": fixture.authorization_params["state"],
                        "code": "unit-code",
                    }
                )
                return SimpleNamespace(
                    status_code=302, headers={"Location": REDIRECT + "?" + query}
                )
            assert url == ISSUER + "/protocol/openid-connect/token"
            assert data["grant_type"] == "authorization_code"
            assert data["client_id"] == "omnidash"
            assert data["client_secret"] == "unit-fixture-client-secret"
            assert data["redirect_uri"] == REDIRECT
            expected_challenge = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(data["code_verifier"].encode()).digest()
                )
                .rstrip(b"=")
                .decode()
            )
            assert fixture.authorization_params["code_challenge"] == expected_challenge
            assert fixture.authorization_params["code_challenge_method"] == "S256"
            fixture.exchanged = True
            return SimpleNamespace(
                status_code=200, json=lambda: {"access_token": TOKEN}
            )

    class FakeOIDCAuth:
        def __init__(self, issuer: str, audience: str) -> None:
            assert issuer == ISSUER
            self.audience = audience

        def decode_and_resolve(self, token: str) -> SimpleNamespace:
            fixture.verifier_calls.append((self.audience, token))
            if self.audience != "onex-api" and not fixture.accept_wrong_audience:
                raise FakeOIDCAuthError("unit audience refusal")
            if token != TOKEN and not fixture.accept_tamper:
                raise FakeOIDCAuthError("unit signature refusal")
            return SimpleNamespace(
                tenant=SimpleNamespace(
                    tenant_id=fixture.tenant, tenant_slug=fixture.slug
                ),
                subject=fixture.subject,
                claims=fixture.claims,
            )

    requests_module = ModuleType("requests")
    requests_module.Session = FakeSession
    auth_module = ModuleType("auth_oidc")
    auth_module.OIDCAuth = FakeOIDCAuth
    auth_module.OIDCAuthError = FakeOIDCAuthError
    tenant_module = ModuleType("tenant_identity")

    def fixture_principal(tenant: UUID) -> str:
        return f"t-{tenant.hex}"

    tenant_module.derive_principal_id = fixture_principal
    for name, module in (
        ("requests", requests_module),
        ("auth_oidc", auth_module),
        ("tenant_identity", tenant_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    previous_logging_disable = logging.root.manager.disable
    yield fixture
    logging.disable(previous_logging_disable)


def test_unit_flow_verifies_identity_and_both_negatives_before_private_token_write(
    flow: UnitFlow,
) -> None:
    result = prove(flow.credentials, flow.realm, flow.output)
    assert flow.exchanged
    assert all(result.values())
    assert [aud for aud, _ in flow.verifier_calls] == [
        "onex-api",
        "wrong-disposable-audience",
        "onex-api",
    ]
    assert flow.verifier_calls[-1][1] != TOKEN
    assert flow.output.read_text() == TOKEN
    assert flow.output.stat().st_mode & 0o777 == 0o600


def test_browser_token_mode_verifies_existing_token_without_login_or_rewrite(
    flow: UnitFlow,
) -> None:
    flow.output.write_text(TOKEN)
    flow.output.chmod(0o600)

    result = verify_browser_token(flow.output, flow.realm)

    assert not flow.exchanged
    assert all(result.values())
    assert result["browser_user_token_consumed"]
    assert [aud for aud, _ in flow.verifier_calls] == [
        "onex-api",
        "wrong-disposable-audience",
        "onex-api",
    ]
    assert flow.verifier_calls[-1][1] != TOKEN
    assert flow.output.read_text() == TOKEN


@pytest.mark.parametrize(
    "field", ["tenant", "subject", "slug", "principal_id", "azp", "sub"]
)
def test_unit_flow_refuses_wrong_identity_without_output(
    flow: UnitFlow, field: str
) -> None:
    if field in ("tenant", "subject", "slug"):
        setattr(flow, field, "unit-wrong-identity")
    else:
        flow.claims[field] = "unit-wrong-identity"
    with pytest.raises(ValueError, match="verified user identity"):
        prove(flow.credentials, flow.realm, flow.output)
    assert not flow.output.exists()


def test_unit_flow_requires_actual_sub_even_when_fallback_subject_matches(
    flow: UnitFlow,
) -> None:
    flow.claims.pop("sub")
    flow.claims["preferred_username"] = SUBJECT
    assert flow.subject == SUBJECT  # Models OIDCAuth.extract_subject's fallback.
    with pytest.raises(ValueError, match="verified user identity"):
        prove(flow.credentials, flow.realm, flow.output)
    assert not flow.output.exists()


@pytest.mark.parametrize("accepted", ["accept_wrong_audience", "accept_tamper"])
def test_unit_flow_fails_if_verifier_accepts_negative_probe(
    flow: UnitFlow, accepted: str
) -> None:
    setattr(flow, accepted, True)
    with pytest.raises(ValueError, match="was accepted"):
        prove(flow.credentials, flow.realm, flow.output)
    assert not flow.output.exists()
