# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Onboarding must leave the machine authenticated (OMN-17028).

THE DEFECT THIS FILE EXISTS TO PIN
    ``connect_cloud`` completed successfully and left a machine that could not
    make a single authenticated call: the policy wrote an overlay plus
    ``~/.onex/credentials.json`` and NOTHING wrote the ``~/.onex/config.yaml``
    block the credential reader resolves from, so ``onex auth status`` failed
    immediately after a run the operator was told had succeeded. A successful
    onboarding that leaves an unauthenticated machine is worse than a failed
    one, because the operator stops looking.

WHY THE ASSERTIONS ARE END-TO-END AND NOT PER-FILE
    The bug was never inside either half. The writer wrote a valid file and the
    reader read a valid file -- they were different files. Only a test that
    runs the real policy through the real handler and then resolves through the
    real store (and the real ``onex auth status`` command) can catch a
    writer/reader handoff, so nothing here asserts on a path or a key name that
    it did not first obtain by running the actual onboarding.

NO MANUAL INTERVENTION
    Every test below runs the policy and then reads back, with zero hand-written
    config between the two. The predecessor test for OMN-16038 hand-wrote the
    ``gateway:`` block between the write and the read, which is exactly the
    manual step a customer does not know to perform -- and is why the gap
    survived a green suite.
"""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli import cli_auth
from omnibase_infra.cli.cli_auth import auth_group
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.handlers.handler_onboarding import (
    handle_onboarding,
)
from omnibase_infra.nodes.node_onboarding_orchestrator.models.model_onboarding_input import (
    ModelOnboardingInput,
)
from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput

from .conftest import FakeHttpResponse

pytestmark = pytest.mark.unit

_API_KEY = "onxk_not-a-real-key-omn17028"  # pragma: allowlist secret
_BASE_URL = "https://dev.api.invalid"
_TENANT_SLUG = "acme"


@pytest.fixture
def adapter() -> AdapterFakeInput:
    """Drive every prompt ``connect_cloud`` can ask, in either revision.

    Keyed by step id, so a superset is harmless: the adapter answers the steps
    the policy actually declares and ignores the rest. The superset is
    deliberate and load-bearing rather than defensive. The pre-fix revision of
    this policy prompted for the attach-plane triple below, so an adapter that
    answered only the current three would make this file fail against that
    revision with ``KeyError: 'gateway_client_id'`` -- a fixture mismatch, not
    the defect. With every step answered, the pre-fix revision runs to a
    reported SUCCESS and the read-back below is what fails, which is precisely
    the reported bug: a completed onboarding that leaves the machine
    unauthenticated.
    """
    return AdapterFakeInput(
        responses={
            "gateway_base_url": _BASE_URL,
            "tenant_slug": _TENANT_SLUG,
            "gateway_api_key": _API_KEY,
            # Pre-fix (attach-plane) prompts; unused by the current policy.
            "gateway_client_id": "unused-principal",
            "gateway_token_endpoint": "https://keycloak.invalid/token",
            "gateway_client_secret": "unused-secret",  # pragma: allowlist secret
        }
    )


@pytest.mark.asyncio
async def test_onboarding_alone_makes_the_credential_resolvable(
    adapter: AdapterFakeInput, tmp_path: Path
) -> None:
    """The DoD probe: run the policy, then read the credential back. No edits."""
    onex_home = tmp_path / ".onex"

    output = await handle_onboarding(
        ModelOnboardingInput(
            policy_name="connect_cloud",
            dry_run=False,
            legacy_env_output=False,
            credentials_output_path=str(onex_home / "credentials.json"),
        ),
        input_adapter=adapter,
    )
    assert output.success is True

    credential = StoreGatewayCredential(onex_home=onex_home).load_read_credential()

    assert isinstance(credential, ModelGatewayApiKeyCredential)
    assert credential.base_url == _BASE_URL
    assert credential.tenant_slug == _TENANT_SLUG
    assert credential.api_key.get_secret_value() == _API_KEY


def _onboard(adapter: AdapterFakeInput, *, tmp_path: Path, home: Path) -> None:
    """Run the real policy through the real handler into ``home/.onex``.

    Synchronous on purpose. The three CLI tests below are plain functions
    because ``onex auth status`` calls ``asyncio.run`` internally, which cannot
    run inside an already-running loop -- an ``async def`` test would fail on
    the harness rather than on the behaviour.
    """
    asyncio.run(
        handle_onboarding(
            ModelOnboardingInput(
                policy_name="connect_cloud",
                dry_run=False,
                legacy_env_output=False,
                credentials_output_path=str(home / ".onex" / "credentials.json"),
            ),
            input_adapter=adapter,
        )
    )


class StubWhoamiTransport:
    """Near-side fake of ``GET /v1/whoami``, standing in for the socket only.

    Answers with the tenant the SERVER resolved, which is the whole point of
    the check: a transport that echoed the local label back would let the
    mismatch test below pass while the real command lied.
    """

    def __init__(self, *, server_slug: str = _TENANT_SLUG) -> None:
        self._server_slug = server_slug
        self.requests: list[tuple[str, dict[str, str]]] = []

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHttpResponse:
        sent = dict(headers or {})
        self.requests.append((url, sent))
        if sent.get("x-api-key") != _API_KEY:
            return FakeHttpResponse(401, '{"detail":"Unauthorized"}')
        return FakeHttpResponse(
            200,
            json.dumps(
                {
                    "tenant_id": "11111111-1111-4111-8111-111111111111",
                    "tenant_slug": self._server_slug,
                }
            ),
        )


def test_auth_status_succeeds_immediately_after_onboarding(
    adapter: AdapterFakeInput, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ticket's own done-proof probe, run as the customer would run it.

    Runs ``status`` in its DEFAULT mode -- the one that actually presents the
    stored key -- rather than the local-only ``--no-verify`` mode, so the
    assertion is "this machine is authenticated", not "this file parses".
    Only the socket is replaced.
    """
    home = tmp_path / "home"
    home.mkdir()
    _onboard(adapter, tmp_path=tmp_path, home=home)

    transport = StubWhoamiTransport()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli_auth, "GatewayTransportHttpx", lambda: transport)

    result = CliRunner().invoke(auth_group, ["status"])

    assert result.exit_code == 0, result.output
    assert _TENANT_SLUG in result.output
    assert _BASE_URL in result.output
    assert _API_KEY not in result.output
    # The key went to the gateway the credential names, in the header.
    url, headers = transport.requests[0]
    assert url == f"{_BASE_URL}/v1/whoami"
    assert headers["x-api-key"] == _API_KEY


def test_auth_status_offline_reports_without_a_network_call(
    adapter: AdapterFakeInput, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-verify`` keeps the pasteable-into-an-issue property, and says so."""
    home = tmp_path / "home"
    home.mkdir()
    _onboard(adapter, tmp_path=tmp_path, home=home)

    def _no_transport() -> None:  # pragma: no cover - reached only on regression
        raise AssertionError("--no-verify must not construct a transport")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(cli_auth, "GatewayTransportHttpx", _no_transport)

    result = CliRunner().invoke(auth_group, ["status", "--no-verify"])

    assert result.exit_code == 0, result.output
    assert "not attempted" in result.output
    assert _API_KEY not in result.output


def test_auth_status_refuses_when_the_gateway_names_another_tenant(
    adapter: AdapterFakeInput, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local label the gateway disagrees with is a failure, not a printout."""
    home = tmp_path / "home"
    home.mkdir()
    _onboard(adapter, tmp_path=tmp_path, home=home)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        cli_auth,
        "GatewayTransportHttpx",
        lambda: StubWhoamiTransport(server_slug="someone-else"),
    )

    result = CliRunner().invoke(auth_group, ["status"])

    assert result.exit_code == 1
    assert "someone-else" in result.output
    assert _API_KEY not in result.output


@pytest.mark.asyncio
async def test_onboarding_never_writes_the_key_into_the_pasteable_config(
    adapter: AdapterFakeInput, tmp_path: Path
) -> None:
    """config.yaml is the file people paste into support threads."""
    onex_home = tmp_path / ".onex"

    await handle_onboarding(
        ModelOnboardingInput(
            policy_name="connect_cloud",
            dry_run=False,
            legacy_env_output=False,
            credentials_output_path=str(onex_home / "credentials.json"),
        ),
        input_adapter=adapter,
    )

    store = StoreGatewayCredential(onex_home=onex_home)
    assert _API_KEY not in store.config_path.read_text()
    # A credential-only run writes exactly two files, both under the store's
    # own root: no overlay, no .env, nothing the credential reader will not
    # open again.
    assert sorted(p.name for p in onex_home.iterdir()) == [
        "config.yaml",
        "credentials.json",
    ]
    assert stat.S_IMODE(store.credentials_path.stat().st_mode) == 0o600
    assert _API_KEY in json.loads(store.credentials_path.read_text()).values()
