# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``gateway_attach`` onboarding verification check (OMN-16036).

What this check claims, and therefore what these tests must pin: that the
credential the onboarding flow just wrote is REAL -- it mints a token the
gateway accepts and opens a session -- and that proving it left nothing
behind. ``http_health`` on the gateway proves the service is up and says
nothing about the credential; this is the check that closes that gap.

The seam replaced here is the socket, and only the socket: the credential
store, the token minter, the session keeper and the probe composition are the
real implementations in every test below. ``FakeGatewayTransport`` is the
near-side fake of a foreign HTTP boundary (fake Keycloak token endpoint, fake
gateway ingress) already used by the ``onex auth`` client tests -- extended in
this ticket to serve detach.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraUnavailableError, ModelInfraErrorContext
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)
from omnibase_infra.probes.probe_gateway_attach import (
    check_gateway_attach,
    prove_gateway_attach,
)
from tests.unit.gateway.conftest import (
    CLIENT_ID,
    CLIENT_SECRET,
    GATEWAY_BASE_URL,
    TOKEN_ENDPOINT,
    FakeGatewayTransport,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_transport() -> FakeGatewayTransport:
    """The gateway-client fixture, re-declared for this package.

    ``tests/unit/gateway/conftest.py`` is not on this directory's conftest
    chain, so the fixture is rebuilt here from the same class rather than the
    fake being forked -- one fake, one wire shape, no drift between the two
    suites that assert against it.
    """
    return FakeGatewayTransport(
        token_endpoint=TOKEN_ENDPOINT,
        gateway_base_url=GATEWAY_BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )


def _credential() -> ModelGatewayCredential:
    return ModelGatewayCredential(
        tenant_slug="acme",
        client_id=CLIENT_ID,
        client_secret=SecretStr(CLIENT_SECRET),
        token_endpoint=TOKEN_ENDPOINT,
        base_url=GATEWAY_BASE_URL,
        edge_instance_id="test-edge",
    )


def _write_credential(onex_home: Path) -> None:
    """Write what ``onex auth login`` writes, at the modes it writes them."""
    onex_home.mkdir(parents=True, exist_ok=True)
    (onex_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "gateway": {
                    "tenant_slug": "acme",
                    "client_id": CLIENT_ID,
                    "client_secret_ref": "acme-gateway",
                    "token_endpoint": TOKEN_ENDPOINT,
                    "base_url": GATEWAY_BASE_URL,
                    "edge_instance_id": "test-edge",
                }
            }
        )
    )
    secrets = onex_home / "credentials.json"
    secrets.touch(mode=0o600)
    secrets.chmod(0o600)
    secrets.write_text(json.dumps({"acme-gateway": CLIENT_SECRET}))


class _UnreachableTransport:
    """A gateway that is not there at all -- no status to classify."""

    async def post_form(self, url: str, **_: object) -> object:
        raise InfraUnavailableError(
            f"gateway transport could not reach {url}",
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="gateway_token_grant",
            ),
        )

    async def post_json(self, url: str, **_: object) -> object:
        raise InfraUnavailableError(
            f"gateway transport could not reach {url}",
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="gateway_request",
            ),
        )


# -- the proof itself ------------------------------------------------------


async def test_the_proof_grants_attaches_and_detaches(
    fake_transport: FakeGatewayTransport,
) -> None:
    message = await prove_gateway_attach(
        credential=_credential(),
        transport=fake_transport,
        now=fake_transport.now,
    )

    grant_urls = [url for url, _ in fake_transport.form_requests]
    assert grant_urls == [TOKEN_ENDPOINT]
    assert fake_transport.form_requests[0][1]["grant_type"] == "client_credentials"

    called = [url for url, _, _ in fake_transport.json_requests]
    assert called == [
        # OMN-16687: the attach audience is obtainable only from the exchange,
        # so the proof's first POST is the exchange, not the attach.
        f"{GATEWAY_BASE_URL}/v1/auth/gateway-token",
        f"{GATEWAY_BASE_URL}/v1/gateway/attach",
        f"{GATEWAY_BASE_URL}/v1/gateway/detach",
    ]
    assert fake_transport.exchange_count == 1
    # Non-destructive: the session the proof opened is gone again.
    assert fake_transport.attach_count == 1
    assert fake_transport.detach_count == 1
    assert fake_transport.current_session is None
    assert "acme" in message
    assert CLIENT_SECRET not in message


async def test_the_proof_message_never_carries_the_secret_or_the_token(
    fake_transport: FakeGatewayTransport,
) -> None:
    message = await prove_gateway_attach(
        credential=_credential(),
        transport=fake_transport,
        now=fake_transport.now,
    )

    assert CLIENT_SECRET not in message
    assert all(token not in message for token in fake_transport.issued_tokens)


async def test_a_rejected_grant_raises_rather_than_returning_a_proof(
    fake_transport: FakeGatewayTransport,
) -> None:
    fake_transport.token_endpoint_status = 401

    with pytest.raises(ModelOnexError):
        await prove_gateway_attach(
            credential=_credential(),
            transport=fake_transport,
            now=fake_transport.now,
        )

    assert fake_transport.attach_count == 0


async def test_a_failed_detach_names_the_session_it_could_not_close(
    fake_transport: FakeGatewayTransport,
) -> None:
    """A dangling session is reported, not swallowed.

    The check's whole non-destructive claim rests on the detach. If the attach
    succeeded and the teardown did not, the operator needs the session id and
    the ceiling -- a swallowed detach failure would leave a session open while
    reporting a clean pass.
    """
    fake_transport.detach_status = 500

    with pytest.raises(ModelOnexError) as caught:
        await prove_gateway_attach(
            credential=_credential(),
            transport=fake_transport,
            now=fake_transport.now,
        )

    assert fake_transport.current_session is not None
    session_id = str(fake_transport.current_session["session_id"])
    assert session_id in str(caught.value)


# -- the check_type entry point -------------------------------------------


async def test_check_reports_a_pass_for_a_working_credential(
    tmp_path: Path, fake_transport: FakeGatewayTransport
) -> None:
    _write_credential(tmp_path / ".onex")

    passed, message = await check_gateway_attach(
        str(tmp_path / ".onex"), 10, transport=fake_transport
    )

    assert passed is True
    assert fake_transport.current_session is None
    assert CLIENT_SECRET not in message


async def test_check_reports_a_failure_when_the_credential_is_rejected(
    tmp_path: Path, fake_transport: FakeGatewayTransport
) -> None:
    _write_credential(tmp_path / ".onex")
    fake_transport.token_endpoint_status = 401

    passed, message = await check_gateway_attach(
        str(tmp_path / ".onex"), 10, transport=fake_transport
    )

    assert passed is False
    assert "401" in message
    assert CLIENT_SECRET not in message


async def test_check_reports_a_failure_when_the_minted_token_cannot_attach(
    tmp_path: Path, fake_transport: FakeGatewayTransport
) -> None:
    _write_credential(tmp_path / ".onex")
    # A token the gateway will not accept: the grant and the exchange both
    # succeed, but what the exchange minted does not carry the attach audience
    # (OMN-16687 -- a server-side mapper drift, not a bad credential).
    fake_transport.exchange_audiences = ["redpanda-events"]

    passed, message = await check_gateway_attach(
        str(tmp_path / ".onex"), 10, transport=fake_transport
    )

    assert passed is False
    assert "gateway-attach" in message


async def test_check_reports_a_failure_when_the_gateway_is_unreachable(
    tmp_path: Path,
) -> None:
    _write_credential(tmp_path / ".onex")

    passed, message = await check_gateway_attach(
        str(tmp_path / ".onex"), 10, transport=_UnreachableTransport()
    )

    assert passed is False
    assert "could not reach" in message


async def test_check_reports_a_failure_when_no_credential_was_written(
    tmp_path: Path, fake_transport: FakeGatewayTransport
) -> None:
    """The onboarding step that should have written the credential did not."""
    passed, message = await check_gateway_attach(
        str(tmp_path / ".onex"), 10, transport=fake_transport
    )

    assert passed is False
    assert "onex auth login" in message
    assert fake_transport.form_requests == []


async def test_check_refuses_a_world_readable_secret_file(
    tmp_path: Path, fake_transport: FakeGatewayTransport
) -> None:
    onex_home = tmp_path / ".onex"
    _write_credential(onex_home)
    (onex_home / "credentials.json").chmod(0o644)

    passed, message = await check_gateway_attach(
        str(onex_home), 10, transport=fake_transport
    )

    assert passed is False
    assert "0600" in message
    assert stat.S_IMODE((onex_home / "credentials.json").stat().st_mode) == 0o644


async def test_check_bounds_the_whole_sequence_by_the_spec_timeout(
    tmp_path: Path,
) -> None:
    """A hung control plane fails the check; it does not hang onboarding."""
    _write_credential(tmp_path / ".onex")

    class _HangingTransport:
        async def post_form(self, url: str, **_: object) -> object:
            import asyncio

            await asyncio.sleep(30)
            raise AssertionError("unreachable")

        async def post_json(self, url: str, **_: object) -> object:
            raise AssertionError("never reached: the grant hangs first")

    passed, message = await check_gateway_attach(
        str(tmp_path / ".onex"), 1, transport=_HangingTransport()
    )

    assert passed is False
    assert "timed out" in message


async def test_check_defaults_to_the_onex_home_the_cli_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_transport: FakeGatewayTransport,
) -> None:
    """A blank target means ``~/.onex`` -- the path ``onex auth login`` uses."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_credential(tmp_path / ".onex")

    passed, _message = await check_gateway_attach("", 10, transport=fake_transport)

    assert passed is True


async def test_the_proof_refuses_an_attach_without_the_renewal_directive(
    fake_transport: FakeGatewayTransport,
) -> None:
    """AC4 as behaviour rather than as a claim about the source.

    The probe drives ``GatewaySessionKeeper`` -- the same object the
    unattended runtime uses for the OMN-15952 re-grant + re-attach cycle -- so
    it inherits that client's contract checks. The sharpest of them is this
    one: an attach response carrying no ``renewal`` directive has no defined
    behaviour at session expiry and is refused. A bespoke one-shot
    attach/detach written inside the handler would happily accept it, so this
    test fails the moment the reuse is unwound.
    """
    fake_transport.omit_renewal = True

    with pytest.raises(ModelOnexError) as caught:
        await prove_gateway_attach(
            credential=_credential(),
            transport=fake_transport,
            now=fake_transport.now,
        )

    assert "renewal" in str(caught.value)


async def test_the_proof_grants_a_fresh_token_for_the_attach(
    fake_transport: FakeGatewayTransport,
) -> None:
    """Also inherited from the shared client, and load-bearing for the proof.

    The gateway stamps ``expires_at`` from ``min(token exp,
    max_session_ttl_seconds)``, so attaching on a nearly-expired cached token
    would prove a session the runtime could not actually use. The keeper
    forces a fresh grant at attach; the detach that follows reuses it.
    """
    await prove_gateway_attach(
        credential=_credential(),
        transport=fake_transport,
        now=fake_transport.now,
    )

    assert len(fake_transport.form_requests) == 1
    assert len(fake_transport.issued_tokens) == 1
    assert len(fake_transport.exchanged_tokens) == 1

    exchange_headers = fake_transport.json_requests[0][2]
    assert exchange_headers["Authorization"] == (
        f"Bearer {fake_transport.issued_tokens[0]}"
    )
    # Both gateway calls present the EXCHANGED token; the machine token never
    # reaches the gateway, and the exchange is not re-run for the detach.
    gateway_presented = {
        headers["Authorization"] for _, _, headers in fake_transport.json_requests[1:]
    }
    assert gateway_presented == {f"Bearer {fake_transport.exchanged_tokens[0]}"}
