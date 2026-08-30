# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The operator's read authenticates with a tenant API key (OMN-17205 AC1/AC2).

WHY AN API KEY AND NOT A MINTED BEARER
    The first build of this read minted an RFC 6749 s4.4 ``client_credentials``
    bearer. That path cannot close AC1 on the live plane: the only credential a
    tenant can actually hold is the P0B machine client, whose grant carries
    ``aud=redpanda-events`` (OMN-16687), while onex-api's OIDC resolver wants a
    control-plane audience. onex-api's projection route already accepts a
    SECOND credential kind on exactly equal footing -- a tenant API key in the
    ``x-api-key`` header, resolved by ``resolve_tenant_from_api_key`` -- and
    that kind is mintable by a tenant without a bearer. So the operator read is
    an API-key read.

    This is not a weaker credential path. It is tenant-scoped, revocable, and
    resolved by the same server-side tenant authority; what it removes is a
    dependency on an audience the tenant cannot obtain.

WHAT THESE TESTS PIN
    1. The store resolves an api-key credential BY REFERENCE, exactly like the
       client-secret one: the value lives only in the 0600 credentials file.
    2. An inline ``api_key`` in config.yaml is REFUSED, not warned about --
       config.yaml is the file operators paste into issues.
    3. The reader presents ``x-api-key`` and mints NO token: an api-key read
       that also hit the token endpoint would fail for a reason unrelated to
       reading.
    4. The secret never appears in the emitted document, the URL, or the
       verdict detail.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict
from omnibase_infra.gateway.client.cloud_ledger_reader import (
    CLOUD_LEDGER_CORRELATION_PATH,
    CloudLedgerReader,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)

from .conftest import FakeHttpResponse

pytestmark = pytest.mark.unit

_CID = "01J8ZC9K7Q0000000000000009"
_NOW = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
_BASE = "https://dev.api.omninode.ai"
_KEY = "onxk_test-not-a-real-key"  # pragma: allowlist secret


class ApiKeyTransport:
    """Near-side fake of onex-api's projection route, API-key arm only.

    ``post_form`` raises rather than returning: the whole point of this path is
    that no token is minted, and a fake that quietly answered a grant would let
    a regression reintroduce the token hop without a test noticing.
    """

    def __init__(self, *, status: int = 200, body: str | None = None) -> None:
        self.get_requests: list[tuple[str, Mapping[str, str]]] = []
        self._status = status
        self._body = body

    async def post_form(
        self,
        url: str,
        *,
        form: Mapping[str, str],
        headers: Mapping[str, str],
    ) -> FakeHttpResponse:
        raise AssertionError(
            "an API-key read must not call the token endpoint; it presented a "
            f"grant to {url}"
        )

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHttpResponse:
        sent = dict(headers or {})
        self.get_requests.append((url, sent))
        if sent.get("x-api-key") != _KEY:
            return FakeHttpResponse(401, '{"detail":"Unauthorized"}')
        body = self._body
        if body is None:
            body = json.dumps(
                {
                    "correlation_id": _CID,
                    "projection": "hook_events",
                    "data_state": "not_found",
                    "count": 0,
                    "rows": [],
                    "generated_at": "2026-08-30T12:00:00Z",
                }
            )
        return FakeHttpResponse(self._status, body)


def _credential() -> ModelGatewayApiKeyCredential:
    return ModelGatewayApiKeyCredential(
        tenant_slug="operator-ledger-probe",
        api_key=SecretStr(_KEY),
        api_key_ref="operator-ledger-probe-api-key",
        base_url=_BASE,
    )


# --- store ----------------------------------------------------------------


def _write_store(
    tmp_path: Path, *, block: dict[str, object], secrets: dict[str, str]
) -> StoreGatewayCredential:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"gateway": block}))
    creds = tmp_path / "credentials.json"
    creds.write_text(json.dumps(secrets))
    creds.chmod(0o600)
    return StoreGatewayCredential(onex_home=tmp_path)


def test_store_resolves_an_api_key_credential_by_reference(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path,
        block={
            "tenant_slug": "operator-ledger-probe",
            "api_key_ref": "operator-ledger-probe-api-key",
            "base_url": _BASE,
        },
        secrets={"operator-ledger-probe-api-key": _KEY},
    )

    credential = store.load_read_credential()

    assert isinstance(credential, ModelGatewayApiKeyCredential)
    assert credential.api_key.get_secret_value() == _KEY
    assert credential.base_url == _BASE
    # The secret must not have been written back into the readable config.
    assert _KEY not in (tmp_path / "config.yaml").read_text()


def test_store_refuses_an_inline_api_key_in_config(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path,
        block={
            "tenant_slug": "operator-ledger-probe",
            "api_key": _KEY,
            "base_url": _BASE,
        },
        secrets={},
    )

    with pytest.raises(ModelOnexError) as excinfo:
        store.load_read_credential()

    assert "api_key" in str(excinfo.value)


def test_read_credential_names_both_remediations_when_nothing_is_stored(
    tmp_path: Path,
) -> None:
    store = _write_store(tmp_path, block={"base_url": _BASE}, secrets={})

    with pytest.raises(ModelOnexError) as excinfo:
        store.load_read_credential()

    message = str(excinfo.value)
    assert "api_key_ref" in message
    assert "client_secret_ref" in message


def test_save_api_key_writes_reference_only_config_and_0600_secret(
    tmp_path: Path,
) -> None:
    store = StoreGatewayCredential(onex_home=tmp_path)
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"kafka": {"bootstrap": "x"}}))

    store.save_api_key(
        tenant_slug="operator-ledger-probe", api_key=_KEY, base_url=_BASE
    )

    document = yaml.safe_load((tmp_path / "config.yaml").read_text())
    # An unrelated writer's block survives the round trip (OMN-16037).
    assert document["kafka"] == {"bootstrap": "x"}
    assert "api_key" not in document["gateway"]
    assert _KEY not in (tmp_path / "config.yaml").read_text()
    assert (tmp_path / "credentials.json").stat().st_mode & 0o777 == 0o600

    reloaded = store.load_read_credential()
    assert isinstance(reloaded, ModelGatewayApiKeyCredential)
    assert reloaded.api_key.get_secret_value() == _KEY


# --- reader ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_read_presents_the_header_and_mints_no_token() -> None:
    transport = ApiKeyTransport()
    reader = CloudLedgerReader(transport=transport, credential=_credential())

    result = await reader.read(correlation_id=_CID, now=_NOW)

    assert result.verdict is EnumCloudLedgerVerdict.NOT_FOUND
    assert result.exit_code == 1
    url, headers = transport.get_requests[0]
    assert headers["x-api-key"] == _KEY
    assert "Authorization" not in headers
    assert url.startswith(_BASE + CLOUD_LEDGER_CORRELATION_PATH)


@pytest.mark.asyncio
async def test_api_key_refusal_is_unauthenticated_and_leaks_nothing() -> None:
    transport = ApiKeyTransport()
    bad = ModelGatewayApiKeyCredential(
        tenant_slug="operator-ledger-probe",
        api_key=SecretStr("onxk_wrong"),  # pragma: allowlist secret
        api_key_ref="operator-ledger-probe-api-key",
        base_url=_BASE,
    )
    reader = CloudLedgerReader(transport=transport, credential=bad)

    result = await reader.read(correlation_id=_CID, now=_NOW)

    assert result.verdict is EnumCloudLedgerVerdict.UNAUTHENTICATED
    assert result.exit_code == 3
    emitted = json.dumps(result.model_dump(mode="json"))
    assert "onxk_wrong" not in emitted
    assert _KEY not in emitted


@pytest.mark.asyncio
async def test_api_key_read_returns_the_row_when_the_projection_has_one() -> None:
    body = json.dumps(
        {
            "correlation_id": _CID,
            "projection": "hook_events",
            "data_state": "found",
            "count": 1,
            "rows": [
                {
                    "correlation_id": _CID,
                    "event_id": "e1",
                    "run_id": "r1",
                    "event_type": "tool-executed.v1",
                    "source": "claude-code",
                    "tenant_id": "operator-ledger-probe",
                    "occurred_at": "2026-08-30T11:59:00Z",
                    "captured_at": "2026-08-30T11:59:01Z",
                    "payload": None,
                }
            ],
            "generated_at": "2026-08-30T12:00:00Z",
        }
    )
    transport = ApiKeyTransport(body=body)
    reader = CloudLedgerReader(transport=transport, credential=_credential())

    result = await reader.read(correlation_id=_CID, now=_NOW)

    assert result.verdict is EnumCloudLedgerVerdict.FOUND
    assert result.exit_code == 0
    assert result.count == 1
