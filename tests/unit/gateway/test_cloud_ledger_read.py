# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The operator's cloud-ledger read, from this Mac, with no cluster access (OMN-17205).

WHAT THIS PROVES
    ``beta/GOAL.md`` row 0b's probe terminates in a read of the CLOUD
    projection by correlation id. Until OMN-17205 there was no way to run it:
    ``kubectl`` against the staging cluster times out from the operator's
    machine, and the deployed onex-api served no projection route at all. This
    is the client half -- one command, credential resolved from the ``~/.onex``
    store by reference, route resolved from that same stored config, and a
    typed verdict for every outcome including the two that are not a row.

WHY EVERY OUTCOME IS TYPED
    A probe that renders "no row" and "the projection does not exist" and "your
    credential was refused" as the same empty output cannot catch a drop -- it
    reports the same thing whether the pipeline is healthy-but-idle or dead.
    So ``EnumCloudLedgerVerdict`` has five members and the exit code is derived
    from it, and only ``FOUND`` exits 0.

WHAT IS FAKED AND WHAT IS NOT
    Only the socket. ``FakeGatewayTransport`` (tests/unit/gateway/conftest.py)
    is the near-side fake of the Keycloak token endpoint; this module extends it
    with the near-side fake of the onex-api projection route. The reader, the
    verdict classification, the URL construction and the exit-code mapping are
    the real implementations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict
from omnibase_infra.gateway.client.cloud_ledger_reader import (
    CLOUD_LEDGER_CORRELATION_PATH,
    CloudLedgerReader,
)
from omnibase_infra.gateway.models.model_gateway_credential import (
    ModelGatewayCredential,
)

from .conftest import (
    CLIENT_ID,
    CLIENT_SECRET,
    GATEWAY_BASE_URL,
    TOKEN_ENDPOINT,
    FakeGatewayTransport,
    FakeHttpResponse,
)

pytestmark = pytest.mark.unit

_CID = "01J8ZC9K7Q0000000000000001"
_NOW = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)


def _credential() -> ModelGatewayCredential:
    return ModelGatewayCredential(
        tenant_slug="acme",
        client_id=CLIENT_ID,
        client_secret=SecretStr(CLIENT_SECRET),
        token_endpoint=TOKEN_ENDPOINT,
        base_url=GATEWAY_BASE_URL,
        edge_instance_id="test-edge",
    )


class ProjectionTransport(FakeGatewayTransport):
    """Adds the near-side fake of onex-api's projection read route.

    The route accepts ONLY a bearer the fake realm actually issued. A fake that
    accepted anything would let an unauthenticated client pass here and fail on
    the wire -- the same trap the attach fake already guards against.
    """

    def __init__(self, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.get_requests: list[tuple[str, Mapping[str, str]]] = []
        self.projection_status = 200
        self.projection_body: str | None = None

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> FakeHttpResponse:
        sent = dict(headers or {})
        self.get_requests.append((url, sent))
        authorization = sent.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            return FakeHttpResponse(401, '{"detail":"Unauthorized"}')
        presented = authorization.removeprefix("Bearer ")
        if presented not in self.issued_tokens:
            return FakeHttpResponse(401, '{"detail":"Unauthorized"}')
        if self.projection_status != 200:
            return FakeHttpResponse(
                self.projection_status, '{"detail":"projection refused"}'
            )
        body = self.projection_body
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
        return FakeHttpResponse(200, body)


@pytest.fixture
def transport() -> ProjectionTransport:
    return ProjectionTransport(
        token_endpoint=TOKEN_ENDPOINT,
        gateway_base_url=GATEWAY_BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )


def _reader(transport: ProjectionTransport) -> CloudLedgerReader:
    return CloudLedgerReader(transport=transport, credential=_credential())


def _found_body(cid: str = _CID) -> str:
    return json.dumps(
        {
            "correlation_id": cid,
            "projection": "hook_events",
            "data_state": "found",
            "count": 1,
            "rows": [
                {
                    "correlation_id": cid,
                    "event_id": "evt-1",
                    "run_id": "run-1",
                    "event_type": "onex.evt.omniclaude.tool-executed.v1",
                    "source": "omniclaude-hook",
                    "tenant_id": "acme",
                    "occurred_at": "2026-08-30T11:59:00Z",
                    "captured_at": "2026-08-30T11:59:01Z",
                    "payload": None,
                }
            ],
            "generated_at": "2026-08-30T12:00:00Z",
        }
    )


# ---------------------------------------------------------------------------
# Route resolution — from the stored config, never a literal host
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_is_built_from_the_stored_base_url() -> None:
    t = ProjectionTransport(
        token_endpoint=TOKEN_ENDPOINT,
        gateway_base_url=GATEWAY_BASE_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    await _reader(t).read(correlation_id=_CID, now=_NOW)
    url, _ = t.get_requests[0]
    assert url.startswith(f"{GATEWAY_BASE_URL}{CLOUD_LEDGER_CORRELATION_PATH}")
    assert f"correlation_id={_CID}" in url


@pytest.mark.asyncio
async def test_no_hardcoded_omninode_host_in_the_module() -> None:
    """The route must come from the credential, not from a literal in source."""
    source = Path("src/omnibase_infra/gateway/client/cloud_ledger_reader.py").read_text(
        encoding="utf-8"
    )
    assert "omninode.ai" not in source


# ---------------------------------------------------------------------------
# Credential — from the store, presented as a Bearer, never printed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_presents_a_minted_bearer_and_never_the_secret(
    transport: ProjectionTransport,
) -> None:
    await _reader(transport).read(correlation_id=_CID, now=_NOW)
    url, headers = transport.get_requests[0]
    authorization = headers["Authorization"]
    assert authorization.startswith("Bearer ")
    presented = authorization.removeprefix("Bearer ")
    assert presented in transport.issued_tokens
    # The client secret goes to the token endpoint and nowhere else.
    assert CLIENT_SECRET not in json.dumps({"url": url, "headers": dict(headers)})


@pytest.mark.asyncio
async def test_refused_credential_is_a_typed_verdict_not_an_empty_result(
    transport: ProjectionTransport,
) -> None:
    transport.projection_status = 401
    result = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert result.verdict is EnumCloudLedgerVerdict.UNAUTHENTICATED
    assert result.count == 0
    assert result.rows == []
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_token_endpoint_refusal_is_reported_not_swallowed(
    transport: ProjectionTransport,
) -> None:
    transport.token_endpoint_status = 401
    result = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert result.verdict is EnumCloudLedgerVerdict.UNAUTHENTICATED
    assert transport.get_requests == []


# ---------------------------------------------------------------------------
# The three server-side data states map to three distinct verdicts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_found_returns_the_row_and_exits_zero(
    transport: ProjectionTransport,
) -> None:
    transport.projection_body = _found_body()
    result = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert result.verdict is EnumCloudLedgerVerdict.FOUND
    assert result.count == 1
    assert result.correlation_id == _CID
    assert result.rows[0]["event_type"] == "onex.evt.omniclaude.tool-executed.v1"
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_not_found_is_distinct_from_projection_absent(
    transport: ProjectionTransport,
) -> None:
    result = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert result.verdict is EnumCloudLedgerVerdict.NOT_FOUND
    assert result.exit_code != 0

    transport.projection_body = json.dumps(
        {
            "correlation_id": _CID,
            "projection": "hook_events",
            "data_state": "projection_absent",
            "count": 0,
            "rows": [],
            "generated_at": "2026-08-30T12:00:00Z",
        }
    )
    absent = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert absent.verdict is EnumCloudLedgerVerdict.PROJECTION_ABSENT
    assert absent.verdict is not EnumCloudLedgerVerdict.NOT_FOUND
    assert absent.exit_code != 0
    assert absent.exit_code != result.exit_code


@pytest.mark.asyncio
async def test_service_unavailable_is_its_own_verdict(
    transport: ProjectionTransport,
) -> None:
    transport.projection_status = 503
    result = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert result.verdict is EnumCloudLedgerVerdict.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_route_that_does_not_exist_is_not_reported_as_not_found(
    transport: ProjectionTransport,
) -> None:
    """404/`Unauthorized`-on-an-unmatched-path must never read as an empty answer.

    This is the exact confusion the ticket records: onex-api 401s every
    unmatched ``/v1`` path, so "the route is missing" and "your credential is
    bad" looked identical from outside. The reader must not additionally
    collapse either of them into "no such row".
    """
    transport.projection_status = 404
    result = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert result.verdict is EnumCloudLedgerVerdict.UNAVAILABLE
    assert result.verdict is not EnumCloudLedgerVerdict.NOT_FOUND


@pytest.mark.asyncio
async def test_unparseable_body_is_unavailable_not_a_crash(
    transport: ProjectionTransport,
) -> None:
    transport.projection_body = "<html>gateway error</html>"
    result = await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert result.verdict is EnumCloudLedgerVerdict.UNAVAILABLE


# ---------------------------------------------------------------------------
# Query shaping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_include_payload_is_off_unless_asked(
    transport: ProjectionTransport,
) -> None:
    await _reader(transport).read(correlation_id=_CID, now=_NOW)
    assert "include_payload=true" not in transport.get_requests[0][0]

    await _reader(transport).read(correlation_id=_CID, now=_NOW, include_payload=True)
    assert "include_payload=true" in transport.get_requests[1][0]


@pytest.mark.asyncio
async def test_correlation_id_is_url_encoded(
    transport: ProjectionTransport,
) -> None:
    await _reader(transport).read(correlation_id="a b&c=d", now=_NOW)
    url, _ = transport.get_requests[0]
    assert "a+b%26c%3Dd" in url or "a%20b%26c%3Dd" in url
