# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts.edge_delegation_worker.gateway_control_plane.

Uses ``pytest-httpserver`` as the fake gateway control-plane server -- a
real local HTTP socket, not a mocked transport, so these tests exercise the
actual outbound httpx call path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

import httpx
import pytest

from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraProtocolError,
    InfraUnavailableError,
)
from scripts.edge_delegation_worker.gateway_control_plane import (
    attach,
    detach,
    heartbeat,
    resolve_access_token,
)
from scripts.edge_delegation_worker.models import ModelWorkerCredential

pytestmark = pytest.mark.unit

_SESSION_ID = UUID("11111111-1111-1111-1111-111111111111")


def _attach_response_body(*, with_renewal: bool = True) -> dict[str, object]:
    session: dict[str, object] = {
        "session_id": str(_SESSION_ID),
        "expires_at": "2026-08-16T12:00:00+00:00",
        "heartbeat_interval_seconds": 15,
    }
    body: dict[str, object] = {"session": session}
    if with_renewal:
        body["renewal"] = {
            "mode": "RE_ATTACH",
            "renew_not_before": "2026-08-16T11:30:00+00:00",
            "renew_at": "2026-08-16T11:50:00+00:00",
            "session_expires_at": "2026-08-16T12:00:00+00:00",
        }
    return body


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


@pytest.mark.asyncio
async def test_attach_success(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/attach", method="POST"
    ).respond_with_json(_attach_response_body())

    session = await attach(
        httpserver.url_for(""),  # type: ignore[attr-defined]
        access_token="token-abc",
        edge_instance_id="edge-1",
        http_client=http_client,
    )
    assert session.session_id == _SESSION_ID
    assert session.heartbeat_interval_seconds == 15
    assert session.renewal is not None
    assert session.renewal.mode == "RE_ATTACH"


@pytest.mark.asyncio
async def test_attach_401_raises_authentication_error(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/attach", method="POST"
    ).respond_with_data("unauthorized", status=401)

    with pytest.raises(InfraAuthenticationError):
        await attach(
            httpserver.url_for(""),  # type: ignore[attr-defined]
            access_token="bad-token",
            edge_instance_id="edge-1",
            http_client=http_client,
        )


@pytest.mark.asyncio
async def test_attach_malformed_body_raises_protocol_error(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/attach", method="POST"
    ).respond_with_json({"session": {"session_id": "not-a-uuid"}})

    with pytest.raises(InfraProtocolError):
        await attach(
            httpserver.url_for(""),  # type: ignore[attr-defined]
            access_token="token-abc",
            edge_instance_id="edge-1",
            http_client=http_client,
        )


@pytest.mark.asyncio
async def test_attach_unreachable_host_raises_unavailable(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(InfraUnavailableError):
        await attach(
            "http://127.0.0.1:1",  # nothing listens on TCP port 1
            access_token="token-abc",
            edge_instance_id="edge-1",
            http_client=http_client,
        )


@pytest.mark.asyncio
async def test_heartbeat_active_session(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/heartbeat", method="POST"
    ).respond_with_json({"session_id": str(_SESSION_ID), "termination_reason": None})

    result = await heartbeat(
        httpserver.url_for(""),  # type: ignore[attr-defined]
        access_token="token-abc",
        session_id=_SESSION_ID,
        http_client=http_client,
    )
    assert result.session_id == _SESSION_ID
    assert not result.is_terminated


@pytest.mark.asyncio
async def test_heartbeat_revoked_session(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/heartbeat", method="POST"
    ).respond_with_json(
        {"session_id": str(_SESSION_ID), "termination_reason": "REVOKED"}
    )

    result = await heartbeat(
        httpserver.url_for(""),  # type: ignore[attr-defined]
        access_token="token-abc",
        session_id=_SESSION_ID,
        http_client=http_client,
    )
    assert result.is_terminated
    assert result.termination_reason == "REVOKED"


@pytest.mark.asyncio
async def test_detach_success(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/detach", method="POST"
    ).respond_with_json({"session_id": str(_SESSION_ID), "detached": True})

    await detach(
        httpserver.url_for(""),  # type: ignore[attr-defined]
        access_token="token-abc",
        session_id=_SESSION_ID,
        reason="worker shutdown",
        http_client=http_client,
    )


@pytest.mark.asyncio
async def test_detach_failure_raises(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/gateway/detach", method="POST"
    ).respond_with_data("service unavailable", status=503)

    with pytest.raises(InfraAuthenticationError):
        await detach(
            httpserver.url_for(""),  # type: ignore[attr-defined]
            access_token="token-abc",
            session_id=_SESSION_ID,
            reason="worker shutdown",
            http_client=http_client,
        )


@pytest.mark.asyncio
async def test_resolve_access_token_bearer_mode_skips_network(
    http_client: httpx.AsyncClient,
) -> None:
    credential = ModelWorkerCredential(
        auth_mode="bearer_token", bearer_token="pre-issued"
    )
    token = await resolve_access_token(credential, http_client=http_client)
    assert token == "pre-issued"


@pytest.mark.asyncio
async def test_resolve_access_token_client_credentials_mode(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/token", method="POST"
    ).respond_with_json({"access_token": "minted-token", "expires_in": 300})

    credential = ModelWorkerCredential(
        auth_mode="client_credentials",
        client_id="ga-tenant-1",
        client_secret="s3cr3t",
        token_endpoint=httpserver.url_for("/token"),  # type: ignore[attr-defined]
    )
    token = await resolve_access_token(credential, http_client=http_client)
    assert token == "minted-token"


@pytest.mark.asyncio
async def test_resolve_access_token_client_credentials_missing_field_fails_closed(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/token", method="POST"
    ).respond_with_json({"token_type": "bearer"})  # no access_token field

    credential = ModelWorkerCredential(
        auth_mode="client_credentials",
        client_id="ga-tenant-1",
        client_secret="s3cr3t",
        token_endpoint=httpserver.url_for("/token"),  # type: ignore[attr-defined]
    )
    with pytest.raises(InfraProtocolError):
        await resolve_access_token(credential, http_client=http_client)
