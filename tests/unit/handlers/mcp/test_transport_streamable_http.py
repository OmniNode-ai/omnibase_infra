# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for MCPAuthMiddleware in TransportMCPStreamableHttp.

Tests cover:
- Unauthenticated request to /mcp is rejected with HTTP 401
- Request with invalid token is rejected with HTTP 401
- Request with valid Bearer token is passed through (200 from inner app)
- Request with valid X-API-Key header is passed through (200 from inner app)
- Request with valid X-MCP-API-Key header is passed through (OMN-1419)
- Multiple API keys: any listed key is accepted (OMN-1419)
- Unknown key is rejected even when other valid keys are configured (OMN-1419)
- /health endpoint is exempt from auth (200 without credentials)
- When auth_enabled=False, all requests pass through (no auth check)
- 401 response body is valid JSON with error key
- Auth rejection logs include remote_ip and reason
- Successful auth logs include masked token (last 4 chars)
- Empty api_keys configured on server causes 401 (misconfiguration guard)
- ModelMcpHandlerConfig / ModelMCPServerConfig validators enforce non-empty
  api_keys when auth_enabled=True (OMN-1419).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.handlers.mcp.transport_streamable_http import MCPAuthMiddleware

if TYPE_CHECKING:
    from starlette.types import Receive, Send

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_KEY = "test-secret-token-abc"
_VALID_KEY_B = "second-client-token-xyz"


def _make_http_scope(
    path: str = "/mcp",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("127.0.0.1", 12345),
) -> dict[str, object]:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers or [],
        "client": client,
    }


class _RecordingApp:
    """Minimal ASGI app that records calls and returns 200."""

    def __init__(self) -> None:
        self.called: bool = False

    async def __call__(self, scope: object, receive: Receive, send: Send) -> None:
        self.called = True
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b"OK"})


def _collect_sends() -> tuple[list[dict[str, object]], Send]:
    """Return (messages, send_fn) where send_fn appends to messages."""
    messages: list[dict[str, object]] = []

    async def _send(msg: dict[str, object]) -> None:
        messages.append(msg)

    return messages, _send


# ---------------------------------------------------------------------------
# Tests: token rejection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_token_returns_401() -> None:
    """Unauthenticated request to /mcp -> 401."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    await middleware(_make_http_scope(path="/mcp"), receive, send)

    assert not inner.called
    assert msgs[0]["status"] == 401


@pytest.mark.asyncio
async def test_invalid_bearer_token_returns_401() -> None:
    """Wrong Bearer token -> 401."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"authorization", b"Bearer wrong-token")])
    await middleware(scope, receive, send)

    assert not inner.called
    assert msgs[0]["status"] == 401


@pytest.mark.asyncio
async def test_invalid_api_key_header_returns_401() -> None:
    """Wrong X-API-Key -> 401."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"x-api-key", b"bad-key")])
    await middleware(scope, receive, send)

    assert not inner.called
    assert msgs[0]["status"] == 401


@pytest.mark.asyncio
async def test_invalid_mcp_api_key_header_returns_401() -> None:
    """Wrong X-MCP-API-Key -> 401 (OMN-1419)."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"x-mcp-api-key", b"bad-key")])
    await middleware(scope, receive, send)

    assert not inner.called
    assert msgs[0]["status"] == 401


@pytest.mark.asyncio
async def test_empty_server_api_keys_returns_401() -> None:
    """Server configured with empty api_keys is a misconfiguration — reject all."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=())
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"authorization", b"Bearer anything")])
    await middleware(scope, receive, send)

    assert not inner.called
    assert msgs[0]["status"] == 401


# ---------------------------------------------------------------------------
# Tests: valid auth - single key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_bearer_token_passes_through() -> None:
    """Valid Bearer token -> inner app called, 200 returned."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(
        headers=[(b"authorization", f"Bearer {_VALID_KEY}".encode())]
    )
    await middleware(scope, receive, send)

    assert inner.called
    assert msgs[0]["status"] == 200


@pytest.mark.asyncio
async def test_valid_x_api_key_passes_through() -> None:
    """Valid X-API-Key (legacy header) -> inner app called, 200 returned."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"x-api-key", _VALID_KEY.encode())])
    await middleware(scope, receive, send)

    assert inner.called
    assert msgs[0]["status"] == 200


@pytest.mark.asyncio
async def test_valid_x_mcp_api_key_passes_through() -> None:
    """Valid X-MCP-API-Key -> inner app called, 200 returned (OMN-1419).

    Covers the canonical MCP header required by the ticket contract.
    """
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"x-mcp-api-key", _VALID_KEY.encode())])
    await middleware(scope, receive, send)

    assert inner.called
    assert msgs[0]["status"] == 200


# ---------------------------------------------------------------------------
# Tests: multi-key support (OMN-1419)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_key_first_key_accepted() -> None:
    """Multiple configured keys: first key authenticates (OMN-1419)."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY, _VALID_KEY_B))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"x-mcp-api-key", _VALID_KEY.encode())])
    await middleware(scope, receive, send)

    assert inner.called
    assert msgs[0]["status"] == 200


@pytest.mark.asyncio
async def test_multi_key_second_key_accepted() -> None:
    """Multiple configured keys: second key also authenticates (OMN-1419)."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY, _VALID_KEY_B))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"x-mcp-api-key", _VALID_KEY_B.encode())])
    await middleware(scope, receive, send)

    assert inner.called
    assert msgs[0]["status"] == 200


@pytest.mark.asyncio
async def test_multi_key_unknown_key_rejected() -> None:
    """Multi-key config rejects an unknown token even with valid keys present."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY, _VALID_KEY_B))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    scope = _make_http_scope(headers=[(b"x-mcp-api-key", b"not-in-allowlist")])
    await middleware(scope, receive, send)

    assert not inner.called
    assert msgs[0]["status"] == 401


# ---------------------------------------------------------------------------
# Tests: /health exemption
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint_exempt_no_credentials() -> None:
    """/health passes through without any auth headers."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    await middleware(_make_http_scope(path="/health"), receive, send)

    assert inner.called
    assert msgs[0]["status"] == 200


# ---------------------------------------------------------------------------
# Tests: non-HTTP scopes pass through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_http_scope_passes_through() -> None:
    """Lifespan and other non-http scope types are forwarded without auth check."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    _, send = _collect_sends()
    receive = AsyncMock()

    lifespan_scope: dict[str, object] = {"type": "lifespan"}
    await middleware(lifespan_scope, receive, send)

    assert inner.called


@pytest.mark.asyncio
async def test_websocket_scope_passes_through() -> None:
    """WebSocket scopes are forwarded without auth — only HTTP is auth-gated."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    _, send = _collect_sends()
    receive = AsyncMock()

    ws_scope: dict[str, object] = {
        "type": "websocket",
        "path": "/mcp",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }
    await middleware(ws_scope, receive, send)

    assert inner.called


# ---------------------------------------------------------------------------
# Tests: 401 response body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_401_body_is_valid_json() -> None:
    """401 response body must be valid JSON with an 'error' key."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    msgs, send = _collect_sends()
    receive = AsyncMock()

    await middleware(_make_http_scope(), receive, send)

    body_msg = next(m for m in msgs if m.get("type") == "http.response.body")
    body = json.loads(body_msg["body"])
    assert "error" in body
    # OMN-1419: detail must mention the X-MCP-API-Key header as an option
    assert "X-MCP-API-Key" in body.get("detail", "")


# ---------------------------------------------------------------------------
# Tests: audit logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_failure_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Auth rejections must produce a WARNING log with remote_ip and reason."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    receive = AsyncMock()
    _, send = _collect_sends()

    with caplog.at_level(
        logging.WARNING, logger="omnibase_infra.handlers.mcp.transport_streamable_http"
    ):
        await middleware(_make_http_scope(client=("10.0.0.1", 9999)), receive, send)

    assert any("MCP auth rejected" in r.message for r in caplog.records)
    rejection_record = next(
        r for r in caplog.records if "MCP auth rejected" in r.message
    )
    assert hasattr(rejection_record, "remote_ip")
    assert rejection_record.remote_ip == "10.0.0.1"  # type: ignore[attr-defined]
    assert hasattr(rejection_record, "correlation_id")
    assert rejection_record.correlation_id  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_auth_rejection_includes_client_correlation_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When X-Correlation-ID is supplied, it is propagated in rejection logs."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    receive = AsyncMock()
    _, send = _collect_sends()

    scope = _make_http_scope(headers=[(b"x-correlation-id", b"client-corr-id-123")])

    with caplog.at_level(
        logging.WARNING, logger="omnibase_infra.handlers.mcp.transport_streamable_http"
    ):
        await middleware(scope, receive, send)

    rejection_record = next(
        r for r in caplog.records if "MCP auth rejected" in r.message
    )
    assert rejection_record.correlation_id == "client-corr-id-123"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_auth_success_logs_masked_token(caplog: pytest.LogCaptureFixture) -> None:
    """Successful auth must log the last 4 chars of the token (masked)."""
    inner = _RecordingApp()
    middleware = MCPAuthMiddleware(inner, api_keys=(_VALID_KEY,))
    receive = AsyncMock()
    _, send = _collect_sends()

    scope = _make_http_scope(
        headers=[(b"authorization", f"Bearer {_VALID_KEY}".encode())]
    )

    with caplog.at_level(
        logging.INFO, logger="omnibase_infra.handlers.mcp.transport_streamable_http"
    ):
        await middleware(scope, receive, send)

    assert any("MCP auth accepted" in r.message for r in caplog.records)
    accepted_record = next(
        r for r in caplog.records if "MCP auth accepted" in r.message
    )
    assert hasattr(accepted_record, "masked_token")
    assert accepted_record.masked_token.endswith(_VALID_KEY[-4:])  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests: ModelMcpHandlerConfig auth fields (OMN-1419 multi-key)
# ---------------------------------------------------------------------------


def test_model_mcp_handler_config_auth_defaults_rejects_empty_keys() -> None:
    """auth_enabled defaults True; empty api_keys must now fail validation."""
    from pydantic import ValidationError

    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    with pytest.raises(ValidationError):
        ModelMcpHandlerConfig()


def test_model_mcp_handler_config_auth_disabled_allows_empty_keys() -> None:
    """auth_enabled=False keeps empty api_keys valid (local dev)."""
    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    cfg = ModelMcpHandlerConfig(auth_enabled=False)
    assert cfg.auth_enabled is False
    assert cfg.api_keys == ()


def test_model_mcp_handler_config_multi_key_set() -> None:
    """api_keys accepts multiple distinct tokens (OMN-1419)."""
    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    cfg = ModelMcpHandlerConfig(
        auth_enabled=True, api_keys=("alpha-token", "beta-token", "gamma-token")
    )
    assert cfg.api_keys == ("alpha-token", "beta-token", "gamma-token")


def test_model_mcp_handler_config_rejects_whitespace_key() -> None:
    """Whitespace-only keys are rejected at validation time."""
    from pydantic import ValidationError

    from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

    with pytest.raises(ValidationError):
        ModelMcpHandlerConfig(auth_enabled=True, api_keys=("valid", "   "))


# ---------------------------------------------------------------------------
# Tests: ModelMCPServerConfig auth fields (OMN-1419 multi-key)
# ---------------------------------------------------------------------------


def test_model_mcp_server_config_auth_disabled_allows_empty_keys() -> None:
    """Server config: auth_enabled=False + empty api_keys is valid."""
    from omnibase_infra.models.mcp.model_mcp_server_config import ModelMCPServerConfig

    cfg = ModelMCPServerConfig(auth_enabled=False)
    assert cfg.auth_enabled is False
    assert cfg.api_keys == ()


def test_model_mcp_server_config_auth_enabled_requires_keys() -> None:
    """Server config: auth_enabled=True with no keys raises ValidationError."""
    from pydantic import ValidationError

    from omnibase_infra.models.mcp.model_mcp_server_config import ModelMCPServerConfig

    with pytest.raises(ValidationError):
        ModelMCPServerConfig(auth_enabled=True)


def test_model_mcp_server_config_accepts_multi_key_tuple() -> None:
    """Server config: api_keys tuple is propagated end-to-end (OMN-1419)."""
    from omnibase_infra.models.mcp.model_mcp_server_config import ModelMCPServerConfig

    cfg = ModelMCPServerConfig(auth_enabled=True, api_keys=("svc-a", "svc-b"))
    assert cfg.api_keys == ("svc-a", "svc-b")
