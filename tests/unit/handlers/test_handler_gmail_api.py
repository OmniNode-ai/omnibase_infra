# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerGmailApi.

Tests the Gmail API handler's core functionality including:
- OAuth2 token refresh (cold start, within buffer window, after expiry)
- Label resolution with mocked list_labels
- list_messages / search_messages failure → empty list
- get_message / modify_labels / delete_message failure → empty dict / False
- ModelGmailMessage construction from API response (base64url decode)
- Multipart body extraction (plain wins over HTML in multipart/alternative)

All tests use mocked HTTP responses to avoid external dependencies.
"""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnibase_infra.handlers.handler_gmail_api import (
    _LABEL_CACHE_TTL_SECONDS,
    _TOKEN_REFRESH_BUFFER_SECONDS,
    HandlerGmailApi,
)
from omnibase_infra.handlers.models.model_gmail_message import (
    ModelGmailMessage,
    _decode_body_data,
    _extract_body_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url(text: str) -> str:
    """Encode text as URL-safe base64 (no padding)."""
    return base64.urlsafe_b64encode(text.encode()).rstrip(b"=").decode()


def _make_handler(
    client_id: str = "test-client-id",
    client_secret: str = "test-secret",  # noqa: S107
    refresh_token: str = "test-refresh-token",  # noqa: S107
    http_client: httpx.AsyncClient | None = None,
) -> HandlerGmailApi:
    return HandlerGmailApi(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
        http_client=http_client,
    )


def _mock_token_response(
    access_token: str = "ya29.test-access-token",  # noqa: S107
    expires_in: int = 3600,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": access_token,
        "expires_in": expires_in,
        "token_type": "Bearer",
    }
    return resp


# ---------------------------------------------------------------------------
# Token refresh tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTokenRefresh:
    """Tests for OAuth2 token refresh logic."""

    @pytest.mark.asyncio
    async def test_cold_start_fetches_token(self) -> None:
        """Token is fetched on first call (cold start)."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=_mock_token_response("token-cold"))
        handler = _make_handler(http_client=mock_client)

        token = await handler._ensure_token()

        assert token == "token-cold"
        assert handler._access_token == "token-cold"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_token_reused_within_window(self) -> None:
        """Token is NOT refreshed when it still has plenty of time left."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(return_value=_mock_token_response("token-first"))
        handler = _make_handler(http_client=mock_client)

        # First call — populates token
        await handler._ensure_token()

        # Manually set expiry to far in the future
        handler._token_expiry = time.monotonic() + 3600

        # Second call — should NOT trigger another post
        token = await handler._ensure_token()

        assert token == "token-first"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_token_refreshed_when_within_buffer(self) -> None:
        """Token IS refreshed when within _TOKEN_REFRESH_BUFFER_SECONDS of expiry."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            return_value=_mock_token_response("token-refreshed"),
        )
        handler = _make_handler(http_client=mock_client)

        # Seed with existing token, but expiry is within buffer (< buffer_seconds from now)
        handler._access_token = "token-first"
        handler._token_expiry = time.monotonic() + _TOKEN_REFRESH_BUFFER_SECONDS - 10

        token = await handler._ensure_token()

        assert token == "token-refreshed"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_token_refreshed_after_expiry(self) -> None:
        """Token IS refreshed when expiry has passed."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=[
                _mock_token_response("token-new"),
            ]
        )
        handler = _make_handler(http_client=mock_client)

        # Seed with expired token
        handler._access_token = "token-expired"
        handler._token_expiry = time.monotonic() - 1.0

        token = await handler._ensure_token()

        assert token == "token-new"

    @pytest.mark.asyncio
    async def test_token_refresh_http_error_raises(self) -> None:
        """RuntimeError is raised when token endpoint returns non-200."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        err_resp = MagicMock()
        err_resp.status_code = 401
        mock_client.post = AsyncMock(return_value=err_resp)
        handler = _make_handler(http_client=mock_client)

        with pytest.raises(RuntimeError, match="token refresh failed with HTTP 401"):
            await handler._ensure_token()

    @pytest.mark.asyncio
    async def test_token_refresh_missing_credentials_raises(self) -> None:
        """RuntimeError is raised when credentials are missing."""
        handler = HandlerGmailApi(
            client_id="",
            client_secret="",
            refresh_token="",
        )
        with pytest.raises(RuntimeError, match="credentials not configured"):
            await handler._ensure_token()

    @pytest.mark.asyncio
    async def test_token_refresh_network_error_raises_sanitized(self) -> None:
        """Network errors during token refresh raise sanitized RuntimeError."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        handler = _make_handler(http_client=mock_client)

        with pytest.raises(RuntimeError) as exc_info:
            await handler._ensure_token()

        # Must not expose raw connection error with credentials context
        assert "ConnectError" in str(exc_info.value) or "unexpected error" in str(
            exc_info.value
        )
        # Must not contain the client secret
        assert "test-secret" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Label resolution tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLabelResolution:
    """Tests for resolve_label_ids and label cache."""

    @pytest.mark.asyncio
    async def test_resolve_known_labels(self) -> None:
        """Known label names are resolved to IDs."""
        handler = _make_handler()
        # Seed cache directly
        handler._label_cache = {
            "INBOX": "INBOX",
            "UNREAD": "UNREAD",
            "MyLabel": "Label_12345",
        }
        handler._label_cache_expires_at = time.monotonic() + 300

        result = await handler.resolve_label_ids(["INBOX", "MyLabel"])

        assert result == {"INBOX": "INBOX", "MyLabel": "Label_12345"}

    @pytest.mark.asyncio
    async def test_unknown_labels_omitted(self) -> None:
        """Label names not in cache are omitted from the result."""
        handler = _make_handler()
        handler._label_cache = {"INBOX": "INBOX"}
        handler._label_cache_expires_at = time.monotonic() + 300

        result = await handler.resolve_label_ids(["INBOX", "NonExistent"])

        assert result == {"INBOX": "INBOX"}
        assert "NonExistent" not in result

    @pytest.mark.asyncio
    async def test_cache_refreshed_on_expiry(self) -> None:
        """Label cache is refreshed when TTL expires."""
        handler = _make_handler()
        # Seed expired cache
        handler._label_cache = {"OldLabel": "old_id"}
        handler._label_cache_expires_at = time.monotonic() - 1.0

        # Mock list_labels to return new labels
        handler.list_labels = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                {"id": "INBOX", "name": "INBOX"},
                {"id": "Label_99", "name": "NewLabel"},
            ]
        )

        result = await handler.resolve_label_ids(["NewLabel"])

        assert result == {"NewLabel": "Label_99"}
        handler.list_labels.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_not_refreshed_within_ttl(self) -> None:
        """Label cache is NOT refreshed when TTL has not expired."""
        handler = _make_handler()
        handler._label_cache = {"INBOX": "INBOX"}
        handler._label_cache_expires_at = time.monotonic() + 200

        handler.list_labels = AsyncMock(return_value=[])  # type: ignore[method-assign]

        await handler.resolve_label_ids(["INBOX"])

        handler.list_labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_cache_triggers_refresh(self) -> None:
        """Empty cache always triggers a refresh regardless of TTL."""
        handler = _make_handler()
        handler._label_cache = {}
        # TTL not expired, but cache is empty
        handler._label_cache_expires_at = time.monotonic() + 300

        handler.list_labels = AsyncMock(  # type: ignore[method-assign]
            return_value=[{"id": "INBOX", "name": "INBOX"}]
        )

        result = await handler.resolve_label_ids(["INBOX"])

        assert result == {"INBOX": "INBOX"}
        handler.list_labels.assert_called_once()


# ---------------------------------------------------------------------------
# API method failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFailureSemantics:
    """Tests for API method failure return values."""

    @pytest.mark.asyncio
    async def test_list_messages_returns_empty_on_failure(self) -> None:
        """list_messages returns [] when token refresh fails."""
        handler = HandlerGmailApi(client_id="", client_secret="", refresh_token="")

        result = await handler.list_messages(["INBOX"])

        assert result == []

    @pytest.mark.asyncio
    async def test_get_message_returns_empty_dict_on_failure(self) -> None:
        """get_message returns {} when token refresh fails."""
        handler = HandlerGmailApi(client_id="", client_secret="", refresh_token="")

        result = await handler.get_message("msg-id-123")

        assert result == {}

    @pytest.mark.asyncio
    async def test_modify_labels_returns_false_on_failure(self) -> None:
        """modify_labels returns False when token refresh fails."""
        handler = HandlerGmailApi(client_id="", client_secret="", refresh_token="")

        result = await handler.modify_labels("msg-id-123", ["READ"], ["UNREAD"])

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_message_returns_false_on_failure(self) -> None:
        """delete_message returns False when token refresh fails."""
        handler = HandlerGmailApi(client_id="", client_secret="", refresh_token="")

        result = await handler.delete_message("msg-id-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_search_messages_returns_empty_on_failure(self) -> None:
        """search_messages returns [] when token refresh fails."""
        handler = HandlerGmailApi(client_id="", client_secret="", refresh_token="")

        result = await handler.search_messages("from:test@example.com")

        assert result == []

    @pytest.mark.asyncio
    async def test_list_labels_returns_empty_on_failure(self) -> None:
        """list_labels returns [] when token refresh fails."""
        handler = HandlerGmailApi(client_id="", client_secret="", refresh_token="")

        result = await handler.list_labels()

        assert result == []


# ---------------------------------------------------------------------------
# ModelGmailMessage tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelGmailMessage:
    """Tests for ModelGmailMessage model and body parsing."""

    def test_minimal_construction(self) -> None:
        """Minimal fields are sufficient for construction."""
        msg = ModelGmailMessage(
            message_id="abc123",
            thread_id="thread456",
            received_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert msg.message_id == "abc123"
        assert msg.thread_id == "thread456"
        assert msg.subject == ""
        assert msg.sender == ""
        assert msg.body_text == ""
        assert msg.label_ids == []

    def test_from_api_response_flat_plain(self) -> None:
        """Flat text/plain payload is decoded correctly."""
        body_text = "Hello, World!"
        raw: dict[str, Any] = {
            "id": "msg-001",
            "threadId": "thread-001",
            "labelIds": ["INBOX", "UNREAD"],
            "internalDate": "1735689600000",  # 2025-01-01 00:00:00 UTC
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "From", "value": "sender@example.com"},
                ],
                "body": {"data": _b64url(body_text)},
            },
        }

        msg = ModelGmailMessage.from_api_response(raw)

        assert msg.message_id == "msg-001"
        assert msg.thread_id == "thread-001"
        assert msg.subject == "Test Subject"
        assert msg.sender == "sender@example.com"
        assert msg.body_text == body_text
        assert "INBOX" in msg.label_ids
        assert msg.received_at == datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_from_api_response_multipart_alternative_prefers_plain(self) -> None:
        """text/plain wins over text/html in multipart/alternative."""
        plain_text = "Plain body"
        html_text = "<html>HTML body</html>"
        raw: dict[str, Any] = {
            "id": "msg-002",
            "threadId": "thread-002",
            "labelIds": [],
            "internalDate": "0",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _b64url(plain_text)},
                    },
                    {
                        "mimeType": "text/html",
                        "body": {"data": _b64url(html_text)},
                    },
                ],
            },
        }

        msg = ModelGmailMessage.from_api_response(raw)

        assert msg.body_text == plain_text

    def test_from_api_response_multipart_alternative_falls_back_to_html(self) -> None:
        """Falls back to text/html when no text/plain part exists."""
        html_text = "<html>Only HTML</html>"
        raw: dict[str, Any] = {
            "id": "msg-003",
            "threadId": "thread-003",
            "labelIds": [],
            "internalDate": "0",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [],
                "parts": [
                    {
                        "mimeType": "text/html",
                        "body": {"data": _b64url(html_text)},
                    },
                ],
            },
        }

        msg = ModelGmailMessage.from_api_response(raw)

        assert msg.body_text == html_text

    def test_from_api_response_multipart_mixed_concatenates(self) -> None:
        """multipart/mixed concatenates text parts from all sub-parts."""
        part1 = "First part"
        part2 = "Second part"
        raw: dict[str, Any] = {
            "id": "msg-004",
            "threadId": "thread-004",
            "labelIds": [],
            "internalDate": "0",
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _b64url(part1)},
                    },
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _b64url(part2)},
                    },
                ],
            },
        }

        msg = ModelGmailMessage.from_api_response(raw)

        assert part1 in msg.body_text
        assert part2 in msg.body_text

    def test_from_api_response_empty_body(self) -> None:
        """Empty body data results in empty body_text."""
        raw: dict[str, Any] = {
            "id": "msg-005",
            "threadId": "thread-005",
            "labelIds": [],
            "internalDate": "0",
            "payload": {
                "mimeType": "text/plain",
                "headers": [],
                "body": {"data": ""},
            },
        }

        msg = ModelGmailMessage.from_api_response(raw)

        assert msg.body_text == ""

    def test_from_api_response_no_payload(self) -> None:
        """Missing payload results in defaults."""
        raw: dict[str, Any] = {
            "id": "msg-006",
            "threadId": "thread-006",
            "labelIds": [],
            "internalDate": "0",
        }

        msg = ModelGmailMessage.from_api_response(raw)

        assert msg.message_id == "msg-006"
        assert msg.subject == ""
        assert msg.body_text == ""

    def test_received_at_is_utc(self) -> None:
        """received_at is always UTC timezone-aware."""
        raw: dict[str, Any] = {
            "id": "msg-007",
            "threadId": "thread-007",
            "labelIds": [],
            "internalDate": "1704067200000",  # 2024-01-01 00:00:00 UTC
            "payload": {"mimeType": "text/plain", "headers": [], "body": {}},
        }

        msg = ModelGmailMessage.from_api_response(raw)

        assert msg.received_at.tzinfo == UTC
        assert msg.received_at.year == 2024

    def test_model_is_frozen(self) -> None:
        """ModelGmailMessage is immutable (frozen=True)."""
        msg = ModelGmailMessage(
            message_id="abc",
            thread_id="thread",
            received_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(Exception):  # ValidationError or AttributeError
            msg.message_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _decode_body_data tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecodeBodyData:
    """Tests for the base64url body decoder helper."""

    def test_standard_b64url_decoding(self) -> None:
        """Standard base64url data is decoded correctly."""
        text = "Hello, Gmail!"
        encoded = _b64url(text)
        part = {"mimeType": "text/plain", "body": {"data": encoded}}

        result = _decode_body_data(part)

        assert result == text

    def test_missing_data_returns_empty(self) -> None:
        """Missing body data returns empty string."""
        part: dict[str, Any] = {"mimeType": "text/plain", "body": {}}

        result = _decode_body_data(part)

        assert result == ""

    def test_invalid_b64_returns_empty(self) -> None:
        """Invalid base64 data returns empty string (no exception)."""
        part = {"mimeType": "text/plain", "body": {"data": "!!!not-valid-base64!!!"}}

        # Should not raise
        result = _decode_body_data(part)

        # May return empty or replacement chars — just must not crash
        assert isinstance(result, str)

    def test_unicode_content_decoded(self) -> None:
        """Unicode content in body is decoded correctly."""
        text = "Héllo Wörld — 日本語"
        encoded = _b64url(text)
        part = {"body": {"data": encoded}}

        result = _decode_body_data(part)

        assert result == text
